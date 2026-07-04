#!/usr/bin/env python3
"""Interactive studio backend (Milestone 15).

A standard-library HTTP server that backs the three-tab browser studio in
app.html. It is a thin API layer over the compiled C++ solvers -- no solver
logic lives here. The headline feature is Server-Sent Events: MAPF and
annealer solves run as background jobs that stream progress line by line, so
the browser animates a solve while it is still running instead of waiting for
a finished result.

Endpoints
  GET  /                     the studio single-page app
  GET  /maxcut               legacy standalone Max-Cut page (still served)
  GET  /api/maps             list maps with dimensions
  GET  /api/scenarios?map=X  scenarios for a map, with agent counts
  GET  /api/gset             Gset instances with best-known cut
  POST /api/solve            start a MAPF job (hybrid or cbs) -> {job_id}
  POST /api/anneal           start an annealer job -> {job_id}
  GET  /api/status/{id}      {status, events_seen, result?}
  GET  /api/result/{id}      final result (plan + metrics) when done
  POST /api/cancel/{id}      cancel an in-flight job
  GET  /api/stream/{id}      Server-Sent Events: one event per solver line
  POST /api/maxcut           run annealer vs classical on a Gset instance
  POST /api/gif              render a plan to a GIF (needs matplotlib)

No third-party dependencies. Run:  python3 mapf/viz/serve.py
"""

import argparse
import http.server
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from urllib.parse import urlsplit, parse_qs

VIZ = pathlib.Path(__file__).resolve().parent
ROOT = VIZ.parent.parent
BUILD = ROOT / "build"

# Compiled solvers. Missing binaries produce a clear error, not a crash.
SOLVE_STREAM_BIN = BUILD / "solve_stream"
SOLVE_CBS_BIN = BUILD / "solve_cbs"
ANNEAL_STREAM_BIN = BUILD / "anneal_stream"
COMPARE_BIN = BUILD / "compare_maxcut"

# Directories the studio is allowed to read maps and scenarios from. Anything
# outside these is refused (path-traversal guard).
MAP_DIRS = [ROOT / "data" / "maps", VIZ, ROOT / "mapf" / "bench" / "maps"]
SCEN_DIRS = [ROOT / "data" / "scenarios", VIZ, ROOT / "tests" / "data"]
GSET_DIR = ROOT / "data" / "gset"
BEST_KNOWN = {"G1": 11624, "G22": 13359, "G39": 2408, "G55": 10294}

# Clamp user-supplied budgets so a stray value cannot wedge the machine.
LIMITS = {
    "sweeps": (1, 200000), "replicas": (1, 64), "threads": (1, 64),
    "candidates": (1, 12), "window": (2, 32), "execute": (1, 16),
    "obstacles": (0, 20), "num_agents": (1, 40), "n": (2, 800),
    "points": (10, 1000), "restarts": (1, 5000),
}
CBS_TIME_LIMIT_MS = 8000  # keep the server responsive on hard instances


def clamp(name, value, fallback):
    lo, hi = LIMITS[name]
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return fallback


# ---- map / scenario discovery -------------------------------------------

def map_dims(path):
    """(width, height) from a MovingAI .map header, or (0, 0)."""
    w = h = 0
    try:
        with open(path) as f:
            for ln in f:
                p = ln.split()
                if not p:
                    continue
                if p[0] == "height":
                    h = int(p[1])
                elif p[0] == "width":
                    w = int(p[1])
                elif p[0] == "map":
                    break
    except (OSError, ValueError):
        pass
    return w, h


def scenario_agent_count(path):
    try:
        with open(path) as f:
            return sum(1 for ln in f if ln.strip() and not ln.startswith("version"))
    except OSError:
        return 0


def list_maps():
    out = []
    seen = set()
    for d in MAP_DIRS:
        if not d.is_dir():
            continue
        for m in sorted(d.glob("*.map")):
            rel = str(m.relative_to(ROOT))
            if rel in seen:
                continue
            seen.add(rel)
            w, h = map_dims(m)
            out.append({"path": rel, "name": m.stem, "w": w, "h": h})
    return out


def list_scenarios(map_path):
    """Scenarios whose filename stem matches the map's stem."""
    stem = pathlib.Path(map_path).stem
    out = []
    for d in SCEN_DIRS:
        if not d.is_dir():
            continue
        for s in sorted(d.glob("*.scen")):
            if s.stem != stem:
                continue
            out.append({"path": str(s.relative_to(ROOT)), "name": s.name,
                        "agents": scenario_agent_count(s)})
    return out


def resolve_under(path_str, dirs, suffix):
    cand = (ROOT / path_str).resolve()
    for d in dirs:
        if str(cand).startswith(str(d.resolve())) and cand.suffix == suffix:
            return cand
    raise ValueError(f"path not allowed: {path_str}")


def list_gset():
    out = []
    if GSET_DIR.is_dir():
        for p in sorted(GSET_DIR.iterdir()):
            if p.is_file():
                out.append({"name": p.name, "best": BEST_KNOWN.get(p.name, 0)})
    return out


# ---- background jobs (SSE) -----------------------------------------------

class Job:
    """A running solver subprocess whose output is streamed as events.

    A reader thread consumes the child's stdout, one JSON object per line,
    appending normalized event dicts. The SSE endpoint replays buffered
    events then blocks on the condition for new ones; status/result read the
    stored state. Everything is guarded by one condition variable.
    """

    def __init__(self, job_id, kind):
        self.id = job_id
        self.kind = kind          # hybrid | cbs | anneal
        self.status = "running"   # running | done | error | cancelled
        self.events = []          # list of event dicts, in order
        self.result = None        # the terminal "done" event
        self.error = None
        self.proc = None
        self.cv = threading.Condition()

    def push(self, ev):
        with self.cv:
            self.events.append(ev)
            if ev.get("event") == "done":
                self.result = ev
                if self.status == "running":
                    self.status = "done"
            self.cv.notify_all()

    def finish(self, status, error=None):
        with self.cv:
            if self.status == "running":
                self.status = status
                self.error = error
            self.cv.notify_all()

    def cancel(self):
        with self.cv:
            self.status = "cancelled"
            proc = self.proc
        if proc and proc.poll() is None:
            proc.terminate()
        with self.cv:
            self.cv.notify_all()


JOBS = {}
JOBS_LOCK = threading.Lock()


def new_job(kind):
    job = Job(uuid.uuid4().hex, kind)
    with JOBS_LOCK:
        JOBS[job.id] = job
    return job


def get_job(job_id):
    with JOBS_LOCK:
        return JOBS.get(job_id)


def run_ndjson_job(job, cmd):
    """Run a subprocess that emits one JSON object per line; push each as an
    event. Used for the hybrid MAPF stream and the annealer stream."""
    try:
        job.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1)
    except OSError as e:
        job.finish("error", f"cannot start solver: {e}")
        return
    for line in job.proc.stdout:
        line = line.strip()
        if not line:
            continue
        if job.status == "cancelled":
            break
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        job.push(ev)
    job.proc.wait()
    if job.status == "running":
        if job.result is not None:
            job.finish("done")
        else:
            err = (job.proc.stderr.read() or "")[-400:] if job.proc.stderr else ""
            job.finish("error", err or "solver produced no result")


def run_cbs_job(job, cmd_prefix, plan_path, grid_meta, agents):
    """Run one-shot CBS, then synthesize the stream events the frontend
    expects: a meta event (grid + agents, so the canvas can draw the start
    state) followed by a done event carrying the plan and metrics."""
    job.push({"event": "meta", "grid": grid_meta, "solver": "cbs", "agents": agents})
    try:
        job.proc = subprocess.Popen(cmd_prefix, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
        out, err = job.proc.communicate()
    except OSError as e:
        job.finish("error", f"cannot start cbs: {e}")
        return
    if job.status == "cancelled":
        return
    metrics = {}
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                metrics = json.loads(ln)
            except ValueError:
                pass
    paths = []
    if plan_path.exists():
        paths = parse_plan_text(plan_path.read_text())
    done = {"event": "done", "solver": "cbs", "paths": paths,
            "success": metrics.get("success", False),
            "sum_of_costs": metrics.get("sum_of_costs", 0),
            "sum_of_optimal": metrics.get("sum_of_optimal", 0),
            "overhead_pct": metrics.get("overhead_pct", 0),
            "conflicts": metrics.get("conflicts", 0),
            "expansions": metrics.get("expansions", 0),
            "wall_ms": metrics.get("wall_ms", 0)}
    job.push(done)
    if job.status == "running":
        job.finish("done")


def parse_plan_text(text):
    paths = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        p = ln.split()
        if p[0] in ("map", "agents", "makespan", "obstacles", "obstacle"):
            continue
        paths.append([[int(v) for v in tok.split(",")] for tok in p[1:]])
    return paths


def parse_scenario_agents(path, k):
    """First k agents of a .scen file as {sx,sy,gx,gy} dicts (for meta)."""
    agents = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("version"):
                continue
            p = ln.split()
            if len(p) >= 8:
                agents.append({"sx": int(p[4]), "sy": int(p[5]),
                               "gx": int(p[6]), "gy": int(p[7])})
            if len(agents) >= k:
                break
    return agents


def grid_meta_from_map(path):
    w, h = map_dims(path)
    blocked = []
    with open(path) as f:
        lines = f.read().split("\n")
    body = 0
    for i, ln in enumerate(lines):
        if ln.split()[:1] == ["map"]:
            body = i + 1
            break
    for y in range(h):
        row = lines[body + y] if body + y < len(lines) else ""
        for x in range(w):
            if x < len(row) and row[x] in "@T":
                blocked.append([x, y])
    return {"w": w, "h": h, "blocked": blocked}


def start_solve(body):
    """Start a MAPF job from {map, scenario, num_agents, solver, params}."""
    map_p = resolve_under(body.get("map", ""), MAP_DIRS, ".map")
    scen_p = resolve_under(body.get("scenario", ""), SCEN_DIRS, ".scen")
    k = clamp("num_agents", body.get("num_agents"), 8)
    solver = body.get("solver", "hybrid")
    params = body.get("params", {})
    seed = int(params.get("seed", 1))

    job = new_job(solver)
    if solver == "cbs":
        tmp = pathlib.Path(tempfile.mkdtemp())
        plan_path = tmp / "cbs.plan"
        cmd = [str(SOLVE_CBS_BIN), str(map_p), str(scen_p), str(k),
               "--time-limit-ms", str(CBS_TIME_LIMIT_MS), "--json",
               "--out", str(plan_path)]
        grid_meta = grid_meta_from_map(map_p)
        agents = parse_scenario_agents(scen_p, k)
        threading.Thread(target=run_cbs_job,
                         args=(job, cmd, plan_path, grid_meta, agents),
                         daemon=True).start()
    else:  # hybrid
        n_obs = clamp("obstacles", params.get("obstacles", 0), 0)
        cmd = [str(SOLVE_STREAM_BIN), str(map_p), str(scen_p), str(k),
               "--sweeps", str(clamp("sweeps", params.get("sweeps"), 4000)),
               "--replicas", str(clamp("replicas", params.get("replicas"), 8)),
               "--threads", str(clamp("threads", params.get("threads"), 4)),
               "--candidates", str(clamp("candidates", params.get("candidates"), 5)),
               "--seed", str(seed)]
        if n_obs > 0:
            # Moving obstacles: use the rolling-horizon driver, which streams a
            # window per cycle and dodges obstacles under a deadline.
            cmd += ["--window", str(clamp("window", params.get("window"), 8)),
                    "--execute", str(clamp("execute", params.get("execute"), 3)),
                    "--deadline-ms", str(float(params.get("deadline_ms", 0) or 0)),
                    "--obstacles", str(n_obs),
                    "--motion", "random" if params.get("motion") == "random" else "scripted"]
        else:
            # Static instance: use the one-shot hybrid solver, which is
            # near-optimal (a few percent over CBS) rather than the rolling
            # driver's much larger overhead. The frontend animates the plan.
            cmd += ["--static"]
        threading.Thread(target=run_ndjson_job, args=(job, cmd), daemon=True).start()
    return job.id


def start_anneal(body):
    """Start an annealer job from {problem, params}."""
    problem = body.get("problem", "partition")
    params = body.get("params", {})
    cmd = [str(ANNEAL_STREAM_BIN), "--problem", problem,
           "--sweeps", str(clamp("sweeps", params.get("sweeps"), 2000)),
           "--t0", str(float(params.get("t0", 10.0))),
           "--alpha", str(float(params.get("alpha", 0.99))),
           "--schedule", "linear" if params.get("schedule") == "linear" else "geometric",
           "--t-end", str(float(params.get("t_end", 0.01))),
           "--seed", str(int(params.get("seed", 1))),
           "--points", str(clamp("points", params.get("points"), 200)),
           "--threads", str(clamp("threads", params.get("threads", 1), 1))]
    if problem == "partition":
        cmd += ["--n", str(clamp("n", params.get("n", 24), 24))]
    elif problem == "maxcut-random":
        cmd += ["--n", str(clamp("n", params.get("n", 60), 60)),
                "--p", str(float(params.get("p", 0.5)))]
    elif problem == "maxcut-gset":
        gset = resolve_under("data/gset/" + pathlib.Path(body.get("gset", "")).name,
                             [GSET_DIR], "")
        cmd += ["--gset", str(gset)]
    job = new_job("anneal")
    threading.Thread(target=run_ndjson_job, args=(job, cmd), daemon=True).start()
    return job.id


# ---- Max-Cut comparison (legacy tab) -------------------------------------

def run_compare(instance, budget):
    if not COMPARE_BIN.exists():
        return {"error": "build/compare_maxcut not found; build the project first"}
    name = pathlib.Path(instance).name
    path = GSET_DIR / name
    if not path.is_file():
        return {"error": f"unknown instance {name!r}; run data/download_gset.sh"}
    best = BEST_KNOWN.get(name, 0)
    cmd = [str(COMPARE_BIN), str(path), "--best", str(best),
           "--sweeps", str(clamp("sweeps", budget.get("sweeps"), 20000)),
           "--replicas", str(clamp("replicas", budget.get("replicas"), 16)),
           "--restarts", str(clamp("restarts", budget.get("restarts"), 200)),
           "--seed", str(int(budget.get("seed", 1)))]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "compare_maxcut produced no result", "stderr": proc.stderr[-400:]}
    result["instance"] = name
    return result


# ---- GIF export ----------------------------------------------------------

def grid_to_map_text(w, h, blocked_set):
    out = ["type octile", f"height {h}", f"width {w}", "map"]
    for y in range(h):
        out.append("".join("@" if (x, y) in blocked_set else "." for x in range(w)))
    return "\n".join(out) + "\n"


def render_gif(grid, paths):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return None
    w, h = grid["w"], grid["h"]
    blocked_set = {(c[0], c[1]) for c in grid.get("blocked", [])}
    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        (tdp / "m.map").write_text(grid_to_map_text(w, h, blocked_set))
        plan = ["# mapf plan", "map m.map", f"agents {len(paths)}",
                f"makespan {max((len(p) - 1) for p in paths) if paths else 0}"]
        for a, p in enumerate(paths):
            plan.append(str(a) + " " + " ".join(f"{c[0]},{c[1]}" for c in p))
        (tdp / "m.plan").write_text("\n".join(plan) + "\n")
        gif = tdp / "m.gif"
        subprocess.run([sys.executable, str(VIZ / "render_plan.py"), str(tdp / "m.plan"),
                        "--map", str(tdp / "m.map"), "--out", str(gif)], check=True)
        return gif.read_bytes()


# ---- HTTP ----------------------------------------------------------------

def make_handler():
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, "application/json", json.dumps(obj).encode())

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n)) if n else {}

        # --- Server-Sent Events for a job ---
        def _stream_job(self, job):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            idx = 0
            try:
                while True:
                    with job.cv:
                        while idx >= len(job.events) and job.status == "running":
                            job.cv.wait(timeout=1.0)
                        batch = job.events[idx:]
                        idx = len(job.events)
                        status = job.status
                    if not batch:
                        # heartbeat comment keeps proxies from closing us
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    for ev in batch:
                        self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                        self.wfile.flush()
                    if status != "running" and idx >= len(job.events):
                        self.wfile.write(b"event: end\ndata: {}\n\n")
                        self.wfile.flush()
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass  # client navigated away; the job keeps running/finishes

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", (VIZ / "app.html").read_bytes())
            elif path == "/maxcut":
                self._send(200, "text/html; charset=utf-8", (VIZ / "maxcut.html").read_bytes())
            elif path == "/api/maps":
                self._json({"maps": list_maps()})
            elif path == "/api/scenarios":
                self._json({"scenarios": list_scenarios(query.get("map", ""))})
            elif path == "/api/gset":
                self._json({"instances": list_gset()})
            elif path.startswith("/api/status/"):
                job = get_job(path.rsplit("/", 1)[-1])
                if not job:
                    self._json({"error": "unknown job"}, 404)
                else:
                    self._json({"status": job.status, "events_seen": len(job.events),
                                "error": job.error})
            elif path.startswith("/api/result/"):
                job = get_job(path.rsplit("/", 1)[-1])
                if not job:
                    self._json({"error": "unknown job"}, 404)
                else:
                    self._json({"status": job.status, "result": job.result,
                                "error": job.error})
            elif path.startswith("/api/stream/"):
                job = get_job(path.rsplit("/", 1)[-1])
                if not job:
                    self._json({"error": "unknown job"}, 404)
                else:
                    self._stream_job(job)
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError):
                self._json({"error": "bad JSON"}, 400)
                return

            try:
                if path == "/api/solve":
                    self._json({"job_id": start_solve(body)})
                elif path == "/api/anneal":
                    self._json({"job_id": start_anneal(body)})
                elif path.startswith("/api/cancel/"):
                    job = get_job(path.rsplit("/", 1)[-1])
                    if not job:
                        self._json({"error": "unknown job"}, 404)
                    else:
                        job.cancel()
                        self._json({"status": job.status})
                elif path == "/api/maxcut":
                    self._json(run_compare(body.get("instance", ""), body.get("budget", {})))
                elif path == "/api/gif":
                    paths = body.get("paths", [])
                    if not paths:
                        self._json({"error": "no plan to render"}, 400)
                        return
                    data = render_gif(body.get("grid", {}), paths)
                    if data is None:
                        self._json({"error": "matplotlib not available on the server"}, 501)
                    else:
                        self._send(200, "image/gif", data)
                else:
                    self._send(404, "text/plain", b"not found")
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except subprocess.TimeoutExpired:
                self._json({"error": "solver timed out"}, 504)
            except Exception as e:  # keep the server alive on any failure
                self._json({"error": f"request failed: {e}"}, 500)

        def log_message(self, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler())
    url = f"http://localhost:{args.port}"
    print(f"MAPF/annealer studio at {url}  (Ctrl-C to stop)")
    for name, b in (("solve_stream", SOLVE_STREAM_BIN), ("solve_cbs", SOLVE_CBS_BIN),
                    ("anneal_stream", ANNEAL_STREAM_BIN)):
        if not b.exists():
            print(f"  note: build/{name} not found -- build first")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
