#!/usr/bin/env python3
"""Interactive MAPF GUI backend.

A small standard-library HTTP server that backs the browser app in
app.html. It does three things the static GIF gallery could not:

  1. Drives the real solver. POST /api/solve takes a grid + a list of
     agents, runs the compiled build/solve_mapf on them, and returns the
     resulting plan and metrics as JSON. The browser then animates it.
  2. Serves maps to edit. GET /api/maps lists the bundled maps;
     GET /api/example loads a demo map + scenario into the editor.
  3. Optional GIF export. POST /api/gif renders the current plan to a GIF
     via render_plan.py (needs matplotlib; returns 501 if unavailable).

No third-party dependencies for the server itself. Run:
    python3 mapf/viz/serve.py            # opens http://localhost:8000
"""

import argparse
import collections
import http.server
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import webbrowser

VIZ = pathlib.Path(__file__).resolve().parent
ROOT = VIZ.parent.parent
SOLVE_BIN = ROOT / "build" / "solve_mapf"
MAP_DIRS = [VIZ, ROOT / "mapf" / "bench" / "maps"]
EXAMPLES = {
    "crossing": ("demo_crossing.map", "demo_crossing.scen"),
    "rooms": ("demo_rooms.map", "demo_rooms.scen"),
}

# Clamp user-supplied budgets so a stray value cannot wedge the machine.
LIMITS = {"sweeps": (1, 200000), "replicas": (1, 64), "candidates": (1, 12),
          "iters": (1, 50)}
MAX_AGENTS = 40


def clamp(name, value, fallback):
    lo, hi = LIMITS[name]
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return fallback


# ---- map / scenario parsing ---------------------------------------------

def parse_map_text(text):
    lines = text.split("\n")
    w = h = body = 0
    for i, ln in enumerate(lines):
        p = ln.split()
        if not p:
            continue
        if p[0] == "height":
            h = int(p[1])
        elif p[0] == "width":
            w = int(p[1])
        elif p[0] == "map":
            body = i + 1
            break
    blocked = [[False] * w for _ in range(h)]
    for y in range(h):
        row = lines[body + y] if body + y < len(lines) else ""
        for x in range(w):
            if x < len(row) and row[x] in "@T":
                blocked[y][x] = True
    return w, h, blocked


def grid_to_map_text(w, h, blocked):
    out = [f"type octile", f"height {h}", f"width {w}", "map"]
    for y in range(h):
        out.append("".join("@" if blocked[y][x] else "." for x in range(w)))
    return "\n".join(out) + "\n"


def parse_scenario_text(text):
    agents = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("version"):
            continue
        p = ln.split()
        if len(p) >= 9:
            agents.append({"sx": int(p[4]), "sy": int(p[5]),
                           "gx": int(p[6]), "gy": int(p[7])})
    return agents


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


def resolve_map(path_str):
    """Resolve a client-provided map path, refusing anything outside the
    allowed map directories."""
    cand = (ROOT / path_str).resolve()
    for d in MAP_DIRS:
        if str(cand).startswith(str(d.resolve())) and cand.suffix == ".map":
            return cand
    raise ValueError(f"map path not allowed: {path_str}")


# ---- shortest path (for exact per-agent optimal / overhead) --------------

def bfs_len(w, h, blocked, start, goal):
    if start == goal:
        return 0
    seen = [[False] * w for _ in range(h)]
    q = collections.deque([(start[0], start[1], 0)])
    seen[start[1]][start[0]] = True
    while q:
        x, y, d = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not blocked[ny][nx] and not seen[ny][nx]:
                if (nx, ny) == goal:
                    return d + 1
                seen[ny][nx] = True
                q.append((nx, ny, d + 1))
    return None


# ---- the solve driver ----------------------------------------------------

def run_solve(w, h, blocked, agents, budget):
    if not SOLVE_BIN.exists():
        return {"error": "build/solve_mapf not found; build the project first"}
    if not agents:
        return {"error": "place at least one agent"}
    if len(agents) > MAX_AGENTS:
        return {"error": f"too many agents (max {MAX_AGENTS})"}

    # Exact per-agent shortest path length -> correct overhead. Also the
    # earliest place we catch an unreachable goal.
    optimal = []
    for i, a in enumerate(agents):
        d = bfs_len(w, h, blocked, (a["sx"], a["sy"]), (a["gx"], a["gy"]))
        if d is None:
            return {"error": f"agent {i} goal ({a['gx']},{a['gy']}) is unreachable"}
        optimal.append(d)

    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        (tdp / "m.map").write_text(grid_to_map_text(w, h, blocked))
        scen = ["version 1"]
        for a, opt in zip(agents, optimal):
            scen.append(f"0\tm.map\t{w}\t{h}\t{a['sx']}\t{a['sy']}\t{a['gx']}\t{a['gy']}\t{opt}")
        (tdp / "m.scen").write_text("\n".join(scen) + "\n")

        cmd = [str(SOLVE_BIN), str(tdp / "m.map"), str(tdp / "m.scen"), str(len(agents)),
               "--sweeps", str(clamp("sweeps", budget.get("sweeps"), 2000)),
               "--replicas", str(clamp("replicas", budget.get("replicas"), 8)),
               "--candidates", str(clamp("candidates", budget.get("candidates"), 4)),
               "--iters", str(clamp("iters", budget.get("iters"), 10)),
               "--seed", str(int(budget.get("seed", 1))),
               "--out", str(tdp / "m.plan")]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        metrics = {}
        for ln in proc.stdout.splitlines():
            parts = ln.split(None, 1)
            if len(parts) == 2:
                metrics[parts[0]] = parts[1].strip()

        plan_file = tdp / "m.plan"
        paths = parse_plan_text(plan_file.read_text()) if plan_file.exists() else []

    return {
        "success": metrics.get("success") == "yes",
        "metrics": metrics,
        "grid": {"w": w, "h": h, "blocked": blocked},
        "paths": paths,
    }


def render_gif(w, h, blocked, paths):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return None
    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        (tdp / "m.map").write_text(grid_to_map_text(w, h, blocked))
        plan = ["# mapf plan", "map m.map", f"agents {len(paths)}",
                f"makespan {max((len(p) - 1) for p in paths) if paths else 0}"]
        for a, p in enumerate(paths):
            plan.append(str(a) + " " + " ".join(f"{c[0]},{c[1]}" for c in p))
        (tdp / "m.plan").write_text("\n".join(plan) + "\n")
        gif = tdp / "m.gif"
        # Same interpreter that runs the server, so render_plan.py sees the
        # same matplotlib we just confirmed is importable.
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

        def do_GET(self):
            # parse_qs percent-decodes values, so a path sent as
            # "mapf%2Fbench%2F..." arrives as "mapf/bench/...".
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8",
                           (VIZ / "app.html").read_bytes())
            elif path == "/api/maps":
                maps = []
                for d in MAP_DIRS:
                    for m in sorted(d.glob("*.map")):
                        maps.append(str(m.relative_to(ROOT)))
                self._json({"maps": maps})
            elif path == "/api/map":
                try:
                    p = resolve_map(query.get("path", ""))
                    w, h, blocked = parse_map_text(p.read_text())
                    self._json({"grid": {"w": w, "h": h, "blocked": blocked}})
                except (ValueError, OSError) as e:
                    self._json({"error": str(e)}, 400)
            elif path == "/api/example":
                name = query.get("name", "")
                if name not in EXAMPLES:
                    self._json({"error": "unknown example"}, 404)
                    return
                mp, sp = EXAMPLES[name]
                w, h, blocked = parse_map_text((VIZ / mp).read_text())
                agents = parse_scenario_text((VIZ / sp).read_text())
                self._json({"grid": {"w": w, "h": h, "blocked": blocked}, "agents": agents})
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError):
                self._json({"error": "bad JSON"}, 400)
                return

            if self.path == "/api/solve":
                g = body.get("grid", {})
                try:
                    result = run_solve(g["w"], g["h"], g["blocked"],
                                       body.get("agents", []), body.get("budget", {}))
                    self._json(result)
                except subprocess.TimeoutExpired:
                    self._json({"error": "solver timed out"}, 504)
                except Exception as e:  # keep the server alive on any failure
                    self._json({"error": f"solve failed: {e}"}, 500)
            elif self.path == "/api/gif":
                g = body.get("grid", {})
                paths = body.get("paths", [])
                if not paths:
                    self._json({"error": "no plan to render"}, 400)
                    return
                try:
                    data = render_gif(g["w"], g["h"], g["blocked"], paths)
                except Exception as e:
                    self._json({"error": f"render failed: {e}"}, 500)
                    return
                if data is None:
                    self._json({"error": "matplotlib not available on the server"}, 501)
                else:
                    self._send(200, "image/gif", data)
            else:
                self._send(404, "text/plain", b"not found")

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
    print(f"MAPF GUI at {url}  (Ctrl-C to stop)")
    if not SOLVE_BIN.exists():
        print("  note: build/solve_mapf not found -- build first so Solve works")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
