#!/usr/bin/env python3
"""Render a MAPF plan to an animated GIF.

Reads a plan file written by `solve_mapf --out` and the map it references,
then draws one frame per timestep (obstacles shaded, each agent a colored
dot moving toward its goal marker) and assembles them into a GIF.

Usage:
    python3 mapf/viz/render_plan.py <plan_file> [--map MAP] [--out OUT.gif]
                                    [--fps N] [--cell PIXELS]

If --map is omitted, the map path recorded in the plan file is used.
Requires matplotlib and pillow (plotting only; the solver has no deps).
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colormaps


def load_map(path):
    """Return (width, height, blocked) where blocked[y][x] is bool."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    width = height = 0
    body_start = 0
    for i, ln in enumerate(lines):
        parts = ln.split()
        if not parts:
            continue
        if parts[0] == "height":
            height = int(parts[1])
        elif parts[0] == "width":
            width = int(parts[1])
        elif parts[0] == "map":
            body_start = i + 1
            break
    blocked = [[False] * width for _ in range(height)]
    for y in range(height):
        row = lines[body_start + y]
        for x in range(width):
            if row[x] in "@T":
                blocked[y][x] = True
    return width, height, blocked


def load_plan(path):
    """Return (map_name, paths) where paths[a] is a list of (x, y)."""
    map_name = None
    paths = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if parts[0] == "map":
                map_name = parts[1]
            elif parts[0] in ("agents", "makespan"):
                continue
            else:
                # "<agent> x0,y0 x1,y1 ..."
                cells = [tuple(map(int, tok.split(","))) for tok in parts[1:]]
                paths.append(cells)
    return map_name, paths


def render(plan_file, map_path, out_path, fps, cell_px):
    map_name, paths = load_plan(plan_file)
    if map_path is None:
        map_path = map_name
    width, height, blocked = load_map(map_path)

    makespan = max(len(p) for p in paths) - 1
    n_agents = len(paths)
    colors = [colormaps["tab20"](i % 20) for i in range(n_agents)]

    def pos(a, t):  # park on final cell after the path ends
        p = paths[a]
        return p[t] if t < len(p) else p[-1]

    fig, ax = plt.subplots(figsize=(width * cell_px / 100.0, height * cell_px / 100.0))
    # Background: obstacles dark, free light. extent lines cells up to
    # integer coordinates; origin upper so y increases downward (MovingAI).
    bg = [[0.15 if blocked[y][x] else 0.95 for x in range(width)] for y in range(height)]
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, extent=(0, width, height, 0))
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="0.7", linewidth=0.5)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    # Static goal markers (X) in each agent's color.
    for a in range(n_agents):
        gx, gy = paths[a][-1]
        ax.plot(gx + 0.5, gy + 0.5, marker="x", color=colors[a], markersize=8,
                markeredgewidth=2)

    # Agent dots, updated each frame.
    scatter = ax.scatter(
        [pos(a, 0)[0] + 0.5 for a in range(n_agents)],
        [pos(a, 0)[1] + 0.5 for a in range(n_agents)],
        c=colors, s=(cell_px * 3), edgecolors="black", zorder=3,
    )
    title = ax.set_title(f"t = 0 / {makespan}")

    def update(t):
        scatter.set_offsets(
            [[pos(a, t)[0] + 0.5, pos(a, t)[1] + 0.5] for a in range(n_agents)]
        )
        title.set_text(f"t = {t} / {makespan}")
        return scatter, title

    # A short pause on the final frame so the GIF ends on the goal state.
    frames = list(range(makespan + 1)) + [makespan] * max(1, fps)
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out_path} ({n_agents} agents, makespan {makespan})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_file")
    ap.add_argument("--map", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--cell", type=int, default=40, help="pixels per cell")
    args = ap.parse_args()
    out = args.out or str(pathlib.Path(args.plan_file).with_suffix(".gif"))
    render(args.plan_file, args.map, out, args.fps, args.cell)


if __name__ == "__main__":
    main()
