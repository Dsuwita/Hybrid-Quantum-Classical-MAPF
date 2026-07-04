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
    """Return (map_name, paths, obstacles, goals). Agent lines start with an
    integer index; obstacle trajectory lines start with 'obstacle'; per-agent
    goal-trajectory lines (lifelong mode) start with 'goal'."""
    map_name = None
    paths = []
    obstacles = []
    goals = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if parts[0] == "map":
                map_name = parts[1]
            elif parts[0] in ("agents", "makespan", "obstacles"):
                continue
            elif parts[0] == "obstacle":
                obstacles.append([tuple(map(int, tok.split(","))) for tok in parts[1:]])
            elif parts[0] == "goal":
                # "goal <agent> x0,y0 x1,y1 ..."
                goals.append([tuple(map(int, tok.split(","))) for tok in parts[2:]])
            else:
                # "<agent> x0,y0 x1,y1 ..."
                paths.append([tuple(map(int, tok.split(","))) for tok in parts[1:]])
    return map_name, paths, obstacles, goals


def render(plan_file, map_path, out_path, fps, cell_px, trail=6):
    map_name, paths, obstacles, goals = load_plan(plan_file)
    if map_path is None:
        map_path = map_name
    width, height, blocked = load_map(map_path)

    makespan = max(len(p) for p in paths) - 1
    n_agents = len(paths)
    colors = [colormaps["tab20"](i % 20) for i in range(n_agents)]
    lifelong = len(goals) == n_agents  # goal-history present -> moving goals

    def pos(a, t):  # park on final cell after the path ends
        p = paths[a]
        return p[t] if t < len(p) else p[-1]

    def goal_pos(a, t):  # current goal at time t (lifelong) or final cell
        g = goals[a] if lifelong else paths[a]
        return g[t] if t < len(g) else g[-1]

    def obs_pos(o, t):
        p = obstacles[o]
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

    # Goal markers (X) in each agent's color. In lifelong mode they move each
    # frame to the agent's current goal; otherwise they are the fixed goals.
    goal_scatter = ax.scatter(
        [goal_pos(a, 0)[0] + 0.5 for a in range(n_agents)],
        [goal_pos(a, 0)[1] + 0.5 for a in range(n_agents)],
        c=colors, marker="x", s=(cell_px * 2), linewidths=2, zorder=2,
    )

    # Motion trails: one fading line per agent showing its last `trail` steps,
    # which makes the continuous replanning read as motion rather than a jump.
    trail_lines = []
    for a in range(n_agents):
        (ln,) = ax.plot([], [], color=colors[a], linewidth=2, alpha=0.5, zorder=1)
        trail_lines.append(ln)

    # Moving obstacles: dark squares redrawn each frame (via a Rectangle
    # per obstacle whose position is updated in update()).
    from matplotlib.patches import Rectangle
    obstacle_patches = []
    for o in range(len(obstacles)):
        ox, oy = obs_pos(o, 0)
        rect = Rectangle((ox + 0.1, oy + 0.1), 0.8, 0.8, color="#c92a2a", zorder=2)
        ax.add_patch(rect)
        obstacle_patches.append(rect)

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
        goal_scatter.set_offsets(
            [[goal_pos(a, t)[0] + 0.5, goal_pos(a, t)[1] + 0.5] for a in range(n_agents)]
        )
        for a in range(n_agents):
            seg = [pos(a, tau) for tau in range(max(0, t - trail), t + 1)]
            trail_lines[a].set_data([c[0] + 0.5 for c in seg], [c[1] + 0.5 for c in seg])
        for o, rect in enumerate(obstacle_patches):
            ox, oy = obs_pos(o, t)
            rect.set_xy((ox + 0.1, oy + 0.1))
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
