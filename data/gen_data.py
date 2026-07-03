#!/usr/bin/env python3
"""Generate MovingAI-format maps and scenarios for the studio.

The studio's map/scenario dropdowns read data/maps/ and data/scenarios/.
This script writes a small, self-contained benchmark set there so a fresh
clone has something to load without downloading the full MovingAI corpus.
Everything is seeded, so re-running reproduces the same files exactly.

Maps (all 32x32, MovingAI .map format):
  empty-32-32     open grid
  room-32-32      four rooms joined by door gaps
  random-32-32    ~12% random blocked cells (single connected component)

Scenarios (.scen): 40 agents each, random reachable start/goal pairs with
the exact BFS shortest-path length as the published `optimal` field, so the
verifier's overhead metric is meaningful.
"""

import collections
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
MAPS = HERE / "maps"
SCENS = HERE / "scenarios"


def write_map(name, w, h, blocked):
    MAPS.mkdir(parents=True, exist_ok=True)
    lines = ["type octile", f"height {h}", f"width {w}", "map"]
    for y in range(h):
        lines.append("".join("@" if blocked[y][x] else "." for x in range(w)))
    (MAPS / f"{name}.map").write_text("\n".join(lines) + "\n")


def empty_map(w, h):
    return [[False] * w for _ in range(h)]


def room_map(w, h):
    b = [[False] * w for _ in range(h)]
    midx, midy = w // 2, h // 2
    for x in range(w):
        b[midy][x] = True
    for y in range(h):
        b[y][midx] = True
    # Door gaps in each wall segment so all four rooms connect.
    for gap in (midy // 2, midy + midy // 2):
        b[gap][midx] = False
    for gap in (midx // 2, midx + midx // 2):
        b[midy][gap] = False
    return b


def random_map(w, h, density, seed):
    rng = random.Random(seed)
    while True:
        b = [[rng.random() < density for _ in range(w)] for _ in range(h)]
        # Keep the four corners open as candidate endpoints.
        for (x, y) in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            b[y][x] = False
        if largest_component(w, h, b) >= 0.75 * w * h:
            return b


def passable(w, h, b, x, y):
    return 0 <= x < w and 0 <= y < h and not b[y][x]


def bfs(w, h, b, start):
    """Distances from start to every reachable cell."""
    dist = {start: 0}
    q = collections.deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if passable(w, h, b, nx, ny) and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def largest_component(w, h, b):
    seen = set()
    best = 0
    for sy in range(h):
        for sx in range(w):
            if b[sy][sx] or (sx, sy) in seen:
                continue
            comp = bfs(w, h, b, (sx, sy))
            seen |= set(comp)
            best = max(best, len(comp))
    return best


def write_scenario(name, w, h, blocked, n_agents, seed):
    SCENS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    free = [(x, y) for y in range(h) for x in range(w) if not blocked[y][x]]
    lines = ["version 1"]
    made = 0
    guard = 0
    while made < n_agents and guard < n_agents * 200:
        guard += 1
        sx, sy = rng.choice(free)
        gx, gy = rng.choice(free)
        if (sx, sy) == (gx, gy):
            continue
        dist = bfs(w, h, blocked, (sx, sy))
        if (gx, gy) not in dist:
            continue
        opt = dist[(gx, gy)]
        lines.append(f"0\t{name}.map\t{w}\t{h}\t{sx}\t{sy}\t{gx}\t{gy}\t{opt}")
        made += 1
    (SCENS / f"{name}.scen").write_text("\n".join(lines) + "\n")
    return made


def main():
    W = H = 32
    specs = [
        ("empty-32-32", empty_map(W, H)),
        ("room-32-32", room_map(W, H)),
        ("random-32-32", random_map(W, H, 0.12, seed=7)),
    ]
    for name, blocked in specs:
        write_map(name, W, H, blocked)
        made = write_scenario(name, W, H, blocked, n_agents=40, seed=hash(name) & 0xFFFF)
        print(f"{name}: map {W}x{H}, scenario with {made} agents")


if __name__ == "__main__":
    main()
