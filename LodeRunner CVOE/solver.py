"""Game-aware Solver for Lode Runner Levels.

Features:
- Discrete transition model with support, falling, ladder climbing, rope traversal, and stepping down.
- Monotone fixpoint digging solver ('b' only, diagonally adjacent, opened cells become passable).
- Completable check using directed SCC graph analysis for one-way pits.
- A* shortest path with Manhattan heuristic for greedy nearest-gold tour solving.
"""

from collections import deque
import heapq
import json
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any, Optional, Union

import constants

LevelType = Union[str, List[str], List[List[str]]]


def normalize_grid(level: LevelType) -> List[str]:
    """Convert level representation into a list of strings of dimensions (ROWS, COLS)."""
    if isinstance(level, str):
        lines = [line.strip() for line in level.strip().splitlines() if line.strip()]
    elif isinstance(level, list):
        if len(level) > 0 and isinstance(level[0], list):
            lines = ["".join(row) for row in level]
        else:
            lines = [str(line).strip() for line in level if str(line).strip()]
    else:
        raise ValueError("Unsupported level format")

    # Pad or crop to exact dimensions if necessary
    processed_lines = []
    for r in range(constants.ROWS):
        if r < len(lines):
            line = lines[r]
            if len(line) < constants.COLS:
                line = line + "." * (constants.COLS - len(line))
            elif len(line) > constants.COLS:
                line = line[: constants.COLS]
        else:
            line = "." * constants.COLS
        # Replace any unknown characters with '.'
        clean_chars = [c if c in constants.TILES_DICT else "." for c in line]
        processed_lines.append("".join(clean_chars))

    return processed_lines


def is_solid(r: int, c: int, grid: List[str], opened: Set[Tuple[int, int]]) -> bool:
    """Return True if tile at (r, c) is currently solid (not dug out)."""
    if (r, c) in opened:
        return False
    return grid[r][c] in constants.SOLID


def is_supported(r: int, c: int, grid: List[str], opened: Set[Tuple[int, int]]) -> bool:
    """Check if player at (r, c) is supported by ground, ladder, rope, or level boundary."""
    R = len(grid)
    tile = "." if (r, c) in opened else grid[r][c]
    if tile in {"#", "-"}:
        return True
    if r + 1 < R:
        if grid[r + 1][c] in constants.SOLID or (grid[r + 1][c] == "#" and (r + 1, c) not in opened):
            return True
    elif r == R - 1:
        # Bottom row rests on the bottom floor boundary of the game screen
        return True
    return False


def get_transitions(
    r: int, c: int, grid: List[str], opened: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return all valid single-step destination coordinates from (r, c)."""
    R = len(grid)
    C = len(grid[0])
    moves: List[Tuple[int, int]] = []
    tile = "." if (r, c) in opened else grid[r][c]
    supp = is_supported(r, c, grid, opened)

    # If the player is currently inside a dug hole in the floor:
    # In Lode Runner, dug holes are vertical pits. The player cannot walk horizontally
    # through subterranean brick layers! They can only climb up into open air above, or fall down if open.
    if (r, c) in opened:
        if r - 1 >= 0 and not is_solid(r - 1, c, grid, opened) and (r - 1, c) not in opened:
            moves.append((r - 1, c))
        if r + 1 < R and not is_solid(r + 1, c, grid, opened):
            moves.append((r + 1, c))
        return moves

    if not supp:
        # Not supported: fall straight down until landing on solid, ladder, or rope
        if r + 1 < R and not is_solid(r + 1, c, grid, opened):
            moves.append((r + 1, c))
    else:
        # Supported: move left and right into any non-solid cell (cannot enter subterranean dug pits sideways)
        if c - 1 >= 0 and not is_solid(r, c - 1, grid, opened) and (r, c - 1) not in opened:
            moves.append((r, c - 1))
        if c + 1 < C and not is_solid(r, c + 1, grid, opened) and (r, c + 1) not in opened:
            moves.append((r, c + 1))

        # On ladder: move up or down into any non-solid cell
        # (stepping up out of ladder top into open air is naturally included)
        if tile == "#":
            if r - 1 >= 0 and not is_solid(r - 1, c, grid, opened):
                moves.append((r - 1, c))
            if r + 1 < R and not is_solid(r + 1, c, grid, opened):
                moves.append((r + 1, c))
        # On rope: can drop down if cell below is non-solid
        elif tile == "-":
            if r + 1 < R and not is_solid(r + 1, c, grid, opened):
                moves.append((r + 1, c))
        # Supported on open air directly above a ladder: can step down into ladder
        elif r + 1 < R and grid[r + 1][c] == "#" and (r + 1, c) not in opened:
            moves.append((r + 1, c))

    return moves


def find_reachable_with_digging(
    start: Tuple[int, int], grid: List[str]
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Monotone fixpoint algorithm for reachable cells and diggable 'b' tiles.
    Returns (all_reachable_cells, opened_cells).
    """
    R = len(grid)
    C = len(grid[0])
    opened: Set[Tuple[int, int]] = set()
    all_visited: Set[Tuple[int, int]] = set()
    queue = deque([start])
    visited = {start}

    while True:
        # Expand reachable states
        while queue:
            curr = queue.popleft()
            for nxt in get_transitions(curr[0], curr[1], grid, opened):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        all_visited.update(visited)

        # Identify newly diggable 'b' tiles from all reached supported states
        newly_opened: Set[Tuple[int, int]] = set()
        for r, c in all_visited:
            # Player can only dig while standing in legitimate open space (NOT from inside a dug hole!)
            if (r, c) not in opened and is_supported(r, c, grid, opened):
                for dc in [-1, 1]:
                    nc = c + dc
                    if 0 <= nc < C and r + 1 < R:
                        # Adjacent side cell must be clear air (so player's digging beam can pass)
                        adj_clear = not is_solid(r, nc, grid, opened) and (r, nc) not in opened
                        target_b = (
                            grid[r + 1][nc] in constants.DIGGABLE
                            and (r + 1, nc) not in opened
                        )
                        if adj_clear and target_b:
                            newly_opened.add((r + 1, nc))

        if not newly_opened:
            return all_visited, opened

        opened.update(newly_opened)
        # Continue BFS expanding from all visited states through newly opened cells
        queue = deque(list(all_visited))


def analyze(level: LevelType) -> Dict[str, Any]:
    """Analyze a level's playability and reachability.

    Returns:
        dict:
          - spawn_ok: bool (exactly one 'M')
          - n_gold: int
          - reachable_cells: Set of (r, c)
          - unreachable_gold: List of (r, c)
          - gold_reachable_fraction: float
          - is_playable: bool
          - opened_cells: Set of (r, c) dug out
    """
    grid = normalize_grid(level)
    R = len(grid)
    C = len(grid[0])

    spawns = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == "M"]
    golds = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == "G"]

    spawn_ok = len(spawns) == 1
    if not spawn_ok:
        return {
            "spawn_ok": False,
            "n_gold": len(golds),
            "reachable_cells": set(),
            "unreachable_gold": golds,
            "gold_reachable_fraction": 0.0,
            "is_playable": False,
            "opened_cells": set(),
        }

    start = spawns[0]
    reachable_cells, opened_cells = find_reachable_with_digging(start, grid)
    unreachable_gold = [g for g in golds if g not in reachable_cells]
    gold_reachable_frac = (
        (len(golds) - len(unreachable_gold)) / len(golds) if len(golds) > 0 else 1.0
    )
    is_playable = spawn_ok and (len(golds) >= 1) and (len(unreachable_gold) == 0)

    return {
        "spawn_ok": spawn_ok,
        "n_gold": len(golds),
        "reachable_cells": reachable_cells,
        "unreachable_gold": unreachable_gold,
        "gold_reachable_fraction": gold_reachable_frac,
        "is_playable": is_playable,
        "opened_cells": opened_cells,
    }


def is_completable(level: LevelType) -> bool:
    """Strict reachability check: detects one-way dead-end traps containing gold.

    A level is completable if:
    1. It is playable (all gold is reachable).
    2. At most one terminal Strongly Connected Component (SCC) in the directed reachability
       graph contains gold (meaning gold in dead-end pits can be collected last).
    """
    res = analyze(level)
    if not res["is_playable"]:
        return False

    grid = normalize_grid(level)
    reachable = res["reachable_cells"]
    opened = res["opened_cells"]
    golds = set(
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] == "G"
    )

    # Build directed graph restricted to reachable states
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for node in reachable:
        adj[node] = [
            nxt
            for nxt in get_transitions(node[0], node[1], grid, opened)
            if nxt in reachable
        ]

    # Tarjan's SCC algorithm
    index = 0
    indices: Dict[Tuple[int, int], int] = {}
    lowlink: Dict[Tuple[int, int], int] = {}
    on_stack: Dict[Tuple[int, int], bool] = {}
    stack: List[Tuple[int, int]] = []
    sccs: List[List[Tuple[int, int]]] = []

    def strongconnect(v: Tuple[int, int]):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for node in reachable:
        if node not in indices:
            strongconnect(node)

    # Map each node to its SCC id
    node_to_scc: Dict[Tuple[int, int], int] = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = i

    # Determine which SCCs are terminal (no outgoing edge to a different SCC)
    is_terminal = [True] * len(sccs)
    for u, neighbors in adj.items():
        scc_u = node_to_scc[u]
        for v in neighbors:
            scc_v = node_to_scc[v]
            if scc_u != scc_v:
                is_terminal[scc_u] = False

    # Count terminal SCCs that contain at least one gold piece
    terminal_sccs_with_gold = 0
    for i, scc in enumerate(sccs):
        if is_terminal[i]:
            if any(node in golds for node in scc):
                terminal_sccs_with_gold += 1

    # Completable if at most one terminal SCC contains gold
    return terminal_sccs_with_gold <= 1


def astar_shortest_path(
    src: Tuple[int, int],
    is_dst_func,
    adj_func,
    heuristic_func,
) -> Optional[List[Tuple[int, int]]]:
    """Clean heapq A* shortest path finder.

    Args:
        src: Starting coordinate (r, c).
        is_dst_func: Predicate function f(node) -> bool.
        adj_func: Function f(node) -> List[node].
        heuristic_func: Function f(node) -> float.

    Returns:
        List of coordinates along the shortest path, or None if unreachable.
    """
    if is_dst_func(src):
        return [src]

    counter = 0
    # Entry: (f_score, g_score, counter, node, path)
    start_f = heuristic_func(src)
    heap = [(start_f, 0, counter, src, [src])]
    best_g: Dict[Tuple[int, int], int] = {src: 0}

    while heap:
        f, g, _, curr, path = heapq.heappop(heap)
        if g > best_g.get(curr, float("inf")):
            continue

        if is_dst_func(curr):
            return path

        for nxt in adj_func(curr):
            nxt_g = g + 1
            if nxt_g < best_g.get(nxt, float("inf")):
                best_g[nxt] = nxt_g
                counter += 1
                nxt_f = nxt_g + heuristic_func(nxt)
                heapq.heappush(heap, (nxt_f, nxt_g, counter, nxt, path + [nxt]))

    return None


def solve_path(
    level: LevelType,
) -> Tuple[List[Tuple[int, int]], int, List[Tuple[int, int]]]:
    """Greedy nearest-gold-first tour of the level using A* with Manhattan heuristic.

    Returns:
        (path, total_steps, collected_order):
          - path: full sequence of (r, c) states traversed (for animation)
          - total_steps: total number of movement actions (len(path) - 1)
          - collected_order: list of gold coordinates in the order collected
    """
    analysis = analyze(level)
    if not analysis["is_playable"]:
        return [], 0, []

    grid = normalize_grid(level)
    opened = analysis["opened_cells"]
    spawns = [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] == "M"
    ]
    curr = spawns[0]

    remaining_gold = set(
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] == "G"
    )

    full_path: List[Tuple[int, int]] = [curr]
    collected_order: List[Tuple[int, int]] = []

    def adj(node):
        return get_transitions(node[0], node[1], grid, opened)

    while remaining_gold:
        # Target closest gold by Manhattan distance as heuristic priority
        def heuristic(node):
            if not remaining_gold:
                return 0
            return min(
                abs(node[0] - g[0]) + abs(node[1] - g[1]) for g in remaining_gold
            )

        def is_dst(node):
            return node in remaining_gold

        leg = astar_shortest_path(curr, is_dst, adj, heuristic)
        if not leg or len(leg) <= 1:
            # Fallback BFS to find any reachable remaining gold
            q = deque([(curr, [curr])])
            seen = {curr}
            found_leg = None
            while q:
                c_node, c_path = q.popleft()
                if c_node in remaining_gold:
                    found_leg = c_path
                    break
                for nxt in adj(c_node):
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, c_path + [nxt]))
            if not found_leg:
                break
            leg = found_leg

        destination = leg[-1]
        collected_order.append(destination)
        remaining_gold.remove(destination)
        full_path.extend(leg[1:])
        curr = destination

    total_steps = max(0, len(full_path) - 1)
    return full_path, total_steps, collected_order
