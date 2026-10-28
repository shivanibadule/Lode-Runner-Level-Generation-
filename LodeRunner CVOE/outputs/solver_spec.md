# Solver Specification: Lode Runner Reachability & Tour Finder

Following Sorochan et al. (FDG 2021) and VGLC specifications:

## 1. State Representation
- **State**: `(row, col)` tuple where `0 <= row < 22` and `0 <= col < 32`.
- **Dynamic Entities**: Enemies (`'E'`) are treated as open air in accordance with Sorochan et al. (2021) and Snodgrass & Ontanón (2016) to avoid artificial instant-death false negatives in static reachability.
- **Digging State**: Tracked via monotone fixpoint `opened` set of `(r, c)` coordinates.

## 2. Movement & Physics Rules
- **Support Condition `is_supported(r, c)`**:
  - The cell contains a ladder `'#'` or rope `'-'`, OR
  - The cell below `(r+1, c)` is solid (`'B'` or `'b'`), OR ladder `'#'`, OR
  - `r == ROWS - 1` (bottom row rests on the ground boundary of the game screen).
- **Transitions**:
  - **Unsupported**: The only valid move is falling straight down to `(r+1, c)` if it is non-solid. Fall ceases when landing on ground, ladder, or rope.
  - **Supported**: Can move left `(r, c-1)` or right `(r, c+1)` into any non-solid cell.
  - **On Ladder `'#'`**: Can climb up to `(r-1, c)` or down to `(r+1, c)` into any non-solid cell. (Includes stepping out of ladder top into open air).
  - **On Rope `'-'`**: Can move horizontally or drop down to `(r+1, c)`.
  - **Above Ladder**: Standing in open air directly above a ladder `(r+1, c) == '#'` allows stepping down onto the ladder.
  - **Out of Bounds**: Moves beyond the 22x32 grid boundaries are disallowed.
- **Digging Rules**:
  - Only `'b'` (diggable brick) can be dug; `'B'` (bedrock) is indestructible.
  - From a supported position `(r, c)`, the player can dig `(r+1, c-1)` or `(r+1, c+1)` if that tile is `'b'` and the diagonally adjacent tile `(r, c±1)` is clear.
  - Dug tiles are added to the opened set and treated as passable air in subsequent reachability steps.

## 3. Heuristic & Tour Search (A*)
- **Leg Objective**: Shortest path from current player position to nearest uncollected gold.
- **Heuristic**: Admissible Manhattan distance:
  $$h((r, c)) = \min_{g \in \text{RemainingGold}} (|r - g_r| + |c - g_c|)$$
- **Tour Policy**: Greedy nearest-gold-first tour until all reachable gold is collected.

## 4. Failure Classification Taxonomy
- `INVALID_ACTIONS`: Structural level error (e.g., spawn count $\neq 1$, no gold).
- `NO_SOLUTION_FOUND`: One or more gold pieces are physically unreachable (e.g. enclosed in indestructible bedrock or isolated on inaccessible platforms).
- `INCORRECT_GOAL_TEST`: Completable check failure (gold located in two or more separate terminal sink SCCs; collecting one permanently traps the player from the others).
- `TIMEOUT`: Exceeded maximum search iteration budget (default: 10,000 steps).
