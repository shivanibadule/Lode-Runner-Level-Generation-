"""Test suite and diagnostic logger for solver.py.
Categorizes level failures into:
- INVALID_ACTIONS
- NO_SOLUTION_FOUND
- INCORRECT_GOAL_TEST
- TIMEOUT
Saves complete audit to outputs/solver_report.json and outputs/clean_levels.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import constants
import solver


def create_mini_levels():
    def empty_level():
        return [list("." * constants.COLS) for _ in range(constants.ROWS)]

    # 1. Solvable Mini-Level
    lvl1 = empty_level()
    for c in range(constants.COLS):
        lvl1[21][c] = "b"  # Ground floor
    lvl1[20][5] = "M"     # Player spawn
    lvl1[20][2] = "G"     # Ground gold
    for c in range(8, 16):
        lvl1[15][c] = "b" # Upper platform
    for r in range(15, 21):
        lvl1[r][10] = "#" # Ladder
    lvl1[14][8] = "G"     # Gold on platform

    # 2. Gold sealed in indestructible 'B'
    lvl2 = [row.copy() for row in lvl1]
    lvl2[14][8] = "G"
    lvl2[13][8] = "B"
    lvl2[14][7] = "B"
    lvl2[14][9] = "B"
    lvl2[15][8] = "B"

    # 3. Ladder one tile short of platform (unreachable)
    lvl3 = empty_level()
    for c in range(constants.COLS):
        lvl3[21][c] = "b"
    lvl3[20][5] = "M"
    for c in range(8, 16):
        lvl3[14][c] = "b"
    lvl3[13][10] = "G"
    for r in range(16, 21): # Ladder ends 1 tile short of row 14
        lvl3[r][10] = "#"

    # 4. Multi-trap level (INCORRECT_GOAL_TEST: 2 dead-end pits each with gold)
    lvl4 = empty_level()
    for c in range(constants.COLS):
        lvl4[21][c] = "B"
    lvl4[20][16] = "M" # Player in middle
    # Pit 1 on left
    lvl4[20][5] = "G"
    lvl4[19][4] = "B"; lvl4[20][4] = "B"; lvl4[19][6] = "B"; lvl4[20][6] = "B"
    # Pit 2 on right
    lvl4[20][25] = "G"
    lvl4[19][24] = "B"; lvl4[20][24] = "B"; lvl4[19][26] = "B"; lvl4[20][26] = "B"

    return {
        "Mini 1 (Solvable)": ["".join(r) for r in lvl1],
        "Mini 2 (Gold sealed in B)": ["".join(r) for r in lvl2],
        "Mini 3 (Ladder short by 1 tile)": ["".join(r) for r in lvl3],
        "Mini 4 (Two dead-end gold pits)": ["".join(r) for r in lvl4],
    }


def classify_level(level_name: str, content: str) -> Dict[str, Any]:
    """Run analysis and solver, and classify result into failure taxonomy."""
    analysis = solver.analyze(content)
    comp = solver.is_completable(content)

    if not analysis["spawn_ok"] or analysis["n_gold"] == 0:
        return {
            "level": level_name,
            "status": "FAIL",
            "reason": "INVALID_ACTIONS",
            "detail": f"spawn_ok={analysis['spawn_ok']}, n_gold={analysis['n_gold']}",
            "gold_reachable_frac": analysis["gold_reachable_fraction"],
            "unreachable_gold": analysis["unreachable_gold"],
        }

    if not analysis["is_playable"]:
        return {
            "level": level_name,
            "status": "FAIL",
            "reason": "NO_SOLUTION_FOUND",
            "detail": f"{len(analysis['unreachable_gold'])} of {analysis['n_gold']} gold unreachable",
            "gold_reachable_frac": analysis["gold_reachable_fraction"],
            "unreachable_gold": analysis["unreachable_gold"],
        }

    if not comp:
        return {
            "level": level_name,
            "status": "FAIL",
            "reason": "INCORRECT_GOAL_TEST",
            "detail": "Multiple terminal sink SCCs contain gold (one-way trap hazard)",
            "gold_reachable_frac": analysis["gold_reachable_fraction"],
            "unreachable_gold": [],
        }

    # Test A* path finding for timeouts
    path, steps, order = solver.solve_path(content)
    if len(order) < analysis["n_gold"]:
        return {
            "level": level_name,
            "status": "FAIL",
            "reason": "TIMEOUT",
            "detail": f"A* collected only {len(order)} of {analysis['n_gold']} gold",
            "gold_reachable_frac": len(order) / analysis["n_gold"],
            "unreachable_gold": [],
        }

    return {
        "level": level_name,
        "status": "PASS",
        "reason": "NONE",
        "detail": f"Solved in {steps} steps",
        "gold_reachable_frac": 1.0,
        "unreachable_gold": [],
        "steps": steps,
        "order": order,
    }


def run_diagnostics():
    print("=" * 75)
    print("1. RUNNING SYNTHETIC MINI-LEVEL DIAGNOSTICS")
    print("=" * 75)
    mini_levels = create_mini_levels()
    for name, lines in mini_levels.items():
        res = classify_level(name, "\n".join(lines))
        print(f"[{name}] -> Status: {res['status']}, Reason: {res['reason']}")
        print(f"  Detail: {res['detail']}")

    print("\n" + "=" * 75)
    print("2. AUDITING ALL 150 REAL LEVELS")
    print("=" * 75)

    level_files = sorted(
        list(constants.LEVELS_DIR.glob("*.txt")),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )

    results = []
    category_counts = {
        "PASS": 0,
        "NO_SOLUTION_FOUND": 0,
        "INVALID_ACTIONS": 0,
        "INCORRECT_GOAL_TEST": 0,
        "TIMEOUT": 0,
    }

    clean_levels = []
    for f in level_files:
        content = f.read_text(encoding="utf-8")
        res = classify_level(f.name, content)
        results.append(res)
        if res["status"] == "PASS":
            category_counts["PASS"] += 1
            clean_levels.append(f.name)
        else:
            category_counts[res["reason"]] += 1

    total = len(level_files)
    print("\n--- SUMMARY BREAKDOWN ---")
    print(f"Total Levels:           {total}")
    print(f"PASS (Clean & Solved):  {category_counts['PASS']} ({category_counts['PASS']/total*100:.1f}%)")
    print(f"NO_SOLUTION_FOUND:      {category_counts['NO_SOLUTION_FOUND']} ({category_counts['NO_SOLUTION_FOUND']/total*100:.1f}%)")
    print(f"INVALID_ACTIONS:        {category_counts['INVALID_ACTIONS']} ({category_counts['INVALID_ACTIONS']/total*100:.1f}%)")
    print(f"INCORRECT_GOAL_TEST:    {category_counts['INCORRECT_GOAL_TEST']} ({category_counts['INCORRECT_GOAL_TEST']/total*100:.1f}%)")
    print(f"TIMEOUT:                {category_counts['TIMEOUT']} ({category_counts['TIMEOUT']/total*100:.1f}%)")

    # Save outputs
    report_path = constants.OUTPUTS_DIR / "solver_report.json"
    clean_path = constants.OUTPUTS_DIR / "clean_levels.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "counts": category_counts,
                "levels": results,
            },
            f,
            indent=2,
        )

    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "clean_count": len(clean_levels),
                "clean_levels": clean_levels,
            },
            f,
            indent=2,
        )

    print(f"\nSaved solver report to: {report_path}")
    print(f"Saved clean subset ({len(clean_levels)} levels) to: {clean_path}")


if __name__ == "__main__":
    run_diagnostics()
