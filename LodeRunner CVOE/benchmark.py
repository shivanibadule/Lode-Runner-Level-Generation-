"""Benchmark evaluation module comparing Real levels vs Raw vs Rejection vs Repair CVAE modes.

Metrics computed across N=100 samples per mode:
1. Playability Rate (%)
2. Completable Rate (%)
3. Mean Gold Reachable Fraction
4. Novelty (% not exact copies of training levels)
5. Diversity (Mean Pairwise Tile Difference)
6. Mean Solver Path Length

Saves:
- outputs/benchmark_results.csv
- outputs/benchmark_comparison.png
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants
import solver
import generate


def compute_novelty(sample_grid: List[str], real_grids: List[List[str]]) -> bool:
    """Return True if sample_grid is not an exact copy of any real level."""
    sample_str = "".join(sample_grid)
    for real in real_grids:
        if sample_str == "".join(real):
            return False
    return True


def compute_diversity(grids: List[List[str]], max_pairs: int = 500) -> float:
    """Compute mean pairwise normalized Hamming tile distance across levels."""
    N = len(grids)
    if N <= 1:
        return 0.0
    total_cells = constants.ROWS * constants.COLS

    # Convert to integer arrays
    arrays = []
    for g in grids:
        arr = np.array([[constants.TILE_TO_IDX[c] for c in row] for row in g], dtype=np.int32)
        arrays.append(arr)

    rng = np.random.RandomState(42)
    diffs = []
    pairs_tested = 0
    while pairs_tested < max_pairs:
        i, j = rng.choice(N, size=2, replace=False)
        diff = np.mean(arrays[i] != arrays[j])
        diffs.append(diff)
        pairs_tested += 1

    return float(np.mean(diffs))


def run_benchmark(N: int = 50): # N=50 for fast execution, robust statistics
    print("=" * 75)
    print(f"RUNNING BENCHMARK EVALUATION (N={N} samples per mode)")
    print("=" * 75)

    model_tuple = generate.load_cvae()
    model, max_cond, _ = model_tuple

    # 1. Load Real Levels
    real_files = sorted(list(constants.LEVELS_DIR.glob("*.txt")))
    real_grids = [solver.normalize_grid(f.read_text(encoding="utf-8")) for f in real_files]

    # Generate fixed random conditions for fair comparison across modes
    rng = np.random.RandomState(42)
    condition_samples = []
    for _ in range(N):
        # Realistic conditions within observed dataset percentiles
        gold = rng.randint(4, 12)
        ladders = rng.randint(15, 35)
        ropes = rng.randint(5, 20)
        enemies = rng.randint(1, 4)
        density = rng.uniform(0.20, 0.40)
        condition_samples.append(np.array([gold, ladders, ropes, enemies, density], dtype=np.float32))

    modes = ["Real Levels", "CVAE (Raw)", "CVAE (Rejection)", "CVAE (Repair)"]
    records = []
    collected_grids: Dict[str, List[List[str]]] = {m: [] for m in modes}

    # Evaluate Real Levels
    print("\nEvaluating Real Levels...")
    real_subset = real_grids[:N]
    collected_grids["Real Levels"] = real_subset
    playable_flags = []
    comp_flags = []
    reach_fracs = []
    path_lengths = []

    for grid in real_subset:
        ana = solver.analyze(grid)
        comp = solver.is_completable(grid) if ana["is_playable"] else False
        playable_flags.append(ana["is_playable"])
        comp_flags.append(comp)
        reach_fracs.append(ana["gold_reachable_fraction"])
        if ana["is_playable"]:
            _, steps, _ = solver.solve_path(grid)
            path_lengths.append(steps)

    records.append({
        "Mode": "Real Levels",
        "Playability Rate (%)": np.mean(playable_flags) * 100,
        "Completable Rate (%)": np.mean(comp_flags) * 100,
        "Gold Reachable (%)": np.mean(reach_fracs) * 100,
        "Novelty (%)": 0.0, # Baseline
        "Diversity": compute_diversity(real_subset),
        "Mean Path Steps": np.mean(path_lengths) if path_lengths else 0.0,
    })

    # Evaluate Synthetic Modes
    mode_keys = [("CVAE (Raw)", "raw"), ("CVAE (Rejection)", "rejection"), ("CVAE (Repair)", "repair")]

    for display_name, mode_key in mode_keys:
        print(f"Evaluating {display_name}...")
        p_flags = []
        c_flags = []
        r_fracs = []
        p_steps = []
        nov_flags = []
        mode_grids = []

        for i, cond in enumerate(condition_samples):
            res = generate.generate(
                cond, mode=mode_key, seed=1000 + i, model_tuple=model_tuple, max_rejection=25
            )
            grid = res["grid"]
            mode_grids.append(grid)
            ana = res["analysis"]
            comp = res["is_completable"]

            p_flags.append(ana["is_playable"])
            c_flags.append(comp)
            r_fracs.append(ana["gold_reachable_fraction"])
            nov_flags.append(compute_novelty(grid, real_grids))

            if ana["is_playable"]:
                _, steps, _ = solver.solve_path(grid)
                p_steps.append(steps)

        collected_grids[display_name] = mode_grids
        records.append({
            "Mode": display_name,
            "Playability Rate (%)": np.mean(p_flags) * 100,
            "Completable Rate (%)": np.mean(c_flags) * 100,
            "Gold Reachable (%)": np.mean(r_fracs) * 100,
            "Novelty (%)": np.mean(nov_flags) * 100,
            "Diversity": compute_diversity(mode_grids),
            "Mean Path Steps": np.mean(p_steps) if p_steps else 0.0,
        })

    # Save to CSV
    df = pd.DataFrame(records)
    csv_path = constants.OUTPUTS_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to: {csv_path}")

    # Print Table
    print("\n" + "=" * 80)
    print(f"{'Mode':<18} {'Playable %':<12} {'Completable %':<15} {'Reachable %':<14} {'Novelty %':<12} {'Diversity':<12} {'Path Steps':<12}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(
            f"{row['Mode']:<18} "
            f"{row['Playability Rate (%)']:<12.1f} "
            f"{row['Completable Rate (%)']:<15.1f} "
            f"{row['Gold Reachable (%)']:<14.1f} "
            f"{row['Novelty (%)']:<12.1f} "
            f"{row['Diversity']:<12.3f} "
            f"{row['Mean Path Steps']:<12.1f}"
        )
    print("=" * 80)

    # Plot Comparison Bar Chart
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Benchmark Comparison: Real vs CVAE Generative Modes", fontsize=16, fontweight="bold")

    metrics_plot = [
        ("Playability Rate (%)", "#1f77b4", axes[0, 0]),
        ("Completable Rate (%)", "#2ca02c", axes[0, 1]),
        ("Gold Reachable (%)", "#ff7f0e", axes[0, 2]),
        ("Novelty (%)", "#9467bd", axes[1, 0]),
        ("Diversity", "#8c564b", axes[1, 1]),
        ("Mean Path Steps", "#e377c2", axes[1, 2]),
    ]

    for metric_name, color, ax in metrics_plot:
        ax.bar(df["Mode"], df[metric_name], color=color, alpha=0.85, edgecolor="black", width=0.55)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    chart_path = constants.OUTPUTS_DIR / "benchmark_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved benchmark comparison chart to: {chart_path}")


if __name__ == "__main__":
    run_benchmark(N=50)
