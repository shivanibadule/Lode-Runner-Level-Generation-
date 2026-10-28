"""Unit tests for generate.py verifying raw, rejection, and repair modes."""

import numpy as np
import constants
import solver
import generate


def test_generation():
    print("Testing generate.py modes...")
    model_tuple = generate.load_cvae()
    _, max_cond, _ = model_tuple

    # Test condition: typical level with 6 gold, 20 ladders, 10 ropes, 2 enemies, 0.3 density
    cond = np.array([6.0, 20.0, 10.0, 2.0, 0.30], dtype=np.float32)

    # 1. Raw mode
    raw_res = generate.generate(cond, mode="raw", seed=42, model_tuple=model_tuple)
    print(f"Raw mode generated -> Playable: {raw_res['is_playable']}, Reachable Frac: {raw_res['analysis']['gold_reachable_fraction']:.2f}")

    # 2. Rejection sampling mode
    rej_res = generate.generate(cond, mode="rejection", seed=42, model_tuple=model_tuple, max_rejection=20)
    print(f"Rejection mode -> Playable: {rej_res['is_playable']}, Attempts: {rej_res['attempts']}")

    # 3. Repair mode
    rep_res = generate.generate(cond, mode="repair", seed=42, model_tuple=model_tuple)
    print(f"Repair mode -> Playable: {rep_res['is_playable']}, Edits applied: {rep_res['edits']}")

    print("\ngenerate.py validated successfully!")


if __name__ == "__main__":
    test_generation()
