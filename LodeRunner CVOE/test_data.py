"""Unit tests for data.py."""

import torch
import numpy as np
import constants
import data


def test_data_pipeline():
    print("Testing data pipeline on all levels...")
    train_loader, val_loader, meta = data.get_dataset_splits(
        use_clean_subset=False, train_ratio=0.85, seed=42, batch_size=16
    )

    print("Dataset metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # Inspect a training batch
    for x_batch, c_batch, names in train_loader:
        print(f"\nTraining batch shapes:")
        print(f"  x_batch: {x_batch.shape} (dtype: {x_batch.dtype})")
        print(f"  c_batch: {c_batch.shape} (dtype: {c_batch.dtype})")
        print(f"  names (first 4): {names[:4]}")
        
        # Verify one-hot validity: sum across channels at each cell should be 1.0
        channel_sums = torch.sum(x_batch, dim=1)
        assert torch.allclose(channel_sums, torch.ones_like(channel_sums)), "One-hot sum across channels != 1"
        
        # Verify condition vector in [0, 1]
        assert (c_batch >= 0.0).all() and (c_batch <= 1.0).all(), "Condition values out of range [0, 1]"
        break

    # Inspect validation batch
    for x_val, c_val, names in val_loader:
        print(f"\nValidation batch shapes:")
        print(f"  x_val: {x_val.shape}")
        print(f"  c_val: {c_val.shape}")
        break

    print("\nData module successfully validated!")


if __name__ == "__main__":
    test_data_pipeline()
