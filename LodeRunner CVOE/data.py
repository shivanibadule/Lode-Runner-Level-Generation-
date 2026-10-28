"""Dataset and Data Loading for Lode Runner CVAE.

Features:
- One-hot encoding of 22x32 levels into (8, 22, 32) tensors.
- Train/Validation split by level (85% / 15%).
- Horizontal-flip augmentation on the training set only.
- 5-dimensional condition vector c: [gold_count, ladder_count, rope_count, enemy_count, solid_density],
  normalized by dataset maximums.
- Supports both all-levels training and clean-subset training via flag.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import constants
import solver


def extract_features(grid: List[str]) -> np.ndarray:
    """Extract raw condition features from 22x32 character grid:
    [gold_count, ladder_count, rope_count, enemy_count, solid_density]
    """
    total_cells = constants.ROWS * constants.COLS
    gold_count = sum(row.count("G") for row in grid)
    ladder_count = sum(row.count("#") for row in grid)
    rope_count = sum(row.count("-") for row in grid)
    enemy_count = sum(row.count("E") for row in grid)
    solid_count = sum(sum(1 for c in row if c in constants.SOLID) for row in grid)
    solid_density = solid_count / total_cells

    return np.array(
        [gold_count, ladder_count, rope_count, enemy_count, solid_density],
        dtype=np.float32,
    )


def grid_to_onehot(grid: List[str]) -> np.ndarray:
    """Convert (22, 32) char grid to (8, 22, 32) one-hot float32 array."""
    onehot = np.zeros((constants.NUM_CLASSES, constants.ROWS, constants.COLS), dtype=np.float32)
    for r in range(constants.ROWS):
        for c in range(constants.COLS):
            tile = grid[r][c]
            idx = constants.TILE_TO_IDX.get(tile, constants.TILE_TO_IDX["."])
            onehot[idx, r, c] = 1.0
    return onehot


def onehot_to_grid(onehot: np.ndarray) -> List[str]:
    """Convert (8, 22, 32) one-hot or probability array back to list of strings."""
    indices = np.argmax(onehot, axis=0) # (22, 32)
    lines = []
    for r in range(constants.ROWS):
        chars = [constants.IDX_TO_TILE[indices[r, c]] for c in range(constants.COLS)]
        lines.append("".join(chars))
    return lines


class LodeRunnerDataset(Dataset):
    """PyTorch Dataset for Lode Runner levels and condition vectors."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, np.ndarray, str]],
        max_condition: np.ndarray,
        is_train: bool = True,
        augment_flip: bool = True,
    ):
        self.is_train = is_train
        self.augment_flip = augment_flip and is_train
        self.max_condition = max_condition

        self.data: List[Tuple[np.ndarray, np.ndarray, str]] = []
        for x, c_raw, name in samples:
            # Normalize condition vector by dataset MAX
            c_norm = c_raw / np.maximum(self.max_condition, 1e-6)
            self.data.append((x, c_norm.astype(np.float32), name))
            if self.augment_flip:
                # Horizontal flip: flip width dimension (axis 2)
                x_flipped = np.flip(x, axis=2).copy()
                # Condition features (counts and density) remain invariant under horizontal flip
                self.data.append((x_flipped, c_norm.astype(np.float32), f"{name}_flip"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        x, c, name = self.data[idx]
        return torch.from_numpy(x), torch.from_numpy(c), name


def get_dataset_splits(
    levels_dir: Path = constants.LEVELS_DIR,
    use_clean_subset: bool = False,
    clean_levels_json: Path = constants.OUTPUTS_DIR / "clean_levels.json",
    train_ratio: float = 0.85,
    seed: int = 42,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Prepare train and validation DataLoaders with condition normalization."""
    rng = np.random.RandomState(seed)

    # 1. Collect files
    if use_clean_subset and clean_levels_json.exists():
        with open(clean_levels_json, "r", encoding="utf-8") as f:
            clean_meta = json.load(f)
        target_files = [levels_dir / name for name in clean_meta["clean_levels"]]
    else:
        target_files = sorted(list(levels_dir.glob("*.txt")))

    raw_samples = []
    raw_conditions = []

    for p in target_files:
        content = p.read_text(encoding="utf-8")
        grid = solver.normalize_grid(content)
        onehot = grid_to_onehot(grid)
        cond = extract_features(grid)
        raw_samples.append((onehot, cond, p.name))
        raw_conditions.append(cond)

    raw_conditions_arr = np.array(raw_conditions, dtype=np.float32)
    max_condition = np.max(raw_conditions_arr, axis=0)
    # Ensure no division by zero
    max_condition[max_condition == 0] = 1.0

    # 2. Train / Val split by level (deterministic)
    indices = np.arange(len(raw_samples))
    rng.shuffle(indices)

    n_train = int(round(len(raw_samples) * train_ratio))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_samples = [raw_samples[i] for i in train_idx]
    val_samples = [raw_samples[i] for i in val_idx]

    train_ds = LodeRunnerDataset(
        train_samples, max_condition, is_train=True, augment_flip=True
    )
    val_ds = LodeRunnerDataset(
        val_samples, max_condition, is_train=False, augment_flip=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    metadata = {
        "total_source_levels": len(raw_samples),
        "use_clean_subset": use_clean_subset,
        "n_train_levels": len(train_samples),
        "n_train_samples_augmented": len(train_ds),
        "n_val_levels": len(val_samples),
        "max_condition": max_condition.tolist(),
        "condition_names": [
            "gold_count",
            "ladder_count",
            "rope_count",
            "enemy_count",
            "solid_density",
        ],
    }

    return train_loader, val_loader, metadata
