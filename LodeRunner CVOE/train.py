"""Training pipeline for Lode Runner CVAE.

Features:
- Class-weighted Cross-Entropy loss (sqrt inverse frequency clipped to max 20).
- Linear KL annealing (0 -> beta_max over the first 30% of epochs).
- Early stopping on validation loss (patience 40).
- Periodic visual logging: saves colored tile grid comparisons (original vs reconstructed) every 25 epochs.
- Hyperparameter sweep over beta_max in {0.25, 0.5, 1.0}.
- Selects the best checkpoint based on generated level playability rate (tie-breaker: lowest val loss).
- Stores all checkpoints, loss curves, and diagnostic JSONs in ./outputs/.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import constants
import solver
import data
from model import LodeRunnerCVAE


def compute_class_weights(train_loader: data.DataLoader, clip_max: float = 20.0) -> torch.Tensor:
    """Compute class weights from sqrt inverse frequency, clipped to clip_max."""
    counts = np.zeros(constants.NUM_CLASSES, dtype=np.float64)
    for x_batch, _, _ in train_loader:
        # x_batch: (B, 8, 22, 32)
        indices = torch.argmax(x_batch, dim=1).numpy()
        for idx in range(constants.NUM_CLASSES):
            counts[idx] += np.sum(indices == idx)

    total_tiles = counts.sum()
    freqs = counts / total_tiles
    # Sqrt inverse frequency
    weights = np.sqrt(np.max(freqs) / np.maximum(freqs, 1e-6))
    weights = np.clip(weights, 1.0, clip_max)
    # Normalize so mean weight is around 1.0 - 2.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def render_level_image(grid: List[str]) -> np.ndarray:
    """Render a 22x32 character level into an RGB image array (scaled up by tile size)."""
    tile_size = 12
    img = np.zeros((constants.ROWS * tile_size, constants.COLS * tile_size, 3), dtype=np.uint8)
    for r in range(constants.ROWS):
        for c in range(constants.COLS):
            tile = grid[r][c]
            rgb = constants.TILE_COLORS.get(tile, (0, 0, 0))
            img[r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size] = rgb
    return img


def save_reconstruction_grid(
    model: LodeRunnerCVAE,
    val_loader: data.DataLoader,
    epoch: int,
    output_dir: Path = constants.OUTPUTS_DIR,
):
    """Save side-by-side comparison of original vs reconstructed levels."""
    model.eval()
    with torch.no_grad():
        for x_batch, c_batch, names in val_loader:
            x_sample = x_batch[:4]
            c_sample = c_batch[:4]
            logits, _, _ = model(x_sample, c_sample)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = torch.argmax(x_sample, dim=1).cpu().numpy()
            break

    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    fig.suptitle(f"Epoch {epoch}: Originals (Left) vs Reconstructions (Right)", fontsize=14)

    for i in range(min(4, len(preds))):
        orig_grid = [
            "".join(constants.IDX_TO_TILE[targets[i, r, c]] for c in range(constants.COLS))
            for r in range(constants.ROWS)
        ]
        recon_grid = [
            "".join(constants.IDX_TO_TILE[preds[i, r, c]] for c in range(constants.COLS))
            for r in range(constants.ROWS)
        ]

        axes[i, 0].imshow(render_level_image(orig_grid))
        axes[i, 0].set_title(f"Original {names[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(render_level_image(recon_grid))
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")

    plt.tight_layout()
    save_path = output_dir / f"reconstructions_epoch_{epoch:03d}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()


def train_single_model(
    beta_max: float,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    meta: Dict[str, Any],
    epochs: int = 150,
    patience: int = 40,
    lr: float = 1e-3,
    device: str = "cpu",
    output_dir: Path = constants.OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Train a CVAE instance with a specified beta_max and evaluate generated playability."""
    print(f"\n" + "=" * 70)
    print(f"TRAINING CVAE WITH beta_max = {beta_max}")
    print("=" * 70)

    model = LodeRunnerCVAE(num_classes=8, cond_dim=5, latent_dim=32, dropout_rate=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights = compute_class_weights(train_loader).to(device)

    anneal_epochs = max(1, int(0.3 * epochs))

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "val_recon": [],
        "train_kl": [],
        "val_kl": [],
        "beta": [],
    }

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # KL Annealing schedule
        beta = beta_max * min(1.0, epoch / anneal_epochs)

        # Training phase
        model.train()
        train_recon_total = 0.0
        train_kl_total = 0.0
        n_train_samples = 0

        for x_batch, c_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            targets = torch.argmax(x_batch, dim=1) # (B, 22, 32)

            optimizer.zero_grad()
            logits, mu, logvar = model(x_batch, c_batch)

            recon_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction="mean")
            # KL divergence per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

            total_loss = recon_loss + beta * kl_loss
            total_loss.backward()
            optimizer.step()

            B = x_batch.size(0)
            train_recon_total += recon_loss.item() * B
            train_kl_total += kl_loss.item() * B
            n_train_samples += B

        train_recon_avg = train_recon_total / n_train_samples
        train_kl_avg = train_kl_total / n_train_samples
        train_loss_avg = train_recon_avg + beta * train_kl_avg

        # Validation phase
        model.eval()
        val_recon_total = 0.0
        val_kl_total = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for x_val, c_val, _ in val_loader:
                x_val = x_val.to(device)
                c_val = c_val.to(device)
                targets = torch.argmax(x_val, dim=1)

                logits, mu, logvar = model(x_val, c_val)
                recon_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction="mean")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

                B = x_val.size(0)
                val_recon_total += recon_loss.item() * B
                val_kl_total += kl_loss.item() * B
                n_val_samples += B

        val_recon_avg = val_recon_total / n_val_samples
        val_kl_avg = val_kl_total / n_val_samples
        val_loss_avg = val_recon_avg + beta * val_kl_avg

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["train_recon"].append(train_recon_avg)
        history["val_recon"].append(val_recon_avg)
        history["train_kl"].append(train_kl_avg)
        history["val_kl"].append(val_kl_avg)
        history["beta"].append(beta)

        # Early stopping check
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}/{epochs:3d} | Beta: {beta:.3f} | "
                f"Train Loss: {train_loss_avg:.4f} (Recon: {train_recon_avg:.4f}, KL: {train_kl_avg:.4f}) | "
                f"Val Loss: {val_loss_avg:.4f} (Recon: {val_recon_avg:.4f}, KL: {val_kl_avg:.4f})"
            )
            save_reconstruction_grid(model, val_loader, epoch, output_dir)

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")

    # Load best weights
    model.load_state_dict(best_weights)

    # Evaluate raw sample playability across 100 generated levels
    model.eval()
    print("Generating 100 raw test levels to evaluate playability...")
    playable_count = 0
    rng = np.random.RandomState(42)

    with torch.no_grad():
        for i in range(100):
            # Sample realistic condition vector within [min, max] observed
            sample_cond = rng.uniform(0.1, 0.8, size=(1, 5)).astype(np.float32)
            c_t = torch.from_numpy(sample_cond).to(device)
            z = torch.randn(1, 32, device=device)
            logits = model.decode(z, c_t)
            pred_indices = torch.argmax(logits, dim=1)[0].cpu().numpy()

            grid = [
                "".join(constants.IDX_TO_TILE[pred_indices[r, c]] for c in range(constants.COLS))
                for r in range(constants.ROWS)
            ]
            res = solver.analyze(grid)
            if res["is_playable"]:
                playable_count += 1

    playability_rate = playable_count / 100.0
    print(f"Raw Playability Rate for beta_max={beta_max}: {playability_rate * 100:.1f}%")

    return {
        "beta_max": beta_max,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "playability_rate": playability_rate,
        "duration_seconds": duration,
        "history": history,
        "state_dict": best_weights,
    }


def plot_loss_curves(history: Dict[str, List[float]], save_path: Path):
    """Plot and save loss curves (Reconstruction, KL, Total)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Total loss
    axes[0].plot(epochs, history["train_loss"], label="Train Total", color="#1f77b4", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Total", color="#ff7f0e", lw=2)
    axes[0].set_title("Total Loss (Recon + Beta * KL)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Reconstruction Loss
    axes[1].plot(epochs, history["train_recon"], label="Train Recon (CE)", color="#2ca02c", lw=2)
    axes[1].plot(epochs, history["val_recon"], label="Val Recon (CE)", color="#d62728", lw=2)
    axes[1].set_title("Reconstruction Loss (Weighted Cross-Entropy)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # KL Divergence
    axes[2].plot(epochs, history["train_kl"], label="Train KL", color="#9467bd", lw=2)
    axes[2].plot(epochs, history["val_kl"], label="Val KL", color="#8c564b", lw=2)
    axes[2].plot(epochs, history["beta"], label="Beta Schedule", color="#e377c2", ls="--")
    axes[2].set_title("KL Divergence & Beta Annealing")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Value")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_training_sweep():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader, meta = data.get_dataset_splits(
        use_clean_subset=False, train_ratio=0.85, seed=42, batch_size=16
    )

    betas = [0.25, 0.5, 1.0]
    sweep_results = []
    best_overall_model = None
    best_score = (-1.0, float("inf")) # (playability, -val_loss)

    for b in betas:
        res = train_single_model(
            beta_max=b,
            train_loader=train_loader,
            val_loader=val_loader,
            meta=meta,
            epochs=150,
            patience=40,
            lr=1e-3,
        )
        sweep_results.append({
            "beta_max": res["beta_max"],
            "best_epoch": res["best_epoch"],
            "best_val_loss": res["best_val_loss"],
            "playability_rate": res["playability_rate"],
            "duration_seconds": res["duration_seconds"],
        })

        score = (res["playability_rate"], -res["best_val_loss"])
        if score > best_score:
            best_score = score
            best_overall_model = res

    # Save best checkpoint
    best_ckpt_path = constants.OUTPUTS_DIR / "best.pt"
    torch.save(
        {
            "state_dict": best_overall_model["state_dict"],
            "beta_max": best_overall_model["beta_max"],
            "best_epoch": best_overall_model["best_epoch"],
            "best_val_loss": best_overall_model["best_val_loss"],
            "playability_rate": best_overall_model["playability_rate"],
            "max_condition": meta["max_condition"],
            "condition_names": meta["condition_names"],
            "cond_dim": 5,
            "latent_dim": 32,
            "num_classes": 8,
        },
        best_ckpt_path,
    )
    print(f"\nSaved best model (beta_max={best_overall_model['beta_max']}) to: {best_ckpt_path}")

    # Save loss curves for best model
    loss_curve_path = constants.OUTPUTS_DIR / "loss_curve.png"
    plot_loss_curves(best_overall_model["history"], loss_curve_path)
    print(f"Saved loss curve to: {loss_curve_path}")

    # Save sweep results and configuration
    config_path = constants.OUTPUTS_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_beta_max": best_overall_model["beta_max"],
                "best_epoch": best_overall_model["best_epoch"],
                "best_val_loss": best_overall_model["best_val_loss"],
                "playability_rate": best_overall_model["playability_rate"],
                "latent_dim": 32,
                "cond_dim": 5,
                "max_condition": meta["max_condition"],
                "condition_names": meta["condition_names"],
                "sweep": sweep_results,
            },
            f,
            indent=2,
        )
    print(f"Saved config and sweep summary to: {config_path}")


if __name__ == "__main__":
    run_training_sweep()
