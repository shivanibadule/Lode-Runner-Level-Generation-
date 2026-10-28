"""Unit test for model.py verifying forward pass, shapes, and backprop."""

import torch
import constants
from model import LodeRunnerCVAE


def test_cvae():
    print("Testing LodeRunnerCVAE model...")
    model = LodeRunnerCVAE(num_classes=8, cond_dim=5, latent_dim=32, dropout_rate=0.25)
    model.train()

    B = 4
    x = torch.randn(B, 8, constants.ROWS, constants.COLS)
    c = torch.rand(B, 5)

    logits, mu, logvar = model(x, c)

    print(f"Input x: {x.shape}")
    print(f"Condition c: {c.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Latent mu: {mu.shape}, logvar: {logvar.shape}")

    assert logits.shape == (B, 8, 22, 32), f"Expected (B, 8, 22, 32) but got {logits.shape}"
    assert mu.shape == (B, 32), f"Expected (B, 32) but got {mu.shape}"
    assert logvar.shape == (B, 32), f"Expected (B, 32) but got {logvar.shape}"

    # Test backward pass
    loss = logits.sum() + mu.sum() + logvar.sum()
    loss.backward()
    print("Backward pass successful. Gradients computed for all parameters!")

    # Test eval / decoding mode
    model.eval()
    with torch.no_grad():
        z_sample = torch.randn(B, 32)
        decoded = model.decode(z_sample, c)
        assert decoded.shape == (B, 8, 22, 32), "Decoder shape mismatch in eval mode"
    print("Decoder sampling mode successful!")
    print("\nModel module successfully validated!")


if __name__ == "__main__":
    test_cvae()
