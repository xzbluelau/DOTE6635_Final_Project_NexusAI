"""
03_model_1dcnn.py — 1D-CNN with residual connections and self-attention for fraud detection.

Architecture (adapted from Mohammed et al. 2026):
    Input: (batch, 1, num_features)  — treat each feature as a 1D signal position
    Conv Block 1:  2x [Conv1d(1→64, k=3, pad=1) → BN → ReLU]
    Residual Block: 2x [Conv1d(64→64, k=3, stride=2, pad=1) → BN → ReLU] + skip (projection)
    Self-Attention: SE-block (squeeze-excite): GAP → FC → ReLU → FC → Sigmoid → reweight
    Classifier:    GAP → FC(64→2)  (binary: fraud / legitimate)

Key fix vs. original MATLAB code: TRUE skip connections with projection shortcuts.

Usage: python code/03_model_1dcnn.py
"""

import os
import sys
import json

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from importlib import import_module
setup = import_module("00_setup")

OUTPUT_FIGURES = setup.OUTPUT_FIGURES
SEED = setup.SEED


# ===========================================================================
# Dataset
# ===========================================================================
class FraudDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for fraud detection features.

    Reshapes flat feature vectors (num_features,) into (1, num_features)
    for Conv1d input: 1 channel, num_features length.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (num_samples, num_features) numpy array
        # y: (num_samples,) numpy array of 0/1 labels
        self.X = torch.FloatTensor(np.array(X, dtype=np.float32)).unsqueeze(1)
        self.y = torch.LongTensor(np.array(y, dtype=np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===========================================================================
# Model Components
# ===========================================================================
class ConvBlock(nn.Module):
    """Two stacked Conv1d → BatchNorm → ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2  # 'same' padding
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with TRUE skip connection (projection shortcut).

    Main path:  Conv1d(stride=2) → BN → ReLU → Conv1d → BN
    Skip path:  Conv1d(1x1, stride=2) to match dimensions
    Output:     ReLU(main + skip)
    """

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2

        # Main path
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

        # Projection shortcut (1x1 conv to match spatial dims after stride)
        self.shortcut = nn.Conv1d(channels, channels, kernel_size=1, stride=stride)

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity  # TRUE skip connection
        out = self.relu_out(out)
        return out


class SelfAttention(nn.Module):
    """Squeeze-and-Excitation (SE) style self-attention.

    GAP → FC(reduction) → ReLU → FC(expansion) → Sigmoid → channel reweighting.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # Squeeze: (N, C, L) → (N, C, 1)
        reduced = max(channels // reduction, 8)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N, C, L)
        b, c, _ = x.shape
        s = self.gap(x).view(b, c)       # (N, C)
        s = self.excitation(s).view(b, c, 1)  # (N, C, 1)
        return x * s                       # channel-wise reweighting


# ===========================================================================
# Full Model
# ===========================================================================
class FraudDetectionCNN1D(nn.Module):
    """Enhanced 1D-CNN with residual connections and self-attention.

    Adapted from Mohammed et al. (2026) with true residual connections.

    Args:
        num_features: Number of input features (sequence length for Conv1d)
        num_classes: Number of output classes (default=2 for binary)
        conv_filters: Number of convolutional filters (default=64)
        kernel_size: Conv1d kernel size (default=3)
        stride: Stride for residual block downsampling (default=2)
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        conv_filters: int = 64,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()
        self.num_features = num_features

        # Initial convolutional block: (N, 1, L) → (N, 64, L)
        self.conv_block1 = ConvBlock(
            in_channels=1,
            out_channels=conv_filters,
            kernel_size=kernel_size,
        )

        # Residual block with true skip: (N, 64, L) → (N, 64, L//2)
        self.residual_block = ResidualBlock(
            channels=conv_filters,
            kernel_size=kernel_size,
            stride=stride,
        )

        # Self-attention (SE-block): (N, 64, L//2) → (N, 64, L//2)
        self.attention = SelfAttention(channels=conv_filters, reduction=4)

        # Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)  # (N, 64, L//2) → (N, 64, 1)
        self.classifier = nn.Linear(conv_filters, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch_size, 1, num_features)

        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.conv_block1(x)       # (N, 64, L)
        x = self.residual_block(x)    # (N, 64, L//2)
        x = self.attention(x)         # (N, 64, L//2) — reweighted
        x = self.gap(x)              # (N, 64, 1)
        x = x.squeeze(-1)            # (N, 64)
        logits = self.classifier(x)  # (N, num_classes)
        return logits


# ===========================================================================
# Verification
# ===========================================================================
def verify_model(num_features: int):
    """Build model, verify forward pass, print summary."""
    print("=" * 60)
    print("MODEL VERIFICATION")
    print("=" * 60)

    model = FraudDetectionCNN1D(num_features=num_features)
    device = setup.DEVICE
    model = model.to(device)

    # Print architecture
    print(f"\nDevice: {device}")
    print(f"Input features: {num_features}")
    print(f"\nModel architecture:\n{model}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Forward pass test
    batch_size = 8
    dummy_input = torch.randn(batch_size, 1, num_features).to(device)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, 2)")
    assert output.shape == (batch_size, 2), f"Shape mismatch! Got {output.shape}"

    # Verify probabilities sum to ~1
    probs = torch.softmax(output, dim=1)
    print(f"  Softmax output sample:\n{probs[:3].cpu().numpy()}")
    print(f"  Prob sums: {probs.sum(dim=1)[:3].cpu().numpy()}")
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size).to(device), atol=1e-5)

    # Layer-by-layer shape trace
    print(f"\nLayer-by-layer shape trace:")
    x = dummy_input
    x = model.conv_block1(x)
    print(f"  After conv_block1: {x.shape}")
    x = model.residual_block(x)
    print(f"  After residual_block: {x.shape}")
    x = model.attention(x)
    print(f"  After self_attention: {x.shape}")
    x = model.gap(x)
    print(f"  After GAP: {x.shape}")
    x = x.squeeze(-1)
    print(f"  After squeeze: {x.shape}")
    x = model.classifier(x)
    print(f"  After classifier: {x.shape}")

    print("\n[PASS] All checks passed!")

    # Save architecture diagram info
    arch_info = {
        "model": "FraudDetectionCNN1D",
        "num_features": num_features,
        "num_classes": 2,
        "conv_filters": 64,
        "kernel_size": 3,
        "residual_stride": 2,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "conv_block1": "2x [Conv1d(1→64, k=3, pad=1) → BN → ReLU]",
        "residual_block": "2x [Conv1d(64→64, k=3, stride=2) → BN → ReLU] + 1x1 projection shortcut",
        "self_attention": "SE-block: GAP → FC(64→16) → ReLU → FC(16→64) → Sigmoid → reweight",
        "classifier": "GAP → FC(64→2)",
    }
    arch_path = os.path.join(setup.MODELS_DIR, "architecture_info.json")
    with open(arch_path, "w") as f:
        json.dump(arch_info, f, indent=2)
    print(f"\nArchitecture info saved to: {arch_path}")

    return model


def main():
    setup.set_seed()

    # Load metadata to get feature count
    meta_path = os.path.join(setup.DATA_PROCESSED, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    num_features = meta["n_features"]

    print(f"Feature count from metadata: {num_features}")

    model = verify_model(num_features)

    # Also verify DataLoader works
    print("\n" + "=" * 60)
    print("DATALOADER VERIFICATION")
    print("=" * 60)

    import pandas as pd

    X_test = pd.read_parquet(
        os.path.join(setup.DATA_PROCESSED, "X_test.parquet")
    ).values
    y_test = pd.read_parquet(
        os.path.join(setup.DATA_PROCESSED, "y_test.parquet")
    ).values.ravel()

    dataset = FraudDataset(X_test, y_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    batch_x, batch_y = next(iter(loader))
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch X shape: {batch_x.shape}")  # (16, 1, 46)
    print(f"  Batch y shape: {batch_y.shape}")    # (16,)
    print(f"  Label values in batch: {batch_y.tolist()}")

    model.eval()
    with torch.no_grad():
        out = model(batch_x.to(setup.DEVICE))
    print(f"  Model output shape: {out.shape}")
    print(f"  Predictions: {torch.argmax(out, dim=1).cpu().tolist()}")

    print("\n[PASS] DataLoader verification passed!")


if __name__ == "__main__":
    main()
