"""
00_setup.py — Project path constants, seed initialization, and configuration.
All scripts import from this module to ensure consistent paths and seeds.
"""

import os
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root — resolve to work/ directory (one level up from code/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_EXTENSION = os.path.join(PROJECT_ROOT, "data", "extension")
NOTES_DIR = os.path.join(PROJECT_ROOT, "notes")
OUTPUT_TABLES = os.path.join(PROJECT_ROOT, "output", "tables")
OUTPUT_FIGURES = os.path.join(PROJECT_ROOT, "output", "figures")
OUTPUT_PAPER = os.path.join(PROJECT_ROOT, "output", "paper")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ---------------------------------------------------------------------------
# Auto-create directories
# ---------------------------------------------------------------------------
for _dir in [
    DATA_RAW, DATA_PROCESSED, DATA_EXTENSION,
    NOTES_DIR, OUTPUT_TABLES, OUTPUT_FIGURES, OUTPUT_PAPER,
    LOGS_DIR, MODELS_DIR,
]:
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for numpy, torch, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Paper-specific constants (from Mohammed et al. 2026)
# ---------------------------------------------------------------------------
ORIGINAL_PAPER = "Mohammed et al. (2026) — A deep residual 1D-CNN with self-attention for fraud transaction detection in virtual economies"

# Model hyperparameters
CONV_FILTERS = 64          # Number of filters in conv blocks
KERNEL_SIZE = 3            # Conv1d kernel size
STRIDE_RESIDUAL = 2        # Stride in residual block (downsampling)

# Training hyperparameters
BATCH_SIZE = 16
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_DROP_FACTOR = 0.5
LR_DROP_PATIENCE = 5

# Data split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Data balancing
SMOTE_STRATEGY = "auto"    # Equalize minority class to majority

# Kaggle dataset file names
RAW_FILES = {
    "transactions": "transactions.csv",
    "customers": "customers.csv",
    "products": "products.csv",
    "behavior": "behavior.csv",
}

# Target variable
TARGET_COL = "fraud_label"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    set_seed()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Seed set to: {SEED}")
    print(f"Data raw directory: {DATA_RAW}")
    print("All output directories created.")
