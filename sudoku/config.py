"""Central configuration: every seed, threshold, path and limit lives here.

No other module should hard-code a tunable number. Import as::

    from sudoku import config
    config.SEED
"""

from pathlib import Path
import random

import numpy as np

# ---------------------------------------------------------------- seeds
SEED = 7  # team generator seed; every random step derives from this by default

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"              # MNIST download (git-ignored)
DATA_PROCESSED = ROOT / "data" / "processed"  # splits, puzzle banks, digit pools (git-ignored)
MODELS = ROOT / "models"                      # trained CNN / baseline weights
RESULTS = ROOT / "results"                    # CSV tables + PNG plots (committed)

# ---------------------------------------------------------------- data
# The brief's 50,000 / 10,000 split refers to the full 60,000-image MNIST train set. Only 54,077 of
# those are digits 1-9 (5,923 are zeros), so 50k + 10k cannot both be filled AFTER dropping 0.
# Rule used: seeded 50k/10k split of all 60k first, then drop digit 0 inside each part
# (about 45k train / 9k validation remain). MNIST test stays untouched.
MNIST_TRAIN_SIZE = 60_000
MNIST_TEST_SIZE = 10_000
MNIST_TRAIN_NONZERO = 54_077   # digits 1-9 in MNIST train; asserted when the data is loaded
MNIST_TEST_NONZERO = 9_020     # digits 1-9 in MNIST test (980 zeros dropped); asserted on load
PIXEL_MAX = 255.0              # raw MNIST pixels are uint8 0..255; scaled to 0..1 on load
SPLITS_FILE = DATA_PROCESSED / "mnist_digits.npz"      # written by make_data.py
PUZZLES_FILE = DATA_PROCESSED / "puzzles.npz"          # full puzzle bank
PUZZLES_QUICK_FILE = DATA_PROCESSED / "puzzles_quick.npz"  # small bank from make_data.py --quick
DATA_PREVIEW_DIR = RESULTS / "data"                    # sample mosaic PNG from make_data.py
SPLIT_TRAIN = 50_000           # train part of the full MNIST train set, before dropping 0
SPLIT_VAL = 10_000             # validation part, before dropping 0
DIGITS = tuple(range(1, 10))   # digit 0 is dropped; class index = digit - 1
N_CLASSES = 9
TILE = 28               # MNIST image side in pixels
GRID = 9                # sudoku side length
BOX = 3                 # sub-box side length

GIVEN_TARGETS = (25, 30, 35)          # number of givens in generated puzzles
DEFAULT_GIVENS = 30                   # used by the demo and the search/correction experiments
GENERATOR_MAX_ATTEMPTS = 20           # new full grids to try if a given-count target is not reached
MIN_GIVENS = 17                       # fewest givens any uniquely solvable Sudoku can have
BLANK_THRESHOLD = 0.01                # tile mean pixel (0..1) below this = empty cell

# ---------------------------------------------------------------- hard-digit injection
MISREAD_LEVELS = (0, 1, 2, 3, 4)
HARD_PROB_THRESHOLD = 0.6   # pool rule from the brief: wrong read, or top-1 prob below this
HARD_TOPK = 3               # ...and the true digit must be within the model's top-k
# True: "k misreads" means exactly k cells the model reads WRONG. The k injected tiles come from the
# wrong-read part of the hard pool and every other given uses a test image the model reads correctly.
# False: inject from the whole hard pool (brief's literal rule) and use any test image for the other
# givens; injected tiles may still be read correctly and other tiles may be misread, so the actual
# number of misreads can differ from k (below or above).
EXACT_MISREADS = True

# ---------------------------------------------------------------- neural network
CNN_EPOCHS = 3
CNN_BATCH_SIZE = 128
CNN_LR = 1e-3
CNN_CONV_CHANNELS = (16, 32)
CNN_HIDDEN = 128
CNN_KERNEL = 3                 # conv kernel size (padding keeps 28x28 before each 2x2 max-pool)
CNN_EVAL_BATCH = 1024          # batch size when only predicting (no gradients)
LOGREG_MAX_ITER = 1000         # lbfgs converges in ~205 iterations on the 20k subset
LOGREG_TRAIN_SUBSET = 20_000   # logistic regression on raw pixels is slow; subset of the train split
ECE_BINS = 15
USE_TEMPERATURE = True         # reader applies temperature scaling fitted on validation
TEMPERATURE_BOUNDS = (0.05, 20.0)  # search range for the fitted temperature T
PROB_EPS = 1e-12               # floor before taking log of a probability (avoids log 0)
# train.py --quick: tiny run for smoke tests, saved under models/quick and results/nn/quick
QUICK_EPOCHS = 1
QUICK_TRAIN_SUBSET = 5_000
QUICK_LOGREG_SUBSET = 2_000
CNN_FILE = "cnn.pt"            # file names inside the model directory
CNN_META_FILE = "cnn_meta.json"
LOGREG_FILE = "logreg.joblib"
POOLS_FILE = DATA_PROCESSED / "digit_pools.npz"        # reader probs on MNIST test + pool masks
POOLS_QUICK_FILE = DATA_PROCESSED / "digit_pools_quick.npz"
NN_RESULTS = RESULTS / "nn"                            # metrics CSV, confusion + reliability PNGs

# ---------------------------------------------------------------- CSP
ORDERING = "static"     # "static" (row-major) or "mrv"; identical in every BT/FC/AC-3 mode
USE_MAC = False         # extension: maintain arc consistency during search
CSP_NODE_LIMIT = 200_000    # max assignments before status "timeout"
CSP_TIME_LIMIT = 10.0       # seconds before status "timeout"

# ---------------------------------------------------------------- correction search
CANDIDATE_CELLS = 15     # least-confident given cells that may be corrected
ALTS_PER_CELL = 3        # top-k non-current digits tried per candidate cell
MAX_CORRECTIONS = 3      # max cells changed in one correction set
# Full correction space with 15 cells x 3 alternatives and <= 3 corrections is
# 1 + 45 + 945 + 12,285 = 13,276 states. The node cap sits above that, so "unreadable" normally
# means the 3-correction budget is exhausted; hitting a cap is reported as its own stop reason.
SEARCH_NODE_LIMIT = 15_000   # max nodes expanded before stopping with reason "node_limit"
SEARCH_TIME_LIMIT = 60.0     # seconds before stopping with reason "time_limit" (revisit in Phase 8)
GOAL_MODE = "unique"     # "unique" = exactly one solution, "solvable" = at least one
DEFAULT_ALGORITHM = "astar"   # "ucs", "greedy" or "astar"
RANDOM_TRIALS = 200      # random-correction baseline: correction sets tried
N_UNREADABLE_REPORT = 5  # least-confident cells listed when no correction works

# ---------------------------------------------------------------- experiments
N_SEEDS = 5
PUZZLES_PER_SEED = 20    # 5 seeds x 20 = 100 puzzles per misread level
QUICK_SEEDS = 2
QUICK_PUZZLES = 3
PUZZLE_BANK_SIZE = N_SEEDS * PUZZLES_PER_SEED      # puzzles per given count in puzzles.npz
QUICK_BANK_SIZE = QUICK_SEEDS * QUICK_PUZZLES      # puzzles per given count in puzzles_quick.npz


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch global generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # CSP/search code runs without torch
        return
    torch.manual_seed(seed)


def make_rng(seed: int = SEED) -> np.random.Generator:
    """Return a fresh, independent NumPy generator for one random task."""
    return np.random.default_rng(seed)
