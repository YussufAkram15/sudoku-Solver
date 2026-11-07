"""MNIST loading, the seeded train/validation split, and dropping digit 0.

Conventions used everywhere in the project:
- a *digit* is the real label 1..9 (what a Sudoku cell contains);
- a *class* is the network's output index 0..8, with ``class = digit - 1``.
Only the NN code works with classes; everything else uses digits.

On disk (``config.SPLITS_FILE``) images are stored as uint8 0..255 to keep the file small;
``load_splits`` scales them to float32 0..1.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sudoku import config


def digit_to_class(digits):
    """Map Sudoku digits 1..9 to network classes 0..8."""
    return np.asarray(digits) - 1


def class_to_digit(classes):
    """Map network classes 0..8 back to Sudoku digits 1..9."""
    return np.asarray(classes) + 1


@dataclass
class DigitSplit:
    """One part of the data (train, val or test), digits 1-9 only.

    images:     (N, 28, 28) float32 in 0..1
    digits:     (N,) int64 true digit 1..9
    source_idx: (N,) int64 index of each image in the original MNIST train or test set
    """

    images: np.ndarray
    digits: np.ndarray
    source_idx: np.ndarray

    @property
    def classes(self) -> np.ndarray:
        """Network targets 0..8 for these images."""
        return digit_to_class(self.digits)

    def __len__(self) -> int:
        return len(self.digits)


def download_mnist(root: Path = config.DATA_RAW):
    """Download MNIST with torchvision (if not cached) and return raw uint8 arrays.

    Returns (train_images, train_labels, test_images, test_labels); labels still include 0.
    """
    from torchvision.datasets import MNIST  # imported here so CSP/search code never needs torch

    train = MNIST(str(root), train=True, download=True)
    test = MNIST(str(root), train=False, download=True)
    return train.data.numpy(), train.targets.numpy(), test.data.numpy(), test.targets.numpy()


def split_indices(n_total: int, seed: int = config.SEED):
    """Seeded split of ``range(n_total)`` into train and validation index arrays.

    The first ``config.SPLIT_TRAIN`` positions of a random permutation are train, the next
    ``config.SPLIT_VAL`` are validation. Applied to all 60,000 MNIST train images (zeros included).
    """
    if n_total < config.SPLIT_TRAIN + config.SPLIT_VAL:
        raise ValueError(f"need at least {config.SPLIT_TRAIN + config.SPLIT_VAL} images, got {n_total}")
    perm = config.make_rng(seed).permutation(n_total)
    train_idx = perm[: config.SPLIT_TRAIN]
    val_idx = perm[config.SPLIT_TRAIN : config.SPLIT_TRAIN + config.SPLIT_VAL]
    return train_idx, val_idx


def drop_zeros(indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Keep only the indices whose label is a digit 1..9 (order preserved)."""
    indices = np.asarray(indices)
    return indices[labels[indices] != 0]


def make_splits(train_images, train_labels, test_images, test_labels, seed: int = config.SEED):
    """Build the three uint8 parts: split train 50k/10k, then drop 0 from train, val and test.

    Returns a dict ``{"train"|"val"|"test": (images_uint8, digits, source_idx)}``.
    """
    train_idx, val_idx = split_indices(len(train_labels), seed)
    parts = {
        "train": (train_images, train_labels, drop_zeros(train_idx, train_labels)),
        "val": (train_images, train_labels, drop_zeros(val_idx, train_labels)),
        "test": (test_images, test_labels, drop_zeros(np.arange(len(test_labels)), test_labels)),
    }
    return {
        name: (images[idx], labels[idx].astype(np.int64), idx.astype(np.int64))
        for name, (images, labels, idx) in parts.items()
    }


def save_splits(splits: dict, seed: int, path: Path = config.SPLITS_FILE) -> None:
    """Save the output of ``make_splits`` (made with ``seed``) as one compressed .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"seed": np.int64(seed)}
    for name, (images, digits, source_idx) in splits.items():
        arrays[f"{name}_images"] = images.astype(np.uint8)
        arrays[f"{name}_digits"] = digits
        arrays[f"{name}_source_idx"] = source_idx
    np.savez_compressed(path, **arrays)


def load_splits(path: Path = config.SPLITS_FILE) -> dict:
    """Load train/val/test ``DigitSplit`` objects with pixels scaled to 0..1.

    The split seed is stored in the file and can be read with ``split_seed(path)``.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python make_data.py` first")
    with np.load(path) as data:
        return {
            name: DigitSplit(
                images=data[f"{name}_images"].astype(np.float32) / config.PIXEL_MAX,
                digits=data[f"{name}_digits"],
                source_idx=data[f"{name}_source_idx"],
            )
            for name in ("train", "val", "test")
        }


def split_seed(path: Path = config.SPLITS_FILE) -> int:
    """Seed that produced the saved split."""
    with np.load(path) as data:
        return int(data["seed"])


def check_split_sizes(splits: dict, train_labels, test_labels) -> None:
    """Assert the split has the sizes documented in config / DATA_DICTIONARY.md."""
    n_train, n_val = len(splits["train"][1]), len(splits["val"][1])
    assert n_train + n_val == config.MNIST_TRAIN_NONZERO == int((train_labels != 0).sum())
    assert len(splits["test"][1]) == config.MNIST_TEST_NONZERO == int((test_labels != 0).sum())
    for name in ("train", "val", "test"):
        assert splits[name][1].min() >= 1, f"{name} still contains digit 0"
    assert not set(splits["train"][2]) & set(splits["val"][2]), "train/val overlap"
