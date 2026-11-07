"""Plots for the NN deliverable: confusion matrices and reliability diagrams."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from sudoku import config  # noqa: E402
from sudoku.nn.calibration import expected_calibration_error, reliability_bins  # noqa: E402


def digit_confusion(probs, labels) -> np.ndarray:
    """9x9 confusion matrix, rows = true digit 1..9, columns = predicted digit 1..9."""
    return confusion_matrix(labels, np.asarray(probs).argmax(axis=1), labels=range(config.N_CLASSES))


def plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    """Heat map of a confusion matrix with counts written in each cell."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(np.log1p(matrix), cmap="Blues")  # log scale so small off-diagonal counts show
    ticks = np.arange(config.N_CLASSES)
    ax.set_xticks(ticks, labels=config.DIGITS)
    ax.set_yticks(ticks, labels=config.DIGITS)
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("true digit")
    big = matrix.max() / 2
    for (r, c), v in np.ndenumerate(matrix):
        if v:
            ax.text(c, r, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > big else "black")
    ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(named_probs: dict, labels, path: Path, n_bins: int = config.ECE_BINS) -> None:
    """One reliability diagram per model: bin accuracy vs bin confidence, with ECE in the title."""
    fig, axes = plt.subplots(1, len(named_probs), figsize=(4 * len(named_probs), 4), squeeze=False)
    for ax, (name, probs) in zip(axes[0], named_probs.items()):
        bins = reliability_bins(probs, labels, n_bins)
        centers = (bins["edges"][:-1] + bins["edges"][1:]) / 2
        used = bins["count"] > 0
        ax.bar(centers[used], bins["accuracy"][used], width=1 / n_bins, edgecolor="black",
               color="tab:blue", label="accuracy")
        ax.plot(centers[used], bins["confidence"][used], "o", color="tab:red", ms=4,
                label="mean confidence in bin")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ece = expected_calibration_error(probs, labels, n_bins)
        ax.set_title(f"{name}\nECE = {ece:.4f}", fontsize=10)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
