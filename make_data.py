"""Reproducible data preparation for P7.

Steps (all seeded with --seed, default config.SEED):
1. Download MNIST with torchvision into data/raw/ (skipped if already there).
2. Seeded 50,000 / 10,000 train/validation split of the 60,000 MNIST train images, then drop digit 0
   in train, validation and the untouched MNIST test set. Saved to data/processed/mnist_digits.npz.
3. Check the blank-cell brightness rule against every MNIST test digit.
4. Generate the puzzle bank (25 / 30 / 35 givens, unique solutions). Saved to
   data/processed/puzzles.npz (or puzzles_quick.npz with --quick).
5. Save a preview mosaic to results/data/sample_mosaic.png.

Hard-digit pools need the trained reader, so they are built by train.py, not here.

Usage:  python make_data.py [--seed 7] [--quick]
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sudoku import config  # noqa: E402
from sudoku.data import bank, mnist, mosaic  # noqa: E402


def prepare_mnist(seed: int) -> dict:
    """Download, split, drop zeros, check sizes, save. Returns the uint8 splits."""
    train_x, train_y, test_x, test_y = mnist.download_mnist()
    splits = mnist.make_splits(train_x, train_y, test_x, test_y, seed)
    mnist.check_split_sizes(splits, train_y, test_y)
    mnist.save_splits(splits, seed)
    for name, (images, _, _) in splits.items():
        print(f"  {name:5s}: {len(images):6d} images (digits 1-9)")
    print(f"  saved {config.SPLITS_FILE.relative_to(config.ROOT)}")
    return splits


def check_blank_rule(test_images_uint8: np.ndarray) -> None:
    """Every real digit must be brighter than the blank threshold, and an empty tile darker."""
    means = (test_images_uint8.astype(np.float32) / config.PIXEL_MAX).mean(axis=(1, 2))
    n_dark = int((means < config.BLANK_THRESHOLD).sum())
    print(f"  darkest test digit mean pixel = {means.min():.4f}; "
          f"threshold = {config.BLANK_THRESHOLD}; digits below threshold: {n_dark}")
    if n_dark:
        raise SystemExit("blank rule would hide real digits - lower BLANK_THRESHOLD in config.py")


def make_puzzle_bank(seed: int, quick: bool) -> dict:
    """Generate and save the puzzle bank; returns ``{G: [Puzzle, ...]}``."""
    n = config.QUICK_BANK_SIZE if quick else config.PUZZLE_BANK_SIZE
    path = config.PUZZLES_QUICK_FILE if quick else config.PUZZLES_FILE
    start = time.perf_counter()
    puzzles = bank.build_bank(n, seed)
    bank.save_bank(puzzles, path, seed)
    for g, items in puzzles.items():
        attempts = [p.attempts for p in items]
        print(f"  {g} givens: {len(items)} puzzles, generator attempts mean {np.mean(attempts):.2f}"
              f" max {max(attempts)}")
    print(f"  {time.perf_counter() - start:.1f}s; saved {path.relative_to(config.ROOT)}")
    return puzzles


def save_preview(puzzle: np.ndarray, test_images: np.ndarray, test_digits: np.ndarray, seed: int):
    """Render one puzzle with random MNIST-test digits and save it as a PNG."""
    pools = mosaic.index_pools_by_digit(test_digits)
    image, _ = mosaic.make_mosaic(puzzle, test_images, pools, config.make_rng(seed))
    assert np.array_equal(mosaic.detect_blanks(image), puzzle == 0), "blank rule failed on preview"
    config.DATA_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_PREVIEW_DIR / "sample_mosaic.png"
    side = config.GRID * config.TILE
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    for k in range(0, config.GRID + 1, config.BOX):  # 3x3 box borders
        ax.axhline(k * config.TILE - 0.5, color="tab:red", lw=1)
        ax.axvline(k * config.TILE - 0.5, color="tab:red", lw=1)
    ax.set_xlim(-0.5, side - 0.5)
    ax.set_ylim(side - 0.5, -0.5)
    ax.set_axis_off()
    ax.set_title(f"Sample mosaic ({int((puzzle != 0).sum())} givens, seed {seed})")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(config.ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--quick", action="store_true", help="small puzzle bank for smoke tests")
    args = parser.parse_args()

    print("[1/4] MNIST download + split")
    splits = prepare_mnist(args.seed)
    test_images, test_digits, _ = splits["test"]
    print("[2/4] blank-cell rule check")
    check_blank_rule(test_images)
    print("[3/4] puzzle bank")
    puzzles = make_puzzle_bank(args.seed, args.quick)
    print("[4/4] preview mosaic")
    save_preview(puzzles[config.DEFAULT_GIVENS][0].puzzle,
                 test_images.astype(np.float32) / config.PIXEL_MAX, test_digits, args.seed)
    print("done")


if __name__ == "__main__":
    main()
