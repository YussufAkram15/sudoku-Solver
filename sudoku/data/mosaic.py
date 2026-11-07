"""Turn a puzzle grid into a 252x252 image mosaic of MNIST test digits, and back into tiles.

A mosaic is described by a *tile index* grid: (9, 9) int array holding, for each given cell, the
row of the MNIST-test ``DigitSplit`` whose image is drawn there; -1 marks an empty cell (an all-zero
tile). Keeping the index (not just pixels) lets us trace every tile back to its real MNIST image.

Blank detection is a brightness rule (mean pixel < ``config.BLANK_THRESHOLD``), not the network.
"""

import numpy as np

from sudoku import config

N = config.GRID
T = config.TILE
BLANK = -1  # tile index of an empty cell


def index_pools_by_digit(digits: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Group image rows by digit: ``{digit: array of row indices}``, optionally only where ``mask``."""
    digits = np.asarray(digits)
    keep = np.ones(len(digits), bool) if mask is None else np.asarray(mask, bool)
    return {d: np.flatnonzero(keep & (digits == d)) for d in config.DIGITS}


def assign_tiles(puzzle: np.ndarray, pools: dict, rng: np.random.Generator) -> np.ndarray:
    """Pick an image for every given cell: a random row from ``pools[digit]``.

    Returns the (9, 9) tile index grid (``BLANK`` for empty cells).
    """
    tile_index = np.full((N, N), BLANK, dtype=np.int64)
    for r, c in zip(*np.nonzero(puzzle)):
        pool = pools[int(puzzle[r, c])]
        if len(pool) == 0:
            raise ValueError(f"no image available for digit {puzzle[r, c]}")
        tile_index[r, c] = rng.choice(pool)
    return tile_index


def render_mosaic(tile_index: np.ndarray, images: np.ndarray) -> np.ndarray:
    """Paste the chosen 28x28 images into a (252, 252) float32 mosaic; blank cells stay 0."""
    mosaic = np.zeros((N * T, N * T), dtype=np.float32)
    for r, c in zip(*np.nonzero(tile_index != BLANK)):
        mosaic[r * T : (r + 1) * T, c * T : (c + 1) * T] = images[tile_index[r, c]]
    return mosaic


def split_tiles(mosaic: np.ndarray) -> np.ndarray:
    """Cut a (252, 252) mosaic into a (9, 9, 28, 28) array of tiles (inverse of ``render_mosaic``)."""
    if mosaic.shape != (N * T, N * T):
        raise ValueError(f"mosaic must be {N * T}x{N * T}, got {mosaic.shape}")
    return mosaic.reshape(N, T, N, T).transpose(0, 2, 1, 3)


def detect_blanks(mosaic: np.ndarray, threshold: float = config.BLANK_THRESHOLD) -> np.ndarray:
    """Brightness rule: (9, 9) bool mask, True where the tile's mean pixel is below ``threshold``."""
    return split_tiles(mosaic).mean(axis=(2, 3)) < threshold


def make_mosaic(puzzle: np.ndarray, images: np.ndarray, pools: dict, rng: np.random.Generator):
    """Convenience: choose tiles for ``puzzle`` and render them. Returns (mosaic, tile_index)."""
    tile_index = assign_tiles(puzzle, pools, rng)
    return render_mosaic(tile_index, images), tile_index
