"""Hard-digit injection: put images the network finds hard into a puzzle on purpose.

A well-trained CNN reads almost every MNIST digit correctly, so random mosaics would leave nothing to
correct. The rule (also in DATA_DICTIONARY.md):

* **Hard pool** (brief's rule): MNIST-test images where the trained reader is wrong, OR its top-1
  probability is below ``config.HARD_PROB_THRESHOLD`` - AND the true digit is within its top
  ``config.HARD_TOPK`` digits (so a correction to the true digit is reachable by the search).
* **Misread pool**: the part of the hard pool the reader gets wrong (argmax != true digit).
* **Clean pool**: test images the reader gets right.
* A puzzle at misread level k: pick k given cells uniformly at random and draw their images from
  the misread pool of the cell's true digit (``config.EXACT_MISREADS = True``, so exactly k cells
  are read wrong; the other givens use the clean pool), or from the whole hard pool
  (``EXACT_MISREADS = False``; the other givens then use any test image).

The pool functions only need the reader's probability matrix, so they do not depend on the NN code.
The pools themselves are computed after training (``train.py``).
"""

from dataclasses import dataclass

import numpy as np

from sudoku import config
from sudoku.data.mosaic import BLANK, index_pools_by_digit
from sudoku.data.mnist import digit_to_class


def _true_in_topk(probs: np.ndarray, true_classes: np.ndarray, topk: int) -> np.ndarray:
    """Bool per row: is the true class among the ``topk`` most probable classes?"""
    top = np.argsort(-probs, axis=1)[:, :topk]
    return (top == true_classes[:, None]).any(axis=1)


def hard_mask(probs, digits, threshold=config.HARD_PROB_THRESHOLD, topk=config.HARD_TOPK):
    """Hard pool rule: (wrong read OR top-1 prob < threshold) AND true digit in top-k."""
    probs, true = np.asarray(probs), digit_to_class(digits)
    wrong = probs.argmax(axis=1) != true
    unsure = probs.max(axis=1) < threshold
    return (wrong | unsure) & _true_in_topk(probs, true, topk)


def misread_mask(probs, digits, topk=config.HARD_TOPK):
    """Misread pool rule: wrong read AND true digit in top-k (a subset of the hard pool)."""
    probs, true = np.asarray(probs), digit_to_class(digits)
    return (probs.argmax(axis=1) != true) & _true_in_topk(probs, true, topk)


def clean_mask(probs, digits):
    """Images the reader classifies correctly."""
    return np.asarray(probs).argmax(axis=1) == digit_to_class(digits)


@dataclass
class DigitPools:
    """Row indices into the MNIST-test split, grouped by digit, for building mosaics."""

    base: dict    # images for normal given cells
    inject: dict  # images for injected hard cells

    def sizes(self) -> dict:
        """Number of images per digit in each pool (for printing / the data dictionary)."""
        return {
            "base": {d: len(v) for d, v in self.base.items()},
            "inject": {d: len(v) for d, v in self.inject.items()},
        }


def build_pools(probs, digits, exact_misreads: bool = config.EXACT_MISREADS) -> DigitPools:
    """Build base and injection pools from the reader's probabilities on the MNIST test split."""
    if exact_misreads:
        return DigitPools(
            base=index_pools_by_digit(digits, clean_mask(probs, digits)),
            inject=index_pools_by_digit(digits, misread_mask(probs, digits)),
        )
    return DigitPools(
        base=index_pools_by_digit(digits),
        inject=index_pools_by_digit(digits, hard_mask(probs, digits)),
    )


def inject_hard_digits(
    puzzle: np.ndarray,
    tile_index: np.ndarray,
    k: int,
    inject_pools: dict,
    rng: np.random.Generator,
):
    """Replace the images of k random given cells with hard images of the SAME digit.

    Only cells whose digit has a non-empty injection pool are eligible. Returns
    ``(new_tile_index, injected_cells)`` where ``injected_cells`` is a sorted list of (row, col).
    """
    eligible = [
        (int(r), int(c))
        for r, c in zip(*np.nonzero(puzzle))
        if len(inject_pools[int(puzzle[r, c])]) > 0
    ]
    if k > len(eligible):
        raise ValueError(f"cannot inject {k} hard digits: only {len(eligible)} eligible givens")
    new_index = tile_index.copy()
    chosen = rng.choice(len(eligible), size=k, replace=False)
    cells = sorted(eligible[i] for i in chosen)
    for r, c in cells:
        new_index[r, c] = rng.choice(inject_pools[int(puzzle[r, c])])
    assert (new_index == BLANK).sum() == (tile_index == BLANK).sum()
    return new_index, cells
