"""The reader: mosaic image -> read grid + probability vector per non-blank cell.

1. Blank cells come from the brightness rule (``sudoku.data.mosaic.detect_blanks``), not the NN.
2. Every other tile goes through the CNN; logits are divided by the fitted temperature T (if
   ``config.USE_TEMPERATURE``) and turned into probabilities over digits 1..9.
3. The read digit of a cell is the most probable one.

Also holds save/load helpers for the trained CNN (weights + small JSON with T and training info).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sudoku import config
from sudoku.data.mnist import class_to_digit
from sudoku.data.mosaic import detect_blanks, split_tiles
from sudoku.nn.calibration import softmax
from sudoku.nn.model import DigitCNN, predict_logits

N = config.GRID


@dataclass
class ReadGrid:
    """What the reader saw in one mosaic.

    grid:  (9, 9) int, read digit 1..9, 0 for blank cells
    probs: (9, 9, 9) float, probs[r, c, d - 1] = P(cell (r, c) shows digit d); all 0 for blanks
    blank: (9, 9) bool, True where the brightness rule found an empty cell
    """

    grid: np.ndarray
    probs: np.ndarray
    blank: np.ndarray

    def confidence(self) -> np.ndarray:
        """(9, 9) probability of the read digit; 0 for blank cells."""
        return self.probs.max(axis=2)

    def top_k(self, r: int, c: int, k: int = config.HARD_TOPK) -> list:
        """The k most probable (digit, probability) pairs of cell (r, c), best first."""
        order = np.argsort(-self.probs[r, c])[:k]
        return [(int(class_to_digit(i)), float(self.probs[r, c, i])) for i in order]


class DigitReader:
    """Wraps a trained CNN and its temperature."""

    def __init__(self, model: DigitCNN, temperature: float = 1.0):
        self.model = model.eval()
        self.temperature = float(temperature)

    @classmethod
    def load(cls, model_dir: Path = config.MODELS, use_temperature: bool = config.USE_TEMPERATURE):
        """Load ``cnn.pt`` + ``cnn_meta.json`` written by train.py."""
        model, meta = load_cnn(model_dir)
        return cls(model, meta["temperature"] if use_temperature else 1.0)

    def predict_proba(self, images) -> np.ndarray:
        """(N, 28, 28) images in 0..1 -> (N, 9) probabilities over digits 1..9 (rows sum to 1)."""
        return softmax(predict_logits(self.model, images), self.temperature)

    def read_mosaic(self, mosaic: np.ndarray) -> ReadGrid:
        """Read a 252x252 mosaic: blank rule first, then the CNN on every non-blank tile."""
        blank = detect_blanks(mosaic)
        tiles = split_tiles(mosaic)[~blank]  # (n_filled, 28, 28), row-major order
        probs = np.zeros((N, N, config.N_CLASSES))
        grid = np.zeros((N, N), dtype=np.int64)
        if len(tiles):
            p = self.predict_proba(tiles)
            probs[~blank] = p
            grid[~blank] = class_to_digit(p.argmax(axis=1))
        return ReadGrid(grid=grid, probs=probs, blank=blank)


def save_cnn(model: DigitCNN, meta: dict, model_dir: Path) -> None:
    """Write weights to ``cnn.pt`` and ``meta`` (temperature, seed, history...) to JSON."""
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / config.CNN_FILE)
    (model_dir / config.CNN_META_FILE).write_text(json.dumps(meta, indent=2))


def load_cnn(model_dir: Path = config.MODELS):
    """Return (model, meta) saved by ``save_cnn``."""
    path = model_dir / config.CNN_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python train.py` first")
    model = DigitCNN()
    model.load_state_dict(torch.load(path, weights_only=True))
    meta = json.loads((model_dir / config.CNN_META_FILE).read_text())
    return model.eval(), meta
