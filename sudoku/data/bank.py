"""Save and load the puzzle bank written by make_data.py.

File layout (``config.PUZZLES_FILE``, one .npz):
    g{G}_puzzles    (n, 9, 9) int64  givens, 0 = empty, for G in config.GIVEN_TARGETS
    g{G}_solutions  (n, 9, 9) int64  the unique solution of each puzzle
    g{G}_attempts   (n,)      int64  full grids tried by the generator
    seed            ()        int64  generator seed; puzzle i of G is made from rng (seed, G, i)
"""

from pathlib import Path

import numpy as np

from sudoku import config
from sudoku.data.generator import generate_bank


def build_bank(n_puzzles: int, seed: int = config.SEED, givens=config.GIVEN_TARGETS) -> dict:
    """Generate ``n_puzzles`` puzzles for each given count. Returns ``{G: [Puzzle, ...]}``."""
    return {g: generate_bank(g, n_puzzles, seed) for g in givens}


def save_bank(bank: dict, path: Path, seed: int) -> None:
    """Write a bank from ``build_bank`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"seed": np.int64(seed)}
    for g, puzzles in bank.items():
        arrays[f"g{g}_puzzles"] = np.stack([p.puzzle for p in puzzles])
        arrays[f"g{g}_solutions"] = np.stack([p.solution for p in puzzles])
        arrays[f"g{g}_attempts"] = np.array([p.attempts for p in puzzles], dtype=np.int64)
    np.savez_compressed(path, **arrays)


def load_bank(path: Path = config.PUZZLES_FILE) -> dict:
    """Load a bank: ``{G: (puzzles, solutions)}`` plus ``"seed"``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python make_data.py` first")
    with np.load(path) as data:
        bank = {"seed": int(data["seed"])}
        for key in data.files:
            if key.endswith("_puzzles"):
                g = int(key[1 : -len("_puzzles")])
                bank[g] = (data[key], data[f"g{g}_solutions"])
    return bank
