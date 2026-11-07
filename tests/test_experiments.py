"""Tests for the experiment helpers: statistics over seeds, recovery table, small end-to-end runs."""

import numpy as np
import pandas as pd
import pytest

from sudoku import config
from sudoku.data import bank as bank_io, injection, mnist
from sudoku.eval import experiments as ex
from sudoku.nn.reader import DigitReader
from tests.test_nn import needs_model


def test_mean_std_over_seeds_averages_per_seed_first():
    df = pd.DataFrame({"seed": [0, 0, 0, 1], "algo": ["a"] * 4, "x": [1.0, 2.0, 3.0, 10.0]})
    out = ex.mean_std_over_seeds(df, ["algo"], ["x"]).iloc[0]
    # seed means are 2 and 10 (not the pooled mean 4): mean 6, sample std of (2, 10) = 5.657
    assert out.n_seeds == 2
    assert out.x_mean == pytest.approx(6.0)
    assert out.x_std == pytest.approx(np.std([2, 10], ddof=1))


def test_recovery_table_breakdown_adds_up():
    rows = []
    for seed, (status, exact) in enumerate([("solved", True), ("corrected", True),
                                            ("solved", False), ("unreadable", False)]):
        rows.append({"seed": seed, "level": 1, "algorithm": config.DEFAULT_ALGORITHM,
                     "status": status, "exact": exact, "misreads_in_candidates": True})
    rows.append({"seed": 0, "level": 1, "algorithm": "ucs", "status": "solved", "exact": True,
                 "misreads_in_candidates": True})  # other algorithms are ignored
    t = ex.recovery_table(pd.DataFrame(rows)).iloc[0]
    assert t.puzzles == 4 and t.exact_recovered == 2 and t.exact_rate == 0.5
    assert (t.exact_as_read, t.exact_after_correction, t.wrong_solution, t.unreadable) == (1, 1, 1, 1)
    assert t.exact_as_read + t.exact_after_correction + t.wrong_solution + t.unreadable == t.puzzles


def test_csp_experiment_rows_and_correctness():
    built = bank_io.build_bank(2, seed=3)
    bank = {g: (np.stack([p.puzzle for p in ps]), np.stack([p.solution for p in ps]))
            for g, ps in built.items()}
    df = ex.run_csp_experiment(bank, n_seeds=2, per_seed=1, log=lambda *_: None)
    assert len(df) == len(config.GIVEN_TARGETS) * 2 * len(ex.CSP_MODES)
    solved = df[df.status == "solved"]
    assert solved.correct.all()
    bt = df[df["mode"] == "BT"]
    assert (bt.fc_pruned == 0).all() and (bt.ac3_revisions == 0).all()


@needs_model
def test_reading_experiments_small_run():
    reader = DigitReader.load()
    puzzles, solutions = bank_io.load_bank()[config.DEFAULT_GIVENS]
    exp1, exp3 = ex.run_reading_experiments(
        reader, DigitReader(reader.model, 1.0), mnist.load_splits()["test"].images,
        injection.load_pools(), puzzles, solutions, n_seeds=1, per_seed=1, log=lambda *_: None)
    levels = len(config.MISREAD_LEVELS)
    assert len(exp1) == levels * 3 and len(exp3) == levels * len(ex.EXP3_ARMS)
    assert (exp1.actual_misreads == exp1.level).all()              # exactly k misreads
    level0 = exp1[exp1.level == 0]
    assert level0.exact.all() and (level0.nodes_expanded == 0).all()
    off = exp3[(exp3.arm == "no_correction") & (exp3.level > 0)]
    assert not off.exact.any()                                      # NN off cannot fix misreads
