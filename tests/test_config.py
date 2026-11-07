"""Sanity checks on the central configuration."""

import random

import numpy as np

from sudoku import config


def test_paths_are_inside_repo():
    for path in (config.DATA_RAW, config.DATA_PROCESSED, config.MODELS, config.RESULTS):
        assert config.ROOT in path.parents


def test_split_and_budget_values_are_consistent():
    assert config.N_TRAIN + config.N_VAL <= 60_000
    assert config.N_SEEDS * config.PUZZLES_PER_SEED == 100
    assert config.GOAL_MODE in ("unique", "solvable")
    assert config.ORDERING in ("static", "mrv")
    assert max(config.MISREAD_LEVELS) > config.MAX_CORRECTIONS  # level 4 shows the budget limit


def test_set_seed_is_deterministic():
    config.set_seed(123)
    first = (random.random(), np.random.rand())
    config.set_seed(123)
    assert (random.random(), np.random.rand()) == first


def test_make_rng_is_deterministic_and_independent():
    a = config.make_rng(5).integers(0, 1000, size=10)
    b = config.make_rng(5).integers(0, 1000, size=10)
    c = config.make_rng(6).integers(0, 1000, size=10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
