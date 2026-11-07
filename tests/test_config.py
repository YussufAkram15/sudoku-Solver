"""Sanity checks on the central configuration."""

import random
from math import comb

import numpy as np
import torch

from sudoku import config


def test_split_fits_mnist_train():
    # 50k/10k is taken from the full 60k MNIST train set; digit 0 is dropped afterwards.
    assert config.SPLIT_TRAIN + config.SPLIT_VAL == config.MNIST_TRAIN_SIZE
    assert config.MNIST_TRAIN_NONZERO == config.MNIST_TRAIN_SIZE - 5_923  # 5,923 zeros in MNIST train


def test_search_node_cap_covers_full_correction_space():
    k, alts = config.CANDIDATE_CELLS, config.ALTS_PER_CELL
    n_states = sum(comb(k, r) * alts**r for r in range(config.MAX_CORRECTIONS + 1))
    assert n_states == 13_276
    assert config.SEARCH_NODE_LIMIT > n_states


def test_option_values_are_valid():
    assert config.N_SEEDS * config.PUZZLES_PER_SEED == 100
    assert config.GOAL_MODE in ("unique", "solvable")
    assert config.ORDERING in ("static", "mrv")
    assert config.DEFAULT_ALGORITHM in ("ucs", "greedy", "astar")
    assert max(config.MISREAD_LEVELS) > config.MAX_CORRECTIONS  # level 4 shows the budget limit


def test_set_seed_is_deterministic_for_python_numpy_and_torch():
    config.set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    config.set_seed(123)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == first


def test_make_rng_is_deterministic_and_independent():
    a = config.make_rng(5).integers(0, 1000, size=10)
    b = config.make_rng(5).integers(0, 1000, size=10)
    c = config.make_rng(6).integers(0, 1000, size=10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
