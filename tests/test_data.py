"""Tests for the data layer: split, generator, mosaic, blank rule, hard-digit injection.

All tests except the last use synthetic arrays, so they run without downloading MNIST.
"""

import numpy as np
import pytest

from sudoku import config
from sudoku.data import bank, generator, injection, mnist, mosaic

T, N = config.TILE, config.GRID


# ---------------------------------------------------------------- labels and split
def test_digit_class_mapping_round_trip():
    digits = np.arange(1, 10)
    assert np.array_equal(mnist.digit_to_class(digits), np.arange(9))
    assert np.array_equal(mnist.class_to_digit(mnist.digit_to_class(digits)), digits)


@pytest.mark.normal
def test_split_is_seeded_disjoint_and_drops_zeros():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=config.MNIST_TRAIN_SIZE)
    images = np.zeros((config.MNIST_TRAIN_SIZE, 1, 1), np.uint8)
    test_labels = np.array([0, 3, 0, 9])
    splits = mnist.make_splits(images, labels, images[:4], test_labels, seed=1)
    again = mnist.make_splits(images, labels, images[:4], test_labels, seed=1)
    other = mnist.make_splits(images, labels, images[:4], test_labels, seed=2)

    train_idx, val_idx = splits["train"][2], splits["val"][2]
    assert np.array_equal(train_idx, again["train"][2])          # same seed -> same split
    assert not np.array_equal(train_idx, other["train"][2])      # other seed -> other split
    assert not set(train_idx) & set(val_idx)                     # disjoint
    assert len(train_idx) + len(val_idx) == (labels != 0).sum()  # only zeros were removed
    assert splits["train"][1].min() >= 1 and splits["val"][1].min() >= 1
    assert np.array_equal(splits["test"][1], [3, 9])             # test keeps order, no shuffling
    assert np.array_equal(splits["test"][2], [1, 3])


@pytest.mark.normal
# ---------------------------------------------------------------- generator
@pytest.mark.parametrize("n_givens", config.GIVEN_TARGETS)
def test_generated_puzzle_has_unique_solution_and_target_givens(n_givens):
    p = generator.generate_puzzle(n_givens, config.make_rng(3))
    assert p.n_givens == n_givens
    assert generator.is_valid_solution(p.solution)
    given = p.puzzle != 0
    assert np.array_equal(p.puzzle[given], p.solution[given])  # givens agree with the solution
    assert generator.count_solutions(p.puzzle, limit=5) == 1


@pytest.mark.normal
def test_generator_is_reproducible_per_puzzle_seed():
    a = generator.generate_bank(30, 3, seed=11)
    b = generator.generate_bank(30, 3, seed=11)
    c = generator.generate_bank(30, 3, seed=12)
    assert all(np.array_equal(x.puzzle, y.puzzle) for x, y in zip(a, b))
    assert not np.array_equal(a[0].puzzle, c[0].puzzle)
    assert not np.array_equal(a[0].puzzle, a[1].puzzle)  # puzzles in one bank differ


@pytest.mark.edge
def test_already_solved_grid_has_exactly_one_solution():
    solution = generator.generate_full_grid(config.make_rng(0))
    assert generator.count_solutions(solution, limit=5) == 1


@pytest.mark.impossible
def test_count_solutions_detects_clash_and_multiple_solutions():
    empty = np.zeros((N, N), int)
    assert generator.count_solutions(empty, limit=2) == 2     # many solutions, stops at limit
    clash = empty.copy()
    clash[0, 0] = clash[0, 5] = 4                             # two equal givens in a row
    assert generator.count_solutions(clash) == 0


def _unavoidable_rectangle(solution):
    """Find rows r1, r2 in one band and columns c1, c2 in different stacks where the four cells hold
    a, b / b, a. Swapping a and b there gives a second valid grid."""
    for r1 in range(N):
        for r2 in range(r1 + 1, (r1 // 3 + 1) * 3):
            for c1 in range(N):
                for c2 in range((c1 // 3 + 1) * 3, N):
                    if solution[r1, c1] == solution[r2, c2] and solution[r1, c2] == solution[r2, c1]:
                        return r1, r2, c1, c2
    return None


def test_count_solutions_counts_exactly_two():
    for seed in range(50):
        solution = generator.generate_full_grid(config.make_rng(seed))
        found = _unavoidable_rectangle(solution)
        if found:
            break
    assert found, "no grid with an unavoidable rectangle in 50 seeds"
    r1, r2, c1, c2 = found
    puzzle = solution.copy()
    puzzle[[r1, r1, r2, r2], [c1, c2, c1, c2]] = 0
    assert generator.count_solutions(puzzle, limit=10) == 2


@pytest.mark.impossible
def test_impossible_target_raises_clear_error():
    with pytest.raises(ValueError, match="n_givens"):
        generator.generate_puzzle(10, config.make_rng(0))


def test_bank_round_trip(tmp_path):
    built = bank.build_bank(2, seed=5, givens=(30, 35))
    path = tmp_path / "bank.npz"
    bank.save_bank(built, path, seed=5)
    loaded = bank.load_bank(path)
    assert loaded["seed"] == 5
    for g in (30, 35):
        puzzles, solutions = loaded[g]
        assert puzzles.shape == (2, N, N)
        assert np.array_equal(puzzles[1], built[g][1].puzzle)
        assert np.array_equal(solutions[0], built[g][0].solution)


# ---------------------------------------------------------------- mosaic and blank rule
def _fake_images(n_per_digit=4):
    """Synthetic 'digit' images: image of digit d has constant brightness d/10 (never blank)."""
    digits = np.repeat(np.arange(1, 10), n_per_digit)
    images = np.repeat((digits / 10.0).astype(np.float32)[:, None, None], T, 1).repeat(T, 2)
    return images, digits


@pytest.mark.normal
def test_mosaic_layout_blanks_and_round_trip():
    images, digits = _fake_images()
    p = generator.generate_puzzle(30, config.make_rng(4))
    image, tile_index = mosaic.make_mosaic(p.puzzle, images, mosaic.index_pools_by_digit(digits),
                                           config.make_rng(4))
    assert image.shape == (N * T, N * T) and image.dtype == np.float32
    tiles = mosaic.split_tiles(image)
    assert tiles.shape == (N, N, T, T)
    blank = p.puzzle == 0
    assert np.all(tiles[blank] == 0)                                   # blank tiles all-zero
    assert np.array_equal(tile_index == mosaic.BLANK, blank)
    assert np.array_equal(digits[tile_index[~blank]], p.puzzle[~blank])  # right digit per cell
    assert np.array_equal(mosaic.detect_blanks(image), blank)
    r, c = np.argwhere(~blank)[0]
    assert np.array_equal(tiles[r, c], images[tile_index[r, c]])


@pytest.mark.edge
def test_detect_blanks_threshold_edge():
    image = np.zeros((N * T, N * T), np.float32)
    image[:T, :T] = config.BLANK_THRESHOLD * 1.1    # just above threshold -> not blank
    image[:T, T : 2 * T] = config.BLANK_THRESHOLD / 2  # faint smudge -> blank
    mask = mosaic.detect_blanks(image)
    assert not mask[0, 0] and mask[0, 1] and mask.sum() == N * N - 1


# ---------------------------------------------------------------- hard-digit injection
def _probs_for(true_digits, top1, second):
    """Probability rows where class ``top1`` gets 0.55 and ``second`` gets 0.40."""
    probs = np.full((len(true_digits), 9), 0.05 / 7)
    rows = np.arange(len(true_digits))
    probs[rows, top1] = 0.55
    probs[rows, second] = 0.40
    return probs


@pytest.mark.normal
def test_pool_rules_follow_documented_definition():
    digits = np.array([1, 1, 1, 2, 2])
    probs = np.zeros((5, 9))
    probs[0, 0] = 0.99; probs[0, 1:] = 0.01 / 8                 # confident & right -> clean only
    probs[1] = _probs_for([1], [0], [4])[0]                     # right but unsure -> hard, clean
    probs[2] = _probs_for([1], [6], [0])[0]                     # wrong, true 2nd -> hard, misread
    probs[3] = _probs_for([2], [5], [6])[0]                     # wrong, true not top-3 -> neither
    probs[4] = 0.01 / 6                                         # wrong & confident, but the
    probs[4, 2], probs[4, 1], probs[4, 0] = 0.9, 0.06, 0.03     # true class 1 is 2nd -> both
    hard = injection.hard_mask(probs, digits)
    mis = injection.misread_mask(probs, digits)
    clean = injection.clean_mask(probs, digits)
    assert hard.tolist() == [False, True, True, False, True]
    assert mis.tolist() == [False, False, True, False, True]
    assert clean.tolist() == [True, True, False, False, False]
    assert not (mis & ~hard).any()  # misread pool is a subset of the hard pool


def test_build_pools_exact_mode_separates_clean_and_misread():
    digits = np.array([1, 1, 2])
    probs = np.vstack([_probs_for([1], [0], [1]), _probs_for([1], [1], [0]),
                       _probs_for([2], [1], [0])])
    pools = injection.build_pools(probs, digits, exact_misreads=True)
    assert pools.base[1].tolist() == [0] and pools.inject[1].tolist() == [1]
    assert pools.base[2].tolist() == [2] and len(pools.inject[2]) == 0
    loose = injection.build_pools(probs, digits, exact_misreads=False)
    assert loose.base[1].tolist() == [0, 1] and loose.inject[1].tolist() == [0, 1]


@pytest.mark.normal
@pytest.mark.parametrize("k", config.MISREAD_LEVELS)
def test_inject_replaces_exactly_k_givens_with_same_digit(k):
    images, digits = _fake_images()
    is_hard = np.zeros(len(digits), bool)
    is_hard[::4] = True                       # one "hard" image per digit
    base = mosaic.index_pools_by_digit(digits, ~is_hard)
    hard = mosaic.index_pools_by_digit(digits, is_hard)
    p = generator.generate_puzzle(30, config.make_rng(9))
    tile_index = mosaic.assign_tiles(p.puzzle, base, config.make_rng(9))
    new_index, cells = injection.inject_hard_digits(p.puzzle, tile_index, k, hard, config.make_rng(9))
    changed = np.argwhere(new_index != tile_index)
    assert len(cells) == k and sorted(map(tuple, changed.tolist())) == cells
    for r, c in cells:
        assert is_hard[new_index[r, c]]                               # injected image is hard
    given = p.puzzle != 0
    assert np.array_equal(digits[new_index[given]], p.puzzle[given])  # digits never change
    again, _ = injection.inject_hard_digits(p.puzzle, tile_index, k, hard, config.make_rng(9))
    assert np.array_equal(new_index, again)                           # seeded


@pytest.mark.impossible
def test_inject_more_than_eligible_raises():
    p = generator.generate_puzzle(30, config.make_rng(2))
    empty_pools = {d: np.array([], int) for d in config.DIGITS}
    with pytest.raises(ValueError, match="cannot inject"):
        injection.inject_hard_digits(p.puzzle, np.zeros((N, N), int), 1, empty_pools,
                                     config.make_rng(0))


# ---------------------------------------------------------------- real MNIST (after make_data.py)
@pytest.mark.skipif(not config.SPLITS_FILE.exists(), reason="run make_data.py first")
def test_saved_splits_match_documented_sizes():
    splits = mnist.load_splits()
    assert isinstance(mnist.split_seed(), int)
    assert len(splits["train"]) + len(splits["val"]) == config.MNIST_TRAIN_NONZERO
    assert len(splits["test"]) == config.MNIST_TEST_NONZERO
    for part in splits.values():
        assert part.images.dtype == np.float32 and 0 <= part.images.min() and part.images.max() <= 1
        assert part.digits.min() == 1 and part.digits.max() == 9
    test_means = splits["test"].images.mean(axis=(1, 2))
    assert test_means.min() > config.BLANK_THRESHOLD  # no real digit is mistaken for a blank
