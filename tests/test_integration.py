"""Tests for the integrated pipeline (steps 2-4 on hand-made readings, and end-to-end on mosaics)."""

import numpy as np
import pytest

from sudoku import config
from sudoku.data import generator, injection, mnist
from sudoku.integration import UNREADABLE_MESSAGE, format_result, pipeline, solve_reading
from sudoku.nn.reader import DigitReader, ReadGrid
from tests.test_csp import KNOWN, KNOWN_SOLUTION
from tests.test_nn import needs_model
from tests.test_search import ONE_MISREAD, X, Y, make_reading


def reading(grid, probs) -> ReadGrid:
    """Wrap a hand-made grid + probabilities as the reader's output."""
    return ReadGrid(grid=grid, probs=probs, blank=grid == 0)


# ---------------------------------------------------------------- hand-made readings
@pytest.mark.normal
def test_one_misread_is_corrected_and_logged():
    r = solve_reading(reading(*make_reading(**ONE_MISREAD)), algorithm="ucs")
    assert r.status == "corrected" and r.exact_recovery(KNOWN_SOLUTION)
    assert not r.ac3_check.consistent  # step 2 found the contradiction
    assert frozenset({X, Y}) in r.ac3_check.duplicates
    assert [(c.cell, c.read_digit, c.new_digit) for c in r.corrections] == [(X, 2, 3)]
    assert "correction (0, 2): 2 -> 3" in format_result(r)


@pytest.mark.edge
def test_zero_misreads_solved_without_search():
    r = solve_reading(reading(*make_reading()))
    assert r.status == "solved" and r.search is None and r.corrections == []
    assert r.ac3_check.consistent and r.exact_recovery(KNOWN_SOLUTION)


@pytest.mark.edge
def test_already_solved_grid_through_pipeline():
    probs = np.zeros((9, 9, 9))
    for (r, c), d in np.ndenumerate(KNOWN_SOLUTION):
        probs[r, c] = 0.01 / 8
        probs[r, c, d - 1] = 0.99
    result = solve_reading(reading(KNOWN_SOLUTION.copy(), probs))
    assert result.status == "solved" and result.exact_recovery(KNOWN_SOLUTION)


@pytest.mark.normal
def test_nn_off_mode_does_not_correct():
    read = reading(*make_reading(**ONE_MISREAD))
    off = solve_reading(read, nn_correction=False)
    assert off.status == "unreadable" and off.search is None and off.solution is None
    assert "correction is OFF" in format_result(off)
    assert solve_reading(read, nn_correction=True).status == "corrected"


@pytest.mark.impossible
def test_confident_duplicates_give_clear_no_solution_message():
    grid, probs = make_reading(misreads={X: (2, 0.999, 0.0005)}, sure=0.98)
    probs[Y] = 0.0005 / 8
    probs[Y][1] = 0.9995  # the clashing 2 at (0,4) is just as confident
    r = solve_reading(reading(grid, probs), algorithm="astar")
    assert r.status == "unreadable" and r.message == UNREADABLE_MESSAGE == "No solution: puzzle unreadable"
    assert r.solution is None and r.search.status == "exhausted"
    assert len(r.least_confident) == config.N_UNREADABLE_REPORT
    conf = [u.top3[0][1] for u in r.least_confident]
    assert conf == sorted(conf)                                  # least confident first
    assert all(len(u.top3) == 3 for u in r.least_confident)
    text = format_result(r)
    assert "No solution: puzzle unreadable" in text and "least confident cells" in text


@pytest.mark.impossible
def test_too_many_misreads_for_the_budget_is_unreadable():
    misreads = {X: (2, 0.5, 0.4), (8, 6): (5, 0.5, 0.4), (4, 0): (8, 0.5, 0.4)}
    r = solve_reading(reading(*make_reading(misreads=misreads)), max_corrections=2)
    assert r.status == "unreadable" and r.message == UNREADABLE_MESSAGE


# ---------------------------------------------------------------- end to end (trained reader)
@needs_model
@pytest.mark.normal
@pytest.mark.parametrize("algorithm", ["ucs", "greedy", "astar"])
def test_pipeline_recovers_one_misread_mosaic(algorithm):
    reader = DigitReader.load()
    test = mnist.load_splits()["test"]
    pools = injection.load_pools()
    p = generator.generate_puzzle(config.DEFAULT_GIVENS, generator.puzzle_rng(config.SEED, 30, 0))
    image, _, cells = injection.make_level_mosaic(p.puzzle, 1, pools, test.images,
                                                  config.make_rng(1))
    r = pipeline(image, reader, algorithm=algorithm)
    assert r.status == "corrected" and r.exact_recovery(p.solution)
    assert [c.cell for c in r.corrections] == cells
    assert not solve_reading(r.read, nn_correction=False).solved  # NN off cannot fix it


@needs_model
def test_pipeline_on_clean_mosaic_reads_exactly():
    reader = DigitReader.load()
    test = mnist.load_splits()["test"]
    pools = injection.load_pools()
    p = generator.generate_puzzle(25, generator.puzzle_rng(config.SEED, 25, 0))
    image, _, _ = injection.make_level_mosaic(p.puzzle, 0, pools, test.images, config.make_rng(2))
    r = pipeline(image, reader)
    assert np.array_equal(r.read.grid, p.puzzle)
    assert r.status == "solved" and r.exact_recovery(p.solution)


def test_known_puzzle_constant_is_consistent():
    assert generator.count_solutions(KNOWN) == 1
