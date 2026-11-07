"""Tests for the correction search: costs, hand-made example with known answer, budget edge case,
impossible case, admissibility / consistency of the A* heuristic by brute force, random baseline."""

import itertools
import math

import numpy as np
import pytest

from sudoku import config
from sudoku.search import (CorrectionProblem, conflict_heuristic, duplicate_conflicts,
                           random_correction, search)
from tests.test_csp import KNOWN, KNOWN_SOLUTION

SURE = 0.99  # confidence of an ordinary, correctly read given


def make_reading(misreads=None, soft=None, sure=SURE):
    """Hand-made reader output for the KNOWN puzzle.

    misreads: {cell: (read_digit, p_read, p_true)} - the network prefers the wrong digit.
    soft:     {cell: (other_digit, p_true, p_other)} - right read, but a plausible second choice.
    Every other given: p(true digit) = ``sure``, rest spread evenly.
    """
    grid = KNOWN.copy()
    probs = np.zeros((9, 9, 9))
    for r, c in zip(*np.nonzero(KNOWN)):
        probs[r, c] = (1 - sure) / 8
        probs[r, c, KNOWN[r, c] - 1] = sure
    for (r, c), (read, p_read, p_true) in (misreads or {}).items():
        probs[r, c] = (1 - p_read - p_true) / 7
        probs[r, c, read - 1], probs[r, c, KNOWN[r, c] - 1] = p_read, p_true
        grid[r, c] = read
    for (r, c), (other, p_true, p_other) in (soft or {}).items():
        probs[r, c] = (1 - p_true - p_other) / 7
        probs[r, c, KNOWN[r, c] - 1], probs[r, c, other - 1] = p_true, p_other
    return grid, probs


# (0,2) is really 3; read as 2 it clashes with the given 2 at (0,4) in row 0.
X, Y, Z = (0, 2), (0, 4), (8, 4)
ONE_MISREAD = dict(misreads={X: (2, 0.5, 0.4)}, soft={Z: (7, 0.6, 0.35)})


# ---------------------------------------------------------------- costs and candidates
def test_step_costs_are_non_negative_and_sorted():
    grid, probs = make_reading(**ONE_MISREAD)
    problem = CorrectionProblem(grid, probs)
    assert len(problem.candidates) == config.CANDIDATE_CELLS
    assert problem.candidates[:2] == [X, Z]                 # least confident first
    for cell, alts in problem.alternatives.items():
        assert len(alts) == config.ALTS_PER_CELL
        assert all(a.cost >= 0 for a in alts)
        assert [a.cost for a in alts] == sorted(a.cost for a in alts)
        assert all(a.digit != grid[cell] for a in alts)    # never the read digit
    assert problem.alternatives[X][0].digit == 3
    assert problem.alternatives[X][0].cost == pytest.approx(math.log(0.5 / 0.4))


def test_read_digit_must_be_the_argmax():
    grid, probs = make_reading(**ONE_MISREAD)
    grid[Z] = 7  # 7 has p = 0.35 at Z, the argmax is the true 1
    with pytest.raises(ValueError, match="negative"):
        CorrectionProblem(grid, probs)


def test_duplicate_conflicts_found_in_rows_columns_boxes():
    grid, _ = make_reading(**ONE_MISREAD)
    # the misread 2 at (0,2) repeats the 2 at (0,4) in its row and the 2 at (6,2) in its column
    assert set(duplicate_conflicts(grid)) == {frozenset({X, Y}), frozenset({X, (6, 2)})}
    assert duplicate_conflicts(KNOWN) == []


# ---------------------------------------------------------------- normal: known answer
@pytest.mark.normal
def test_ucs_on_hand_made_example_known_cost_and_nodes():
    # 3 candidates: X (p .5), Z (p .6), then Y (first .99 cell in row-major order).
    grid, probs = make_reading(**ONE_MISREAD)
    problem = CorrectionProblem(grid, probs, n_candidates=3)
    assert problem.candidates == [X, Z, Y]
    r = search(problem, "ucs")
    assert r.found and r.state == frozenset({(X, 3)})
    assert r.total_cost == pytest.approx(math.log(0.5 / 0.4))
    # root fails (duplicate) and is expanded into 3 cells x 3 alternatives; the cheapest child
    # (X -> 3, cost 0.223) is the goal.
    assert (r.stats.nodes_expanded, r.stats.nodes_generated, r.stats.goal_tests) == (1, 9, 2)
    assert r.stats.csp_calls == 1  # the root never reaches the CSP: duplicate check fails first
    assert np.array_equal(r.solution, KNOWN_SOLUTION)
    row = r.corrections[0]
    assert (row.cell, row.read_digit, row.new_digit) == (X, 2, 3)
    assert [t.state for t in r.trace] == [frozenset(), frozenset({(X, 3)})]


@pytest.mark.normal
@pytest.mark.parametrize("algorithm", ["greedy", "astar"])
def test_greedy_and_astar_find_the_same_correction(algorithm):
    grid, probs = make_reading(**ONE_MISREAD)
    r = search(CorrectionProblem(grid, probs), algorithm)
    assert r.found and r.state == frozenset({(X, 3)})
    assert np.array_equal(r.solution, KNOWN_SOLUTION)
    assert r.trace[0].h == pytest.approx(math.log(0.5 / 0.4))  # h(root) = cheapest fix of X/Y


@pytest.mark.normal
def test_astar_expands_no_more_than_ucs_and_costs_match():
    grid, probs = make_reading(misreads={X: (2, 0.5, 0.4), (8, 6): (5, 0.45, 0.4)},
                               soft={Z: (7, 0.6, 0.35), (4, 4): (3, 0.7, 0.25)})
    results = {a: search(CorrectionProblem(grid, probs), a) for a in ("ucs", "astar", "greedy")}
    assert all(r.found for r in results.values())
    assert results["astar"].total_cost == pytest.approx(results["ucs"].total_cost)
    assert results["astar"].stats.nodes_expanded <= results["ucs"].stats.nodes_expanded
    assert np.array_equal(results["ucs"].solution, KNOWN_SOLUTION)


@pytest.mark.edge
def test_no_misreads_needs_no_search():
    grid, probs = make_reading()
    for algorithm in ("ucs", "greedy", "astar"):
        r = search(CorrectionProblem(grid, probs), algorithm)
        assert r.found and r.state == frozenset() and r.total_cost == 0
        assert r.stats.nodes_expanded == 0 and r.stats.goal_tests == 1


# ---------------------------------------------------------------- edge: exactly the budget
THREE_MISREADS = {X: (2, 0.5, 0.4), (8, 6): (5, 0.5, 0.4), (4, 0): (8, 0.5, 0.4)}


@pytest.mark.edge
@pytest.mark.parametrize("algorithm", ["ucs", "astar"])
def test_correction_needed_at_max_budget(algorithm):
    grid, probs = make_reading(misreads=THREE_MISREADS)
    assert len(duplicate_conflicts(grid)) >= 3
    r = search(CorrectionProblem(grid, probs, max_corrections=3), algorithm)
    assert r.found and len(r.corrections) == 3
    assert {(row.cell, row.new_digit) for row in r.corrections} == {(X, 3), ((8, 6), 3), ((4, 0), 7)}
    assert np.array_equal(r.solution, KNOWN_SOLUTION)
    assert r.total_cost == pytest.approx(3 * math.log(0.5 / 0.4))
    short = search(CorrectionProblem(grid, probs, max_corrections=2), algorithm)
    assert short.status == "exhausted" and short.solution is None


# ---------------------------------------------------------------- impossible
@pytest.mark.impossible
@pytest.mark.parametrize("algorithm", ["ucs", "greedy", "astar"])
def test_confident_duplicate_cannot_be_fixed_within_budget(algorithm):
    # (0,2) confidently read as 2 next to a confident 2 at (0,4); every other given is less sure,
    # so neither clashing cell is among the 15 candidates and no correction can break the clash.
    grid, probs = make_reading(misreads={X: (2, 0.999, 0.0005)}, sure=0.98)
    probs[Y] = 0.0005 / 8
    probs[Y][1] = 0.9995
    problem = CorrectionProblem(grid, probs)
    assert X not in problem.alternatives and Y not in problem.alternatives
    assert math.isinf(conflict_heuristic(problem, frozenset()))
    r = search(problem, algorithm)
    assert r.status == "exhausted" and not r.found and r.solution is None
    if algorithm != "ucs":  # h = infinity prunes every child: nothing beyond the root
        assert r.stats.nodes_expanded == 1
        assert r.stats.nodes_pruned == r.stats.nodes_generated > 0
    else:
        assert r.stats.nodes_pruned == 0


# ---------------------------------------------------------------- heuristic properties
def _small_problem():
    grid, probs = make_reading(misreads={X: (2, 0.5, 0.3), (8, 6): (5, 0.4, 0.35)},
                               soft={Z: (7, 0.55, 0.4)})
    return CorrectionProblem(grid, probs, n_candidates=5, goal_mode="solvable")


def _all_states(problem):
    """Every state reachable within the budget (each cell at most once)."""
    options = [[(cell, a.digit) for a in problem.alternatives[cell]] for cell in problem.candidates]
    states = [frozenset()]
    for k in range(1, problem.max_corrections + 1):
        for cells in itertools.combinations(range(len(options)), k):
            for choice in itertools.product(*(options[i] for i in cells)):
                states.append(frozenset(choice))
    return states


def test_astar_heuristic_is_admissible_and_consistent_by_brute_force():
    problem = _small_problem()
    states = _all_states(problem)
    assert len(states) == 1 + 15 + 90 + 270
    goal_cost = {s: problem.path_cost(s) for s in states if problem.goal_test(s)}
    assert goal_cost, "the small example must have at least one goal"
    for s in states:
        g = problem.path_cost(s)
        true_remaining = min((c - g for t, c in goal_cost.items() if s <= t), default=math.inf)
        h = conflict_heuristic(problem, s)
        assert h <= true_remaining + 1e-9, (s, h, true_remaining)   # admissible
        for child, cost, _, _ in problem.successors(s):             # consistent
            assert h <= cost + conflict_heuristic(problem, child) + 1e-9


def test_ucs_and_astar_return_the_brute_force_optimum():
    problem = _small_problem()
    best = min(problem.path_cost(s) for s in _all_states(problem) if problem.goal_test(s))
    assert search(problem, "ucs").total_cost == pytest.approx(best)
    assert search(problem, "astar").total_cost == pytest.approx(best)


def test_search_limits_stop_cleanly():
    grid, probs = make_reading(misreads=THREE_MISREADS)
    r = search(CorrectionProblem(grid, probs), "ucs", node_limit=5)
    assert r.status == "node_limit" and r.stats.nodes_expanded == 5
    r = search(CorrectionProblem(grid, probs), "ucs", time_limit=0.0)
    assert r.status == "time_limit"


# ---------------------------------------------------------------- random baseline
def test_random_baseline_is_seeded_and_uses_same_goal_test():
    grid, probs = make_reading(**ONE_MISREAD)
    problem = CorrectionProblem(grid, probs)
    a = random_correction(problem, config.make_rng(1), trials=50)
    b = random_correction(problem, config.make_rng(1), trials=50)
    assert [t.state for t in a.trace] == [t.state for t in b.trace]
    assert a.stats.goal_tests == len(a.trace) <= 51
    if not a.found:  # 1 root + 50 random sets, each generated set was tested
        assert a.stats.goal_tests == 51 and a.stats.nodes_generated == 50
    clean_grid, clean_probs = make_reading()
    r = random_correction(CorrectionProblem(clean_grid, clean_probs), config.make_rng(0))
    assert r.found and r.state == frozenset()  # read grid already fine
