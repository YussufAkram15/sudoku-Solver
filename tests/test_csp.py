"""Tests for the Sudoku CSP: BT / FC / AC-3 / MAC switches, counters, limits, solution counting."""

import numpy as np
import pytest

from sudoku import config
from sudoku.csp import count_solutions, solve
from sudoku.csp.ac3 import ac3, revise
from sudoku.csp.solver import Stats
from sudoku.csp.sudoku_csp import ARCS, PEERS, initial_domains, var
from sudoku.data import generator
from tests.test_data import _unavoidable_rectangle

N = config.GRID


def parse(text: str) -> np.ndarray:
    """81-character string (0 = empty) -> (9, 9) grid."""
    return np.array([int(ch) for ch in text], dtype=np.int64).reshape(N, N)


# A classic puzzle and its known solution (Project Euler 96, grid 01).
KNOWN = parse("003020600900305001001806400008102900700000008006708200002609500800203009005010300")
KNOWN_SOLUTION = parse(
    "483921657967345821251876493548132976729564138136798245372689514814253769695417382")
# A 17-given puzzle (the minimum for a unique Sudoku) from Gordon Royle's list.
SEVENTEEN = parse("000000010400000000020000000000050407008000300001090000300400200050100000000806000")

MODES = {
    "BT": dict(use_fc=False, use_ac3=False),
    "BT+FC": dict(use_fc=True, use_ac3=False),
    "BT+FC+AC3": dict(use_fc=True, use_ac3=True),
    "BT+AC3": dict(use_fc=False, use_ac3=True),
    "BT+FC+AC3+MAC": dict(use_fc=True, use_ac3=True, use_mac=True),
}


def _assert_solves(grid, result):
    assert result.status == "solved", result.reason
    assert generator.is_valid_solution(result.solution)
    given = grid != 0
    assert np.array_equal(result.solution[given], grid[given])


# ---------------------------------------------------------------- structure
def test_constraint_graph_shape():
    assert all(len(p) == 20 for p in PEERS)
    assert len(ARCS) == 81 * 20
    assert var(4, 4) in PEERS[var(3, 3)] and var(4, 4) not in PEERS[var(3, 6)]


# ---------------------------------------------------------------- normal cases
@pytest.mark.normal
@pytest.mark.parametrize("ordering", ["static", "mrv"])
def test_known_puzzle_same_solution_in_every_mode(ordering):
    for name, kw in MODES.items():
        result = solve(KNOWN, ordering=ordering, **kw)
        _assert_solves(KNOWN, result)
        assert np.array_equal(result.solution, KNOWN_SOLUTION), name


@pytest.mark.normal
def test_bank_puzzles_solved_identically_by_bt_fc_ac3():
    for i in range(5):
        p = generator.generate_puzzle(30, generator.puzzle_rng(config.SEED, 30, i))
        solutions = [solve(p.puzzle, **MODES[m]).solution for m in ("BT", "BT+FC", "BT+FC+AC3")]
        for s in solutions:
            assert np.array_equal(s, p.solution)


@pytest.mark.normal
def test_pruning_never_increases_search_with_static_order():
    # Same variable and value order: FC only cuts subtrees BT would fail in, and AC-3 only removes
    # values that cannot be part of a solution, so the search trees are nested.
    for i in range(5):
        p = generator.generate_puzzle(30, generator.puzzle_rng(config.SEED, 30, i)).puzzle
        bt, fc, ac = (solve(p, ordering="static", **MODES[m]).stats
                      for m in ("BT", "BT+FC", "BT+FC+AC3"))
        assert bt.assignments >= fc.assignments >= ac.assignments
        assert bt.fc_pruned == 0 and bt.ac3_revisions == 0      # switches really off
        assert fc.fc_pruned > 0 and fc.ac3_revisions == 0
        assert ac.ac3_revisions > 0 and ac.ac3_pruned > 0


def test_counters_are_consistent():
    r = solve(KNOWN, use_fc=False, use_ac3=False, ordering="static")
    empties = int((KNOWN == 0).sum())
    # every assignment is either kept in the final solution or undone by exactly one backtrack
    assert r.stats.assignments - r.stats.backtracks == empties
    assert r.stats.values_pruned == r.stats.ac3_pruned + r.stats.fc_pruned
    assert r.stats.seconds > 0


def test_ac3_shrinks_domains_before_search():
    grid = KNOWN_SOLUTION.copy()
    grid[0, 0] = 0  # one empty cell: AC-3 alone narrows it to the missing digit
    stats = Stats()
    domains = initial_domains(grid)
    ok, _ = ac3(domains, stats)
    assert ok and domains[0] == {KNOWN_SOLUTION[0, 0]}
    assert stats.ac3_pruned == 8 and stats.ac3_revisions >= 20


def test_revise_removes_only_unsupported_value():
    domains = [{1, 2, 3}, {2}, {2, 3}]
    assert revise(domains, 0, 1) == 2 and domains[0] == {1, 3}   # 2 has no support in {2}
    assert revise(domains, 0, 2) is None and domains[0] == {1, 3}  # {2, 3} supports everything


def test_mac_requires_ac3():
    with pytest.raises(ValueError, match="use_mac needs use_ac3"):
        solve(KNOWN, use_fc=True, use_ac3=False, use_mac=True)


def test_ac3_queue_has_no_duplicate_arcs():
    # From scratch every arc is revised once; a re-queued arc is never queued twice at a time,
    # so revisions stay far below the old duplicate-allowing count.
    stats = Stats()
    ok, _ = ac3(initial_domains(KNOWN), stats)
    assert ok and len(ARCS) <= stats.ac3_revisions < 3 * len(ARCS)


def test_counting_does_not_call_found_solutions_backtracks():
    # Exactly two solutions: blank an "unavoidable rectangle" (a b / b a in two boxes). Static BT
    # tries a then b in the first cell; both branches end in a solution, so the 8 assignments are
    # all on solution paths and none of the undos is a failure.
    for seed in range(50):
        solution = generator.generate_full_grid(config.make_rng(seed))
        found = _unavoidable_rectangle(solution)
        if found:
            break
    r1, r2, c1, c2 = found
    grid = solution.copy()
    grid[[r1, r1, r2, r2], [c1, c2, c1, c2]] = 0
    r = count_solutions(grid, limit=10, use_fc=False, use_ac3=False, ordering="static")
    assert r.count == 2
    assert r.stats.assignments == 8 and r.stats.backtracks == 0


@pytest.mark.normal
def test_count_solutions_unique_and_multiple():
    assert count_solutions(KNOWN, limit=2).count == 1
    four_blanks = KNOWN_SOLUTION.copy()
    four_blanks[[0, 0, 1, 1], [0, 1, 0, 1]] = 0  # cells held 4 8 / 9 6: no swap possible
    assert count_solutions(four_blanks).count == 1
    empty = np.zeros((N, N), np.int64)
    r = count_solutions(empty, limit=2)
    assert r.count == 2 and r.status == "solved"
    assert all(generator.is_valid_solution(s) for s in r.solutions)
    assert not np.array_equal(r.solutions[0], r.solutions[1])


def test_count_solutions_matches_independent_counter():
    # Cross-check against the generator's separate bitmask counter on random partial grids.
    rng = config.make_rng(5)
    for _ in range(10):
        full = generator.generate_full_grid(rng)
        grid = full.copy()
        grid.ravel()[rng.choice(81, size=50, replace=False)] = 0
        assert count_solutions(grid, limit=3).count == generator.count_solutions(grid, limit=3)


# ---------------------------------------------------------------- edge cases
@pytest.mark.edge
def test_already_solved_grid_needs_no_search():
    for name, kw in MODES.items():
        r = solve(KNOWN_SOLUTION, **kw)
        assert r.status == "solved" and np.array_equal(r.solution, KNOWN_SOLUTION), name
        assert r.stats.assignments == 0 and r.stats.backtracks == 0


@pytest.mark.edge
def test_minimum_givens_puzzle():
    assert (SEVENTEEN != 0).sum() == config.MIN_GIVENS
    assert generator.count_solutions(SEVENTEEN, limit=2) == 1  # independent check: unique
    r = solve(SEVENTEEN, use_fc=True, use_ac3=True, ordering="mrv")
    _assert_solves(SEVENTEEN, r)
    assert count_solutions(SEVENTEEN, limit=2, ordering="mrv").count == 1


@pytest.mark.edge
def test_timeout_is_reported_not_hung():
    empty = np.zeros((N, N), np.int64)
    clash_free_hard = SEVENTEEN  # static-order BT needs far more than 50 assignments here
    r = solve(clash_free_hard, use_fc=False, use_ac3=False, ordering="static", node_limit=50)
    assert r.status == "timeout" and r.solution is None and r.stats.assignments == 50
    assert "node limit" in r.reason
    r = solve(empty, use_fc=False, use_ac3=False, time_limit=0.0)
    assert r.status == "timeout" and "time limit" in r.reason
    c = count_solutions(SEVENTEEN, limit=2, use_fc=False, use_ac3=False, node_limit=50)
    assert c.status == "timeout"


# ---------------------------------------------------------------- impossible cases
def _two_equal_givens():
    grid = KNOWN.copy()
    grid[0, 0] = 2  # row 0 already has a 2 at (0, 4)
    return grid


@pytest.mark.impossible
def test_ac3_reports_contradictory_givens_before_search():
    r = solve(_two_equal_givens(), use_fc=True, use_ac3=True)
    assert r.status == "unsolvable" and r.solution is None
    assert "AC-3" in r.reason and r.stats.assignments == 0
    assert r.conflict_cell in [(0, 0), (0, 4)]


@pytest.mark.impossible
@pytest.mark.parametrize("mode", ["BT", "BT+FC"])
def test_contradictory_givens_without_ac3_are_still_caught(mode):
    r = solve(_two_equal_givens(), **MODES[mode])
    assert r.status == "unsolvable" and "givens conflict" in r.reason
    assert r.stats.assignments == 0


@pytest.mark.impossible
def test_unsolvable_without_duplicate_givens():
    # Change one given to a digit that clashes with nothing directly but breaks the puzzle.
    grid = KNOWN.copy()
    grid[0, 2] = 5  # was 3; 5 is not in row 0, column 2 or the top-left box
    assert generator.count_solutions(grid, limit=1) == 0  # independent check
    for name, kw in MODES.items():
        r = solve(grid, **kw)
        assert r.status == "unsolvable" and r.solution is None, name
        assert r.reason
    assert count_solutions(grid).count == 0
