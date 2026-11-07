"""The correction search problem: which given cells did the network misread?

- Read grid: the argmax digit of every given (non-blank) cell; 0 for blanks.
- Candidate cells: the ``n_candidates`` given cells with the LOWEST confidence (ties: row-major).
- Alternatives of a candidate: its ``alts_per_cell`` most probable digits other than the read one.
- State: frozenset of corrections {(cell, new_digit)}, at most one per cell, at most
  ``max_corrections`` in total. The initial state is the empty set (trust the network).
- Step cost of changing cell c from read digit r to d:  log p_c(r) - log p_c(d)  (>= 0, r is argmax).
  The path cost is the sum, i.e. -log of how much less likely the corrected reading is.
- Goal test: no two equal givens in a unit, AC-3 finds no contradiction, and the CSP finds a
  solution - exactly one if ``goal_mode == "unique"`` (extension), at least one if "solvable".
"""

from dataclasses import dataclass

import numpy as np

from sudoku import config
from sudoku.csp import count_solutions, solve
from sudoku.csp.sudoku_csp import given_conflict

N = config.GRID


@dataclass(frozen=True)
class Alternative:
    """One possible correction of a candidate cell."""

    digit: int
    cost: float


@dataclass
class GoalStats:
    """Work done by goal tests (filled in by the problem, read by the search)."""

    goal_tests: int = 0       # states tested
    csp_calls: int = 0        # goal tests that reached the CSP (no duplicate givens)
    csp_timeouts: int = 0     # CSP calls stopped by the goal-test node/time limit


def step_cost(probs_cell: np.ndarray, read_digit: int, new_digit: int) -> float:
    """log p(read) - log p(new) for one cell's probability vector over digits 1..9."""
    p_read = max(probs_cell[read_digit - 1], config.PROB_EPS)
    p_new = max(probs_cell[new_digit - 1], config.PROB_EPS)
    return float(np.log(p_read) - np.log(p_new))


class CorrectionProblem:
    """Search problem built from one read grid and its probabilities."""

    def __init__(self, read_grid, probs, n_candidates: int = config.CANDIDATE_CELLS,
                 alts_per_cell: int = config.ALTS_PER_CELL,
                 max_corrections: int = config.MAX_CORRECTIONS,
                 goal_mode: str = config.GOAL_MODE):
        if goal_mode not in ("unique", "solvable"):
            raise ValueError(f"unknown goal_mode {goal_mode!r}")
        self.read_grid = np.asarray(read_grid, dtype=np.int64)
        self.probs = np.asarray(probs, dtype=np.float64)
        self._check_read_is_argmax()
        self.max_corrections = max_corrections
        self.goal_mode = goal_mode
        self.alternatives = self._candidate_alternatives(n_candidates, alts_per_cell)
        self.candidates = list(self.alternatives)  # least confident first
        self.goal_stats = GoalStats()
        self.solutions = {}  # state -> solved grid, filled by goal_test

    def reset_counters(self) -> None:
        """Forget goal-test counts and cached solutions (called at the start of every search)."""
        self.goal_stats = GoalStats()
        self.solutions = {}

    # ------------------------------------------------------------ set-up
    def _check_read_is_argmax(self) -> None:
        """Step costs are only >= 0 if every read digit is its cell's most probable digit."""
        for r, c in zip(*np.nonzero(self.read_grid)):
            p = self.probs[r, c]
            if p[self.read_grid[r, c] - 1] < p.max():
                raise ValueError(f"cell {(int(r), int(c))}: read digit {self.read_grid[r, c]} is not "
                                 "the most probable one, step costs would be negative")

    def _candidate_alternatives(self, n_candidates: int, alts_per_cell: int) -> dict:
        """{cell: [Alternative, ...] cheapest first} for the least confident given cells."""
        givens = [(int(r), int(c)) for r, c in zip(*np.nonzero(self.read_grid))]
        confidence = {cell: self.probs[cell][self.read_grid[cell] - 1] for cell in givens}
        ranked = sorted(givens, key=lambda cell: (confidence[cell], cell))[:n_candidates]
        alternatives = {}
        for cell in ranked:
            read = int(self.read_grid[cell])
            order = np.argsort(-self.probs[cell], kind="stable") + 1  # digits, most probable first
            others = [int(d) for d in order if d != read][:alts_per_cell]
            alternatives[cell] = [Alternative(d, step_cost(self.probs[cell], read, d)) for d in others]
        return alternatives

    # ------------------------------------------------------------ states
    initial_state = frozenset()

    def apply(self, state: frozenset) -> np.ndarray:
        """The read grid with the corrections of ``state`` written in."""
        grid = self.read_grid.copy()
        for cell, digit in state:
            grid[cell] = digit
        return grid

    def path_cost(self, state: frozenset) -> float:
        """Sum of the step costs of all corrections in ``state``."""
        return sum(self.cost(cell, digit) for cell, digit in state)

    def cost(self, cell: tuple, digit: int) -> float:
        """Step cost of correcting ``cell`` to ``digit``."""
        return step_cost(self.probs[cell], int(self.read_grid[cell]), digit)

    def successors(self, state: frozenset) -> list:
        """All (child_state, step_cost, cell, digit): one more correction on an uncorrected cell."""
        if len(state) >= self.max_corrections:
            return []
        corrected = {cell for cell, _ in state}
        children = []
        for cell in self.candidates:
            if cell in corrected:
                continue
            for alt in self.alternatives[cell]:
                children.append((state | {(cell, alt.digit)}, alt.cost, cell, alt.digit))
        return children

    # ------------------------------------------------------------ goal test
    def goal_test(self, state: frozenset) -> bool:
        """Is the corrected puzzle consistent and solvable (uniquely, in "unique" mode)?"""
        self.goal_stats.goal_tests += 1
        grid = self.apply(state)
        if given_conflict(grid):  # same contradiction AC-3 would report, found without the CSP
            return False
        self.goal_stats.csp_calls += 1
        settings = dict(use_fc=config.GOAL_USE_FC, use_ac3=config.GOAL_USE_AC3,
                        use_mac=config.GOAL_USE_MAC, ordering=config.GOAL_ORDERING,
                        node_limit=config.GOAL_NODE_LIMIT, time_limit=config.GOAL_TIME_LIMIT)
        if self.goal_mode == "unique":
            result = count_solutions(grid, limit=2, **settings)
            ok = result.status != "timeout" and result.count == 1
            solution = result.solutions[0] if ok else None
        else:
            result = solve(grid, **settings)
            ok, solution = result.status == "solved", result.solution
        self.goal_stats.csp_timeouts += result.status == "timeout"
        if ok:
            self.solutions[state] = solution
        return ok


@dataclass
class CorrectionRow:
    """One line of the correction log."""

    cell: tuple
    read_digit: int
    new_digit: int
    cost: float


def correction_log(problem: CorrectionProblem, state: frozenset) -> list:
    """Correction log rows for ``state``, sorted by cell."""
    return [CorrectionRow(cell, int(problem.read_grid[cell]), digit, problem.cost(cell, digit))
            for cell, digit in sorted(state)]


def describe(state: frozenset) -> str:
    """Short text form of a state, e.g. '{(0,3)->7, (4,4)->1}'."""
    if not state:
        return "{}"
    return "{" + ", ".join(f"({r},{c})->{d}" for (r, c), d in sorted(state)) + "}"

