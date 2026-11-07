"""Backtracking Sudoku solver with switchable Forward Checking, AC-3 preprocessing and MAC.

    solve(grid, use_fc, use_ac3)            -> Result(status, solution, stats, reason)
    count_solutions(grid, limit=2, ...)     -> CountResult(count, solutions, status, stats)

Parts:
- Backtracking (always on): pick an unassigned variable, try its domain values in increasing order,
  keep a value only if no assigned peer already has it, recurse, undo on failure.
- Forward Checking (use_fc): after assigning v = d, remove d from the domains of v's unassigned
  peers; if a domain becomes empty, undo at once. Givens are forward-checked before search.
- AC-3 (use_ac3): run once on the initial domains before search; an empty domain means the puzzle
  is contradictory (e.g. two equal givens in a unit) and search never starts.
- MAC (use_mac, extension, needs use_ac3): after each assignment, run AC-3 again starting from
  the arcs into v and into every cell FC just pruned.

Variable ordering is the SAME function in every mode (``config.ORDERING``):
- "static": first unassigned cell in row-major order;
- "mrv": unassigned cell with the fewest values not used by assigned peers (ties: row-major).
  It is computed from the assignment only, never from pruned domains, so FC/AC-3 cannot change
  the rule itself.

Counters (``Stats``):
- assignments:   values given to a variable during search (givens are not counted);
- backtracks:    assignments undone because they failed (no solution below them, or FC/MAC
                 wipe-out); when counting several solutions, undoing an assignment that led to
                 a solution is not a backtrack;
- ac3_revisions: revise(xi, xj) calls, in preprocessing and in MAC;
- ac3_pruned / fc_pruned: domain values removed by AC-3 (incl. MAC) / by forward checking;
- seconds:       wall-clock time.
Limits: ``node_limit`` assignments or ``time_limit`` seconds -> status "timeout" (never hangs).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from sudoku import config
from sudoku.csp.ac3 import ac3
from sudoku.csp.sudoku_csp import DIGITS, N, N_VARS, PEERS, cell, given_conflict, initial_domains

SOLVED, UNSOLVABLE, TIMEOUT = "solved", "unsolvable", "timeout"


@dataclass
class Stats:
    """Work done by one solver run (see module docstring for exact definitions)."""

    assignments: int = 0
    backtracks: int = 0
    ac3_revisions: int = 0
    ac3_pruned: int = 0
    fc_pruned: int = 0
    seconds: float = 0.0

    @property
    def values_pruned(self) -> int:
        """All domain values removed, by AC-3/MAC and forward checking together."""
        return self.ac3_pruned + self.fc_pruned

    def as_dict(self) -> dict:
        """Plain dict for tables, including the derived ``values_pruned``."""
        return {**self.__dict__, "values_pruned": self.values_pruned}


@dataclass
class Result:
    """Outcome of ``solve``."""

    status: str                      # "solved", "unsolvable" or "timeout"
    solution: np.ndarray | None      # (9, 9) grid if solved
    stats: Stats
    reason: str = ""                 # human-readable explanation when not solved
    conflict_cell: tuple | None = None  # cell whose domain AC-3/FC emptied before search


@dataclass
class CountResult:
    """Outcome of ``count_solutions``. If status is "timeout", ``count`` is only a lower bound."""

    count: int
    solutions: list = field(default_factory=list)
    status: str = SOLVED
    stats: Stats = field(default_factory=Stats)


class _Timeout(Exception):
    """Raised inside the search when a node or time limit is hit."""


class _Search:
    """One run of the backtracking search; holds domains, the assignment and counters."""

    def __init__(self, grid, use_fc, use_ac3, use_mac, ordering, node_limit, time_limit,
                 solution_limit):
        self.grid = np.asarray(grid)
        if use_mac and not use_ac3:
            raise ValueError("use_mac needs use_ac3: MAC maintains the arc consistency AC-3 sets up")
        self.use_fc, self.use_ac3, self.use_mac = use_fc, use_ac3, use_mac
        if ordering not in ("static", "mrv"):
            raise ValueError(f"unknown ordering {ordering!r}")
        self.ordering = ordering
        self.node_limit, self.time_limit = node_limit, time_limit
        self.solution_limit = solution_limit
        self.stats = Stats()
        self.domains = initial_domains(self.grid)
        self.value = [int(d) for d in self.grid.ravel()]  # 0 = unassigned
        self.solutions = []
        self.start = time.perf_counter()

    # ------------------------------------------------------------ preprocessing
    def preprocess(self):
        """Checks before search. Returns (ok, reason, conflict_variable)."""
        if not self.use_ac3:
            # Without AC-3 nothing compares two givens with each other (search and FC only look
            # at unassigned cells), so check the givens directly.
            clash = given_conflict(self.grid)
            if clash:
                v1, v2, d = clash
                return False, (f"No solution: givens conflict: digit {d} at {cell(v1)} "
                               f"and {cell(v2)}"), v2
        if self.use_ac3:
            ok, wiped = ac3(self.domains, self.stats)
            if not ok:
                return False, f"No solution: AC-3 emptied the domain of cell {cell(wiped)}", wiped
        if self.use_fc:
            for v in range(N_VARS):
                if self.value[v]:
                    ok, wiped = self.forward_check(v, self.value[v], [])
                    if not ok:
                        return False, ("No solution: forward checking emptied the domain of "
                                       f"cell {cell(wiped)}"), wiped
        return True, "", None

    # ------------------------------------------------------------ search pieces
    def legal_count(self, v: int) -> int:
        """Values not already used by an assigned peer (MRV measure, same in every mode)."""
        return len(DIGITS - {self.value[u] for u in PEERS[v]})

    def select_variable(self):
        """Next unassigned variable by ``self.ordering``; None when all are assigned."""
        unassigned = [v for v in range(N_VARS) if not self.value[v]]
        if not unassigned:
            return None
        if self.ordering == "static":
            return unassigned[0]
        return min(unassigned, key=lambda v: (self.legal_count(v), v))

    def consistent(self, v: int, d: int) -> bool:
        """Backtracking check: no assigned peer of v holds d."""
        return all(self.value[u] != d for u in PEERS[v])

    def forward_check(self, v: int, d: int, removed: list):
        """Remove d from unassigned peers of v. Returns (ok, emptied_variable)."""
        for u in PEERS[v]:
            if not self.value[u] and d in self.domains[u]:
                self.domains[u].discard(d)
                removed.append((u, d))
                self.stats.fc_pruned += 1
                if not self.domains[u]:
                    return False, u
        return True, None

    @staticmethod
    def mac_queue(v: int, removed: list) -> list:
        """Arcs to start MAC from: every arc (xk, x) into v or into a cell FC just pruned.

        Starting only from arcs (u, v) would do nothing after FC, because FC already removed v's
        value from its peers; the new information is in the domains that shrank.
        """
        changed = {v} | {u for u, _ in removed}
        return [(xk, x) for x in sorted(changed) for xk in PEERS[x]]

    def check_limits(self):
        """Raise _Timeout if the node or time budget is used up."""
        if self.stats.assignments >= self.node_limit:
            raise _Timeout(f"node limit {self.node_limit} reached")
        if time.perf_counter() - self.start > self.time_limit:
            raise _Timeout(f"time limit {self.time_limit}s reached")

    def search(self) -> bool:
        """Depth-first backtracking. Returns True when enough solutions were found (stop)."""
        v = self.select_variable()
        if v is None:
            self.solutions.append(np.array(self.value).reshape(N, N))
            return len(self.solutions) >= self.solution_limit
        for d in sorted(self.domains[v]):
            if not self.consistent(v, d):
                continue
            self.check_limits()
            self.stats.assignments += 1
            self.value[v] = d
            saved_domain, self.domains[v] = self.domains[v], {d}
            removed = []
            ok = True
            if self.use_fc:
                ok, _ = self.forward_check(v, d, removed)
            if ok and self.use_mac:
                ok, _ = ac3(self.domains, self.stats, queue=self.mac_queue(v, removed),
                            removed=removed)
            found_before = len(self.solutions)
            if ok and self.search():
                return True
            for u, val in removed:  # undo FC / MAC pruning
                self.domains[u].add(val)
            self.domains[v] = saved_domain
            self.value[v] = 0
            if len(self.solutions) == found_before:  # nothing found below: a real failure
                self.stats.backtracks += 1
        return False


def _run(grid, use_fc, use_ac3, use_mac, ordering, node_limit, time_limit, solution_limit):
    """Preprocess and search; returns (search object, status, reason, conflict_cell)."""
    s = _Search(grid, use_fc, use_ac3, use_mac, ordering, node_limit, time_limit, solution_limit)
    ok, reason, wiped = s.preprocess()
    if not ok:
        status, conflict = UNSOLVABLE, cell(wiped)
    else:
        conflict = None
        try:
            s.search()
            status = SOLVED if s.solutions else UNSOLVABLE
            reason = "" if s.solutions else ("No solution: search exhausted, no assignment "
                                             "satisfies all constraints")
        except _Timeout as stop:
            status, reason = TIMEOUT, str(stop)
    s.stats.seconds = time.perf_counter() - s.start
    return s, status, reason, conflict


def solve(grid, use_fc: bool = True, use_ac3: bool = True, use_mac: bool = config.USE_MAC,
          ordering: str = config.ORDERING, node_limit: int = config.CSP_NODE_LIMIT,
          time_limit: float = config.CSP_TIME_LIMIT) -> Result:
    """Solve a 9x9 grid (0 = empty) with backtracking plus the chosen FC / AC-3 / MAC switches."""
    s, status, reason, conflict = _run(grid, use_fc, use_ac3, use_mac, ordering,
                                       node_limit, time_limit, solution_limit=1)
    solution = s.solutions[0] if s.solutions else None
    return Result(status, solution, s.stats, reason, conflict)


def count_solutions(grid, limit: int = 2, use_fc: bool = True, use_ac3: bool = True,
                    use_mac: bool = config.USE_MAC, ordering: str = config.ORDERING,
                    node_limit: int = config.CSP_NODE_LIMIT,
                    time_limit: float = config.CSP_TIME_LIMIT) -> CountResult:
    """Count solutions up to ``limit`` (extension, for the uniqueness goal).

    status "solved" means the search finished (count is exact, or reached ``limit``);
    "unsolvable" means 0 solutions; "timeout" means count is a lower bound.
    """
    s, status, _, _ = _run(grid, use_fc, use_ac3, use_mac, ordering, node_limit, time_limit,
                           solution_limit=limit)
    if status != TIMEOUT:
        status = SOLVED if s.solutions else UNSOLVABLE
    return CountResult(len(s.solutions), s.solutions, status, s.stats)
