"""The integrated system: mosaic image -> NN reading -> AC-3 check -> correction search -> solution.

    pipeline(mosaic)           reads the image with the CNN, then calls solve_reading
    solve_reading(read)        steps 2-4 on an existing reading (experiments reuse one reading)

1. Blank detection (brightness rule) + CNN reading -> read grid + probabilities per cell.
2. AC-3 check of the read grid. If it is consistent and the CSP solves it (uniquely, in "unique"
   goal mode) -> status "solved", no corrections.
3. Otherwise, with ``nn_correction=True``, run the correction search (UCS / Greedy / A*); the first
   goal state popped wins -> status "corrected".
4. If nothing works within the budget (or ``nn_correction=False``, the "NN off" mode) -> status
   "unreadable", message "No solution: puzzle unreadable", plus the 5 least confident cells with
   their top-3 probabilities.
"""

from dataclasses import dataclass, field

import numpy as np

from sudoku import config
from sudoku.csp.ac3 import ac3
from sudoku.csp.solver import Stats
from sudoku.csp.sudoku_csp import cell as var_to_cell, initial_domains
from sudoku.nn.reader import DigitReader, ReadGrid
from sudoku.search import CorrectionProblem, SearchResult, duplicate_conflicts, search

SOLVED, CORRECTED, UNREADABLE = "solved", "corrected", "unreadable"
UNREADABLE_MESSAGE = "No solution: puzzle unreadable"


@dataclass
class AC3Check:
    """Step 2: what AC-3 says about the raw read grid."""

    consistent: bool
    wiped_cell: tuple | None        # cell whose domain AC-3 emptied
    duplicates: list                # duplicate-digit conflicts among givens (sets of cells)
    stats: Stats


@dataclass
class UnsureCell:
    """A low-confidence cell listed when the puzzle is unreadable."""

    cell: tuple
    read_digit: int
    top3: list  # [(digit, probability), ...] best first


@dataclass
class PipelineResult:
    """Everything the demo and the experiments need about one run."""

    status: str                         # "solved", "corrected" or "unreadable"
    message: str
    read: ReadGrid
    ac3_check: AC3Check
    nn_correction: bool
    algorithm: str
    search: SearchResult | None = None  # None if no search was needed / allowed
    corrections: list = field(default_factory=list)  # CorrectionRow list
    total_cost: float = 0.0
    solution: np.ndarray | None = None
    least_confident: list = field(default_factory=list)  # UnsureCell list (when unreadable)

    @property
    def solved(self) -> bool:
        return self.status in (SOLVED, CORRECTED)

    def exact_recovery(self, true_solution) -> bool:
        """Did we output exactly the generator's original solution?"""
        return self.solution is not None and np.array_equal(self.solution, true_solution)


def check_ac3(grid) -> AC3Check:
    """Run AC-3 on the read grid (step 2) and list duplicate givens for the demo."""
    stats = Stats()
    ok, wiped = ac3(initial_domains(grid), stats)
    return AC3Check(ok, None if ok else var_to_cell(wiped), duplicate_conflicts(grid), stats)


def least_confident_cells(read: ReadGrid, n: int = config.N_UNREADABLE_REPORT) -> list:
    """The n given cells with the lowest top-1 probability, with their top-3 digits."""
    conf = read.confidence()
    givens = [(int(r), int(c)) for r, c in zip(*np.nonzero(read.grid))]
    givens.sort(key=lambda cell: (conf[cell], cell))
    return [UnsureCell(cell, int(read.grid[cell]), read.top_k(*cell, 3)) for cell in givens[:n]]


def solve_reading(read: ReadGrid, algorithm: str = config.DEFAULT_ALGORITHM,
                  nn_correction: bool = True, goal_mode: str = config.GOAL_MODE,
                  **search_kwargs) -> PipelineResult:
    """Steps 2-4 on a reading. ``search_kwargs`` go to the CorrectionProblem / search."""
    problem_kwargs = {k: search_kwargs.pop(k) for k in
                      ("n_candidates", "alts_per_cell", "max_corrections") if k in search_kwargs}
    problem = CorrectionProblem(read.grid, read.probs, goal_mode=goal_mode, **problem_kwargs)
    result = PipelineResult(UNREADABLE, UNREADABLE_MESSAGE, read, check_ac3(read.grid),
                            nn_correction, algorithm)

    # Step 2: consistent and solvable as read -> no corrections.
    if result.ac3_check.consistent and problem.goal_test(problem.initial_state):
        result.status, result.message = SOLVED, "Solved as read (no corrections needed)"
        result.solution = problem.solutions[problem.initial_state]
        return result

    # Step 3: NN-guided correction search.
    if nn_correction:
        found = search(problem, algorithm, **search_kwargs)
        result.search = found
        if found.found:
            result.status = CORRECTED
            result.message = (f"Solved after {len(found.corrections)} correction(s), "
                              f"total cost {found.total_cost:.3f}")
            result.corrections, result.total_cost = found.corrections, found.total_cost
            result.solution = found.solution
            return result

    # Step 4: unreadable.
    result.least_confident = least_confident_cells(read)
    return result


def pipeline(mosaic: np.ndarray, reader: DigitReader | None = None, **kwargs) -> PipelineResult:
    """Full system on one 252x252 mosaic (step 1 + ``solve_reading``)."""
    reader = reader or DigitReader.load()
    return solve_reading(reader.read_mosaic(mosaic), **kwargs)


def format_result(result: PipelineResult) -> str:
    """Readable multi-line summary (used by run_demo.py)."""
    lines = [f"status: {result.status} - {result.message}"]
    check = result.ac3_check
    if check.consistent:
        lines.append("AC-3 on the read grid: consistent")
    else:
        lines.append(f"AC-3 on the read grid: CONTRADICTION (domain of cell {check.wiped_cell} emptied)")
    for conflict in check.duplicates:
        lines.append(f"  duplicate digit {result.read.grid[next(iter(conflict))]} at "
                     f"{sorted(conflict)}")
    for row in result.corrections:
        lines.append(f"  correction {row.cell}: {row.read_digit} -> {row.new_digit}"
                     f"  cost {row.cost:.3f}")
    if result.status == UNREADABLE:
        if result.search is not None:
            lines.append(f"  search stopped: {result.search.status} "
                         f"({result.search.stats.nodes_expanded} nodes expanded)")
        elif not result.nn_correction:
            lines.append("  NN-guided correction is OFF: the raw reading was not solvable")
        lines.append(f"  {len(result.least_confident)} least confident cells:")
        for u in result.least_confident:
            top = ", ".join(f"{d}: {p:.2f}" for d, p in u.top3)
            lines.append(f"    {u.cell} read {u.read_digit}  top-3 [{top}]")
    return "\n".join(lines)
