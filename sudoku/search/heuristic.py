"""A* heuristic for the correction search: at least one cell of every duplicate must change.

A *conflict* is a set of given cells in one row, column or box that show the same digit. No goal
state can keep a conflict, so at least one of its cells must still be corrected. Only cells that
can still change count: candidate cells not yet corrected (each cell is corrected at most once).

For one conflict, the cheapest way to break it costs at least
    m(conflict) = min over its changeable cells of that cell's cheapest alternative.
Different conflicts may share a cell, so adding the m's could count one change twice; the maximum
never overestimates:
    h(state) = max over conflicts of m(conflict),   0 if there are no conflicts,
               infinity if some conflict has no changeable cell (no goal below this state).

h is also consistent: correcting a cell of the max conflict costs at least m of that conflict, and
correcting any other cell leaves that conflict in place with no cheaper changeable cell.
"""

import math

import numpy as np

from sudoku import config

N = config.GRID
B = config.BOX


def units():
    """All 27 units (rows, columns, boxes) as lists of (row, col) cells."""
    rows = [[(r, c) for c in range(N)] for r in range(N)]
    cols = [[(r, c) for r in range(N)] for c in range(N)]
    boxes = [[(br + i, bc + j) for i in range(B) for j in range(B)]
             for br in range(0, N, B) for bc in range(0, N, B)]
    return rows + cols + boxes


UNITS = units()


def duplicate_conflicts(grid) -> list:
    """Every (unit, digit) that appears more than once among filled cells, as a frozenset of cells.

    A pair of cells in the same row and box appears twice (once per unit); both are kept, it does
    not matter for a max.
    """
    grid = np.asarray(grid)
    conflicts = []
    for unit in UNITS:
        by_digit = {}
        for cell in unit:
            if grid[cell]:
                by_digit.setdefault(int(grid[cell]), []).append(cell)
        conflicts.extend(frozenset(cells) for cells in by_digit.values() if len(cells) > 1)
    return conflicts


def conflict_heuristic(problem, state: frozenset) -> float:
    """h(state) for A* and Greedy (see module docstring)."""
    corrected = {cell for cell, _ in state}
    best = 0.0
    for conflict in duplicate_conflicts(problem.apply(state)):
        changeable = [cell for cell in conflict
                      if cell in problem.alternatives and cell not in corrected]
        if not changeable:
            return math.inf
        cheapest = min(problem.alternatives[cell][0].cost for cell in changeable)
        best = max(best, cheapest)
    return best
