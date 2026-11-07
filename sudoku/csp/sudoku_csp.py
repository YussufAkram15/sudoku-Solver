"""Sudoku as a CSP: 81 variables, domains {1..9}, binary "!=" constraints between peers.

Variable v = cell index ``r * 9 + c`` (row-major). Two cells are *peers* if they share a row,
column or 3x3 box; every peer pair is one binary inequality constraint, i.e. the all-different
constraint of each unit written as binary constraints (what AC-3 needs).
"""

import numpy as np

from sudoku import config

N = config.GRID
B = config.BOX
N_VARS = N * N
DIGITS = frozenset(config.DIGITS)


def cell(v: int) -> tuple:
    """Variable index -> (row, col)."""
    return divmod(v, N)


def var(r: int, c: int) -> int:
    """(row, col) -> variable index."""
    return r * N + c


def _units_of(r: int, c: int) -> list:
    """The row, column and box of cell (r, c), each as a list of variables."""
    br, bc = (r // B) * B, (c // B) * B
    row = [var(r, j) for j in range(N)]
    col = [var(i, c) for i in range(N)]
    box = [var(br + i, bc + j) for i in range(B) for j in range(B)]
    return [row, col, box]


# PEERS[v] = the 20 other cells that must differ from v (computed once at import).
PEERS = tuple(
    tuple(sorted({u for unit in _units_of(*cell(v)) for u in unit} - {v})) for v in range(N_VARS)
)
# Every directed arc (xi, xj) of the binary constraint graph: 81 * 20 = 1620 arcs.
ARCS = tuple((xi, xj) for xi in range(N_VARS) for xj in PEERS[xi])


def initial_domains(grid) -> list:
    """Domain per variable: {given} for givens, {1..9} for empty cells. Returns a list of sets."""
    flat = np.asarray(grid).ravel()
    return [{int(d)} if d else set(DIGITS) for d in flat]


def given_conflict(grid):
    """First pair of equal givens that are peers, as ``(v1, v2, digit)``, or None if givens agree."""
    flat = np.asarray(grid).ravel()
    for v in range(N_VARS):
        if flat[v]:
            for u in PEERS[v]:
                if u > v and flat[u] == flat[v]:
                    return v, u, int(flat[v])
    return None
