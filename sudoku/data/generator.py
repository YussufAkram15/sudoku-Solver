"""Seeded Sudoku puzzle generator.

1. Build a full valid grid by randomised backtracking (cells in row-major order, digits tried in a
   random order).
2. Visit the 81 cells in a random order and blank each one; keep the removal only if the puzzle
   still has exactly one solution, otherwise put the digit back.
3. Stop as soon as the target number of givens is reached. If a full pass ends above the target,
   start again from a new full grid (at most ``config.GENERATOR_MAX_ATTEMPTS`` grids).

Grids are (9, 9) int arrays, 0 = empty cell.

``count_solutions`` here is a small, fast bitmask solver used only to check uniqueness while
generating. The instrumented CSP solver with BT / FC / AC-3 switches lives in ``sudoku.csp``.
"""

from dataclasses import dataclass

import numpy as np

from sudoku import config

N = config.GRID
B = config.BOX
ALL_DIGITS_MASK = sum(1 << d for d in config.DIGITS)  # bit d set = digit d available


def box_index(r: int, c: int) -> int:
    """Index 0..8 of the 3x3 box that contains cell (r, c)."""
    return (r // B) * B + c // B


def can_place(grid: np.ndarray, r: int, c: int, d: int) -> bool:
    """True if digit d does not already appear in row r, column c or the box of (r, c)."""
    br, bc = (r // B) * B, (c // B) * B
    return (
        d not in grid[r, :]
        and d not in grid[:, c]
        and d not in grid[br : br + B, bc : bc + B]
    )


def is_valid_solution(grid: np.ndarray) -> bool:
    """True if ``grid`` is completely filled and every row, column and box holds 1..9 once."""
    grid = np.asarray(grid)
    target = set(config.DIGITS)
    for i in range(N):
        br, bc = (i // B) * B, (i % B) * B
        units = (grid[i, :], grid[:, i], grid[br : br + B, bc : bc + B].ravel())
        if any(set(unit.tolist()) != target for unit in units):
            return False
    return True


def generate_full_grid(rng: np.random.Generator) -> np.ndarray:
    """Return a random complete Sudoku grid using randomised backtracking."""
    grid = np.zeros((N, N), dtype=np.int64)

    def fill(pos: int) -> bool:
        if pos == N * N:
            return True
        r, c = divmod(pos, N)
        for d in rng.permutation(config.DIGITS):
            if can_place(grid, r, c, int(d)):
                grid[r, c] = d
                if fill(pos + 1):
                    return True
                grid[r, c] = 0
        return False

    fill(0)
    return grid


def count_solutions(grid: np.ndarray, limit: int = 2) -> int:
    """Count solutions of ``grid`` (0 = empty), stopping once ``limit`` are found.

    Uses bitmasks per row/column/box and always branches on the empty cell with the fewest
    candidates. Returns 0 if the givens already clash.
    """
    rows, cols, boxes = [0] * N, [0] * N, [0] * N
    empties = []
    for r in range(N):
        for c in range(N):
            d = int(grid[r][c])
            if d == 0:
                empties.append((r, c))
                continue
            bit, b = 1 << d, box_index(r, c)
            if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                return 0  # two equal givens in one unit
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

    found = 0

    def search(remaining: list) -> bool:
        """Depth-first search; returns True once ``limit`` solutions were found."""
        nonlocal found
        if not remaining:
            found += 1
            return found >= limit
        best_i, best_mask, best_n = -1, 0, N + 1
        for i, (r, c) in enumerate(remaining):  # most constrained cell first
            mask = ALL_DIGITS_MASK & ~(rows[r] | cols[c] | boxes[box_index(r, c)])
            n = mask.bit_count()
            if n == 0:
                return False
            if n < best_n:
                best_i, best_mask, best_n = i, mask, n
        r, c = remaining[best_i]
        rest = remaining[:best_i] + remaining[best_i + 1 :]
        b = box_index(r, c)
        for d in config.DIGITS:
            bit = 1 << d
            if not best_mask & bit:
                continue
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            done = search(rest)
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            if done:
                return True
        return False

    search(empties)
    return found


@dataclass
class Puzzle:
    """A generated puzzle and its unique solution."""

    puzzle: np.ndarray    # (9, 9) givens, 0 = empty
    solution: np.ndarray  # (9, 9) complete grid the puzzle was cut from
    attempts: int         # full grids tried before the target was reached

    @property
    def n_givens(self) -> int:
        return int((self.puzzle != 0).sum())


def remove_cells(solution: np.ndarray, n_givens: int, rng: np.random.Generator) -> np.ndarray:
    """Blank cells in random order, keeping each removal only if the solution stays unique.

    Stops when ``n_givens`` cells remain. May end with more givens if no further cell can be
    removed without breaking uniqueness.
    """
    puzzle = solution.copy()
    givens = N * N
    for cell in rng.permutation(N * N):
        if givens == n_givens:
            break
        r, c = divmod(int(cell), N)
        digit = puzzle[r, c]
        puzzle[r, c] = 0
        if count_solutions(puzzle, limit=2) == 1:
            givens -= 1
        else:
            puzzle[r, c] = digit  # removal would allow a second solution: undo
    return puzzle


def generate_puzzle(
    n_givens: int,
    rng: np.random.Generator,
    max_attempts: int = config.GENERATOR_MAX_ATTEMPTS,
) -> Puzzle:
    """Generate a puzzle with exactly ``n_givens`` givens and a unique solution."""
    if not config.MIN_GIVENS <= n_givens <= N * N:
        raise ValueError(f"n_givens must be in {config.MIN_GIVENS}..{N * N}, got {n_givens}")
    for attempt in range(1, max_attempts + 1):
        solution = generate_full_grid(rng)
        puzzle = remove_cells(solution, n_givens, rng)
        if (puzzle != 0).sum() == n_givens:
            return Puzzle(puzzle=puzzle, solution=solution, attempts=attempt)
    raise RuntimeError(f"no puzzle with {n_givens} givens after {max_attempts} grids")


def puzzle_rng(seed: int, n_givens: int, index: int) -> np.random.Generator:
    """Independent generator for puzzle number ``index`` with ``n_givens`` givens.

    Every puzzle in the bank can be regenerated on its own from (seed, n_givens, index).
    """
    return np.random.default_rng([seed, n_givens, index])


def generate_bank(n_givens: int, n_puzzles: int, seed: int = config.SEED) -> list:
    """Generate ``n_puzzles`` puzzles with ``n_givens`` givens, puzzle i seeded by (seed, n_givens, i)."""
    return [generate_puzzle(n_givens, puzzle_rng(seed, n_givens, i)) for i in range(n_puzzles)]
