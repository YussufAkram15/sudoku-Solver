"""Sudoku CSP: backtracking with switchable forward checking, AC-3 and MAC, plus counters."""

from sudoku.csp.solver import CountResult, Result, Stats, count_solutions, solve

__all__ = ["solve", "count_solutions", "Result", "CountResult", "Stats"]
