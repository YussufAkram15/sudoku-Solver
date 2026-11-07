"""Random-correction baseline for Experiment 3: guess corrections without using the network.

First the uncorrected read grid is goal-tested (as every search does). Then each trial:
1. pick k uniformly from 1..max_corrections;
2. pick k distinct GIVEN cells uniformly at random (any given, not only the low-confidence ones);
3. give each a uniformly random digit different from the read one;
4. goal-test the corrected grid. The first success wins; after ``trials`` failures it gives up.
Same goal test and correction budget as UCS / Greedy / A*, but no probabilities guide the choice.
"""

import time

import numpy as np

from sudoku import config
from sudoku.search.algorithms import EXHAUSTED, FOUND, SearchResult, SearchStats, TraceRow
from sudoku.search.problem import CorrectionProblem, correction_log


def random_state(problem: CorrectionProblem, givens: list, rng: np.random.Generator) -> frozenset:
    """One random correction set (steps 1-3 above)."""
    k = int(rng.integers(1, problem.max_corrections + 1))
    picked = rng.choice(len(givens), size=min(k, len(givens)), replace=False)
    state = set()
    for i in picked:
        cell = givens[i]
        others = [d for d in config.DIGITS if d != problem.read_grid[cell]]
        state.add((cell, int(rng.choice(others))))
    return frozenset(state)


def random_correction(problem: CorrectionProblem, rng: np.random.Generator,
                      trials: int = config.RANDOM_TRIALS) -> SearchResult:
    """Try the read grid, then up to ``trials`` random correction sets; first goal wins."""
    problem.reset_counters()
    start = time.perf_counter()
    givens = [(int(r), int(c)) for r, c in zip(*np.nonzero(problem.read_grid))]
    result = SearchResult("random", EXHAUSTED)
    states = [problem.initial_state]
    for step in range(trials + 1):
        state = states[-1]
        is_goal = problem.goal_test(state)
        result.trace.append(TraceRow(step + 1, state, problem.path_cost(state), 0.0, is_goal))
        if is_goal:
            result.status, result.state = FOUND, state
            result.corrections = correction_log(problem, state)
            result.total_cost = problem.path_cost(state)
            result.solution = problem.solutions[state]
            break
        states.append(random_state(problem, givens, rng))
    result.stats = SearchStats(nodes_expanded=0, nodes_generated=len(states) - 1,
                               goal_tests=problem.goal_stats.goal_tests,
                               csp_calls=problem.goal_stats.csp_calls,
                               csp_timeouts=problem.goal_stats.csp_timeouts,
                               seconds=time.perf_counter() - start)
    return result
