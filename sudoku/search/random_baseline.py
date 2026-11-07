"""Random-correction baseline for Experiment 3: guess corrections without using the network.

First the uncorrected read grid is goal-tested (as every search does). Then each trial:
1. pick k uniformly from 1..max_corrections;
2. pick k distinct cells uniformly at random: any GIVEN cell (default), or - with
   ``candidates_only=True`` - only among the search's candidate cells;
3. give each a uniformly random digit different from the read one (default), or one of that
   cell's top-3 alternatives (``candidates_only=True``);
4. goal-test the corrected grid. The first success wins; after ``trials`` failures it gives up.
Same goal test and correction budget as UCS / Greedy / A*, but no costs order the guesses.
The ``candidates_only`` variant uses the same choices the search can make, so comparing it with
A* isolates the value of ordering by NN cost from the value of the candidate cut-off.
"""

import time

import numpy as np

from sudoku import config
from sudoku.search.algorithms import EXHAUSTED, FOUND, SearchResult, SearchStats, TraceRow
from sudoku.search.problem import CorrectionProblem, correction_log


def random_state(problem: CorrectionProblem, cells: list, rng: np.random.Generator,
                 candidates_only: bool = False) -> frozenset:
    """One random correction set over ``cells`` (steps 1-3 above)."""
    k = int(rng.integers(1, problem.max_corrections + 1))
    picked = rng.choice(len(cells), size=min(k, len(cells)), replace=False)
    state = set()
    for i in picked:
        cell = cells[i]
        if candidates_only:
            options = [alt.digit for alt in problem.alternatives[cell]]
        else:
            options = [d for d in config.DIGITS if d != problem.read_grid[cell]]
        state.add((cell, int(rng.choice(options))))
    return frozenset(state)


def random_correction(problem: CorrectionProblem, rng: np.random.Generator,
                      trials: int = config.RANDOM_TRIALS,
                      candidates_only: bool = False) -> SearchResult:
    """Try the read grid, then up to ``trials`` random correction sets; first goal wins."""
    problem.reset_counters()
    start = time.perf_counter()
    if candidates_only:
        cells = list(problem.candidates)
    else:
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(problem.read_grid))]
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
        if step < trials:  # draw the next random set only if it will be tested
            states.append(random_state(problem, cells, rng, candidates_only))
    result.stats = SearchStats(nodes_expanded=0, nodes_generated=len(states) - 1,
                               goal_tests=problem.goal_stats.goal_tests,
                               csp_calls=problem.goal_stats.csp_calls,
                               csp_timeouts=problem.goal_stats.csp_timeouts,
                               seconds=time.perf_counter() - start)
    return result
