"""UCS, Greedy Best-First and A* over correction sets - one best-first graph search, three priorities.

    UCS     priority = g          (path cost; optimal)
    Greedy  priority = h          (heuristic only; fast, not optimal)
    A*      priority = g + h      (optimal because h is admissible and consistent)

Graph search: the frontier is a heap of (priority, insertion order, state, g). A state can be pushed
more than once (the same set reached in a different order); when it is popped again after being
expanded it is skipped. The goal test is done when a state is POPPED (not when generated), which
is what makes UCS and A* return the cheapest goal.

Greedy and A* never push a child with h = infinity (a conflict no remaining correction can fix).

Counters:
- nodes_expanded:  states popped, goal-tested negatively, and whose children were generated;
- nodes_generated: children created by ``successors`` (including ones pruned by h = infinity);
- goal_tests / csp_calls / csp_timeouts: from the problem (CSP calls = goal tests that reached it);
- max_frontier:    largest heap size seen;
- seconds:         wall-clock time.
Stops with status "found", "exhausted" (no goal within the correction budget), "node_limit" or
"time_limit".
"""

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from sudoku import config
from sudoku.search.heuristic import conflict_heuristic
from sudoku.search.problem import CorrectionProblem, correction_log

FOUND, EXHAUSTED, NODE_LIMIT, TIME_LIMIT = "found", "exhausted", "node_limit", "time_limit"
ALGORITHMS = ("ucs", "greedy", "astar")


@dataclass
class SearchStats:
    """Search counters (see module docstring)."""

    nodes_expanded: int = 0
    nodes_generated: int = 0
    goal_tests: int = 0
    csp_calls: int = 0
    csp_timeouts: int = 0
    max_frontier: int = 0
    seconds: float = 0.0


@dataclass
class TraceRow:
    """One popped state, in the order the search tested it."""

    step: int
    state: frozenset
    g: float
    h: float
    is_goal: bool


@dataclass
class SearchResult:
    """Outcome of one correction search."""

    algorithm: str
    status: str
    state: frozenset | None = None
    corrections: list = field(default_factory=list)  # CorrectionRow list
    total_cost: float = math.nan
    solution: np.ndarray | None = None
    stats: SearchStats = field(default_factory=SearchStats)
    trace: list = field(default_factory=list)        # TraceRow list

    @property
    def found(self) -> bool:
        return self.status == FOUND


def priority(algorithm: str, g: float, h: float) -> float:
    """Frontier priority of a node for each algorithm."""
    if algorithm == "ucs":
        return g
    if algorithm == "greedy":
        return h
    if algorithm == "astar":
        return g + h
    raise ValueError(f"unknown algorithm {algorithm!r}; use one of {ALGORITHMS}")


def search(problem: CorrectionProblem, algorithm: str = config.DEFAULT_ALGORITHM,
           node_limit: int = config.SEARCH_NODE_LIMIT,
           time_limit: float = config.SEARCH_TIME_LIMIT) -> SearchResult:
    """Run UCS / Greedy / A* on ``problem``; the first goal state popped wins."""
    priority(algorithm, 0.0, 0.0)  # validates the name
    problem.reset_counters()
    use_h = algorithm != "ucs"
    heuristic = (lambda s: conflict_heuristic(problem, s)) if use_h else (lambda s: 0.0)
    stats, trace = SearchStats(), []
    start = time.perf_counter()
    order = itertools.count()  # FIFO tie-break -> deterministic

    root = problem.initial_state
    h0 = heuristic(root)
    frontier = [(priority(algorithm, 0.0, h0), next(order), root, 0.0, h0)]
    expanded = set()
    status, goal = EXHAUSTED, None
    while frontier:
        if stats.nodes_expanded >= node_limit:
            status = NODE_LIMIT
            break
        if time.perf_counter() - start > time_limit:
            status = TIME_LIMIT
            break
        _, _, state, g, h = heapq.heappop(frontier)
        if state in expanded:
            continue  # already reached more cheaply (or earlier, for Greedy)
        is_goal = problem.goal_test(state)
        trace.append(TraceRow(len(trace) + 1, state, g, h, is_goal))
        if is_goal:
            status, goal = FOUND, (state, g)
            break
        expanded.add(state)
        stats.nodes_expanded += 1
        for child, cost, _, _ in problem.successors(state):
            stats.nodes_generated += 1
            if child in expanded:
                continue
            h_child = heuristic(child)
            if math.isinf(h_child):
                continue  # this conflict can never be fixed below the child
            heapq.heappush(frontier, (priority(algorithm, g + cost, h_child), next(order),
                                      child, g + cost, h_child))
        stats.max_frontier = max(stats.max_frontier, len(frontier))

    stats.goal_tests = problem.goal_stats.goal_tests
    stats.csp_calls = problem.goal_stats.csp_calls
    stats.csp_timeouts = problem.goal_stats.csp_timeouts
    stats.seconds = time.perf_counter() - start
    result = SearchResult(algorithm, status, stats=stats, trace=trace)
    if goal:
        state, g = goal
        result.state, result.total_cost = state, g
        result.corrections = correction_log(problem, state)
        result.solution = problem.solutions[state]
    return result
