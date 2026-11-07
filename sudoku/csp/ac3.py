"""AC-3 arc consistency for the Sudoku CSP (used as preprocessing, and inside MAC).

For a "!=" constraint, arc (xi, xj) is inconsistent only when xj's domain is a single value y:
then y has no support and must be removed from xi's domain. ``revise`` does exactly that, so one
revision removes at most one value.
"""

from collections import deque

from sudoku.csp.sudoku_csp import ARCS, PEERS


def revise(domains: list, xi: int, xj: int):
    """Make arc (xi, xj) consistent. Returns the value removed from D(xi), or None.

    A value x in D(xi) has no support in D(xj) only if D(xj) == {x}.
    """
    if len(domains[xj]) == 1:
        (y,) = domains[xj]
        if y in domains[xi]:
            domains[xi].discard(y)
            return y
    return None


def ac3(domains: list, stats, queue=None, removed: list | None = None):
    """Make ``domains`` arc consistent (in place).

    queue:   arcs to start from (default: every arc of the constraint graph).
    stats:   object with ``ac3_revisions`` (revise calls) and ``ac3_pruned`` (values removed).
    removed: if given, every (variable, value) removed is appended, so a caller can undo it.

    Returns ``(True, None)`` if consistent, or ``(False, v)`` where v is the variable whose domain
    became empty (a contradiction, e.g. two equal givens in one unit).
    """
    queue = deque(ARCS if queue is None else queue)
    while queue:
        xi, xj = queue.popleft()
        stats.ac3_revisions += 1
        value = revise(domains, xi, xj)
        if value is None:
            continue
        stats.ac3_pruned += 1
        if removed is not None:
            removed.append((xi, value))
        if not domains[xi]:
            return False, xi
        # D(xi) shrank: arcs (xk, xi) pointing into xi must be checked again.
        queue.extend((xk, xi) for xk in PEERS[xi] if xk != xj)
    return True, None
