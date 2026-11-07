"""Integrated demo: one handwritten Sudoku from image to solution, with plots of how NN, search and
CSP cooperate.

    python run_demo.py --seed 7 --misreads 2 [--givens 30] [--algorithm astar]

Console: input puzzle, read grid with confidences, AC-3 conflict, search trace (popped nodes in order
with g / h / f), correction log, solved grid, CSP pruning table, NN on/off table.
Figures (results/demo/):
    demo_pipeline.png   input mosaic | read grid coloured by confidence | solved grid
    demo_search.png     g and h of popped nodes + UCS / Greedy / A* node counts on this puzzle
    demo_csp.png        CSP work with BT / BT+FC / BT+FC+AC-3 (+ MAC) on the solved puzzle
    demo_nn_on_off.png  exact recovery with NN-guided correction on vs off, misread levels 0-4

Needs `python make_data.py` and `python train.py` first.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sudoku import config  # noqa: E402
from sudoku.csp import solve  # noqa: E402
from sudoku.data import generator, injection, mnist  # noqa: E402
from sudoku.eval.plots import ALGO_COLORS, draw_grid, plot_bars, save  # noqa: E402
from sudoku.integration import format_result, solve_reading  # noqa: E402
from sudoku.integration.pipeline import format_ac3  # noqa: E402
from sudoku.nn.reader import DigitReader  # noqa: E402
from sudoku.search import ALGORITHMS, describe  # noqa: E402

CSP_MODES = {
    "BT": dict(use_fc=False, use_ac3=False),
    "BT+FC": dict(use_fc=True, use_ac3=False),
    "BT+FC+AC-3": dict(use_fc=True, use_ac3=True),
    "+MAC (ext.)": dict(use_fc=True, use_ac3=True, use_mac=True),
}


# ---------------------------------------------------------------- building the input
def make_instance(seed: int, givens: int, misreads: int, pools, test_images):
    """Seeded puzzle + mosaic with ``misreads`` injected hard digits."""
    rng = config.make_rng(seed)
    puzzle = generator.generate_puzzle(givens, rng)
    mosaic, _, injected = injection.make_level_mosaic(puzzle.puzzle, misreads, pools, test_images, rng)
    return puzzle, mosaic, injected


# ---------------------------------------------------------------- console text
def grid_text(grid, conf=None, marks=()) -> str:
    """Grid as text; with ``conf`` every given shows its confidence in %, ``*`` marks cells."""
    lines = []
    for r in range(config.GRID):
        if r and r % config.BOX == 0:
            lines.append("-" * (len(lines[-1])))
        cells = []
        for c in range(config.GRID):
            d = grid[r][c]
            text = str(d) if d else "."
            if conf is not None:  # 5 characters per cell: digit, mark, confidence
                text = f"{d}{'*' if (r, c) in marks else ' '}{conf[r][c] * 100:3.0f}" if d \
                    else "  .  "
            cells.append(text)
            if c % config.BOX == config.BOX - 1 and c < config.GRID - 1:
                cells.append("|")
        lines.append(" ".join(cells))
    return "\n".join(lines)


PRIORITY_NAME = {"ucs": "g", "greedy": "h", "astar": "f = g + h"}


def print_trace(search_result, rows: int = config.DEMO_TRACE_ROWS) -> None:
    """Popped nodes in order with g and h (and f = g + h for A*)."""
    astar = search_result.algorithm == "astar"
    print(f"  frontier ordered by {PRIORITY_NAME[search_result.algorithm]}")
    print(f"  {'step':>4}  {'g':>7}  {'h':>7}  {'f' if astar else '':>7}  goal  state")
    for t in search_result.trace[:rows]:
        f = f"{t.g + t.h:7.3f}" if astar else " " * 7
        print(f"  {t.step:4d}  {t.g:7.3f}  {t.h:7.3f}  {f}  {'YES' if t.is_goal else ' no'}"
              f"   {describe(t.state)}")
    if len(search_result.trace) > rows:
        print(f"  ... {len(search_result.trace) - rows} more popped nodes")


# ---------------------------------------------------------------- the three comparisons
def compare_algorithms(read) -> dict:
    """Run UCS, Greedy and A* on the same reading."""
    return {a: solve_reading(read, algorithm=a) for a in ALGORITHMS}


def compare_csp_modes(grid) -> dict:
    """Solve ``grid`` with each CSP mode (static ordering, as in Experiment 2)."""
    return {name: solve(grid, ordering=config.ORDERING, **kw) for name, kw in CSP_MODES.items()}


def nn_on_off_batch(reader, pools, test_images, givens: int, seed: int) -> dict:
    """Exact recovery rate per misread level with NN-guided correction on and off."""
    rates = {"on": [], "off": []}
    for k in config.MISREAD_LEVELS:
        hits = {"on": 0, "off": 0}
        for i in range(config.DEMO_BATCH_PUZZLES):
            rng = np.random.default_rng([seed, k, i])
            p = generator.generate_puzzle(givens, rng)
            mosaic, _, _ = injection.make_level_mosaic(p.puzzle, k, pools, test_images, rng)
            read = reader.read_mosaic(mosaic)
            hits["on"] += solve_reading(read, nn_correction=True).exact_recovery(p.solution)
            hits["off"] += solve_reading(read, nn_correction=False).exact_recovery(p.solution)
        for mode in rates:
            rates[mode].append(hits[mode] / config.DEMO_BATCH_PUZZLES)
    return rates


# ---------------------------------------------------------------- figures
def figure_pipeline(mosaic, result, puzzle, injected, path) -> None:
    """Input mosaic | read grid (confidence shading, conflicts red, injected blue) | solution."""
    read = result.read
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mosaic, cmap="gray")
    axes[0].set_title(f"1. Input mosaic ({len(injected)} hard digits injected)", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    marks = {cell: "red" for conflict in result.ac3_check.duplicates for cell in conflict}
    injected_marks = {cell: "tab:blue" for cell in injected}
    conf = read.confidence()
    small = {(r, c): f"{conf[r, c] * 100:.0f}%" for r, c in zip(*np.nonzero(read.grid))}
    unsure = np.clip((1 - conf) * 2, 0, 1)  # 100% -> white, 50% or less -> darkest
    draw_grid(axes[1], read.grid, "2. CNN reading (purple = unsure, % = confidence)\n"
              "red = AC-3 conflict, blue dashed = injected hard digit", shade=unsure, marks=marks,
              inner_marks=injected_marks, small=small, cmap="Purples")
    if result.solution is not None:
        fixed = {row.cell: "tab:orange" for row in result.corrections}
        ok = "exact" if result.exact_recovery(puzzle.solution) else "NOT the original solution"
        draw_grid(axes[2], result.solution, f"3. Solution ({result.status}, {ok})\n"
                  "orange = corrected by search", marks=fixed)
    else:
        unsure = {u.cell: "tab:orange" for u in result.least_confident}
        second = {u.cell: f"or {u.top3[1][0]}? {u.top3[1][1]:.0%}" for u in result.least_confident}
        draw_grid(axes[2], read.grid, f"3. {result.message}\norange = {len(unsure)} least confident "
                  "cells with 2nd choice (top-3 in console)", marks=unsure, small=second)
    save(fig, path)


def figure_search(results: dict, chosen: str, path) -> None:
    """g/h of popped nodes for the chosen algorithm + node counts for all three."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    trace = results[chosen].search.trace if results[chosen].search else []
    steps = [t.step for t in trace]
    axes[0].plot(steps, [t.g for t in trace], "o-", ms=3, label="g (cost so far)")
    axes[0].plot(steps, [t.h for t in trace], "s-", ms=3, label="h (conflict lower bound)")
    if chosen == "astar":
        axes[0].plot(steps, [t.g + t.h for t in trace], "--", label="f = g + h (A* priority)")
    axes[0].set_xlabel("pop order")
    axes[0].set_ylabel("cost")
    axes[0].set_title(f"{chosen.upper()} search trace ({len(trace)} nodes popped, "
                      f"ordered by {PRIORITY_NAME[chosen]})", fontsize=10)
    axes[0].legend(fontsize=8)
    names = list(results)
    expanded = [results[a].search.stats.nodes_expanded if results[a].search else 0 for a in names]
    generated = [results[a].search.stats.nodes_generated if results[a].search else 0 for a in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, expanded, 0.4, label="expanded", color=[ALGO_COLORS[a] for a in names])
    axes[1].bar(x + 0.2, generated, 0.4, label="generated", color="lightgray", edgecolor="black")
    axes[1].set_xticks(x, labels=[a.upper() for a in names])
    for xi, e, g in zip(x, expanded, generated):
        axes[1].text(xi - 0.2, e, str(e), ha="center", va="bottom", fontsize=8)
        axes[1].text(xi + 0.2, g, str(g), ha="center", va="bottom", fontsize=8)
    axes[1].set_yscale("symlog")
    axes[1].set_title("Nodes on this puzzle (coloured: expanded, grey: generated)", fontsize=10)
    save(fig, path)


def figure_csp(csp: dict, path) -> None:
    """Assignments, backtracks and pruned values per CSP mode (log scale)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = list(csp)
    for ax, (key, label) in zip(axes, [("assignments", "assignments"), ("backtracks", "backtracks"),
                                       ("values_pruned", "domain values pruned")]):
        values = [getattr(csp[n].stats, key) for n in names]
        plot_bars(ax, names, values, f"CSP {label} (static order)", label, log=True)
    save(fig, path)


def figure_nn_on_off(rates: dict, path) -> None:
    """Exact recovery per misread level with correction on vs off."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(config.MISREAD_LEVELS))
    ax.bar(x - 0.2, rates["on"], 0.4, label="NN-guided correction ON", color="tab:green")
    ax.bar(x + 0.2, rates["off"], 0.4, label="correction OFF (raw reading)", color="tab:red")
    for xi, on, off in zip(x, rates["on"], rates["off"]):
        ax.text(xi - 0.2, on, f"{on:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, off, f"{off:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels=[str(k) for k in config.MISREAD_LEVELS])
    ax.set_xlabel("injected misreads")
    ax.set_ylabel("exact recovery rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"System with NN correction on vs off ({config.DEMO_BATCH_PUZZLES} puzzles/level, "
                 "illustration;\nfull study: Experiment 3)", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, path)


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--misreads", type=int, default=config.DEMO_MISREADS,
                        choices=config.MISREAD_LEVELS)
    parser.add_argument("--givens", type=int, default=config.DEFAULT_GIVENS,
                        choices=config.GIVEN_TARGETS)
    parser.add_argument("--algorithm", default=config.DEFAULT_ALGORITHM, choices=ALGORITHMS)
    args = parser.parse_args()
    out = config.DEMO_DIR

    reader = DigitReader.load()
    test = mnist.load_splits()["test"]
    pools = injection.load_pools()
    puzzle, mosaic, injected = make_instance(args.seed, args.givens, args.misreads, pools, test.images)

    print(f"=== Input: {args.givens}-given puzzle, seed {args.seed}, {args.misreads} hard digits "
          f"injected at {injected}")
    print(grid_text(puzzle.puzzle))

    results = compare_algorithms(reader.read_mosaic(mosaic))
    result = results[args.algorithm]
    read = result.read
    misread = [tuple(map(int, rc)) for rc in np.argwhere((read.grid != puzzle.puzzle) & (puzzle.puzzle > 0))]
    print(f"\n=== 1. NN reading (digit, * = misread, confidence %); temperature T = {reader.temperature:.3f}")
    print(grid_text(read.grid, read.confidence(), set(misread)))
    print(f"misread cells: {misread}")

    print("\n=== 2. AC-3 check on the read grid")
    print("\n".join(format_ac3(result.ac3_check, read.grid)))

    print(f"\n=== 3. Correction search ({args.algorithm.upper()}), popped nodes in order")
    if result.search:
        print_trace(result.search)
        s = result.search.stats
        print(f"  expanded {s.nodes_expanded}, generated {s.nodes_generated}, pruned (h=inf) "
              f"{s.nodes_pruned}, goal tests {s.goal_tests}, CSP calls {s.csp_calls}, "
              f"max frontier {s.max_frontier}, {s.seconds:.3f}s")
    print("\n  algorithm  status      expanded  generated  cost")
    for a, r in results.items():
        st = r.search.stats if r.search else None
        print(f"  {a:9s}  {r.status:10s}  {st.nodes_expanded if st else 0:8d}  "
              f"{st.nodes_generated if st else 0:9d}  "
              f"{f'{r.total_cost:.3f}' if r.solved else '-'}")

    print("\n=== 4. Result")
    print(format_result(result))
    if result.solution is not None:
        print(grid_text(result.solution))
        print("exact recovery of the original solution:", result.exact_recovery(puzzle.solution))

    corrected = read.grid.copy()
    for row in result.corrections:
        corrected[row.cell] = row.new_digit
    csp_grid = corrected if result.solved else puzzle.puzzle
    csp = compare_csp_modes(csp_grid)
    which = "corrected puzzle" if result.solved else "original (ground-truth) puzzle"
    print(f"\n=== 5. CSP pruning on the {which} (static order)")
    print(f"  {'mode':12s} {'status':9s} {'assign':>8s} {'backtr':>8s} {'AC3 rev':>8s} {'pruned':>8s}")
    for name, r in csp.items():
        st = r.stats
        print(f"  {name:12s} {r.status:9s} {st.assignments:8d} {st.backtracks:8d} "
              f"{st.ac3_revisions:8d} {st.values_pruned:8d}")

    print(f"\n=== 6. NN-guided correction on vs off ({config.DEMO_BATCH_PUZZLES} puzzles per level)")
    rates = nn_on_off_batch(reader, pools, test.images, args.givens, args.seed)
    for k, on, off in zip(config.MISREAD_LEVELS, rates["on"], rates["off"]):
        print(f"  {k} misreads: exact recovery ON {on:.0%}   OFF {off:.0%}")

    figure_pipeline(mosaic, result, puzzle, injected, out / "demo_pipeline.png")
    figure_search(results, args.algorithm, out / "demo_search.png")
    figure_csp(csp, out / "demo_csp.png")
    figure_nn_on_off(rates, out / "demo_nn_on_off.png")
    shown = out.relative_to(config.ROOT) if out.is_relative_to(config.ROOT) else out
    print(f"\nfigures saved to {shown}/")


if __name__ == "__main__":
    main()
