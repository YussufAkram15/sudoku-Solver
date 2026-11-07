"""All experiments of the brief -> CSV tables + PNG plots in results/experiments/.

    python run_experiments.py           5 seeds x 20 puzzles (100 puzzles per misread level)
    python run_experiments.py --quick   2 seeds x 3 puzzles, written to results/experiments/quick/

Experiment 1: UCS vs Greedy vs A* (nodes, recovery) for misread levels 0-4.
Experiment 2: BT vs BT+FC vs BT+FC+AC-3 (+MAC) on 25 / 30 / 35-given puzzles.
Experiment 3: NN-guided vs no correction vs random correction (+ NN without temperature).
Recovery:     exact-solution recovery on 100 puzzles per misread level.
Every table reports mean and std over seeds (see sudoku/eval/experiments.py).
Needs `python make_data.py` and `python train.py` first.
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sudoku import config  # noqa: E402
from sudoku.data import bank as bank_io, injection, mnist  # noqa: E402
from sudoku.eval import experiments as ex  # noqa: E402
from sudoku.eval.plots import ALGO_COLORS, grouped_bars, save  # noqa: E402
from sudoku.nn.reader import DigitReader  # noqa: E402
from sudoku.search import ALGORITHMS  # noqa: E402

LEVELS = list(config.MISREAD_LEVELS)
# cost / n_corrections are NaN for unsolved puzzles, so their means cover solved puzzles only
EXP1_METRICS = ["nodes_expanded", "nodes_generated", "nodes_pruned", "goal_tests", "csp_calls",
                "seconds", "cost", "n_corrections", "exact"]
EXP2_METRICS = ["assignments", "backtracks", "ac3_revisions", "values_pruned", "seconds",
                "timeout", "correct"]


def series(summary: pd.DataFrame, key: str, names, metric: str, groups, group_col: str) -> dict:
    """{name: (means, stds)} in ``groups`` order, from a mean_std_over_seeds table."""
    out = {}
    for name in names:
        rows = summary[summary[key] == name].set_index(group_col).reindex(groups)
        out[name] = (rows[f"{metric}_mean"].to_numpy(), rows[f"{metric}_std"].fillna(0).to_numpy())
    return out


def plot_exp1(summary, out) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, metric, title in zip(axes, ["nodes_expanded", "nodes_generated", "exact"],
                                 ["Nodes expanded", "Nodes generated", "Exact recovery rate"]):
        grouped_bars(ax, LEVELS, series(summary, "algorithm", ALGORITHMS, metric, LEVELS, "level"),
                     f"Exp. 1: {title}", title.lower(), log=metric != "exact",
                     colors=ALGO_COLORS)
        ax.set_xlabel("injected misreads")
    axes[0].legend(fontsize=8)
    axes[2].set_ylim(0, 1.1)
    save(fig, out / "exp1_search.png")


def plot_exp2(summary, out) -> None:
    modes = list(ex.CSP_MODES)
    givens = list(config.GIVEN_TARGETS)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, metric, title, log in zip(axes, ["assignments", "backtracks", "seconds", "timeout"],
                                      ["Assignments", "Backtracks", "Time (s)", "Timeout rate"],
                                      [True, True, True, False]):
        grouped_bars(ax, givens, series(summary, "mode", modes, metric, givens, "givens"),
                     f"Exp. 2: {title} (static order)", title.lower(), log=log)
        ax.set_xlabel("givens")
    axes[0].legend(fontsize=7)
    save(fig, out / "exp2_csp.png")


def plot_exp3(summary, out) -> None:
    labels = {"nn_guided": f"NN-guided ({config.DEFAULT_ALGORITHM.upper()})",
              "no_correction": "no correction (NN off)", "random": "random correction",
              "random_in_candidates (ext.)": "random within candidates (ext.)",
              "nn_guided_no_T (ext.)": "NN-guided, no temperature (ext.)"}
    colors = dict(zip(labels.values(),
                      ["tab:green", "tab:red", "tab:gray", "silver", "tab:olive"]))
    data = series(summary, "arm", ex.EXP3_ARMS, "exact", LEVELS, "level")
    fig, ax = plt.subplots(figsize=(10, 4))
    grouped_bars(ax, LEVELS, {labels[arm]: v for arm, v in data.items()},
                 "Exp. 3: exact recovery - NN-guided vs no correction vs random", "exact recovery rate",
                 colors=colors, label_values=True)
    ax.set_xlabel("injected misreads")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    save(fig, out / "exp3_recovery.png")


def plot_recovery(table, out) -> None:
    parts = [("exact_as_read", "exact, no correction needed", "tab:blue"),
             ("exact_after_correction", "exact after correction", "tab:green"),
             ("wrong_solution", "wrong solution (unique but not the original)", "tab:orange"),
             ("unreadable", "unreadable", "tab:red")]
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(table))
    for col, label, colour in parts:
        ax.bar(table.level, table[col], bottom=bottom, label=label, color=colour)
        bottom += table[col].to_numpy()
    for x, n, rate in zip(table.level, table.puzzles, table.exact_rate):
        ax.text(x, n, f"{rate:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, table.puzzles.max() * 1.12)  # room for the % labels
    ax.set_xlabel("injected misreads")
    ax.set_ylabel("puzzles")
    ax.set_title(f"Exact-solution recovery, {int(table.puzzles.iloc[0])} puzzles per level "
                 f"(NN-guided, {config.DEFAULT_ALGORITHM.upper()})", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, out / "recovery.png")


def plot_all(s1, s2, s3, rec, out) -> None:
    """All four experiment figures from the summary tables."""
    plot_exp1(s1, out)
    plot_exp2(s2, out)
    plot_exp3(s3, out)
    plot_recovery(rec, out)


def fmt(summary: pd.DataFrame, metrics) -> pd.DataFrame:
    """'mean ± std' strings for the console."""
    shown = summary.drop(columns=[c for c in summary if c.endswith(("_mean", "_std"))])
    for m in metrics:
        shown[m] = [f"{a:.3g} ± {b:.2g}" for a, b in zip(summary[f"{m}_mean"],
                                                         summary[f"{m}_std"].fillna(0))]
    return shown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    parser.add_argument("--replot", action="store_true",
                        help="recompute summaries and plots from the saved per-puzzle CSVs")
    args = parser.parse_args()
    n_seeds = config.QUICK_SEEDS if args.quick else config.N_SEEDS
    per_seed = config.QUICK_PUZZLES if args.quick else config.PUZZLES_PER_SEED
    out = config.EXPERIMENTS_DIR / "quick" if args.quick else config.EXPERIMENTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if args.replot:
        summarize(*(pd.read_csv(out / f"{n}_per_puzzle.csv") for n in ("exp1", "exp2", "exp3")),
                  out)
        return

    bank_file = config.PUZZLES_FILE
    if not bank_file.exists() and args.quick:
        bank_file = config.PUZZLES_QUICK_FILE
    bank = bank_io.load_bank(bank_file)
    if len(bank[config.DEFAULT_GIVENS][0]) < n_seeds * per_seed:
        raise SystemExit(f"{bank_file.name} has too few puzzles - run `python make_data.py`")
    reader = DigitReader.load()
    reader_no_t = DigitReader(reader.model, temperature=1.0)
    test = mnist.load_splits()["test"]
    pools = injection.load_pools()
    print(f"{n_seeds} seeds x {per_seed} puzzles per misread level, output {out}")
    start = time.perf_counter()

    print("[1/2] Experiments 1 + 3 and recovery study (30 givens, misread levels 0-4)")
    puzzles, solutions = bank[config.DEFAULT_GIVENS]
    exp1, exp3 = ex.run_reading_experiments(reader, reader_no_t, test.images, pools, puzzles,
                                            solutions, n_seeds, per_seed)
    print("[2/2] Experiment 2 (CSP modes)")
    exp2 = ex.run_csp_experiment(bank, n_seeds, per_seed)

    for name, df in [("exp1", exp1), ("exp2", exp2), ("exp3", exp3)]:
        df.to_csv(out / f"{name}_per_puzzle.csv", index=False)
    summarize(exp1, exp2, exp3, out)
    print(f"\ndone in {time.perf_counter() - start:.0f}s; tables and plots in {out}")


def summarize(exp1, exp2, exp3, out) -> None:
    """Per-puzzle rows -> summary CSVs (mean ± std over seeds), plots and console tables."""
    rec = ex.recovery_table(exp1.assign(exact=exp1.exact.astype(bool)))
    exp1 = exp1.assign(exact=exp1.exact.astype(float))
    exp3 = exp3.assign(exact=exp3.exact.astype(float))
    exp2 = exp2.assign(timeout=exp2.timeout.astype(float), correct=exp2.correct.astype(float))
    s1 = ex.mean_std_over_seeds(exp1, ["level", "algorithm"], EXP1_METRICS)
    s2 = ex.mean_std_over_seeds(exp2, ["givens", "mode"], EXP2_METRICS)
    s3 = ex.mean_std_over_seeds(exp3, ["level", "arm"], ["exact"])
    for name, df in [("exp1_summary", s1), ("exp2_summary", s2), ("exp3_summary", s3),
                     ("recovery", rec)]:
        df.to_csv(out / f"{name}.csv", index=False)
    plot_all(s1, s2, s3, rec, out)

    print("\nExperiment 1 (mean ± std over seeds)")
    print(fmt(s1, ["nodes_expanded", "nodes_generated", "nodes_pruned", "seconds", "exact"])
          .to_string(index=False))
    print("\nExperiment 2 (mean ± std over seeds; timed-out runs count with the capped work)")
    print(fmt(s2, ["assignments", "backtracks", "seconds", "timeout"]).to_string(index=False))
    print("\nExperiment 3 (exact recovery, mean ± std over seeds)")
    print(fmt(s3, ["exact"]).to_string(index=False))
    print("\nExact-solution recovery (NN-guided)")
    print(rec.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
