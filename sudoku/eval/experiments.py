"""The three experiments of the brief plus the 100-puzzle exact-recovery study.

Seeds and puzzles. "Seed" s = 0..N_SEEDS-1 is one independent run: it uses bank puzzles
s*PUZZLES_PER_SEED .. (s+1)*PUZZLES_PER_SEED - 1 and seeds every mosaic with (SEED, s, k, i).
Statistics: first the mean (or rate) over the puzzles of one seed, then mean ± std over seeds.

Experiment 1  UCS vs Greedy vs A*: nodes expanded / generated / pruned, goal tests, time and exact
              recovery for misread levels 0-4 (30-given puzzles).
Experiment 2  BT vs BT+FC vs BT+FC+AC-3 (+ MAC, extension) with static variable order on the
              25 / 30 / 35-given bank puzzles: assignments, backtracks, time, timeouts.
Experiment 3  NN-guided correction (A*) vs no correction (NN off) vs random correction, plus the
              extension arm "NN-guided without temperature scaling": exact recovery per level.
Recovery      NN-guided (A*) exact-solution recovery on 100 puzzles per misread level, with a
              breakdown that adds up to the number of puzzles: exact without corrections /
              exact after corrections / wrong solution (a grid was produced but it is not the
              original) / unreadable.

Every reading is computed once and shared by Experiments 1 and 3 and the recovery study.
"""

import time

import numpy as np
import pandas as pd

from sudoku import config
from sudoku.csp import solve
from sudoku.data import injection
from sudoku.integration import solve_reading
from sudoku.search import ALGORITHMS, CorrectionProblem, random_correction

CSP_MODES = {
    "BT": dict(use_fc=False, use_ac3=False),
    "BT+FC": dict(use_fc=True, use_ac3=False),
    "BT+FC+AC-3": dict(use_fc=True, use_ac3=True),
    "BT+FC+AC-3+MAC (ext.)": dict(use_fc=True, use_ac3=True, use_mac=True),
}
EXP3_ARMS = ("nn_guided", "no_correction", "random", "nn_guided_no_T (ext.)")


def instance_rng(seed: int, s: int, k: int, i: int, *extra) -> np.random.Generator:
    """Generator for puzzle i of seed s at misread level k (independent of every other instance)."""
    return np.random.default_rng([seed, s, k, i, *extra])


def misread_cells_are_candidates(read, injected) -> bool:
    """Are all injected cells among the CANDIDATE_CELLS least confident givens (search can reach them)?"""
    problem = CorrectionProblem(read.grid, read.probs)
    return all(cell in problem.alternatives for cell in injected)


# ---------------------------------------------------------------- Experiments 1 and 3 + recovery
def run_reading_experiments(reader, reader_no_t, test_images, pools, puzzles, solutions,
                            n_seeds: int, per_seed: int, seed: int = config.SEED, log=print):
    """One pass over (seed, level, puzzle). Returns (exp1_rows, exp3_rows) as DataFrames."""
    exp1, exp3 = [], []
    for s in range(n_seeds):
        for k in config.MISREAD_LEVELS:
            start = time.perf_counter()
            for i in range(per_seed):
                idx = s * per_seed + i
                rng = instance_rng(seed, s, k, i)
                mosaic, _, injected = injection.make_level_mosaic(puzzles[idx], k, pools,
                                                                  test_images, rng)
                read = reader.read_mosaic(mosaic)
                truth = solutions[idx]
                base = {"seed": s, "level": k, "puzzle": idx,
                        "actual_misreads": int(((read.grid != puzzles[idx]) & (puzzles[idx] > 0)).sum()),
                        "misreads_in_candidates": misread_cells_are_candidates(read, injected)}
                results = {a: solve_reading(read, algorithm=a) for a in ALGORITHMS}
                for a, r in results.items():
                    st = r.search.stats if r.search else None
                    exp1.append({**base, "algorithm": a, "status": r.status,
                                 "exact": r.exact_recovery(truth),
                                 "n_corrections": len(r.corrections),
                                 "cost": r.total_cost if r.solved else np.nan,
                                 "search_status": r.search.status if r.search else "none",
                                 "nodes_expanded": st.nodes_expanded if st else 0,
                                 "nodes_generated": st.nodes_generated if st else 0,
                                 "nodes_pruned": st.nodes_pruned if st else 0,
                                 "goal_tests": st.goal_tests if st else 0,
                                 "csp_calls": st.csp_calls if st else 0,
                                 "max_frontier": st.max_frontier if st else 0,
                                 "seconds": st.seconds if st else 0.0})
                arms = {
                    "nn_guided": results[config.DEFAULT_ALGORITHM],
                    "no_correction": solve_reading(read, nn_correction=False),
                    "nn_guided_no_T (ext.)": solve_reading(
                        reader_no_t.read_mosaic(mosaic), algorithm=config.DEFAULT_ALGORITHM),
                }
                for arm, r in arms.items():
                    exp3.append({**base, "arm": arm, "status": r.status,
                                 "exact": r.exact_recovery(truth)})
                rand = random_correction(CorrectionProblem(read.grid, read.probs),
                                         instance_rng(seed, s, k, i, config.RANDOM_ARM_SEED_OFFSET))
                exp3.append({**base, "arm": "random", "status": rand.status,
                             "exact": rand.solution is not None and np.array_equal(rand.solution, truth)})
            log(f"  seed {s} level {k}: {per_seed} puzzles in {time.perf_counter() - start:.1f}s")
    return pd.DataFrame(exp1), pd.DataFrame(exp3)


# ---------------------------------------------------------------- Experiment 2
def run_csp_experiment(bank: dict, n_seeds: int, per_seed: int, log=print) -> pd.DataFrame:
    """Every CSP mode on the first n_seeds * per_seed bank puzzles of each given count."""
    rows = []
    for g in config.GIVEN_TARGETS:
        puzzles, solutions = bank[g]
        start = time.perf_counter()
        for s in range(n_seeds):
            for i in range(per_seed):
                idx = s * per_seed + i
                for mode, kw in CSP_MODES.items():
                    r = solve(puzzles[idx], ordering=config.ORDERING, **kw)
                    rows.append({"givens": g, "seed": s, "puzzle": idx, "mode": mode,
                                 "status": r.status, "timeout": r.status == "timeout",
                                 "correct": r.solution is not None
                                 and np.array_equal(r.solution, solutions[idx]),
                                 **r.stats.as_dict()})
        log(f"  {g} givens: {n_seeds * per_seed} puzzles x {len(CSP_MODES)} modes in "
            f"{time.perf_counter() - start:.1f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics
def mean_std_over_seeds(df: pd.DataFrame, by: list, metrics: list) -> pd.DataFrame:
    """Per seed: mean of each metric over its puzzles. Then mean and std (ddof=1) over seeds.

    Output columns: by..., n_seeds, <metric>_mean, <metric>_std.
    """
    per_seed = df.groupby(by + ["seed"])[metrics].mean().reset_index()
    grouped = per_seed.groupby(by)[metrics]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    out.insert(0, "n_seeds", per_seed.groupby(by)["seed"].nunique())
    ordered = ["n_seeds"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    return out[ordered].reset_index()


def recovery_table(exp1: pd.DataFrame) -> pd.DataFrame:
    """Exact-recovery study for the NN-guided system (default algorithm), pooled over all seeds."""
    d = exp1[exp1.algorithm == config.DEFAULT_ALGORITHM]
    rows = []
    for k, g in d.groupby("level"):
        per_seed = g.groupby("seed")["exact"].mean()
        rows.append({
            "level": k, "puzzles": len(g), "exact_recovered": int(g.exact.sum()),
            "exact_rate": g.exact.mean(), "rate_mean_over_seeds": per_seed.mean(),
            "rate_std_over_seeds": per_seed.std(ddof=1),
            "exact_as_read": int(((g.status == "solved") & g.exact).sum()),
            "exact_after_correction": int(((g.status == "corrected") & g.exact).sum()),
            "wrong_solution": int(((g.status != "unreadable") & ~g.exact).sum()),
            "wrong_as_read": int(((g.status == "solved") & ~g.exact).sum()),  # part of wrong_solution
            "unreadable": int((g.status == "unreadable").sum()),
            "misreads_all_in_candidates": int(g.misreads_in_candidates.sum()),
        })
    return pd.DataFrame(rows)
