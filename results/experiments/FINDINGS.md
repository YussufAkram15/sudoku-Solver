# Experiment findings (full run, seed 7)

`python run_experiments.py` — 5 seeds × 20 puzzles = 100 puzzles per misread level (30 givens) and
100 puzzles per given count for Experiment 2. Runtime ≈ 3.5 min on a 12-core CPU. Every number below
comes from the CSVs in this folder; "±" is the standard deviation over the 5 seeds.

What a "seed" is: seed s uses its own slice of 20 bank puzzles (puzzles 20s .. 20s+19, all made
with generator seed 7) and its own mosaic / injection / random-baseline draws. The CNN is trained
once and the CSP has no randomness, so the spread is over puzzle subsets and image draws, not over
retrained models or regenerated banks.

## Exact-solution recovery (P7 deliverable) — `recovery.csv`, `recovery.png`

| Misreads | Exact | Exact w/o correction | Exact after correction | Wrong solution | Unreadable |
|---|---|---|---|---|---|
| 0 | 100 % | 100 | 0 | 0 | 0 |
| 1 | 95 % (± 5) | 0 | 95 | 5 (4 accepted as read) | 0 |
| 2 | 89 % (± 7) | 0 | 89 | 11 | 0 |
| 3 | 87 % (± 8) | 0 | 87 | 13 | 0 |
| 4 | 0 % | 0 | 0 | 40 | 60 |

- In every puzzle all misread cells were among the 15 least-confident candidates, so failures are
  not caused by the candidate cut-off.
- "Wrong solution": the system returned a uniquely solvable grid that is not the generator's
  original. At level 1, 4 of these needed no correction at all: the misread digit happened to give
  another valid puzzle with a unique solution, which no consistency check can detect. At levels
  2–3 the search found a different correction set whose puzzle is also uniquely solvable.
- Level 4 exceeds the 3-correction budget by design: 60 puzzles end "unreadable", 40 end with a
  wrong solution found within 3 corrections. The uniqueness goal cannot catch these: the
  corrected puzzle really has exactly one solution, just not the original one.
- The misread pool has only 133 test images (8–32 per digit), so the same hard images recur
  across the 100 puzzles.

## Correction log (P7 deliverable) — `correction_log.csv`

685 corrections on 336 puzzles (NN-guided, A*), one row each with cell, read digit, new digit and
cost. 94.9 % of them set the true digit and 97.5 % are on a cell where a hard digit was injected.
No goal test hit its CSP node/time limit in the whole run (`csp_timeouts` = 0 in
`exp1_per_puzzle.csv`).

## Experiment 1 — UCS vs Greedy vs A* (`exp1_summary.csv`, `exp1_search.png`)

- Nodes expanded (mean ± std): level 1: UCS 2.0, Greedy 1.2, A* 1.1; level 2: 8.8 / 6.4 / 3.5;
  level 3: 52.5 / 23.3 / 11.2; level 4 (exhausted): 8,200 / 2,130 / 2,150.
- A* always found the same cost as UCS (optimal); Greedy found a more expensive correction set in
  26 of 440 puzzles both solved (mean cost of solved puzzles, `cost_mean` in `exp1_summary.csv`:
  level 3 A* 4.55 vs Greedy 5.32), and has lower recovery at levels 2–3 (84 %, 77 % vs 89 %, 87 %).
- Part of A*/Greedy's saving comes from pruning children with h = ∞ (an unfixable duplicate),
  reported separately as `nodes_pruned`; UCS never prunes. On exhausted searches UCS's "expanded"
  includes every 3-correction leaf.
- Wall-clock time includes computing h, so A* is slower per node than UCS: at level 4 A* takes
  about twice as long as UCS (0.36 s vs 0.17 s in `exp1_summary.csv`; timings vary between runs)
  although it expands about 4× fewer nodes.

## Experiment 2 — BT vs BT+FC vs BT+FC+AC-3 (`exp2_summary.csv`, `exp2_csp.png`)

Static row-major variable order in all modes (with assignment-based MRV the three modes make
identical assignments, see `config.ORDERING`). Timeouts = 200,000-assignment cap; timed-out runs
count with the capped work, so BT means at 25 givens are lower bounds.

| Givens | BT assignments | BT+FC | BT+FC+AC-3 | +MAC (ext.) | Timeouts BT / FC / FC+AC-3 / MAC |
|---|---|---|---|---|---|
| 25 | 44,600 | 13,600 | 9,460 | 271 | 10 / 1 / 1 / 0 of 100 |
| 30 | 5,520 | 1,270 | 540 | 60 | 0 |
| 35 | 587 | 205 | 96 | 47 | 0 |

Every solved run returned the generator's solution. Each switch cuts search; AC-3's per-call cost
means BT+FC+AC-3 is not always faster in seconds than BT+FC on easy (35-given) puzzles.

## Experiment 3 — NN-guided vs no correction vs random (`exp3_summary.csv`, `exp3_recovery.png`)

| Misreads | NN-guided (A*) | No correction | Random (any given) | Random within candidates (ext.) | NN-guided, no T (ext.) |
|---|---|---|---|---|---|
| 0 | 100 % | 100 % | 100 % | 100 % | 100 % |
| 1 | 95 % | 0 % | 28 % | 66 % | 95 % |
| 2 | 89 % | 0 % | 0 % | 9 % | 89 % |
| 3 | 87 % | 0 % | 0 % | 0 % | 87 % |
| 4 | 0 % | 0 % | 0 % | 0 % | 0 % |

Both random arms get up to 201 goal tests (the read grid plus 200 random draws) per puzzle; A* needs on average 2.1 / 4.5 / 12.2
(max 48) at levels 1 / 2 / 3. "Random
within candidates" may only choose the same 15 cells and top-3 alternatives the search uses, so
the gap to NN-guided (66 vs 95 %, 9 vs 89 %, 0 vs 87 %) is the value of ordering corrections by
NN cost, beyond what the candidate cut-off already gives.

Temperature scaling (T = 0.868) made no difference to recovery, and this is expected: a step cost
is log p(r) − log p(d) = (z_r − z_d) / T for logits z, so every cost is multiplied by the same 1/T
and the order of correction sets cannot change. T can only change which 15 cells are the least
confident candidates, and here it never did in a way that mattered. Calibration matters for the
reported probabilities (ECE), not for which correction wins.
