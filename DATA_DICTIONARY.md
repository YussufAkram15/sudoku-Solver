# Data dictionary

Every file, array and field the project uses. **REAL** = taken unchanged from MNIST (images and
labels drawn and labelled by real people). **GENERATED** = produced by our code, with the exact rule
below. All randomness uses the seed passed to the script (default `SEED = 7` in `sudoku/config.py`).

Conventions: a *digit* is the label 1–9 a Sudoku cell holds; a *class* is the network output index
0–8 (`class = digit - 1`). Grids are 9×9 integer arrays, row-major, `0` = empty cell; cell (r, c)
has r, c in 0..8.

## 1. Raw MNIST — `data/raw/MNIST/raw/` (REAL, git-ignored)

Downloaded by `make_data.py` through `torchvision.datasets.MNIST` (the yann.lecun.com page is often
unreachable). Four IDX files (plus their `.gz`):

| File | Content | Type |
|---|---|---|
| `train-images-idx3-ubyte` | 60,000 images, 28×28, grey levels 0–255 | REAL |
| `train-labels-idx1-ubyte` | 60,000 labels 0–9 | REAL |
| `t10k-images-idx3-ubyte` | 10,000 test images (different writers from train) | REAL |
| `t10k-labels-idx1-ubyte` | 10,000 test labels 0–9 | REAL |

## 2. Digit splits — `data/processed/mnist_digits.npz` (REAL images/labels, GENERATED split)

Written by `make_data.py` (`sudoku/data/mnist.py`). Rule:
1. Seeded permutation of the 60,000 MNIST train indices (`numpy.random.default_rng(seed)`);
   first 50,000 = train, next 10,000 = validation.
2. Drop every digit-0 image in each part. Only 54,077 train images are non-zero, so 50k + 10k cannot
   both be filled after dropping 0; what remains is ≈45k train / ≈9k validation (exact counts are
   printed by `make_data.py`; they sum to 54,077).
3. MNIST test: all 10,000 kept in original order, digit 0 dropped → 9,020 images. Not shuffled, not
   used for training, validation or temperature fitting. It is the **only** source of mosaic tiles.

| Array (P = `train`, `val`, `test`) | Shape / dtype | Meaning | Kind |
|---|---|---|---|
| `P_images` | (n, 28, 28) uint8 | pixel values 0–255 exactly as in MNIST | REAL |
| `P_digits` | (n,) int64 | true digit 1–9 | REAL |
| `P_source_idx` | (n,) int64 | row of the image in the original MNIST train (train/val) or test file | GENERATED (bookkeeping) |
| `seed` | scalar int64 | seed of the train/validation permutation | GENERATED |

Preprocessing on load (`load_splits`): pixels converted to float32 and divided by 255 → range 0–1.
No other normalisation, no augmentation.

## 3. Puzzle bank — `data/processed/puzzles.npz` (GENERATED)

Written by `make_data.py` (`sudoku/data/generator.py`, `sudoku/data/bank.py`).
`make_data.py --quick` writes a small bank to `puzzles_quick.npz` with the same layout.

| Array | Shape / dtype | Meaning |
|---|---|---|
| `g{G}_puzzles` | (n, 9, 9) int64 | givens, 0 = empty; G ∈ {25, 30, 35} givens |
| `g{G}_solutions` | (n, 9, 9) int64 | the unique solution (the full grid the puzzle was cut from) |
| `g{G}_attempts` | (n,) int64 | full grids the generator needed to reach G givens |
| `seed` | scalar int64 | generator seed |

n = 100 per G (`N_SEEDS × PUZZLES_PER_SEED`), 6 per G in the quick bank.

How a puzzle is generated (`generate_puzzle`):
1. Puzzle i with G givens gets its own generator `numpy.random.default_rng([seed, G, i])`, so any
   single puzzle can be regenerated on its own.
2. Full grid by randomised backtracking: fill cells in row-major order, trying digits 1–9 in a random
   order, undoing on a dead end.
3. Visit all 81 cells in random order; blank each one and keep the blank only if the puzzle still has
   exactly one solution (checked by a solution counter that stops at 2); otherwise restore the digit.
   Stop as soon as G givens remain.
4. If a full pass ends with more than G givens, start again from a new full grid
   (at most `GENERATOR_MAX_ATTEMPTS = 20` grids).

Guarantees (tested): exactly G givens, givens agree with the solution, exactly one solution.

## 4. Mosaics (GENERATED, built in memory)

A mosaic is a 252×252 float32 image (0–1) made of 9×9 tiles of 28×28 (`sudoku/data/mosaic.py`).

| Object | Shape / dtype | Meaning |
|---|---|---|
| `tile_index` | (9, 9) int64 | row of the MNIST-**test** split drawn in each cell; `-1` = empty cell |
| `mosaic` | (252, 252) float32 | tile (r, c) occupies pixel rows `28r..28r+27`, cols `28c..28c+27` |

Rule: for every given cell holding digit d, choose a random test image (seeded `rng.choice`) from the
pool of digit d (pools in §5). Empty cells are all-zero tiles. The image is real; its placement is
generated. `results/data/sample_mosaic.png` shows one example.

**Blank detection** (not the NN): a tile is empty if its mean pixel < `BLANK_THRESHOLD = 0.01`.
`make_data.py` checks that every one of the 9,020 test digits is brighter than this (the darkest
mean is printed) and stops with an error otherwise.

## 5. Hard-digit injection (GENERATED placement of REAL images)

A well-trained CNN reads almost every MNIST digit correctly, so random mosaics would leave nothing to
correct. Hard images are injected with this documented rule (`sudoku/data/injection.py`); the pools
depend on the trained reader, so they are computed after training (`train.py`) from the reader's
probabilities on the MNIST test split (the same calibrated probabilities the pipeline uses).

- **Hard pool** (rule from the brief): test images where the reader is wrong **or** its top-1
  probability is < `HARD_PROB_THRESHOLD = 0.6`, **and** the true digit is within the reader's top
  `HARD_TOPK = 3` digits (so the search can still reach the true digit).
- **Misread pool**: the part of the hard pool that the reader gets wrong (argmax ≠ true digit).
- **Clean pool**: test images the reader classifies correctly.

Building a puzzle at misread level k ∈ {0, 1, 2, 3, 4}:
1. Assign tiles to all givens (§4).
2. Pick k distinct given cells uniformly at random among those whose digit has a non-empty
   injection pool, and replace each one's image with a random injection-pool image **of the same
   digit**. The Sudoku digits never change; only which handwritten image shows them.

With `EXACT_MISREADS = True` (default) the base pool is the clean pool and the injection pool is the
misread pool, so a level-k puzzle has **exactly k** cells read wrong by the reader. With
`EXACT_MISREADS = False` the base pool is all test images and the injection pool is the whole hard
pool (the brief's literal rule); the actual misread count can then differ from k.

### Saved pools — `data/processed/digit_pools.npz` (GENERATED from REAL test images, git-ignored)

Written by `train.py` step 6 (`--quick` writes `digit_pools_quick.npz` from the smoke-test model).
`injection.load_pools()` rebuilds the base/injection pools from `probs` with the rules above.

| Array | Shape / dtype | Meaning |
|---|---|---|
| `probs` | (9020, 9) float64 | reader probabilities (CNN + temperature T) on each MNIST-test row, column i = digit i+1 |
| `digits` | (9020,) int64 | true digit of each MNIST-test row (REAL) |
| `temperature` | scalar float64 | T used for `probs` (same T the pipeline's reader uses) |
| `hard`, `misread`, `clean` | (9020,) bool | membership of each test row in the three pools |

Pool sizes with the default seed 7 (also in `results/nn/pool_sizes.csv`; CNN test accuracy 98.49%):

| Digit | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | all |
|---|---|---|---|---|---|---|---|---|---|---|
| hard | 11 | 18 | 27 | 21 | 19 | 14 | 39 | 13 | 14 | 176 |
| misread | 8 | 12 | 18 | 14 | 15 | 13 | 32 | 12 | 9 | 133 |
| clean | 1127 | 1020 | 990 | 968 | 877 | 945 | 995 | 962 | 1000 | 8884 |

The misread pool is small (8–32 images per digit), so across 100 puzzles the same hard image is
reused many times. Every digit has at least one, so every given cell is eligible for injection.

## 6. Trained models — `models/` (GENERATED, git-ignored)

Written by `train.py` (`--quick` writes `models/quick/` instead).

| File | Content |
|---|---|
| `cnn.pt` | PyTorch `state_dict` of the CNN (`sudoku/nn/model.py`) |
| `cnn_meta.json` | `temperature` (fitted on validation), `seed`, `epochs`, `train_size`, per-epoch `history`, `test_accuracy` |
| `logreg.joblib` | dict `{"model": LogisticRegression, "temperature": float}` for the baseline |

## 7. Results — `results/` (GENERATED, committed)

All results are computed from REAL MNIST test images placed into GENERATED puzzles; no field is
taken directly from a data set.

### `results/data/`, `results/demo/`
PNG figures only: `sample_mosaic.png` (make_data.py) and the four `demo_*.png` of
`python run_demo.py` with default arguments (seed 7, 2 misreads).

### `results/nn/` (train.py, MNIST test split, digits 1–9)

| File | Columns / content |
|---|---|
| `metrics.csv` | `model` (cnn, cnn+temperature, logreg, logreg+temperature), `temperature`, `accuracy`, `nll` (mean negative log-likelihood), `ece` (15 equal-width bins), `train_size`, `train_seconds` |
| `cnn_history.csv` | `epoch`, `train_loss` (mean cross-entropy), `val_acc`, `seconds` |
| `confusion_cnn.csv`, `confusion_logreg.csv` | 9×9 counts, rows = true digit 1–9, columns = predicted digit 1–9 |
| `pool_sizes.csv` | `digit` (1–9 or `all`), `hard`, `misread`, `clean`: test images per pool (§5) |
| `confusion_*.png`, `reliability.png` | confusion matrices; reliability diagrams for all four model variants |

### `results/experiments/` (run_experiments.py)

"Seed" `s` = run number 0–4: bank puzzles `20s .. 20s+19` and mosaics seeded with
`(SEED, s, level, i)`. Per-puzzle files have one row per (seed, puzzle, method); summary files
hold, for each group, the mean over the puzzles of each seed, then `<metric>_mean` and
`<metric>_std` (sample std, ddof = 1) over the `n_seeds` seeds.

| File | Columns |
|---|---|
| `exp1_per_puzzle.csv` | `seed`, `level` (injected misreads), `puzzle` (bank index, 30 givens), `actual_misreads` (cells the reader got wrong), `misreads_in_candidates` (all misread cells among the 15 search candidates), `algorithm` (ucs/greedy/astar), `status` (solved/corrected/unreadable), `exact` (output equals the generator's solution), `n_corrections`, `cost` (total correction cost; empty if unsolved), `search_status` (found/exhausted/node_limit/time_limit), `nodes_expanded`, `nodes_generated`, `nodes_pruned` (children dropped because h = ∞), `goal_tests`, `csp_calls`, `max_frontier`, `seconds` |
| `exp1_summary.csv` | `level`, `algorithm`, `n_seeds`, mean/std of nodes expanded/generated/pruned, goal tests, CSP calls, seconds, cost, n_corrections (cost and n_corrections over solved puzzles only), exact |
| `exp2_per_puzzle.csv` | `givens` (25/30/35), `seed`, `puzzle`, `mode` (`BT`, `BT+FC`, `BT+FC+AC-3`, `BT+FC+AC-3+MAC (ext.)`), `status` (solved/unsolvable/timeout), `timeout`, `correct`, `assignments`, `backtracks`, `ac3_revisions`, `ac3_pruned`, `fc_pruned`, `seconds`, `values_pruned` (= ac3_pruned + fc_pruned) |
| `exp2_summary.csv` | `givens`, `mode`, `n_seeds`, mean/std of assignments, backtracks, AC-3 revisions, values pruned, seconds, timeout rate, correct rate |
| `exp3_per_puzzle.csv` | `seed`, `level`, `puzzle`, `actual_misreads`, `misreads_in_candidates`, `arm` (`nn_guided`, `no_correction`, `random`, `random_in_candidates (ext.)`, `nn_guided_no_T (ext.)`), `status`, `exact`. `status` uses the pipeline's values (solved/corrected/unreadable) for the nn_guided, no_correction and no-T arms, and the search's values (found = a goal state was tested positive, exhausted = all trials failed) for the two random arms |
| `exp3_summary.csv` | `level`, `arm`, `n_seeds`, `exact_mean`, `exact_std` |
| `recovery.csv` | per `level`: `puzzles`, `exact_recovered`, `exact_rate`, `rate_mean_over_seeds`, `rate_std_over_seeds`, breakdown `exact_as_read` + `exact_after_correction` + `wrong_solution` + `unreadable` (= puzzles), `wrong_as_read` (part of wrong_solution), `misreads_all_in_candidates` |
| `*.png`, `FINDINGS.md` | plots of the tables above and a factual summary of the results |

`results/experiments/quick/` and `results/nn/quick/` hold smoke-test outputs and are git-ignored.
