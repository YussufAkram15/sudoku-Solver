# P7 — Handwritten Sudoku Reader and Solver (Search + CSP + NN)

A Sudoku drawn as a 9×9 mosaic of MNIST digits is read by a CNN, checked with AC-3, repaired by a
search over the cheapest cell corrections (UCS / Greedy / A*), and solved by a CSP solver.

> Work in progress — the full run guide, module map and results are added in the final phase.

## Setup

Requires **Python 3.12** (pins tested on 3.12.3, Linux, CPU only). Recent numpy/pandas pins may not
install on older Python versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins every package version that was installed and tested. It contains
`--extra-index-url https://download.pytorch.org/whl/cpu`, so pip finds the CPU-only
`torch==…+cpu` / `torchvision==…+cpu` wheels (avoids the multi-GB CUDA build) while all other
packages still come from PyPI in the same command. `--index-url` is deliberately not used: it would
replace PyPI and break the other installs.

## Run order

```bash
python make_data.py        # download MNIST, split, generate puzzles (seeded)
python train.py            # train CNN + logistic-regression baseline, calibrate, build digit pools
pytest                     # run the test suite
python run_demo.py         # integrated demo (e.g. --seed 7 --misreads 2)
python run_experiments.py  # all experiments (add --quick for a short smoke run)
```

All seeds, thresholds and limits live in `sudoku/config.py` (default seed 7).
