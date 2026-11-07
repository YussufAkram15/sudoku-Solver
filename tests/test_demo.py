"""Smoke test: run_demo.py runs end to end and writes its four figures (needs the trained model)."""

import sys

import pytest

from sudoku import config
from tests.test_nn import needs_model

import run_demo


@needs_model
@pytest.mark.parametrize("misreads", [2, 4])
def test_demo_runs_and_saves_figures(tmp_path, monkeypatch, capsys, misreads):
    monkeypatch.setattr(config, "DEMO_DIR", tmp_path)
    monkeypatch.setattr(config, "DEMO_BATCH_PUZZLES", 1)
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--seed", "7", "--misreads", str(misreads)])
    run_demo.main()
    out = capsys.readouterr().out
    for section in ("=== 1. NN reading", "=== 2. AC-3 check", "=== 3. Correction search",
                    "=== 4. Result", "=== 5. CSP pruning", "=== 6. NN-guided correction"):
        assert section in out
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"demo_pipeline.png", "demo_search.png", "demo_csp.png", "demo_nn_on_off.png"}


def test_grid_text_marks_and_blanks():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    conf = [[1.0] * 9 for _ in range(9)]
    text = run_demo.grid_text(grid, conf, marks={(0, 0)})
    first = text.splitlines()[0]
    assert first.startswith("5*100") and len(set(len(line) for line in text.splitlines())) == 1
