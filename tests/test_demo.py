"""Smoke test: run_demo.py runs end to end and writes its four figures (needs the trained model)."""

import sys

import pytest

from sudoku import config
from sudoku.integration import UNREADABLE_MESSAGE
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
    if misreads == 2:
        assert "exact recovery of the original solution: True" in out
        assert "correction (" in out
    else:  # 4 misreads exceed the 3-correction budget
        assert UNREADABLE_MESSAGE in out and "least confident cells" in out
    names = {p.name for p in tmp_path.rglob("*.png")}
    if misreads != 2:  # non-default arguments go to their own sub-folder
        assert (tmp_path / "runs").is_dir() and not list(tmp_path.glob("*.png"))
    assert names == {"demo_pipeline.png", "demo_search.png", "demo_csp.png", "demo_nn_on_off.png"}


def test_minimum_version_goal_flag(tmp_path, monkeypatch, capsys):
    """--goal solvable runs the brief's minimum version (at least one solution)."""
    if not (config.MODELS / config.CNN_FILE).exists():
        pytest.skip("run train.py first")
    monkeypatch.setattr(config, "DEMO_DIR", tmp_path)
    monkeypatch.setattr(config, "DEMO_BATCH_PUZZLES", 1)
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--misreads", "1", "--algorithm", "ucs",
                                      "--goal", "solvable"])
    run_demo.main()
    out = capsys.readouterr().out
    assert "goal: solvable" in out and "status: corrected" in out


def test_grid_text_marks_and_blanks():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    conf = [[1.0] * 9 for _ in range(9)]
    text = run_demo.grid_text(grid, conf, marks={(0, 0)})
    first = text.splitlines()[0]
    assert first.startswith("5*100") and len(set(len(line) for line in text.splitlines())) == 1
