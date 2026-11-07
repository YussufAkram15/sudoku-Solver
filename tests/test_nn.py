"""Tests for the NN part: model shapes, training determinism, ECE, temperature scaling, the reader.

Tests marked ``needs_model`` use the models trained by ``python train.py`` and are skipped otherwise.
"""

import joblib
import numpy as np
import pytest
import torch

from sudoku import config
from sudoku.data import generator, injection, mnist, mosaic
from sudoku.nn import baseline, calibration
from sudoku.nn.model import DigitCNN, predict_logits
from sudoku.nn.reader import DigitReader, ReadGrid
from sudoku.nn.training import train_cnn

T, N = config.TILE, config.GRID
HAVE_MODEL = (config.MODELS / config.CNN_FILE).exists() and config.POOLS_FILE.exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="run make_data.py and train.py first")


def _untrained_reader(seed=0):
    torch.manual_seed(seed)
    return DigitReader(DigitCNN(), temperature=1.0)


# ---------------------------------------------------------------- model and training
def test_cnn_outputs_one_logit_per_digit():
    logits = predict_logits(DigitCNN(), np.zeros((5, T, T), np.float32))
    assert logits.shape == (5, config.N_CLASSES)


def test_training_is_seeded():
    rng = np.random.default_rng(0)
    x = rng.random((64, T, T), dtype=np.float32)
    y = rng.integers(0, config.N_CLASSES, 64)
    m1, h1 = train_cnn(x, y, x[:8], y[:8], epochs=1, seed=3, verbose=False)
    m2, h2 = train_cnn(x, y, x[:8], y[:8], epochs=1, seed=3, verbose=False)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    assert h1[0]["train_loss"] == h2[0]["train_loss"]


# ---------------------------------------------------------------- calibration
def test_ece_is_zero_when_confidence_matches_accuracy():
    # 10 predictions at confidence 0.8, exactly 8 of them right -> ECE 0
    probs = np.full((10, 9), 0.2 / 8)
    probs[:, 0] = 0.8
    labels = np.array([0] * 8 + [1] * 2)
    assert calibration.expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_ece_hand_computed_value():
    # bin of conf 0.9 with 50% right (gap 0.4) and bin of conf 0.5 with 100% right (gap 0.5),
    # equal sizes -> ECE = 0.5 * 0.4 + 0.5 * 0.5 = 0.45
    probs = np.zeros((4, 9))
    probs[:2, 0], probs[:2, 1:] = 0.9, 0.1 / 8
    probs[2:, 0], probs[2:, 1:] = 0.5, 0.5 / 8
    labels = np.array([0, 3, 0, 0])
    assert calibration.expected_calibration_error(probs, labels) == pytest.approx(0.45)


def test_temperature_recovers_known_overconfidence():
    # Labels are sampled from softmax(z) (calibrated). The model reports 3*z (overconfident),
    # so the best temperature is about 3.
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, size=(5000, 9))
    p = calibration.softmax(z)
    labels = np.array([rng.choice(9, p=row) for row in p])
    t = calibration.fit_temperature(3 * z, labels)
    assert t == pytest.approx(3.0, rel=0.1)
    assert (calibration.expected_calibration_error(calibration.softmax(3 * z, t), labels)
            < calibration.expected_calibration_error(calibration.softmax(3 * z), labels))


def test_temperature_never_changes_the_prediction():
    z = np.random.default_rng(2).normal(size=(200, 9))
    for t in (0.3, 1.0, 4.0):
        assert np.array_equal(calibration.softmax(z, t).argmax(1), z.argmax(1))


# ---------------------------------------------------------------- reader
@pytest.mark.normal
def test_reader_gives_probability_vector_per_nonblank_cell():
    puzzle = generator.generate_puzzle(30, config.make_rng(0)).puzzle
    images = np.random.default_rng(0).random((9, T, T)).astype(np.float32)  # never blank
    pools = {d: np.array([d - 1]) for d in config.DIGITS}
    image, _ = mosaic.make_mosaic(puzzle, images, pools, config.make_rng(0))
    read = _untrained_reader().read_mosaic(image)
    assert isinstance(read, ReadGrid)
    assert read.probs.shape == (N, N, config.N_CLASSES)
    filled = puzzle != 0
    assert np.array_equal(read.blank, ~filled)
    assert np.allclose(read.probs[filled].sum(axis=1), 1.0)       # distribution over 1..9
    assert np.all(read.probs[~filled] == 0) and np.all(read.grid[~filled] == 0)
    assert read.grid[filled].min() >= 1 and read.grid[filled].max() <= 9
    assert np.array_equal(read.grid[filled], read.probs[filled].argmax(1) + 1)
    (d0, p0), (d1, p1), _ = read.top_k(*np.argwhere(filled)[0])
    assert p0 >= p1 and d0 == read.grid[tuple(np.argwhere(filled)[0])]


@pytest.mark.edge
def test_reader_on_empty_mosaic_reads_nothing():
    read = _untrained_reader().read_mosaic(np.zeros((N * T, N * T), np.float32))
    assert read.blank.all() and not read.grid.any() and not read.probs.any()


# ---------------------------------------------------------------- trained model (after train.py)
@needs_model
def test_trained_cnn_beats_logreg_baseline_on_test():
    reader = DigitReader.load()
    test = mnist.load_splits()["test"]
    logreg = joblib.load(config.MODELS / config.LOGREG_FILE)["model"]
    cnn_acc = (reader.predict_proba(test.images).argmax(1) == test.classes).mean()
    lr_acc = (logreg.predict(baseline.flatten(test.images)) == test.classes).mean()
    assert cnn_acc > 0.97
    assert cnn_acc > lr_acc


@needs_model
@pytest.mark.normal
@pytest.mark.parametrize("k", config.MISREAD_LEVELS)
def test_level_k_mosaic_has_exactly_k_misreads(k):
    """End-to-end: pools + injection + real reader -> exactly k cells read wrong."""
    reader = DigitReader.load()
    test = mnist.load_splits()["test"]
    pools = injection.load_pools()
    for i in range(3):
        p = generator.generate_puzzle(config.DEFAULT_GIVENS, generator.puzzle_rng(99, k, i))
        image, _, cells = injection.make_level_mosaic(p.puzzle, k, pools, test.images,
                                                      config.make_rng(i))
        read = reader.read_mosaic(image)
        assert np.array_equal(read.blank, p.puzzle == 0)
        wrong = np.argwhere((read.grid != p.puzzle) & (p.puzzle != 0))
        assert sorted(map(tuple, wrong.tolist())) == cells
        for r, c in cells:  # the true digit stays reachable by the correction search
            assert p.puzzle[r, c] in [d for d, _ in read.top_k(r, c, config.HARD_TOPK)]


@needs_model
def test_saved_pool_probs_match_reader():
    reader = DigitReader.load()
    test = mnist.load_splits()["test"]
    with np.load(config.POOLS_FILE) as data:
        assert data["temperature"] == pytest.approx(reader.temperature)
        assert np.allclose(data["probs"][:200], reader.predict_proba(test.images[:200]), atol=1e-5)
