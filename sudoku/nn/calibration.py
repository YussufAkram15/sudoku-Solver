"""Calibration: softmax, Expected Calibration Error, reliability bins, temperature scaling.

A classifier is *calibrated* if, among predictions made with confidence ~0.8, about 80% are right.
That matters here because the correction search turns probabilities into costs.

ECE (equal-width bins over the top-1 confidence):
    ECE = sum over bins b of (n_b / N) * |accuracy_b - mean_confidence_b|

Temperature scaling divides the logits by one number T > 0 fitted on the VALIDATION split by
minimising negative log-likelihood. It never changes the argmax, so accuracy is unchanged.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from sudoku import config


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of ``logits / temperature`` (numerically stable)."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_bins(probs, labels, n_bins: int = config.ECE_BINS) -> dict:
    """Per-bin count, accuracy and mean confidence of the top-1 prediction.

    Bin b covers confidences in (b/n_bins, (b+1)/n_bins]; empty bins have accuracy/confidence NaN.
    """
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=True puts conf == edge into the lower bin -> bins (lo, hi]
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    counts = np.bincount(which, minlength=n_bins)
    acc = np.full(n_bins, np.nan)
    mean_conf = np.full(n_bins, np.nan)
    for b in np.flatnonzero(counts):
        acc[b] = correct[which == b].mean()
        mean_conf[b] = conf[which == b].mean()
    return {"edges": edges, "count": counts, "accuracy": acc, "confidence": mean_conf}


def expected_calibration_error(probs, labels, n_bins: int = config.ECE_BINS) -> float:
    """ECE with ``n_bins`` equal-width confidence bins (0 = perfectly calibrated)."""
    bins = reliability_bins(probs, labels, n_bins)
    used = bins["count"] > 0
    gaps = np.abs(bins["accuracy"][used] - bins["confidence"][used])
    return float((bins["count"][used] / bins["count"].sum() * gaps).sum())


def negative_log_likelihood(logits, labels, temperature: float) -> float:
    """Mean -log p(true class) after temperature scaling."""
    probs = softmax(logits, temperature)
    p_true = probs[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.log(np.clip(p_true, config.PROB_EPS, None)).mean())


def fit_temperature(val_logits, val_labels, bounds=config.TEMPERATURE_BOUNDS) -> float:
    """Temperature T minimising validation NLL (bounded 1-D search)."""
    result = minimize_scalar(lambda t: negative_log_likelihood(val_logits, val_labels, t),
                             bounds=bounds, method="bounded")
    return float(result.x)
