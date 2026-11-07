"""Baseline: multinomial logistic regression on the 784 raw pixels (scikit-learn)."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from sudoku import config


def flatten(images) -> np.ndarray:
    """(N, 28, 28) -> (N, 784) float array for scikit-learn."""
    images = np.asarray(images, dtype=np.float32)
    return images.reshape(len(images), -1)


def train_logreg(train_images, train_classes, subset: int | None = config.LOGREG_TRAIN_SUBSET,
                 seed: int = config.SEED) -> LogisticRegression:
    """Fit softmax regression on the training split (or a seeded random subset of ``subset`` images).

    With the lbfgs solver and 9 classes scikit-learn fits the multinomial (softmax) model.
    """
    idx = config.make_rng(seed).permutation(len(train_classes))[:subset]  # [:None] = all
    model = LogisticRegression(max_iter=config.LOGREG_MAX_ITER, random_state=seed)
    model.fit(flatten(train_images[idx]), np.asarray(train_classes)[idx])
    return model


def logreg_logits(model: LogisticRegression, images) -> np.ndarray:
    """Per-class scores (N, 9); softmax of these equals ``predict_proba``."""
    return model.decision_function(flatten(images))
