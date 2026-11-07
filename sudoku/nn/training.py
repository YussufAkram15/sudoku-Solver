"""Seeded training loop for the CNN (Adam + cross-entropy, a few epochs on CPU)."""

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sudoku import config
from sudoku.nn.model import DigitCNN, predict_logits, to_tensor


def train_cnn(train_images, train_classes, val_images, val_classes,
              epochs: int = config.CNN_EPOCHS, seed: int = config.SEED, verbose: bool = True):
    """Train a fresh ``DigitCNN``; returns (model, history).

    ``*_classes`` are network targets 0..8. ``history`` holds one dict per epoch with the mean
    training loss, validation accuracy and seconds taken. Same seed -> same weights.
    """
    torch.manual_seed(seed)
    model = DigitCNN()
    data = TensorDataset(to_tensor(train_images), torch.as_tensor(train_classes, dtype=torch.long))
    loader = DataLoader(data, batch_size=config.CNN_BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimiser = torch.optim.Adam(model.parameters(), lr=config.CNN_LR)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        start, total, n = time.perf_counter(), 0.0, 0
        model.train()
        for x, y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(y)
            n += len(y)
        val_acc = float((predict_logits(model, val_images).argmax(1) == np.asarray(val_classes)).mean())
        history.append({"epoch": epoch, "train_loss": total / n, "val_acc": val_acc,
                        "seconds": time.perf_counter() - start})
        if verbose:
            h = history[-1]
            print(f"  epoch {epoch}: train loss {h['train_loss']:.4f}  val acc {val_acc:.4f}"
                  f"  ({h['seconds']:.1f}s)")
    return model, history
