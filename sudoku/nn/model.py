"""Small CNN digit classifier: 2 conv layers + 2 fully connected layers.

Input  (N, 1, 28, 28) float pixels in 0..1.
Output (N, 9) logits, class i = digit i + 1 (digit 0 is not in the data).
"""

import torch
from torch import nn

from sudoku import config


class DigitCNN(nn.Module):
    """conv(1->16) -> ReLU -> pool -> conv(16->32) -> ReLU -> pool -> FC(128) -> ReLU -> FC(9)."""

    def __init__(self, channels=config.CNN_CONV_CHANNELS, hidden=config.CNN_HIDDEN):
        super().__init__()
        c1, c2 = channels
        pad = config.CNN_KERNEL // 2  # "same" padding: 28 -> 28 -> pool 14 -> 14 -> pool 7
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, config.CNN_KERNEL, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, config.CNN_KERNEL, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        side = config.TILE // 4  # two 2x2 poolings
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * side * side, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no softmax) for a batch of images."""
        return self.classifier(self.features(x))


def to_tensor(images) -> torch.Tensor:
    """(N, 28, 28) array in 0..1 -> (N, 1, 28, 28) float32 tensor."""
    return torch.as_tensor(images, dtype=torch.float32).unsqueeze(1)


@torch.no_grad()
def predict_logits(model: nn.Module, images, batch_size: int = config.CNN_EVAL_BATCH):
    """Logits for many images as a NumPy array (N, 9), computed in batches without gradients."""
    model.eval()
    x = to_tensor(images)
    out = [model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy() if out else torch.zeros((0, config.N_CLASSES)).numpy()
