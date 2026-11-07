"""Matplotlib figures shared by run_demo.py and run_experiments.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from sudoku import config  # noqa: E402

N, B = config.GRID, config.BOX
ALGO_COLORS = {"ucs": "tab:blue", "greedy": "tab:orange", "astar": "tab:green", "random": "tab:gray"}


def save(fig, path: Path) -> None:
    """Save a figure (creating the folder) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def draw_grid(ax, grid, title: str, shade=None, marks=None, inner_marks=None, small=None,
              cmap="Greens") -> None:
    """Draw a 9x9 Sudoku grid.

    shade:       optional (9, 9) values in 0..1 used as light cell background (e.g. confidence).
    marks:       {cell: colour} - thick coloured border around those cells.
    inner_marks: {cell: colour} - dashed border just inside the cell (a second kind of mark).
    small:       {cell: text} - small text under the digit (e.g. confidence in %).
    """
    if shade is not None:
        ax.imshow(np.ma.masked_where(np.asarray(grid) == 0, shade), cmap=cmap, vmin=0, vmax=1,
                  alpha=0.45, extent=(-0.5, N - 0.5, N - 0.5, -0.5))
    for (r, c), d in np.ndenumerate(np.asarray(grid)):
        if d:
            ax.text(c, r - (0.1 if small else 0), str(d), ha="center", va="center", fontsize=12)
        if small and (r, c) in small:
            ax.text(c, r + 0.3, small[(r, c)], ha="center", va="center", fontsize=6, color="dimgray")
    for (r, c), colour in (marks or {}).items():
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=colour, lw=2.5))
    for (r, c), colour in (inner_marks or {}).items():
        ax.add_patch(Rectangle((c - 0.38, r - 0.38), 0.76, 0.76, fill=False, edgecolor=colour,
                               lw=1.8, linestyle="--"))
    for k in range(N + 1):
        lw = 2 if k % B == 0 else 0.5
        ax.axhline(k - 0.5, color="black", lw=lw)
        ax.axvline(k - 0.5, color="black", lw=lw)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def plot_bars(ax, labels, values, title: str, ylabel: str, log: bool = False, errors=None,
              colors=None) -> None:
    """Simple labelled bar chart (value written above each bar)."""
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=3, color=colors)
    ax.set_xticks(x, labels=labels, fontsize=8)
    if log:
        ax.set_yscale("symlog")
    for xi, v in zip(x, values):
        ax.text(xi, v, f"{v:,.0f}" if float(v).is_integer() or v >= 10 else f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)


def grouped_bars(ax, groups, series: dict, title: str, ylabel: str, log: bool = False,
                 colors: dict | None = None, label_values: bool = False) -> None:
    """Bars for several series side by side per group, with std error bars.

    series: {name: (means, stds)} with one value per group.
    log:    log y axis; bars with mean 0 are not drawn (log scale cannot show 0).
    Lower error bars are clipped so they never go below 0 (counts and rates are non-negative).
    """
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    for j, (name, (means, stds)) in enumerate(series.items()):
        pos = x - 0.4 + width * (j + 0.5)
        means, stds = np.asarray(means, float), np.asarray(stds, float)
        lower = np.minimum(stds, means * 0.9) if log else np.minimum(stds, means)  # never below 0
        yerr = np.vstack([lower, stds])
        ax.bar(pos, means, width, yerr=yerr, capsize=2, label=name,
               color=(colors or {}).get(name))
        if label_values:
            for p, m in zip(pos, means):
                ax.text(p, m, f"{m:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, labels=[str(g) for g in groups])
    if log:
        ax.set_yscale("log", nonpositive="clip")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
