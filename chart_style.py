# chart_style.py
#
# What it does: holds the colours and the tidy-up step shared by every chart
#   in the project, so that all charts look like one family.
# What it reads: nothing.
# What it produces: nothing on disk.
# Which files use it: none yet. The validation charts of Phase 2 will.

import matplotlib

matplotlib.use("Agg")  # draw to files, never open a window
import matplotlib.pyplot as plt

# One fixed colour per thing, used the same way in every chart.
BLUE = "#2a78d6"  # the model
ORANGE = "#eb6834"  # a second series
AQUA = "#1baf7a"  # a third series
BASELINE = "#898781"  # whatever the model is being compared against
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECOND_INK = "#52514e"
GRID = "#e6e5e0"


def new_figure(columns: int = 1, width: float = 7.0, height: float = 4.5, **options):
    """A figure with the project's background and text colours."""
    plt.rcParams.update({
        "font.size": 10, "text.color": INK, "axes.labelcolor": SECOND_INK,
        "xtick.color": SECOND_INK, "ytick.color": SECOND_INK, "axes.edgecolor": GRID,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    figure, axes = plt.subplots(1, columns, figsize=(width, height), **options)
    return figure, axes


def tidy(axes, title: str, grid_axis: str = "x"):
    """Left aligned title, light grid lines behind the marks, no box."""
    axes.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(length=0)


def save(figure, file):
    figure.tight_layout()
    file.parent.mkdir(exist_ok=True)
    figure.savefig(file, dpi=130)
    plt.close(figure)
    print(f"Chart saved to {file}")
