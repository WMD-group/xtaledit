"""Shared plotting style for notebooks."""

import matplotlib.pyplot as plt
import seaborn as sns

GRAY = "#9FA0A0"
BLACK = "#000000"
WHITE = "#FFFFFF"

PALETTE = [
    "#316745",
    "#F39800",
    "#2CA9E1",
    "#c53d43",
    "#B8D200",
    "#19448E",
    "#884898",
    GRAY,
    "#E597B2",
]

CATEGORY_COLORS = {
    "1": "#316745",
    "2-1": "#F39800",
    "2-2": "#F39800",
    "2-3": "#2CA9E1",
    "3": GRAY,
}


def apply_plot_style() -> None:
    """Apply the shared notebook plotting style."""
    plt.rcParams["font.family"] = "cmr10"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    sns.set_palette(PALETTE)
