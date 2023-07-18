############## Imports Packages ##############
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{cmbright}",
            ]
        ),
    }
)

from utils import *
import itertools
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnchoredText
from sklearn.utils import shuffle

#############################################


def plot_hist2d(value, canal, idx, dyn, name):
    canals = list(itertools.combinations(canal, 2))

    f, ax = plt.subplots(
        2, 6, figsize=(5 * 8.5 / 2.54, 5 * 3 / 2.54), sharex=True, sharey=True
    )
    b = 80
    print(canals)
    for n, chanel in enumerate(canals):
        indice = canal.index(chanel[0]), canal.index(chanel[1])

        w = ax[0, n].hist2d(
            value[indice[0]][idx],
            value[indice[1]][idx],
            bins=b,
            cmap="RdBu_r",
            alpha=0.65,
            range=[dyn, dyn],
            norm=LogNorm(1, 600),
            rasterized=True,
        )
        d = ax[1, n].hist2d(
            value[indice[0]][~idx],
            value[indice[1]][~idx],
            bins=b,
            cmap="RdBu_r",
            alpha=0.65,
            range=[dyn, dyn],
            norm=LogNorm(1, 600),
            rasterized=True,
        )

        at = AnchoredText(
            "wet",
            prop=dict(size=10, fontweight="bold"),
            frameon=True,
            loc="lower right",
        )
        at.patch.set(linewidth=0.5, edgecolor="gray", alpha=0.5)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[0, n].add_artist(at)

        at = AnchoredText(
            "dry",
            prop=dict(size=10, fontweight="bold"),
            frameon=True,
            loc="lower right",
        )
        at.patch.set(linewidth=0.5, edgecolor="gray", alpha=0.5)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[1, n].add_artist(at)
        ax[1, n].set_xlabel(chanel[0], fontsize=12, fontweight="bold")
        ax[1, n].set_ylabel(chanel[1], fontsize=12, fontweight="bold")
        ax[0, n].set_xlabel(chanel[0], fontsize=12, fontweight="bold")
        ax[0, n].set_ylabel(chanel[1], fontsize=12, fontweight="bold")

    new_ax = f.add_axes([0.91, 0.2, 0.01, 0.6])
    plt.colorbar(w[3], cax=new_ax)
    # plt.tight_layout()
    exist_create_folder("fig")
    plt.savefig(f"fig/{name}.pdf", backend="pgf", dpi=150)
