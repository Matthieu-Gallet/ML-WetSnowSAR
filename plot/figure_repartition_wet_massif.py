############## Imports Packages ##############
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.dates as mdates
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

import pandas as pd
import numpy as np
from utils.files_management import open_pkl

#############################################


def prepare_data(path):
    label_array = open_pkl(os.path.join(path, "label_array.pkl"))
    name_array = open_pkl(os.path.join(path, "name_array.pkl"))
    namdat = open_pkl(os.path.join(path, "namdat.pkl"))
    label_array = np.array(label_array).astype(str)
    name_array = np.array(name_array).astype(str)
    namdat = np.array(namdat)
    name_mass = np.array([i.split("_")[0] for i in name_array])
    carac = pd.DataFrame(
        [name_mass, namdat, label_array], index=["mass", "date", "label"]
    ).T
    return carac


def plot_carac_wet_massif(carac, name):
    f, ax = plt.subplots(
        2, 1, figsize=(2 * 8 / 2.54, 2 * 5.5 / 2.54), sharex=True, layout="constrained"
    )
    # f, ax = plt.subplots(1,1,figsize=(10,4))
    t = pd.DataFrame(
        carac.loc[carac.label == "wet"].groupby("date").indices.keys(), columns=["date"]
    )
    t["bottom"] = np.zeros_like(t.date).astype(int)
    t.set_index("date", inplace=True)
    ax[0].grid(True, zorder=0, linewidth=0.75)

    for m in carac.mass.unique():
        q = pd.DataFrame(
            carac.loc[(carac["label"] == "wet") & (carac["mass"] == m)][
                "date"
            ].value_counts()
        ).sort_index()
        labl = m[0] + m[1:].lower()
        ax[0].bar(
            q.index,
            q["count"],
            width=5,
            align="edge",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.75,
            label=labl,
            bottom=t.loc[q.index, "bottom"].values,
            zorder=3,
        )
        g = pd.merge(t, q, on="date", how="left").sort_index()
        t["bottom"] = np.nansum([g["bottom"], g["count"]], axis=0)

    locator = mdates.AutoDateLocator()  # minticks=12, maxticks=24)
    formatter = mdates.ConciseDateFormatter(locator)
    ax[0].xaxis.set_major_locator(locator)
    ax[0].xaxis.set_major_formatter(formatter)
    # [0]ax.legend()
    ax[0].set_xlabel("date")
    ax[0].set_ylabel("wet samples")
    ax[0].set_title("Number of wet samples per date")

    t = pd.DataFrame(
        carac.loc[carac.label != "wet"].groupby("date").indices.keys(), columns=["date"]
    )
    t["bottom"] = np.zeros_like(t.date).astype(int)
    t.set_index("date", inplace=True)
    ax[1].grid(True, zorder=0, linewidth=0.75)

    for m in carac.mass.unique():
        q = pd.DataFrame(
            carac.loc[(carac["label"] != "wet") & (carac["mass"] == m)][
                "date"
            ].value_counts()
        ).sort_index()
        ax[1].bar(
            q.index,
            q["count"],
            width=5,
            align="edge",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.75,
            bottom=t.loc[q.index, "bottom"].values,
            zorder=3,
        )
        g = pd.merge(t, q, on="date", how="left").sort_index()
        t["bottom"] = np.nansum([g["bottom"], g["count"]], axis=0)

    # ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.DateFormatter("%b-%y-%d")))
    locator = mdates.AutoDateLocator(minticks=6, maxticks=24)
    formatter = mdates.ConciseDateFormatter(locator)
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    # ax.set_xticklabels(rotation=45,ha="right")
    # ax[1].legend()
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("non-wet samples")
    ax[1].set_title("Number of non-wet samples per date")
    f.legend(
        loc="outside right center",
        title="Massif",
        title_fontsize=12,
    )
    os.makedirs("../data/fig/", exist_ok=True)
    plt.savefig(f"fig/{name}.pdf", backend="pgf")
