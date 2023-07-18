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
#############################################
from utils import *


def correlate_dataframe(dftel, dftmin, threshold=0.5):
    df1 = dftel.dropna().copy()
    df2 = dftmin.dropna().copy()
    m = np.where(df1.tel.values > threshold, 1, 0)
    n = np.where(df2.tmin.values > 0, 1, 0)
    df1.loc[:, "tel"] = m
    df2.loc[:, "tmin"] = n
    df = pd.concat([df1, df2.loc[:, "tmin"]], axis=1, join="inner")
    return df


def correlation_tel_tmin(tel, ts, thresholds, save=True):
    corr = []
    for thresh in tqdm(thresholds):
        m = []
        df_full_tel = pd.DataFrame()
        for i in tel.keys():
            m.extend(
                Parallel(n_jobs=4)(
                    delayed(correlate_dataframe)(tel[i][j], ts[i][j], thresh)
                    for j in tel[i].keys()
                )
            )
        df_full_tel = pd.concat(m, axis=0)
        corr.append(df_full_tel.corr(method="spearman").iloc[0, 1])
        del m, df_full_tel
    if save:
        exist_create_folder("data")
        dump_pkl([corr, thresholds], "data/corr_tel_tmin.pkl")
    return corr


def plot_tn_two_massifs(M1, M2, ts, hs, name):
    fig, ax = plt.subplots(2, 1, figsize=(2 * 8.5 / 2.54, 2 * 6 / 2.54))
    ax1 = (
        ts[M1][(2700, 20, 180)]
        .loc["2020-07":"2021-08"]
        .tmin.plot(fontsize=15, subplots=True, ax=ax[0], sharex=True, sharey=True)
    )
    d = hs[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].hs > 0.4

    ts[M2][(2700, 20, 180)].loc["2020-07":"2021-08"].tmin.plot(
        fontsize=15,
        subplots=True,
        ax=ax[0],
        sharex=True,
        sharey=True,
        color="tab:orange",
    )
    ax1[0].set_xlabel("Time", fontsize=15)
    ax1[0].set_ylabel("Temperature (°C)", fontsize=15)
    ax1[0].set_title(
        "$T_n$ at 2700m, $\\theta$=180°, $\\nabla$=20°", fontsize=15, fontweight="bold"
    )

    ax[0].vlines(
        hs[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].index[d],
        -50,
        10,
        color="black",
        label="Hs",
        zorder=0,
        alpha=0.5,
        linewidth=0.125,
    )
    # ax[0].scatter(hs[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].hs[d].index, ts[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].tmin[d], color="black", label="Hs",s=1, alpha=0.5,marker="x",zorder=10)
    ax1[0].set_ylim(-30, 2)
    ax1[0].grid()
    lbl = [M1[0] + M1[1:].lower(), M2[0] + M2[1:].lower(), "Snow Heigh >0.4m"]
    ax1[0].legend(lbl, fontsize=11, loc="lower right")

    ax2 = (
        ts[M1][(1800, 20, 180)]
        .loc["2020-07":"2021-08"]
        .tmin.plot(fontsize=15, subplots=True, ax=ax[1], sharex=True, sharey=True)
    )
    ts[M2][(1800, 20, 180)].loc["2020-07":"2021-08"].tmin.plot(
        fontsize=15,
        subplots=True,
        ax=ax[1],
        sharex=True,
        sharey=True,
        color="tab:orange",
    )
    ax2[0].set_xlabel("Time", fontsize=15)
    ax2[0].set_ylabel("Temperature (°C)", fontsize=15)
    ax2[0].set_title(
        "$T_n$ at 1800m, $\\theta$=180°, $\\nabla$=20°", fontsize=15, fontweight="bold"
    )
    d = hs[M2][(2700, 20, 180)].loc["2020-07":"2021-08"].hs > 0.4
    ax[1].vlines(
        hs[M2][(2700, 20, 180)].loc["2020-07":"2021-08"].index[d],
        -50,
        10,
        color="black",
        label="Hs",
        zorder=0,
        alpha=0.5,
        linewidth=0.125,
    )

    ax2[0].set_ylim(-30, 2)
    ax2[0].grid()

    ax2[0].legend(lbl, fontsize=11, loc="lower right")

    plt.tight_layout()
    exist_create_folder("fig")
    plt.savefig(f"fig/{name}.pdf")


#### new form
def plot_tn_two_massifs(M1, M2, ts, hs, name):
    fig, ax = plt.subplots(2, 2, figsize=(4 * 8.5 / 2.54, 2 * 6 / 2.54))
    name_mass = [M1[0] + M1[1:].lower(), M2[0] + M2[1:].lower()]
    ax1 = (
        ts[M1][(2700, 20, 180)]
        .loc["2020-07":"2021-08"]
        .tmin.plot(fontsize=15, subplots=True, ax=ax[0][0], sharex=True, sharey=True)
    )
    d1 = (hs[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].hs > 0.4) & (
        ts[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].tmin > 0
    )
    d2 = (hs[M2][(2700, 20, 180)].loc["2020-07":"2021-08"].hs > 0.4) & (
        ts[M2][(2700, 20, 180)].loc["2020-07":"2021-08"].tmin > 0
    )
    ts[M2][(2700, 20, 180)].loc["2020-07":"2021-08"].tmin.plot(
        fontsize=15,
        subplots=True,
        ax=ax[0][1],
        sharex=True,
        sharey=True,
        color="tab:orange",
    )
    for i in range(2):
        ax[0][i].set_xlabel("Time", fontsize=15)
        ax[0][i].set_ylabel("Temperature (°C)", fontsize=15)
        ax[0][i].set_title(
            "$T_n$ at 2700m, $\\theta$=180°, $\\nabla$=20°",
            fontsize=15,
            fontweight="bold",
        )

    ax[0][0].vlines(
        hs[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].index[d1],
        -50,
        10,
        color="black",
        label="Hs",
        zorder=0,
        alpha=0.5,
        linewidth=0.125,
    )
    ax[0][1].vlines(
        hs[M2][(2700, 20, 180)].loc["2020-07":"2021-08"].index[d2],
        -50,
        10,
        color="black",
        label="Hs",
        zorder=0,
        alpha=0.5,
        linewidth=0.125,
    )

    # ax[0].scatter(hs[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].hs[d].index, ts[M1][(2700, 20, 180)].loc["2020-07":"2021-08"].tmin[d], color="black", label="Hs",s=1, alpha=0.5,marker="x",zorder=10)
    for i in range(2):
        ax[0][i].set_ylim(-30, 2)
        ax[0][i].grid()
        lbl = [name_mass[i], "Labels"]
        ax[0][i].legend(lbl, fontsize=11, loc="lower left")

    ax2 = (
        ts[M1][(1800, 20, 180)]
        .loc["2020-07":"2021-08"]
        .tmin.plot(fontsize=15, subplots=True, ax=ax[1][0], sharex=True, sharey=True)
    )
    ts[M2][(1800, 20, 180)].loc["2020-07":"2021-08"].tmin.plot(
        fontsize=15,
        subplots=True,
        ax=ax[1][1],
        sharex=True,
        sharey=True,
        color="tab:orange",
    )
    for i in range(2):
        ax[1][i].set_xlabel("Time", fontsize=15)
        ax[1][i].set_ylabel("Temperature (°C)", fontsize=15)
        ax[1][i].set_title(
            "$T_n$ at 1800m, $\\theta$=180°, $\\nabla$=20°",
            fontsize=15,
            fontweight="bold",
        )

    d3 = (hs[M1][(1800, 20, 180)].loc["2020-07":"2021-08"].hs > 0.4) & (
        ts[M1][(1800, 20, 180)].loc["2020-07":"2021-08"].tmin > 0
    )
    d4 = (hs[M2][(1800, 20, 180)].loc["2020-07":"2021-08"].hs > 0.4) & (
        ts[M2][(1800, 20, 180)].loc["2020-07":"2021-08"].tmin > 0
    )

    ax[1][0].vlines(
        hs[M1][(1800, 20, 180)].loc["2020-07":"2021-08"].index[d3],
        -50,
        10,
        color="black",
        label="Hs",
        zorder=0,
        alpha=0.5,
        linewidth=0.125,
    )
    ax[1][1].vlines(
        hs[M2][(1800, 20, 180)].loc["2020-07":"2021-08"].index[d4],
        -50,
        10,
        color="black",
        label="Hs",
        zorder=0,
        alpha=0.5,
        linewidth=0.125,
    )

    for i in range(2):
        lbl = [name_mass[i], "Labels"]
        ax[1][i].set_ylim(-30, 2)
        ax[1][i].grid()
        ax[1][i].legend(lbl, fontsize=11, loc="lower left")

    plt.tight_layout()
    exist_create_folder("fig")
    plt.savefig(f"fig/{name}.pdf", backend="pgf", transparent=True)


def plot_tel_two_massifs(M1, M2, tel, name):
    fig, ax = plt.subplots(2, 1, figsize=(2 * 8.5 / 2.54, 2 * 6 / 2.54))
    ax1 = (
        tel[M1][(2700, 20, 180)]
        .loc["2020-07":"2021-08"]
        .plot(fontsize=15, subplots=True, ax=ax[0], sharex=True, sharey=True)
    )
    tel[M2][(2700, 20, 180)].loc["2020-07":"2021-08"].plot(
        fontsize=15,
        subplots=True,
        ax=ax[0],
        sharex=True,
        sharey=True,
        color="tab:orange",
    )
    ax1[0].set_xlabel("Time", fontsize=15)
    ax1[0].set_ylabel("LWC ($kg/m^2$)", fontsize=15)
    ax1[0].set_title(
        "LWC at 2700m, $\\theta$=180°, $\\nabla$=20°", fontsize=15, fontweight="bold"
    )
    ax1[0].set_ylim(0, 95)
    ax1[0].grid()
    lbl = [M1[0] + M1[1:].lower(), M2[0] + M2[1:].lower()]
    ax1[0].legend(lbl, fontsize=12)

    ax2 = (
        tel[M1][(1800, 20, 180)]
        .loc["2020-07":"2021-08"]
        .plot(fontsize=15, subplots=True, ax=ax[1], sharex=True, sharey=True)
    )
    tel[M2][(1800, 20, 180)].loc["2020-07":"2021-08"].plot(
        fontsize=15,
        subplots=True,
        ax=ax[1],
        sharex=True,
        sharey=True,
        color="tab:orange",
    )
    ax2[0].set_xlabel("Time", fontsize=15)
    ax2[0].set_ylabel("LWC ($kg/m^2$)", fontsize=15)
    ax2[0].set_title(
        "LWC at 1800m, $\\theta$=180°, $\\nabla$=20°", fontsize=15, fontweight="bold"
    )
    ax2[0].set_ylim(0, 95)
    ax2[0].grid()

    ax2[0].legend(lbl, fontsize=12, loc="upper left")

    plt.tight_layout()
    exist_create_folder("fig")
    plt.savefig(f"fig/{name}.pdf")


def plot_correlation_tn_tel(thresh, corr_spear, name):
    f, ax = plt.subplots(1, 1, figsize=(2 * 8.5 / 2.54, 2 * 3 / 2.54))
    ax.semilogx(
        thresh,
        corr_spear,
        "+",
        label="Spearman",
        color="black",
    )
    ax.set_xlabel("LWC threshold", fontsize=15)
    ax.set_ylabel("Correlation", fontsize=15)
    ax.set_title("Correlation between LWC and $T_n$", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="lower left")
    ax.grid()
    plt.tight_layout()
    exist_create_folder("fig")
    plt.savefig(f"fig/{name}.pdf")
