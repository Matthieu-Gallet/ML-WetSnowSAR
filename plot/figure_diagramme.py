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
from matplotlib.colors import LogNorm

from geo_tools import load_data, count_value_pen_ori_alti
from joblib import Parallel, delayed

#############################################


def evaluate_dataset(
    input_path,
    pente,
    step_alt,
    step_or,
):
    refd = glob.glob(input_path + "*.tif")[0]
    org, _ = load_data(refd)
    print(org.max(axis=(0, 1)))
    gradient = org[:, :, 1]
    dem = org[:, :, 2]
    theta = org[:, :, 0]
    alti_max = dem.max()
    alti_min = dem[dem > 0].min() - step_alt
    orientation = np.arange(-step_or, 360, step_or) % 360
    Parallel(n_jobs=8)(
        delayed(wrap_eval)(
            file,
            gradient,
            dem,
            theta,
            orientation,
            pente,
            alti_min,
            alti_max,
            step_alt,
            input_path,
        )
        for file in glob.glob(input_path + "*/*GR*.tif")
    )

    return 1


def wrap_eval(
    file,
    gradient,
    dem,
    theta,
    orientation,
    pente,
    alti_min,
    alti_max,
    step_alt,
    input_path,
):
    name = basename(file).split("_")[0] + "_" + basename(file).split("_")[3]
    img, _ = load_data(file)
    img = np.where(img < 0, np.nan, img)
    print(img.min(), img.max())
    print(np.nanmax(img), np.nanmin(img))
    prediction = prepare_prediction(img)
    print(np.nanmax(prediction), np.nanmin(prediction))
    eval_results_diagramme(
        prediction,
        gradient,
        dem,
        theta,
        orientation,
        pente,
        alti_min,
        alti_max,
        step_alt,
        name,
        input_path,
    )


def prepare_prediction(result_map):
    if np.nanmax(result_map) > 1:
        print("scale image")
        prediction = scale_image(result_map)
    else:
        print("no scale")
        prediction = result_map
    return prediction


def eval_results_diagramme(
    prediction,
    gradient,
    dem,
    theta,
    orientation,
    pente,
    alti_min,
    alti_max,
    step_alt,
    name,
    input_path,
):
    d = {}
    alt_p = count_value_pen_ori_alti(
        prediction,
        gradient,
        dem,
        theta,
        orientation,
        pente,
        alti_min,
        alti_max,
        step_alt,
    )
    # dump_pkl(alt_p, name + ".pkl")
    d[name] = [
        np.arange(alti_min, alti_max, step_alt),
        orientation,
        alt_p[1:, 1:, :],
        pente,
        prediction.size,
    ]
    dump_pkl(d, input_path + f"{name}_diagramme.pkl")
    return 1


def plot_diagramme_4prod(dico, figname):
    fig, ax1 = plt.subplots(1, 4, subplot_kw={"projection": "polar"}, figsize=(30, 6))

    at, rt = [], []
    print(dico.keys())
    for i in ["FCROC", "BAROC", "SWS"]:
        r, _, alt, _, lengt = dico[i]
        at.append(alt)
        rt.append(r)
    at = np.array(at)
    alt_max = np.max(at)
    alt_min = at[at > 0].min()
    r_max = np.max(rt)
    r_min = np.min(rt)
    alt_max = alt_max * 100 / lengt
    alt_min = alt_min * 100 / lengt

    for i, name in enumerate(["FCROC", "BAROC", "SWS"]):
        r, th, alt, _, lengt = dico[name]
        th = np.radians(th)
        print(i, alt.shape, r.shape, th.shape)
        v = alt[:, :, 0] * 100 / lengt
        ax1[i].grid(False)
        # m = ax1[i].pcolormesh(
        #     th, r, v.T, cmap="terrain_r",  shading="flat",vmin=0, vmax=alt_max,norm=LogNorm(alt_min, alt_max)
        # )
        # m = ax1[i].pcolormesh(
        #     th, r, v.T,  shading="flat",cmap="terrain_r", vmin=0, vmax=alt_max
        # )
        m = ax1[i].pcolormesh(
            th,
            r,
            v.T,
            shading="flat",
            cmap="terrain",
            norm=LogNorm(alt_min * 100, alt_max),
            rasterized=True,
        )
        ax1[i].set_theta_zero_location("N")
        ax1[i].set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
        ax1[i].set_xticklabels(
            ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            fontsize=15,
            fontweight="bold",
            # position=(0.21, 0.1),
        )
        ax1[i].set_yticks(np.arange(0, r_max, 600))
        ax1[i].set_yticklabels(
            np.arange(0, r_max, 600, dtype=int),
            fontsize=14,
            fontweight="bold",
            color="red",
        )
        ax1[i].set_rlim(r_min, 1.1 * r_max)
        ax1[i].set_rlabel_position(157.5)
        ax1[i].grid(True)
        ax1[i].set_theta_direction(-1)
        ax1[i].set_title(f"{name}", va="bottom", fontsize=18, fontweight="bold")
        ax1[i].invert_yaxis()

    fig.subplots_adjust(bottom=-0.175)
    cbar_ax = fig.add_axes([0.15, -0.15, 0.475, 0.05])

    p = fig.colorbar(m, cax=cbar_ax, orientation="horizontal")
    p.ax.tick_params(labelsize=15)
    p.set_label("% wet pixels", fontsize=20, fontweight="bold")
    i = 3
    name = "FSC"
    r, th, alt, _, lengt = dico[name]
    print(alt.min(), alt.max())
    th = np.radians(th)
    print(i, alt.shape, r.shape, th.shape)
    v = alt[:, :, 0] * 100 / lengt
    ax1[i].grid(False)
    dm = ax1[i].pcolormesh(
        th,
        r,
        v.T,
        shading="flat",
        cmap="terrain",
        norm=LogNorm(alt_min * 100, alt_max),
        rasterized=True,
    )
    # (alt[alt>0].min()*100/lengt, alt.max()*100/lengt))
    ax1[i].set_theta_zero_location("N")
    ax1[i].set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
    ax1[i].set_xticklabels(
        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        fontsize=13,
        fontweight="bold",
        # position=(0.21, 0.1),
    )
    ax1[i].set_yticks(np.arange(0, r_max, 600))
    ax1[i].set_yticklabels(
        np.arange(0, r_max, 600, dtype=int), fontsize=14, fontweight="bold", color="red"
    )
    ax1[i].set_rlim(r_min, 1.1 * r_max)
    ax1[i].set_rlabel_position(157.5)
    ax1[i].grid(True)
    ax1[i].set_theta_direction(-1)
    ax1[i].set_title(f"{name}", va="bottom", fontsize=18, fontweight="bold")
    ax1[i].invert_yaxis()
    fig.subplots_adjust(right=0.83)
    cbar_ax2 = fig.add_axes([0.85, 0.0, 0.01, 0.7])
    q = fig.colorbar(dm, cax=cbar_ax2, orientation="vertical")
    q.ax.tick_params(labelsize=15)
    q.set_label("% snow cover", fontsize=20, fontweight="bold")
    exist_create_folder("fig")
    plt.savefig(
        f"fig/{figname}.pdf", bbox_inches="tight", pad_inches=0.1, backend="pgf"
    )
