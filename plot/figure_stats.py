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


def open_log(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()
    return lines


def find_band(dicband, band):
    for key in dicband:
        if np.all(band == dicband[key]):
            return key
    return None


def convert_str_to_list(s):
    return [int(e) for e in s.strip("][").split(", ")]


def sort_dic(dic):
    return {k: dic[k] for k in sorted(dic)}


def parse_f1_score_log(f, dicband):
    f1_score = {"M": [], "N": [], "O": [], "L": []}
    for i in f:
        try:
            id_stat = i.split("_")[-3]
        except:
            id_stat = ""
        idx = find_band(dicband, id_stat)
        if idx != None:
            if "F1 score" in i:
                f1 = np.float64(i.split("test ")[1][:-1])
                f1_score[idx].append(f1)
    return f1_score


def plot_boxplot_channel(f1score, name):
    f, ax = plt.subplots(1, 1, figsize=(2 * 8.5 / 2.54, 2 * 4 / 2.54))
    t = list(f1score.keys())
    x = list(f1score.values())
    # t = [t[i] for i in range(len(t))]
    t = ["$\mathbf{" + f"{t[i].upper()}" + "}$" for i in range(len(t))]
    ax.grid(True, linestyle="--", linewidth=8 * 0.005, zorder=0, alpha=0.75)
    bp1 = ax.boxplot(
        x,
        labels=t,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgrey", color="black", alpha=0.5),
        medianprops=dict(color="black"),
        positions=np.arange(0, len(t), 1),
    )
    ax.set_xlabel("Combination auxilary data", fontsize=14)
    ax.set_ylabel("F1-score", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylim(0.55, 0.82)
    ax.legend([bp1["boxes"][0]], ["F1-score"], loc="lower right")
    plt.tight_layout()
    exist_create_folder("fig/")
    plt.savefig(f"fig/{name}.pdf", backend="pgf")
