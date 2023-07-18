############## Imports Packages ##############
import sys, os, glob

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

from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from utils.files_management import open_pkl
from utils.metrics import BAROC, FRCROC

# #############################################


def extract_mean_pred(dtrain):
    y_true = []
    y_est = []
    for folds in dtrain.keys():
        a = dtrain[folds]["y_true"]
        b = dtrain[folds]["y_est"]
        y_true.extend(a)
        y_est.extend(b)

    am, tm = BAROC(y_true, y_est)
    ac, tc = FRCROC(y_true, y_est, rate=0.05)
    # print(np.min(y_est), np.max(y_est))
    return [am, tm], [ac, tc], y_true, y_est


def prepare_ROC(in_path):
    path_test = in_path + "*/*/prob*est*pkl"
    path_thres = in_path + "*/*/*train*pkl"
    nagler_path = in_path + "*nagler*pkl"

    dic_thresh = {}
    for i in glob.glob(path_thres):
        ID_m = i.split("/")[2][-1].upper()
        pred = open_pkl(i)
        [am, tm], [ac, tc], _, _ = extract_mean_pred(pred)
        print(
            ID_m,
            "Accuracy max",
            am,
            "threshold: ",
            tm,
            "FPR constant",
            ac,
            "threshold: ",
            tc,
        )
        dic_thresh[ID_m] = {"thresh_max": tm, "thresh_con": tc}
    dic_test = {}
    for i in glob.glob(path_test):
        ID_m = i.split("/")[2][-1].upper()
        pred = open_pkl(i)
        _, _, y_true, y_est = extract_mean_pred(pred)
        dic_test[ID_m] = {
            "y_true": y_true,
            "y_est": y_est,
            "thresh_max": dic_thresh[ID_m]["thresh_max"],
            "thresh_con": dic_thresh[ID_m]["thresh_con"],
        }
    nagler = open_pkl(glob.glob(nagler_path)[0])
    nagler_VV = {k: v for k, v in nagler.items() if "VV" in k}
    nagler_VH = {k: v for k, v in nagler.items() if "VH" in k}

    [am, tm], [ac, tc], y_true, y_est = extract_mean_pred(nagler_VV)
    print(am, tm)
    print(ac, tc)
    dic_test["Nagler_VV"] = {
        "y_true": y_true,
        "y_est": y_est,
        "thresh_max": tm,
        "thresh_con": tc,
    }

    [am, tm], [ac, tc], y_true, y_est = extract_mean_pred(nagler_VH)
    dic_test["Nagler_VH"] = {
        "y_true": y_true,
        "y_est": y_est,
        "thresh_max": tm,
        "thresh_con": tc,
    }
    print(am, tm)
    print(ac, tc)
    try:
        stat = open_pkl(in_path + "stat.pkl")
        vvha = np.array(stat["max"])[:, [6, 7]]
        vvhi = np.array(stat["min"])[:, [6, 7]]
        vh = vvhi[vvhi[:, 1] > -998, 1].min(), vvha[:, 1].max()
        vv = vvhi[vvhi[:, 0] > -998, 0].min(), vvha[:, 0].max()
        print("VV min max: ", vv, "VH min max: ", vh)
        thmax = (1 - dic_test["Nagler_VV"]["thresh_max"]) * (vv[1] - vv[0]) + vv[0]
        print("Nagler VV threshold max accuracy dB: ", thmax)
        thmax2 = (1 - dic_test["Nagler_VH"]["thresh_max"]) * (vh[1] - vh[0]) + vh[0]
        print("Nagler VH threshold max accuracy dB: ", thmax2)
        thcon = (1 - dic_test["Nagler_VV"]["thresh_con"]) * (vv[1] - vv[0]) + vv[0]
        print("Nagler VV threshold constant FPR dB: ", thcon)
        thcon2 = (1 - dic_test["Nagler_VH"]["thresh_con"]) * (vh[1] - vh[0]) + vh[0]
        print("Nagler VH threshold constant FPR dB: ", thcon2)
        with open(
            in_path + f"info{os.path.basename(glob.glob(nagler_path)[0])}.txt", "w"
        ) as f:
            f.write(f"Nagler VV threshold max accuracy dB: {thmax}\n")
            f.write(f"Nagler VH threshold max accuracy dB: {thmax2}\n")
            f.write(f"Nagler VV threshold constant FPR dB: {thcon}\n")
            f.write(f"Nagler VH threshold constant FPR dB: {thcon2}\n")
    except:
        print("No stat file")
    return dic_test


def plot_roc_multi(dic_test, name):
    f, ax = plt.subplots(
        1, 2, figsize=(4 * 8.5 / 2.54, 2 * 5 / 2.54), sharey=True, sharex=True
    )
    bckp_metric = {}
    c, d = 0, 0
    for i in dic_test.keys():
        if not ("Nagler" in i):
            a = dic_test[i]["y_true"]
            b = dic_test[i]["y_est"]
            auc_score = roc_auc_score(a, b)
            fpr, tpr, thresholds = roc_curve(a, b)

            xmax, ymax = (
                fpr[dic_test[i]["thresh_max"] <= thresholds][-1],
                tpr[dic_test[i]["thresh_max"] <= thresholds][-1],
            )
            xcon, ycon = (
                fpr[dic_test[i]["thresh_con"] >= thresholds][0],
                tpr[dic_test[i]["thresh_con"] >= thresholds][0],
            )
            ax[0].plot(
                fpr,
                tpr,
                label="$\mathbf{"
                + f"{i.upper()}"
                + "}$"
                + " - ROC (area = %0.2f)" % auc_score,
            )
            ax[0].set_xlabel("False Positive Rate", fontsize=14)
            ax[0].set_ylabel("True Positive Rate", fontsize=14)
            ax[0].grid(True, linestyle="--", linewidth=8 * 0.051, zorder=0)
            ax[0].hlines(
                ymax,
                -10,
                xmax,
                linestyles="--",
                color="black",
                alpha=0.75,
                linewidth=0.5,
                zorder=10,
            )

            ax[0].hlines(
                ycon,  # -adj[c],
                -10,
                xcon,
                linestyles="--",
                color="black",
                alpha=0.75,
                linewidth=0.5,
                zorder=10,
            )

            if c == 3:
                ax[0].scatter(
                    xmax,
                    ymax,
                    label="Best accuracy",
                    marker="x",
                    color="black",
                    zorder=10,
                )
                ax[0].scatter(
                    xcon,
                    ycon,  # -adj[c],
                    label="FPR = 5\%",
                    marker="+",
                    color="black",
                    zorder=10,
                )

            else:
                ax[0].scatter(
                    xmax,
                    ymax,
                    marker="x",
                    color="black",
                    zorder=10,
                )
                ax[0].scatter(
                    xcon,
                    ycon,  # -adj[c],
                    marker="+",
                    color="black",
                    zorder=10,
                )
            c += 1
        else:
            a = dic_test[i]["y_true"]
            b = dic_test[i]["y_est"]
            auc_score = roc_auc_score(a, b)
            fpr, tpr, thresholds = roc_curve(a, b)

            xmax, ymax = (
                fpr[dic_test[i]["thresh_max"] <= thresholds][-1],
                tpr[dic_test[i]["thresh_max"] <= thresholds][-1],
            )
            xcon, ycon = (
                fpr[dic_test[i]["thresh_con"] >= thresholds][0],
                tpr[dic_test[i]["thresh_con"] >= thresholds][0],
            )
            ax[1].plot(fpr, tpr, label=f"{i} - ROC (area = %0.2f)" % auc_score)
            ax[1].set_xlabel("False Positive Rate", fontsize=14)
            ax[1].set_ylabel("True Positive Rate", fontsize=14)
            ax[1].grid(True, linestyle="--", linewidth=8 * 0.051, zorder=0)
            ax[1].hlines(
                ymax,
                -10,
                xmax,
                linestyles="--",
                color="black",
                alpha=0.75,
                linewidth=0.5,
                zorder=10,
            )

            ax[1].hlines(
                ycon,  # -adj[c],
                -10,
                xcon,
                linestyles="--",
                color="black",
                alpha=0.75,
                linewidth=0.5,
                zorder=10,
            )

            if d > 0:
                ax[1].scatter(
                    xmax,
                    ymax,
                    label="Best accuracy",
                    marker="x",
                    color="black",
                    zorder=10,
                )
                ax[1].scatter(
                    xcon,
                    ycon,  # -adj[c],
                    label="FPR = 5\%",
                    marker="+",
                    color="black",
                    zorder=10,
                )
            else:
                ax[1].scatter(
                    xmax,
                    ymax,
                    marker="x",
                    color="black",
                    zorder=10,
                )
                ax[1].scatter(
                    xcon,
                    ycon,  # -adj[c],
                    marker="+",
                    color="black",
                    zorder=10,
                )
            d += 1
    xt = [0, 0.05] + list(np.arange(0.1, 1.1, 0.1).round(2))
    ax[0].set_xticks(xt)
    ax[1].set_xticks(xt)
    ax[0].set_xticklabels(xt)
    ax[1].set_xticklabels(xt)
    ax[0].legend(loc="lower right", fontsize=14)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc="lower right", fontsize=14)
    plt.tight_layout()
    os.makedirs("../data/fig/", exist_ok=True)
    plt.savefig(f"fig/{name}.pdf", backend="pgf")


def ROC_plot(y, path_save):
    bckp_metric = {}
    f, ax = plt.subplots(1, figsize=(12, 12))
    ax.plot([0, 1], [0, 1], "--", color="black")
    for i in y.keys():
        a = y[i]["y_true"]
        b = y[i]["y_est"]
        auc_score = roc_auc_score(a, b)
        fpr, tpr, thresholds = roc_curve(a, b)
        num_pos_class = a[a > 0].sum()
        num_neg_class = len(a) - num_pos_class
        methode = i.split("_")[0]
        bckp_metric[methode] = {
            "tpr": tpr,
            "fpr": fpr,
            "thresholds": thresholds,
            "num_pos_class": num_pos_class,
            "num_neg_class": num_neg_class,
        }
        ax.plot(fpr, tpr, label=f"{i} - ROC curve (area = %0.2f)" % auc_score)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
    plt.tight_layout()
    os.makedirs("../data/fig/", exist_ok=True)
    outd = os.path.join(path_save, "ROC.pdf")
    plt.savefig(outd, backend="pgf")
