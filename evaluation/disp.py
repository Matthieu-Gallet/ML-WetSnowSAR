from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import roc_curve


# import plotly.graph_objects as go
import matplotlib.pyplot as plt


def roc_curve_plot(y, path_save):
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
    plt.savefig(path_save, dpi=250, bbox_inches="tight")
    plt.close()
    dump_pkl(bckp_metric, join(dirname(path_save), "bckp_metric.pkl"))


def make_confusion_matrix(y_true, y_pred, classes, labels, name_pip, kappa, output_dir):
    cm = confusion_matrix(y_true, y_pred)  # , labels=[classes])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        cmap=plt.cm.Blues
    )
    path_save = join(output_dir, f"CM_{name_pip}_k_{kappa:.2f}.png")
    plt.savefig(path_save, dpi=250, bbox_inches="tight")
    plt.close()
    return 1
