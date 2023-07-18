import sys, os, time
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sklearn.preprocessing import LabelEncoder

from estimators.statistical_descriptor import Nagler_WS
from plot.figure_roc import ROC_plot
from utils.dataset_management import load_train, load_test, parser_pipeline, BFold
from utils.files_management import (
    load_yaml,
    dump_pkl,
    init_logger,
    open_param_set_dir,
    report_prediction,
    report_metric_from_log,
    write_report,
)


def prediction_dataset(
    X_train,
    y_train,
    X_test,
    Y_test,
    label_encoder,
    output_dir,
    methods_param,
    logg,
    save=True,
):
    bkf = BFold(shuffle=False, random_state=42)

    kfold = 0
    y_est_save, metrics = {}, {}
    kappa, f1sc, acc = [], [], []
    pos_class = label_encoder.transform(["wet"])[0]

    for count in range(len(methods_param["pipeline"])):
        name_pip = methods_param["name_pip"][count]
        logg.info(f"Pipeline : {name_pip}")
        y_est_save[name_pip] = {"y_true": [], "y_est": []}

        for train_index in bkf.split(X_train, y_train):
            logg.info(f"Kfold : {kfold}")
            X_train_K, y_train_k = X_train[train_index], y_train[train_index]
            logg.info(f" y_train_k : {np.unique(y_train_k, return_counts=True)}")
            logg.info(f" X_train_K : {X_train_K.shape}")

            pipeline = parser_pipeline(methods_param, count)

            try:
                id_pip = name_pip + f"_kfold_{kfold}"
                pipeline.fit(X_train_K, y_train_k)

                y_prob = pipeline.predict_proba(X_test)[:, pos_class]

                logg, f1, ac, ka = report_prediction(
                    Y_test, y_prob, label_encoder, logg
                )
                f1sc.append(f1)
                acc.append(ac)
                kappa.append(ka)

                y_est_save[name_pip]["y_est"].extend(y_prob)
                y_est_save[name_pip]["y_true"].extend(Y_test)

            except Exception as e:
                logg.error(f"Pipeline {id_pip} failed")
                logg.error(e)
            kfold += 1
        metrics[name_pip] = {"f1": f1sc, "acc": acc, "kappa": kappa}
        logg = report_metric_from_log(metrics, logg)
        if save:
            dump_pkl(pipeline, os.path.join(output_dir, f"{name_pip}.pkl"))
            dump_pkl(metrics, os.path.join(output_dir, f"metrics.pkl"))
    return y_est_save


def Nagler_estimation(data_path):
    y_est_save = {}
    X_trainU, y_train, label_encoder = load_train(
        data_path, -1, balanced=False, shffle=True, encode=True
    )
    X_test, y_test = load_test(
        data_path, -1, balanced=True, shffle=True, encoder=label_encoder
    )

    pos_class = label_encoder.transform(["wet"])[0]

    NGS_VV = Nagler_WS(bands=6)
    name_pip = "Nagler_VV"
    prob_test = NGS_VV.predict_proba(X_test)[:, pos_class]
    prob_train = NGS_VV.predict_proba(X_trainU)[:, pos_class]
    y_prob = np.concatenate([prob_train, prob_test])
    y_true = np.concatenate([y_train, y_test])

    y_est_save[name_pip] = {"y_true": y_true, "y_est": y_prob}

    NGS_VH = Nagler_WS(bands=7)
    name_pip = "Nagler_VH"
    prob_test = NGS_VH.predict_proba(X_test)[:, pos_class]
    prob_train = NGS_VH.predict_proba(X_trainU)[:, pos_class]
    y_prob = np.concatenate([prob_train, prob_test])
    y_true = np.concatenate([y_train, y_test])

    y_est_save[name_pip] = {"y_true": y_true, "y_est": y_prob}

    return y_est_save


def logg_info(log_F, X_trainU, y_train, X_test, y_test, label_encoder, bands_max):
    """Logg information about the dataset"""
    log_F.info("############################################")
    log_F.info(f"Loaded {X_trainU.shape} train samples and {X_test.shape} test samples")
    log_F.info(f"Y_train: {np.unique(y_train, return_counts=True)}")
    log_F.info(f"Y_test: {np.unique(y_test, return_counts=True)}")
    log_F.info(f"Classes {label_encoder.classes_}")
    log_F.info(f"Labels {label_encoder.transform(label_encoder.classes_)}")
    log_F.info(f"List of bands {bands_max}")
    log_F.info("############################################")
    return log_F


def evaluate_methods(log_F, data_path, bands_max, methods_param, y_nagler):
    X_trainU, y_train, label_encoder = load_train(
        data_path, bands_max, balanced=False, shffle=True, encode=True
    )
    X_test, y_test = load_test(
        data_path, bands_max, balanced=True, shffle=True, encoder=label_encoder
    )

    log_F = logg_info(
        log_F, X_trainU, y_train, X_test, y_test, label_encoder, bands_max
    )
    y_est = prediction_dataset(
        X_trainU,
        y_train,
        X_test,
        y_test,
        label_encoder,
        output_path,
        methods_param,
        log_F,
    )
    y_est.update(y_nagler)

    return y_est


if "__main__" == __name__:
    param_path = "parameter/Y_fit_S_param.yml"
    methods_param = load_yaml(param_path)
    try:
        data_path = methods_param["data_path"]
        out_dir = methods_param["out_dir"]
        seed = methods_param["seed"]
        BANDS_MAX = methods_param["BANDS_MAX"]
    except KeyError as e:
        print("KeyError: %s undefine" % e)

    start_line = 0

    y_nagler = Nagler_estimation(data_path)

    for keys in list(BANDS_MAX.keys()):
        np.random.seed(seed)
        log_F.info(f"================== Fitting model {keys} ==================")
        list_bands = BANDS_MAX[keys]
        subout = out_dir + f"_{keys}"
        os.makedev(subout, exist_ok=True)
        output_path = open_param_set_dir(param_path, subout)

        log_F, path_log = init_logger(output_path)
        log_F.info("Data path : %s" % data_path)
        log_F.info("seed : %s" % seed)
        log_F.info("Output path : %s" % output_path)
        log_F.info("keys : %s" % keys)
        log_F.info("Number of bands : %s" % list_bands)

        t = time.time()
        y_est = evaluate_methods(log_F, data_path, list_bands, methods_param, y_nagler)
        end = time.time() - t
        log_F.info(f"============= Done in {end} seconds ============")

        dump_pkl(y_est, os.path.join(output_path, f"proba_y_est_{keys}.pkl"))
        ROC_plot(y_est, output_path)
        _, start_line = write_report(
            path_log, os.path.join(output_path, "report.txt"), begin_line=start_line
        )
