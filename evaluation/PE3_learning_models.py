from old.ml_manage.display import *
from utils import *


from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import os, time, shutil
from datetime import datetime
import numpy as np


def open_param_set_dir(i_path_param, out_dir):
    now = datetime.now()
    saveto = os.path.join(out_dir, "R_" + now.strftime("%d%m%y_%HH%MM%S"))
    os.makedirs(saveto, exist_ok=True)
    shutil.copyfile(i_path_param, os.path.join(saveto, "param.yaml"))
    return saveto


def test_prediction(kfo, X_test, Y_test, classes, labels, output_dir, opt, logging):
    kappa = {}
    y_est = {}
    y_train = {}
    y_roc = {}
    y_pred_s = {}
    for count in range(len(opt["pipeline"])):
        for i in range(len(kfo[0])):
            pipeline = parser_pipeline(opt, count)
            try:
                name_pip = opt["name_pip"][count] + "_fold_" + str(i)
                xkfo = kfo[0][i]
                ykfo = kfo[1][i]
                pipeline.fit(xkfo, ykfo)
                y_pred = pipeline.predict(X_test)
                try:
                    y_est[name_pip] = {"y_est": pipeline.predict_proba(X_test)[:, 1]}
                    y_est[name_pip]["y_true"] = Y_test
                    y_train[name_pip] = {"y_est": pipeline.predict_proba(xkfo)[:, 1]}
                    y_train[name_pip]["y_true"] = ykfo
                except:
                    logging.warning(f"Pipeline {name_pip} no predict_proba")
                kappa[name_pip] = cohen_kappa_score(Y_test, y_pred)
                y_pred_s[name_pip] = accuracy_score(Y_test, y_pred)

                logging.info("############################################")
                logging.info(f"Pipeline {name_pip} accuracy test {y_pred_s[name_pip]}")
                logging.info(f"Pipeline {name_pip} Kappa test{kappa[name_pip]}")
                logging.info(
                    f"Pipeline {name_pip} F1 score test {f1_score(Y_test, y_pred)}"
                )

            except Exception as e:
                logging.error(f"Pipeline {name_pip} failed")
                logging.error(e)

        try:
            y_roc[name_pip] = {"y_est": pipeline.predict_proba(X_test)[:, 1]}
            y_roc[name_pip]["y_true"] = Y_test
        except:
            logging.warning(f"Pipeline {name_pip} no predict_proba")
        print(Y_test.shape, y_pred.shape)
        print(classes, labels)
        print(Y_test, y_pred)
        make_confusion_matrix(
            Y_test, y_pred, classes, labels, name_pip, kappa[name_pip], output_dir
        )
        dump_pkl(pipeline, os.path.join(output_dir, f"{name_pip}.pkl"))
        dump_pkl(y_train, os.path.join(output_dir, f"{name_pip}_y_train.pkl"))
    return y_est, y_roc


def nagler(data_path, mxtest, y_roc):
    X_test, Y_test = load_test(data_path, mxtest, [6, 7], True)
    LabelEnc = LabelEncoder()
    y_test = LabelEnc.fit_transform(Y_test)
    name_pip = ["Nagler_VV", "Nagler_VH"]
    for i in range(2):
        y_pred = 1 - X_test.mean(axis=(1, 2))[:, i]
        y_roc[name_pip[i]] = {"y_est": y_pred}
        y_roc[name_pip[i]]["y_true"] = y_test
    X_trainU, Y_trainU = load_train(data_path, mxtrain, [6, 7], False)
    LabelEnc = LabelEncoder()
    y_train = LabelEnc.fit_transform(Y_trainU)
    kfo = kfold_train(X_trainU, y_train)
    ytr_prd = []
    ytr_true = []
    for i in range(len(kfo[0])):
        xkfo = kfo[0][i]
        ykfo = kfo[1][i]
        y_pred = 1 - xkfo.mean(axis=(1, 2))[:, 0]
        ytr_prd.append(y_pred)
        ytr_true.append(ykfo)
    y_roc[f"Nagler_VV_train"] = {"y_est": np.concatenate(ytr_prd)}
    y_roc[f"Nagler_VV_train"]["y_true"] = np.concatenate(ytr_true)
    ytr_prd = []
    ytr_true = []
    for i in range(len(kfo[0])):
        xkfo = kfo[0][i]
        ykfo = kfo[1][i]
        y_pred = 1 - xkfo.mean(axis=(1, 2))[:, 1]
        ytr_prd.append(y_pred)
        ytr_true.append(ykfo)
    y_roc[f"Nagler_VH_train"] = {"y_est": np.concatenate(ytr_prd)}
    y_roc[f"Nagler_VH_train"]["y_true"] = np.concatenate(ytr_true)
    return y_roc


def evaluation(log_F, data_path, mxtest, mxtrain, bands_max):
    X_trainU, Y_trainU = load_train(data_path, mxtrain, bands_max, False)
    X_test, Y_test = load_test(data_path, mxtest, bands_max, True)

    LabelEnc = LabelEncoder()
    y_train = LabelEnc.fit_transform(Y_trainU)
    y_test = LabelEnc.transform(Y_test)
    classes = LabelEnc.transform(LabelEnc.classes_)
    labels = LabelEnc.inverse_transform(classes)

    kfo = kfold_train(X_trainU, y_train)

    log_F.info("############################################")
    log_F.info(f"Loaded {X_trainU.shape} train samples and {X_test.shape} test samples")
    log_F.info(
        f"wet test label {np.sum(y_test == 1)}/{len(y_test)}, dry : {np.sum(y_test == 0)}/{len(y_test)}"
    )
    log_F.info(f"Kfold train {len(kfo[0])}")
    for i in range(len(kfo[0])):
        xkfo = kfo[0][i]
        ykfo = kfo[1][i]
        log_F.info(
            f"Kfold {i} train {np.sum(ykfo == 1)}/{len(ykfo)}, dry : {np.sum(ykfo == 0)}/{len(ykfo)}"
        )
        log_F.info(f"Kfold {i} train {xkfo.shape}")
    log_F.info("############################################")

    y_est, y_roc = test_prediction(
        kfo, X_test, y_test, classes, labels, output_path, opt, log_F
    )

    log_F.info(f"Test prediction done")
    log_F.info(f"class: {classes}, labels : {labels}")
    return y_est, y_roc


if "__main__" == __name__:
    param_path = "parameter/Y_fit_S_param.yml"
    opt = load_yaml(param_path)
    try:
        data_path = opt["data_path"]
        out_dir = opt["out_dir"]
        mxtest = opt["mxtest"]
        mxtrain = opt["mxtrain"]
        seed = opt["seed"]
        bands_max = opt["bands_max"]
    except KeyError as e:
        print("KeyError: %s undefine" % e)
    BANDS_MAX = {
        # "a": [0, 1],
        # "b": [0, 1, 2],
        # "c": [0, 1, 3],
        # "d": [0, 1, 4],
        # "e": [0, 1, 5],
        # "f": [0, 1, 3, 4, 5],
        # "g": [6, 7],
        "h": [0, 1, 6, 7],
        # "h_bis": [0, 1, 2, 6, 7, 8],
        # "i": [0, 1, 4, 6, 7],
        # "i_dem": [0, 1, 3, 6, 7],
        # "i_ori": [0, 1, 5, 6, 7],
        # "j": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }
    for keys in list(BANDS_MAX.keys()):
        np.random.seed(seed)
        bands_max = BANDS_MAX[keys]
        # keys = "max_I"
        subout = out_dir + f"_{keys}"
        os.makedev(subout, exist_ok=True)
        output_path = open_param_set_dir(param_path, subout)
        log_F = init_logger(output_path)
        log_F.info("Output path : %s" % output_path)
        log_F.info("Data path : %s" % data_path)
        log_F.info("seed : %s" % seed)
        log_F.info("keys : %s" % keys)
        log_F.info("Number of bands : %s" % bands_max)
        t = time.time()
        print("================== Fitting model to data ==================")
        y_est, y_roc = evaluation(log_F, data_path, mxtest, mxtrain, bands_max)
        y_roc = nagler(data_path, mxtest, y_roc)
        dump_pkl(y_est, os.path.join(output_path, "proba_y_est.pkl"))
        dump_pkl(y_roc, os.path.join(output_path, "proba_y_roc.pkl"))

        roc_curve_plot(y_roc, os.path.join(output_path, "ROC.png"))

        end = time.time() - t
        log_F.info(f"============= Done in {end} seconds ============")
        del log_F
