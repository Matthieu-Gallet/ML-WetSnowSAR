import sys, os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from PE3_learning_models import *


if __name__ == "__main__":
    comparison0 = 1
    comparison1 = 1
    comparison2 = 1

    ################ Comparison of the different methods ################
    if comparison0:
        param_path = "../parameter/comparison_algos.yml"
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
            y_est = evaluate_methods(
                log_F, data_path, list_bands, methods_param, y_nagler
            )
            end = time.time() - t
            log_F.info(f"============= Done in {end} seconds ============")

            dump_pkl(y_est, os.path.join(output_path, f"proba_y_est_{keys}.pkl"))
            ROC_plot(y_est, output_path)
            _, start_line = write_report(
                path_log, os.path.join(output_path, "report.txt"), begin_line=start_line
            )

    ################ Comparison of the channel ################
    if comparison1:
        param_path = "../parameter/comparison_channels.yml"
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
            y_est = evaluate_methods(
                log_F, data_path, list_bands, methods_param, y_nagler
            )
            end = time.time() - t
            log_F.info(f"============= Done in {end} seconds ============")

            dump_pkl(y_est, os.path.join(output_path, f"proba_y_est_{keys}.pkl"))
            ROC_plot(y_est, output_path)
            _, start_line = write_report(
                path_log, os.path.join(output_path, "report.txt"), begin_line=start_line
            )

    ################ Comparison of the stats ################
    if comparison2:
        param_path = "../parameter/comparison_stats.yml"
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
            y_est = evaluate_methods(
                log_F, data_path, list_bands, methods_param, y_nagler
            )
            end = time.time() - t
            log_F.info(f"============= Done in {end} seconds ============")

            dump_pkl(y_est, os.path.join(output_path, f"proba_y_est_{keys}.pkl"))
            ROC_plot(y_est, output_path)
            _, start_line = write_report(
                path_log, os.path.join(output_path, "report.txt"), begin_line=start_line
            )
