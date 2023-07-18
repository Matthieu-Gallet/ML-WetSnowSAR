from figure_canaux import *
from figure_repartition import *
from figure_measureTN import *
from figure_distribution import *
from figure_roc import *
from figure_stats import *
from figure_diagramme import *
from figure_repartition_wet_massif import *

if __name__ == "__main__":
    figure0 = 0
    figure1 = 0
    figure2 = 0
    figure3 = 0
    figure4 = 0
    figure5 = 0
    figure6 = 1
    figure7 = 0
    figure8 = 0
    figure9 = 0

    if figure0:
        print("######################## Figure 0 ########################")
        dicband = {
            "A": [0, 1],
            "B": [0, 1, 2],
            "C": [0, 1, 3],
            "D": [0, 1, 4],
            "E": [0, 1, 5],
            "F": [0, 1, 3, 4, 5],
            "G": [6, 7],
            "H": [0, 1, 6, 7],
            "H_r": [0, 1, 2, 6, 7, 8],
            "I": [0, 1, 4, 6, 7],
            "I_d": [0, 1, 3, 6, 7],
            "I_p": [0, 1, 5, 6, 7],
            "J": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
        # log_path = "../data/results/log_200223_12H16M08.log"
        log_path = "../data/results/log_200623_15H18M45.log"
        f = open_log(log_path)
        f1score = parse_f1(f, dicband)
        f1score = sort_dic(f1score)
        for key in f1score:
            print(
                key,
                f"${100*np.mean(f1score[key]):.3f} \pm {100*np.std(f1score[key]):.3f}$",
            )
        plot_boxplot_channel_aux(f1score)

    if figure1:
        print("######################## Figure 1 ########################")
        fp = "../data/results/repartition/"
        train = [
            "BAUGES",
            "BEAUFORTAIN",
            "BELLEDONNE",
            "CHARTEUSE",
            "VANOISE",
            "MAURIENNE",
        ]
        test = ["GRANDES-ROUSSES"]

        df_train, df_test = extract_topo_dataset(fp, train, test)
        slope_full_rose("stat_train", df_train, (5e-5, 5e-1), cb=False)
        slope_full_rose("stat_test", df_test, (5e-5, 5e-1), cb=True)

    if figure2:
        print("######################## Figure 2 ########################")
        M1 = "BELLEDONNE"
        M2 = "GRANDES-ROUSSES"
        name = "ASC_TN_2700_1800"
        tn = open_pkl("../data/results/FC_21X22_asc_tn.pkl")
        hs = open_pkl("../data/results/FC_21X22_asc_hs.pkl")
        plot_tn_two_massifs(M1, M2, tn, hs, name)

    if figure3:
        print("######################## Figure 3 ########################")
        M1 = "BELLEDONNE"
        M2 = "GRANDES-ROUSSES"
        name = "ASC_LWC_2700_1800_4"
        tel = open_pkl("../data/results/FC_21X22_asc_tel.pkl")
        plot_tel_two_massifs(M1, M2, tel, name)

    if figure4:
        print("######################## Figure 4 ########################")
        name = "corr_tn_tel_wet_snow"
        try:
            corr_spear, thresholds = open_pkl("../data/results/corr_tel_tmin.pkl")
        except:
            thresholds = np.logspace(np.log10(0.1), np.log10(50), 50)
            save = True
            tel = open_pkl("../data/results/FC_21X22_asc_tel.pkl")
            ts = open_pkl("../data/results/FC_21X22_asc_tn.pkl")
            corr_spear = correlation_tel_tmin(tel, ts, thresholds, save)
        print(
            "Max correlation: ",
            np.max(corr_spear),
            " at ",
            thresholds[np.argmax(corr_spear)],
        )
        plot_correlation_tn_tel(thresholds, corr_spear, name)

    if figure5:
        print("######################## Figure 5 ########################")
        # X_train, Y_train = open_pkl("../data/results/bal_train.pkl")
        # X_test, Y_test = open_pkl("../data/results/bal_test.pkl")

        # print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        # X_test = X_test[:, :, :, [0, 1, 6, 7]]
        # X_train = X_train[:, :, :, [0, 1, 6, 7]]

        # x = np.concatenate((X_train, X_test), axis=0)
        # y = np.concatenate((Y_train, Y_test), axis=0)
        x, y = open_pkl("../data/results/xy_un.pkl")
        canal = ["VV", "VH", "R_VV", "R_VH"]

        idx = y == "wet"
        xb = x[idx]
        xa = x[~idx]
        xa = shuffle(xa)

        xa = xa[: xb.shape[0]]

        # mean study
        xf = np.concatenate((xa, xb), axis=0)
        yf = np.concatenate((np.zeros(xa.shape[0]), np.ones(xb.shape[0])), axis=0)
        idx = yf == 1
        dyn = [0.2, 0.8]
        value = (xf.mean(axis=(1, 2))).T
        plot_hist2d(value, canal, idx, dyn, name="hist2d_mean_wet")

        # std study
        canal = ["VV", "VH", "R_VV", "R_VH"]
        value = (xf.std(axis=(1, 2))).T
        dyn = [0, 0.125]
        plot_hist2d(value, canal, idx, dyn, name="hist2d_std_wet")

    if figure6:
        print("######################## Figure 6 ########################")
        inpath = "../data/results/comp_aux_roc/"
        dic_test = prepare_ROC(inpath)
        plot_roc_multi(dic_test, "ROC_multi_aux")

    if figure7:
        print("######################## Figure 7 ########################")
        dicband = {
            "L": "mean",
            "M": "meanstd",
            "N": "meanstdskew",
            "O": "meanstdskewkurt",
        }
        log_path = "../data/results/log_210623_14H08M50.log"
        f = open_log(log_path)
        f1score = parse_f1_score_log(f, dicband)
        f1score = sort_dic(f1score)
        for key in f1score:
            print(
                key,
                f"${100*np.mean(f1score[key]):.3f} \pm {100*np.std(f1score[key]):.3f}$",
            )
        plot_boxplot_channel(f1score, "comp_stats")

    if figure8:
        print("######################## Figure 8 ########################")
        dat = ["0118", "0330"]
        for d in dat:
            PATH = f"../data/results/diag/*{d}*.pkl"
            files = glob.glob(PATH)
            if len(files) == 0:
                path = dirname(PATH) + "/"
                pente = [0, 45]
                step_alt = 100
                step_or = 22.5 / 2
                evaluate_dataset(
                    path,
                    pente,
                    step_alt,
                    step_or,
                )
                files = glob.glob(PATH)
            dico = {}
            for i in files:
                dico[i.split("/")[-1].split("_")[0]] = open_pkl(i)[
                    list(open_pkl(i).keys())[0]
                ]
            figname = f"diagramme_{d}"
            plot_diagramme_4prod(dico, figname)

    if figure9:
        print("######################## Figure 9 ########################")
        data_p = "../data/results/"
        caracl = prepare_data(data_p)
        plot_carac_wet_massif(caracl, "repartition_wet_massif")
