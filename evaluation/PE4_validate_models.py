from img_processing import *
from utils.geo_tools import load_data, array2raster, count_value_pen_ori_alti
import os, glob


def validation_files(input_file, path_model, output_file, nbands, winsize):
    img, geodata = load_data(input_file)
    patches = extract_patches(img, winsize, nbands)
    model = open_pkl(path_model)
    ypred = model.predict_proba(patches)[:, 1]
    reconstruct = reconstruct_image(ypred, img.shape[:2], winsize)
    array2raster(reconstruct, geodata, output_file)
    return 1


def validation_dataset(input_path, path_model, nbands, winsize):
    for file in glob.glob(input_path):
        out_dir_r = os.path.join(os.path.dirname(path_model), "validation")
        os.makedirs(out_dir_r, exist_ok=True)
        model_n = os.path.basename(path_model).split(".")[0]
        output_file = os.path.join(
            out_dir_r, f"PRED_{model_n}" + os.path.basename(file)
        )
        validation_files(file, path_model, output_file, nbands, winsize)
    return 1


if "__main__" == __name__:
    path_area = "../dataset_/dataset_A_HD16_TN0HS40_LOG/select/*2021*.tif"
    path = "../results/AUX_KNN/comp_hist_aux_i/R_200623_16H58M05"
    nbands = [0, 1, 4, 6, 7]
    winsize = 16
    binarize = False
    for p_model in glob.glob(os.path.join(path, "K*.pkl")):
        try:
            validation_dataset(path_area, p_model, nbands, winsize)
        except Exception as e:
            print(e)
            print(f"{p_model} failed")
            continue
