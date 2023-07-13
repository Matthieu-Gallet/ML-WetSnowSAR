from osgeo_utils import gdal_merge
from osgeo import gdal
from joblib import Parallel, delayed

import subprocess

from utils import *


def load_data(file_name, gdal_driver="GTiff"):
    """
    Converts a GDAL compatable file into a numpy array and associated geodata.
    The rray is provided so you can run with your processing - the geodata consists of the geotransform and gdal dataset object
    If you're using an ENVI binary as input, this willr equire an associated .hdr file otherwise this will fail.
    This needs modifying if you're dealing with multiple bands.

    VARIABLES
    file_name : file name and path of your file

    RETURNS
    image array
    (geotransform, inDs)
    """
    driver = gdal.GetDriverByName(gdal_driver)  ## http://www.gdal.org/formats_list.html
    driver.Register()

    inDs = gdal.Open(file_name, gdal.GA_ReadOnly)

    if inDs is None:
        print("Couldn't open this file: %s" % (file_name))
    else:
        pass
    # Extract some info form the inDs
    geotransform = inDs.GetGeoTransform()
    projection = inDs.GetProjection()

    # Get the data as a numpy array
    cols = inDs.RasterXSize
    rows = inDs.RasterYSize

    channel = inDs.RasterCount
    image_array = np.zeros((rows, cols, channel), dtype=np.float32)
    for i in range(channel):
        data_array = inDs.GetRasterBand(i + 1).ReadAsArray(0, 0, cols, rows)
        image_array[:, :, i] = data_array
    inDs = None
    return image_array, (geotransform, projection)


def array2raster(data_array, geodata, file_out, gdal_driver="GTiff"):
    """
    Converts a numpy array to a specific geospatial output
    If you provide the geodata of the original input dataset, then the output array will match this exactly.
    If you've changed any extents/cell sizes, then you need to amend the geodata variable contents (see below)

    VARIABLES
    data_array = the numpy array of your data
    geodata = (geotransform, inDs) # this is a combined variable of components when you opened the dataset
                            inDs = gdal.Open(file_name, GA_ReadOnly)
                            geotransform = inDs.GetGeoTransform()
                            see data2array()
    file_out = name of file to output to (directory must exist)
    gdal_driver = the gdal driver to use to write out the data (default is geotif) - see: http://www.gdal.org/formats_list.html

    RETURNS
    None
    """

    if not exists(dirname(file_out)):
        print("Your output directory doesn't exist - please create it")
        print("No further processing will take place.")
    else:
        post = geodata[0][1]
        original_geotransform, projection = geodata

        rows, cols, bands = data_array.shape
        # adapt number of bands to input data

        # Set the gedal driver to use
        driver = gdal.GetDriverByName(gdal_driver)
        driver.Register()

        # Creates a new raster data source
        outDs = driver.Create(file_out, cols, rows, bands, gdal.GDT_Float32)

        # Write metadata
        originX = original_geotransform[0]
        originY = original_geotransform[3]

        outDs.SetGeoTransform([originX, post, 0.0, originY, 0.0, -post])
        outDs.SetProjection(projection)

        # Write raster datasets
        for i in range(bands):
            outBand = outDs.GetRasterBand(i + 1)
            outBand.WriteArray(data_array[:, :, i])

        print("Output saved: %s" % file_out)


def mask_layover(data_path, mask_path):
    """Mask the layover pixels in the data with the mask

    Parameters
    ----------
    data_path : str
        Path to the data to be masked
    mask_path : str
        Path to the mask to be used

    Returns
    -------
    str
        Path to the masked data
    """
    data_files = glob.glob(data_path)
    mask_img, _ = load_data(mask_path)
    for data_file in tqdm(data_files):
        im, ge = load_data(data_file)
        for i in range(im.shape[2]):
            im[:, :, i] = np.where(mask_img[:, :, 0] == 1, -999, im[:, :, i])
        new_dir = join(dirname(data_file), "temp")
        new_path = join(new_dir, basename(data_file))
        exist_create_folder(new_dir)
        array2raster(im, ge, new_path)
    return join(new_dir, "*.tif")


# def linear_to_db(value):
#    array = np.where(value <= 0, -999, 10 * np.log10(value))
#    return array


def linear_to_db(value, bands):
    array = value.copy()
    array[:, :, bands - 1] = np.where(
        value[:, :, bands - 1] <= 0, -999, 10 * np.log10(value[:, :, bands - 1])
    )
    return array


def ref_to_ratio(value, band):
    array = value.copy()
    log_data = np.where(
        value[:, :, band - 1] <= 0, -999, 10 * np.log10(value[:, :, band - 1])
    )
    array[:, :, band - 1] = np.where(
        value[:, :, band - 1] <= 0,
        -999,
        value[:, :, (band - 1) % 3] / log_data,
    )
    return array


def apply_function(file_name, dic_bands):
    for nbands in dic_bands:
        name_func = dic_bands[nbands]
        func = eval(name_func)
        array, geo = load_data(file_name)
        remove(file_name)
        # array[:, :, nbands - 1] = func(array[:, :, nbands - 1])
        array = func(array, nbands)
        array2raster(array, geo, file_name)
    return 1


def raster2windows(input_filename, output_path, winsize, step):
    ds = gdal.Open(input_filename)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    name_base = basename(input_filename)[:-4]
    for i in range(0, xsize, step):
        for j in range(0, ysize, step):
            output_filename = join(output_path, f"{name_base}_{i}_{j}.tif")
            com_string = (
                f"gdal_translate -of GTIFF -a_nodata -999 -srcwin {i}, {j}, "
                + f"{winsize}, {winsize} {input_filename}"
                + f" {output_filename}"
            )
            system(com_string)


def normalize_raster(input_filename, output_path, nbands, min_s, max_s):
    input_im = gdal.Open(input_filename, gdal.GA_ReadOnly)
    name_base = basename(input_filename)[:-4]
    output_filename = join(output_path, f"{name_base}_N.tif")
    driver_tiff = gdal.GetDriverByName("GTiff")
    output_im_band = driver_tiff.CreateCopy(output_filename, input_im, strict=0)

    for i in range(1, nbands + 1):
        input_im_band_ar = input_im.GetRasterBand(i).ReadAsArray()
        output_im_band_ar = np.array(
            (((input_im_band_ar) - (min_s[i - 1])) / ((max_s[i - 1]) - (min_s[i - 1]))),
            dtype=np.float32,
        )
        output_im_band_ar = np.where(output_im_band_ar < 0, 0, output_im_band_ar)
        output_im_band.GetRasterBand(i).WriteArray(output_im_band_ar)
    output_im_band = input_im_band_ar = None
    return 1


def gdal_resolution(inraster):
    ds = gdal.Open(inraster)
    gt = ds.GetGeoTransform()
    res = gt[1]
    return res


def check_resolution(inref, inraster):
    raster_ref = gdal_resolution(inref)
    raster_c = gdal_resolution(inraster)
    return raster_ref == raster_c


def gdal_projection(inraster):
    ds = gdal.Open(inraster)
    pr = ds.GetProjection()
    return pr


def check_projection(inref, inraster):
    pr_ref = gdal_projection(inref)
    pr_c = gdal_projection(inraster)
    return pr_ref == pr_c


def check_ResProj_files(aux_path, data_path):
    re_d, pr_d = check_data(data_path)
    re_a, pr_a = check_data(aux_path)
    if re_d == re_a and pr_d == pr_a:
        return 1
    else:
        print("Resolution or projection not compatible")
        print(re_d, re_a)
        print(pr_d, pr_a)
        return 0


def check_data(data_path):
    data_files = glob.glob(data_path)
    dataR = data_files[0]
    for i in data_files[1:]:
        condR = check_resolution(dataR, i)
        condP = check_projection(dataR, i)
        if condR and condP:
            dataR = i
        else:
            print(f"data not compatible in projection {condP} or resolution {condR}")
            print(i)
            return 0
    return gdal_resolution(dataR), gdal_projection(dataR)


def check_with_nan(img):
    f = False
    try:
        c2, _ = load_data(img)
        for i in range(2):
            for j in range(2):
                cond = ((c2[:, :, i][c2[:, :, j] == -999] == -999).all()) & (
                    len(c2[:, :, i][c2[:, :, j] == -999] == -999) > 0
                )
                f = f | cond
    except:
        f = True
    return f


def clean_data_nan(path):
    list_files = glob.glob(path)
    for i in tqdm(list_files):
        if check_with_nan(i):
            remove(i)


def stats_dataset(path, dico):
    list_files = glob.glob(join(path, "*.tif"))
    for i in tqdm(list_files):
        c, _ = load_data(i)
        nbands = c.shape[2]
        try:
            dico["mean"].append(
                [np.mean(c[:, :, v][c[:, :, v] > -999]) for v in range(nbands)]
            )
            dico["std"].append(
                [np.std(c[:, :, v][c[:, :, v] > -999]) for v in range(nbands)]
            )
            dico["min"].append(
                [np.min(c[:, :, v][c[:, :, v] > -999]) for v in range(nbands)]
            )
            dico["max"].append(
                [np.max(c[:, :, v][c[:, :, v] > -999]) for v in range(nbands)]
            )
        except:
            print(i)
    return dico


def extract_patch_clean(path, output_path, winsize, step):
    list_files = glob.glob(path)
    Parallel(n_jobs=-1)(
        delayed(raster2windows)(name, output_path, winsize, step) for name in list_files
    )
    clean_data_nan(join(output_path, "*.tif"))


def gdal_clip_shp_raster(inraster, inshape, outraster, country_name):
    subprocess.call(
        [
            "gdalwarp",
            "-of",
            "Gtiff",
            "-dstnodata",
            "value -999",
            "-ot",
            "Float32",
            inraster,
            outraster,
            "-cutline",
            inshape,
            "-crop_to_cutline",
            "-cwhere",
            f"id='{country_name}'",
        ]
    )


def gdal_merge_rasters(in_R1, in_R2, outraster):
    gdal_merge.main(
        [
            "",
            "-o",
            outraster,
            "-separate",
            "-ot",
            "Float32",
            "-of",
            "GTiff",
            "-n",
            "0",
            "-a_nodata",
            "-999",
            "-ot",
            "Float32",
            "-of",
            "GTiff",
            "-co",
            "COMPRESS=NONE",
            "-co",
            "BIGTIFF=IF_NEEDED",
            in_R1,
            in_R2,
        ]
    )


def normalize_tensor(X, stat):
    max_D = np.max(stat["max"], axis=0)
    min_D = np.min(stat["min"], axis=0)
    for j in range(X.shape[0]):
        for i in range(X.shape[3]):
            X[j, :, :, i] = np.array(
                (X[j, :, :, i] - min_D[i]) / (max_D[i] - min_D[i]), dtype=np.float32
            )
            X[j, :, :, i] = np.where(X[j, :, :, i] < 0, 0, X[j, :, :, i])
    return X


def normalize_dataset(input_path, output_path, stat):
    list_files = glob.glob(input_path)
    max_D = np.max(stat["max"], axis=0)
    min_D = np.min(stat["min"], axis=0)
    for name in tqdm(list_files):
        normalize_raster(name, output_path, len(max_D), min_D, max_D)
    return 1


def count_value_altitude(prediction, dem, alti_min, alti_max, step, condition):
    lower = alti_min
    total = 0
    n_pix = []
    for upper in np.arange(alti_min + step, alti_max, step):
        count_alt = prediction[(dem >= lower) & (dem < upper) & condition]
        n_pix.append(np.nansum(count_alt))
        total += count_alt.size
        lower = upper
    count_alt = prediction[(dem >= lower) & condition]
    n_pix.append(np.nansum(count_alt))
    #total += count_alt.size
    #alti = np.array(n_pix) * 100 / total
    return n_pix# alti


def count_value_orien_alti(
    prediction, theta, dem, orientation, alti_min, alti_max, step, cond_pente
):
    lower_o = orientation[0]
    alti_theta = []
    for upper_o in orientation[1:]:
        cond_theta = (theta >= lower_o) & (theta < upper_o) & cond_pente
        array_alti = count_value_altitude(
            prediction, dem, alti_min, alti_max, step, cond_theta
        )
        alti_theta.append(array_alti)
        lower_o = upper_o
    cond_theta = (theta >= lower_o) & cond_pente
    array_alti = count_value_altitude(
        prediction, dem, alti_min, alti_max, step, cond_theta
    )
    alti_theta.append(array_alti)
    return np.array(alti_theta)


def count_value_pen_ori_alti(
    prediction, gradient, dem, theta, orientation, pente, alti_min, alti_max, step
):
    alpeor = []
    lower_p = pente[0]
    for upper_p in tqdm(pente[1:]):
        cond_pente = (gradient >= lower_p) & (gradient < upper_p)
        alpeor.append(
            count_value_orien_alti(
                prediction,
                theta,
                dem,
                orientation,
                alti_min,
                alti_max,
                step,
                cond_pente,
            )
        )
        lower_p = upper_p
    cond_pente = gradient >= lower_p
    alpeor.append(
        count_value_orien_alti(
            prediction, theta, dem, orientation, alti_min, alti_max, step, cond_pente
        )
    )
    alpeor = np.array(alpeor)
    return np.moveaxis(alpeor, 0, 2)


def gdal_resampling(i_path, resolution):
    o_path = i_path[:-4] + "_resampled_" + str(resolution) + ".tif"
    cmd = f"gdalwarp -tr {resolution} {resolution} -of GTiff {i_path} {o_path}"
    system(cmd)
    remove(i_path)
    return o_path
