from osgeo_utils import gdal_merge
from osgeo import gdal
import subprocess
import numpy as np
import os


def load_data(file_name, gdal_driver="GTiff"):
    """
    Converts a GDAL compatable file into a numpy array and associated geodata.
    The rray is provided so you can run with your processing - the geodata consists of the geotransform and gdal dataset object
    If you're using an ENVI binary as input, this willr equire an associated .hdr file otherwise this will fail.
    This needs modifying if you're dealing with multiple bands.

    Parameters
    ----------
    file_name : str
        The file name of the input data

    gdal_driver : str
        The gdal driver to use to read the data (default is geotif) - see: http://www.gdal.org/formats_list.html

    Returns
    -------
    image_array : numpy array
        The data as a numpy array

    geodata : tuple
        (geotransform, inDs) # this is a combined variable of components when you opened the dataset

    """
    driver = gdal.GetDriverByName(gdal_driver)
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

     Parameters
     ----------
     data_array : numpy array
         The data as a numpy array

    geodata : tuple
         (geotransform, inDs) # this is a combined variable of components when you opened the dataset

     file_out : str
         The file name of the output data

     gdal_driver : str
         The gdal driver to use to write out the data (default is geotif) - see: http://www.gdal.org/formats_list.html

     Returns
     -------
     None
    """

    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    post = geodata[0][1]
    original_geotransform, projection = geodata

    if len(data_array.shape) == 2:
        data_array = data_array[:, :, np.newaxis]

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
    return None


def calculate_E_distribution(prediction, dem, alti_min, alti_max, step, condition):
    """Calculate the distribution of the prediction map for a given elevation range

    Parameters
    ----------
    prediction : numpy array
        prediction map (between 0 and 1)

    dem : numpy array
        digital elevation model: elevation in meters

    alti_min : float
        minimum elevation in meters to consider

    alti_max : float
        maximum elevation in meters to consider

    step : float
        step in meters to consider

    condition : numpy array
        boolean array of the same shape as dem

    Returns
    -------
    alti : numpy array
        distribution of the prediction map over the elevation range
    """
    lower = alti_min
    total = 0
    n_pix = []

    alt_ranges = np.arange(alti_min + step, alti_max, step)

    for upper in alt_ranges:
        count_alt = prediction[(dem >= lower) & (dem < upper) & condition]
        n_pix.append(np.nansum(count_alt))
        total += count_alt.size
        lower = upper
    count_alt = prediction[(dem >= lower) & condition]
    n_pix.append(np.nansum(count_alt))
    total += count_alt.size
    alti = np.array(n_pix) * 100 / total
    return alti


def calculate_OE_distribution(
    prediction, theta, dem, orientation, alti_min, alti_max, step, cond_pente
):
    """Calculate the distribution of the prediction map for a given orientation range
    and elevation range

    Parameters
    ----------
    prediction : numpy array
        prediction map (between 0 and 1)

    theta : numpy array
        orientation map in degrees

    dem : numpy array
        digital elevation model: elevation in meters

    orientation : numpy array
        array of the orientation range in degrees

    alti_min : float
        minimum elevation in meters to consider

    alti_max : float
        maximum elevation in meters to consider

    step : float
        step in meters to consider

    cond_pente : numpy array
        boolean array of the same shape as dem

    Returns
    -------
    alti_theta : numpy array
        distribution of the prediction map over the orientation range and elevation range
    """
    alti_theta = []

    for i in range(len(orientation) - 1):
        cond_theta = (
            (theta >= orientation[i]) & (theta < orientation[i + 1]) & cond_pente
        )
        array_alti = calculate_E_distribution(
            prediction, dem, alti_min, alti_max, step, cond_theta
        )
        alti_theta.append(array_alti)

    cond_theta = (theta >= orientation[-1]) & cond_pente
    array_alti = calculate_E_distribution(
        prediction, dem, alti_min, alti_max, step, cond_theta
    )
    alti_theta.append(array_alti)

    return np.array(alti_theta)


def calculate_SOE_distribution(
    prediction, gradient, dem, theta, orientation, pente, alti_min, alti_max, step
):
    """Calculate the distribution of the prediction map for a given orientation range,
    elevation range and slope range using step.
    For example if you want to the number of pixels with a slope between 0 and 5 degrees,
    a orientation between 0 and 22.5 degrees and an elevation between 0 and 100 meters.

    Example::
    prediction = np.random.rand(500, 500)
    dem = np.random.rand(500, 500) * 4000
    gradient = np.random.rand(500, 500) * 90
    theta = np.random.rand(500, 500) * 360
    pente = [22.5]
    step_alt = 50
    step_or = 22.5 / 2
    alti_max = dem.max()
    alti_min = dem[dem > 0].min() - step_alt
    orientation = np.arange(-step_or, 360, step_or) % 360
    alpeor_II = calculate_SOE_distribution(
        prediction, gradient, dem, theta, orientation, pente, alti_min, alti_max, step_alt
    )

    Parameters
    ----------
    prediction : numpy array
        prediction map (between 0 and 1)

    gradient : numpy array
        slope map in degrees

    dem : numpy array
        digital elevation model: elevation in meters

    theta : numpy array
        orientation map in degrees

    orientation : numpy array
        array of the orientation range in degrees

    pente : numpy array
        array of the slope range in degrees

    alti_min : float
        minimum elevation in meters to consider

    alti_max : float
        maximum elevation in meters to consider

    step : float
        step in meters to consider

    Returns
    -------
    alpeor : numpy array
        distribution of the prediction map over the orientation range, elevation range and slope range
    """

    alpeor = []

    for i in range(len(pente) - 1):
        cond_pente = (gradient >= pente[i]) & (gradient < pente[i + 1])
        alpeor.append(
            calculate_OE_distribution(
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

    cond_pente = gradient >= pente[-1]
    alpeor.append(
        calculate_OE_distribution(
            prediction, theta, dem, orientation, alti_min, alti_max, step, cond_pente
        )
    )

    alpeor = np.array(alpeor)
    return np.moveaxis(alpeor, 0, 2)
