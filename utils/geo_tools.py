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
