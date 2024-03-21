"""
    The module to extract and produce geotiff file from a dataset.
    The original data is in regular grid in geographicl coordinate (lat/lon).
    This program utilizes xarray and rasterio to read and output the data.
"""

import xarray as xr
import numpy as np

def add_cli_arg(cmd_parsers):
    """
        A subcommand geo_grid_to_geotiff: Using xarray & rastertio to write out geotiff.
    """
    cmd_togeotiff = cmd_parsers.add_parser("geo_grid_to_geotiff",
                                           description="Extract a dataset or a variable. Convert to"
                                                      " GeoTIFF. If there is time dimension, they "
                                                      "will be treated as bands.",
                                           help="Extract a variable")
    cmd_togeotiff.add_argument('-i', '--input', nargs=1, type=str,
                               required=True,
                               help="Input as a dataset source. It can be a datapyth for GDAL.")
    cmd_togeotiff.add_argument('-g', '--group', nargs=1, type=str,
                               required=False,
                               help=("Name of the group in input dataset. If not given, "
                                     "it will use the root group."))
    cmd_togeotiff.add_argument('-v', '--variable', nargs=1, type=str,
                               required=False,
                               help=("Name of the variable in input dataset. If not given, it will "
                                     "loop through all variables with the same dimension."))
    cmd_togeotiff.add_argument('-o', '--output', nargs=1, type=str,
                               required=True,
                               help=("Name of the output GeoTIFF file. Without extention. tif "
                                     "extension will be appended. If looping through all "
                                     "variables, variable name will be append as well."))

def process(the_args):
    """
        A subcommand plugin - process.
    """
    return to_geotiff(the_args)


def to_geotiff(args):
    """
        The method to create geotiff.
    """
    inputdataset=args.input[0]
    outfilename=args.output[0]
    ingroup=None
    if args.group:
        ingroup=args.group[0]
    invariables=args.variable
    with xr.open_dataset(inputdataset, group=ingroup, decode_cf=False) as the_ds:
        if invariables is None:
            invariables=_to_geotiff_get_variables(the_ds)
        for the_v in invariables:
            _to_geotiff(the_ds[the_v], outfilename+"."+the_v)


def _to_geotiff_get_variables(the_rs):
    the_ret=list(the_rs.data_vars)
    for var_name in list(the_rs.data_vars):
        if list(the_rs[var_name].coords) != list(the_rs.coords):
            the_ret.remove(var_name)
    return the_ret

def _gdal_dataset_to_tif(gdal_dataset, outpath, cust_projection = None,
                         cust_geotransform = None, force_custom = False,
                         nodata_value = None):
    """
    This function takes a gdal dataset object as returned from the
    "_extract_HDF_layer_data" OR "_extractNetCDF_layer_data functions
    and writes it to tif with either the embedded projection and
    geotransform or custom ones. This function should be wrapped in another
    function for a specific datatype.

    :param gdal_dataset:        A gdal.Dataset object
    :param outpath:             Output filepath for this dataset (tif)
    :param cust_projection:     A projection string, see datatype_library
    :param cust_geotransform:   A geotransform array, see datatype_library
    :param force_custom:        If True, forces the custom geotransform and
                                projections to be used even if valid
                                geotransforms and projections can be read
                                from the gdal.dataset. If False, custom
                                projections and geotransforms will be ignored
                                if valid variables can be pulled from the
                                gdal.dataset metadata.
    :param nodata_value:        The value to set to Nodata

    :return outpath:            The local system filepath to output dataset
    """

    # set up the projection and geotransform
    if force_custom is True:
        projection = cust_projection
        geotransform = cust_geotransform
    else:

        gdal_projection = gdal_dataset.GetProjection()

        if gdal_projection == "":
            projection = cust_projection
        else:
            projection = gdal_projection

        gdal_geotransform = gdal_dataset.GetGeoTransform()

        if gdal_geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            geotransform = cust_geotransform
        else:
            geotransform = gdal_geotransform

    numpy_array = gdal_dataset.ReadAsArray()
    shape = numpy_array.shape

    if len(shape) == 2:
        xsize = shape[1]
        ysize = shape[0]
        numbands = 1
    elif len(shape) == 3:
        xsize = shape[2]
        ysize = shape[1]
        numbands = shape[0]
    else:
        raise Exception("cannot write 1 dimensional data to tif")

    gtiff = gdal.GetDriverByName("GTiff")
    outdata = gtiff.Create(outpath, xsize, ysize, numbands, convert_dtype(numpy_array.dtype))
    outdata.SetProjection(projection)
    outdata.SetGeoTransform(geotransform)

    for i in range(numbands):
        outraster = outdata.GetRasterBand(i+1)
        outraster.WriteArray(numpy_array, 0, 0)
        if nodata_value is not None:
            outraster.SetNoDataValue(nodata_value)
        outraster.FlushCache()

    return outpath

def _to_geotiff(the_var, the_out_tif):
    the_time = 'time'
    if not hasattr(the_var, the_time):
        print("Warning: it should not be here. Not prcessed.") # to be processed
    else:
        for the_t in range(len(the_var[the_time])):
            the_out = the_out_tif + "."+ str(the_t)+".tif"
            the_var_t=the_var[the_t].rio.set_spatial_dims('lon', 'lat')
            if the_var_t.rio.crs is None:
                the_var_t.rio.set_crs("epsg:4326")
            if list(the_var_t.dims)==list(["lon","lat"]):
                the_var_t = the_var_t.transpose('lat','lon')
            try:
                if the_var_t.dtype.type == np.timedelta64:
                    the_fill_value, the_value_unit=_get_timedelta_fill_value(the_var_t)
                    if the_fill_value is not None:
                        the_var_t = the_var_t.fillna(value=the_fill_value)
                    if the_value_unit is not None:
                        the_var_t = the_var_t / the_value_unit
                    the_var_t = the_var_t.astype(np.float32) # to float32 or np.int32
                the_var_t.rio.to_raster(the_out)
            except Exception as the_e:
                print("ERROR - not saved properly - "+the_out)
                print(the_var_t.dtype)
                print(f"Error {str(the_e)} occured. Arguments {the_e.args}.")

def _calc_timedelta_fill_value(the_val, the_unit_str):
    the_ret = None
    if the_val is not None:
        the_ret = np.timedelta64(the_val,the_unit_str)
    the_ret_unit = np.timedelta64(1,the_unit_str)
    return the_ret, the_ret_unit

def _get_timedelta_fill_value(the_var_t):
    the_ret = None #np.timedelta64(0)
    the_ret_unit = None #np.timedelta64(1)

    the_val = None
    the_val_str = the_var_t.CodeMissingValue
    if the_val_str is None:
        print("Warning: no CodeMissingValue attribute found!")
    else:
        the_val = int(the_val_str)

    the_unit_str = the_var_t.Units
    if the_unit_str is None:
        print("Warning: no Units attribute found!")
        return the_ret, the_ret_unit
    the_unit = the_unit_str.lower()

    if the_unit.startswith("minute"):
        return _calc_timedelta_fill_value(the_val, 'm')
    elif the_unit.startswith("year"):
        return _calc_timedelta_fill_value(the_val, 'Y')
    elif the_unit.startswith("month"):
        return _calc_timedelta_fill_value(the_val, 'M')
    elif the_unit.startswith("week"):
        return _calc_timedelta_fill_value(the_val, 'W')
    elif the_unit.startswith("day"):
        return _calc_timedelta_fill_value(the_val, 'D')
    elif the_unit.startswith("hour"):
        return _calc_timedelta_fill_value(the_val, 'h')
    elif the_unit.startswith("second"):
        return _calc_timedelta_fill_value(the_val, 's')
    elif the_unit.startswith("millisecond"):
        return _calc_timedelta_fill_value(the_val, 'ms')
    elif the_unit.startswith("nanosecond"):
        return _calc_timedelta_fill_value(the_val, 'ns')
    elif the_unit.startswith("picosecond"):
        return _calc_timedelta_fill_value(the_val, 'ps')
    elif the_unit.startswith("femtosecond"):
        return _calc_timedelta_fill_value(the_val, 'fs')
    elif the_unit.startswith("attosecond"):
        return _calc_timedelta_fill_value(the_val, 'as')
    else:
        print("ERROR: Unknown unit type - "+the_unit)
    return the_ret, the_ret_unit
