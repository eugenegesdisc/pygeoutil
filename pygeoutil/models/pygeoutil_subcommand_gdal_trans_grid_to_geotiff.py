"""
    The module to extract and produce geotiff file from a dataset.
    This plugin use gdal.Translate. It uses the lib version of this utility.
    Refs:
    https://github.com/OSGeo/gdal/blob/master/autotest/utilities/test_gdaltranslate_lib.py
    
    The original data is in regular grid in geographicl coordinate (lat/lon).
    Required function:
        - add_cli_arg(cmd_parsers)

"""
import re
import os
import pyproj
import argparse
import affine
import numpy as np
from osgeo import gdal, osr

def add_cli_arg(cmd_parsers):
    """
        A subcommand gdal_trans_grid_to_geotiff.
    """
    cmd_togeotiff = cmd_parsers.add_parser("gdal_trans_grid_to_geotiff",
                                           description="Extract a dataset or a variable. Convert to"
                                                      " GeoTIFF. If there is time dimension, they "
                                                      "will be treated as bands.",
                                           help="Extract a variable")
    cmd_togeotiff.add_argument('-i', '--input', nargs=1, type=str,
                               required=True,
                               help="Input as a dataset source. It can be a dataset description "
                               "for GDAL.")
    cmd_togeotiff.add_argument('-g', '--group', nargs="?", type=str,
                               required=False, default="",
                               help=("Name of the group in input dataset. If not given, "
                                     "it will use the root group. If multiple group options are "
                                     "given, the last one would be picked up. Default: ''"))
    cmd_togeotiff.add_argument('-v', '--variable', nargs="?", type=str,
                               required=False,
                               help=("Name of the variable in input dataset. If not given, it will "
                                     "loop through all variables with the same group. If multiple "
                                     "variable options are given, the last one would be picked up"))
    cmd_togeotiff.add_argument('--valid-dimension',
                               dest="valid_dimension", default=True,
                               action=argparse.BooleanOptionalAction,
                               help=("Valid variables with dimensions with equal dimensions to the "
                                     "largest sum of rasterXSize and rasterYSize. "
                                     "Default is true."))
    cmd_togeotiff.add_argument('--projection',
                               dest="projection", nargs=2, type=str,
                               required=False,
                               metavar=("authority", "value"),
                               help=("Projection. <authority> can be pyproj, proj4, esri, wkt, "
                                     "epsg, crs. crs only takes 84 as value."))
    cmd_togeotiff.add_argument('--geotransform',
                               dest="geotransform", nargs=6, type=float,
                               required=False,
                               help=("Geotransform coeeficients."))
    cmd_togeotiff.add_argument('-o', '--output', nargs=1, type=str,
                               required=True,
                               help=("Name of the output GeoTIFF file. Without extention. tif "
                                     "extension will be appended. If looping through all "
                                     "variables, variable name will be append as well."))

def process(the_args):
    """
        A subcommand plugin - gdal_trans_grid_to_geotiff.
    """
    if the_args.debug:
        gdal.UseExceptions()
    else:
        gdal.DontUseExceptions()

    the_variables = _get_valid_subdatasets(the_args)
    the_variables = _filter_variable_by_group(the_args, the_variables)
    the_variables = _filter_variable_by_variable_name(the_args, the_variables)
    the_variables = _filter_variable_by_valid_dimension(the_args, the_variables)
    if the_args.debug:
        print("the_variables=", the_variables)
    _gdal_variable_to_tif(the_args,the_variables)

def _gdal_variable_to_tif(the_args, the_variables:list)->None:
    the_output = the_args.output[0]
    the_proj_auth = None
    the_proj_value = None
    if the_args.projection:
        the_proj_auth, the_proj_value = the_args.projection
    if the_args.geotransform:
        the_geotransform = the_args.geotransform
    for the_var in the_variables:
        the_var_path = the_var[1].replace('/','_')
        the_output_file = (f"{the_output}"
                           f"{the_var_path}.tif")
        _gdal_translate_variable_to_tif(the_var, the_output_file,
                                        the_proj_auth=the_proj_auth,
                                        the_proj_value=the_proj_value,
                                        the_geotransform=the_args.geotransform)

def _gdal_translate_variable_to_tif(the_var:tuple, the_output:str,
                                    the_proj_auth:str=None,
                                    the_proj_value:str=None,
                                    the_geotransform:list=None)->None:
    the_ds = gdal.Open(the_var[0], gdal.GA_ReadOnly)
    the_proj = _get_wkt_spatialreference(the_proj_auth,the_proj_value)
    if not the_proj:
        the_proj = _guess_wkt_spatialreference(the_ds)

    the_gdal_geotransform = the_ds.GetGeoTransform()
    if the_gdal_geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        print("No transformation")
        if the_geotransform:
            the_gdal_geotransform=the_geotransform
    the_ref = None
    if the_proj:
        the_ref = osr.SpatialReference()
        the_ref.ImportFromWkt(the_proj)
    the_ds = gdal.Translate(the_output, the_ds,
                            outputSRS=the_ref)
    the_ds.SetProjection(the_proj)
    the_ds.SetGeoTransform(the_gdal_geotransform)
    the_ds = None

def _transpose_raster(the_ds):
    for _i_band in range(the_ds.RasterCount):
        _band = the_ds.GetRasterBand(_i_band + 1)
        ar = _band.ReadAsArray()
        _band.WriteArray(ar.T,0,0)  # .T means transpose 2D array
    _band = None  # save, close

def _raster_center(raster):
    """This function return the pixel coordinates of the raster center 
    """

    width, height = raster.RasterXSize, raster.RasterYSize

    xmed = width / 2
    ymed = height / 2

    return (xmed, ymed)


def _rotate_gt(affine_matrix, angle, pivot=None):
    """This function generate a rotated affine matrix
    """

    affine_src = affine.Affine.from_gdal(*affine_matrix)
    affine_dst = affine_src * affine_src.rotation(angle, pivot)
    return affine_dst.to_gdal()

def _rotate_dataset_degree(the_ds, angle:int):
    """
    # Import the raster to rotate
    # Here I use a sample of GDAL, dowloaded here : https://download.osgeo.org/geotiff/samples/spot/chicago/SP27GTIF.TIF
    # and transformed in nc with qgis
    # NB: the transformation in nc is specific for the original question,
    # this step is not neccecary if you copy/past this code
    dataset_src = gdal.Open("SP27GTIF.NC")

    # For no overwriting the original raster I make a copy

    # I Get the reading/writing driver (GTIFF)
    driver = gdal.GetDriverByName("GTiff")
    # A new raster, the destination file, is created.
    # This raster is a copy of the source raster (same size, values...)
    datase_dst = driver.CreateCopy("SP27GTIF_rotate.TIF", dataset_src, strict=0)
    # Now we can rotate the raster

    # First step, we get the affine tranformation matrix of the initial fine
    # More info here : https://gdal.org/tutorials/geotransforms_tut.html#geotransforms-tut
    gt_affine = dataset_src.GetGeoTransform()
   
    """

    gt_affine = the_ds.GetGeoTransform()

    center = _raster_center(the_ds)

    the_ds.SetGeoTransform(_rotate_gt(gt_affine, angle, center))

def _guess_wkt_spatialreference(the_ds)->str:
    """
        Extract CRS from metadata.
        Typical data from NASA
        * GPM in Grid_GridHeader

    """
    print("Error: not implemented")
    return None

def _get_wkt_spatialreference(the_proj_auth:str, the_proj_val:str)->str:
    if not isinstance(the_proj_auth, str):
        return None
    if the_proj_auth.lower() == "pyproj":
        return _get_wkt_spatialreference_pyproj(the_proj_val)
    return None

def _get_wkt_spatialreference_pyproj(the_proj_val:str)->str:
    the_proj = pyproj.CRS.from_string(the_proj_val)
    return the_proj.to_wkt(pyproj.enums.WktVersion.WKT2_2019)

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

def _filter_variable_by_valid_dimension(the_args, the_variables:list)->list:
    """
        the_variables: list of
        (
        dataset_file_name, #0
        dataset, #1
        bandcount, #2
        xsize, #3
        ysize #4
        )
    """
    the_ret_variables = the_variables
    the_x, the_y = _get_maximum_size_from_variables(the_variables)
    if the_args.debug:
        print("the_x=", the_x, "the_y=", the_y)
        print("valid_dimension", the_args.valid_dimension)
    if the_args.valid_dimension:
        the_ret_variables = _filter_variable_by_x_y_size(the_ret_variables,the_x, the_y)

    return the_ret_variables

def _filter_variable_by_x_y_size(
        the_variables:list, the_x_size:int, the_y_size:int)->list:
    """
        the_variables: list of
        (
        dataset_file_name, #0
        dataset, #1
        bandcount, #2
        xsize, #3
        ysize #4
        )
    """
    the_ret_variables = the_variables
    the_ret_variables = [the_val for the_val in the_ret_variables
                         if (the_val[3] == the_x_size and
                             the_val[4] == the_y_size)]
    return the_ret_variables

def _get_maximum_size_from_variables(the_variables:list)->tuple[int, int]:
    """
        the_variables: list of
        (
        dataset_file_name,
        dataset,
        bandcount,
        xsize,
        ysize
        )
        @return: (xsize, ysize)
    """
    the_x_size = 0
    the_y_size = 0
    the_max_sum = the_x_size + the_y_size
    for the_val in the_variables:
        the_s_x_size = the_val[3]
        the_s_y_size = the_val[4]
        the_s_sum = the_s_x_size + the_s_y_size
        if the_s_sum > the_max_sum:
            the_x_size = the_s_x_size
            the_y_size = the_s_y_size
            the_max_sum = the_s_sum
    return (the_x_size, the_y_size)

def _filter_variable_by_group(the_args, the_variables:list)->list:
    """
        the_variables: list of
        (
        dataset_file_name,
        dataset,
        bandcount,
        xsize,
        ysize
        )
    """
    the_ret_variables = the_variables
    the_group = the_args.group
    if not the_group:
        the_group = ""
    the_group = the_group.strip()

    the_ret_variables = [the_val for the_val in the_ret_variables
                         if re.search(f"^[/]*{the_group}",the_val[1])]
    return the_ret_variables

def _filter_variable_by_variable_name(the_args, the_variables:list)->list:
    """
        the_variables: list of
        (
        dataset_file_name,
        dataset,
        bandcount,
        xsize,
        ysize
        )
    """
    the_ret_variables = the_variables
    the_variable = the_args.variable
    if the_variable:
        the_ret_variables = [the_val for the_val in the_ret_variables
                             if the_val[1].endswith(f"{the_variable}")]
    return the_ret_variables

def _get_valid_subdatasets(the_args)->list:
    """
        Retrieve a list of available subdataset tuples.
        (
        dataset_file_name,
        dataset,
        bandcount,
        xsize,
        ysize
        )
    """
    the_ret_variables = []
    inputdataset=the_args.input[0]
    if the_args.debug:
        print("inputfile=", inputdataset)
    the_ds = gdal.Open(inputdataset, gdal.GA_ReadOnly)

    if the_ds.RasterCount>0:
        the_ret_variables.append(
            (inputdataset, "", the_ds.RasterCount, the_ds.RasterXSize, the_ds.RasterYSize))

    the_subdatasets = the_ds.GetSubDatasets()
    for the_sds in the_subdatasets:
        the_dataset, the_shape, the_dtype =_parse_gdal_info_subset_value(the_sds[1])
        if len(the_shape) == 2:
            the_ret_variables.append((the_sds[0], the_dataset, 1, the_shape[1], the_shape[0]))
        elif len(the_shape) == 3:
            the_ret_variables.append((the_sds[0], the_dataset, the_shape[0],
                                      the_shape[2], the_shape[1]))
    return the_ret_variables

def _parse_gdal_info_subset_value(the_value:str)->tuple[str, list[int], str]:
    """
        Parse example: [1x3600x1800] //Grid/randomError (32-bit floating-point)
        To produce tuple (
            dataset="//Grid/randomError",
            shape=[1,3600,1800],
            dtype="(32-bit floating-point)")
    """
    the_ret_dataset = ""
    the_ret_dtype = ""
    the_ret_shape = []
    the_shape_strs = re.findall('^\[[^\]]*\]', the_value)
    if len(the_shape_strs) > 0:
        the_ret_shape=_parse_gdal_info_subset_value_shape(the_shape_strs[0])
    the_dtype_strs = re.findall('\([^(]*\)$', the_value)

    the_val = the_value.replace(the_shape_strs[0], "", 1)
    the_val = _replace_last(the_val, the_dtype_strs[0], "")
    the_ret_dataset = the_val.strip()
    the_ret_dtype = the_dtype_strs[0].lstrip('(').rstrip(')')
    return (the_ret_dataset, the_ret_shape, the_ret_dtype)

def _replace_last(a_string:str, old:str, new:str):
    if old not in a_string:
        return a_string

    index = a_string.rfind(old)

    return a_string[:index] + new + a_string[index+len(old):]

def _parse_gdal_info_subset_value_shape(the_shape_str:str)->list[int]:
    the_shape_strs = the_shape_str.strip('[').rstrip(']').split('x')
    the_ret_list = [int(v) for v in the_shape_strs]
    return the_ret_list

def _to_geotiff_get_variables(the_rs):
    the_ret=list(the_rs.data_vars)
    for var_name in list(the_rs.data_vars):
        if list(the_rs[var_name].coords) != list(the_rs.coords):
            the_ret.remove(var_name)
    return the_ret

def _to_geotiff(the_var, the_out_tif):
    the_time = 'time'
    # print(the_var)
    if not hasattr(the_var, the_time):
        print("Warning: it should not be here. Not prcessed.") # to be processed
    else:
        for the_t in range(len(the_var[the_time])):
            the_out = the_out_tif + "."+ str(the_t)+".tif"
            print("producing...",the_out)
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
                # the_var_t.rio.to_raster(the_out)
                the_no_data = _get_fill_value(the_var_t)
                _gdal_to_geotiff(the_out, the_var_t, nodata=the_no_data)

            except Exception as the_e:
                print("ERROR - not saved properly - "+the_out)
                print(the_var_t.dtype)
                print(f"Error {str(the_e)} occured. Arguments {the_e.args}.")

def _gdal_to_geotiff(outname, result, nodata=None):
    driv = gdal.GetDriverByName('GTiff')
    the_d_type = eval("gdal.GDT_"+result.dtype.name.capitalize())

    height,width = result.shape
    noDataVal = nodata
    if nodata is None:
        noDataVal = 0

    dstds = driv.Create(outname, width, height, 1, the_d_type)

    gt = [-180, 0.1, 0, 90, 0, -0.1]
    dstds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    dstds.GetRasterBand(1).SetNoDataValue(noDataVal)
    dstds.GetRasterBand(1).WriteArray(result.to_numpy())
    dstds = None


def _get_fill_value(the_var_t):
    the_ret = None
    the_ret = the_var_t._FillValue
    if the_ret is None:
        the_str = the_var_t.CodeMissingValue
        if the_str is not None:
            the_ret = the_var_t.dtype.type(the_str).item()
    else:
        the_ret = the_ret.item()
    return the_ret

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
