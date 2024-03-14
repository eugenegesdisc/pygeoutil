"""
    The module to extract and produce geotiff file from a dataset.
"""
import dask
import copy
import xarray as xr
import numpy as np
from pyresample import (
    geometry,image,kd_tree,ewa, bilinear
)
from osgeo import (
    gdal, osr, gdal_array
)

def add_cli_arg(cmd_parsers):
    """
        A subcommand plugin.
    """
    cmd_togeotiff2 = cmd_parsers.add_parser("swath_to_geotiff",
                                           description="Extract a dataset or a variable. Convert to"
                                                      " GeoTIFF. If there is time dimension, they "
                                                      "will be treated as bands.",
                                           help="Extract a variable")
    cmd_togeotiff2.add_argument('-i', '--input', nargs=1, type=str,
                               required=True,
                               help="Input as a dataset source. It can be a dataset for GDAL.")
    cmd_togeotiff2.add_argument('-g', '--group', nargs=1, type=str,
                               required=False, default=["/"],
                               help=("Name of the group in input dataset. If not given, "
                                     "it will use the root group or the group defined a "
                                     "GDAL path."))
    cmd_togeotiff2.add_argument('-v', '--variables', nargs='+', type=str,
                               required=False,
                               help=("Name of the variable in input dataset. If not given, it will "
                                     "loop through all variables with the same dimension."))
    cmd_togeotiff2.add_argument('-xg', '--longitude_group', nargs=1, type=str,
                               required=False, default=["/"],
                               help=("Name of the group for x/longitude variable in input dataset. "
                                     "If not given, it will use the root group or the group "
                                     "defined by GDAL path."))
    cmd_togeotiff2.add_argument('-xv', '--longitude_variable', nargs=1, type=str,
                               required=False, default=["longitude"],
                               help=("Name of the variable for x/longitude dimension in input "
                                     "dataset. If not given, it will "
                                     "use 'longitude' as the dimension variable name."))
    cmd_togeotiff2.add_argument('-xd', '--longitude_dimensions_layers', nargs='+', type=str,
                               required=False,
                               help=("List of dimension names that are not considerd"
                                     " as part of the coordinates, "
                                     "but they will be extracted as layers, "
                                     "such as 'time' dimension."))
    cmd_togeotiff2.add_argument('-yg', '--latitude_group', nargs=1, type=str,
                               required=False, default=["/"],
                               help=("Name of the group for y/latitude in input dataset. "
                                     "If not given, it will use the root group."))
    cmd_togeotiff2.add_argument('-yv', '--latitude_variable', nargs=1, type=str,
                               required=False, default=["latitude"],
                               help=("Name of the variable for x/latitude in input dataset."
                                     " If not given, it will use 'latitude' as the "
                                     "dimension variable name."))
    cmd_togeotiff2.add_argument('-yd', '--latitude_dimensions_layers', nargs='+', type=str,
                               required=False,
                               help=("List of dimension names that are not considerd as "
                                     "part of the coordinates, "
                                     "but they will be extracted as layers, such as 'time' "
                                     "dimension. They should "
                                     "the same as that for x/longitude axis."))
    cmd_togeotiff2.add_argument('-o', '--output', nargs=1, type=str,
                               required=True,
                               help=("Name of the output GeoTIFF file. Without extention. tif "
                                     "extension will be appended. If looping through all "
                                     "variables, variable name will be append as well."))
    resample_group = cmd_togeotiff2.add_argument_group("resample")
    resample_group.add_argument('-ia', '--interpolation_algorithm',
                                choices=["nearest","bilinear","kd_nearest",
                                         "kd_gauss","ewa","ewa_legacy_dask",
                                         "ewa_legacy_function"],
                                required=False,
                                default="nearest",
                                help="Interpolation algrithms")

    resample_group.add_argument('-rd', '--radius_of_influence', nargs='?', type=float,
                               required=False,
                               help="Radius of influence")
    resample_group.add_argument('-scan', '--rows_per_scan', nargs=1, type=int,
                               required=False,
                               help="Rows per scan. It needs at least two rows per scan"
                                " to compute the footprints. This number of scanlines should "
                                "be divided by an integer of rows_per_scan. For  example, GPM, 4172 "
                                "scanlines can be divided by 2, 4, 7, and 28, which could be used. "
                                "The recommended rows_per_scan for GPM may be 2. Default: the smallest "
                                "factor of the rows will be used. If no factor, 0 will be used."
                                )
    resample_group.add_argument('-er', '--epsilon', nargs='?', type=float,
                               required=False,
                               help="Allowed uncertainty. Default will be ")
    resample_group.add_argument('--sigmas', nargs='*', type=float,
                               required=False,
                               help="List of sigmas to use for the gauss weighting of each channel "
                               "1 to k, w_k = exp(-dist^2/sigma_k^2). If only one channel is resampled "
                               "sigmas is a single float value. If there are more than one channel but only "
                               "one single float value is given, the float value will be applied to "
                               "all channels.")

    proj_group = cmd_togeotiff2.add_argument_group("target_projection")
    proj_group.add_argument('-aid', '--area_id', nargs=1, type=str,
                               required=False, default=['sinu_esri54008'],
                               help="Area ID")
    proj_group.add_argument('-desc', '--description', nargs=1, type=str,
                               required=False,
                               default=['Sinusoidal Equal Area world projection'],
                               help="Area description - description of the projection (sinusoidal)")
    proj_group.add_argument('-pid', '--proj_id', nargs=1, type=str,
                               required=False, default=['sinu'],
                               help="Projection ID")
    proj_group.add_argument('-pj4', '--projection', nargs=1, type=str,
                               required=False,
                               default=["+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
                                        "+datum=WGS84 +units=m +no_defs +type=crs"],
                               help=("Projection accepted by PyProj 3.5. For example, "
                                     "proj4 projection string,"
                                     " espg code, authority:code."))
    proj_group.add_argument('-gw', '--grid_width', nargs=1, type=int,
                               required=False, default=[3600],
                               help="Grid width. This determines the resolution along x/longitude "
                                "axiss together with extent.")
    proj_group.add_argument('-gh', '--grid_height', nargs=1, type=int,
                               required=False, default=[1800],
                               help="Grid height. This determines the resolution along y/latitude "
                                "axis together with extent.")
    proj_group.add_argument('-ge', '--grid_extent', nargs=4, type=float,
                               required=False,
                               default=[-20037508.34,-10001965.73,20037508.34,10001965.73],
                               help=("Earea extent in lower_left_x lower_left_y "
                                     "upper_right_x upper_right_y"))

    uncertainty_group = cmd_togeotiff2.add_argument_group("uncertainty")
    uncertainty_group.add_argument('-ue', '--uncertainty_estimate',
                                choices=["standard_deviation","bias"],
                                required=False,
                                default=None,
                                help="Uncertainty estimate measure")


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
    if args.debug:
        print("Processing input=", inputdataset, "outputfile=", outfilename)

    ingroup=args.group[0]
    invariables=args.variables
    if args.debug:
        print("IN group=", ingroup, "selected variables=", invariables)

    xgrp = args.longitude_group[0]
    x_variable = args.longitude_variable[0]
    if args.debug:
        print("x_group=", xgrp, "x_variable=", x_variable)
    
    ygrp = args.latitude_group[0]
    y_variable = args.latitude_variable[0]
    if args.debug:
        print("y_group=", ygrp, "y_variable=", y_variable)


    if args.debug:
        print("area_id=",args.area_id,
              "proj_id=",args.proj_id,
              "description=", args.description,
              "projection=", args.projection,
              "width=", args.grid_width,
              "height=", args.grid_height,
              "area_extent=", args.grid_extent)
    area_def = geometry.AreaDefinition(
        area_id=args.area_id[0], description=args.description[0],
        proj_id=args.proj_id[0], projection=args.projection[0],
        width=args.grid_width[0], height=args.grid_height[0],
        area_extent=(args.grid_extent[0],
                     args.grid_extent[1],
                     args.grid_extent[2],
                     args.grid_extent[3])
    )
    if args.debug:
        print("AREA DEFINITION=")
        print("area_id=", args.area_id[0],
            "description=", args.description[0],
            "proj_id=",args.proj_id[0],
            "projection=", args.projection[0],
            "width=",args.grid_width[0],
            "height=",args.grid_height[0],
            "area_extent=(", 
            args.grid_extent[0],args.grid_extent[1],
            args.grid_extent[2],args.grid_extent[3],
            ")")

    if args.debug:
        print("area_def=", area_def)

    the_radius = getattr(args,
                         "radius_of_influence",
                         area_def.pixel_size_x+area_def.pixel_size_y)
    if the_radius is None:
        the_radius=area_def.pixel_size_x+area_def.pixel_size_y
    if args.debug:
        print("THE_RADIUS=", the_radius)

    the_epsilon = getattr(args, "epsilon",
                          area_def.pixel_size_x/2.0)
    if the_epsilon is None:
        the_epsilon=area_def.pixel_size_x/2.0
    if args.debug:
        print("THE_EPSILON=", the_epsilon)

    the_algorithm = args.interpolation_algorithm
    if args.debug:
        print("the_algorithm=", the_algorithm)

    xlongitude = None
    if args.debug:
        print("dimension layers for longitude=",
              args.longitude_dimensions_layers)
    the_dim_layers = None
    with xr.open_dataset(inputdataset, group=xgrp,
                         decode_cf=False) as the_ds:
        xlongitude=the_ds[x_variable]
        the_longitude_dimensions_layers = args.longitude_dimensions_layers
        if the_longitude_dimensions_layers is None:
            the_longitude_dimensions_layers=[]
        the_dim_layers = _get_layers_from_dimension(
            xlongitude, the_longitude_dimensions_layers)
    if args.debug:
        print("the_dim_layers=", the_dim_layers)

    ylatitude = None
    if args.debug:
        print("dimension layers for latitude=",
              args.latitude_dimensions_layers)
    with xr.open_dataset(inputdataset, group=ygrp,
                         decode_cf=False) as the_ds:
        ylatitude=the_ds[y_variable]


    if args.debug:
        print("coords for xlongitude=", xlongitude.coords)

    with xr.open_dataset(inputdataset, group=ingroup, decode_cf=False) as the_ds:
        if args.debug:
            print("the dataset opened=", the_ds)
        if invariables is None:
            invariables=_to_geotiff_get_variables(the_ds, xlongitude)
        if args.debug:
            print("invaraibles=", invariables)
            print("the_DS dims=", the_ds.dims)
        for the_v in invariables:
            the_v_data=the_ds[the_v]
            if args.debug:
                print("first=",the_v_data)
            the_fill_value = getattr(the_v_data,"_FillValue", 0)
            if args.debug:
                print("The_Fill_Value=", the_fill_value)
            the_layers = _get_layers(the_ds[the_v], xlongitude.dims)

            _process_layers(args, the_v_data, the_layers, 
                            xlongitude, ylatitude, the_dim_layers,
                            area_def,args.output[0], the_v,
                            the_algorithm,
                            radius=the_radius,
                            fill_value=the_fill_value,
                            epsilon=the_epsilon)

def _process_layers(args, the_v_data, the_layers, 
                    the_xlongitude, the_ylatitude, the_dim_layers,
                    grid_def, outname, the_v_name,
                    algorithm,
                    radius=0.1, fill_value=0, epsilon=0.5):
    the_sigmas = getattr(args, "sigmas", [radius]) # default to radius_of_influence
    if the_sigmas is None:
        the_sigmas = [radius]
    if args.debug:
        print("THE SIGMAS=",the_sigmas)
    if len(the_sigmas) == 1:
        n_layers = len(the_layers)
        the_sigma = the_sigmas[0]
        the_sigmas=[]
        for n in range(n_layers):
            the_sigmas.append(the_sigma)
    if args.debug:
        print("THE_SIGMAS=", the_sigmas)
        print("channels: ", len(the_layers))

    the_current_layer=0
    for the_layer in the_layers:
        if args.debug:
            print("the_layer=", the_layer)
            print("shape for the_v_data=", the_v_data.shape)
        the_data = the_v_data[the_layer.get('slice')]
        if args.debug:
            print("shape for the_data=", the_data.shape)
            print("the_data dtype:", the_data.dtype)
        for the_dim_layer in the_dim_layers:
            outfilename = outname+"_"+the_dim_layer.get('lname'
                            )+"_"+the_v_name+"_"+the_layer.get('lname')+".tif"
            if args.debug:
                print("the_dim_layer=",the_dim_layer)
            xlongitude1 = the_xlongitude[the_dim_layer.get('slice')]
            ylatitude1 = the_ylatitude[the_dim_layer.get('slice')]
            swath_def = geometry.SwathDefinition(lons=xlongitude1,lats=ylatitude1)
            the_data1 = the_data[the_dim_layer.get('slice')]
            the_rows_per_scan = _get_rows_per_scan(args,the_data1.shape[0])
            if args.debug:
                print("ROWS per scan: ", the_rows_per_scan)
            the_data_con = None # result for data
            the_uncertainty_estimate = None # uncertainty estimate
            the_uncertainty_count = None # Number of data vaules used in weighting
            if algorithm.lower()=="nearest":
                the_data_con, the_uncertainty_estimate, the_uncertainty_count = _resample_image_nearest_data(
                    args,
                    the_data1, swath_def, grid_def,
                    radius, fill_value, epsilon
                )
            elif algorithm.lower()=="bilinear":
                the_data_con, the_uncertainty_estimate, the_uncertainty_count = _resample_numpy_bilinear_data(
                    args,
                    the_data1, swath_def, grid_def,
                    radius, fill_value, epsilon
                )
            elif algorithm.lower()=="kd_nearest":
                the_data_con, the_uncertainty_estimate, the_uncertainty_count = _resample_kd_tree_nearest_data(
                    args,
                    the_data1, swath_def, grid_def,
                    radius, fill_value, epsilon
                )
            elif algorithm.lower()=="kd_gauss":
                the_data_con, the_uncertainty_estimate, the_uncertainty_count  = _resample_kd_tree_gauss_data(
                    args,
                    the_data1, swath_def, grid_def,
                    radius, fill_value, sigmas=the_sigmas[the_current_layer],
                    epsilon=epsilon
                )
            elif algorithm.lower()=="ewa":
                the_data_con, the_uncertainty_estimate, the_uncertainty_count = _resample_ewa_data(
                    args, the_data1, swath_def, grid_def,
                    rows_per_scan=the_rows_per_scan,
                    radius=radius,
                    fill_value=fill_value, sigmas=the_sigmas[the_current_layer],
                    epsilon=epsilon
                )
                if args.debug:
                    print("shape for the_data_con=",the_data_con)
            elif algorithm.lower()=="ewa_legacy_dask":
                the_data_con, the_uncertainty_estimate, the_uncertainty_count = _resample_ewa_legacy_dask_data(
                    args, the_data1, swath_def, grid_def,
                    rows_per_scan=the_rows_per_scan,
                    radius=radius,
                    fill_value=fill_value, sigmas=the_sigmas[the_current_layer],
                    epsilon=epsilon
                )
            elif algorithm.lower()=="ewa_legacy_function":
                the_data_con, the_uncertainty_estimate, the_uncertainty_count = _resample_ewa_legacy_function_data(
                    args, the_data1, swath_def, grid_def,
                    rows_per_scan=the_rows_per_scan,
                    radius=radius,
                    fill_value=fill_value, sigmas=epsilon
                )
            else:
                if args.debug:
                    print("ERROR: unrecognized algorithm=", algorithm)
                return
            if args.debug:
                print("shape for the data_con=",
                    the_data_con.shape)
            if the_data_con is not None:
                _gdal_save_to_geotiff(args, outfilename, the_data_con, grid_def, fill_value=fill_value)
                if args.debug:
                    if hasattr(the_data_con, "fill_value"):
                        print("Output data fill_value=", the_data_con.fill_value)
            if the_uncertainty_estimate is not None:
                outfilename_ue = outname+"_"+the_dim_layer.get('lname'
                                )+"_"+the_v_name+"_"+the_layer.get('lname')+"_ue"+".tif"
                _gdal_save_to_geotiff(args, outfilename_ue,
                                      the_uncertainty_estimate, grid_def, fill_value=fill_value)
            if the_uncertainty_count is not None:
                outfilename_uc = outname+"_"+the_dim_layer.get('lname'
                                )+"_"+the_v_name+"_"+the_layer.get('lname')+"_uc"+".tif"
                _gdal_save_to_geotiff(args, outfilename_uc, the_uncertainty_count, grid_def)
            the_current_layer += 1

def _get_rows_per_scan(args, the_row_num):
    """
        retrieve the rows_per_scan.
        If no input, a rows_per_scan will be estimated.
    """
    if args.rows_per_scan:
        return args.rows_per_scan[0]
    return _get_smallest_factor(the_row_num)

def _get_smallest_factor(the_num):
    """
        if no factor but itself, zero will be returned.
        the_num - the integer to find smallest factor from.
    """
    my_list = []

    for i in range(2,the_num+1):
        if(the_num%i==0):
            my_list.append(i)
    if len(my_list) > 0:
        my_list.sort()
        return(my_list[0])
    return 0

def _gdal_save_to_geotiff(
        args, outfilename, the_data, grid_def,
        fill_value=None):
    the_drive = gdal.GetDriverByName('GTiff')
    height, width=the_data.shape
    the_type = gdal_array.NumericTypeCodeToGDALTypeCode(the_data.dtype)
    if args.debug:
        print("dtype=", the_data.dtype, "gdaltype=",the_type)
    out_ds = the_drive.Create(outfilename, width, height, 1, the_type)

    # define and set outpu geotransform
    geo_transform = [grid_def.area_extent[0],
                    grid_def.pixel_size_x,0,
                    grid_def.area_extent[3],0,
                    -grid_def.pixel_size_y]
    if args.debug:
        print("gt=",geo_transform)
    out_ds.SetGeoTransform(geo_transform)

    # create and set output projection
    srs = osr.SpatialReference()
    srs.ImportFromProj4(grid_def.proj4_string)
    srs.SetProjCS(grid_def.proj_id)
    srs_wkt = srs.ExportToWkt()
    if args.debug:
        print("srs_wkt=",srs_wkt)
    out_ds.SetProjection(srs_wkt)

    if args.debug:
        print("type of the_data=", type(the_data))
        print(" SHAPE=", the_data.shape)
    out_ds.GetRasterBand(1).WriteArray(the_data)
    if fill_value:
        the_fill_val = fill_value
        if hasattr(fill_value,"dtype"):
            the_fill_val = fill_value.item()
        out_ds.GetRasterBand(1).SetNoDataValue(the_fill_val)
    srs = None
    out_ds = None

def _resample_image_nearest_data(args, the_data, swath_def,
                                 grid_def, radius=0.1,
                                 fill_value=0, epsilon=0.05):
    the_result = None
    the_ue = None
    the_ue_count = None

    if args.debug:
        print("In resample_image_nearest:",the_data)
    if (args.uncertainty_estimate and
        args.uncertainty_estimate.lower() == "bias"):
        the_data_np = the_data.to_numpy()
        swath_con = image.ImageContainerNearest(
                image_data=the_data_np, geo_def=swath_def,
                radius_of_influence=radius,fill_value=fill_value,
                epsilon=epsilon)
        area_con = swath_con.resample(grid_def)
        the_result = area_con.image_data
        swath_con2 = image.ImageContainerNearest(
                image_data=the_result, geo_def=grid_def,
                radius_of_influence=radius,fill_value=fill_value,
                epsilon=epsilon)
        area_con2 = swath_con2.resample(swath_def)
        the_result2 = area_con2.image_data
        swath_con3 = image.ImageContainerNearest(
                image_data=the_result2, geo_def=swath_def,
                radius_of_influence=radius,fill_value=fill_value,
                epsilon=epsilon)
        area_con3 = swath_con3.resample(grid_def)
        the_result3 = area_con3.image_data
        the_init_value = 0
        if fill_value:
            the_init_value = fill_value
        the_ue = np.full(the_result.shape,the_init_value)
        the_condition = (the_result!=fill_value)&(the_result3!=fill_value)
        the_ue = np.subtract(the_result,
                             the_result3, out=the_ue,
                             where=the_condition)
        if args.debug:
            print("shape of the_bias=", the_ue.shape)
            print("value of 200,200", the_ue[200][200])
    else:
        the_data_np = the_data.to_numpy()
        swath_con = image.ImageContainerNearest(
                image_data=the_data_np, geo_def=swath_def,
                radius_of_influence=radius,fill_value=fill_value,
                epsilon=epsilon)
        area_con = swath_con.resample(grid_def)
        the_result = area_con.image_data
        if args.debug:
            if hasattr(the_result,"fill_value"):
                print("FILL_VALUE=",the_result.fill_value, "fill_value0=", fill_value) 
    return the_result, the_ue, the_ue_count

def _resample_numpy_bilinear_data(args, the_data, swath_def,
                                 grid_def, radius=0.1,
                                 fill_value=0, epsilon=0.05):
    the_result = None
    the_ue = None
    the_ue_count = None

    if args.debug:
        print("In resample_numpy_bilinear:",the_data)
    if (args.uncertainty_estimate and
        args.uncertainty_estimate.lower() == "bias"):
        the_data_np = the_data.to_numpy()
        the_data_np = np.ma.masked_where(the_data_np == fill_value, the_data_np)
        np.ma.set_fill_value(the_data_np,np.nan)
        resampler = bilinear.NumpyBilinearResampler(
            source_geo_def=swath_def,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            epsilon=epsilon)
        the_result=resampler.resample(
            data=the_data_np,fill_value=np.nan)
        the_result2 = kd_tree.resample_nearest(
            source_geo_def=grid_def,
            data=the_result,
            target_geo_def=swath_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=np.nan)
        resampler3 = bilinear.NumpyBilinearResampler(
            source_geo_def=swath_def,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            epsilon=epsilon)
        the_result3=resampler3.resample(
            data=the_result2,fill_value=np.nan)
        the_ue = the_result - the_result3
        if args.debug:
            print("shape of the_bias=", the_ue.shape)
            print("value of 200,200", the_ue[200][200])
    else:
        the_data_np = the_data.to_numpy()
        the_data_np = np.ma.masked_where(the_data_np == fill_value, the_data_np)
        np.ma.set_fill_value(the_data_np,np.nan)
        resampler = bilinear.NumpyBilinearResampler(
            source_geo_def=swath_def,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            epsilon=epsilon)
        the_result=resampler.resample(
            data=the_data_np,fill_value=np.nan)
    return the_result, the_ue, the_ue_count

def _resample_image_bilinear_data(args, the_data, swath_def,
                                 grid_def, radius=0.1,
                                 fill_value=0, epsilon=0.05):
    the_result = None
    the_ue = None
    the_ue_count = None

    if args.debug:
        print("In resample_image_bilinear_data:",the_data)
    if (args.uncertainty_estimate and
        args.uncertainty_estimate.lower() == "bias"):
        swath_con = image.ImageContainerBilinear(
                image_data=the_data.to_numpy(), geo_def=swath_def,
                radius_of_influence=radius,fill_value=fill_value,
                epsilon=epsilon)
        area_con = swath_con.resample(grid_def)
        the_result = area_con.image_data
        the_result2 = kd_tree.resample_nearest(
            source_geo_def=grid_def,
            data=the_result,
            target_geo_def=swath_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
        swath_con3 = image.ImageContainerBilinear(
                image_data=the_result2, geo_def=swath_def,
                radius_of_influence=radius,fill_value=fill_value,
                epsilon=epsilon)
        area_con3 = swath_con3.resample(grid_def)
        the_result3 = area_con3.image_data
        the_ue = the_result - the_result3
        if args.debug:
            print("shape of the_bias=", the_ue.shape)
            print("value of 200,200", the_ue[200][200])
    else:
        swath_con = image.ImageContainerBilinear(
                image_data=the_data.to_numpy(), geo_def=swath_def,
                radius_of_influence=radius,fill_value=fill_value,
                epsilon=epsilon)
        area_con = swath_con.resample(grid_def)
        the_result = area_con.image_data
    return the_result, the_ue, the_ue_count

def _resample_kd_tree_nearest_data(args, the_data, swath_def,
                                 grid_def, radius=0.1,
                                 fill_value=0, epsilon=0.05):
    the_result = None
    the_ue = None
    the_ue_count = None

    if args.debug:
        print("In resample_kd_tree_nearest:",the_data)
    if (args.uncertainty_estimate and
        args.uncertainty_estimate.lower() == "bias"):
        the_data_np = the_data.to_numpy()
        the_result = kd_tree.resample_nearest(
            source_geo_def=swath_def,
            data=the_data_np,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
        the_result2 = kd_tree.resample_nearest(
            source_geo_def=grid_def,
            data=the_result,
            target_geo_def=swath_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
        the_result3 = kd_tree.resample_nearest(
            source_geo_def=swath_def,
            data=the_result2,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
        the_init_value = 0
        if fill_value:
            the_init_value = fill_value
        the_ue = np.full(the_result.shape,the_init_value)
        # ignore those where it is a pixel of fill value on either of the inputs
        the_condition = (the_result!=fill_value)&(the_result3!=fill_value)
        the_ue = np.subtract(the_result,
                             the_result3, out=the_ue,
                             where=the_condition)
        if args.debug:
            print("shape of the_bias=", the_ue.shape)
            print("value of 200,200", the_ue[200][200])
    else:
        the_data_np = the_data.to_numpy()
        the_result = kd_tree.resample_nearest(
            source_geo_def=swath_def,
            data=the_data_np,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
    return the_result, the_ue, None

def _resample_kd_tree_gauss_data(args, the_data, swath_def,
                                 grid_def, radius=0.1,
                                 fill_value=0, sigmas=None,
                                 epsilon=0.5):
    the_result = None
    the_ue = None
    the_ue_count = None

    if args.debug:
        print("In resample_kd_gauss:",the_data)
    if (args.uncertainty_estimate and
        args.uncertainty_estimate.lower() == "bias"):
        the_data_np = the_data.to_numpy()
        the_result = kd_tree.resample_gauss(
            source_geo_def=swath_def,
            data=the_data_np,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            sigmas=sigmas,
            epsilon=epsilon,
            fill_value=fill_value)
        the_result2 = kd_tree.resample_nearest(
            source_geo_def=grid_def,
            data=the_result,
            target_geo_def=swath_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
        the_result3 = kd_tree.resample_gauss(
            source_geo_def=swath_def,
            data=the_result2,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            sigmas=sigmas,
            epsilon=epsilon,
            fill_value=fill_value)
        the_init_value = 0
        if fill_value:
            the_init_value = fill_value
        the_ue = np.full(the_result.shape,the_init_value)
        the_condition = (the_result!=fill_value)&(the_result3!=fill_value)
        the_ue = np.subtract(the_result,
                             the_result3, out=the_ue,
                             where=the_condition)
        if args.debug:
            print("shape of the_bias=", the_ue.shape)
            print("value of 200,200", the_ue[200][200])
    elif (args.uncertainty_estimate and
          args.uncertainty_estimate.lower() == "standard_deviation"):
        the_data_np = the_data.to_numpy()
        the_result, the_ue, the_ue_count = kd_tree.resample_gauss(
            source_geo_def=swath_def,
            data=the_data_np,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            sigmas=sigmas,
            epsilon=epsilon,
            fill_value=fill_value,
            with_uncert=True)
    else:
        the_data_np = the_data.to_numpy()
        # the_data_np[the_data_np == fill_value] = np.nan
        the_data_np = np.ma.masked_where(the_data_np == fill_value, the_data_np)
        np.ma.set_fill_value(the_data_np,np.nan)
        the_result = kd_tree.resample_gauss(
            source_geo_def=swath_def,
            data=the_data_np,
            target_geo_def=grid_def,
            radius_of_influence=radius,
            sigmas=sigmas,
            epsilon=epsilon,
            fill_value=np.nan)
    return the_result, the_ue, the_ue_count


def _resample_ewa_data(args, the_data, swath_def,
                        grid_def, rows_per_scan=0,
                        radius=10000,
                        fill_value=0, sigmas=0.05,
                        epsilon=0.05):
    """"
        Works pm numpy array, dask array, or xarray DataArray object.

        Status: Not working
    """
    the_result = None
    the_ue = None
    the_ue_count = None

    if args.debug:
        print("INNNN resample_ewa:",the_data)
    if (args.uncertainty_estimate and
        args.uncertainty_estimate.lower() == "bias"):
        resampler = ewa.DaskEWAResampler(source_geo_def=swath_def,
                                        target_geo_def=grid_def)
        the_data = the_data.rename({"scanline":"y","ground_pixel":"x"})
        the_da_array = the_data
        result = resampler.resample(data=the_da_array,
                                    rows_per_scan=rows_per_scan)
        the_result = result.compute().to_numpy()
        the_result2 = kd_tree.resample_nearest(
            source_geo_def=grid_def,
            data=the_result,
            target_geo_def=swath_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
        the_result3 = resampler.resample(data=the_result2,
                                    rows_per_scan=rows_per_scan)
        the_ue = the_result - the_result3
        if args.debug:
            print("shape of the_bias=", the_ue.shape)
            print("value of 200,200", the_ue[200][200])
    else:
        resampler = ewa.DaskEWAResampler(source_geo_def=swath_def,
                                        target_geo_def=grid_def)
        if args.debug:
            print("BEFORE dims=", the_data.dims)
        the_data = the_data.rename({"scanline":"y","ground_pixel":"x"})
        if args.debug:
            print("AFTER dims=", the_data.dims)
        if args.debug:
            dims = [d for d in the_data.dims if d not in ('y', 'x')] + ['y', 'x']
            print("dims=", dims)
            print("THE_DATA_DIMS=",the_data.dims, "ndim=", the_data.ndim)
            print("swath_def ndim=",swath_def.ndim)
            print("grid_def ndim=",grid_def.ndim)
            print("rows_per_scan=",rows_per_scan)
            print("dims=", the_data.dims)
        the_da_array = the_data
        result = resampler.resample(data=the_da_array,
                                    rows_per_scan=rows_per_scan)
        the_result = result.compute().to_numpy()
    return the_result, the_ue, the_ue_count

def _resample_ewa_legacy_dask_data(args, the_data, swath_def,
                                   grid_def, rows_per_scan=0,
                                   radius=10000,
                                   fill_value=0,
                                   sigmas=0.05,
                                   epsilon=0.05):
    """
        Works only on xarray DataArray. It doesn't use dask optimally and
        in most cases will use a lot of memory.

        Status: not working. Data readers need to be revised.
    """
    the_result = None
    the_ue = None
    the_ue_count = None

    if args.debug:
        print("INNNN resample_ewa_legacy_dask:",the_data)
    if (args.uncertainty_estimate and
        args.uncertainty_estimate.lower() == "bias"):
        the_lons, the_lats = swath_def.get_lonlats()
        the_lons_r = xr.DataArray(dask.array.from_array(the_lons, chunks={}))
        the_lats_r = xr.DataArray(dask.array.from_array(the_lats, chunks={}))
        the_swath_def = geometry.SwathDefinition(lons=the_lons_r,lats=the_lats_r)
        resampler = ewa.LegacyDaskEWAResampler(source_geo_def=the_swath_def,
                                        target_geo_def=grid_def)
        the_data = the_data.rename({"scanline":"y","ground_pixel":"x"})
        the_da_array = xr.DataArray(dask.array.from_array(the_data, chunks={}))
        result = resampler.resample(data=the_da_array,
                                    rows_per_scan=rows_per_scan)
        #,
        #                            fill_value=fill_value)
        the_result = result.compute().to_numpy()
        the_result2 = kd_tree.resample_nearest(
            source_geo_def=grid_def,
            data=the_result,
            target_geo_def=swath_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
        the_da_array3 = xr.DataArray(dask.array.from_array(the_result2, chunks={}))
        result3 = resampler.resample(data=the_da_array3,
                                    rows_per_scan=rows_per_scan)
        #,
        #                            fill_value=fill_value)
        the_result3 = result3.compute().to_numpy()
        the_ue = the_result - the_result3
    else:
        the_lons, the_lats = swath_def.get_lonlats()
        the_lons_r = xr.DataArray(dask.array.from_array(the_lons, chunks={}))
        the_lats_r = xr.DataArray(dask.array.from_array(the_lats, chunks={}))
        the_swath_def = geometry.SwathDefinition(lons=the_lons_r,lats=the_lats_r)
        resampler = ewa.LegacyDaskEWAResampler(source_geo_def=the_swath_def,
                                        target_geo_def=grid_def)
        if args.debug:
            print("BEFORE dims=", the_data.dims)
            print("dims")
        the_data = the_data.rename({"scanline":"y","ground_pixel":"x"})
        if args.debug:
            print("AFTER dims=", the_data.dims)
            print("AFTER data=", the_data)
        if args.debug:
            dims = [d for d in the_data.dims if d not in ('y', 'x')] + ['y', 'x']
            print("dims=", dims)
            print("THE_DATA_DIMS=",the_data.dims, "ndim=", the_data.ndim)
            print("swath_def ndim=",swath_def.ndim)
            print("grid_def ndim=",grid_def.ndim)
            print("rows_per_scan=",rows_per_scan)
            print("dims=", the_data.dims)
        the_da_array = xr.DataArray(dask.array.from_array(the_data, chunks={}))
        result = resampler.resample(data=the_da_array,
                                    rows_per_scan=rows_per_scan)
        #,
        #                            fill_value=fill_value)
        the_result = result.compute().to_numpy()
    return the_result, the_ue, the_ue_count


def _resample_ewa_legacy_function_data(args, the_data, swath_def,
                        grid_def, rows_per_scan=1,
                        radius=10000,
                        fill_value=0,
                        epsilon=0.5,
                        sigmas=0.05):
    """
        Works only on numpy array. It uses low-level fucntions (ll2cr and fornav)
        directly. Fast, but use a lot of memory.
    """
    the_result = None
    the_ue = None
    the_ue_count = None

    if args.debug:
        print("INNNN resample_ewa:",the_data)
    if (args.uncertainty_estimate and
        args.uncertainty_estimate.lower() == "bias"):
        swath_points_in_grid, cols, rows = ewa.ll2cr(swath_def, grid_def)
        the_data_np = the_data.to_numpy()
        # the_data_np[the_data_np == fill_value] = np.nan
        the_data_np = np.ma.masked_where(the_data_np == fill_value, the_data_np)
        np.ma.set_fill_value(the_data_np,np.nan)
        num_valid_points, gridded_data= ewa.fornav(
            cols=cols, rows=rows, area_def=grid_def,
            data_in=the_data_np,
            rows_per_scan=rows_per_scan,fill=fill_value
        )
        the_result = gridded_data
        the_result2 = kd_tree.resample_nearest(
            source_geo_def=grid_def,
            data=the_result,
            target_geo_def=swath_def,
            radius_of_influence=radius,
            epsilon=epsilon,
            fill_value=fill_value)
        swath_points_in_grid3, cols3, rows3 = ewa.ll2cr(swath_def, grid_def)
        num_valid_points3, gridded_data3= ewa.fornav(
            cols=cols3, rows=rows3, area_def=grid_def,
            data_in=the_result2,
            rows_per_scan=rows_per_scan,fill=fill_value
        )
        the_result3 = gridded_data3
        the_init_value = 0
        if fill_value:
            the_init_value = fill_value
        the_ue = np.full(the_result.shape,the_init_value)
        # ignore those where it is a pixel of fill value on either of the inputs
        the_condition = (the_result!=fill_value)&(the_result3!=fill_value)
        the_ue = np.subtract(the_result,
                             the_result3, out=the_ue,
                             where=the_condition)
        if args.debug:
            print("shape of the_bias=", the_ue.shape)
            print("value of 200,200", the_ue[200][200])
    else:
        if args.debug:
            dims = [d for d in the_data.dims if d not in ('y', 'x')] + ['y', 'x']
            print("dims=", dims)
            print("THE_DATA_DIMS=",the_data.dims, "ndim=", the_data.ndim)
            print("swath_def ndim=",swath_def.ndim)
            print("grid_def ndim=",grid_def.ndim)
            print("rows_per_scan=",rows_per_scan)
            print("dims=", the_data.dims)

        # ll2cr convert swath longitudes and latitudes to grid columns and rows
        swath_points_in_grid, cols, rows = ewa.ll2cr(swath_def, grid_def)
        # resampler = ewa.LegacyDaskEWAResampler(source_geo_def=swath_def,
        #                                 target_geo_def=grid_def)
        if args.debug:
            print("swath_points_in_grid=", swath_points_in_grid)
            print("cols=", cols, " rows=", rows)
            print("rows_per_scan=", rows_per_scan)
            print("shape of the_data=", the_data.shape)

        the_data_np = the_data.to_numpy()
        # the_data_np[the_data_np == fill_value] = np.nan
        #the_data_np = np.ma.masked_where(the_data_np == fill_value, the_data_np)
        #np.ma.set_fill_value(the_data_np,np.nan)

        num_valid_points, gridded_data= ewa.fornav(
            cols=cols, rows=rows, area_def=grid_def,
            data_in=the_data_np,
            rows_per_scan=rows_per_scan,fill=fill_value
        )
        the_result = gridded_data
    return the_result, the_ue, the_ue_count


def _get_layers_from_dimension(the_v, the_layer_dims):
    """
        Generate the list of layers for a given variable. 
        @param: the_v - the variable
        @param: the_layer_dims list of dimensions for layers.
    """

    the_indices = [{}]
    for i in range(the_v.ndim):
        the_dim = the_v.dims[i]
        if the_dim in the_layer_dims:
            the_indices = _expand_indices_1_dim(
                the_indices, i, the_dim, the_v.shape[i])
    the_ret_layers = _form_slice_indices(the_indices, the_v.ndim)
    return the_ret_layers


def _get_layers(the_v, the_dims):
    """
        Generate the list of layers for a given variable.
    """

    the_indices = [{}]
    for i in range(the_v.ndim):
        the_dim = the_v.dims[i]
        if not the_dim in the_dims:
            the_indices = _expand_indices_1_dim(
                the_indices, i, the_dim, the_v.shape[i])
    the_ret_layers = _form_slice_indices(the_indices, the_v.ndim)
    return the_ret_layers

def _get_layer_name_suffix(the_idx):
    the_ret = "_layer"
    for key, val in the_idx.items():
        the_ret += "_D_"+str(key)+"_v_"+str(val)
    return the_ret


def _form_slice_indices(the_indices, the_ndim):
    the_ret = []
    for the_indx in the_indices:
        ix = [the_indx.get(dim, slice(None)) for dim in range(the_ndim)]
        the_layer_name = _get_layer_name_suffix(the_indx)
        adict = {}
        adict["lname"]=the_layer_name
        adict["slice"]=tuple(ix)
        the_ret.append(adict)
    return the_ret

def _expand_indices_1_dim(the_indices, the_index, the_dim, the_size):
    the_ret = []
    for item in the_indices:
        for i in range(the_size):
            adict = copy.deepcopy(item)
            adict[the_index] = i
            the_ret.append(adict)
    return the_ret

def _to_geotiff_get_variables(the_rs, the_ref_axis):
    the_ret=list(the_rs.data_vars)
    for var_name in list(the_rs.data_vars):
        if list(the_rs[var_name].coords) != list(the_rs.coords):
            the_ret.remove(var_name)
            continue
        if the_rs[var_name].ndim < the_ref_axis.ndim:
            the_ret.remove(var_name)
            continue
        for the_dim in the_ref_axis.dims:
            if not (the_dim in the_rs[var_name].dims):
                    the_ret.remove(var_name)
                    break 
    return the_ret
