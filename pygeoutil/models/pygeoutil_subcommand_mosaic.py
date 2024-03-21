"""
    A subcommand plugin.
"""
import glob
import os
import xarray as xr

def add_cli_arg(cmd_parsers):
    """
        A subcommand plugin.
    """
    cmd_mosaic = cmd_parsers.add_parser("mosaic",
                                        description="Mosaic a stack of files with exactly "
                                        "the same projection and the same extent. "
                                        "This may be the 2nd step after re-gridding to the "
                                        "same extent.",
                                        help="Mosaic a stack of files with the same extent")
    cmd_mosaic.add_argument('-i', '--input', nargs="+", type=str,
                               required=True,
                               help="Input as a list of file or files. The file name will be "
                               "expanded with glob. In other words, wildcards are aceeptable.")
    cmd_mosaic.add_argument('-o', '--output', nargs=1, type=str,
                               required=True,
                               help="Output file.")
    cmd_mosaic.add_argument('-m', '--method', type=str,
                               required=False, default="max",
                               choices=["min","max","mean","median","count","sum"],
                               help="mosaic method. Default will be assigning the value to "
                               "the max.")
    cmd_mosaic.add_argument('-u', '--uncertainty', nargs="+", type=str,
                               required=False,
                               help="Uncertainty file suffix, such as _ue or _uc."
                               " It will be used as suffix before extension in the input "
                               "to form corresponding files. For example, input /some/path/"
                               "filename.ext, uncertainty=_ue, then the ue file would be "
                               "/some/path/filename_ue.ext")


def process(the_args):
    """
        A subcommand plugin - process.
    """
    if the_args.debug:
        print("the_args=", the_args)
    mosaic(the_args)

def mosaic(the_args):
    """
        The main entrance for mosaic function.
    """
    the_files=[]
    for the_i in the_args.input:
        the_files.extend(glob.glob(the_i))
    if the_args.debug:
        print("the_files=", the_files)
    the_dss = _load_data_sets(the_args,the_files)
    if the_args.debug:
        print("the_ds=", the_dss)
    the_final=xr.merge(the_dss,join="outer",
                       combine_attrs="drop_conflicts",
                       compat="no_conflicts")
    the_same_spatial_ref = _verify_spatial_ref(the_dss)
    the_spatial_ref = the_dss[0]['spatial_ref']
    _mosaic_main_data(the_args,
                       the_same_spatial_ref,
                       the_spatial_ref,
                       the_final)
    # Not implemented for all
    if the_args.uncertainty:
        for the_ue_suffix in the_args.uncertainty:
            print("THE_UE_FILES=",_retrieve_and_save_uncertainty(
                the_args,the_ue_suffix,the_files,
                the_same_spatial_ref,
                the_spatial_ref,
                the_final))

def _get_ue_files(the_args,the_files,the_ue_suffix):
    the_ue_files = list()
    i = 0
    print("TTTTTthe_files=", the_files)
    for the_f0 in the_files:
        the_f0_parts = os.path.splitext(the_f0)
        the_f = the_f0_parts[0]+the_ue_suffix+the_f0_parts[1]
        print("THE_F=", the_f)
        the_ue_files.append(the_f)
    return the_ue_files

def _load_data_sets(the_args,the_files):
    the_dss = list()
    i = 0
    for the_f in the_files:
        the_ds = xr.open_dataset(the_f)
        if the_args.debug:
            print("the_f=",the_f)
            print("the_ds=", the_ds)
            print("variables=",the_ds.variables)
            # the_ds = the_ds.rename({"band_data":"var_"+str(i)})
        i += 1
        the_ds['band']=the_ds['band']*i
        the_dss.append(the_ds)
    return the_dss

def _retrieve_and_save_uncertainty(the_args,
                                   the_ue_suffix,
                                   the_files,
                                   the_same_spatial_ref,
                                   the_spatial_ref,
                                   the_final_data):
    if (the_args.method == 'min'):
        _retrieve_and_save_uncertainty_min(
             the_args, the_ue_suffix, the_files,
             the_same_spatial_ref,
             the_spatial_ref,
             the_final_data)

def  _retrieve_and_save_uncertainty_min(
        the_args, the_ue_suffix, the_files,
        the_same_spatial_ref,
        the_spatial_ref,
        the_final_data):
        the_ue_files = _get_ue_files(the_args,
                                      the_files,
                                      the_ue_suffix)
        if the_args.debug:
            print("the_ue_files=",the_ue_files)
        the_ues = _load_data_sets(the_args,the_ue_files)
        the_final_ue=xr.merge(the_ues,join="outer",
                    combine_attrs="drop_conflicts",
                    compat="no_conflicts")
        if the_same_spatial_ref:
            the_final_ue['spatial_ref']=the_spatial_ref

        the_idx = the_final_data.argmin(
                            dim='band',
                            skipna=False,
                            keep_attrs=True)
        print("the_idx=", the_idx)
        print("")
        print("the_final_ue=", the_final_ue)
        #the_result = the_final_data["band"].isel(
        #    band=the_idx["band_data"]
        #)
        the_result = the_final_ue["band"].isel(
            band=the_idx["band_data"]
        )
        print("THE_RESULT=", the_result)
        #the_result = the_final_ue["band"].isel(
        #    band=the_idx)
        #the_result = xr.DataArray.isel(the_ues,indexers=the_idx)
        #print("THE_RESULT=",the_result)
        the_result.rio.to_raster(
            _get_output_ue_filename(
                the_args.output[0], the_ue_suffix))

def _get_output_ue_filename(out_datafile:str, the_suffix:str)->str:
    the_f0_parts = os.path.splitext(out_datafile)
    the_f = the_f0_parts[0]+the_suffix+the_f0_parts[1]
    return the_f

def _mosaic_main_data(the_args,
                       the_same_spatial_ref,
                       the_spatial_ref,
                       the_final):
    """
        Sub-mosaic function for combining data.
    """
    if the_same_spatial_ref:
        the_final['spatial_ref']=the_spatial_ref
    if the_args.debug:
        print("the_final=", the_final)
        print("final dims:", the_final.dims)
    if the_args.method == 'min':
        the_result = the_final.min(dim='band',skipna=True, keep_attrs=True)
        if the_args.debug:
            print("Writing (min)...", the_args.output[0])
        the_result.rio.to_raster(the_args.output[0])
    elif the_args.method == 'max':
        the_result = the_final.max(dim='band',skipna=True, keep_attrs=True)
        if the_args.debug:
            print("Writing (max)...", the_args.output[0])
        the_result.rio.to_raster(the_args.output[0])
    elif the_args.method == 'mean':
        the_result = the_final.mean(dim='band',skipna=True, keep_attrs=True)
        if the_args.debug:
            print("Writing (mean)...", the_args.output[0])
        the_result.rio.to_raster(the_args.output[0])
    elif the_args.method == 'median':
        the_result = the_final.median(dim='band',skipna=True, keep_attrs=True)
        if the_args.debug:
            print("Writing (median)...", the_args.output[0])
        the_result.rio.to_raster(the_args.output[0])
    elif the_args.method == 'count':
        the_result = the_final.count(dim='band',keep_attrs=True)
        the_result=the_result.astype("uint16")
        if the_args.debug:
            print("Writing (count)...", the_args.output[0])
        the_result.rio.to_raster(the_args.output[0])
    elif the_args.method == 'sum':
        the_result = the_final.sum(dim='band',skipna=True,
                                   min_count=1,keep_attrs=True)
        if the_args.debug:
            print("Writing (sum)...", the_args.output[0])
        the_result.rio.to_raster(the_args.output[0])
    else:
        print("Error: method is not supported - ", the_args.method)
    
def _verify_spatial_ref(the_dss):
    """
        Verify if the list of dataset has the same spatial_ref.
    """
    the_spatial_ref = the_dss[0]['spatial_ref']
    for the_ds in the_dss:
        if the_spatial_ref != the_ds['spatial_ref']:
            return False
    return True
