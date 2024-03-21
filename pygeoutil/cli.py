"""
    The main command line program for pygeoutil.
"""

import argparse
from pygeoutil import plugin_manager


def parse_args(prog_name="python cli.py"):
    """
        Parsing the arguments.
    """
    parser = argparse.ArgumentParser(
        prog=prog_name,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Utility programs for preparing geospatial dataset for serving. 
""",
        epilog=f"""
Examples:
    {prog_name} --help 
"""
    )

    # options applicable to all sub-commands
    parser.add_argument(
        '--debug', action="store_true", required=False, default=False,
        help='Enable debug mode')

    # sub-command: enable sum-command
    cmd_parsers = parser.add_subparsers(title="subcommands",
                                        description="A set of sub-commands",
                                        dest="cmd",
                                        required=True,
                                        help="List of sub-commands")
    for plugin_name in plugin_manager.discovered_subcommand_plugins:
        the_plugin=plugin_manager.discovered_subcommand_plugins[plugin_name]
        the_plugin.add_cli_arg(cmd_parsers)

    args = parser.parse_args()
    return args


def process_args(args):
    """
        Processing all the options of arguments.
    """
    if args.debug:
        print("args=", args)
    the_sub_cmd = "pygeoutil.models.pygeoutil_subcommand_"+args.cmd
    if (the_sub_cmd in plugin_manager.discovered_subcommand_plugins):
        the_plugin = plugin_manager.discovered_subcommand_plugins.get(the_sub_cmd)
        if the_plugin is not None:
            the_plugin.process(args)
    #if args.cmd == "geogridtogeotiff":
    #    geo_grid_to_geotiff.to_geotiff(args)
    #elif args.cmd == "gdalgeogridtogeotiff":
    #    gdal_geo_grid_to_geotiff.to_geotiff(args)
    #elif args.cmd == "swathtogeotiff":
    #    swath_to_geotiff.to_geotiff(args)
    #else:
    #    print(f"Unrecognized command {args.cmd}")

def cli(prog_name="python cli.py"):
    """
        Processing all the options of arguments.
    """
    args = parse_args(prog_name)
    if args.debug:
        print("discovered_plugins=",plugin_manager.discovered_subcommand_plugins)
    process_args(args)


if __name__ == '__main__':
    cli(prog_name="python pygeoutil/cli.py")
