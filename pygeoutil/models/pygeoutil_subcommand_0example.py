"""
    A subcommand plugin.
"""

def add_cli_arg(the_main_parsers):
    """
        A subcommand plugin.
    """
    #print("Add argument here - ", the_main_parsers)
    the_subcommand = the_main_parsers.add_parser("0example",
                                        description="This is a template. 'example0' will be "
                                        "the name of the subcommand to be added. There should be "
                                        "a file with name 'pygeoutil.models.pygeoutil_subcommand_0example.py' "
                                        "to support the command",
                                        help="Mosaic a stack of files with the same extent")
    the_subcommand.add_argument('-a', '--argument', nargs=1, type=str,
                                required=True,
                                help="Add any argument as necessary.")


def process(the_args):
    """
        A subcommand plugin - process.
    """
    print("This will be the entrypoint to process.")
    print("Example to process args=", the_args)
