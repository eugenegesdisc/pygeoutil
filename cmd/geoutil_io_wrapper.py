"""
    A subcommand plugin for utility routine to run program by expanding and updating input/output.
"""
import sys, getopt
import argparse
import subprocess
import ast
import glob
import copy
import pathlib

def main(inargs=None):
    """
        main program
    """
    inputfile = ''
    outputfile = ''
    program, subcommand, args = __retrieve_prog_subcommand(inargs)
    the_args_list = __expand_args_on_input(
        inargs[0], args, program, subcommand)
    for the_args in the_args_list:
        __execute_program(program, the_args)

def __expand_args_on_input(
        wrap_program: str,
        args:list, the_program:str, the_subcommand:str)->list:
    the_ret_args_list = list()
    parser = argparse.ArgumentParser(
        usage= (wrap_program + " [-h] --program \"" + the_program +
                "\" [...] "+" -i INPUT -o OUTPUT [...]"),
        description=("A wrapper executor to execute program and expand on input and output. "
            "(-i INPUT -o OUTPUT) are required. [...] can be any other arguments/options to be "
            "used in running the program. The given output should be directory.")
    )
    if the_subcommand is not None:
        parser = argparse.ArgumentParser(
            usage= (wrap_program + " [-h] --program \"" + the_program+"\" -s "+the_subcommand+
                    " [...] "+the_subcommand+" -i INPUT -o OUTPUT [...]"),
            description=("A wrapper executor to execute program and expand on input and output. "
                "(-i INPUT -o OUTPUT) are required. [...] can be any other arguments/options to be "
                "used in running the program. The given output should be directory.")
        )
    if the_subcommand is not None:
        sc_parsers = parser.add_subparsers(prog="cmds",title="subcommands")
        sc_parser = sc_parsers.add_parser(
            the_subcommand,
            usage= (wrap_program + " [-h] --program \"" + the_program+"\" -s "+the_subcommand+
                    " [...] "+the_subcommand+" -i INPUT -o OUTPUT [...]"),
            description=("A wrapper executor to execute program and expand on input and output. "
                "(-i INPUT -o OUTPUT) are required. [...] can be any other arguments/options to be "
                "used in running the program. The given output should be directory."),
            help="Extract a variable")
        sc_parser.add_argument('-i', '--input', nargs=1, type=str,
                                required=True,
                                help="Input as a dataset source. It can be a dataset for GDAL.")
        sc_parser.add_argument('-o', '--output', nargs=1, type=str,
                                required=True,
                                help=("Name of the output GeoTIFF file. Without extention. tif "
                                        "extension will be appended. If looping through all "
                                        "variables, variable name will be append as well."))
    else:
        parser.add_argument('-i', '--input', nargs=1, type=str,
                                required=True,
                                help="Input as a dataset source. It can be a dataset for GDAL.")
        parser.add_argument('-o', '--output', nargs=1, type=str,
                                required=True,
                                help=("Name of the output GeoTIFF file. Without extention. tif "
                                        "extension will be appended. If looping through all "
                                        "variables, variable name will be append as well."))
    s_args, s_unknown = parser.parse_known_args()

    print("s_args=", s_args, "s_unknown=", s_unknown)

    #the_input = copy.deepcopy(s_args.input[0])
    the_files = glob.glob(s_args.input[0])
    if len(the_files)<1:
        print("Error: cannot find any file with pattern \""
              +s_args.input[0]+"\"")
        return list()
    for the_f in the_files:
        path = pathlib.Path(the_f)
        path2 = pathlib.Path(s_args.output[0]).joinpath(path.stem)
        the_arg = copy.deepcopy(args)
        the_arg = __update_io_in_args(the_arg, the_subcommand, the_f, str(path2))
        if the_arg is not None:
            the_ret_args_list.append(the_arg)
    return the_ret_args_list

def __update_io_in_args(args:list,
                        subcommand:str,
                        new_input:str, new_output:str)->list:
    the_subcommand_idx = 0
    if subcommand:
        the_subcommand_idx = __find_first_in_list([subcommand],args)
    if the_subcommand_idx < 0:
        print("Warning: subcommand \""+subcommand+"\" not found.")
        the_subcommand_idx = 0
    the_input_idx = __find_first_in_list(["-i","--input"],args[the_subcommand_idx:])
    if the_input_idx < 0:
        print("Error: Cannot find input option in args - ", args)
        return None
    the_input_idx = the_input_idx + the_subcommand_idx
    the_output_idx = __find_first_in_list(["-o","--output"],args[the_subcommand_idx:])
    if the_output_idx < 0:
        print("Error: Cannot find output option in args - ", args)
        return None
    the_output_idx = the_output_idx + the_subcommand_idx
    the_ret_args = args
    the_ret_args[the_input_idx+1] = new_input
    the_ret_args[the_output_idx+1] = new_output
    return the_ret_args


def __execute_program(program, args)->bool:
    the_command = [program]
    if args is not None:
        the_command = the_command + args
    the_cmd_to_run = __join_command_list(the_command) #" ".join(the_command)
    result = subprocess.run(
        the_cmd_to_run, shell=True, check=True)
    print("Executed result = ", result)
    return True

def __join_command_list(the_command_list:list)->str:
    the_ret_string = ""
    i=0
    for the_cmd in the_command_list:
        if i==0:
            the_ret_string = the_cmd
            i = i + 1
            continue
        if the_cmd[0] == '-':
            the_ret_string = the_ret_string + " " + the_cmd
        elif __test_list_string_in_string([" ","\\","/","*","?"], the_cmd):
            the_ret_string = the_ret_string + " " + "\""+the_cmd+"\""
        else:
            the_ret_string = the_ret_string + " " + the_cmd
    return the_ret_string

def __test_list_string_in_string(the_strings:list, the_string:str)->bool:
    for the_str in the_strings:
        if the_str in the_string:
            return True
    return False

def __retrieve_prog_subcommand(inargs:list)->tuple[str,str,list]:
    the_ret_prog = None
    the_ret_subcommand = None
    the_ret_args = None
    parser = argparse.ArgumentParser(
        usage=inargs[0] + " [-h] --program [PROGRAM ...] [--subcommand SUBCOMMAND] [...]",
        description=("A wrapper executor to execute program and expand on input and output. "
            "[...] can be any arguments/options to be used in running the program, inlcuding "
            "subcommand, input and output options (-i INPUT -o OUTPUT ). The given output should "
            "be directory.")
    )
    parser.add_argument('--program', action="append", nargs='?', type=str,
                        required=True,
                        help="Program name")
    parser.add_argument('--subcommand', action="append", nargs='?', type=str,
                        help="optional subcommand")
    s_args = inargs
    if len(inargs)>6:
        s_args = inargs[:6]
    ss_args, ss_unknown = parser.parse_known_args(s_args)
    if ss_args.program[0] is None:
        print("Error: The first program value is None")
        parser.parse_args(['-h'])
        exit()
    the_ret_program = ss_args.program[0]
    the_pg_index = __find_first_in_list(['--program'], inargs)

    if ss_args.subcommand is not None:
        if ss_args.subcommand[0] is None:
            print("Error: The first subcommand value is None")
            parser.parse_args(['-h'])
            exit()
        the_ret_subcommand = ss_args.subcommand[0]
    the_sc_index = __find_first_in_list(['--subcommand'], inargs)

    the_index = the_pg_index
    if the_sc_index > the_index:
        the_index = the_sc_index
    the_index = the_index + 2
    if len(inargs) > the_index:
        the_ret_args = inargs[the_index:]

    return the_ret_program, the_ret_subcommand, the_ret_args

def __find_first_in_list(param:list[str], inargs:list)->int:
    i = 0
    for the_arg in inargs:
        if the_arg in param:
            return i
        i = i+1
    return -1

if __name__ == "__main__":
    main(sys.argv)
