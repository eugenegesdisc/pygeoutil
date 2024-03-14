"""
    The module is to search for modules under "pygeoutil.modles"
    starting with "pygeoutil_subcommand_".
"""

import importlib
import pkgutil

import pygeoutil.models

def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__+".")

discovered_subcommand_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in iter_namespace(pygeoutil.models)
    if name.startswith("pygeoutil.models."+"pygeoutil_subcommand_")
}
