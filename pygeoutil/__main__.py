"""
    Module __main__: The default entry for module pygeoutil.
"""
# pygeoutil/__main__.py

from pygeoutil import cli, __app_name__

def main():
    """
        The entry main function for module pygeoutil.
    """
    cli.cli(prog_name="python -m "+__app_name__)

if __name__ == '__main__':
    main()
