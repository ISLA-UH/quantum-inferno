"""
Utilities related to exporting data.
"""

import os


def check_dir(dir_name: str) -> None:
    """
    Check if directory exists, and if not, create it.

    :param dir_name: name of directory to check
    """
    existing_dir: bool = os.path.isdir(dir_name)
    if not existing_dir:
        os.makedirs(dir_name)
