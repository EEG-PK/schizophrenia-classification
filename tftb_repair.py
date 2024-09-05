import os
import sys
import shutil
from typing import Optional


def find_tftb_dir(base_lib_path: str) -> str:
    """
    Recursively search for the 'tftb' directory within the specified base library path.

    :param base_lib_path: Path to the base library directory.
    :return: Path to the 'tftb' directory if found.
    :raises FileNotFoundError: If the 'tftb' directory is not found.
    """
    for root, dirs, _ in os.walk(base_lib_path):
        if 'site-packages' in root and 'tftb' in dirs:
            return os.path.join(root, 'tftb')
    raise FileNotFoundError("tftb directory not found in the specified base library path.")


def copy_and_replace(local_path: str, target_path: str) -> None:
    """
    Copy a file from the local path to the target path, replacing the target file if it exists.

    :param local_path: Path to the local file to be copied.
    :param target_path: Path to the target location where the file should be copied.
    :raises FileNotFoundError: If the local file does not exist.
    :raises SystemExit: If an unexpected error occurs during the file copy process.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Required file '{local_path}' not found.")
    try:
        shutil.copyfile(local_path, target_path)
        print(f"The file '{target_path}' has been successfully updated.")
    except Exception as e:
        print(f"Unexpected error when copying the file: {local_path} - {e}")
        sys.exit(1)


def main() -> None:
    """
    Main function to update 'tftb' library files in the active Conda or venv environment.

    The function checks if the script is run within an active Conda or venv environment,
    locates the 'tftb' library directory, and replaces specific files with updated versions.

    :raises SystemExit: If the script is not run within an active Conda or venv environment
                        or if required directories/files are not found.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    venv_prefix = os.environ.get("VIRTUAL_ENV")

    if conda_prefix:
        base_lib_path = os.path.join(conda_prefix, 'lib')
    elif venv_prefix:
        base_lib_path = os.path.join(venv_prefix, 'lib')
    else:
        print("Error: The script must be run within an active Conda or venv environment.")
        sys.exit(1)

    base_dir = find_tftb_dir(base_lib_path)

    repair_dir = os.path.join(os.path.dirname(__file__), 'tftb_repair')
    file_paths = {
        'utils.py': os.path.join(repair_dir, 'tftb_generators_utils.py'),
        'base.py': os.path.join(repair_dir, 'tftb_processing_base.py'),
        'cohen.py': os.path.join(repair_dir, 'tftb_processing_cohen.py')
    }

    # Update all files
    for file_name, local_file_path in file_paths.items():
        # Determine the location of the file based on the path
        if file_name == 'utils.py':
            target_file_path = os.path.join(base_dir, 'generators', file_name)
        else:
            target_file_path = os.path.join(base_dir, 'processing', file_name)

        if not os.path.exists(os.path.dirname(target_file_path)):
            raise FileNotFoundError(f"Target directory '{os.path.dirname(target_file_path)}' does not exist.")

        copy_and_replace(local_file_path, target_file_path)


if __name__ == "__main__":
    main()
