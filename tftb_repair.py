import os
import sys
import shutil
# import urllib.request


def copy_and_replace(local_path, target_path):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Required file '{local_path}' not found.")
    try:
        shutil.copyfile(local_path, target_path)
        print(f"The file '{target_path}' has been successfully updated.")
    except Exception as e:
        print(f"Unexpected error when copying the file: {local_path} - {e}")
        sys.exit(1)


def main():
    if "CONDA_PREFIX" not in os.environ:
        print("Error: The script must be run within an active Conda environment.")
        sys.exit(1)

    conda_prefix = os.environ["CONDA_PREFIX"]
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # tftb library path
    base_dir = os.path.join(conda_prefix, f'lib/python{py_version}/site-packages/tftb')

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
