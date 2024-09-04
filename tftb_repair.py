import os
import sys
import urllib.request


def download_and_replace(url, local_path):
    try:
        with urllib.request.urlopen(url) as response:
            content = response.read()
            with open(local_path, 'wb', encoding="utf-8") as file:
                file.write(content)
        print(f"The file '{local_path}' has been successfully updated.")
    except urllib.error.URLError as e:
        print(f"Error downloading a file from a URL: {url} - {e.reason}")
    except Exception as e:
        print(f"Unexpected error when downloading a file from a URL: {url} - {e}")


def main(env_name, py_version):
    # tftb corrected files on github
    file_paths = {
        'utils.py': 'https://raw.githubusercontent.com/EEG-PK/schizophrenia-classification/preprocessing/Preprocessing/tftb_repairs/tftb_generators_utils.py',
        'base.py': 'https://raw.githubusercontent.com/EEG-PK/schizophrenia-classification/preprocessing/Preprocessing/tftb_repairs/tftb_processing_base.py',
        'cohen.py': 'https://raw.githubusercontent.com/EEG-PK/schizophrenia-classification/preprocessing/Preprocessing/tftb_repairs/tftb_processing_cohen.py'
    }

    # tftb library path
    base_dir = os.path.expanduser(f'~/miniconda3/envs/{env_name}/lib/python{py_version}/site-packages/tftb')

    # Update all files
    for file_name, url in file_paths.items():
        # Determine the location of the file based on the path
        if file_name == 'utils.py':
            local_file_path = os.path.join(base_dir, 'generators', file_name)
        else:
            local_file_path = os.path.join(base_dir, 'processing', file_name)

        # Check if the directories exist, if not, create them
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        download_and_replace(url, local_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python <script_name.py> <env_name> <python_env_version>")
        sys.exit(1)

    env_name = sys.argv[1]
    py_version = sys.argv[2]
    main(env_name, py_version)
