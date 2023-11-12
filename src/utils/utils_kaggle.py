from pathlib import Path
import shutil
import subprocess
import os


def setup_kaggle(kaggle_json_path: Path):
    kaggle_path = Path("~/.kaggle").expanduser()
    if not kaggle_path.exists():
        print("Creating ~/.kaggle directory...")
        kaggle_path.mkdir(parents=True)

    target_json = kaggle_path / "kaggle.json"
    if not target_json.exists():
        print("Moving kaggle.json to ~/.kaggle...")
        shutil.move(kaggle_json_path, target_json)

    if not (target_json.stat().st_mode & 0o600 == 0o600):
        print("Setting permissions for kaggle.json...")
        target_json.chmod(0o600)

    try:
        import kaggle

        print("Kaggle package is already installed.")
    except ImportError:
        print("Installing Kaggle package...")
        subprocess.run(["pip", "install", "-q", "kaggle"])


# if download_kaggle_data:
#     dataset_name = "ravi20076/optiver-memoryreduceddatasets"
#     kaggle_json_path = os.path.join(path_project_dir, "kaggle.json")
#     get_data(
#         kaggle_json_path,
#         path_data_project_dir,
#         dataset_name=dataset_name,
#         specific_file=None,
#     )


def download_data(
    dest_folder, competition_name=None, dataset_name=None, specific_file=None
):
    if specific_file:
        if dataset_name:
            target_file_path = os.path.join(dest_folder, specific_file)

            if not os.path.exists(target_file_path):
                print(
                    f"Downloading specific file {specific_file} from dataset {dataset_name}..."
                )
                subprocess.run(
                    [
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        dataset_name,
                        "-f",
                        specific_file,
                        "-p",
                        dest_folder,
                    ]
                )

                # Unzip the specific file
                subprocess.run(
                    ["unzip", f"{dest_folder}/{specific_file}", "-d", dest_folder]
                )

                subprocess.run(["rm", f"{dest_folder}/{specific_file}.zip"])
            else:
                print(f"File {specific_file} already exists. Skipping download.")

        else:
            print("To download a specific file, dataset_name must also be provided.")
            return
    else:
        if competition_name:
            target_folder_path = os.path.join(dest_folder, competition_name)

            if not os.path.exists(target_folder_path):
                print(f"Downloading all files from competition {competition_name}...")
                subprocess.run(
                    [
                        "kaggle",
                        "competitions",
                        "download",
                        "-c",
                        competition_name,
                        "-p",
                        dest_folder,
                    ]
                )
                subprocess.run(
                    [
                        "unzip",
                        f"{dest_folder}/{competition_name}.zip",
                        "-d",
                        dest_folder,
                    ]
                )
                subprocess.run(["rm", f"{dest_folder}/{competition_name}.zip"])
            else:
                print(
                    f"Files from competition {competition_name} already exist. Skipping download."
                )

        elif dataset_name:
            dataset_file_name = dataset_name.split("/")[-1]
            target_folder_path = os.path.join(dest_folder, dataset_file_name)

            if not os.path.exists(target_folder_path):
                print(f"Downloading all files from dataset {dataset_name}...")
                subprocess.run(
                    [
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        dataset_name,
                        "-p",
                        dest_folder,
                    ]
                )
                subprocess.run(
                    [
                        "unzip",
                        f"{dest_folder}/{dataset_file_name}.zip",
                        "-d",
                        dest_folder,
                    ]
                )
                subprocess.run(["rm", f"{dest_folder}/{dataset_file_name}.zip"])
            else:
                print(
                    f"Files from dataset {dataset_name} already exist. Skipping download."
                )

        else:
            print(
                "Either competition_name or dataset_name must be provided to download data."
            )


def make_directories(dir_path: Path):
    if not dir_path.exists():
        print(f"Creating directory {dir_path}...")
        dir_path.mkdir(parents=True)
    else:
        print(f"Directory {dir_path} already exists.")


def get_data(
    kaggle_json_path,
    dest_folder,
    competition_name=None,
    dataset_name=None,
    specific_file=None,
):
    setup_kaggle(kaggle_json_path)
    make_directories(dest_folder)
    download_data(
        dest_folder,
        competition_name=competition_name,
        dataset_name=dataset_name,
        specific_file=specific_file,
    )


def clean_directory_except_one(dir_path, file_to_keep):
    """
    Remove all files and folders in a directory except for one specified file.

    Parameters:
    - dir_path (str): The path of the directory to clean.
    - file_to_keep (str): The name of the file to keep.

    clean_directory_except_one('/kaggle/working/', 'submission.csv')
    """
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Loop through each file and folder in the directory
        for filename in os.listdir(dir_path):
            # Skip the file you want to keep
            if filename == file_to_keep:
                continue

            file_path = os.path.join(dir_path, filename)

            # Remove file or directory
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        print(
            f"All files and folders in {dir_path} have been removed, except for {file_to_keep}."
        )
    else:
        print(f"Directory {dir_path} does not exist.")
