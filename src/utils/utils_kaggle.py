import os
import subprocess
import shutil

def setup_kaggle(kaggle_json_path):
    if not os.path.exists(os.path.expanduser("~/.kaggle")):
        print("Creating ~/.kaggle directory...")
        subprocess.run(["mkdir", "-p", "~/.kaggle"])

    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("Moving kaggle.json to ~/.kaggle...")
        subprocess.run(["mv", kaggle_json_path, "~/.kaggle/"])

    file_stat = os.stat(os.path.expanduser("~/.kaggle/kaggle.json")).st_mode
    if not (file_stat & 0o600 == 0o600):
        print("Setting permissions for kaggle.json...")
        subprocess.run(["chmod", "600", "~/.kaggle/kaggle.json"])
    
    try:
        import kaggle
        print("Kaggle package is already installed.")
    except ImportError:
        print("Installing Kaggle package...")
        subprocess.run(["pip", "install", "-q", "kaggle"])
import subprocess

def make_directories(dir_path):
    if not os.path.exists(dir_path):
        print(f"Creating directory {dir_path}...")
        subprocess.run(["mkdir", "-p", dir_path])
    else:
        print(f"Directory {dir_path} already exists.")


def download_data(dest_folder, competition_name=None, dataset_name=None, specific_file=None):
    if specific_file:
        if dataset_name:
            target_file_path = os.path.join(dest_folder, specific_file)
            
            if not os.path.exists(target_file_path):
                print(f"Downloading specific file {specific_file} from dataset {dataset_name}...")
                subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-f", specific_file, "-p", dest_folder])
                
                # Unzip the specific file
                subprocess.run(["unzip", f"{dest_folder}/{specific_file}", "-d", dest_folder])
                
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
                subprocess.run(["kaggle", "competitions", "download", "-c", competition_name, "-p", dest_folder])
                subprocess.run(["unzip", f"{dest_folder}/{competition_name}.zip", "-d", dest_folder])
                subprocess.run(["rm", f"{dest_folder}/{competition_name}.zip"])
            else:
                print(f"Files from competition {competition_name} already exist. Skipping download.")
                
        elif dataset_name:
            dataset_file_name = dataset_name.split('/')[-1]
            target_folder_path = os.path.join(dest_folder, dataset_file_name)
            
            if not os.path.exists(target_folder_path):
                print(f"Downloading all files from dataset {dataset_name}...")
                subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", dest_folder])
                subprocess.run(["unzip", f"{dest_folder}/{dataset_file_name}.zip", "-d", dest_folder])
                subprocess.run(["rm", f"{dest_folder}/{dataset_file_name}.zip"])
            else:
                print(f"Files from dataset {dataset_name} already exist. Skipping download.")
                
        else:
            print("Either competition_name or dataset_name must be provided to download data.")


def get_data(kaggle_json_path, dest_folder, competition_name=None, dataset_name=None, specific_file=None):
    setup_kaggle(kaggle_json_path)
    make_directories(dest_folder)
    download_data(dest_folder, competition_name=competition_name, dataset_name=dataset_name, specific_file=specific_file)

