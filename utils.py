from pathlib import Path
import shutil
import os
import yaml
import pandas as pd
import mlflow
import subprocess
from termcolor import colored
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def load_config(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        logger.error(f"Configuration file {path} not found. Exiting.")
        raise e


def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        logger.error(f"Data file {path} not found. Exiting.")
        raise e


# ---------------------------------------------------------------------------- #
#                                    kaggle                                    #
# ---------------------------------------------------------------------------- #


def setup_kaggle(kaggle_json_path: Path):
    kaggle_path = Path("~/.kaggle").expanduser()
    if not kaggle_path.exists():
        logger.info("Creating ~/.kaggle directory...")
        kaggle_path.mkdir(parents=True)

    target_json = kaggle_path / "kaggle.json"
    if not target_json.exists():
        logger.info("Moving kaggle.json to ~/.kaggle...")
        shutil.move(kaggle_json_path, target_json)

    if not (target_json.stat().st_mode & 0o600 == 0o600):
        logger.info("Setting permissions for kaggle.json...")
        target_json.chmod(0o600)

    try:
        import kaggle

        logger.info("Kaggle package is already installed.")
    except ImportError:
        logger.info("Installing Kaggle package...")
        subprocess.run(["pip", "install", "-q", "kaggle"])


def make_directories(dir_path: Path):
    if not dir_path.exists():
        logger.info(f"Creating directory {dir_path}...")
        dir_path.mkdir(parents=True)
    else:
        logger.info(f"Directory {dir_path} already exists.")


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


# ---------------------------------------------------------------------------- #
#                              feature importance                              #
# ---------------------------------------------------------------------------- #


def log_feature_importance(model, X, fold_n, exp_purpose, exp_date_str):
    """
    Logs the feature importances for a given model and fold number.
    """

    feature_importances = model.feature_importances_
    new_importance_df = pd.DataFrame(
        {"feat": X.columns, f"imp_fold_{fold_n+1}": feature_importances}
    )

    csv_path = f"feat_impor_{exp_purpose}_{exp_date_str}.csv"

    # Check if the CSV already exists
    if os.path.exists(csv_path):
        # If so, read it and merge with the new importance values
        existing_df = pd.read_csv(csv_path)
        importance_df = pd.merge(existing_df, new_importance_df, on="feat", how="outer")
    else:
        # If not, create a new DataFrame
        importance_df = new_importance_df

    # Save the updated DataFrame to CSV
    importance_df.to_csv(csv_path, index=False)

    mlflow.log_artifact(csv_path)


# ---------------------------------------------------------------------------- #
#                                   plot logs                                  #
# ---------------------------------------------------------------------------- #


def log_training_details(logger, model, trial, model_name):
    logger.info(colored(f"Training model: {model_name}", "blue"))

    dynamic_params = {key: value for key, value in trial.params.items()}

    logger.info(
        colored(
            f"Trial {trial.number:<4} | "
            + " | ".join(f"{key}: {value}" for key, value in dynamic_params.items()),
            "green",
        )
    )

    logger.info(f"{'Fold':<5} {'|':<2} {'MAE':<20}")
    logger.info(f"{'-----':<5} {'|':<2} {'--------------------':<20}")
