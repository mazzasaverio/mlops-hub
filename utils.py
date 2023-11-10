from pathlib import Path
import shutil
from datetime import datetime
import os
import yaml
import pandas as pd
import numpy as np
import mlflow
import subprocess
from termcolor import colored
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------- #
#                                   general                                    #
# ---------------------------------------------------------------------------- #


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


# if download_kaggle_data:
#     dataset_name = "ravi20076/optiver-memoryreduceddatasets"
#     kaggle_json_path = os.path.join(path_project_dir, "kaggle.json")
#     get_data(
#         kaggle_json_path,
#         path_data_project_dir,
#         dataset_name=dataset_name,
#         specific_file=None,
#     )


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


# ---------------------------------------------------------------------------- #
#                              feature importance                              #
# ---------------------------------------------------------------------------- #


def log_feature_importance(trial_number, model, X, fold_n, exp_purpose, exp_date_str):
    """
    Logs the feature importances for a given model and fold number.
    """

    feature_importances = model.feature_importances_
    new_importance_df = pd.DataFrame(
        {"feat": X.columns, f"t{trial_number}_imp_fold_{fold_n+1}": feature_importances}
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


def aggregate_feature_importance(list_files_feat_importance):
    list_of_dfs = []
    for file_path in list_files_feat_importance:
        feature_importance_df = pd.read_csv(file_path)

        folds = [col for col in feature_importance_df.columns if "imp_fold" in col]

        # Normalize by dividing each score by the sum of scores within its respective fold
        for fold in folds:
            fold_sum = feature_importance_df[fold].sum()
            feature_importance_df[fold] = feature_importance_df[fold] / fold_sum

        list_of_dfs.append(feature_importance_df)

    aggregated_df = pd.concat(list_of_dfs, ignore_index=True)

    df_median_importance = aggregated_df.groupby("feat").median().reset_index()

    df_median_importance["feat_imp_overall_mean"] = df_median_importance.loc[
        :, df_median_importance.columns != "feat"
    ].median(axis=1, skipna=True)
    cols = ["feat", "feat_imp_overall_mean"] + [
        col
        for col in df_median_importance.columns
        if col not in ["feat_imp_overall_mean", "feat"]
    ]
    df_median_importance = df_median_importance[cols]

    df_median_importance.sort_values(
        "feat_imp_overall_mean", ascending=False, inplace=True
    )
    return df_median_importance


# ---------------------------------------------------------------------------- #
#                                   plot logs                                  #
# ---------------------------------------------------------------------------- #


def log_training_details(logger, model, trial, model_name):
    logger.info(colored(f"Training model: {model_name}", "blue"))

    dynamic_params = {key: round(value, 4) for key, value in trial.params.items()}

    logger.info(
        colored(
            f"\n| "
            + " | ".join(f"{key}: {value}" for key, value in dynamic_params.items()),
            "green",
        )
    )


# ---------------------------------------------------------------------------- #
#                                   models                                     #
# ---------------------------------------------------------------------------- #


def create_model(trial, model_class, static_params, dynamic_params):
    dynamic_params_values = {}
    for param_name, suggestions in dynamic_params.items():
        suggestion_type = suggestions["type"]
        if suggestion_type == "int":
            dynamic_params_values[param_name] = trial.suggest_int(
                param_name, suggestions["low"], suggestions["high"]
            )
        elif suggestion_type == "float":
            dynamic_params_values[param_name] = trial.suggest_float(
                param_name, suggestions["low"], suggestions["high"]
            )
        elif suggestion_type == "categorical":
            dynamic_params_values[param_name] = trial.suggest_categorical(
                param_name, suggestions["choices"]
            )
        elif suggestion_type == "discrete_uniform":
            dynamic_params_values[param_name] = trial.suggest_discrete_uniform(
                param_name, suggestions["low"], suggestions["high"], suggestions["q"]
            )
        elif suggestion_type == "loguniform":
            dynamic_params_values[param_name] = trial.suggest_loguniform(
                param_name, suggestions["low"], suggestions["high"]
            )
        else:
            raise ValueError(f"Unsupported suggestion type: {suggestion_type}")

    model_params = {**static_params, **dynamic_params_values}
    return model_class(**model_params)


# ---------------------------------------------------------------------------- #
#                                    mlflow                                    #
# ---------------------------------------------------------------------------- #
def experiments_data(client, list_experiment_id=None, save_df=None, list_columns=None):
    """
    Every time this function is called, it reads all experiments and a new version of the file returns with all the historical experiments
    """
    experiments = client.search_experiments()
    all_runs_data = []
    for exp in experiments:
        experiment_id = exp.experiment_id
        if (list_experiment_id == None) or (experiment_id in list_experiment_id):
            run_infos = client.search_runs(experiment_ids=[experiment_id])

            for run_info in run_infos:
                run_data = {
                    "experiment_id": experiment_id,
                    "experiment_name": exp.name,
                    "run_id": run_info.info.run_id,
                }

                # Add metrics to run_data
                for key, value in run_info.data.metrics.items():
                    run_data[f"{key}"] = value

                # Add params to run_data
                for key, value in run_info.data.params.items():
                    run_data[f"{key}"] = value

                # Add tags to run_data
                for key, value in run_info.data.tags.items():
                    run_data[f"{key}"] = value

                all_runs_data.append(run_data)

    df_runs_new = pd.DataFrame(all_runs_data)

    if list_columns:
        df_runs_new = df_runs_new[list_columns]

    if save_df:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_filename = f"df_runs_{timestamp}.csv"
        df_runs_new.to_csv(csv_filename, index=False)

        print(f"DataFrame saved to {csv_filename}, Shape: {df_runs_new.shape}")

    return df_runs_new


# ---------------------------------------------------------------------------- #
#                              reduce memory usage                             #
# ---------------------------------------------------------------------------- #


def reduce_mem_usage(df, verbose=0):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        if col not in ["time_id", "date_id", "target", "stock_id", "seconds_in_bucket"]:
            col_type = df[col].dtype

            if (col_type != object) and (col != "target"):
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)

        if verbose:
            logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
            end_mem = df.memory_usage().sum() / 1024**2
            logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
            decrease = 100 * (start_mem - end_mem) / start_mem
            logger.info(f"Decreased by {decrease:.2f}%")

    return df
