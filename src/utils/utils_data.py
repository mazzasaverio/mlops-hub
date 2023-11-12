import yaml
import pandas as pd
import numpy as np
from pathlib import Path


class PathManager:
    def __init__(self, path_project_dir):
        self.path_root_project = Path(path_project_dir)

        self.path_data_dir = None
        self.path_experiments_dir = None
        self.name_train_file = None
        self.name_test_file = None

        self.name_project_dir = None

        self.path_experiments_dir = (
            "http://ec2-13-38-228-107.eu-west-3.compute.amazonaws.com:5000"
        )
        self.path_artifact_location = (
            f"s3://mlflow-v1/kaggle_optiver_trading_at_the_close"
        )

        self.path_model_production = None

        self.initialize_paths()

    def initialize_paths(self):
        if self.path_root_project.name == "working":
            self.setup_kaggle_paths()
        else:
            self.setup_local_paths()

    def setup_kaggle_paths(self):
        self.name_project_dir = "optiver-trading-at-the-close"

        self.name_train_file = "train.csv"
        self.name_test_file = "test.csv"

        self.data_dir = Path("/kaggle/input") / self.name_project_dir
        self.path_data_train_raw = self.data_dir / self.name_train_file
        self.path_data_test_raw = self.data_dir / self.name_test_file

        self.path_dataset_processed = "/kaggle/working/processed_data"

        self.path_model_production = Path("/kaggle/input")
        # path_dataset_train = os.path.join(path_dataset_processed, "train.csv")
        # path_dataset_test = os.path.join(path_dataset_processed, "test.csv")

    def setup_local_paths(self):
        self.name_project_dir = "kaggle_optiver_trading_at_the_close"
        self.data_dir = self.path_root_project / "data" / self.name_project_dir

        self.path_data_train_raw = self.data_dir / "raw" / "train.csv"
        self.path_data_test_raw = self.data_dir / "raw" / "test.csv"

        self.path_dataset_processed = self.path_root_project / "data" / "processed"

        self.path_model_production = self.path_root_project
        # path_dataset_train = path_dataset_processed / "train.csv"
        # path_dataset_test = path_dataset_processed / "test.csv"


def load_config(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Configuration file {path} not found. Exiting.")
        raise e


def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        print(f"Data file {path} not found. Exiting.")
        raise e


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
            print(f"Memory usage of dataframe is {start_mem:.2f} MB")
            end_mem = df.memory_usage().sum() / 1024**2
            print(f"Memory usage after optimization is: {end_mem:.2f} MB")
            decrease = 100 * (start_mem - end_mem) / start_mem
            print(f"Decreased by {decrease:.2f}%")

    return df
