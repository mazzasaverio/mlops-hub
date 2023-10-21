# utils.py
import os
import sys
import yaml
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_project_directory():
    current_directory = os.getcwd()
    
    # Define environment-specific directories
    env_directories = {
        '/content': '/content/',
        '/kaggle/working': '/kaggle/'
    }
    
    return env_directories.get(current_directory, os.getenv("ROOT_PATH"))


def load_config(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file {path} not found. Exiting.")
        sys.exit()

def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Data file {path} not found. Exiting.")
        sys.exit()




def prepare_features_and_target(df_train: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    cols_feat = [c for c in df_train.columns if c not in ['datetime', 'target', 'ticker']]
    X = df_train[cols_feat]
    y = df_train['target']
    y = y.map({-1: 0, 0: 1, 1: 2})
    return X, y