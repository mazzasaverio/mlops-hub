# utils.py
import os
import sys
import yaml
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


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

