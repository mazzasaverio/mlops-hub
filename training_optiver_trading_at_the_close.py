# main.py
from utils.utils_general import load_config, load_dataset
from experiments.mlflow_optuna_init import initialize_mlflow, initialize_optuna
from validation.cv_setup import initialize_cv_method
from experiments.optuna_objective import objective
from utils.utils_general import prepare_features_and_target
import os
from dotenv import load_dotenv
import warnings
# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load environment variables
load_dotenv()
# Load configurations and datasets
config_path = os.path.join(os.getenv('ROOT_PATH'), 'config/train_config.yaml')
config = load_config(config_path)

dataset_path = os.path.join(os.getenv('ROOT_PATH'), 'data/processed/synthetic_ticker_data.csv')
df_train = load_dataset(dataset_path)

print(config_path)
print(dataset_path)
# Prepare features and target
X, y = prepare_features_and_target(df_train)  # Assuming you have this function


# Initialize MLFlow and Optuna
path_experiments_storage = os.path.join(os.getenv('ROOT_PATH', 'default/path/to/config'), 'data/experiments_storage')
print(path_experiments_storage)
initialize_mlflow(path_experiments_storage, config)
study = initialize_optuna(path_experiments_storage, config)
print("SONO QUI")
# Initialize Cross-Validation
splits = initialize_cv_method(config['validation_method'], X, y)

# Optuna Study
study.optimize(lambda trial: objective(trial, splits, X, y), n_trials=config.get('n_trials', 3))
