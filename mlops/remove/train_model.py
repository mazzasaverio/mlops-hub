import os
import sys
import yaml
import warnings
import pandas as pd
import mlflow
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier
from dotenv import load_dotenv
from mlops.validation.model_validation import CombPurgedKFoldCV

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load environment variables
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

def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

config_path = os.path.join(os.getenv('CONFIG_PATH', 'default/path/to/config'), 'config/train_config.yaml')
config = load_config(config_path)

# Ensure directory exists
path_experiments_storage = config['path_experiments_storage']
ensure_directory_exists(path_experiments_storage)

# Initialize MLflow and Optuna
mlflow_artifact_location = os.path.join(path_experiments_storage, "mlruns")
mlflow.set_tracking_uri(mlflow_artifact_location)
mlflow.set_experiment(config.get('experiment_name', 'Default_Experiment'))

try:
    db_path = os.path.join(path_experiments_storage, "optuna.db")
    study = optuna.create_study(
        study_name=config.get('study_name', 'Default_Study_Name'),
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True
    )
    print("Optuna study created.")
except Exception as e:
    print(f"Could not initialize Optuna study. Error: {e}")

# Load and prepare dataset
dataset_path = config.get('dataset_path', '/default/path/to/dataset.csv')
df_train = load_dataset(dataset_path)
cols_feat = [c for c in df_train.columns if c not in ['datetime', 'target', 'ticker']]
X = df_train[cols_feat]
y = df_train['target']

# Re-map target values
y = y.map({-1: 0, 0: 1, 1: 2})

# Initialize Cross-Validation
cv_method = config.get('validation_method', 'kfold')
splits = []

if cv_method == 'comb_purged_kfold':
    cv = CombPurgedKFoldCV(n_splits=3, n_test_splits=1, embargo_td=2)
    t1_ = df_train.index
    t1 = pd.Series(t1_).shift(100).fillna(0).astype(int)
    t2 = pd.Series(t1_).shift(-100).fillna(1e12).astype(int)
    splits = list(cv.split(df_train, pred_times=t1, eval_times=t2))
elif cv_method == 'kfold':
    cv = KFold(n_splits=5)
elif cv_method == 'stratified_kfold':
    cv = StratifiedKFold(n_splits=5)
else:
    print(f"Invalid cross-validation method: {cv_method}. Exiting.")
    sys.exit()

# Objective function for Optuna optimization
def objective(trial):
    with mlflow.start_run() as run:
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
        cv_scores = []
        for train_index, test_index in splits:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)
        
        avg_score = sum(cv_scores) / len(cv_scores)
        
        mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
        mlflow.log_metric('cv_score', avg_score)
        
        return avg_score

# Optimize
study.optimize(objective, n_trials=config.get('n_trials', 10))

# Save the best parameters
best_params = study.best_params
with open("../config/best_params.yaml", 'w') as f:
    yaml.dump(best_params, f)
