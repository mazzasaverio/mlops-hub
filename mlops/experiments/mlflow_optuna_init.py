# mlflow_optuna_init.py
import mlflow
import optuna

def initialize_mlflow(path:str, config: dict):

    mlflow.set_tracking_uri(path)
    mlflow.set_experiment(config.get('experiment_name', 'Default_Experiment'))

def initialize_optuna(path:str, config: dict):

    try:
        study = optuna.create_study(
            study_name=config.get('study_name', 'Default_Study_Name'),
            storage=f"sqlite:///{path}/optuna.db",
            direction="maximize",
            load_if_exists=True
        )
        return study
    except Exception as e:
        print(f"Could not initialize Optuna study. Error: {e}")
