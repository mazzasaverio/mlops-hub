import mlflow
import pandas as pd
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError
import os
import pickle

from mlflow.exceptions import MlflowException

from collections import OrderedDict
from utils_models import EnsembleModel


def get_or_create_experiment(client, experiment_name, artifact_location):
    """
    Get the ID of an existing MLflow experiment with the given name or create a new one if it
    does not exist.

    Parameters:
    experiment_name (str): The name of the experiment.
    artifact_location (str): The location for storing artifacts for the experiment.

    Returns:
    str: The experiment ID of the existing or newly created experiment.
    """

    try:
        # Check if the experiment already exists
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(
                f"Experiment '{experiment_name}' already exists with ID {experiment_id}."
            )
        else:
            # If the experiment does not exist, create it
            experiment_id = client.create_experiment(
                name=experiment_name, artifact_location=artifact_location
            )
            print(f"Created new experiment with ID {experiment_id}.")

    except MlflowException as e:
        raise e

    return experiment_id


def log_model_parameters(model, priority_params, excluded_params, verbose=None):
    """
    Logs the model's parameters to MLflow, with priority parameters logged first.

    Parameters:
    model: The model object with a get_params() method.
    priority_params (list): A list of parameter names to log first.
    excluded_params (list): A list of parameter names to exclude from logging.
    """
    params_to_log = model.get_params()

    # Create an OrderedDict to keep the priority parameters first
    ordered_params = OrderedDict()

    # Add the priority parameters with rounding if they are floats or ints
    for key in priority_params:
        if key in params_to_log:
            value = params_to_log[key]
            if isinstance(value, (int, float)):
                ordered_params[key] = round(value, 5)
            else:
                ordered_params[key] = value

    # Add the remaining parameters, excluding the ones in excluded_params and already added priority keys
    for key, value in params_to_log.items():
        if key not in excluded_params and key not in priority_params:
            if isinstance(value, (int, float)):
                ordered_params[key] = round(value, 5)
            else:
                ordered_params[key] = value

    if verbose:
        formatted_params = " | ".join(
            f"{key}: {value}" for key, value in ordered_params.items()
        )
        print(f"\n{formatted_params}\n")
    return ordered_params


def download_and_load_model(s3_client, s3_path):
    """
    Function to download and load a model from S3
    """
    bucket_name = s3_path.split("/")[2]
    object_key = "/".join(s3_path.split("/")[3:])
    try:
        with open("temp_model.pkl", "wb") as f:
            s3_client.download_fileobj(bucket_name, object_key, f)
        with open("temp_model.pkl", "rb") as f:
            model = pickle.load(f)
        os.remove("temp_model.pkl")
        print(f"Model at {s3_path} loaded successfully.")
        return model
    except NoCredentialsError:
        print("Credentials not available.")
    except Exception as e:
        print(f"Error occurred while loading model from {s3_path}: {e}")


def load_models_and_create_ensemble(s3_client, model_paths):
    """
    Function to load all models and create an ensemble model.
    """
    loaded_models = []
    for path in model_paths:
        model = download_and_load_model(s3_client, path)
        if model is not None:
            loaded_models.append(model)
    return EnsembleModel(loaded_models)


def save_and_register_model(ensemble_model, model_name):
    """
    Function to save and register the ensemble model
    """
    temp_ensemble_path = "ensemble_model.pkl"
    with open(temp_ensemble_path, "wb") as f:
        pickle.dump(ensemble_model, f)

    with mlflow.start_run() as run:
        mlflow.log_artifact(temp_ensemble_path, "model")
        run_id = run.info.run_id

        # Records the model in the Model Registry
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)

    # Clean up the local file system
    # if os.path.exists(temp_ensemble_path):
    #     os.remove(temp_ensemble_path)

    print(f"Ensemble model registered under run_id: {run_id}")


def get_experiments_df(client):
    experiments = client.search_experiments()
    data = []
    for exp in experiments:
        exp_detail = {
            "Experiment ID": exp.experiment_id,
            "Creation Time": datetime.fromtimestamp(exp.creation_time / 1000.0),
            "Name": exp.name,
            "Artifact Location": exp.artifact_location,
            "Lifecycle Stage": exp.lifecycle_stage,
        }
        data.append(exp_detail)

    df = pd.DataFrame(data)
    return df


def delete_runs_and_artifacts(client, experiment_ids_to_remove, bucket_name):
    """
    Deletes all the runs and their associated artifacts for the experiments listed in `experiment_ids_to_remove`
    Deletes MLflow runs with a 'FAILED' status or with a 'debug' tag set to 'True' for all others experiments.
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    for exp in client.search_experiments():
        exp_id = exp.experiment_id
        if exp_id not in experiment_ids_to_remove:
            runs = client.search_runs([exp.experiment_id], "")
            for run in runs:
                # Check run status and tags
                if run.info.status == "FAILED" or (
                    run.data.tags.get("debug") == "True"
                ):
                    try:
                        # Delete artifacts from S3 corresponding to the run ID
                        artifact_uri = run.info.artifact_uri
                        if "s3://" in artifact_uri:
                            artifact_path = artifact_uri.replace(
                                f"s3://{bucket_name}/", ""
                            )
                            bucket.objects.filter(Prefix=artifact_path).delete()

                        # Delete the run
                        client.delete_run(run.info.run_id)
                        print(f"Deleted run {run.info.run_id} and its artifacts.")

                    except MlflowException as e:
                        print(f"Error deleting run {run.info.run_id}: {e}")
        else:
            try:
                # Get experiment data to find the artifact location
                experiment_data = client.get_experiment(exp_id)
                artifact_uri = experiment_data.artifact_location

                # Assuming the artifact URI is an S3 path
                if "s3://" in artifact_uri:
                    artifact_path = artifact_uri.replace(f"s3://{bucket_name}/", "")

                    # Delete artifacts from S3 corresponding to the experiment ID
                    # It's crucial that the artifact_path is specific to the experiment
                    if exp_id in artifact_path:
                        bucket.objects.filter(Prefix=artifact_path).delete()

                # Delete the experiment from MLflow
                client.delete_experiment(exp_id)
                print(f"Deleted experiment {exp_id} and its artifacts.")

            except MlflowException as e:
                print(f"Error deleting experiment {exp_id}: {e}")


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

    return df_runs_new


def list_path_models(client, list_experiment_id, run_name_startswith):
    df_exp = experiments_data(
        client, list_experiment_id=list_experiment_id, save_df=None, list_columns=None
    )

    return list(
        df_exp[df_exp["mlflow.runName"].str.startswith(run_name_startswith)][
            "model_path"
        ]
    )
