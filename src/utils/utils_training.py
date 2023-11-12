from datetime import datetime


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
