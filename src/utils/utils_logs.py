from termcolor import colored


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
