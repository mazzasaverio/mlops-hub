import numpy as np


class EnsembleModel:
    def __init__(self, models):
        self.models = models
        self.feature_names = self.models[0].feature_name_

        # Since all models have the same parameters, take the parameters of the first model
        self.model_params = self.models[0].get_params()

    def predict(self, X, type):
        # Average predictions from all models
        if type == "mean":
            predictions = np.mean([model.predict(X) for model in self.models], axis=0)
        elif type == "median":
            predictions = np.median([model.predict(X) for model in self.models], axis=0)
        else:
            raise ValueError(
                f"Invalid type '{type}' for EnsembleModel.predict(). Valid types are 'mean' and 'median'."
            )
        return predictions

    def get_feature_names(self):
        return self.feature_names

    def get_params(self):
        # Returns the parameters of the first model
        return self.model_params
