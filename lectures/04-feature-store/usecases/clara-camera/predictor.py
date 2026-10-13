"""Predictor script for the claras02spot deployment (Hopsworks Python model server).

Request: {"instances": [{<current column>: amps, ...}, ...]}, one dict per setting with the
model's fourteen currents by name. Response: one [mu_x, mu_y, sigma_x, sigma_y] per setting,
in mm on CLA-S02-DIA-CAM-03.
"""

import os

import joblib
import pandas as pd


class Predictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(os.environ["MODEL_FILES_PATH"], "model.pkl"))

    def predict(self, inputs):
        # Ordered by the names the model was fitted with, so the caller's key order is free.
        X = pd.DataFrame(inputs)[list(self.model.feature_names_in_)]
        return self.model.predict(X).tolist()
