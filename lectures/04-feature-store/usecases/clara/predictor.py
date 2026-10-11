"""KServe predictor for clara_s04_surrogate. Inputs: a list of {SETI column: current} dicts."""

import os

import joblib
import pandas as pd


class Predictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(os.environ["MODEL_FILES_PATH"], "model.pkl"))

    def predict(self, inputs):
        X = pd.DataFrame(inputs)[list(self.model.feature_names_in_)]
        return self.model.predict(X).tolist()
