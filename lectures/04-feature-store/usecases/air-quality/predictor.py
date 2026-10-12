"""Predictor script for the pm25forecast deployment (Hopsworks Python model server).

A request carries keys only, [{"station_id": ...}, ...]. The predictor looks up each station's
latest feature vector in the online store through the model's feature view and returns the
predicted PM2.5 of that day, in µg/m³, per key.
"""

import os

import joblib


class Predictor:
    def __init__(self, model):
        self.model = joblib.load(os.path.join(os.environ["MODEL_FILES_PATH"], "model.pkl"))
        # inside a deployment this initialises online serving
        self.fv = model.get_feature_view()

    def predict(self, inputs):
        X = self.fv.get_feature_vectors(entry=inputs, return_type="pandas")
        return self.model.predict(X[self.model.feature_names_in_]).round(2).tolist()
