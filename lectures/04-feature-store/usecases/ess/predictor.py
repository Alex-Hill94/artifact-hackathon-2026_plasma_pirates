"""Predictor script for the ess_llrf_anomaly deployment.

A request carries keys only, [{"event_id": ..., "cycle_id": ...}, ...]. The predictor looks up
each feature vector in the online store through the model's feature view and returns
[anomaly_score, flagged] per cycle: score is the flipped IsolationForest score (higher is more
anomalous), flagged is true above the threshold set by the training contamination.
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
        X = X[self.model.feature_names_in_]
        scores = -self.model.score_samples(X)
        flags = self.model.predict(X) == -1
        return [[float(s), bool(f)] for s, f in zip(scores, flags)]
