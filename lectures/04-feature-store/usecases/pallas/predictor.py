"""Predictor script for the pallas_direct_b deployment (Hopsworks Python model server).

Request: {"instances": [[P_max, cN2_max, L_inj, dip_frac, x_of], ...]} in the units of the pack
(Pa, fraction, mm, fraction, um). Response: one [E_med_MeV, dE_mad, q_end] per instance,
in MeV, relative spread and coulombs.
"""

import os

import joblib
import numpy as np


class Predictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(os.environ["MODEL_FILES_PATH"], "model.pkl"))

    def predict(self, inputs):
        return self.model.predict(np.asarray(inputs, dtype=float)).tolist()
