# 05 · Simulation campaigns: runs to surrogate to optimiser

For the optimisation (CLEAR, CLARA) and surrogate (PALLAS) teams, 30 minutes. Every simulation run becomes a row with its inputs, outputs, lattice and code version. The surrogate trains from a feature view over those rows, the optimiser calls it through a deployment.

The lattice and beam are the ones from lecture 02, notebook 03: drift, QF, drift, QD, drift, screen; 150 MeV protons; k1 in [-15, 15] T/m².

## Run a campaign

```bash
uv pip install cheetah-accelerator torch
python generators/sim_campaign.py --runs 2000 --campaign scan_a --insert quad_scan_runs
```

Feature group `quad_scan_runs`, primary key `run_id`, event time `run_ts`, partitioned by `campaign`:

| column | meaning |
|---|---|
| `k1_qf`, `k1_qd` | quadrupole strengths, T/m² |
| `sigma_x_mm`, `sigma_y_mm` | beam size at the screen |
| `diverged` | the beam blew up |
| `lattice_hash`, `cheetah_version` | what produced the row |

Run a second campaign with another seed or range, same feature group. Share it with the other teams (track 02): their surrogate trains on your runs.

## Feature view per target

```python
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
from hsfs.builtin_transformations import standard_scaler

runs = fs.get_feature_group("quad_scan_runs", version=1)

query = runs.select(["k1_qf", "k1_qd", "sigma_x_mm", "sigma_y_mm"]).filter(runs.diverged == False)

fv = fs.get_or_create_feature_view(
    name="quad_surrogate", version=1,
    query=query, labels=["sigma_x_mm", "sigma_y_mm"],
    transformation_functions=[standard_scaler("k1_qf"), standard_scaler("k1_qd")],
)
X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
```

The offline copy of an online group lands a minute or two after the insert. If `train_test_split` fails with "Could not read data using Hopsworks Query Service", wait and run it again.

The filter and the scaling live in the view. The scaler learns its mean and deviation from this training data only, and the deployment applies the same numbers. Anyone who reads `quad_surrogate` v1 gets the same rows, including the runs added after it was created.

## Surrogate

```python
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import joblib, os

surrogate = MLPRegressor(hidden_layer_sizes=(128, 128, 128), max_iter=2000)
surrogate.fit(X_train, y_train)
r2 = r2_score(y_test, surrogate.predict(X_test))

os.makedirs("surrogate", exist_ok=True)
joblib.dump(surrogate, "surrogate/model.pkl")

mr = project.get_model_registry()
mr.python.create_model(
    name="quad_surrogate", metrics={"r2": r2},
    feature_view=fv, input_example=X_train.head(1),
).save("surrogate")
```

The notebook 03 PyTorch MLP works too: register it with `mr.torch.create_model` and a `predictor.py`.

## Optimise against the deployment

A Python model is served by a predictor script. Save this as `predictor.py`: it takes the two strengths, applies the view's scaler with the statistics of the training data, and predicts.

```python
import os
import joblib
import pandas as pd


class Predictor:
    def __init__(self, model):
        self.model = joblib.load(os.path.join(os.environ["MODEL_FILES_PATH"], "model.pkl"))
        # In a deployment this also loads the scaler statistics from the training data.
        self.fv = model.get_feature_view()

    def predict(self, inputs):
        return self.model.predict(self.fv.transform(pd.DataFrame(inputs))).tolist()
```

```python
import os

entry = project.get_model_registry().get_model("quad_surrogate", version=1)
uploaded = project.get_dataset_api().upload("predictor.py", "Resources", overwrite=True)
deployment = entry.deploy(name="quadsurrogate", script_file=os.path.join("/Projects", project.name, uploaded))
deployment.start(await_running=600)

from skopt import gp_minimize
from skopt.space import Real

def beam_size(k):
    sx, sy = deployment.predict(inputs=[{"k1_qf": k[0], "k1_qd": k[1]}])["predictions"][0]
    return sx + sy

best = gp_minimize(beam_size, [Real(-15, 15), Real(-15, 15)], n_calls=60, random_state=42)
print(best.x, best.fun)
```

Validate `best.x` with one real Cheetah run and write that run back to `quad_scan_runs`, tagged `--campaign bo_check`. The next surrogate trains on it.

## Your own simulator

Same pattern for Smilei (PALLAS) or your facility's lattice: one row per run, inputs and outputs as columns, a hash of the input deck, the code version. Keep the heavy output (fields, particle dumps) in files and store the path in the row.
