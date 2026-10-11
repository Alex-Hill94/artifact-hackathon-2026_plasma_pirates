# 04 · Serve: live events, online features, a running model

20 minutes. Stream synthetic events into the same feature groups, deploy the model from track 03, score events by id as they arrive.

## Stream

```bash
python generators/llrf_stream.py --data data/llrf --n 3            # look first
python generators/llrf_stream.py --data data/llrf --insert --follow
```

Each event copies a real SPIRAL2 event from the same cavity and class, with a new `event_id`, the current time, `synthetic=True` and `source_event_id` pointing at the original. Real inter-event gaps per cavity, compressed 3600x by default (`--speedup`). `--hot CMA07_CAV1` makes one cavity fault five times more often than it really does.

The generator writes online only. Live rows are in RonDB within a second and the offline copy follows on the materialisation job's schedule:

```bash
hops job list
```

## Deploy

The model is registered with its feature view, so a short predictor script does the rest: take an `event_id`, look up the feature vector in the online store, predict. Save it as `predictor.py`:

```python
import os
import joblib


class Predictor:
    def __init__(self, model):
        self.model = joblib.load(os.path.join(os.environ["MODEL_FILES_PATH"], "model.pkl"))
        # In a deployment this also initialises online lookups for the feature view.
        self.fv = model.get_feature_view()

    def predict(self, inputs):
        return self.model.predict(self.fv.get_feature_vectors(entry=inputs, return_type="pandas")).tolist()
```

```python
import os
import hopsworks

project = hopsworks.login()
mr = project.get_model_registry()
entry = mr.get_model("llrf_fault", version=1)

uploaded = project.get_dataset_api().upload("predictor.py", "Resources", overwrite=True)
deployment = entry.deploy(name="llrffault", script_file=os.path.join("/Projects", project.name, uploaded))
deployment.start(await_running=600)
```

```bash
hops deployment list
hops deployment status llrffault
```

## Score

```python
live_id = "syn_cma07_cav1_..."          # copy one from the generator output
deployment.predict(inputs=[{"event_id": live_id}])
```

Same call on a real event, a September 2025 fault (Absence autorisation RF, RF authorization absent):

```python
deployment.predict(inputs=[{"event_id": "cma07_cav1_20250904T151333"}])
# {'predictions': [1]}
```

Training read `llrf_physics` offline, from Delta, as it was on each label's date. Serving reads the same features online, from RonDB, as they are now. One definition, two stores, no skew.

Check against the truth: `source_event_id` tells you which real event a synthetic one copies, and `llrf_labels` holds its label.

## Look up features directly

```bash
hops fv get llrf_fault --version 1 --entry "event_id=<live_id>"
```

## Stretch

- Per-cavity health, a windowed aggregation: events and mean `microphonics_rms` over the last hour per cavity. Write it as `llrf_cavity_health` (primary key `cavity_id`, online) from a scheduled job:

  ```bash
  hops job deploy cavity_health cavity_health.py --cron "0 */5 * ? * *" --run
  ```

- Run the generator with `--hot` on one cavity and watch its health row move.
- Stop the deployment when done: `hops deployment stop llrffault`.
