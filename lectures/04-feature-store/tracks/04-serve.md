# 04 · Deploy and call your model

15 minutes, in the terminal of your own project. The model of track 03 becomes an HTTP endpoint.

```bash
python tracks/own_project.py
```

```
deployment running after 74 s
predict {'predictions': [29.86145796488902]} true 31.26613796465147
```

## What it does

```python
entry = project.get_model_registry().python.create_model(
    name="p1_from_beam", metrics={"r2": r2}, feature_view=fv, input_example=X_train.head(1)
)
entry.save("model")                      # the folder with model.pkl

script = project.get_dataset_api().upload("usecases/pallas/predictor.py", "Resources", overwrite=True)
deployment = entry.deploy(name="p1frombeam", script_file=f"/Projects/{project.name}/{script}")
deployment.start(await_running=600)
deployment.predict(inputs=X_test.head(1).values.tolist())
```

- A Python model needs a predictor script on this version: `deploy()` without `script_file` is refused. `usecases/pallas/predictor.py` loads `model.pkl` and calls `predict`; it fits any scikit-learn model saved that way.
- A second run reuses the deployment `p1frombeam` if it exists. Delete it to serve a newer model version.
- scikit-learn is pinned in `requirements.txt` to the serving image's version, so a model trained on your laptop loads in the deployment.
- Call the deployment from the terminal or from a job. The serving gateway is not reachable from outside the cluster.

## Clean up

```python
deployment.stop()
deployment.delete()
```

A running deployment holds a pod. Stop it when you are done.

## Next: your challenge

Each use case folder holds the same three pipelines on its own data:

| Folder | Feature pipeline | Training pipeline | Inference pipeline |
|---|---|---|---|
| `usecases/ess` | waveform features of every cycle and chain | IsolationForest, the challenge autoencoder | scores cycles by key |
| `usecases/clara` | Cheetah captures (the Easy objective) | MLP surrogate | Bayesian optimisation on the deployment |
| `usecases/clara-camera` | real camera captures | extra trees surrogate | optimiser on the deployment, checked on real bursts |
| `usecases/pallas` | the Smilei pack | Gaussian process, gradient boosting | search for a target beam |

`usecases/pallas` runs on the full data as it is. In the terminal of your own project, after `training_pipeline.py --store m4cast_pallas_featurestore`:

```bash
cd usecases/pallas
python inference_pipeline.py --target-mev 200 --min-charge-pc 20
```

```
p_max       6999.7362
cn2_max        0.0565
l_inj          0.7427
dip_frac       0.3341
x_of         889.6666
predicted: E_med 204.6 MeV, dE_mad 0.034, q_end 66.9 pC (loss 0.0347)
```

It deploys the direct surrogate as `pallasdirectb` and lets an optimiser call it 60 times, about ten minutes in all. The proposal changes with the machine that runs the optimiser, and campaign B already holds real runs with a lower loss (config 5746, 0.0318). Stop the deployment afterwards: `hops deployment stop pallasdirectb`.

`usecases/ess` and `usecases/clara-camera` run on the real feature groups the same way, from a project of your own. `usecases/clara` is the optimisation challenge's Easy objective, on the Cheetah simulation.

Your challenge is scored by its own script (`score.py`, `pallas_score.py`) and logged to Weights & Biases: see `GRADING.md` at the root of the repository.
