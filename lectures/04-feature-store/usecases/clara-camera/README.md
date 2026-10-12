# CLARA cameras: a surrogate and an optimiser on real captures

Feature, training and inference pipelines on the real CLARA camera archive. The challenge's Easy objective is on the Cheetah simulation: that one is `../clara`.

## Data

In `m4cast_clara`, read-only:

| Feature group | Key | Rows | One row is |
|---|---|---|---|
| `clara_camera_captures` v1 | `capture_id` | 225,421 | one camera shot: spot moments, light, every recorded magnet current |
| `clara_camera_files` v1 | `file_id` | 17,878 | one burst of ten shots at fixed settings, with a thumbnail |

The files record no magnet downstream of S02. So only `CLA-S02-DIA-CAM-03` can be modelled: it sits after the five S02 quadrupoles, and the operators swept them in 30 scans (2,030 bursts).

## Run

In a project you own, with the real store shared in ([track 03](../../tracks/03-your-project.md)), from its cluster terminal:

```bash
cd usecases/clara-camera
uv pip install -r ../../requirements.txt

python feature_pipeline.py --insert     # flags background bursts, names the scans: clara_s02_bursts
python training_pipeline.py             # extra trees: 14 currents in, 4 spot moments out
python inference_pipeline.py            # deploy, tune a scan the model has not seen
hops deployment stop claras02spot
```

`training_pipeline.py` and `inference_pipeline.py --local model/` also run from a laptop.

## What you get

| Object | Name |
|---|---|
| Feature group | `clara_s02_bursts`, key `file_id`: `background`, `scan_id`, `scan_step` |
| Feature view | `clara_s02_spot`: the real `clara_camera_files` joined to your flags, backgrounds filtered out |
| Model | `clara_s02_spot`, `ExtraTreesRegressor`, split on event time |
| Deployment | `claras02spot` |

559 of the 2,030 bursts are backgrounds nobody marked: the same currents, a few seconds before the beam burst, a third of the light or less.

## Results

```
$ python training_pipeline.py
        same scans        later scans
                r2 mae_um          r2  mae_um
sigma_x       0.88  91.55       -4.44  172.22
sigma_y       0.82 131.26       -0.32  545.46
```

The model holds inside scans it has seen and fails on a later day: the beam entering S02 changes, and the files do not say how. Retrain with the day's first scans:

```bash
python training_pipeline.py --test-start "2026-04-24 14:48"
python inference_pipeline.py
```

| Model | Knows of 24 April | Predicted | Measured there | Best measured |
|---|---|---|---|---|
| v1 | nothing | 1204 µm | 959 µm | 787 µm |
| v2 | its first scan | 1167 µm | 1173 µm | 1152 µm |
| v3 | two scans | 1090 µm | 1109 µm | 1070 µm |

Objective: mean distance of `sigma_x` and `sigma_y` to 100 µm.

## Limits

- Extra trees do not predict outside what they saw, so the optimiser cannot propose a setting better than anything measured. For that, use the simulation in `../clara`.
- Steering is not scored: the S02 correctors keep one value through each scan.
- A view that filters on a helper column must be read with `training_helper_columns=True`.

## Full archive

`cca/` holds the Slurm job and the ingest job that filled the two feature groups from 2.17 TB of HDF5 files at CC-IN2P3. `camera_pipeline.py` computes the same rows from raw files. Participants run neither.
