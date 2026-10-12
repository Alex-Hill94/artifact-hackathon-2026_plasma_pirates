# CLARA cameras: real captures, a surrogate, an optimiser

The optimisation use case on the real CLARA camera archive: 225,421 captures from 36 cameras, 2 March to 30 April 2026, served as feature groups from the project `m4cast_clara` on `https://aissai26.hops.live`. Nothing here is simulated. The Cheetah version of the same three steps is `../clara`.

## The real feature groups

Project `m4cast_clara`, feature store `m4cast_clara_featurestore`, read-only for participants.

| feature group | key | rows | one row is |
|---|---|---|---|
| `clara_camera_captures` v1 | `capture_id` | 225,421 | one camera shot: spot moments, light, 193 columns with every recorded magnet PV |
| `clara_camera_files` v1 | `file_id` | 17,878 | one burst (one HDF5 file, ten shots at fixed settings): mean and shot-to-shot spread of the moments, PV medians, a thumbnail |

Select the columns you need from `clara_camera_files`: the 2,030 bursts of one camera without the `thumbnail` column read in 3 s, the whole group with it in two minutes.

## What the data supports

The files record 24 commanded currents (`SETI`): the gun and linac solenoids, the S01 correctors, the five S02 quadrupoles, two S02 corrector pairs and the C2V line. No magnet downstream of S02 is in the files.

| camera | captures | what can be learned |
|---|---|---|
| `CLA-S02-DIA-CAM-03` | 20,300 in 2,030 bursts, 10 days | sits after the five S02 quadrupoles, and the operators swept them: 30 scans. This use case. |
| `CLA-S04-*`, `CLA-S07-*`, `CLA-FE*` | more than 190,000 | the magnets between S02 and these cameras are not recorded, so their quad scans cannot be joined to settings. Spot statistics only. |

For everything downstream of S02, and for any setting the operators never tried, the simulation in `../clara` is still what answers.

Two things about the S02 scans that no PV says:

- Operators often saved two bursts per magnet setting, a few seconds apart: a faint one without the photo-injector beam, then the beam. 559 of the 2,030 bursts are such backgrounds. Their median light is 1.6e7 counts per shot against 3.1e8 for the beam bursts, and their spot is the dark current halo, several mm wide. `beam_present` is true for both.
- The beam that enters S02 changes from one scan to the next, and the files do not hold what changed. The same quadrupole currents give a different spot on another day.

## FTI map

| step | program | reads | writes | runs |
|---|---|---|---|---|
| feature | `feature_pipeline.py` | `clara_camera_captures` (shared) | feature group `clara_s02_bursts` v1 in your project | cluster terminal or job |
| training | `training_pipeline.py` | feature view `clara_s02_spot` v1 | model `clara_s02_spot` | laptop or cluster terminal |
| inference | `inference_pipeline.py` + `predictor.py` | the model's own test rows | deployment `claras02spot` | cluster terminal (`--local`: laptop) |

You need a project of your own with the real feature stores shared into it: track 03.

`clara_s02_bursts`, key `file_id`, one row per burst of `CLA-S02-DIA-CAM-03`:

| column | what |
|---|---|
| `background` | true when the next burst has the same 24 currents (to 1 mA), starts within 15 s and has at least 3 times the light |
| `scan_id`, `scan_step` | a scan sweeps quadrupoles and leaves everything else alone. A pause of more than ten minutes or a change of any other recorded current starts a new scan, named by its start time. `scan_step` is the burst's place in it. |
| `shots`, `beam_shots`, `signal_sum` | shots in the burst, shots with a spot, median light per shot |
| `event_ts`, `camera` | copied from the captures |

`clara_s02_spot` joins the real `clara_camera_files` and your `clara_s02_bursts` on `file_id` and keeps the beam bursts:

```python
beam = bursts.select(HELPERS).filter(bursts.background == False)
query = files.select(FEATURES + LABELS).join(beam, on=["file_id"], join_type="inner")
fv = fs.get_or_create_feature_view(name="clara_s02_spot", version=1, query=query,
                                   labels=LABELS, training_helper_columns=HELPERS)
```

- Features, 14 currents in A: the five S02 quadrupoles (the knobs), then the context: the S02 and S01 correctors and the gun solenoid.
- Labels, mm on the camera: `mu_x_mm_mean`, `mu_y_mm_mean`, `sigma_x_mm_mean`, `sigma_y_mm_mean`.
- Helper columns: `scan_id`, `scan_step`, `background`. They pick the rows and name the scan; the model does not see them.

The model is `ExtraTreesRegressor` (300 trees). The split is in time, on the bursts' event time: `fv.train_test_split(test_start=...)`.

## Commands

Setup is in `tracks/00-setup.md`: clone, `uv pip install -r requirements.txt`, `hops setup --host https://aissai26.hops.live`.

```bash
# cluster terminal of your project
git clone -b lecture-04-feature-store https://github.com/MagicLex/artifact-hackathon-2026
cd artifact-hackathon-2026/lectures/04-feature-store/usecases/clara-camera
uv pip install -r ../../requirements.txt    # 7 s, adds scikit-optimize to the terminal's environment

# cluster terminal of your project, or a job from your laptop
python feature_pipeline.py --insert
hops job deploy clara_s02_bursts feature_pipeline.py --args "--insert" --run --wait

# laptop (hops project use <your project>) or cluster terminal
python feature_pipeline.py --out data/      # the same rows to parquet, nothing written to the cluster
python training_pipeline.py                 # test: the last day of bursts
python training_pipeline.py --test-start "2026-04-24 14:48"

# cluster terminal: deploys, or reuses, claras02spot
python inference_pipeline.py
# laptop: the same against model/model.pkl
python inference_pipeline.py --local model/

hops deployment stop claras02spot
```

Laptop and cluster terminal both run on `lectures/04-feature-store/requirements.txt`. The terminal image has everything in it but `scikit-optimize`, which the inference pipeline needs. `requirements.txt` in this folder is for `camera_pipeline.py` only.

## Results

Feature pipeline, as a job inside the cluster (48 s for the whole job):

```
read 20300 captures of CLA-S02-DIA-CAM-03 in 2 s
2030 bursts, 559 background, 1471 beam, 30 scans
median light per shot: background 1.59e+07, beam 3.13e+08 counts
inserted 2030 rows into clara_s02_bursts v1 in 7 s
```

Training pipeline, from a laptop, 23 s:

```
1071 beam bursts before 2026-04-24 train, 400 after are the test
        same scans        later scans
                r2 mae_um          r2  mae_um
mu_x          0.88  97.32       -3.28  944.32
mu_y          0.78 187.45      -40.08 1427.98
sigma_x       0.88  91.55       -4.44  172.22
sigma_y       0.82 131.26       -0.32  545.46
registered clara_s02_spot v1
```

"Same scans" is a 5-fold cross-validation inside the training period: the model interpolates between the settings of scans it has seen. "Later scans" are the 400 bursts of 24 April, a day it has not seen: R² below zero means it predicts worse than the mean of that day would. A random split would have reported only the first column.

24 April has five scans of `QUAD-01`, one per gun solenoid setting. Retraining with the first scans of the day:

| model | trained on bursts before | later scans: R² `sigma_x`, `sigma_y` | MAE, µm |
|---|---|---|---|
| v1 | 2026-04-24 | -4.44, -0.32 | 172, 545 |
| v2 | 2026-04-24 14:48 (one scan of the day) | 0.09, 0.86 | 80, 157 |
| v3 | 2026-04-24 14:59 (two scans) | 0.45, 0.82 | 53, 142 |

Inference pipeline: the deployment is ready 31 s after it is created, and the run takes about a minute. The optimiser tunes the first scan the model has not seen, with 40 calls to the deployment, and the answer is checked against the real burst nearest to it. Objective: mean distance of `sigma_x` and `sigma_y` to 100 µm, the kickoff notebook's spot.

```
model clara_s02_spot v2, scan 20260424T144843: 74 real beam bursts the model has not seen
searched quad_01 in [1.57, 2.66] A, 40 surrogate calls, target sigma 100 um in x and y
start of scan   quad_01 1.572 A   measured sigma 331, 2214 um   objective 1173 um
surrogate best  quad_01 1.572 A   predicted sigma 341, 2192 um   objective 1167 um
nearest burst   quad_01 1.572 A   measured sigma 331, 2214 um   objective 1173 um
best measured   quad_01 1.626 A   measured sigma 258, 2246 um   objective 1152 um
```

| model | scan tuned | predicted | measured at the nearest burst | best burst of the scan |
|---|---|---|---|---|
| v1 | 20260424T143328 | 1204 µm | 959 µm | 787 µm |
| v2 | 20260424T144843 | 1167 µm | 1173 µm | 1152 µm |
| v3 | 20260424T145906 | 1090 µm | 1109 µm | 1070 µm |

v1 sends the operators to a setting 172 µm worse than the best they measured, and is 245 µm wrong about it. After one scan of the day the prediction is 6 µm from the measurement. `--local` gives the same v1 lines as the deployment.

A 100 µm round spot is out of reach in these scans: one quadrupole focuses one plane and defocuses the other, and `sigma_y` stays at 1 mm or more.

A request to the deployment, 103 ms median over 20 calls from a laptop on the VPN:

```python
deployment.predict(inputs=[{"cla_s02_mag_quad_01_seti": 2.005, "cla_s02_mag_quad_02_seti": 0.0, ..., "cla_gun_mag_sol_02_seti": 223.0}])
# {'predictions': [[-0.7467, -1.2024, 0.2003, 2.3032]]}    measured: -0.9833, -1.3308, 0.1660, 2.3182
```

## Limits

- The surrogate holds inside scans it has seen. It does not carry to another day, because the beam entering S02 is not in the files. Retrain on the day's first scan, or give the model that beam: a quad scan is the measurement it can be fitted from, with the organisers' lattice and `utils.py`.
- Steering is not scored: the S02 correctors keep one value through each scan.
- Extra trees never predict outside the spots they were trained on. The optimiser therefore searches the range the operators swept, and cannot propose a setting better than anything measured. For that, use the simulation.
- `clara_camera_files` and `clara_s02_bursts` both carry `event_ts`, but `clara_s02_bursts` declares no event time, so the view joins on the key alone.

## Where each step runs

- Reads, the feature view, training data, training and the model registry: from a laptop.
- `feature_pipeline.py --insert`: from the terminal of your project (2,030 rows in 4 to 5 s) and as a Python job inside the cluster. From a laptop on the VPN the same insert failed twice on the HopsFS connection.
- `inference_pipeline.py`: from the project terminal, with model v3, after `uv pip install scikit-optimize` (without it: `ModuleNotFoundError: No module named 'skopt'`). Also from a laptop on the VPN, three rounds, same numbers.
- From a new terminal session and a fresh clone of the branch (the clone takes 36 s): `uv pip install -r ../../requirements.txt` adds `scikit-optimize` 0.10.2 in 7 s, the insert writes 2,030 rows in 5 s, and `inference_pipeline.py` takes 1 min 35 s with the deployment stopped at the start. Same proposal: 1090 µm predicted, 1109 µm at the nearest burst.
- Found on the way: the query service loads only the columns a query selects. A view that filters on a helper column must be read with `training_helper_columns=True`, or the read fails with `Binder Error: ... does not have a column named "background"`.

## From the raw archive: `camera_pipeline.py`

`camera_pipeline.py` computes capture rows from the HDF5 files themselves: frame median as pedestal, spot threshold at 5 noise sigmas and 5 % of the smoothed peak, first and second moments in mm, a false-colour thumbnail per capture, and every `SETI` and `READI` of the file. It needs `requirements.txt` in this folder and the raw archive mounted at `/m4cast`.

```bash
python camera_pipeline.py --select "2026/3/19/S07 Quad 03 Scans" --limit 2 --out data/
python camera_pipeline.py --select "2026/3/19/S07 Quad 03 Scans" --insert
```

An earlier run in project `aissai26_clara`: 161 files, 1,610 captures of `CLA-S07-DIA-CAM-03`, about 5 minutes as a job. Since 2026-10-12 the cluster mounts only the samples and the computed tables, so the raw frames are no longer readable from a project. The full archive is processed on CC-IN2P3 instead:

## Full archive on CC-IN2P3

`cca/` holds the version that processes the whole archive (17,885 files, 2.17 TB) next to the data, as a SLURM job submitted from Hopsworks. The raw frames never leave CC-IN2P3.

| file | runs | does |
|---|---|---|
| `cca/extract_captures.sh` | on CC-IN2P3, SLURM job `clara_capture_extract` (64 cores, 128 GB) | reads every HDF5 file, writes `captures.parquet`, `files.parquet` and `manifest.json` to `/sps/m4cast/hopsworks/clara/` |
| `cca/ingest_full.py` | in Hopsworks, Python job `clara_full_ingest` | reads them at `/m4cast-results/clara/`, inserts the two feature groups |

Differences from `camera_pipeline.py`: moments also in pixels (`mu_x_px`...), since some cameras have no `PixMM_RBV`; saturation level from the frame type (uint8 or uint16); every `SETI`, `READI`, `GETSETI` and `RPOWER` PV as a column, null where a file has none; no thumbnail per capture.

| feature group | key | rows | what |
|---|---|---|---|
| `clara_camera_captures` | `capture_id` | 225,421 | one row per capture |
| `clara_camera_files` | `file_id` | 17,878 | one row per HDF5 file: mean and shot-to-shot spread of the moments, PV medians, thumbnail of the mean frame |

```bash
hops files upload cca/extract_captures.sh Resources/jobs/clara_capture_extract
# create the SLURM job in the UI (script above, args: --archive <claraserv3 dir> --out /sps/m4cast/hopsworks/clara), run it
hops job deploy clara_full_ingest cca/ingest_full.py --run --wait
```

Run of 2026-10-12, project `m4cast_clara`: 41 min on 64 cores, 7 files do not open (listed in `manifest.json`). The files record no magnet PV downstream of S02.
