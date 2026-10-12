# PALLAS surrogates on the feature store

PALLAS (IJCLab, Orsay) is a laser-plasma electron injector. The challenge data are Smilei particle-in-cell simulations of it: one row per simulation, the settings in, the electron bunch out. This folder runs the challenge's two Easy objectives as three separate programs that share only the feature store and the model registry.

## Source

The full pack is on the hackathon cluster, `https://aissai26.hops.live`, as four feature groups in the project `m4cast_pallas` (feature store `m4cast_pallas_featurestore`). They were written on 2026-10-12 by `feature_pipeline.py`, unchanged, run as the Python job `pallas_features`:

```
python feature_pipeline.py --insert --source /m4cast/surrogate-models/data/pallas_hackathon_data
```

`/m4cast` is the read-only mount of the organisers' files at CC-IN2P3 (`/sps/m4cast/artifact_hackathon_2026`). The pack is 191 MB, so the job reads it directly and needs no Slurm. You do not run the feature pipeline on the cluster: start at the training pipeline.

Without a cluster, the default source is the organisers' public 10 % sample, read over HTTPS from their repository at a pinned commit:

```
https://raw.githubusercontent.com/aghribi/artifact-hackathon-2026/de9c997f35dffc4b359050ad9298c75e065366f1/challenges/surrogate-models/data/sample/
```

`--source` takes any base URL or a local folder with the same files. Nothing is copied into this repository. The pack's rules (`DATA_CARD.md`, `data/README.md` in `challenges/surrogate-models/`) are applied as written.

Cite the data as the organisers ask in `challenges/surrogate-models/scientific_case.md`, at least:

- G. Kane et al., "Surrogate models study for laser-plasma accelerator electron source design through numerical optimisation", Machine Learning: Science and Technology 7, 030502 (2026). doi:10.1088/2632-2153/ae6603
- J. Derouillat et al., "SMILEI: a collaborative, open-source, multi-purpose particle-in-cell code for plasma simulation", Comput. Phys. Commun. 222, 351 (2018). doi:10.1016/j.cpc.2017.09.024

## FTI map

| program | reads | writes |
|---|---|---|
| `feature_pipeline.py` | the pack: a folder or a URL (`--source`) | feature groups `pallas_campaign_a`, `pallas_campaign_b`, `pallas_b_trajectories`, `pallas_b_moments` |
| `training_pipeline.py` | `pallas_campaign_b`, `pallas_campaign_a` | feature views `pallas_direct_b`, `pallas_inverse_a`; models `pallas_direct_b`, `pallas_inverse_a` |
| `inference_pipeline.py` | model `pallas_direct_b` and its feature view (search bounds) | deployment `pallasdirectb` (Python server, `predictor.py`) |

### Feature groups

| name | primary key | rows in `m4cast_pallas` | rows in the sample | what |
|---|---|---|---|---|
| `pallas_campaign_a` | `config` | 13,073 | 1,307 | `p_1, a_0, c_n2, x_of` and the bunch, both scans (`scan`) |
| `pallas_campaign_b` | `config` | 5,286 | 529 | `p_max, cn2_max, l_inj, dip_frac, x_of` and the bunch |
| `pallas_b_trajectories` | `config, z_um` | 322,615 | 31,866 | the bunch at 113 positions, reference train configs only |
| `pallas_b_moments` | `config, plane_um, particle_group` | 211,360 | 21,160 | mean and 6x6 covariance at 20 planes |

Campaign B holds 5,286 of the 6,000 simulations: the other 714 are the organisers' hidden test set. 2,882 rows are injected and 2,855 are in the reference train split, all of them injected. Campaign A has 10,751 injected rows, 7,208 of them in the random scan.

What the feature pipeline does to the pack:

- Column names are lower-cased, because Hopsworks stores feature names in lower case. `E_med_MeV` is `e_med_mev`, `P_max` is `p_max`.
- `injected` is recomputed as `q_pC >= 3`, the pack's rule.
- Derived, campaigns A and B: `log10_q_pc`, `log10_i_peak`, `log10_n_emit_x`, `log10_sigma_y` (NaN on rows without a bunch). These span decades; the inverse model reads them.
- Campaign B: `in_reference_train`, membership of the organisers' frozen reference train split from `pack_reference.json`.
- Trajectories and moments: positions as integer micrometres (`z_um`, `plane_um`) for the key, since float keys do not join reliably. The scorer joins on z rounded to 1 um too. `z_mm` and `plane_mm` stay as features.
- Moments: `config` parsed from `sim_id = "Config_<id>"`; `sim_id` and the table's own `injection` flag dropped (it disagrees with the pack's flag on 8 simulations); `group` renamed `particle_group` (SQL reserved word).
- Every row carries `source`, the base URL or folder it was read from.
- Not stored: the per-row native spectrum (`spec`, `ener_axis_MeV`) and the 2000-point density curve (`x_p`, `n_e_p`). The spectrum is kept on the shared 200-bin grid as `spec_common`; the curve is built from the four profile knobs.

No event time: a simulation has no wall-clock time, and these groups are never joined point in time. No group is online: the deployment gets all its inputs from the caller (the optimiser), so nothing is looked up.

### Feature views and models

| feature view / model | rows | features -> labels | model |
|---|---|---|---|
| `pallas_direct_b` | campaign B, `injected` and `in_reference_train`: 2,855 | `p_max, cn2_max, l_inj, dip_frac, x_of` -> `e_med_mev, de_mad, q_end` | Gaussian process per target, log targets, physical units out |
| `pallas_inverse_a` | campaign A, `scan == "random"` and `injected`: 7,208 | `e_med_mev, de_mad, log10_q_pc, log10_i_peak, log10_n_emit_x, sigma_z, log10_sigma_y, div_rms, x_of` -> `p_1, a_0, c_n2` | gradient boosting per target |

Scaling lives inside the sklearn pipeline, not in the feature view, so `model.pkl` is self-contained and the predictor script needs nothing else. Campaign A runs with `envelope_validated == False` are kept, as in the kickoff notebook.

R² is measured on a 20 % holdout of the view's rows (sorted by `config`, `random_state=0`), on the scorer's scales: energy linear, `dE_mad` as log10 percent, charge as log10 pC. The local and cluster runs split the same rows. The reference split's test half is hidden, so the organisers' reference numbers (`pack_reference.json`, `scores`) come from another split and are shown for orientation only.

## Commands

Setup is in `tracks/00-setup.md`: clone, `uv pip install -r requirements.txt`, `hops setup --host https://aissai26.hops.live`.

### On the real data, from your laptop

In `m4cast_pallas` you are a Data scientist: you read, create feature views, read training data and register models. The feature groups are already there.

```bash
hops project use m4cast_pallas
cd usecases/pallas
python training_pipeline.py --only pallas_inverse_a     # 26 s
python training_pipeline.py --only pallas_direct_b      # 2 min 27 s
```

```
pallas_inverse_a: trained on 5766, held out 1442
                   p_1    a_0   c_n2
model            0.973  0.916  0.961
linear baseline  0.854  0.409  0.814
registered pallas_inverse_a v1

pallas_direct_b: trained on 2284, held out 571
                 e_med_mev  de_mad  q_end
model                0.996   0.937  0.999
linear baseline      0.946   0.244  0.829
registered pallas_direct_b v1
```

The views and the models are shared with everyone in the project. A second run registers a new model version on the same view.

### In a project you own

In `m4cast_pallas` you cannot create a feature group or a deployment: both answer 403. Create a project of your own and have the real feature stores shared into it (`tracks/03-your-project.md`). `--store` reads the feature groups from the shared store and keeps the views and the models in your project:

```bash
hops project use <your_project>
python training_pipeline.py --store m4cast_pallas_featurestore     # both objectives, 3 min 29 s
```

The scores are the same as above. `hops fv lineage pallas_direct_b --version 1` then lists `pallas_campaign_b` v1 as the upstream feature group and the model `pallas_direct_b` v1 downstream.

### Deploy and search, in the terminal of your project

Deployments are created and called from inside the cluster. Open the terminal of your project (**Terminal**, **Start Terminal**), then:

```bash
git clone -b lecture-04-feature-store https://github.com/MagicLex/artifact-hackathon-2026
cd artifact-hackathon-2026/lectures/04-feature-store/usecases/pallas
uv pip install -r ../../requirements.txt     # 7 s, adds scikit-optimize to the terminal's environment
python inference_pipeline.py --target-mev 200 --min-charge-pc 20
python inference_pipeline.py --target-mev 200 --x-of 700      # focal position fixed by the operator
hops deployment stop pallasdirectb
```

```
p_max       6999.7362
cn2_max        0.0565
l_inj          0.7427
dip_frac       0.3341
x_of         889.6666
predicted: E_med 204.6 MeV, dE_mad 0.034, q_end 66.9 pC (loss 0.0347)
Check it with one Smilei run at these settings and write that run back to pallas_campaign_b.
```

The first run creates the deployment `pallasdirectb` and took 7 min 36 s. A second run reuses it. A running deployment holds a pod: stop it when you are done.

### Without a cluster, on the public sample

```bash
uv pip install requests
python feature_pipeline.py --out pallas_fg
python training_pipeline.py --local pallas_fg --models models
python inference_pipeline.py --local models/pallas_direct_b/model.pkl --fg-dir pallas_fg --target-mev 200 --min-charge-pc 20
```

`notebook.ipynb` walks through these three steps. `pallas_fg/` and `models/` are local outputs and git-ignored.

### Writing feature groups yourself

In a project you own, from its terminal or as a job:

```bash
python feature_pipeline.py --insert --source /m4cast/surrogate-models/data/pallas_hackathon_data
python feature_pipeline.py --insert --only pallas_b_trajectories --source /m4cast/surrogate-models/data/pallas_hackathon_data
hops job deploy pallas_features feature_pipeline.py --args "--insert --source /m4cast/surrogate-models/data/pallas_hackathon_data" --run --wait
```

The optimiser minimises `((E - target) / target)^2 + dE_mad + 10 * shortfall^2`, where shortfall is the relative charge deficit below `--min-charge-pc`. The search space is the range of the training rows, so the surrogate is never asked to extrapolate. Its answer is a proposal: run Smilei at those settings and insert the run as a new row of `pallas_campaign_b`.

## Results on the real data

Hopsworks 5.1.1, scikit-learn 1.8.0, scikit-optimize 0.10.2.

Feature pipeline: job `pallas_features` in `m4cast_pallas`, execution 1050, 1 min 15 s. Its log:

```
pallas_campaign_a          13073 rows x  36 columns, 10751 injected
pallas_campaign_b           5286 rows x  37 columns, 2882 injected
pallas_b_trajectories     322615 rows x  15 columns
pallas_b_moments          211360 rows x  58 columns
pallas_campaign_a: inserted 13073 rows in 8 s
pallas_campaign_b: inserted 5286 rows in 6 s
pallas_b_trajectories: inserted 322615 rows in 7 s
pallas_b_moments: inserted 211360 rows in 8 s
```

Reading each group back from a laptop takes 2 to 4 seconds. No group has an event time and none is online.

Training pipeline: the numbers under Commands, from a laptop, in `m4cast_pallas` and in a project of your own with `--store`. Registry metrics: `r2_e_med_mev` 0.9958, `r2_de_mad` 0.9372, `r2_q_end` 0.9991, `r2_mean` 0.9773 for `pallas_direct_b`; `r2_p_1` 0.9731, `r2_a_0` 0.9162, `r2_c_n2` 0.9613, `r2_mean` 0.9502 for `pallas_inverse_a`.

Inference pipeline, in the terminal of your project: the output under Commands. Run a second time there with the deployment already running, it gave the same settings and the same prediction in 7 min 13 s. Almost all of that time is the optimiser's own Gaussian process in the terminal pod: one request to the deployment answers in about 0.05 s.

```python
deployment.predict(inputs=[[6999.7362, 0.0565, 0.7427, 0.3341, 889.6666]])
# {'predictions': [[204.55058735314452, 0.03413061792184855, 6.691421641315632e-11]]}
```

The local `model.pkl` gives the same numbers at that point. `p_max` of the proposal is the highest value in the view's rows, so the search ended on the edge of its bounds.

For scale, real runs of campaign B scored with the same loss: config 3195 (188.9 MeV, `dE_mad` 0.026, 64.4 pC) has 0.0292 and config 5746 (204.4 MeV, `dE_mad` 0.031, 64.7 pC) has 0.0318. After 60 calls the proposal (0.0347) is close to config 5746 and does not beat it.

The same search run from a laptop on the cluster's network, against the same deployment, ended elsewhere in 19 s: `p_max` 6999.7, `cn2_max` 0.0787, `l_inj` 0.7995, `dip_frac` 0.4320, `x_of` 1246.4, predicted 191.0 MeV, `dE_mad` 0.028, 55.3 pC, loss 0.0303. The Bayesian optimiser follows a different path once its own numerics differ between two machines. Treat one run as one proposal.

Permissions: in `m4cast_pallas` a Data scientist reads, creates feature views, reads training data and registers models. Creating a feature group or a deployment there answers 403.

## Results on the public sample, without a cluster

Python 3.12, hopsworks 5.1.1, scikit-learn 1.8.0, scikit-optimize 0.10.2.

`feature_pipeline.py --out` (HTTPS, and the same counts from the local sample folder):

```
pallas_campaign_a           1307 rows x  36 columns, 1080 injected
pallas_campaign_b            529 rows x  37 columns, 286 injected
pallas_b_trajectories      31866 rows x  15 columns
pallas_b_moments           21160 rows x  58 columns
```

`training_pipeline.py --local` (R² on the 20 % holdout):

```
pallas_direct_b: trained on 225, held out 57
                 e_med_mev  de_mad  q_end
model                0.989   0.805  0.994
linear baseline      0.957   0.348  0.697

pallas_inverse_a: trained on 580, held out 146
                   p_1    a_0   c_n2
model            0.927  0.772  0.885
linear baseline  0.835  0.378  0.782
```

57 and 146 holdout rows are small. On the full campaigns the same code scores higher (see above).

`inference_pipeline.py --local ... --target-mev 200 --min-charge-pc 20` (60 surrogate calls, 10 s):

```
p_max       6574.9747
cn2_max        0.0766
l_inj          0.7076
dip_frac       0.4989
x_of         828.5519
predicted: E_med 192.0 MeV, dE_mad 0.037, q_end 60.1 pC (loss 0.0387)
```

For scale, the closest real simulation in the sample is config 5746: 204 MeV, `dE_mad` 0.031, 65 pC.

`notebook.ipynb` gives the same outputs.

## Results on a cluster with the public sample

Hopsworks 5.1.1. These numbers are from the sample.

Feature pipeline, run as a Python job inside the cluster (`hops job deploy pallas_features feature_pipeline.py --args "--insert" --run --wait`, default environment): the job read the pack from GitHub over HTTPS and wrote the four groups, 6 s each, 59 s for the whole job. Read back: A 1,307 rows with 1,080 injected, B 529 with 286, trajectories 31,866, moments 21,160. `spec_common` is stored as `array<double>`.

Writes run from the project terminal or a job. From a laptop on a lossy line they are slow and can fail, and a write interrupted midway leaves the group unreadable until it is deleted and recreated.

Training pipeline: feature views `pallas_direct_b` v1 and `pallas_inverse_a` v1 created, models `pallas_direct_b` v1 and `pallas_inverse_a` v1 registered with their views. The views return the same rows as the local run (282 and 726 after dropping NaN) and the R² is identical to the local numbers above: `r2_e_med_mev` 0.9891, `r2_de_mad` 0.8054, `r2_q_end` 0.9945; `r2_p_1` 0.9273, `r2_a_0` 0.772, `r2_c_n2` 0.8854. Training and registering both models from the laptop took 136 s.

Inference pipeline, from a laptop on the cluster's network: deployment `pallasdirectb` created with `predictor.py` on the default Python environment, running after 34 s, whole first run 66 s. scikit-learn 1.8 unpickled the model there. A raw request:

```python
deployment.predict(inputs=[[6082.7095, 0.0506, 0.7808, 0.4989, 730.9937]])
# {'predictions': [[194.27143915992133, 0.03599419432421022, 5.404681588120277e-11]]}
```

The same point through the local `model.pkl` agrees to 1e-12. The 60-call search through the deployment:

```
--target-mev 200 --min-charge-pc 20:  p_max 6082.7, cn2_max 0.0506, l_inj 0.7808, dip_frac 0.4989, x_of 731.0
                                      predicted E_med 194.3 MeV, dE_mad 0.036, q_end 54.0 pC (loss 0.0368)
--target-mev 200 --min-charge-pc 20 --x-of 700:
                                      p_max 5184.7, cn2_max 0.0506, l_inj 0.7703, dip_frac 0.4989
                                      predicted E_med 187.2 MeV, dE_mad 0.028, q_end 22.1 pC (loss 0.0324)
```

The first search ends at a different point from the local one (192.0 MeV, loss 0.0387): the Bayesian optimiser follows a different path once predictions differ in the last digits, and both points are near-equal in loss. The deployment was stopped afterwards (`deployment.stop()`); start it again with `inference_pipeline.py`, which reuses it.

Client behaviour found on the way, handled in the code:

- `fv.training_data(primary_key=True)` returns the key as `<project>_<feature group>_<version>_config`, not `config`. The training pipeline renames it back.
- `mr.get_model(name)` without a version returns version 1, not the latest. The inference pipeline takes the version with the best `r2_mean` unless `--model-version` is given.
- `Model.deploy` in 5.1.1 has no `model_server` argument; a Python model with `script_file` gets the Python server.

## Not run

- `hops job deploy pallas_features ... --args "--insert --source ..."` with the `/m4cast` source: the job's arguments are the ones shown in its execution, but the deploy command itself was last run on 2026-10-10 with the sample.
- `--x-of 700` on the real data.
- The six test input files of the pack and the `level5` and `level6` data are not in the feature store.
