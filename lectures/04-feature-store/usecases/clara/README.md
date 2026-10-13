# CLARA S04: captures, surrogate, optimiser

Steering and focusing through section S04 of CLARA (Daresbury), the Easy objective of the optimisation challenge, built as three programs that share only the feature store and the model registry.

Every capture here is simulated with Cheetah. The same three steps on the real camera archive (`m4cast_clara`, 225,421 captures) are in `../clara-camera`: real quadrupole scans in S02, the last section whose magnets the camera files record. S04 and everything downstream still need this simulation.

## Source

All inputs are the organisers' public files in `challenges/optimisation/` (upstream commit `de9c997f35dffc4b359050ad9298c75e065366f1`):

| file | what it gives | provenance (from the file) |
|---|---|---|
| `data/clara/lattice_data/CLARA_cheetah.json` | the CLARA lattice in Cheetah | organisers |
| `data/clara/lattice_data/magnet_ranges.json` | per-magnet operating range | archiver `READI`, 33,159 samples per PV, 2026-01-12 to 2026-07-24, 0.5 to 99.5 percentile |
| `data/clara/lattice_data/reference_tune.json` | starting magnet settings | archiver row 17993, 2026-04-17T14:49:42Z, correctors forced to zero |
| `data/clara/lattice_data/utils.py` | SETI current to k1 / angle conversion | organisers' magnet calibrations |
| `kickoff_notebook.ipynb`, section 4 | section, screen, starting beam, target, objective | organisers |

Section S04 from `CLA-S04-SIM-APER-01`, readout on `CLA-S04-DIA-SCR-03`, eight knobs: `QUAD-01..04` and the H/V supplies of `HVCOR-01..02`. Starting beam: the kickoff `sample_beam(100)` at 193 MeV. Target: centred, 100 um round spot. Objective: mean absolute error on (mu_x, mu_y, sigma_x, sigma_y).

Cite CLARA as: Snedden, E. W., et al. "Specification and design for full energy beam exploitation of the compact linear accelerator for research and applications." Phys. Rev. Accel. Beams 27, 041602 (2024). Cite Cheetah as: Kaiser, J., Xu, C., Eichler, A., Santamaria Garcia, A. "Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations." Phys. Rev. Accel. Beams 27, 054601 (2024).

## Real and synthetic

- Real: lattice, magnet ranges, reference tune, current conversion, PV names.
- Synthetic: every capture. Settings are drawn uniformly inside the real range around the reference tune (`--spread 1.0` = reference plus or minus half the range, clipped to it), converted to `SETI` amps with `utils.py`, applied with `utils.update_cheetah_element`, tracked with Cheetah. Each row carries `synthetic=True`, `lattice_hash` (SHA-1 of `CLARA_cheetah.json`), `cheetah_version`, `campaign`, `beam_seed`, `noise_fraction`.
- The screen reading follows `score.py`: moments over the particles on the camera chip, relative to its centre; below 10 % of the beam on the chip there is no reading (`beam_lost=True`, moments null).
- Camera noise: `--noise-fraction` adds Gaussian noise of that fraction of the spot size to each moment, as the kickoff notebook's `NOISE_FRACTION`. No public file gives a value for CLARA screens, so the default is 0.
- The real captures are the feature groups `clara_camera_captures` and `clara_camera_files` in `m4cast_clara`. They record no magnet downstream of S02, so no real row pairs the S04 magnets with a spot. These rows have that shape (magnet `SETI` settings paired with a centroid and RMS size) for S04.
- Nothing in this folder is read from CC-IN2P3 or the CLARA archiver.

## FTI map

| step | program | reads | writes |
|---|---|---|---|
| feature | `feature_pipeline.py` | lattice files | feature group `clara_s04_captures` v1: key `run_id`, event time `run_ts`, partition `campaign`, offline |
| training | `training_pipeline.py` | feature view `clara_s04_surrogate` v1: 8 `cla_s04_mag_*_seti` features, labels `mu_x_um`, `mu_y_um`, `sigma_x_um`, `sigma_y_um`, filter `beam_lost == False` | model `clara_s04_surrogate` (Python, registered with the view) |
| inference | `inference_pipeline.py` + `predictor.py` | model registry, deployment `clarasurrogate` | one `bo_check` row into `clara_s04_captures` |

`clara.py` holds the lattice, knobs and screen readout used by the feature and inference programs. The model is a scikit-learn pipeline (input scaler, MLP 128x128x128, output scaler): scaling lives in the pickle, so the predictor script only loads it. The deployment receives the eight currents in the request, so the feature group does not need to be online.

## Commands

From this folder, with `lectures/04-feature-store/requirements.txt` plus `torch` and `cheetah-accelerator`:

```bash
# laptop, no cluster
python feature_pipeline.py --runs 2000 --campaign s04_a --seed 1 --out data/
python training_pipeline.py --local data/ --model-dir model/
python inference_pipeline.py --local model/ --out data/

# cluster
python feature_pipeline.py --runs 2000 --campaign s04_a --insert
python training_pipeline.py
python inference_pipeline.py
```

`notebook.ipynb` walks through the same three steps, `LOCAL = True` by default.

## Results, laptop run

2000 captures, seed 1, beam seed 100, Cheetah 0.8.4, scikit-learn 1.8.0:

- Generation: 7.6 s. 0 of 2000 captures lost: inside the real S04 ranges the beam always stays on the chip.
- Surrogate, 1600 train / 400 test: R2 0.997 overall. MAE 51 um (mu_x), 27 um (mu_y), 5.1 um (sigma_x), 3.1 um (sigma_y).
- Optimiser, 80 surrogate evaluations, warm-started at the reference tune:

| setting | mu_x | mu_y | sigma_x | sigma_y | objective |
|---|---|---|---|---|---|
| reference tune, Cheetah | 269.3 | 471.1 | 144.8 | 256.7 | 235.5 um |
| BO best, surrogate | -24.1 | -10.9 | 173.6 | 233.8 | 60.6 um |
| BO best, Cheetah check | 53.2 | -17.4 | 165.4 | 240.3 | 69.1 um |

For scale, on the same box and beam: BO run directly on Cheetah with 80 evaluations reached 53.8 um; differential evolution with 7923 Cheetah evaluations reached 36.3 um. The 100 um round spot is not reachable inside the real operating ranges; sigma_y stays above 200 um at the optimum found.

## Cluster run

Project `aissai26_clara` on a Hopsworks 5.1.1 cluster, `hopsworks` 5.1.1 from PyPI on the laptop, same seeds as above:

- `feature_pipeline.py --insert`: 2000 rows into `clara_s04_captures` v1.
- `training_pipeline.py`: feature view `clara_s04_surrogate` v1, R2 0.997 on the test split (MAE 60 um mu_x, 30 um mu_y, 5.7 um sigma_x, 3.6 um sigma_y), model `clara_s04_surrogate` v1 registered with the view.
- `inference_pipeline.py`: deployment `clarasurrogate` with `predictor.py`, 80 BO calls against it, one Cheetah check written back as `bo_check` (the group then holds 2001 rows).

| setting | mu_x | mu_y | sigma_x | sigma_y | objective |
|---|---|---|---|---|---|
| reference tune, Cheetah | 269.3 | 471.1 | 144.8 | 256.7 | 235.5 um |
| BO best, deployment | -30.8 | -2.3 | 300.3 | 106.2 | 59.9 um |
| BO best, Cheetah check | 141.9 | -45.7 | 311.2 | 104.8 | 100.9 um |

The cluster split and fit differ from the laptop run. Here the optimiser pushed four knobs to the edge of the box, where the surrogate's centroid is off by about 170 um: the check is still 2.3x better than the reference tune, but 41 um worse than predicted. That gap is what the `bo_check` row is for: the next surrogate trains on it. A request with the reference tune returns `{'predictions': [[337.1, 487.6, 145.3, 251.4]]}`.

## Notes

- Two beam energies live in this use case. `utils.py` converts S04 currents at 180 MeV, and the kickoff notebook brings the beam into S04 at 193 MeV. `clara.py` keeps the 180, which is what `utils.update_cheetah_element` converts with. Convert currents yourself at 193 and your quads come out about 7% weaker than the ones the simulation applies.
- The notebook's `LOCAL = False` path calls the same functions as the three programs but was not run as a notebook.
- `Model.deploy` in `hopsworks` 5.1.1 has no `passed_features` and requires `script_file` for Python models, which is why this use case ships `predictor.py`.
