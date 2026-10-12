# Challenge Objectives — Forecasting (CLEAR beam charge)

This file lists the goals of each tier. Background is in
[`scientific_case.md`](scientific_case.md), the data in [`data/README.md`](data/README.md).
There are no fixed numbers to beat: each tier has **goals**, and teams are compared on how
far they get and how well they support their results.

**The subject:** learn how the accelerator's exogenous variables behave and use them to
infer the gun charge `BCM_GUN`. The variables are the laser (power meters, scope trace,
virtual-cathode camera), the gun RF and klystron 15, the magnets, and the ambient
conditions. First infer the charge on the same shot (virtual diagnostic), then ahead of
time (forecasting).

**Common rules for every tier:**
- **Data:** the October 2026 CLEAR acquisitions, as raw chunks on CERNBox. You reduce them
  to tables yourself (Easy tier); `kickoff_notebook.ipynb` shows how for one chunk. Use the
  long runs for results; the two short runs from 2026-10-07 are for quick tests. Report
  every run you use, not just your best one.
- **Chronological split:** test on the last 10 % of each run's shots and train on the
  rest. No shuffling. Fit scalers, feature selection and hyperparameters on the training
  part only.
- **Dropouts:** shots whose charge is below 30 % of the local median are not evaluated.
- **Metric:** MAE in nC is the common reference, so that results can be compared. Try other
  metrics too, and understand their physical meaning.
- **Leakage:** be careful that no information from the test region leaks into training,
  especially with decompositions and smoothing.
- The kickoff notebook implements these rules as functions (section 1b).
- **W&B keys** (proposed, not yet in `GRADING.md`):
  - `forecast/vd_mae` for the virtual diagnostic
  - `forecast/fc_mae` and `forecast/fc_delta_mae` for forecasting
  - `forecast/horizon` for the number of steps ahead
  - set `dataset` in the run config to the run name

## 🟢 Easy — understand and clean the data

Process the raw runs, and understand in which conditions each run was acquired and what
that means for the data. Not every recorded signal is useful: "this signal tells us
nothing" is a valid finding. `kickoff_notebook.ipynb` sections 2–3 do a first pass.

- Reduce the raw runs to tables, one row per shot.
- Work out each run's regime from the data (stable, saturated, single or multi-bunch, laser
  scans).
- Flag dropouts and bad shots; don't delete them.
- Put every reading back on its own shot.
- Go beyond a waveform's peak and integral.
- Rank every signal against the charge: Pearson, Spearman, distance correlation, mutual
  information.

**Goals:**
- Synchronise the data, checking the lag of each variable separately.
- A reason for every variable kept or dropped.
- A ranking that holds across runs, or is explained where it changes.
- A clean dataset ready for the next tiers.

## 🟡 Medium — virtual diagnostic

Infer the charge of a shot from the other variables of that same shot. Examine the
performance of various models, with and without the past `BCM_GUN` in the loop.

- Compare a linear model, a tree ensemble and a neural network.
- Show what drives each model.
- Cross-run: train on one run, test on another.
- Report how the metrics change with more or less past charge history.

**Goals:**
- Good performance as a virtual diagnostic.
- Identify the right metric, and start from basic models (ElasticNet, Random Forest,
  XGBoost, MLP).

## 🔴 Hard — forecasting

Repeat what was done for the virtual diagnostic in a harder setting: predict the charge of
future shots from everything known up to now.

- One step ahead, then up to 50 shots ahead.
- At least one sequence model: TFT, DeepAR, LSTM/GRU or a time-series foundation model.
- Tune the context window and show its effect.

**Goals:**
- Identify the right metric, and start from basic models.
- Beat the rolling mean (the mean of the last 9 good shots).
- A model good enough to deploy in the machine: extremely promising results can earn real
  beam time on CLEAR.

## ⚪ Optional

- **Frequency-band decomposition.** Split the charge with FFT into bands (slow drift, mid,
  fast jitter), forecast each band separately, and recombine. Does it beat forecasting the
  whole signal? Which bands are predictable at all? Watch out for leakage.
- **Probabilistic forecasting.** Prediction intervals or quantiles, with their coverage
  checked on the test region.
- **Close the loop.** Use your forecaster inside one of the parent challenge's optimisation
  methods to choose settings (e.g. laser power or a gun RF phase) that keep the forecast
  charge on target, evaluated offline on the recorded data.
- **Raw images and traces.** Learn features straight from the `BTV0125` images or the full
  waveforms (CNN, autoencoder) instead of hand-made ones.
