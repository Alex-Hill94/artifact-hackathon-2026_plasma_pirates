# ESS LLRF anomaly detection

Feature, training and inference pipelines on the real ESS data. No labels: an IsolationForest on waveform features, and the challenge's autoencoder scored with the organisers' `score.py`.

## Data

`ess_llrf_waveforms` v1 in `m4cast_ess`: 52,842 rows, one per acquisition cycle of one RF chain, `i` and `q` at 256 points.

- Key: `event_id`, `cycle_id`, `pv_base`. Event time: `cycle_ts`.
- 108 captures, 2,380 cycles, 28 chains. Two campaigns: May 2025 and June 2026.
- The 42 rows the organisers score on carry `in_sample_holdout`. Neither model trains on them.

`ess_sds_channel_stats` (497,659 rows, same key) holds full-resolution statistics. These pipelines do not read it.

## Run

In `m4cast_ess` you read only. Run in a project you own, with the real store shared in ([track 03](../../tracks/03-your-project.md)), from its cluster terminal:

```bash
cd usecases/ess
uv pip install -r ../../requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # autoencoder only
export OMP_NUM_THREADS=1                            # the terminal has one CPU

python feature_pipeline.py --insert                 # 14 features per row into ess_waveform_features
python training_pipeline.py                         # both models, registered with their views
python inference_pipeline.py --event 16122602164    # deploy, score a capture by key
hops deployment stop essanomaly
```

`feature_pipeline.py` without `--insert` and `training_pipeline.py` also run from a laptop. `notebook.ipynb` walks the same steps read-only.

## What you get

| Object | Name |
|---|---|
| Feature group, online | `ess_waveform_features`: flat-top level, length, ripple and droop, rise and fall, tail noise, distance to the capture's median cycle |
| Feature views | `ess_llrf_anomaly`, `ess_llrf_autoencoder` |
| Models | `ess_llrf_anomaly` (StandardScaler + IsolationForest, 1% flagged), `ess_llrf_autoencoder` (the challenge's `RFAutoencoder`) |
| Deployment | `essanomaly`: send keys, get a score and a flag |

## Results

```
$ python training_pipeline.py --only ess_llrf_anomaly
                  rows  flagged
all rows         52842      532
holdout             42        4
RF off             269      269
pulse cut short     14       14
```

- **RF off**: 269 rows with almost no amplitude. They are exactly the rows with `trig_code` 14.
- **Pulse cut short**: 14 rows whose flat-top stops after 13 to 17 samples. Ten are the same cycle, `16122596526`, on ten chains.
- Autoencoder on all rows: holdout MSE 0.0136. The challenge's bar, trained on the 168 sample rows, is 0.0194.

```
$ python inference_pipeline.py --event 16122602164
event 16122602164: 328 rows scored by key in 0.3 s, 10 flagged
```

The ten flagged rows are the ten chains cut at cycle `16122596526`.

## Options

```bash
python training_pipeline.py --only ess_llrf_anomaly --end 2026-01-01   # train on May 2025 only
python training_pipeline.py --only ess_llrf_autoencoder --strict       # leave out every cycle of the holdout's captures
hops job deploy ess_features feature_pipeline.py --args "--insert" --run --wait
```

## Open questions

- What `i`, `q` and `trig_code` mean physically. For the case holder.
- Whether the 14 cut pulses are trips.

## Full archive

`cca/` holds the Slurm job and the ingest job that filled the two feature groups from the 49 GB of tars at CC-IN2P3. Participants do not run them.

## Cite

- AISSAI 2026 hackathon, anomaly-detection challenge, case holder Ishkhan Gorgisyan (ESS).
- Gorgisyan I. et al. (2026). EPJ Res. Infrastruct. 10(1):28. https://doi.org/10.1007/s41781-026-00182-7
- Martins J. et al. (2025). ICALEPCS25, TUMG018. https://doi.org/10.18429/JACoW-ICALEPCS2025-TUMG018
