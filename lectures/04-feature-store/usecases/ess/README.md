# ESS LLRF anomaly detection

Unsupervised anomaly scoring of ESS RF post-mortem captures, split into feature, training and inference programs that share only the feature store and the model registry.

## Data

Real data: the 210-row public sample of the AISSAI 2026 anomaly-detection challenge, read over HTTPS from the organisers' repository at a pinned commit:

```
https://raw.githubusercontent.com/aghribi/artifact-hackathon-2026/de9c997f35dffc4b359050ad9298c75e065366f1/challenges/anomaly-detection/data/sample/
    ess_llrf_sample.parquet   210 rows
    holdout_ids.json          42 holdout row positions, seed 42
```

Each row is one acquisition cycle of one RF chain (`DIG-101:Dwn0`) from an ESS Synchronous Data Service (SDS) post-mortem capture: 42 events of 5 cycles each, June 2026, two cavities (`MBL-010RFC`: 190 rows, `MBL-090RFC`: 20 rows), three beam modes (RfTest 115, Probe 50, SlowCommissioning 45). Waveforms `I` and `Q` are decimated to 256 points. In the sample, `I` has the shape of an amplitude envelope (fill, flat-top near 8.16, decay) and `Q` of a phase trace (flat near 93, then drifting); 20 of the 50 Probe rows carry no RF at all. Neither reading is confirmed by the case holder, so the features keep the column names. The full 71 GB raw archive is not used here.

Sources to cite:

- Challenge and sample: AISSAI 2026 hackathon, anomaly-detection challenge, case holder Ishkhan Gorgisyan (ESS), github.com/aghribi/artifact-hackathon-2026 at `de9c997f`.
- Gorgisyan I. et al. (2026). Strategy of AI Application and Current Activities at European Spallation Source Proton Linac. EPJ Res. Infrastruct. 10(1):28. https://doi.org/10.1007/s41781-026-00182-7
- Martins J. et al. (2025). The ESS synchronous data service (SDS) development and first results. ICALEPCS25, TUMG018. https://doi.org/10.18429/JACoW-ICALEPCS2025-TUMG018

## Real and synthetic

| | Real | Synthetic (`generators/ess_sds_stream.py`) |
|---|---|---|
| Waveforms | as published | the 5 cycles of one real event, `I` times a gain, `Q` plus an offset, one draw per event |
| Gain and offset spread | | cycle-to-cycle standard deviation of flat-top `I` level (relative) and flat-top `Q` level inside the source event; setpoint changes between events are not invented |
| Metadata | as published | copied from the source event |
| `event_id` | SDS id | epoch milliseconds at emission, above every real id |
| `cycle_id` | SDS id | `event_id` plus the source event's real cycle offsets |
| `event_ts` | capture time | emission time |
| Gap to next event | | a real gap between consecutive sample events, divided by `--speedup` (default 3600) |
| `synthetic` | False | True |
| `source_event_id` | own `event_id` | the real event copied |
| `injected_anomaly` | False | True when `--hot` fired |

`--hot RATE` is the one invented signal: on that share of events (picked among events with RF), the last cycle gets a linear sag of `I` across the flat-top, depth `--droop` (default 2%). The sample holds no confirmed fault of this kind, so the depth is a parameter, not a measurement. Measured on the sample, the real spread it competes with is small: flat-top `I` level varies by a median 0.002% from cycle to cycle inside an event.

## Feature store objects

| Object | Name | Key | Written by | Read by |
|---|---|---|---|---|
| Feature group | `ess_sds_events` v1, online | `event_id`, `cycle_id`; event time `event_ts` | `feature_pipeline.py`, generator | `inference_pipeline.py` (finds new synthetic cycles) |
| Feature group | `ess_waveform_features` v1, online | `event_id`, `cycle_id`; event time `event_ts` | `feature_pipeline.py`, generator | feature view |
| Feature view | `ess_llrf_anomaly` v1 | serving keys `event_id`, `cycle_id` | `training_pipeline.py` | training, `predictor.py` |
| Model | `ess_llrf_anomaly` | linked to the feature view | `training_pipeline.py` | deployment |
| Deployment | `essanomaly`, `predictor.py` | request: list of `{event_id, cycle_id}` | `inference_pipeline.py` | `inference_pipeline.py` |

`ess_sds_events` columns: `source_tar`, `pv_base`, `trig_code`, `trig_user_tag`, `beam_mode`, `beam_state`, `beam_present`, `synthetic`, `source_event_id`, `injected_anomaly`.

`ess_waveform_features`, 14 per cycle, all computed in `feature_pipeline.build`, also used by the generator:

| Feature | Definition |
|---|---|
| `i_max` | peak of `I` |
| `i_flat_level`, `i_flat_len` | median and length of the flat-top, samples at or above 97% of the peak |
| `i_flat_ripple` | flat-top standard deviation over its level |
| `i_droop` | first quarter minus last quarter of the flat-top, over its level |
| `i_rise`, `i_fall` | samples from 10% of the peak to the flat-top, and from its end back below 10% |
| `i_tail_noise`, `q_tail_noise` | standard deviation of sample-to-sample differences over the last 32 samples |
| `q_flat_mean`, `q_flat_std` | `Q` over the flat-top window |
| `q_decay_slope` | linear slope of `Q` from the end of the flat-top |
| `i_cycle_dev`, `q_cycle_dev` | RMS distance of the cycle from the median waveform of its event's cycles |

The raw 256-point arrays are not stored. Hopsworks 5.1 supports `ARRAY<DOUBLE>` features, but online they map to `VARBINARY(100)` by default; 256 doubles need an explicit `BLOB` online type. The model does not need them.

## Model

`StandardScaler` plus `IsolationForest(n_estimators=300, contamination=0.05, random_state=42)` on the 14 features. No confirmed fault labels exist for this data (`trig_code` is an unconfirmed weak label and here it separates the two cavities exactly), so the model learns the envelope of the real cycles. Score: the flipped `score_samples`, higher is more anomalous. Flag: above the threshold that marks 5% of the training cycles.

Split: the challenge's `holdout_ids.json`, row positions mapped to `(event_id, cycle_id)`. It is stratified on rows, so cycles of one event sit on both sides. Training reads the feature view between `--start 2026-06-01` and `--end 2026-07-01`, the window of the real sample, so synthetic rows never enter training.

The challenge's own score (autoencoder reconstruction MSE on `Q`, bar 0.0194) is not comparable with an IsolationForest score.

## Run

From this folder, with the lecture venv (`pip install -r ../../requirements.txt`):

```bash
python feature_pipeline.py --out data            # local frames
python generators/ess_sds_stream.py --n 1        # one synthetic event as JSON
python generators/ess_sds_stream.py --n 200 --hot 0.3 --seed 7 --out data/synthetic.parquet
python training_pipeline.py --local data         # train, holdout metrics, model/model.pkl
```

On a cluster (`HOPSWORKS_HOST`, `HOPSWORKS_PROJECT`, `HOPSWORKS_API_KEY` set):

```bash
python feature_pipeline.py --insert
python training_pipeline.py                      # feature view, training, model registry
python generators/ess_sds_stream.py --insert --follow --hot 0.2 &
python inference_pipeline.py --follow            # deploys on first run, prints scores
python inference_pipeline.py --key 16096686207:16096686195
```

The generator and the loop run side by side as two clients of the same user. Give each its own certificate folder (`export HOPSWORKS_CERT_FOLDER=$(mktemp -d)` per terminal), or the first to exit deletes the other's certificates.

`notebook.ipynb` walks the same steps locally, cluster calls shown as code blocks.

## Verified locally

Run on 2026-10-10, scikit-learn 1.8, no cluster:

```
$ python feature_pipeline.py --out data
42 events, 210 cycles, 14 features per cycle

$ python training_pipeline.py --local data
train 168 cycles, holdout 42 cycles, 14 features
train_mean_score     0.4378
holdout_mean_score   0.4491
holdout_flag_rate    0.0952
top holdout cycles:
   event_id    cycle_id    score
16096686207 16096686195 0.699664
16096686207 16096686199 0.683636
16096683557 16096683546 0.636402
```

On the real cycles, the flags fall on the rare population: 12 of the 20 `MBL-090RFC` Probe cycles, the only ones without RF, plus 1 `MBL-010RFC` Probe cycle. The top holdout cycles above are `MBL-090RFC` Probe.

Scoring 200 synthetic events (`--hot 0.3 --seed 7`, 61 injected) with that model: drooped cycles average 0.597 against 0.432 for the rest, ROC AUC 0.958, but only 24.6% of drooped cycles cross the 5% threshold (3.9% of the others do, mostly Probe). IsolationForest ranks them well and saturates; a lower threshold, or one model per cavity and beam mode, would flag more.

## Verified on a cluster

Run on 2026-10-10 against a Hopsworks 5.1.1 cluster, external Python client `hopsworks` 5.1.1, scikit-learn 1.8, project `aissai26_ess`, every command above in order:

- `feature_pipeline.py --insert`: both groups created and filled, 210 rows each. The first attempt failed on the offline Delta write (`Connection to HopsFS failed after 3 attempts`) while two other projects were writing; the unchanged retry passed in 5m53s.
- `ess_sds_stream.py --insert --n 10 --hot 0.3`: 10 events online, 6m18s, most of it in the online write.
- `training_pipeline.py`: same split and numbers as the local run (train 168, holdout 42, holdout mean score 0.4491, flag rate 0.0952, same top cycles). Model `ess_llrf_anomaly` v1 registered with the feature view; the model upload took 11 minutes.
- `inference_pipeline.py --key 16096686207:16096686195`: deployment `essanomaly` created with `predictor.py`, running after 44 s, response `{'predictions': [[0.6996643884534517, True]]}`, the same score as the local model.
- `inference_pipeline.py --follow` next to `ess_sds_stream.py --insert --follow --n 4 --hot 0.3 --max-wait 3`: 40 live cycles read from the online store and scored through the deployment. Excerpt (`injected` printed 0/1 in this run, now False/True):

```
event_id       cycle_id       source_event_id  injected  score   flagged
1791632050854  1791632050718  16121644564      1         0.383
1791632050854  1791632050732  16121644564      1         0.578
1791632050886  1791632050874  16096686207      0         0.700   ANOMALY
1791632051121  1791632050994  16122829627      1         0.469
1791632051121  1791632051008  16122829627      1         0.597
```

The drooped last cycle of each injected event scored highest in its event (0.578 and 0.597 against 0.37 to 0.47) but stayed under the threshold, as in the local check. Copies of the no-RF `MBL-090RFC` event were flagged, as the real event is. The deployment was stopped afterwards: `inference_pipeline.py` starts it again.

Fixes made on the way:

- `fv.training_data(..., primary_key=True)` returns the keys as `<project>_<fg>_<version>_<key>`, for example `aissai26_ess_ess_waveform_features_1_event_id`, even for a one-group view. `training_pipeline.py` renames them back.
- The generator wrote `ess_sds_events` before `ess_waveform_features`. The polling loop then found events whose features were not online yet, and the predictor failed with `Found array with 0 sample(s)`. The generator now writes features first.
- Booleans come back from the online store as TINYINT; the loop prints them as booleans.

## Open questions

- What `I`/`Q` (`Cmp0`/`Cmp1`) and `trig_code` mean physically: for the case holder.
- The organisers' repository has no license file; check reuse terms of the sample before publishing slides with its plots.
