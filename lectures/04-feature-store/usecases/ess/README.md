# ESS LLRF anomaly detection

Unsupervised anomaly scoring of ESS RF captures on the full challenge data, split into feature, training and inference programs that share only the feature store and the model registry. The challenge's own autoencoder is trained on the same data and scored with the organisers' `score.py`.

## Data

The feature group `ess_llrf_waveforms` v1 in the project `m4cast_ess` (feature store `m4cast_ess_featurestore`), filled from the SDS archive at CC-IN2P3. Read on 2026-10-12:

| | |
|---|---|
| Rows | 52,842, one per key |
| One row | one acquisition cycle of one RF chain (`DIG-101:Dwn0` of a cavity section), `i` and `q` at 256 decimated points |
| Key | `event_id`, `cycle_id`, `pv_base`. Event time `cycle_ts` |
| Events | 108 captures, 2,380 cycles, 9 to 42 cycles per capture. 65 post-mortem (`ctype` PM) and 43 on demand (DD) |
| Chains | 28 values of `pv_base`: 13 Spoke, 9 MBL and 5 HBL sections, plus one `DODData` channel with 9 rows |
| Campaigns | 32 captures on 28 to 30 May 2025 (31,735 rows, Probe beam), 76 captures on 4 to 30 June 2026 (21,107 rows) |
| Public sample | the 210 rows of the challenge sample carry `in_sample`; the 42 rows the organisers score on carry `in_sample_holdout` |

`i` is an amplitude envelope: fill, flat-top, decay. `q` stays within ±180 and wraps, as a phase in degrees does, and it is noise once the amplitude has decayed. The case holder has not confirmed either reading, so the features keep the column names. The flat-top lasts about 44 samples in May 2025 and about 108 in June 2026.

The same project holds `ess_sds_channel_stats` (497,659 rows, same key): the kickoff notebook's statistics at full resolution for three channels per section. The pipelines here do not read it; it joins on the key.

Sources to cite:

- Challenge and data: AISSAI 2026 hackathon, anomaly-detection challenge, case holder Ishkhan Gorgisyan (ESS), github.com/aghribi/artifact-hackathon-2026.
- Gorgisyan I. et al. (2026). Strategy of AI Application and Current Activities at European Spallation Source Proton Linac. EPJ Res. Infrastruct. 10(1):28. https://doi.org/10.1007/s41781-026-00182-7
- Martins J. et al. (2025). The ESS synchronous data service (SDS) development and first results. ICALEPCS25, TUMG018. https://doi.org/10.18429/JACoW-ICALEPCS2025-TUMG018

## Feature store objects

In `m4cast_ess` you are a Data scientist: you read, and you cannot create a feature group or a deployment. The pipelines run in a project you own, with the real store shared into it ([track 03](../../tracks/03-your-project.md)).

| Object | Name | Where | Key | Written by | Read by |
|---|---|---|---|---|---|
| Feature group | `ess_llrf_waveforms` v1, offline | `m4cast_ess`, read-only | `event_id`, `cycle_id`, `pv_base`; event time `cycle_ts` | the organisers | `feature_pipeline.py`, `training_pipeline.py` |
| Feature group | `ess_waveform_features` v1, online | your project | same key, same event time | `feature_pipeline.py` | feature view `ess_llrf_anomaly` |
| Feature view | `ess_llrf_anomaly` v1 | your project | serving keys `event_id`, `cycle_id`, `pv_base` | `training_pipeline.py` | training, `predictor.py` |
| Feature view | `ess_llrf_autoencoder` v1 | your project | | `training_pipeline.py` | training |
| Model | `ess_llrf_anomaly` | your project | linked to its view | `training_pipeline.py` | deployment |
| Model | `ess_llrf_autoencoder` | your project | linked to its view | `training_pipeline.py` | |
| Deployment | `essanomaly`, `predictor.py` | your project | request: list of `{event_id, cycle_id, pv_base}` | `inference_pipeline.py` | `inference_pipeline.py`, `hops deployment predict` |

`ess_llrf_anomaly` reads `ess_waveform_features` only. A view that also selects columns of `ess_llrf_waveforms` trains but cannot serve: the real group is offline, and `init_serving` refuses a view with an offline group in it. The split flag is therefore read from the real group by key, next to the view.

## Features

`ess_waveform_features`, 14 per row, computed in `feature_pipeline.build`:

| Feature | Definition |
|---|---|
| `i_max` | peak of `i` |
| `i_flat_level`, `i_flat_len` | median and length of the flat-top, samples at or above 97% of the peak |
| `i_flat_ripple` | flat-top standard deviation over its level |
| `i_droop` | first quarter minus last quarter of the flat-top, over its level |
| `i_rise`, `i_fall` | samples from 10% of the peak to the flat-top, and from its end back below 10% |
| `i_tail_noise`, `q_tail_noise` | standard deviation of sample-to-sample differences over the last 32 samples |
| `q_flat_mean`, `q_flat_std` | `q` over the flat-top window, unwrapped |
| `q_decay_slope` | linear slope of unwrapped `q` from the end of the flat-top until `i` is below 10% of the peak |
| `i_cycle_dev`, `q_cycle_dev` | RMS distance of the cycle from the median waveform of the cycles of the same event on the same chain; for `q`, on wrapped differences, where the RF is on |

The waveforms themselves stay in `ess_llrf_waveforms`. The model does not need them online.

## Models

`ess_llrf_anomaly`: `StandardScaler` plus `IsolationForest(n_estimators=300, contamination=0.01, random_state=42)` on the 14 features. No confirmed fault labels exist for this data, so the model learns the envelope of the cycles. Score: the flipped `score_samples`, higher is more anomalous. Flag: above the threshold that marks 1% of the training rows. `--start` and `--end` cut the training rows on `cycle_ts` through the feature view; the model then scores every row.

`ess_llrf_autoencoder`: `RFAutoencoder` and `score_holdout` imported from `challenges/anomaly-detection/score.py`, the same optimiser, batch size and seed as its `train_reference`. The difference is the training rows: every row of `ess_llrf_waveforms` outside the holdout, read through a feature view, where the reference uses the 168 training rows of the public sample. `q` is normalised with the mean and deviation of those 168 rows, as the reference does, so the result is on the scale of the bar. Needs `torch`, as the challenge's own `requirements.txt` does. The cluster terminal has scikit-learn 1.8.0 and the Hopsworks client, and no `torch`: the CPU wheel installs there in about 6 seconds with the command below.

Neither model trains on a row with `in_sample_holdout`.

## Run

| Step | Command | Where |
|---|---|---|
| Features, look only | `python feature_pipeline.py` | laptop or cluster terminal |
| Features, write | `python feature_pipeline.py --insert` | terminal of your project |
| Train both models | `python training_pipeline.py` | laptop or cluster terminal |
| Deploy, score by key | `python inference_pipeline.py --event 16122602164` | terminal of your project |

In the terminal of your own project, after the organisers shared the real stores into it:

```bash
git clone -b lecture-04-feature-store https://github.com/MagicLex/artifact-hackathon-2026
cd artifact-hackathon-2026/lectures/04-feature-store/usecases/ess
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # for the autoencoder only

python feature_pipeline.py --insert
python training_pipeline.py
python inference_pipeline.py --event 16122602164
python inference_pipeline.py --key 16122602164 16122596526 HBL-050RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData
hops deployment stop essanomaly
```

From a laptop, with the lecture venv (`uv pip install -r ../../requirements.txt`) and `hops project use <your project>`: `python feature_pipeline.py` without `--insert`, and `python training_pipeline.py`. The insert and the inference pipeline need the cluster: an insert goes through Kafka and HopsFS, a prediction through the serving gateway.

`notebook.ipynb` walks the same steps from a laptop. It only reads: the real group, the features, the anomaly model fitted in memory.

Variants:

```bash
python feature_pipeline.py --out data                                # keep the frame as parquet
hops job deploy ess_features feature_pipeline.py --args "--insert" --run --wait    # the insert as a job inside the cluster
python training_pipeline.py --only ess_llrf_anomaly --end 2026-01-01   # train on the May 2025 campaign
python training_pipeline.py --only ess_llrf_autoencoder --sample-chains --epochs 30
python training_pipeline.py --only ess_llrf_autoencoder --strict
```

## Results

### Feature pipeline

```
$ python feature_pipeline.py --insert
read 52842 rows in 6 s, 0 repeated keys dropped
108 events, 2380 cycles, 28 chains: 52842 rows, 14 features, 4 s
inserted 52842 rows into ess_waveform_features v1 in 29 s
```

43 s in all from the laptop. From the terminal of the project, in a clone of this branch: read 2 s, features 15 s, insert 10 s. As the job `ess_features`: about one minute from submission to the end.

### Anomaly model

```
$ python training_pipeline.py --only ess_llrf_anomaly --end 2026-01-01

ess_llrf_anomaly: trained on 31735 rows, 14 features, 2 s
                              rows  flagged
all rows                     52842      986
holdout                         42        4
RF off                         269      269
pulse cut short                 14       14
outside the training window  21107      668
registered ess_llrf_anomaly v1

$ python training_pipeline.py --only ess_llrf_anomaly

ess_llrf_anomaly: trained on 52800 rows, 14 features, 2 s
                  rows  flagged
all rows         52842      532
holdout             42        4
RF off             269      269
pulse cut short     14       14
registered ess_llrf_anomaly v2
```

20 s and 27 s. What the rows of the report are:

- **RF off**: 269 rows whose peak is under 10% of their chain's median peak. MBL-090, Spk-010 and Spk-020 in the captures of 4 to 10 June 2026. They are exactly the rows with `trig_code` 14; every other row has 15. The 4 flagged holdout rows are four of them, on MBL-090.
- **Pulse cut short**: 14 rows whose flat-top is under half the median length of the cycles of the same event on the same chain: 13 to 17 samples where the others hold 53 or 108. Ten of them are the same cycle, `16122596526` on 29 June 2026 at 18:13:56, on ten HBL and MBL chains; that `cycle_id` is also the `event_id` of a post-mortem capture. Two more are cycle `16122697297`, also the id of a post-mortem capture. These are facts read from the waveforms. Nobody has confirmed them as faults.
- **Model v1**, trained on May 2025, flags 1.0% of May 2025 and 3.2% of June 2026 (668 of 21,107 rows): the 269 rows with the RF off, the 13 cut pulses of June 2026, and 386 HBL and MBL rows. June 2026 runs pulses 108 samples long where May 2025 ran 44. The 14 cut pulses rank between 1 and 27 of 52,842 rows.
- **Model v2**, trained on both campaigns, flags 532 rows: the 269 rows with the RF off first, then the 14 cut pulses (ranks 254 to 330), then 249 rows, 246 of them Spk-050 in May 2025. On that chain the flat-top climbs by 1% from start to end; on the other Spoke chains by 0.3%.

The model ranks rare operating states above single odd cycles. A per-chain model, or features relative to the chain's own median, would rank the cut pulses first.

### Autoencoder, scored with `score.py`

The bar of the challenge is 0.0194: the reference autoencoder trained on the 168 training rows of the public sample, mean reconstruction MSE on the 42 holdout rows. `python score.py` in `challenges/anomaly-detection/` gives 0.0194 here too, and so does the same model trained on those 168 rows read from the feature group.

| Command | Training rows | Epochs | Holdout MSE, mean | Median | Training |
|---|---|---|---|---|---|
| `python score.py` (the bar) | 168 | 30 | 0.0194 | 0.0164 | 3 s |
| `training_pipeline.py --only ess_llrf_autoencoder` | 52,800 | 5 | 0.0136 | 0.0047 | 64 s |
| `... --sample-chains --epochs 30` | 4,256 | 30 | 0.0130 | 0.0038 | 37 s |
| `... --strict` | 46,250 | 5 | 0.0139 | 0.0049 | 55 s |
| `... --strict --sample-chains --epochs 30` | 3,727 | 30 | 0.0144 | 0.0044 | 28 s |

```
$ python training_pipeline.py --only ess_llrf_autoencoder

ess_llrf_autoencoder: trained on 52800 rows, 5 epochs, 64 s
holdout mean reconstruction MSE 0.0136, median 0.0047, 42 rows
on the scale of the bar (sigma 39.73); 0.0019 with the training rows' own sigma 106.35
registered ess_llrf_autoencoder v5
```

How to read the table:

- The holdout is 38 rows of MBL-010 and 4 of MBL-090. `--sample-chains` trains on those two chains only.
- Every holdout row has 10 to 12 cycles of the same event and chain in the training rows, and consecutive cycles are nearly identical. `--strict` leaves out every cycle of the 25 events that hold a holdout row. The score moves from 0.0136 to 0.0139, and from 0.0130 to 0.0144 on the two chains. The two `--sample-chains` rows were run before the re-ingest, on the same rows. Training is not bit-reproducible: the two runs of the default command gave 0.0138 and 0.0136, the two of `--strict` 0.0141 and 0.0139.
- The mean is held up by the 4 MBL-090 rows with the RF off: their `q` is noise, which no model reconstructs. The median falls by a factor of 3 to 4.
- The scale matters. Normalised with the deviation of all the training rows (106.35 instead of 39.73), the same reconstruction reads 0.0019. Compare numbers on one scale.
- Thirty epochs on all rows with a batch of 256 gave 0.0143 (452 s on a laptop CPU). More passes did not help.

### Inference

```
$ python inference_pipeline.py --key 16122602164 16122596526 HBL-050RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData
deployment essanomaly running after 42 s, model version 2
{'predictions': [[0.6469190320102919, True]]}

$ python inference_pipeline.py --event 16122602164
deployment essanomaly running after 0 s, model version 2
event 16122602164: 328 rows scored by key in 0.3 s, 10 flagged
   event_id    cycle_id                                   pv_base    score  flagged
16122602164 16122596526 MBL-090RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.685356     True
16122602164 16122596526 HBL-040RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.669337     True
16122602164 16122596526 MBL-080RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.668712     True
16122602164 16122596526 HBL-010RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.665018     True
16122602164 16122596526 MBL-060RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.664706     True
16122602164 16122596526 MBL-070RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.664365     True
16122602164 16122596526 HBL-030RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.663034     True
16122602164 16122596526 HBL-020RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.654557     True
16122602164 16122596526 MBL-050RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.651936     True
16122602164 16122596526 HBL-050RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData 0.646919     True
```

The first run creates the deployment: 47 s in all. The second reuses it: 4 s in all, 328 keys in one request. The ten flagged rows are the ten chains whose pulse was cut at cycle `16122596526`; the local `model.pkl` gives the same scores. The same call from the command line:

```
$ hops deployment predict essanomaly --data '{"instances": [{"event_id": 16122602164, "cycle_id": 16122596526, "pv_base": "HBL-050RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData"}, {"event_id": 16122602164, "cycle_id": 16122596512, "pv_base": "HBL-050RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData"}]}'
{
  "predictions": [
    [
      0.6469190320102919,
      true
    ],
    [
      0.46705529406673735,
      false
    ]
  ]
}
```

A deployment keeps the model version it was created with. To serve a newer one, delete it (`hops deployment delete essanomaly`) and run the inference pipeline again.

## Client behaviour handled in the code

- `fv.training_data(primary_key=True)` and `fv.get_batch_data(primary_key=True, event_time=True)` return the keys and the event time as `<project>_<feature group>_<version>_<name>`. `training_pipeline.py` renames them back.
- `feature_pipeline.py` and the autoencoder drop repeated keys after reading. The groups held 1,872 keys four times until the re-ingest of 12 October; the guard now drops nothing.
- `mr.get_model(name)` without a version returns version 1. `inference_pipeline.py` takes the highest version unless `--model-version` is given.

## Not run

- The training pipeline from the cluster terminal. The feature pipeline and the inference pipeline ran there; the terminal has scikit-learn 1.8.0 and no `torch` until you install it.
- Training from a laptop outside the cluster's network. The laptop used here was on it.
- Anything on `ess_sds_channel_stats` beyond reading it (497,659 rows).

## Open questions

- What `i` and `q` (`Cmp0`, `Cmp1`) and `trig_code` mean physically: for the case holder. In this data `trig_code` is 14 on the 269 rows with the RF off and 15 everywhere else, so it is no fault label here.
- Whether the 14 cut pulses are trips, and what cut them.

## Full archive on CC-IN2P3

`cca/` builds the public sample's table for the whole SDS archive (108 events, 49 GB of tars), next to the data, as a SLURM job submitted from Hopsworks. The lecture programs above read the two feature groups it fills.

| file | runs | does |
|---|---|---|
| `cca/extract_sds.sh` | on CC-IN2P3, SLURM job `ess_sds_extract` (16 cores, 96 GB, 10 workers) | reads the parquet members inside the tars in place, writes `llrf_waveforms.parquet`, `channel_stats.parquet` and `manifest.json` to `/sps/m4cast/hopsworks/ess/` |
| `cca/ingest_full.py` | in Hopsworks, Python job `ess_full_ingest` | reads them at `/m4cast-results/ess/`, inserts the two feature groups |

| feature group | key | rows | what |
|---|---|---|---|
| `ess_llrf_waveforms` | `event_id, cycle_id, pv_base` | 52,842 | the sample's schema for every event, cycle and RF section (`DIG-101:Dwn0`), `i` and `q` at 256 points, `in_sample` and `in_sample_holdout` |
| `ess_sds_channel_stats` | `event_id, cycle_id, pv_base` | 497,659 | full-resolution statistics of every RF chain per cycle |

The key has `pv_base` because a cycle has many chains here; the sample has one. The 210 sample rows are in `ess_llrf_waveforms` and match to float32 rounding; the 42 rows the organisers score on carry `in_sample_holdout`, so leave them out of training.

Run of 2026-10-12, project `m4cast_ess`: 17 min. With 16 workers on 48 GB the job runs out of memory (one post-mortem member decodes to more than 1 GB).
