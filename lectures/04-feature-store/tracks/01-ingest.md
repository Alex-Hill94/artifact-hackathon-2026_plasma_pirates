# 01 · Ingest: post-mortem events to feature groups

Everyone, 20 minutes. Turn the SPIRAL2 LLRF release into three feature groups.

Data: Zenodo [22729653](https://zenodo.org/records/22729653), SPIRAL2 LLRF fault diagnosis (GANIL, CNRS, MIT license). 2,865 post-mortem events from 26 cavities, October 2019 to October 2025. One event is one post-mortem file: about 1.1 s of 27 signals around a trigger, captured when the post-mortem system detects a deviation from normal operation or when an operator requests it. 1,478 of the 2,865 events are Normal. The release holds engineered features, not raw waveforms.

## 1. Prepare

```bash
python data/prepare_llrf.py --out data/llrf
```

Downloads 57 MB, checks md5, writes one parquet file per table:

| File | Rows × cols | Content |
|---|---|---|
| `events.parquet` | 2865 × 20 | keys, cavity, LLRF loop settings from the file header |
| `physics.parquet` | 2865 × 45 | 42 physics and status features: decay, detuning, RF mismatch, microphonics, power flow |
| `stats.parquet` | 2865 × 263 | 260 whole-buffer statistics per signal: mean, std, quantiles, shape, energy |
| `precursor.parquet` | 2865 × 504 | pre-trigger trends, CUSUM, segment statistics |
| `labels.parquet` | 2865 × 6 | `is_fault`, `fault_category`, `rf_subtype` |

Every table carries `event_id` (primary key), `cavity_id`, `event_ts` (event time).

## 2. Write feature groups

```python
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

events = fs.get_or_create_feature_group(
    name="llrf_events", version=1,
    primary_key=["event_id"], event_time="event_ts", online_enabled=True,
    description="SPIRAL2 LLRF post-mortem events: cavity and loop settings",
)
events.insert(pd.read_parquet("data/llrf/events.parquet"))

physics = fs.get_or_create_feature_group(
    name="llrf_physics", version=1,
    primary_key=["event_id"], event_time="event_ts", online_enabled=True,
    description="Physics features per event (Zenodo 22729653 feature dictionary)",
)
physics.insert(pd.read_parquet("data/llrf/physics.parquet"))

labels = fs.get_or_create_feature_group(
    name="llrf_labels", version=1,
    primary_key=["event_id"], event_time="event_ts", online_enabled=True,
    description="Fault labels from the GANIL post-mortem classifier",
)
labels.insert(pd.read_parquet("data/llrf/labels.parquet"))
```

Labels live in their own group. They come from the authors' deterministic classifier, `Classify_PostMortemFile`, which decodes the fault register recorded in each post-mortem file. Keeping them apart keeps that register out of the features and gives track 03 one group to name as the label. All three groups are online: serving a feature view needs every group in it online, the label group included.

## 3. Look

```bash
hops fg list
hops fg info llrf_physics --version 1
hops fg preview llrf_physics --version 1
```

Then open the feature group in the UI: schema, statistics, the online and offline stores, the commit that just landed.

## Traps

- `ALM` (the hardware fault register) is left out of `events`. The labels are decoded from it, so it is the label.
- `beam_flag` comes from the header field `BEAM`, which the authors document as feed-forward on/off. It says nothing about beam.
- Statistics over the full buffer include the interlock trip itself: the buffer spans both sides of the trigger. That is why binary accuracy on this data sits near 100%. `precursor.parquet` only uses the pre-trigger window: use it if you want to predict, not describe.
- Four column pairs in `physics` hold identical values: `phase_std_pre` and `phase_jitter_rms`, `rf_mismatch_settling_time` and `control_settling_time`, `reflected_saturation_fraction` and `control_saturation_fraction`, `llrf_loop_closed` and `feed_forward_enabled` (both constant 1). Keep one of each.
- Many physics values come from high-passed, normalised signals. Treat `detuning_std_hz` as a feature, not a measurement in hertz.

## Stretch

- Write `stats` and `precursor` as feature groups too.
- Add a model-independent feature per cavity: faults in the previous 7 days, hours since the last event. Compute it in pandas, write it as `llrf_cavity_history`, keyed on `event_id`.
