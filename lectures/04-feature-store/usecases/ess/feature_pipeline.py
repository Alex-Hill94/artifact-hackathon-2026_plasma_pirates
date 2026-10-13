"""Feature pipeline: ESS LLRF waveforms to one feature group of per-cycle features.

Source: the feature group ess_llrf_waveforms in the shared store m4cast_ess_featurestore, the
full SDS archive of the AISSAI 2026 anomaly-detection challenge. One row is one acquisition
cycle of one RF chain (DIG-101:Dwn0 of a cavity section): i and q at 256 decimated points.
i is an amplitude envelope (fill, flat-top, decay). q stays within +-180 and wraps, as a phase
in degrees does, and it is noise once the amplitude has decayed. The case holder has not
confirmed either reading, so the features keep the column names.

Writes ess_waveform_features into your own project, keyed on (event_id, cycle_id, pv_base)
with event time cycle_ts, the key and event time of the source group, so a feature view joins
the two. 14 features per row: flat-top level, length, ripple and droop, rise and fall, phase
level and drift while the RF is on, noise, and the distance of the cycle from the other cycles
of the same event on the same chain.

    # laptop: read the real group, compute, print the summary
    python feature_pipeline.py

    # same, and keep the frame
    python feature_pipeline.py --out data

    # terminal of your own project: write the feature group (online and offline)
    python feature_pipeline.py --insert
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

STORE = "m4cast_ess_featurestore"
SOURCE_FG = "ess_llrf_waveforms"
KEYS = ["event_id", "cycle_id", "pv_base"]
EVENT_TIME = "cycle_ts"
TAIL = 32  # last samples of the buffer: the RF has decayed, only noise is left
RF_ON = 0.1  # share of the peak above which the phase means something


def wrap(deg: np.ndarray) -> np.ndarray:
    """Phase differences back into [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def flat_top(i: np.ndarray) -> tuple[int, int]:
    """First and last sample at or above 97% of the peak of i: the flat-top, ramp excluded."""
    idx = np.flatnonzero(i >= 0.97 * i.max())
    return int(idx[0]), int(idx[-1])


def cycle_features(i: np.ndarray, q: np.ndarray) -> dict:
    peak = float(i.max())
    first, last = flat_top(i)
    flat = i[first : last + 1]
    level = float(np.median(flat))
    # a chain with the RF off has a level of 0: its relative features are 0, not a division error
    rel = 1.0 / level if level > 0 else 0.0
    quarter = max(1, len(flat) // 4)
    below = np.flatnonzero(i[last:] < RF_ON * peak)
    fall = int(below[0]) if len(below) else len(i) - last
    # q jumps by 360 where it crosses +-180; unwrapped, level and slope are those of the phase
    phase = np.unwrap(q, period=360.0)
    flat_phase, decay = phase[first : last + 1], phase[last : last + fall]
    return {
        "i_max": peak,
        "i_flat_level": level,
        "i_flat_len": last - first + 1,
        "i_flat_ripple": float(flat.std() * rel),
        # relative sag from the first to the last quarter of the flat-top, > 0 means droop
        "i_droop": float((flat[:quarter].mean() - flat[-quarter:].mean()) * rel),
        "i_rise": first - int(np.argmax(i >= RF_ON * peak)),
        "i_fall": fall,
        "i_tail_noise": float(np.diff(i[-TAIL:]).std()),
        "q_flat_mean": float(wrap(flat_phase.mean())),
        "q_flat_std": float(wrap(flat_phase - flat_phase.mean()).std()),
        "q_decay_slope": float(np.polyfit(np.arange(len(decay)), decay, 1)[0]) if len(decay) > 1 else 0.0,
        "q_tail_noise": float(wrap(np.diff(q[-TAIL:])).std()),
    }


def build(waves: pd.DataFrame) -> pd.DataFrame:
    """Rows of ess_llrf_waveforms (keys, cycle_ts, i, q) to one row of features per key."""
    df = waves.reset_index(drop=True)
    i = np.stack(df.i.to_numpy()).astype("float64")
    q = np.stack(df.q.to_numpy()).astype("float64")
    feats = pd.DataFrame([cycle_features(a, b) for a, b in zip(i, q)])
    # an event holds consecutive cycles of each chain; the odd one out is what the buffer caught
    i_dev, q_dev = np.empty(len(df)), np.empty(len(df))
    for rows in df.groupby(["event_id", "pv_base"]).indices.values():
        median_i = np.median(i[rows], axis=0)
        i_dev[rows] = np.sqrt(((i[rows] - median_i) ** 2).mean(axis=1))
        on = median_i >= RF_ON * median_i.max()
        # phase against the first cycle, so the median is taken on differences that do not wrap
        d = wrap(q[rows] - q[rows][0])
        q_dev[rows] = np.sqrt((wrap(d - np.median(d, axis=0))[:, on] ** 2).mean(axis=1))
    feats["i_cycle_dev"], feats["q_cycle_dev"] = i_dev, q_dev
    return pd.concat([df[[*KEYS, EVENT_TIME]], feats], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=STORE, help="feature store that holds ess_llrf_waveforms")
    ap.add_argument("--out", type=Path, help="write waveform_features.parquet here")
    ap.add_argument("--insert", action="store_true", help="write the feature group into your project")
    ap.add_argument("--features-fg", default="ess_waveform_features")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    import hopsworks

    project = hopsworks.login()
    real = project.get_feature_store(name=args.store)
    source = real.get_feature_group(SOURCE_FG, version=1)
    start = time.time()
    waves = source.select([*KEYS, EVENT_TIME, "i", "q"]).read()
    read = len(waves)
    # the archive holds some captures more than once; the copies are identical
    waves = waves.drop_duplicates(KEYS)
    print(f"read {read} rows in {time.time() - start:.0f} s, {read - len(waves)} repeated keys dropped")

    start = time.time()
    features = build(waves)
    print(
        f"{features.event_id.nunique()} events, {len(features[['event_id', 'cycle_id']].drop_duplicates())} cycles, "
        f"{features.pv_base.nunique()} chains: {len(features)} rows, {features.shape[1] - len(KEYS) - 1} features, "
        f"{time.time() - start:.0f} s"
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        features.to_parquet(args.out / "waveform_features.parquet", index=False)
        print(f"wrote {args.out}/waveform_features.parquet {features.shape}")

    if args.insert:
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=args.features_fg, version=args.version,
            primary_key=KEYS, event_time=EVENT_TIME, online_enabled=True,
            description="Per-cycle features from the i and q waveforms of ess_llrf_waveforms, one row per RF chain and cycle",
        )
        start = time.time()
        fg.insert(features, wait=True)
        print(f"inserted {len(features)} rows into {args.features_fg} v{args.version} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
