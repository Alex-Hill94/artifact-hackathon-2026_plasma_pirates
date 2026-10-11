"""Feature pipeline: ESS LLRF post-mortem captures to two feature groups.

Source: the 210-capture public sample of the AISSAI 2026 anomaly-detection challenge, read
over HTTPS from the organisers' repository at a pinned commit. One row is one acquisition
cycle of one RF chain (DIG-101:Dwn0), five cycles per post-mortem event, 256 decimated points
per component. The columns are called I and Q; their shapes suggest an amplitude envelope (I)
and a phase-like trace (Q), which the case holder has not confirmed, so features keep the
column names.

Writes two tables, both keyed on (event_id, cycle_id) with event time event_ts:

    ess_sds_events         capture metadata: chain, trigger code, beam mode and state, plus
                           synthetic, source_event_id, injected_anomaly (generator rows)
    ess_waveform_features  per-cycle features from I and Q: flat-top level, ripple, droop,
                           rise and fall, phase-like level and drift, noise, deviation from
                           the other cycles of the same event

    # look at the tables, no cluster
    python feature_pipeline.py --out data

    # write the feature groups (online and offline)
    python feature_pipeline.py --insert
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_URL = (
    "https://raw.githubusercontent.com/aghribi/artifact-hackathon-2026/"
    "de9c997f35dffc4b359050ad9298c75e065366f1/challenges/anomaly-detection/data/sample/"
)
KEYS = ["event_id", "cycle_id"]
META = ["source_tar", "pv_base", "trig_code", "trig_user_tag", "beam_mode", "beam_state", "beam_present"]
TAIL = 32  # last samples of the buffer: the RF has decayed, only noise is left


def load_sample(source: str = SAMPLE_URL) -> pd.DataFrame:
    return pd.read_parquet(source.rstrip("/") + "/ess_llrf_sample.parquet")


def flat_top(i: np.ndarray) -> tuple[int, int]:
    """First and last sample at or above 97% of the peak of I: the flat-top, ramp excluded."""
    idx = np.flatnonzero(i >= 0.97 * i.max())
    return int(idx[0]), int(idx[-1])


def cycle_features(i: np.ndarray, q: np.ndarray) -> dict:
    first, last = flat_top(i)
    flat = i[first : last + 1]
    level = float(np.median(flat))
    quarter = max(1, len(flat) // 4)
    below = np.flatnonzero(i[last:] < 0.1 * i.max())
    decay = q[last:]
    return {
        "i_max": float(i.max()),
        "i_flat_level": level,
        "i_flat_len": last - first + 1,
        "i_flat_ripple": float(flat.std() / level),
        # relative sag from the first to the last quarter of the flat-top, > 0 means droop
        "i_droop": float((flat[:quarter].mean() - flat[-quarter:].mean()) / level),
        "i_rise": first - int(np.argmax(i >= 0.1 * i.max())),
        "i_fall": int(below[0]) if len(below) else len(i) - last,
        "i_tail_noise": float(np.diff(i[-TAIL:]).std()),
        "q_flat_mean": float(q[first : last + 1].mean()),
        "q_flat_std": float(q[first : last + 1].std()),
        "q_decay_slope": float(np.polyfit(np.arange(len(decay)), decay, 1)[0]) if len(decay) > 1 else 0.0,
        "q_tail_noise": float(np.diff(q[-TAIL:]).std()),
    }


def build(captures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Captures in the sample schema (plus optional generator columns) to (events, features)."""
    df = captures.reset_index(drop=True)
    events = df[KEYS + ["event_ts", *META]].assign(
        synthetic=df.get("synthetic", False),
        # real captures are their own source; synthetic ones point at the real event they copy
        source_event_id=df.get("source_event_id", df.event_id),
        injected_anomaly=df.get("injected_anomaly", False),
    )

    i = np.stack(df.I.to_numpy()).astype("float64")
    q = np.stack(df.Q.to_numpy()).astype("float64")
    feats = pd.DataFrame([cycle_features(a, b) for a, b in zip(i, q)])
    # a post-mortem event holds consecutive cycles; the odd one out is what the buffer caught
    for name, wave in (("i", i), ("q", q)):
        dev = np.empty(len(df))
        for _, rows in df.groupby("event_id").indices.items():
            dev[rows] = np.sqrt(((wave[rows] - np.median(wave[rows], axis=0)) ** 2).mean(axis=1))
        feats[f"{name}_cycle_dev"] = dev
    features = pd.concat([df[KEYS + ["event_ts"]], feats], axis=1)
    return events, features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SAMPLE_URL, help="folder or URL holding ess_llrf_sample.parquet")
    ap.add_argument("--out", type=Path, help="write events.parquet and waveform_features.parquet here")
    ap.add_argument("--insert", action="store_true", help="write the feature groups")
    ap.add_argument("--events-fg", default="ess_sds_events")
    ap.add_argument("--features-fg", default="ess_waveform_features")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()
    if not (args.out or args.insert):
        ap.error("pass --out, --insert or both")

    events, features = build(load_sample(args.source))
    if events.duplicated(KEYS).any():
        raise SystemExit("(event_id, cycle_id) is not unique, check the source sample")
    print(f"{events.event_id.nunique()} events, {len(events)} cycles, {features.shape[1] - 3} features per cycle")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        events.to_parquet(args.out / "events.parquet", index=False)
        features.to_parquet(args.out / "waveform_features.parquet", index=False)
        print(f"wrote {args.out}/events.parquet {events.shape}, {args.out}/waveform_features.parquet {features.shape}")

    if args.insert:
        import hopsworks

        fs = hopsworks.login().get_feature_store()
        events_fg = fs.get_or_create_feature_group(
            name=args.events_fg, version=args.version,
            primary_key=KEYS, event_time="event_ts", online_enabled=True,
            description="ESS SDS post-mortem captures, DIG-101:Dwn0: metadata per cycle, real and synthetic",
        )
        events_fg.insert(events)
        features_fg = fs.get_or_create_feature_group(
            name=args.features_fg, version=args.version,
            primary_key=KEYS, event_time="event_ts", online_enabled=True,
            description="Per-cycle features from the I and Q waveforms of ESS SDS captures",
        )
        features_fg.insert(features)


if __name__ == "__main__":
    main()
