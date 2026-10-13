"""SPIRAL2 feature pipeline: the LLRF post-mortem tables into three feature groups.

One row is one post-mortem file: about 1.1 s of 27 signals around a trigger, as engineered
features. data/prepare_llrf.py downloads the release (Zenodo 22729653) and writes one parquet
per table; this pipeline writes three of them as feature groups, all keyed on event_id with
event_ts as event time, all online so a deployment can look a row up by key:

    llrf_events    cavity and LLRF loop settings from the file header
    llrf_physics   42 physics and status features
    llrf_labels    is_fault, fault_category, rf_subtype

Labels sit in their own group: they are decoded from the hardware fault register, which is
left out of the features.

    python ../../data/prepare_llrf.py --out llrf
    python feature_pipeline.py --data llrf            # shapes only
    python feature_pipeline.py --data llrf --insert
"""

import argparse
import time
from pathlib import Path

import pandas as pd

SPECS = {
    "llrf_events": dict(file="events.parquet", description="SPIRAL2 LLRF post-mortem events: cavity and loop settings"),
    "llrf_physics": dict(file="physics.parquet", description="Physics features per event (Zenodo 22729653 feature dictionary)"),
    "llrf_labels": dict(file="labels.parquet", description="Fault labels from the GANIL post-mortem classifier"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="output folder of data/prepare_llrf.py")
    ap.add_argument("--insert", action="store_true", help="insert into the feature store")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    frames = {name: pd.read_parquet(args.data / spec["file"]) for name, spec in SPECS.items()}
    for name, df in frames.items():
        print(f"{name:14s} {len(df):>5} rows x {df.shape[1]:>2} columns")
    if not args.insert:
        return

    import hopsworks

    fs = hopsworks.login().get_feature_store()
    for name, df in frames.items():
        fg = fs.get_or_create_feature_group(
            name=name, version=args.version, primary_key=["event_id"], event_time="event_ts",
            online_enabled=True, description=SPECS[name]["description"],
        )
        start = time.monotonic()
        fg.insert(df, wait=True)
        print(f"{name}: inserted {len(df)} rows in {time.monotonic() - start:.0f} s")


if __name__ == "__main__":
    main()
