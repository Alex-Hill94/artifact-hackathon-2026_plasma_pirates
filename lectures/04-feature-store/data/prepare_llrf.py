"""Turn the SPIRAL2 LLRF Zenodo release into feature-group-ready parquet files.

Source: Zenodo record 22729653, "SPIRAL2 LLRF Fault Diagnosis: Structured Results Data (V7)",
GANIL / CNRS, MIT license. The release holds engineered features per post-mortem event,
not raw waveforms.

Writes to --out (default ./llrf):
    events.parquet     one row per event: keys, cavity, LLRF settings from the file header,
                       synthetic=False (generators/llrf_stream.py writes synthetic=True rows)
    physics.parquet    42 physics and status features (decay, detuning, microphonics, power flow)
    stats.parquet      260 whole-signal statistics, 17 per channel
    precursor.parquet  501 pre-trigger features (trends, CUSUM, segments)
    labels.parquet     binary label, fault category, RF-regulation subtype

Every table carries event_id (primary key), cavity_id and event_ts (event time).
The ALM header field is left out on purpose: it is set by the interlock that recorded the
fault, so it leaks the label.

    python prepare_llrf.py --out ./llrf
"""

import argparse
import hashlib
import json
import pickle
import re
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RECORD = "https://zenodo.org/api/records/22729653"
FILES = ["features_engineered_v7.pkl", "feature_dictionary.csv"]


def fetch(cache: Path) -> dict[str, Path]:
    """Download the record files once, verify md5 against the Zenodo API before any unpickling."""
    cache.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(RECORD) as r:
        meta = {f["key"]: f for f in json.load(r)["files"]}
    paths = {}
    for name in FILES:
        path = cache / name
        expected = meta[name]["checksum"].removeprefix("md5:")
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != expected:
            print(f"downloading {name} ({meta[name]['size'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(meta[name]["links"]["self"], path)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"{name}: md5 mismatch, refusing to load")
        paths[name] = path
    return paths


def column(name: str) -> str:
    """Hopsworks feature names: lowercase ascii, digits, underscores."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", ascii_name.lower())).strip("_")


def events_table(metadata: list[dict]) -> pd.DataFrame:
    h = pd.DataFrame(metadata)
    # DBNAME looks like LINA-CMA07-CAV1:Llrf
    parts = h["DBNAME"].str.removesuffix(":Llrf").str.split("-", expand=True)
    ts = pd.to_datetime(h["DATE"].str.title(), format="%b %d %Y %H:%M:%S")
    ev = pd.DataFrame({
        "cavity_id": parts[1] + "_" + parts[2],
        "event_ts": ts,
        "linac": parts[0],
        "cryomodule": parts[1],
        "cavity": parts[2],
        "ndec": h["NDEC"].astype(int),
        "nrow": h["NROW"].astype(int),
        "postrow": h["POSTROW"].astype(int),
        "kpi": h["KPI"].astype(float),
        "kii": h["KII"].astype(float),
        "kpuci": h["KPUCI"].astype(float),
        "kiuci": h["KIUCI"].astype(float),
        "ampt": h["AMPT"].astype(float),
        "phit": h["PHIT"].astype(float),
        "beam_flag": h["BEAM"].eq("OUI"),
        "cavite": h["CAVITE"].str.removeprefix("CAVITE_"),
        "mask": h["MASK"],
        "synthetic": False,
    })
    ev.insert(0, "event_id", ev["cavity_id"].str.lower() + "_" + ts.dt.strftime("%Y%m%dT%H%M%S"))
    # real events are their own source; replayed ones point at the real event they copy
    ev["source_event_id"] = ev["event_id"]
    if ev["event_id"].duplicated().any():
        raise SystemExit("event_id is not unique, check the source release")
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("llrf"))
    ap.add_argument("--cache", type=Path, default=Path(".zenodo"))
    args = ap.parse_args()

    paths = fetch(args.cache)
    with open(paths["features_engineered_v7.pkl"], "rb") as f:
        bundle = pickle.load(f)  # md5-verified above
    features = bundle["features_all"].replace([np.inf, -np.inf], np.nan)
    dictionary = pd.read_csv(paths["feature_dictionary.csv"])
    kept = set(bundle["feature_cols"])

    events = events_table(bundle["metadata"])
    keys = events[["event_id", "cavity_id", "event_ts"]]

    groups = {
        "stats": dictionary.physics_group.eq("Statistical"),
        "precursor": dictionary.physics_group.eq("Temporal"),
        # meta_* repeat the header values already in events
        "physics": ~dictionary.physics_group.isin(["Statistical", "Temporal"])
        & ~dictionary.feature_name.str.startswith("meta_"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.out / "events.parquet", index=False)
    print(f"events.parquet     {events.shape}")

    for name, mask in groups.items():
        cols = [c for c in dictionary.feature_name[mask] if c in kept]
        block = features[cols].rename(columns=column).astype("float64")
        if block.columns.duplicated().any():
            raise SystemExit(f"{name}: column names collide after sanitising")
        table = pd.concat([keys, block], axis=1)
        table.to_parquet(args.out / f"{name}.parquet", index=False)
        print(f"{name + '.parquet':<18} {table.shape}")

    labels = keys.assign(
        is_fault=bundle["y_binary"].astype(int),
        fault_category=bundle["y_defaut"].astype(str),
        rf_subtype=pd.Series(bundle["y_physics_subtype"]).fillna("").astype(str),
    )
    labels.to_parquet(args.out / "labels.parquet", index=False)
    print(f"labels.parquet     {labels.shape}")


if __name__ == "__main__":
    main()
