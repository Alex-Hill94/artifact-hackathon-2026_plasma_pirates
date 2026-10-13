"""CLARA S02 bursts: which real bursts show the beam, and which scan they belong to.

Reads the real clara_camera_captures of one camera from the shared m4cast_clara feature store
and writes one row per burst (one HDF5 file, ten shots at fixed settings) into your own
project. Two things the archive does not say anywhere:

    background   operators often save two bursts per magnet setting, seconds apart: a faint
                 one without the photo-injector beam, then the beam. No PV marks which is
                 which. A burst is a background when the next burst has the same magnet
                 settings and several times its light.
    scan_id      a scan sweeps quadrupoles and leaves everything else alone. A pause of more
                 than ten minutes, or a change of any other recorded current, starts a new
                 scan, named by its start time. scan_step is the burst's place in it.

The camera is CLA-S02-DIA-CAM-03: S02 is the last section whose magnets the files record,
and this camera sits after its five quadrupoles.

    # look at the result, write nothing (laptop or cluster terminal)
    python feature_pipeline.py --out data/

    # into the clara_s02_bursts feature group of your project (cluster terminal)
    python feature_pipeline.py --insert
"""

import argparse
import time
from pathlib import Path

import pandas as pd

REAL_STORE = "m4cast_clara_featurestore"
CAPTURES = "clara_camera_captures"
FEATURE_GROUP = "clara_s02_bursts"
CAMERA = "CLA-S02-DIA-CAM-03"


def read_captures(real, camera: str) -> pd.DataFrame:
    captures = real.get_feature_group(CAPTURES, version=1)
    # Every commanded current the files record: two bursts are "at the same settings" when all agree.
    settings = [f.name for f in captures.columns if f.name.endswith("_seti") and not f.name.endswith("_getseti")]
    columns = ["file_id", "event_ts", "camera", "beam_present", "signal_sum"] + settings
    return captures.select(columns).filter(captures.camera == camera).read()


def bursts(captures: pd.DataFrame, pair_gap_s: float, pair_ratio: float, scan_gap_s: float) -> pd.DataFrame:
    settings = [c for c in captures.columns if c.endswith("_seti")]
    by_file = captures.groupby("file_id")
    df = by_file.agg(event_ts=("event_ts", "min"), camera=("camera", "first"), shots=("signal_sum", "size"),
                     beam_shots=("beam_present", "sum"), signal_sum=("signal_sum", "median"))
    # The power supplies are set to 1 mA; anything closer is the same setting.
    df = df.join(by_file[settings].median().round(3)).sort_values("event_ts").reset_index()

    following = df.shift(-1)
    gap_s = (following.event_ts - df.event_ts).dt.total_seconds()
    same_settings = (df[settings].fillna(0) == following[settings].fillna(0)).all(axis=1)
    df["background"] = same_settings & (gap_s <= pair_gap_s) & (following.signal_sum >= pair_ratio * df.signal_sum)

    context = [c for c in settings if "_quad_" not in c]
    paused = df.event_ts.diff().dt.total_seconds().fillna(scan_gap_s + 1) > scan_gap_s
    moved = (df[context].fillna(0) != df[context].shift().fillna(0)).any(axis=1)
    df["scan_id"] = df.event_ts.where(paused | moved).ffill().dt.strftime("%Y%m%dT%H%M%S")
    df["scan_step"] = df.groupby("scan_id").cumcount()
    df["beam_shots"] = df.beam_shots.astype("int64")
    return df[["file_id", "event_ts", "camera", "scan_id", "scan_step", "background", "shots", "beam_shots", "signal_sum"]]


def insert(fs, df: pd.DataFrame, version: int) -> None:
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP,
        version=version,
        primary_key=["file_id"],
        # No event time: a burst's flags never change, so a view joins this group on the key
        # alone instead of point in time.
        description="CLARA S02 camera bursts: background flag and scan, derived from the real captures",
    )
    start = time.time()
    fg.insert(df, wait=True)
    print(f"inserted {len(df)} rows into {FEATURE_GROUP} v{version} in {time.time() - start:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=CAMERA)
    ap.add_argument("--pair-gap", type=float, default=15.0, help="a background and its beam burst are at most this many seconds apart")
    ap.add_argument("--pair-ratio", type=float, default=3.0, help="the beam burst has at least this many times the background's light")
    ap.add_argument("--scan-gap", type=float, default=600.0, help="a pause longer than this many seconds starts a new scan")
    ap.add_argument("--out", type=Path, help="directory to write bursts.parquet into")
    ap.add_argument("--insert", action="store_true", help=f"insert into the {FEATURE_GROUP} feature group of your project")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    import hopsworks

    project = hopsworks.login()
    start = time.time()
    captures = read_captures(project.get_feature_store(name=REAL_STORE), args.camera)
    print(f"read {len(captures)} captures of {args.camera} in {time.time() - start:.0f} s")

    df = bursts(captures, args.pair_gap, args.pair_ratio, args.scan_gap)
    beam = df[~df.background]
    print(f"{len(df)} bursts, {int(df.background.sum())} background, {len(beam)} beam, {df.scan_id.nunique()} scans")
    print(f"median light per shot: background {df[df.background].signal_sum.median():.3g}, beam {beam.signal_sum.median():.3g} counts")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out / "bursts.parquet", index=False)
    if args.insert:
        insert(project.get_feature_store(), df, args.version)
    if not (args.out or args.insert):
        print(df.head(6).to_string())


if __name__ == "__main__":
    main()
