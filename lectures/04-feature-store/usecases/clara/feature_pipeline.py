"""Synthetic CLARA S04 captures from Cheetah, one row per capture.

Each run sets the eight S04 magnets (four quadrupoles, two H/V corrector pairs) to
SETI currents drawn uniformly inside the real archiver operating range around the
reference tune, tracks the kickoff notebook's starting beam to CLA-S04-DIA-SCR-03
and records the spot centroid and RMS size the camera would see. Same shape as the
real clara_s07_sample.parquet: magnet settings paired with a screen reading.

    # 2000 captures to parquet, no cluster
    python feature_pipeline.py --runs 2000 --out data/

    # straight into the clara_s04_captures feature group
    python feature_pipeline.py --runs 2000 --campaign s04_a --insert
"""

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cheetah
import numpy as np
import pandas as pd

import clara

FEATURE_GROUP = "clara_s04_captures"


def rows(segment, beam, currents: list[dict], campaign: str, beam_seed: int, noise_fraction: float, rng) -> pd.DataFrame:
    """One row per capture: settings, reading, provenance."""
    provenance = {
        "campaign": campaign,
        "synthetic": True,
        "screen": clara.SCREEN_PV,
        "beam_seed": beam_seed,
        "noise_fraction": noise_fraction,
        "lattice_hash": clara.lattice_hash(),
        "cheetah_version": cheetah.__version__,
    }
    out = []
    for setting in currents:
        reading = clara.capture(segment, beam, setting, noise_fraction, rng)
        out.append({"run_id": str(uuid.uuid4()), "run_ts": datetime.now(timezone.utc)} | provenance | setting | reading)
    return pd.DataFrame(out)


def campaign(runs: int, name: str, seed: int | None, beam_seed: int, spread: float, noise_fraction: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    box = clara.bounds(spread)
    currents = [{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in box.items()} for _ in range(runs)]
    return rows(clara.section(), clara.starting_beam(beam_seed), currents, name, beam_seed, noise_fraction, rng)


def insert(df: pd.DataFrame, version: int = 1) -> None:
    import hopsworks

    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP,
        version=version,
        primary_key=["run_id"],
        event_time="run_ts",
        partition_key=["campaign"],
        description="Synthetic CLARA S04 captures: SETI currents to CLA-S04-DIA-SCR-03 spot (Cheetah)",
    )
    fg.insert(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--campaign", default=f"s04_{datetime.now():%Y%m%d_%H%M}")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--beam-seed", type=int, default=100, help="kickoff notebook starting beam (100-104)")
    ap.add_argument("--spread", type=float, default=1.0, help="fraction of each magnet's half range around the reference tune")
    # The kickoff notebook's NOISE_FRACTION. Nothing public documents CLARA's camera
    # noise, so it stays off unless you choose a value.
    ap.add_argument("--noise-fraction", type=float, default=0.0)
    ap.add_argument("--out", type=Path, help="directory to write <campaign>.parquet into")
    ap.add_argument("--insert", action="store_true", help=f"insert into the {FEATURE_GROUP} feature group")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    df = campaign(args.runs, args.campaign, args.seed, args.beam_seed, args.spread, args.noise_fraction)
    print(f"{len(df)} captures, {int(df.beam_lost.sum())} lost, lattice {df.lattice_hash[0]}, cheetah {df.cheetah_version[0]}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out / f"{args.campaign}.parquet", index=False)
    if args.insert:
        insert(df, args.version)
    if not (args.out or args.insert):
        print(df.head().to_string())


if __name__ == "__main__":
    main()
