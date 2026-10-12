"""Insert the output of extract_captures.sh (SLURM job on CC-IN2P3) into two offline feature groups.

Reads <results>/captures.parquet and files.parquet from the read-only results mount.

    clara_camera_captures   one row per capture of the whole archive: spot moments (mm and pixels),
                            frame metadata, every SETI/READI/GETSETI/RPOWER PV (null where a file has none)
    clara_camera_files      one row per HDF5 file: moments averaged over its beam captures and their
                            shot-to-shot spread, PV medians, thumbnail of the mean frame

Every row carries slurm_job_id, the CC-IN2P3 job that computed it.

    python ingest_full.py                       # from /m4cast-results/clara
    python ingest_full.py --results <dir> --dry-run
    python ingest_full.py --allow-partial       # insert although the job failed on some files
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("/m4cast-results/clara"))
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="read and check, write nothing")
    ap.add_argument("--allow-partial", action="store_true", help="insert even when the job failed on some files")
    args = ap.parse_args()

    manifest = json.loads((args.results / "manifest.json").read_text())
    job = str(manifest["slurm_job_id"])
    captures = pd.read_parquet(args.results / "captures.parquet").assign(slurm_job_id=job)
    files = pd.read_parquet(args.results / "files.parquet").assign(slurm_job_id=job)
    print(f"slurm job {job}: {len(files)} files, {len(captures)} captures, "
          f"{int(captures.beam_present.sum())} with a beam, {len(manifest['failed'])} files failed")
    for failure in manifest["failed"]:
        print(f"  failed: {failure['file']}: {failure['error']}")
    if args.dry_run:
        return
    if manifest["failed"] and not args.allow_partial:
        raise SystemExit(f"{len(manifest['failed'])} files failed in job {job}: nothing inserted, pass --allow-partial to insert the rest")

    import hopsworks

    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="clara_camera_captures", version=args.version, primary_key=["capture_id"], event_time="event_ts",
        partition_key=["camera"], online_enabled=False,
        description="Real CLARA camera captures, full CC-IN2P3 archive: beam spot moments (mm and pixels), "
                    "frame metadata, magnet SETI/READI/GETSETI/RPOWER",
    )
    fg.insert(captures, wait=True)
    fg = fs.get_or_create_feature_group(
        name="clara_camera_files", version=args.version, primary_key=["file_id"], event_time="event_ts",
        partition_key=["camera"], online_enabled=False,
        description="Real CLARA camera bursts (one HDF5 file each), full CC-IN2P3 archive: mean and shot-to-shot "
                    "spread of the spot moments, PV medians, thumbnail of the mean frame",
        statistics_config={"columns": [c for c in files.columns if c != "thumbnail"]},
    )
    fg.insert(files, wait=True)


if __name__ == "__main__":
    main()
