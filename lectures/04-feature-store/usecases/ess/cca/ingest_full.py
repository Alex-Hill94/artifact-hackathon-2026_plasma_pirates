"""Insert the output of extract_sds.sh (SLURM job on CC-IN2P3) into two offline feature groups.

Reads <results>/llrf_waveforms.parquet and channel_stats.parquet from the read-only results mount.

    ess_llrf_waveforms      the public sample's schema for every event and RF section (I and Q at 256
                            points, lower-cased as i and q), plus in_sample / in_sample_holdout: the
                            organisers score on 42 sample rows, and those rows are in here verbatim
    ess_sds_channel_stats   full-resolution statistics of every RF chain of every cycle

Every row carries slurm_job_id, the CC-IN2P3 job that computed it.

    python ingest_full.py                       # from /m4cast-results/ess
    python ingest_full.py --results <dir> --dry-run
"""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import requests

SAMPLE_URL = (
    "https://raw.githubusercontent.com/aghribi/artifact-hackathon-2026/"
    "de9c997f35dffc4b359050ad9298c75e065366f1/challenges/anomaly-detection/data/sample/"
)
KEYS = ["event_id", "cycle_id", "pv_base"]


def sample_flags(waves: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_parquet(io.BytesIO(requests.get(SAMPLE_URL + "ess_llrf_sample.parquet", timeout=120).content))
    holdout = json.loads(requests.get(SAMPLE_URL + "holdout_ids.json", timeout=120).content)["holdout_idx"]
    sample["in_sample_holdout"] = False
    sample.loc[holdout, "in_sample_holdout"] = True
    flags = sample[KEYS + ["in_sample_holdout"]].assign(in_sample=True)
    out = waves.merge(flags, on=KEYS, how="left")
    out["in_sample"] = out.in_sample.eq(True)
    out["in_sample_holdout"] = out.in_sample_holdout.eq(True)
    matched = int(out.in_sample.sum())
    if matched != len(sample):
        raise SystemExit(f"only {matched} of {len(sample)} sample rows found in the full set")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("/m4cast-results/ess"))
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="read and check, write nothing")
    args = ap.parse_args()

    manifest = json.loads((args.results / "manifest.json").read_text())
    job = str(manifest["slurm_job_id"])
    waves = pd.read_parquet(args.results / "llrf_waveforms.parquet").rename(columns={"I": "i", "Q": "q"})
    stats = pd.read_parquet(args.results / "channel_stats.parquet")
    for df in (waves, stats):
        for c in df.columns:
            if pd.api.types.is_unsigned_integer_dtype(df[c]):
                df[c] = df[c].astype("int64")
        df["slurm_job_id"] = job
    waves = sample_flags(waves)
    print(f"slurm job {job}: {manifest['events']} events, {len(waves)} waveform rows "
          f"({int(waves.in_sample.sum())} in the sample, {int(waves.in_sample_holdout.sum())} holdout), "
          f"{len(stats)} channel rows")
    if args.dry_run:
        return

    import hopsworks

    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="ess_llrf_waveforms", version=args.version, primary_key=KEYS, event_time="cycle_ts",
        online_enabled=False,
        description="ESS SDS captures, full CC-IN2P3 archive: DIG-101:Dwn0 I/Q at 256 points per cycle and RF section "
                    "(public sample schema), sample and holdout flags",
        statistics_config={"columns": [c for c in waves.columns if c not in ("i", "q")]},
    )
    fg.insert(waves, wait=True)
    fg = fs.get_or_create_feature_group(
        name="ess_sds_channel_stats", version=args.version, primary_key=KEYS, online_enabled=False,
        description="ESS SDS captures, full CC-IN2P3 archive: full-resolution statistics of every RF chain per cycle",
    )
    fg.insert(stats, wait=True)


if __name__ == "__main__":
    main()
