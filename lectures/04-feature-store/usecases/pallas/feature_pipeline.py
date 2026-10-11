"""PALLAS feature pipeline: the organisers' simulation pack, read over HTTPS, into four feature groups.

One row of campaign A or B is one Smilei simulation of the PALLAS laser-plasma accelerator.
The pipeline reads the pack as published, applies the data card's rules (config as the only id,
the moments table's "Config_<id>" spelling, injected = q_pC >= 3), adds a few derived features and
writes four feature groups:

    pallas_campaign_a       config                    p_1, a_0, c_N2, x_of -> bunch
    pallas_campaign_b       config                    P_max, cN2_max, L_inj, dip_frac, x_of -> bunch
    pallas_b_trajectories   config, z_um              the bunch at 113 positions
    pallas_b_moments        config, plane_um,         the bunch's 6x6 covariance at 20 planes
                            particle_group            (the pack's `group`: all or cohort)

Hopsworks stores feature names in lower case, so every column is lower-cased here and the local
parquet files match what the feature store returns.

    # the public 10 % sample, pinned commit, to parquet (no cluster needed)
    python feature_pipeline.py --out pallas_fg

    # the same frames into the feature store, all four groups or some of them
    python feature_pipeline.py --insert
    python feature_pipeline.py --insert --only pallas_campaign_b

    # the full pack once the organisers publish it: a URL or an unpacked folder
    python feature_pipeline.py --source https://<full-pack-base-url> --insert
    python feature_pipeline.py --source ~/pallas_hackathon_data --out pallas_fg
"""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SAMPLE = (
    "https://raw.githubusercontent.com/aghribi/artifact-hackathon-2026/"
    "de9c997f35dffc4b359050ad9298c75e065366f1/challenges/surrogate-models/data/sample"
)
INJECTED_PC = 3.0
# Per-row native spectra and the 2000-point density curve are left out: the spectrum is kept on
# the shared 200-bin grid (spec_common), and the curve is built from the four profile knobs.
HEAVY = ["ener_axis_MeV", "spec", "x_p", "n_e_p"]
LOG_COLUMNS = {"log10_q_pc": "q_pC", "log10_i_peak": "i_peak", "log10_n_emit_x": "n_emit_x", "log10_sigma_y": "sigma_y"}


def fetch(source: str, name: str) -> bytes:
    if source.startswith(("http://", "https://")):
        r = requests.get(f"{source.rstrip('/')}/{name}", timeout=120)
        r.raise_for_status()
        return r.content
    return (Path(source).expanduser() / name).read_bytes()


def read_parquet(source: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(fetch(source, name)))


def campaign(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.drop(columns=[c for c in HEAVY if c in df.columns])
    # the pack ships `injected` already; recompute it so a different pack cannot drift from the rule
    df["injected"] = df.q_pC >= INJECTED_PC
    for name, col in LOG_COLUMNS.items():
        df[name] = np.log10(df[col].where(df.injected & (df[col] > 0)))
    df["spec_common"] = df.spec_common.map(lambda a: None if a is None else np.asarray(a, dtype="float32").tolist())
    df["source"] = source
    df.columns = df.columns.str.lower()
    return df


def build(source: str) -> dict[str, pd.DataFrame]:
    ref = json.loads(fetch(source, "pack_reference.json"))
    reference_train = set(ref["sections"]["reference_split"]["train"])

    a = campaign(read_parquet(source, "campaign_A.parquet"), source)

    b = campaign(read_parquet(source, "campaign_B.parquet"), source)
    # the frozen split the organisers' reference numbers use; its test half is not public
    b["in_reference_train"] = b.config.isin(reference_train)

    traj = read_parquet(source, "campaign_B_trajectories.parquet")
    # float positions make poor keys: the scorer joins on z_mm rounded to 1 um, so the key does too
    traj.insert(1, "z_um", (traj.z_mm.astype("float64") * 1000).round().astype("int64"))
    traj["source"] = source
    traj.columns = traj.columns.str.lower()

    mom = read_parquet(source, "campaign_B_moments.parquet")
    mom.insert(0, "config", mom.sim_id.str.removeprefix("Config_").astype("int64"))
    mom.insert(1, "plane_um", (mom.plane_mm * 1000).round().astype("int64"))
    # the table's own `injection` disagrees with the pack's flag on 8 simulations; keep the pack's
    mom = mom.drop(columns=["sim_id", "injection"])
    # `group` is an SQL reserved word and the offline store is queried with SQL
    mom = mom.rename(columns={"group": "particle_group"})
    mom["source"] = source
    mom.columns = mom.columns.str.lower()

    return {
        "pallas_campaign_a": a,
        "pallas_campaign_b": b,
        "pallas_b_trajectories": traj,
        "pallas_b_moments": mom,
    }


SPECS = {
    "pallas_campaign_a": dict(
        primary_key=["config"],
        description="PALLAS campaign A: Smilei PIC runs, p_1, a_0, c_N2, x_of -> electron bunch. One row per simulation.",
    ),
    "pallas_campaign_b": dict(
        primary_key=["config"],
        description="PALLAS campaign B: Smilei PIC runs, density profile knobs -> electron bunch. One row per simulation.",
    ),
    "pallas_b_trajectories": dict(
        primary_key=["config", "z_um"],
        description="PALLAS campaign B: the bunch at 113 positions along the accelerator. z_um = round(z_mm * 1000).",
    ),
    "pallas_b_moments": dict(
        primary_key=["config", "plane_um", "particle_group"],
        description="PALLAS campaign B: the bunch's mean and 6x6 covariance at 20 planes. Moments z is transverse.",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SAMPLE, help="pack base URL or local folder (default: public sample, pinned commit)")
    ap.add_argument("--out", type=Path, help="folder to write one parquet per feature group")
    ap.add_argument("--insert", action="store_true", help="insert into the feature store")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--only", nargs="+", choices=list(SPECS), help="write only these feature groups")
    args = ap.parse_args()

    frames = {k: v for k, v in build(args.source).items() if args.only is None or k in args.only}
    for name, df in frames.items():
        injected = f", {int(df.injected.sum())} injected" if "injected" in df else ""
        print(f"{name:24s} {len(df):>7} rows x {df.shape[1]:>3} columns{injected}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, df in frames.items():
            df.to_parquet(args.out / f"{name}.parquet", index=False)
    if args.insert:
        import hopsworks

        fs = hopsworks.login().get_feature_store()
        for name, df in frames.items():
            # Simulations have no wall-clock time, so no event_time. Nothing is looked up online:
            # the surrogate's inputs come from the caller, so every group stays offline.
            fg = fs.get_or_create_feature_group(name=name, version=args.version, **SPECS[name])
            start = time.monotonic()
            fg.insert(df, wait=True)
            print(f"{name}: inserted {len(df)} rows in {time.monotonic() - start:.0f} s")
    if not (args.out or args.insert):
        for df in frames.values():
            print(df.drop(columns=["spec_common"], errors="ignore").head().to_string())


if __name__ == "__main__":
    main()
