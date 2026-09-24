"""Run a Cheetah quadrupole scan and store every run as a row.

Same lattice and beam as lectures/02-accelerators-and-ai/notebooks/03_surrogate_model:
drift, QF, drift, QD, drift, screen; 150 MeV protons; k1 in [-15, 15] T/m^2.
Each run records its inputs, outputs, lattice hash and Cheetah version, so a surrogate
trained later can say exactly which simulations it learned from.

    # 500 runs to parquet
    python sim_campaign.py --runs 500 --out runs.parquet

    # 2000 runs straight into a feature group, tagged with a campaign name
    python sim_campaign.py --runs 2000 --campaign scan_a --insert quad_scan_runs
"""

import argparse
import hashlib
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import cheetah

PROTON_REST_MASS_MEV = 938.272
KINETIC_MEV = 150.0
K1_MIN, K1_MAX = -15.0, 15.0


def lattice() -> tuple[cheetah.Segment, cheetah.ParameterBeam, str]:
    elements = [
        cheetah.Drift(length=torch.tensor(0.5)),
        cheetah.Quadrupole(length=torch.tensor(0.1), name="QF"),
        cheetah.Drift(length=torch.tensor(1.0)),
        cheetah.Quadrupole(length=torch.tensor(0.1), name="QD"),
        cheetah.Drift(length=torch.tensor(0.5)),
        cheetah.Screen(name="monitor"),
    ]
    beam = cheetah.ParameterBeam.from_twiss(
        energy=torch.tensor((KINETIC_MEV + PROTON_REST_MASS_MEV) * 1e6),
        beta_x=torch.tensor(1.0),
        beta_y=torch.tensor(1.0),
        emittance_x=torch.tensor(1e-6),
        emittance_y=torch.tensor(1e-6),
        species=cheetah.Species("proton"),
    )
    spec = ";".join(f"{type(e).__name__}:{e.name if type(e).__name__ != 'Drift' else ''}:{float(e.length):g}" for e in elements)
    return cheetah.Segment(elements=elements), beam, hashlib.sha1(spec.encode()).hexdigest()[:12]


def run(runs: int, campaign: str, seed: int | None) -> pd.DataFrame:
    segment, beam, lattice_hash = lattice()
    rng = np.random.default_rng(seed)
    k1 = rng.uniform(K1_MIN, K1_MAX, size=(runs, 2))
    rows = []
    for k1_qf, k1_qd in k1:
        segment.QF.k1 = torch.tensor(k1_qf, dtype=torch.float32)
        segment.QD.k1 = torch.tensor(k1_qd, dtype=torch.float32)
        out = segment.track(beam)
        sx, sy = out.sigma_x.item() * 1e3, out.sigma_y.item() * 1e3
        rows.append({
            "run_id": str(uuid.uuid4()),
            "run_ts": datetime.now(timezone.utc),
            "campaign": campaign,
            "lattice_hash": lattice_hash,
            "cheetah_version": cheetah.__version__,
            "k1_qf": float(k1_qf),
            "k1_qd": float(k1_qd),
            "sigma_x_mm": sx,
            "sigma_y_mm": sy,
            "diverged": not (np.isfinite(sx) and np.isfinite(sy)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--campaign", default=f"scan_{datetime.now():%Y%m%d_%H%M}")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", help="parquet file to write")
    ap.add_argument("--insert", metavar="FG", help="feature group to insert into (created if missing)")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    df = run(args.runs, args.campaign, args.seed)
    print(f"{len(df)} runs, {int(df.diverged.sum())} diverged, lattice {df.lattice_hash[0]}")

    if args.out:
        df.to_parquet(args.out, index=False)
    if args.insert:
        import hopsworks

        fs = hopsworks.login().get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=args.insert,
            version=args.version,
            primary_key=["run_id"],
            event_time="run_ts",
            partition_key=["campaign"],
            # online so the surrogate deployment can serve from this view
            online_enabled=True,
            description="Cheetah quadrupole scan runs (lecture 02 lattice)",
        )
        fg.insert(df)
    if not (args.out or args.insert):
        print(df.head().to_string())


if __name__ == "__main__":
    main()
