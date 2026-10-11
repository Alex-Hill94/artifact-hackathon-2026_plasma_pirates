"""PALLAS inference pipeline: deploy the Direct B surrogate, then search it for a target beam.

The deployment serves pallas_direct_b with predictor.py: five density-profile knobs in,
E_med_MeV, dE_mad and q_end out. The optimiser (scikit-optimize, Bayesian) asks it for settings
that give a bunch at a target median energy with a small spread and enough charge. Each call is
one HTTP request to the deployment; a Smilei run takes hours, a surrogate call milliseconds.

The search space is the range of the rows the model was trained on, read from its feature view,
so the optimiser never asks the surrogate to extrapolate. The operator can fix the focal
position x_of, as in the challenge's inverse tasks.

    # deploy (or reuse) and search for a 200 MeV bunch with at least 20 pC
    python inference_pipeline.py --target-mev 200 --min-charge-pc 20

    # same, focal position fixed by the operator
    python inference_pipeline.py --target-mev 200 --x-of 700

    # without a cluster: the model.pkl and parquet folder from the local runs of the other two steps
    python inference_pipeline.py --local models/pallas_direct_b/model.pkl --fg-dir pallas_fg --target-mev 200
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

MODEL = "pallas_direct_b"
DEPLOYMENT = "pallasdirectb"
KNOBS = ["p_max", "cn2_max", "l_inj", "dip_frac", "x_of"]
HERE = Path(__file__).resolve().parent


def model_entry(project, version: int | None):
    mr = project.get_model_registry()
    # get_model without a version returns version 1, not the latest
    return mr.get_model(MODEL, version=version) if version else mr.get_best_model(MODEL, "r2_mean", "max")


def deploy(project, entry):
    ms = project.get_model_serving()
    deployment = ms.get_deployment(DEPLOYMENT)
    if deployment is None:
        uploaded = project.get_dataset_api().upload(os.fspath(HERE / "predictor.py"), "Resources", overwrite=True)
        deployment = entry.deploy(
            name=DEPLOYMENT,
            script_file=os.path.join("/Projects", project.name, uploaded),
            description="PALLAS Direct B surrogate",
        )
    deployment.start(await_running=600)
    return deployment


def loss(beam: list[float], target_mev: float, min_charge_pc: float) -> float:
    energy, spread, charge_c = beam
    charge_pc = charge_c * 1e12
    short = max(0.0, (min_charge_pc - charge_pc) / min_charge_pc)
    return ((energy - target_mev) / target_mev) ** 2 + spread + 10 * short**2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-mev", type=float, required=True, help="target median energy, MeV")
    ap.add_argument("--min-charge-pc", type=float, default=10.0, help="charge below this is penalised, pC")
    ap.add_argument("--x-of", type=float, help="fix the focal position (um) instead of searching it")
    ap.add_argument("--calls", type=int, default=60, help="surrogate evaluations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-version", type=int, help="default: the version with the best r2_mean")
    ap.add_argument("--local", type=Path, metavar="MODEL_PKL", help="call this model.pkl instead of a deployment")
    ap.add_argument("--fg-dir", type=Path, help="with --local: feature_pipeline.py --out folder, for the search bounds")
    args = ap.parse_args()

    if args.local:
        import joblib

        from training_pipeline import rows_local

        model = joblib.load(args.local)
        rows = rows_local(MODEL, pd.read_parquet(args.fg_dir / "pallas_campaign_b.parquet"))[KNOBS]

        def surrogate(x):
            return model.predict(np.asarray([x], dtype=float))[0].tolist()
    else:
        import hopsworks

        project = hopsworks.login()
        entry = model_entry(project, args.model_version)
        rows = entry.get_feature_view().get_batch_data()[KNOBS]
        deployment = deploy(project, entry)

        def surrogate(x):
            return deployment.predict(inputs=[list(map(float, x))])["predictions"][0]

    searched = [k for k in KNOBS if not (k == "x_of" and args.x_of is not None)]
    space = [Real(float(rows[k].min()), float(rows[k].max()), name=k) for k in searched]

    def settings(x):
        point = dict(zip(searched, x))
        if args.x_of is not None:
            point["x_of"] = args.x_of
        return [point[k] for k in KNOBS]

    result = gp_minimize(
        lambda x: loss(surrogate(settings(x)), args.target_mev, args.min_charge_pc),
        space, n_calls=args.calls, random_state=args.seed,
    )
    best = settings(result.x)
    energy, spread, charge = surrogate(best)
    print(pd.Series(best, index=KNOBS).round(4).to_string())
    print(f"predicted: E_med {energy:.1f} MeV, dE_mad {spread:.3f}, q_end {charge * 1e12:.1f} pC (loss {result.fun:.4f})")
    print("Check it with one Smilei run at these settings and write that run back to pallas_campaign_b.")


if __name__ == "__main__":
    main()
