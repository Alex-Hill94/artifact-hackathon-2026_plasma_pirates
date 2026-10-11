"""Centre and focus the beam on CLA-S04-DIA-SCR-03 with Bayesian optimisation on the surrogate.

The optimiser asks the deployed surrogate, not the machine: 80 predictions instead of
80 shots. The best setting then gets one real Cheetah run, which is written back to
clara_s04_captures as campaign "bo_check" so the next surrogate trains on it.
Target and objective are the kickoff notebook's: centred, 100 um round spot, mean
absolute error.

    # against model/model.pkl, check rows appended to data/
    python inference_pipeline.py --local model/ --out data/

    # deploy clara_s04_surrogate, optimise against the deployment, write back
    python inference_pipeline.py
"""

import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

import clara
import feature_pipeline

MODEL = "clara_s04_surrogate"
DEPLOYMENT = "clarasurrogate"  # deployment names are [a-zA-Z0-9] only


def optimise(predict, n_calls: int, spread: float, seed: int):
    """BO over the eight SETI currents. `predict` maps a list of settings to moments in um."""
    box = clara.bounds(spread)
    space = [Real(lo, hi, name=name) for name, (lo, hi) in box.items()]
    reference = clara.reference_currents()

    def objective(x):
        return clara.objective_um(predict([dict(zip(clara.FEATURES, x))])[0])

    warnings.filterwarnings("ignore", category=UserWarning)  # skopt/sklearn GP chatter
    # Start from the reference tune, the way an operator would.
    x0 = [float(np.clip(reference[name], lo, hi)) for name, (lo, hi) in box.items()]
    result = gp_minimize(objective, space, x0=[x0], n_calls=n_calls, n_initial_points=20, acq_func="EI", random_state=seed)
    return dict(zip(clara.FEATURES, map(float, result.x))), float(result.fun)


def check(best: dict, beam_seed: int) -> pd.DataFrame:
    """One Cheetah run at the optimiser's setting, as a capture row tagged bo_check."""
    rng = np.random.default_rng()
    return feature_pipeline.rows(clara.section(), clara.starting_beam(beam_seed), [best], "bo_check", beam_seed, 0.0, rng)


def deployed_predict(model_version: int):
    import hopsworks

    project = hopsworks.login()
    ms = project.get_model_serving()
    deployment = ms.get_deployment(DEPLOYMENT)
    if deployment is None:
        script = Path(__file__).with_name("predictor.py")
        uploaded = project.get_dataset_api().upload(str(script), "Resources", overwrite=True)
        entry = project.get_model_registry().get_model(MODEL, version=model_version)
        deployment = entry.deploy(name=DEPLOYMENT, script_file=os.path.join("/Projects", project.name, uploaded))
    deployment.start(await_running=600)
    return lambda settings: deployment.predict(inputs=settings)["predictions"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", type=Path, metavar="MODEL_DIR", help="use model.pkl from training_pipeline.py --local")
    ap.add_argument("--out", type=Path, help="with --local: directory to append the bo_check row to")
    ap.add_argument("--calls", type=int, default=80, help="surrogate evaluations")
    ap.add_argument("--spread", type=float, default=1.0, help="same box as the feature pipeline")
    ap.add_argument("--beam-seed", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-version", type=int, default=1, help="model to deploy if the deployment does not exist yet")
    args = ap.parse_args()

    if args.local:
        model = joblib.load(args.local / "model.pkl")
        predict = lambda settings: model.predict(pd.DataFrame(settings)[clara.FEATURES])  # noqa: E731
    else:
        predict = deployed_predict(args.model_version)

    best, predicted = optimise(predict, args.calls, args.spread, args.seed)
    df = check(best, args.beam_seed)
    simulated = clara.objective_um(df[clara.LABELS].iloc[0])
    reference = clara.capture(clara.section(), clara.starting_beam(args.beam_seed), clara.reference_currents())

    print(f"target          {clara.TARGET_UM} um (mu_x, mu_y, sigma_x, sigma_y)")
    print(f"reference tune  {np.round([reference[k] for k in clara.LABELS], 1)}  objective {clara.objective_um([reference[k] for k in clara.LABELS]):.1f} um")
    print(f"surrogate says  {np.round(predict([best])[0], 1)}  objective {predicted:.1f} um")
    print(f"cheetah says    {np.round(df[clara.LABELS].iloc[0].to_numpy(), 1)}  objective {simulated:.1f} um")
    print("best SETI (A)   " + ", ".join(f"{k.removeprefix('cla_s04_mag_').removesuffix('_seti')}={v:.3f}" for k, v in best.items()))

    if args.local:
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(args.out / f"bo_check_{datetime.now():%Y%m%dT%H%M%S}.parquet", index=False)
    else:
        feature_pipeline.insert(df)


if __name__ == "__main__":
    main()
