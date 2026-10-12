"""Tune a scan the surrogate has not seen, then check the answer against the real bursts.

The model's own test rows are real scans taken after its training period. One of them is
replayed: the optimiser knows the scan's recorded context (correctors, solenoid) and the range
the operators swept, asks the deployed surrogate instead of the machine, and proposes the
quadrupole currents for the kickoff notebook's spot: 100 um in both planes. The machine cannot
be shot from here, so the check is the real burst of that scan nearest to the proposal, next
to the best burst the operators actually measured.

Only the spot size is scored. The files hold no scan of the S02 correctors, so the surrogate
has nothing to learn steering from.

    # deploy (or reuse) the latest clara_s02_spot and tune the first unseen scan (cluster terminal)
    python inference_pipeline.py

    # the same against model/model.pkl, no deployment (laptop)
    python inference_pipeline.py --local model/
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

from training_pipeline import CONTEXT, FEATURES, KNOBS, LABELS, NAME

DEPLOYMENT = "claras02spot"  # deployment names are [a-zA-Z0-9] only
SIZES = ["sigma_x_mm_mean", "sigma_y_mm_mean"]
INITIAL_POINTS = 10  # random settings before the Gaussian process takes over


def objective_um(moments_mm, target_um: float) -> float:
    """Kickoff objective on the spot size: mean |sigma - target| over x and y, in um."""
    sizes = np.asarray(moments_mm, dtype=float)[[LABELS.index(c) for c in SIZES]] * 1e3
    return float(np.abs(sizes - target_um).mean())


def model_entry(project, version: int | None):
    mr = project.get_model_registry()
    # get_model without a version returns version 1, not the latest
    return mr.get_model(NAME, version=version or max(m.version for m in mr.get_models(NAME)))


def unseen_scans(entry) -> pd.DataFrame:
    """The model's test rows: real bursts after its training period, with their scan."""
    fv = entry.get_feature_view(init=False)
    _, X_test, _, y_test = fv.get_train_test_split(entry.training_dataset_version, training_helper_columns=True)
    return pd.concat([X_test, y_test], axis=1)


def deployed(project, entry):
    deployment = project.get_model_serving().get_deployment(DEPLOYMENT)
    if deployment is not None and deployment.model_version != entry.version:
        # A deployment keeps the model version it was created with.
        print(f"{DEPLOYMENT} serves v{deployment.model_version}, replacing it with v{entry.version}")
        deployment.stop()
        deployment.delete()
        deployment = None
    if deployment is None:
        script = project.get_dataset_api().upload(os.fspath(Path(__file__).with_name("predictor.py")), "Resources", overwrite=True)
        deployment = entry.deploy(name=DEPLOYMENT, script_file=os.path.join("/Projects", project.name, script))
    deployment.start(await_running=600)
    return lambda rows: deployment.predict(inputs=rows)["predictions"]


def line(label: str, knobs: dict, kind: str, moments_mm, target_um: float) -> str:
    setting = ", ".join(f"{k.removeprefix('cla_s02_mag_').removesuffix('_seti')} {v:.3f} A" for k, v in knobs.items())
    sx, sy = (float(moments_mm[LABELS.index(c)]) * 1e3 for c in SIZES)
    return f"{label:16s}{setting}   {kind} sigma {sx:.0f}, {sy:.0f} um   objective {objective_um(moments_mm, target_um):.0f} um"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", help="scan_id to tune (default: the first scan after the model's training period)")
    ap.add_argument("--target-um", type=float, default=100.0, help="spot size wanted in both planes")
    ap.add_argument("--calls", type=int, default=40, help="surrogate evaluations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-version", type=int, help="default: the latest")
    ap.add_argument("--local", type=Path, metavar="MODEL_DIR", help="call model.pkl from training_pipeline.py instead of a deployment")
    args = ap.parse_args()
    if args.calls <= INITIAL_POINTS:  # the start of the scan is evaluated on top of the random settings
        ap.error(f"--calls must be at least {INITIAL_POINTS + 1}")

    import hopsworks

    project = hopsworks.login()
    entry = model_entry(project, args.model_version)
    rows = unseen_scans(entry)
    scan = rows[rows.scan_id == (args.scan or rows.scan_id.min())].sort_values("scan_step").reset_index(drop=True)
    measured = scan[LABELS].to_numpy()

    if args.local:
        import joblib

        model = joblib.load(args.local / "model.pkl")
        predict = lambda settings: model.predict(pd.DataFrame(settings)[FEATURES])  # noqa: E731
    else:
        predict = deployed(project, entry)

    # Search what the operators swept in this scan, inside the range they swept it over.
    # Everything else stays where the scan had it.
    fixed = scan[FEATURES].median().to_dict()
    swept = [k for k in KNOBS if scan[k].round(3).nunique() > 1]
    space = [Real(float(scan[k].min()), float(scan[k].max()), name=k) for k in swept]

    def moments(x):
        return predict([fixed | dict(zip(swept, map(float, x)))])[0]

    warnings.filterwarnings("ignore", category=UserWarning)  # skopt/sklearn GP chatter
    # Start where the operators started the scan.
    start = scan.loc[0, swept].astype(float).tolist()
    result = gp_minimize(lambda x: objective_um(moments(x), args.target_um), space, x0=[start],
                         n_calls=args.calls, n_initial_points=INITIAL_POINTS, acq_func="EI", random_state=args.seed)

    span = np.array([d.high - d.low for d in space])
    distance = np.linalg.norm((scan[swept].to_numpy() - np.array(result.x)) / span, axis=1)
    nearest = int(distance.argmin())
    best = int(np.argmin([objective_um(m, args.target_um) for m in measured]))

    print(f"model {NAME} v{entry.version}, scan {scan.scan_id[0]}: {len(scan)} real beam bursts the model has not seen")
    print("searched " + ", ".join(f"{d.name.removeprefix('cla_s02_mag_').removesuffix('_seti')} in [{d.low:.2f}, {d.high:.2f}] A" for d in space)
          + f", {args.calls} surrogate calls, target sigma {args.target_um:.0f} um in x and y")
    print(line("start of scan", dict(zip(swept, start)), "measured", measured[0], args.target_um))
    print(line("surrogate best", dict(zip(swept, result.x)), "predicted", moments(result.x), args.target_um))
    print(line("nearest burst", scan.loc[nearest, swept].to_dict(), "measured", measured[nearest], args.target_um))
    print(line("best measured", scan.loc[best, swept].to_dict(), "measured", measured[best], args.target_um))
    print(f"context held at the scan's recorded values: {len(CONTEXT)} corrector and solenoid currents, {len(KNOBS) - len(swept)} quadrupoles not swept")


if __name__ == "__main__":
    main()
