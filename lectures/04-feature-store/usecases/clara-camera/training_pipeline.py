"""Train a surrogate of CLARA S02 on real bursts: magnet currents -> spot on CLA-S02-DIA-CAM-03.

One feature view joins two feature groups on file_id: the real clara_camera_files (the
currents and the mean spot of every burst, shared read-only from m4cast_clara) and your
clara_s02_bursts (which bursts are backgrounds, which scan a burst belongs to). Background
bursts are filtered out in the view.

The split is in time, on the bursts' event time: everything before --test-start trains, the
rest is the test. Two scores are printed, because they answer different questions:

    same scans   5-fold cross-validation inside the training period: can the model
                 interpolate between the settings of scans it has seen?
    later scans  the bursts after --test-start: can it predict scans it has not seen?

    # default: the last day of bursts is the test (laptop or cluster terminal)
    python training_pipeline.py

    # the morning's first two scans are known, the rest of the day is the test
    python training_pipeline.py --test-start "2026-04-24 14:59"
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

REAL_STORE = "m4cast_clara_featurestore"
FILES = "clara_camera_files"
BURSTS = "clara_s02_bursts"
NAME = "clara_s02_spot"
SEED = 0

# The five S02 quadrupoles sit between the section entrance and the camera: the knobs.
KNOBS = [f"cla_s02_mag_quad_0{i}_seti" for i in range(1, 6)]
# What else the files record upstream of the camera and the operators changed between scans:
# the S02 and S01 correctors and the gun solenoid. They say which beam a burst was taken with.
CONTEXT = [
    "cla_s02_mag_hcor_01_seti", "cla_s02_mag_vcor_01_seti", "cla_s02_mag_hcor_02_seti", "cla_s02_mag_vcor_02_seti",
    "cla_s01_mag_hcor_01_seti", "cla_s01_mag_vcor_01_seti", "cla_s01_mag_hcor_02_seti", "cla_s01_mag_vcor_02_seti",
    "cla_gun_mag_sol_02_seti",
]
FEATURES = KNOBS + CONTEXT
LABELS = ["mu_x_mm_mean", "mu_y_mm_mean", "sigma_x_mm_mean", "sigma_y_mm_mean"]
# From clara_s02_bursts: not model inputs, they pick the rows and name the scan.
HELPERS = ["scan_id", "scan_step", "background"]


def surrogate() -> ExtraTreesRegressor:
    # Trees need no scaling and never predict outside what they have seen, which is the
    # honest behaviour for a model of a machine: it cannot invent a spot nobody measured.
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)


def scores(y_true: pd.DataFrame, y_pred) -> pd.DataFrame:
    return pd.DataFrame({
        "r2": [r2_score(y_true[c], y_pred[:, i]) for i, c in enumerate(LABELS)],
        "mae_um": [1e3 * mean_absolute_error(y_true[c], y_pred[:, i]) for i, c in enumerate(LABELS)],
    }, index=[c.removesuffix("_mm_mean") for c in LABELS])


def feature_view(project, version: int):
    files = project.get_feature_store(name=REAL_STORE).get_feature_group(FILES, version=1)
    fs = project.get_feature_store()
    bursts = fs.get_feature_group(BURSTS, version=1)
    beam = bursts.select(HELPERS).filter(bursts.background == False)  # noqa: E712
    query = files.select(FEATURES + LABELS).join(beam, on=["file_id"], join_type="inner")
    fv = fs.get_or_create_feature_view(
        name=NAME,
        version=version,
        query=query,
        labels=LABELS,
        training_helper_columns=HELPERS,
        description="S02 magnet currents -> spot on CLA-S02-DIA-CAM-03, real beam bursts only",
    )
    return fv, bursts


def last_day(bursts) -> str:
    beam = bursts.select(["event_ts", "background"]).read()
    return str(beam[~beam.background].event_ts.max().date())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-start", help="bursts from this time on are the test (default: the last day of bursts)")
    ap.add_argument("--model-dir", type=Path, default=Path("model"))
    ap.add_argument("--fv-version", type=int, default=1)
    args = ap.parse_args()

    import hopsworks

    project = hopsworks.login()
    fv, bursts = feature_view(project, args.fv_version)
    test_start = args.test_start or last_day(bursts)
    # The helper columns are read too: the query service loads only the columns a query
    # selects, and the view filters on one of them.
    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_start=test_start, training_helper_columns=True, description=f"beam bursts, test from {test_start}"
    )
    X_train, X_test = X_train[FEATURES], X_test[FEATURES]
    print(f"{len(X_train)} beam bursts before {test_start} train, {len(X_test)} after are the test")

    same = scores(y_train, cross_val_predict(surrogate(), X_train, y_train, cv=KFold(5, shuffle=True, random_state=SEED)))
    model = surrogate().fit(X_train, y_train)
    later = scores(y_test, model.predict(X_test))
    print(pd.concat({"same scans": same, "later scans": later}, axis=1).to_string(float_format=lambda v: f"{v:.2f}"))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / "model.pkl")
    metrics = {f"{k}_{split}_{label}": float(v) for split, table in (("same", same), ("later", later))
               for k, column in table.items() for label, v in column.items()}
    entry = project.get_model_registry().python.create_model(
        name=NAME,
        metrics=metrics,
        feature_view=fv,
        input_example=X_train.head(1),
        description=f"Extra trees, S02 currents to spot moments in mm, trained on bursts before {test_start}",
    )
    # Keep model.pkl here too: inference_pipeline.py --local reads it.
    entry.save(os.fspath(args.model_dir), keep_original_files=True)
    print(f"registered {NAME} v{entry.version}")


if __name__ == "__main__":
    main()
