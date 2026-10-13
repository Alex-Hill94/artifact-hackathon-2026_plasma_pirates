"""Train a surrogate for CLARA S04: eight SETI currents -> SCR-03 centroid and spot size.

Reads the clara_s04_captures feature group through the clara_s04_surrogate feature
view (captures where the beam reached the screen), fits a scaled MLP, registers it
with the view so the model knows which rows it learned from.

    # from parquet written by feature_pipeline.py --out, no cluster
    python training_pipeline.py --local data/ --model-dir model/

    # from the feature store, registered in the model registry
    python training_pipeline.py
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import clara

FEATURE_VIEW = "clara_s04_surrogate"
MODEL = "clara_s04_surrogate"
TEST_SIZE = 0.2


def surrogate():
    # Scaling lives in the model, not the view: the predictor then needs nothing but
    # the pickle, and train and serve apply the same fitted numbers.
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128, 128), max_iter=2000, early_stopping=True, random_state=0)
    return make_pipeline(StandardScaler(), TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler()))


def fit(X_train, X_test, y_train, y_test):
    model = surrogate().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"r2": float(r2_score(y_test, pred))}
    for i, label in enumerate(clara.LABELS):
        metrics[f"r2_{label}"] = float(r2_score(y_test.iloc[:, i], pred[:, i]))
        metrics[f"mae_{label}"] = float(mean_absolute_error(y_test.iloc[:, i], pred[:, i]))
    return model, metrics


def local_split(data: Path):
    df = pd.read_parquet(data)
    df = df[~df.beam_lost]
    return train_test_split(df[clara.FEATURES], df[clara.LABELS], test_size=TEST_SIZE, random_state=0)


def feature_store_split(version: int):
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()
    captures = fs.get_feature_group("clara_s04_captures", version=1)
    query = captures.select(clara.FEATURES + clara.LABELS).filter(captures.beam_lost == False)  # noqa: E712
    fv = fs.get_or_create_feature_view(
        name=FEATURE_VIEW,
        version=version,
        query=query,
        labels=clara.LABELS,
        description="S04 SETI currents -> CLA-S04-DIA-SCR-03 centroid and RMS size, beam on screen",
    )
    # Simulation runs are independent draws, so a random split does not leak.
    return project, fv, fv.train_test_split(test_size=TEST_SIZE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", type=Path, metavar="DIR", help="parquet directory from feature_pipeline.py --out")
    ap.add_argument("--model-dir", type=Path, default=Path("model"))
    ap.add_argument("--version", type=int, default=1, help="feature view version")
    args = ap.parse_args()

    if args.local:
        X_train, X_test, y_train, y_test = local_split(args.local)
    else:
        project, fv, (X_train, X_test, y_train, y_test) = feature_store_split(args.version)
    X_train, X_test = X_train[clara.FEATURES], X_test[clara.FEATURES]
    y_train, y_test = y_train[clara.LABELS], y_test[clara.LABELS]

    model, metrics = fit(X_train, X_test, y_train, y_test)
    print(f"{len(X_train)} train, {len(X_test)} test")
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / "model.pkl")
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if not args.local:
        entry = project.get_model_registry().python.create_model(
            name=MODEL,
            metrics=metrics,
            feature_view=fv,
            input_example=X_train.head(1),
            description="MLP surrogate of Cheetah S04 tracking, SETI currents to SCR-03 moments in um",
        )
        entry.save(str(args.model_dir))
        print(f"registered {MODEL} v{entry.version}")


if __name__ == "__main__":
    main()
