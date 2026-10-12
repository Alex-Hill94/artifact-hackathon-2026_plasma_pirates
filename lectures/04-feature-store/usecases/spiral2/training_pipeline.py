"""SPIRAL2 training pipeline: feature views, two classifiers, the model registry.

Two questions on the same rows, each one feature view and one model:

    llrf_fault            is this post-mortem file a fault or a normal capture
    llrf_fault_category   which of the seven fault categories, fault rows only

Both read four loop settings from llrf_events and the 42 features of llrf_physics, joined to
llrf_labels on event_id. Both split on time, not at random: events of one cavity in one week
look alike, so the models train on 2019 to 2023 and are tested on 2024 and 2025. Gradient
boosting takes the missing values as they are.

    # from the parquet folder written by data/prepare_llrf.py (no cluster needed)
    python training_pipeline.py --local llrf --models models

    # from the feature store, register both models with their feature views
    python training_pipeline.py
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 0
SPLIT = "2024-01-01"
START, END = "2019-01-01", "2026-01-01"
KEYS = ["event_id", "cavity_id", "event_ts"]
HEADER = ["kpi", "kii", "ampt", "phit"]

OBJECTIVES = {
    "llrf_fault": dict(
        label="is_fault",
        faults_only=False,
        description="Fault vs normal from header and physics features",
    ),
    "llrf_fault_category": dict(
        label="fault_category",
        faults_only=True,
        description="Fault category from header and physics features, fault rows only",
    ),
}


def train(name: str, spec: dict, X_train, X_test, y_train, y_test):
    model = HistGradientBoostingClassifier(max_iter=300, random_state=SEED).fit(X_train, y_train)
    predicted = model.predict(X_test)
    majority = y_train.mode()[0]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro")), 4),
        "majority_accuracy": round(float((y_test == majority).mean()), 4),
    }
    if y_train.nunique() == 2:
        metrics["auc"] = round(float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])), 4)

    print(f"\n{name}: trained on {len(X_train)} rows before {SPLIT}, tested on {len(X_test)} after")
    print(pd.Series(metrics).to_string())
    if spec["faults_only"]:
        counts = pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()})
        counts["test, right"] = y_test[predicted == y_test].value_counts()
        print(counts.fillna(0).astype(int).to_string())
    return model, metrics


def frames_local(data: Path, spec: dict):
    events = pd.read_parquet(data / "events.parquet")[["event_id", "event_ts", *HEADER]]
    physics = pd.read_parquet(data / "physics.parquet").drop(columns=["cavity_id", "event_ts"])
    labels = pd.read_parquet(data / "labels.parquet")[["event_id", "is_fault", "fault_category"]]
    rows = labels.merge(events, on="event_id").merge(physics, on="event_id")
    if spec["faults_only"]:
        rows = rows[rows.is_fault == 1]
    past = rows.event_ts < SPLIT
    X = rows.drop(columns=["event_id", "event_ts", "is_fault", "fault_category"])
    y = rows[spec["label"]]
    return X[past], X[~past], y[past], y[~past]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", type=Path, metavar="DIR", help="train from data/prepare_llrf.py parquet, skip the registry")
    ap.add_argument("--models", type=Path, default=Path("models"), help="where model.pkl files are written")
    ap.add_argument("--fg-version", type=int, default=1)
    ap.add_argument("--fv-version", type=int, default=1)
    ap.add_argument("--only", choices=list(OBJECTIVES), help="train one objective")
    args = ap.parse_args()

    objectives = {k: v for k, v in OBJECTIVES.items() if args.only in (None, k)}

    if args.local:
        for name, spec in objectives.items():
            model, _ = train(name, spec, *frames_local(args.local, spec))
            (args.models / name).mkdir(parents=True, exist_ok=True)
            joblib.dump(model, args.models / name / "model.pkl")
        return

    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    events = fs.get_feature_group("llrf_events", version=args.fg_version)
    physics = fs.get_feature_group("llrf_physics", version=args.fg_version)
    labels = fs.get_feature_group("llrf_labels", version=args.fg_version)
    for name, spec in objectives.items():
        query = (
            labels.select([spec["label"]])
            .join(events.select(HEADER), on=["event_id"])
            .join(physics.select_except(KEYS), on=["event_id"])
        )
        if spec["faults_only"]:
            query = query.filter(labels.is_fault == 1)
        fv = fs.get_or_create_feature_view(
            name=name, version=args.fv_version, query=query, labels=[spec["label"]], description=spec["description"]
        )
        X_train, X_test, y_train, y_test = fv.train_test_split(
            train_start=START, train_end=SPLIT, test_start=SPLIT, test_end=END,
            description=f"{name}: train before {SPLIT}, test after",
        )
        model, metrics = train(name, spec, X_train, X_test, y_train[spec["label"]], y_test[spec["label"]])
        (args.models / name).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.models / name / "model.pkl")
        entry = mr.python.create_model(
            name=name,
            metrics=metrics,
            feature_view=fv,
            input_example=X_train.head(1),
            description=f"{spec['description']}. Gradient boosting, trained before {SPLIT}, tested after.",
        )
        entry.save(os.fspath(args.models / name))
        print(f"registered {name} v{entry.version}")


if __name__ == "__main__":
    main()
