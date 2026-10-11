"""Training pipeline: unsupervised anomaly detector on ESS waveform features.

Reads the per-cycle features through the feature view ess_llrf_anomaly (features of
ess_waveform_features, keyed on event_id and cycle_id), trains a scaler plus IsolationForest,
and registers it with the feature view. No confirmed fault labels exist for this data, so the
model learns the envelope of the real captures and scores distance from it.

The split is the challenge's own: holdout_ids.json lists row positions into the public sample,
turned here into (event_id, cycle_id) keys. Training reads only the time window of the real
sample, so synthetic rows the generator streams later never reach training.

    # from the parquet files of feature_pipeline.py --out, no cluster
    python training_pipeline.py --local data

    # from the feature store, register the model
    python training_pipeline.py
"""

import argparse
import json
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_pipeline import KEYS, SAMPLE_URL, load_sample


def holdout_keys(source: str) -> pd.MultiIndex:
    split = json.load(urllib.request.urlopen(source.rstrip("/") + "/holdout_ids.json")) \
        if source.startswith("http") else json.loads((Path(source) / "holdout_ids.json").read_text())
    keys = load_sample(source)[KEYS]
    return pd.MultiIndex.from_frame(keys.iloc[split["holdout_idx"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", type=Path, metavar="DIR", help="train from waveform_features.parquet in DIR")
    ap.add_argument("--source", default=SAMPLE_URL, help="folder or URL holding the sample and holdout_ids.json")
    ap.add_argument("--features-fg", default="ess_waveform_features")
    ap.add_argument("--fg-version", type=int, default=1)
    ap.add_argument("--name", default="ess_llrf_anomaly", help="feature view and model name")
    ap.add_argument("--version", type=int, default=1, help="feature view version")
    ap.add_argument("--start", default="2026-06-01", help="training window start: the public sample")
    ap.add_argument("--end", default="2026-07-01", help="training window end")
    ap.add_argument("--contamination", type=float, default=0.05, help="share of training cycles flagged")
    ap.add_argument("--model-dir", type=Path, default=Path("model"))
    args = ap.parse_args()

    if args.local:
        X = pd.read_parquet(args.local / "waveform_features.parquet").drop(columns="event_ts")
    else:
        import hopsworks

        project = hopsworks.login()
        fs = project.get_feature_store()
        features = fs.get_feature_group(args.features_fg, version=args.fg_version)
        fv = fs.get_or_create_feature_view(
            name=args.name, version=args.version,
            query=features.select_except([*KEYS, "event_ts"]),
            description="Per-cycle ESS LLRF waveform features for unsupervised anomaly scoring",
        )
        X, _ = fv.training_data(start_time=args.start, end_time=args.end, primary_key=True)
        # 5.1.1 returns keys as <project>_<fg>_<version>_<key>, even for a one-group view
        X = X.rename(columns=lambda c: next((k for k in KEYS if c.endswith("_" + k)), c))

    holdout = X.set_index(KEYS).index.isin(holdout_keys(args.source))
    keys = X[KEYS]
    X = X.drop(columns=KEYS)
    X_train, X_test = X[~holdout], X[holdout]
    print(f"train {len(X_train)} cycles, holdout {len(X_test)} cycles, {X.shape[1]} features")

    model = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=300, contamination=args.contamination, random_state=42),
    ).fit(X_train)
    # score_samples is higher for normal points; flip it so higher means more anomalous
    train_score, test_score = -model.score_samples(X_train), -model.score_samples(X_test)
    metrics = {
        "train_mean_score": float(train_score.mean()),
        "holdout_mean_score": float(test_score.mean()),
        "holdout_flag_rate": float((model.predict(X_test) == -1).mean()),
    }
    for k, v in metrics.items():
        print(f"{k:20s} {v:.4f}")
    top = keys[holdout].assign(score=test_score).nlargest(3, "score")
    print("top holdout cycles:\n" + top.to_string(index=False))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / "model.pkl")
    print(f"saved {args.model_dir / 'model.pkl'}")

    if not args.local:
        entry = project.get_model_registry().python.create_model(
            name=args.name, metrics=metrics,
            feature_view=fv, input_example=X_train.head(1),
            description="StandardScaler + IsolationForest on ESS waveform features, challenge holdout split",
        )
        entry.save(str(args.model_dir))
        print(f"registered {args.name} version {entry.version}")


if __name__ == "__main__":
    main()
