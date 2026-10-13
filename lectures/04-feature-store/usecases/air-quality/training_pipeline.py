"""Air quality training pipeline: one feature view, one regressor, the model registry.

The view pm25_forecast takes the day's PM2.5 from pm25_daily as the label and joins, on
station_id and as of the label's day, the rolling features of pm25_rolling and the weather of
weather_daily. The split is on time, not at random: consecutive days at one station look
alike, so the model trains on one period and is tested on the days after it. A station's
first days have no rolling features yet; gradient boosting takes the missing values as they are.

The score to beat is persistence: predicting that today equals yesterday (pm25_lag1).

    # from the parquet folder written by data/prepare_air_quality.py (no cluster needed)
    python training_pipeline.py --local air

    # from the feature store, register the model with its feature view
    python training_pipeline.py

    # the same view, a later cut: the next model version
    python training_pipeline.py --train-end 2025-10-01 --test-end 2026-04-01
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

NAME = "pm25_forecast"
LABEL = "pm25"
KEYS = ["station_id", "event_ts"]
SEED = 0


def train(X_train, X_test, y_train, y_test, args):
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predicted)), 3),
        "r2": round(float(r2_score(y_test, predicted)), 3),
        "persistence_mae": round(float(mean_absolute_error(y_test, X_test.pm25_lag1)), 3),
        "persistence_r2": round(float(r2_score(y_test, X_test.pm25_lag1)), 3),
    }
    print(f"{NAME}: trained on {len(X_train)} rows, {args.train_start} to {args.train_end}; tested on {len(X_test)}, to {args.test_end}")
    print(pd.DataFrame({
        "model": [metrics["mae"], metrics["r2"]],
        "persistence": [metrics["persistence_mae"], metrics["persistence_r2"]],
    }, index=["mae, µg/m³", "r2"]).to_string())
    return model, metrics


def frames_local(data: Path, args):
    from feature_pipeline import rolling

    readings = pd.read_parquet(data / "readings.parquet")[[*KEYS, LABEL]].sort_values("event_ts")
    weather = pd.read_parquet(data / "weather.parquet").sort_values("event_ts")
    # as the view does: the latest row of each group at or before the label's day
    rows = pd.merge_asof(readings, rolling(readings).sort_values("event_ts"), on="event_ts", by="station_id")
    rows = pd.merge_asof(rows, weather, on="event_ts", by="station_id")
    past = rows.event_ts.between(args.train_start, args.train_end, inclusive="left")
    future = rows.event_ts.between(args.train_end, args.test_end, inclusive="left")
    X = rows.drop(columns=[*KEYS, LABEL])
    return X[past], X[future], rows[LABEL][past], rows[LABEL][future]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", type=Path, metavar="DIR", help="train from data/prepare_air_quality.py parquet, skip the registry")
    ap.add_argument("--model", type=Path, default=Path("model"), help="where model.pkl is written")
    ap.add_argument("--train-start", default="2024-10-01")
    ap.add_argument("--train-end", default="2025-04-01", help="first day of the test period")
    ap.add_argument("--test-end", default="2025-10-01", help="first day after the test period")
    ap.add_argument("--fg-version", type=int, default=1)
    ap.add_argument("--fv-version", type=int, default=1)
    args = ap.parse_args()

    args.model.mkdir(parents=True, exist_ok=True)
    if args.local:
        model, _ = train(*frames_local(args.local, args), args)
        joblib.dump(model, args.model / "model.pkl")
        return

    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()
    daily = fs.get_feature_group("pm25_daily", version=args.fg_version)
    history = fs.get_feature_group("pm25_rolling", version=args.fg_version)
    weather = fs.get_feature_group("weather_daily", version=args.fg_version)
    query = (
        daily.select([LABEL])
        .join(history.select_except(KEYS), on=["station_id"])
        .join(weather.select_except(KEYS), on=["station_id"])
    )
    fv = fs.get_or_create_feature_view(
        name=NAME, version=args.fv_version, query=query, labels=[LABEL],
        description="The day's PM2.5 at a station from the days before and the day's weather",
    )
    X_train, X_test, y_train, y_test = fv.train_test_split(
        train_start=args.train_start, train_end=args.train_end, test_start=args.train_end, test_end=args.test_end,
        description=f"{NAME}: train {args.train_start} to {args.train_end}, test to {args.test_end}",
    )
    model, metrics = train(X_train, X_test, y_train[LABEL], y_test[LABEL], args)
    joblib.dump(model, args.model / "model.pkl")
    entry = project.get_model_registry().python.create_model(
        name=NAME,
        metrics=metrics,
        feature_view=fv,
        input_example=X_train.head(1),
        description=f"Gradient boosting, trained on {args.train_start} to {args.train_end}, tested to {args.test_end}.",
    )
    entry.save(os.fspath(args.model))
    print(f"registered {NAME} v{entry.version}")


if __name__ == "__main__":
    main()
