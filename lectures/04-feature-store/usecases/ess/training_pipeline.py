"""Training pipeline: two models on the real ESS data, each with its feature view.

    ess_llrf_anomaly      StandardScaler + IsolationForest on the 14 features of ess_waveform_features
                          (your project). No confirmed fault labels exist, so the model learns the
                          envelope of the cycles and scores distance from it. Served by key.
    ess_llrf_autoencoder  the challenge's reference autoencoder on q, trained on every row of
                          ess_llrf_waveforms outside the holdout instead of the 168 rows of the public
                          sample, and scored with the organisers' score.py. Needs torch.

The 42 rows the organisers score on carry in_sample_holdout in ess_llrf_waveforms. Neither model
trains on them. --start and --end cut the training rows of the anomaly model on cycle_ts; it then
scores every row, inside the window or not.

    # both models, from the feature store, registered with their feature views
    python training_pipeline.py

    # one of them
    python training_pipeline.py --only ess_llrf_anomaly

    # the anomaly model on the May 2025 campaign only
    python training_pipeline.py --only ess_llrf_anomaly --end 2026-01-01
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_pipeline import EVENT_TIME, KEYS, RF_ON, SOURCE_FG, STORE

CHALLENGE = Path(__file__).resolve().parents[4] / "challenges" / "anomaly-detection"
ANOMALY, AUTOENCODER = "ess_llrf_anomaly", "ess_llrf_autoencoder"
SEED = 42
CUT = 0.5  # a flat-top under half its event's median length: the pulse was cut


def with_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """hopsworks 5.1.1 returns keys and event time as <project>_<feature group>_<version>_<name>: give them their names back."""
    return frame.rename(columns=lambda c: next((k for k in [*KEYS, EVENT_TIME] if c.endswith("_" + k)), c))


def train_anomaly(args, project, fs, waves) -> None:
    features = fs.get_feature_group(args.features_fg, version=args.fg_version)
    fv = fs.get_or_create_feature_view(
        name=ANOMALY, version=args.fv_version,
        query=features.select_except([*KEYS, EVENT_TIME]),
        description="Per-cycle ESS LLRF waveform features for unsupervised anomaly scoring",
    )
    holdout_keys = pd.MultiIndex.from_frame(
        waves.select(KEYS).filter(waves.in_sample_holdout == True).read().drop_duplicates()  # noqa: E712
    )
    X, _ = fv.training_data(start_time=args.start, end_time=args.end, primary_key=True)
    X = with_keys(X)
    X_train = X[~X.set_index(KEYS).index.isin(holdout_keys)].drop(columns=KEYS)

    start = time.time()
    model = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=300, contamination=args.contamination, random_state=SEED, n_jobs=-1),
    ).fit(X_train)

    # score every row the view holds, as a batch job would: the holdout and anything outside the window
    rows = with_keys(fv.get_batch_data(primary_key=True, event_time=True))
    names = list(X_train.columns)
    # score_samples is higher for normal points; flip it so higher means more anomalous
    rows["score"] = -model.score_samples(rows[names])
    rows["flagged"] = model.predict(rows[names]) == -1
    print(f"\n{ANOMALY}: trained on {len(X_train)} rows, {len(names)} features, {time.time() - start:.0f} s")

    chain = rows.groupby("pv_base")
    event_chain = rows.groupby(["event_id", "pv_base"])
    in_holdout = rows.set_index(KEYS).index.isin(holdout_keys)
    rf_off = rows.i_max < RF_ON * chain.i_max.transform("median")
    groups = {
        "all rows": np.ones(len(rows), dtype=bool),
        "holdout": in_holdout,
        "RF off": rf_off.to_numpy(),
        "pulse cut short": (~rf_off & (rows.i_flat_len < CUT * event_chain.i_flat_len.transform("median"))).to_numpy(),
    }
    if args.start or args.end:
        ts = rows[EVENT_TIME].dt.tz_localize(None)
        inside = (ts >= pd.Timestamp(args.start or ts.min())) & (ts < pd.Timestamp(args.end or ts.max() + pd.Timedelta(1)))
        groups["outside the training window"] = (~inside).to_numpy()
    report = pd.DataFrame(
        {"rows": {k: int(m.sum()) for k, m in groups.items()},
         "flagged": {k: int(rows.flagged[m].sum()) for k, m in groups.items()}}
    )
    print(report.to_string())
    print("top rows:\n" + rows.nlargest(5, "score")[[*KEYS, "score"]].to_string(index=False))

    metrics = {
        "flag_rate": float(rows.flagged.mean()),
        "holdout_flag_rate": float(rows.flagged[in_holdout].mean()),
        "holdout_mean_score": float(rows.score[in_holdout].mean()),
        "cut_pulses_flagged": float(report.flagged["pulse cut short"]),
    }
    folder = args.models / ANOMALY
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.pkl")
    entry = project.get_model_registry().python.create_model(
        name=ANOMALY, metrics=metrics, feature_view=fv, input_example=X_train.head(1),
        description="StandardScaler + IsolationForest on ESS waveform features, holdout rows left out",
    )
    entry.save(str(folder))
    print(f"registered {ANOMALY} v{entry.version}")


def train_autoencoder(args, project, fs, waves) -> None:
    try:
        import torch
    except ImportError:
        raise SystemExit(f"{AUTOENCODER} needs torch: uv pip install torch, or pass --only {ANOMALY}")
    from torch.utils.data import DataLoader, TensorDataset

    # the organisers' model and scorer, from the challenge folder of this repository
    sys.path.insert(0, str(CHALLENGE))
    from score import RFAutoencoder, score_holdout

    fv = fs.get_or_create_feature_view(
        name=AUTOENCODER, version=args.fv_version,
        query=waves.select(["q", "in_sample", "in_sample_holdout"]),
        description="q waveforms of ess_llrf_waveforms with the public sample's split, for the challenge autoencoder",
    )
    rows, _ = fv.training_data(primary_key=True)
    # the archive holds some captures more than once; the copies are identical
    rows = with_keys(rows).drop_duplicates(KEYS).reset_index(drop=True)
    q = np.stack(rows.q.to_numpy()).astype(np.float32)
    holdout = rows.in_sample_holdout.to_numpy()
    train = ~holdout
    if args.sample_chains:
        train &= rows.pv_base.isin(rows.pv_base[rows.in_sample].unique()).to_numpy()
    if args.strict:
        train &= ~rows.event_id.isin(rows.event_id[holdout].unique()).to_numpy()

    # normalised as the reference bar is, with the public sample's training rows, so the two numbers compare
    reference = (rows.in_sample & ~rows.in_sample_holdout).to_numpy()
    mu, sigma = float(q[reference].mean()), float(q[reference].std() + 1e-8)
    q_n = (q - mu) / sigma

    torch.manual_seed(SEED)
    model = RFAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(q_n[train][:, None, :])), batch_size=args.batch_size, shuffle=True)
    start = time.time()
    model.train()
    for _ in range(args.epochs):
        for (batch,) in loader:
            loss = torch.nn.functional.mse_loss(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    mse, per_row = score_holdout(model, q_n, np.flatnonzero(holdout))
    own_sigma = float(q[train].std())
    print(f"\n{AUTOENCODER}: trained on {int(train.sum())} rows, {args.epochs} epochs, {time.time() - start:.0f} s")
    print(f"holdout mean reconstruction MSE {mse:.4f}, median {np.median(per_row):.4f}, {int(holdout.sum())} rows")
    print(f"on the scale of the bar (sigma {sigma:.2f}); {mse * (sigma / own_sigma) ** 2:.4f} with the training rows' own sigma {own_sigma:.2f}")

    folder = args.models / AUTOENCODER
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "model.pt")
    (folder / "normalisation.json").write_text(json.dumps({"mu": mu, "sigma": sigma}))
    entry = project.get_model_registry().torch.create_model(
        name=AUTOENCODER, metrics={"holdout_mse": mse, "holdout_median_mse": float(np.median(per_row))}, feature_view=fv,
        description=f"Challenge RFAutoencoder on q, {int(train.sum())} training rows, scored with score.py on the holdout",
    )
    entry.save(str(folder))
    print(f"registered {AUTOENCODER} v{entry.version}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=[ANOMALY, AUTOENCODER], help="train one model")
    ap.add_argument("--store", default=STORE, help="feature store that holds ess_llrf_waveforms")
    ap.add_argument("--features-fg", default="ess_waveform_features")
    ap.add_argument("--fg-version", type=int, default=1)
    ap.add_argument("--fv-version", type=int, default=1)
    ap.add_argument("--models", type=Path, default=Path("models"), help="where the model files are written")
    ap.add_argument("--start", help="anomaly model: train on cycles from this date")
    ap.add_argument("--end", help="anomaly model: train on cycles before this date")
    ap.add_argument("--contamination", type=float, default=0.01, help="anomaly model: share of training rows flagged")
    ap.add_argument("--epochs", type=int, default=5, help="autoencoder: passes over the training rows")
    ap.add_argument("--batch-size", type=int, default=32, help="autoencoder")
    ap.add_argument("--sample-chains", action="store_true", help="autoencoder: train on the chains of the public sample only")
    ap.add_argument("--strict", action="store_true", help="autoencoder: also leave out every cycle of an event that holds a holdout row")
    args = ap.parse_args()

    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()
    waves = project.get_feature_store(name=args.store).get_feature_group(SOURCE_FG, version=1)
    if args.only in (None, ANOMALY):
        train_anomaly(args, project, fs, waves)
    if args.only in (None, AUTOENCODER):
        train_autoencoder(args, project, fs, waves)


if __name__ == "__main__":
    main()
