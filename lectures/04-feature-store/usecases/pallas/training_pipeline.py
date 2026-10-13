"""PALLAS training pipeline: feature views, two surrogates, the model registry.

Two Easy objectives of the challenge, each one feature view and one model:

    pallas_direct_b    P_max, cN2_max, L_inj, dip_frac, x_of -> E_med_MeV, dE_mad, q_end
                       injected campaign B rows inside the organisers' reference train split
                       Gaussian process per target, fitted on log targets
    pallas_inverse_a   E_med_MeV, dE_mad, q_end, i_peak, n_emit_x, sigma_z, sigma_y, div_rms, x_of
                       -> p_1, a_0, c_N2, injected rows of campaign A's random scan
                       gradient boosting per target

Charge, current, emittance and vertical size enter the inverse model as the log10 features
written by the feature pipeline. Each model is scored on a 20 % holdout of its own rows
(sorted by config, seed 0, so the local and cluster runs split identically), on the scorer's
scales: energy linear, spread as log10 percent, charge as log10 pC. A linear baseline on the
same split is printed beside it.

    # from the parquet folder written by feature_pipeline.py --out (no cluster needed)
    python training_pipeline.py --local pallas_fg --models models

    # from the feature store, register both models with their feature views
    python training_pipeline.py

    # in a project of your own: read the feature groups from the shared store, keep views and models here
    python training_pipeline.py --store m4cast_pallas_featurestore
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
TEST_SIZE = 0.2

# the scorer's scales (pallas_score.py): multiply, floor, then log10
SCALES = {"de_mad": (100.0, 1e-3), "q_end": (1e12, 0.1)}


def gaussian_process(n_inputs: int):
    kernel = ConstantKernel() * RBF(np.ones(n_inputs)) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED)
    # every target is positive and spans a decade or more: fit in log, predict in physical units
    return TransformedTargetRegressor(
        make_pipeline(StandardScaler(), MultiOutputRegressor(gp)), func=np.log, inverse_func=np.exp
    )


def boosting(n_inputs: int):
    return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED))


OBJECTIVES = {
    "pallas_direct_b": dict(
        fg="pallas_campaign_b",
        features=["p_max", "cn2_max", "l_inj", "dip_frac", "x_of"],
        labels=["e_med_mev", "de_mad", "q_end"],
        model=gaussian_process,
        description="Direct B surrogate: density profile knobs -> E_med_MeV, dE_mad, q_end. GP per target, log targets.",
    ),
    "pallas_inverse_a": dict(
        fg="pallas_campaign_a",
        features=["e_med_mev", "de_mad", "log10_q_pc", "log10_i_peak", "log10_n_emit_x",
                  "sigma_z", "log10_sigma_y", "div_rms", "x_of"],
        labels=["p_1", "a_0", "c_n2"],
        model=boosting,
        description="Inverse A: beam + x_of -> p_1, a_0, c_N2. Gradient boosting per target.",
    ),
}


def rows_filter(name: str, fg):
    """The row selection of each view, as a feature store filter."""
    if name == "pallas_direct_b":
        return (fg.injected == True) & (fg.in_reference_train == True)  # noqa: E712
    return (fg.scan == "random") & (fg.injected == True)  # noqa: E712


def rows_local(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """The same selection on the parquet written by feature_pipeline.py --out."""
    if name == "pallas_direct_b":
        return df[df.injected & df.in_reference_train]
    return df[(df.scan == "random") & df.injected]


def scored(col: str, values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if col not in SCALES:
        return values
    multiply, floor = SCALES[col]
    return np.log10(np.maximum(values * multiply, floor))


def r2_per_label(labels: list[str], y_true: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    return {c: float(r2_score(scored(c, y_true[c]), scored(c, y_pred[:, i]))) for i, c in enumerate(labels)}


def train(name: str, spec: dict, frame: pd.DataFrame):
    frame = frame.dropna(subset=spec["features"] + spec["labels"]).sort_values("config")
    X, y = frame[spec["features"]], frame[spec["labels"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    model = spec["model"](len(spec["features"])).fit(X_train.to_numpy(), y_train.to_numpy())
    r2 = r2_per_label(spec["labels"], y_test, model.predict(X_test.to_numpy()))
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train.to_numpy(), scored_frame(y_train))
    base = {c: float(r2_score(scored(c, y_test[c]), linear.predict(X_test.to_numpy())[:, i]))
            for i, c in enumerate(spec["labels"])}

    print(f"\n{name}: trained on {len(X_train)}, held out {len(X_test)}")
    print(pd.DataFrame({"model": r2, "linear baseline": base}).T.round(3).to_string())
    metrics = {f"r2_{c}": round(v, 4) for c, v in r2.items()}
    metrics["r2_mean"] = round(float(np.mean(list(r2.values()))), 4)
    return model, metrics, X_train


def scored_frame(y: pd.DataFrame) -> np.ndarray:
    return np.column_stack([scored(c, y[c]) for c in y.columns])


def save(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.pkl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", type=Path, metavar="DIR", help="train from feature_pipeline.py --out parquet, skip the registry")
    ap.add_argument("--models", type=Path, default=Path("models"), help="where model.pkl files are written")
    ap.add_argument("--fg-version", type=int, default=1)
    ap.add_argument("--fv-version", type=int, default=1)
    ap.add_argument("--only", choices=list(OBJECTIVES), help="train one objective")
    ap.add_argument("--store", metavar="NAME", help="read the feature groups from this shared feature store")
    args = ap.parse_args()

    objectives = {k: v for k, v in OBJECTIVES.items() if args.only in (None, k)}

    if args.local:
        for name, spec in objectives.items():
            frame = rows_local(name, pd.read_parquet(args.local / f"{spec['fg']}.parquet"))
            model, _, _ = train(name, spec, frame)
            save(model, args.models / name)
        return

    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()
    source = project.get_feature_store(name=args.store) if args.store else fs
    mr = project.get_model_registry()
    for name, spec in objectives.items():
        fg = source.get_feature_group(spec["fg"], version=args.fg_version)
        query = fg.select(spec["features"] + spec["labels"]).filter(rows_filter(name, fg))
        fv = fs.get_or_create_feature_view(
            name=name, version=args.fv_version, query=query, labels=spec["labels"], description=spec["description"]
        )
        X, y = fv.training_data(primary_key=True, description=f"{name} training rows")
        # hopsworks 5.1.1 returns the key as <project>_<feature group>_<version>_config
        X = X.rename(columns=lambda c: "config" if c.endswith("_config") else c)
        model, metrics, X_train = train(name, spec, X.join(y))
        save(model, args.models / name)
        entry = mr.python.create_model(
            name=name,
            metrics=metrics,
            feature_view=fv,
            input_example=X_train.head(1),
            description=spec["description"],
        )
        entry.save(os.fspath(args.models / name))
        print(f"registered {name} v{entry.version}")


if __name__ == "__main__":
    main()
