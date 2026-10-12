"""Your own project: read a real feature group, add a feature, train, register, deploy, predict.

Run it in the terminal of a project you own, after the organisers shared the real feature stores
into it (track 03). It reads PALLAS campaign A from the shared store, writes one derived feature
group into your project, joins both in a feature view, and serves a small model on it.

    python tracks/own_project.py              # everything
    python tracks/own_project.py --no-deploy  # stop after the model registry
"""

import argparse
import os
import time
from pathlib import Path

import hopsworks
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

PREDICTOR = Path(__file__).resolve().parent.parent / "usecases" / "pallas" / "predictor.py"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-deploy", action="store_true", help="register the model and stop")
    args = ap.parse_args()

    project = hopsworks.login()
    fs = project.get_feature_store()
    real = project.get_feature_store(name="m4cast_pallas_featurestore")

    a = real.get_feature_group("pallas_campaign_a", version=1)
    df = a.read()
    print("read", df.shape)

    mine = df[df.injected][["config", "e_med_mev", "q_pc"]].copy()
    mine["charge_per_mev"] = mine.q_pc / mine.e_med_mev
    fg = fs.get_or_create_feature_group(
        name="charge_per_mev",
        version=1,
        primary_key=["config"],
        description="Bunch charge per MeV of median energy, injected rows of campaign A",
    )
    start = time.time()
    fg.insert(mine[["config", "charge_per_mev"]], wait=True)
    print("inserted", len(mine), "rows in", round(time.time() - start), "s")

    query = (
        a.select(["e_med_mev", "de_mad", "x_of", "p_1"])
        .filter(a.injected == True)  # noqa: E712
        .join(fg.select(["charge_per_mev"]), on=["config"])
    )
    fv = fs.get_or_create_feature_view(name="p1_from_beam", version=1, query=query, labels=["p_1"])
    X, y = fv.training_data()
    print("training data", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = HistGradientBoostingRegressor(max_iter=300).fit(X_train, y_train.values.ravel())
    r2 = float(r2_score(y_test, model.predict(X_test)))
    print("r2", round(r2, 3), "features", list(X.columns))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    entry = project.get_model_registry().python.create_model(
        name="p1_from_beam", metrics={"r2": r2}, feature_view=fv, input_example=X_train.head(1)
    )
    entry.save("model")
    if args.no_deploy:
        return

    start = time.time()
    deployment = project.get_model_serving().get_deployment("p1frombeam")
    if deployment is None:
        script = project.get_dataset_api().upload(os.fspath(PREDICTOR), "Resources", overwrite=True)
        deployment = entry.deploy(name="p1frombeam", script_file=os.path.join("/Projects", project.name, script))
    deployment.start(await_running=600)
    print("deployment running after", round(time.time() - start), "s")
    print("predict", deployment.predict(inputs=X_test.head(1).values.tolist()), "true", float(y_test.values.ravel()[0]))


if __name__ == "__main__":
    main()
