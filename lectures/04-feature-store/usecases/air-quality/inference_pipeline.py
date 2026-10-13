"""Air quality inference pipeline: deploy pm25_forecast, predict stations by key.

Creates the deployment on first run and replaces it when it serves another model version.
A request carries station ids only: predictor.py looks up each station's latest row of
pm25_rolling and weather_daily in the online store and returns the PM2.5 predicted for that
day. Run it in the cluster terminal of your project, or as a job: the serving gateway is not
reachable from outside the cluster.

    # every station
    python inference_pipeline.py

    # some stations
    python inference_pipeline.py --station FR21001 FR25053

The day predicted is the latest one in pm25_rolling, a day pm25_daily has no value for yet.
Once the next feature pipeline run has written that value, the last line compares the two.
"""

import argparse
import os
import time
from pathlib import Path

import hopsworks
import pandas as pd

MODEL = "pm25_forecast"
DEPLOYMENT = "pm25forecast"


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
        deployment = entry.deploy(
            name=DEPLOYMENT,
            script_file=os.path.join("/Projects", project.name, script),
            description="PM2.5 of the day per station, online lookup by station_id",
        )
    deployment.start(await_running=600)
    return deployment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", nargs="+", metavar="STATION_ID", help="default: every station")
    ap.add_argument("--model-version", type=int, help="default: the latest version")
    ap.add_argument("--fg-version", type=int, default=1)
    args = ap.parse_args()

    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    # get_model without a version returns version 1, not the latest
    version = args.model_version or max(m.version for m in mr.get_models(MODEL))
    deployment = deployed(project, mr.get_model(MODEL, version=version))

    daily = fs.get_feature_group("pm25_daily", version=args.fg_version).read()
    history = fs.get_feature_group("pm25_rolling", version=args.fg_version).read()
    # a station that stopped reporting has no row on the latest day, and the online store would answer with its last one
    latest = history[history.event_ts == history.event_ts.max()].set_index("station_id")
    stations = args.station or sorted(latest.index)

    start = time.time()
    predicted = deployment.predict(inputs=[{"station_id": s} for s in stations])["predictions"]
    print(f"{DEPLOYMENT} v{version}: {len(stations)} stations predicted by key in {(time.time() - start) * 1000:.0f} ms")

    names = daily.drop_duplicates("station_id").set_index("station_id").station_name
    measured = daily.set_index(["station_id", "event_ts"]).pm25
    table = pd.DataFrame({
        "station": names[stations],
        "day": latest.event_ts[stations].dt.strftime("%Y-%m-%d"),
        "yesterday": latest.pm25_lag1[stations],
        "predicted": predicted,
        "measured": [measured.get((s, latest.event_ts[s]), float("nan")) for s in stations],
    })
    print(table.to_string(na_rep="not yet"))


if __name__ == "__main__":
    main()
