"""Inference pipeline: deploy ess_llrf_anomaly, score cycles by key.

Creates the deployment on first run, reuses it afterwards, and replaces it when it serves another
model version than the one asked for. A request carries keys only:
predictor.py looks up the 14 features of each (event_id, cycle_id, pv_base) in the online store
and returns the anomaly score and the flag. Run it in the terminal of your own project: the
serving gateway is not reachable from outside the cluster.

    # one cycle of one chain
    python inference_pipeline.py --key 16122602164 16122596526 HBL-050RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData

    # every cycle and chain of one event
    python inference_pipeline.py --event 16122602164
"""

import argparse
import os
import time
from pathlib import Path

import hopsworks

from feature_pipeline import KEYS

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ess_llrf_anomaly")
    ap.add_argument("--model-version", type=int, help="default: the latest version")
    ap.add_argument("--deployment", default="essanomaly")
    ap.add_argument("--features-fg", default="ess_waveform_features")
    ap.add_argument("--fg-version", type=int, default=1)
    ap.add_argument("--key", nargs=3, metavar=("EVENT_ID", "CYCLE_ID", "PV_BASE"), help="score one cycle of one chain")
    ap.add_argument("--event", type=int, metavar="EVENT_ID", help="score every cycle and chain of this event")
    args = ap.parse_args()
    if not (args.key or args.event):
        ap.error("pass --key or --event")

    project = hopsworks.login()
    mr = project.get_model_registry()
    # get_model without a version returns version 1, not the latest
    version = args.model_version or max(m.version for m in mr.get_models(args.model))
    deployment = project.get_model_serving().get_deployment(args.deployment)
    if deployment is not None and deployment.model_version != version:
        # A deployment keeps the model version it was created with.
        print(f"deployment {args.deployment} serves model version {deployment.model_version}, replacing it with {version}")
        deployment.stop()
        deployment.delete()
        deployment = None
    if deployment is None:
        script = project.get_dataset_api().upload(os.fspath(HERE / "predictor.py"), "Resources", overwrite=True)
        deployment = mr.get_model(args.model, version=version).deploy(
            name=args.deployment,
            script_file=os.path.join("/Projects", project.name, script),
            description="IsolationForest on ESS waveform features, online lookup by (event_id, cycle_id, pv_base)",
        )
    start = time.time()
    deployment.start(await_running=600)
    print(f"deployment {args.deployment} running after {time.time() - start:.0f} s, model version {deployment.model_version}")

    if args.key:
        event_id, cycle_id, pv_base = args.key
        print(deployment.predict(inputs=[{"event_id": int(event_id), "cycle_id": int(cycle_id), "pv_base": pv_base}]))
        return

    features = project.get_feature_store().get_feature_group(args.features_fg, version=args.fg_version)
    keys = features.select(KEYS).filter(features.event_id == args.event).read().sort_values(KEYS)
    if keys.empty:
        raise SystemExit(f"no row of {args.features_fg} has event_id {args.event}")
    start = time.time()
    predictions = deployment.predict(inputs=keys.astype({"event_id": int, "cycle_id": int}).to_dict("records"))["predictions"]
    keys["score"], keys["flagged"] = zip(*predictions)
    print(f"event {args.event}: {len(keys)} rows scored by key in {time.time() - start:.1f} s, {int(keys.flagged.sum())} flagged")
    shown = keys[keys.flagged] if keys.flagged.any() else keys.nlargest(5, "score")
    print(shown.sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
