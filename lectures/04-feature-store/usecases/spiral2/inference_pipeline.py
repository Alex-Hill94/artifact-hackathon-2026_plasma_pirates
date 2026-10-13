"""SPIRAL2 inference pipeline: deploy llrf_fault, score events by key.

Creates the deployment on first run and reuses it afterwards. A request carries event ids
only: predictor.py looks up the features of each event in the online store and returns the
probability of a fault. Run it in the cluster terminal of your project: the serving gateway
is not reachable from outside the cluster.

    # one or more events
    python inference_pipeline.py --event cma07_cav1_20250904T151333

    # every event of one cavity, compared with its labels
    python inference_pipeline.py --cavity CMA07_CAV1

Events written by generators/llrf_stream.py --insert are scored the same way, by the
event_id the generator prints.
"""

import argparse
import os
import time
from pathlib import Path

import hopsworks
import pandas as pd

MODEL = "llrf_fault"
DEPLOYMENT = "llrffault"
HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", nargs="+", metavar="EVENT_ID", help="score these events")
    ap.add_argument("--cavity", metavar="CAVITY_ID", help="score every event of this cavity")
    ap.add_argument("--model-version", type=int, help="default: the latest version")
    ap.add_argument("--fg-version", type=int, default=1)
    args = ap.parse_args()
    if not (args.event or args.cavity):
        ap.error("pass --event or --cavity")

    project = hopsworks.login()
    deployment = project.get_model_serving().get_deployment(DEPLOYMENT)
    if deployment is None:
        mr = project.get_model_registry()
        # get_model without a version returns version 1, not the latest
        version = args.model_version or max(m.version for m in mr.get_models(MODEL))
        script = project.get_dataset_api().upload(os.fspath(HERE / "predictor.py"), "Resources", overwrite=True)
        deployment = mr.get_model(MODEL, version=version).deploy(
            name=DEPLOYMENT,
            script_file=os.path.join("/Projects", project.name, script),
            description="SPIRAL2 LLRF fault vs normal, online lookup by event_id",
        )
    deployment.start(await_running=600)

    if args.event:
        scores = deployment.predict(inputs=[{"event_id": e} for e in args.event])["predictions"]
        print(pd.Series(scores, index=args.event, name="p_fault").round(3).to_string())
        return

    labels = project.get_feature_store().get_feature_group("llrf_labels", version=args.fg_version)
    rows = labels.select(["event_id", "is_fault"]).filter(labels.cavity_id == args.cavity).read()
    if rows.empty:
        raise SystemExit(f"no row of llrf_labels has cavity_id {args.cavity}")
    start = time.time()
    scores = deployment.predict(inputs=[{"event_id": e} for e in rows.event_id])["predictions"]
    rows["predicted"] = (pd.Series(scores, index=rows.index) >= 0.5).astype(int)
    print(f"{args.cavity}: {len(rows)} events scored by key in {time.time() - start:.1f} s")
    print(pd.crosstab(rows.is_fault, rows.predicted).to_string())


if __name__ == "__main__":
    main()
