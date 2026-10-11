"""Inference pipeline: deploy ess_llrf_anomaly, score live ESS events by key.

Creates the deployment on first run (predictor.py looks up the feature vector of each
(event_id, cycle_id) in the online store and scores it), then polls ess_sds_events online for
synthetic cycles written since the last poll and prints their anomaly scores. Run the
generator with --insert --follow next to it. injected_anomaly is printed as the truth to
check against; the model never sees it.

    # score one cycle
    python inference_pipeline.py --key 16096686207:16096686195

    # follow the live stream
    python inference_pipeline.py --follow
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import hopsworks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ess_llrf_anomaly")
    ap.add_argument("--model-version", type=int, default=1)
    ap.add_argument("--deployment", default="essanomaly")
    ap.add_argument("--events-fg", default="ess_sds_events")
    ap.add_argument("--fg-version", type=int, default=1)
    ap.add_argument("--key", metavar="EVENT_ID:CYCLE_ID", help="score one cycle and exit")
    ap.add_argument("--follow", action="store_true", help="score new synthetic cycles until Ctrl-C")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between two polls")
    args = ap.parse_args()
    if not (args.key or args.follow):
        ap.error("pass --key or --follow")

    project = hopsworks.login()
    deployment = project.get_model_serving().get_deployment(args.deployment)
    if deployment is None:
        entry = project.get_model_registry().get_model(args.model, version=args.model_version)
        deployment = entry.deploy(
            name=args.deployment,
            script_file=str(Path(__file__).with_name("predictor.py")),
            description="IsolationForest on ESS waveform features, online lookup by (event_id, cycle_id)",
        )
    deployment.start(await_running=600)

    if args.key:
        event_id, cycle_id = (int(v) for v in args.key.split(":"))
        print(deployment.predict(inputs=[{"event_id": event_id, "cycle_id": cycle_id}]))
        return

    events = project.get_feature_store().get_feature_group(args.events_fg, version=args.fg_version)
    since = datetime.now(timezone.utc)
    print("event_id       cycle_id       source_event_id  injected  score   flagged")
    while True:
        new = (
            events.select(["event_id", "cycle_id", "event_ts", "source_event_id", "injected_anomaly"])
            .filter((events.synthetic == True) & (events.event_ts > since))  # noqa: E712, builds a filter
            .read(online=True)
            .sort_values(["event_ts", "cycle_id"])
        )
        if len(new):
            since = new.event_ts.max()
            keys = new[["event_id", "cycle_id"]].astype(int).to_dict("records")
            for row, (score, flagged) in zip(new.itertuples(), deployment.predict(inputs=keys)["predictions"]):
                print(f"{row.event_id}  {row.cycle_id}  {row.source_event_id}      "
                      f"{bool(row.injected_anomaly)!s:8}  {score:.3f}   {'ANOMALY' if flagged else ''}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
