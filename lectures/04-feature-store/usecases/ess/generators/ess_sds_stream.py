"""Stream synthetic ESS SDS post-mortem events, each derived from a real event of the public sample.

Each emitted event takes the five cycles of a real event, keeps their I and Q waveforms and
metadata, and applies one event-wide perturbation: I scaled by a gain and Q shifted by an
offset, drawn with the spread that the flat-top I level and Q level show from cycle to cycle
inside that same real event. A new capture of the same machine state varies like consecutive
cycles do; setpoint changes between events are not invented. It gets a fresh event_id (epoch milliseconds, far above the
real ids), cycle ids at the real offsets, the current event_ts, synthetic=True and the id of
the real event it comes from. Gaps between events are real gaps between consecutive sample
events, compressed by --speedup.

--hot injects the one invented thing: an amplitude droop on the last cycle of the event, a
linear sag of I across the flat-top of depth --droop. The public sample has no confirmed fault
of this kind; the depth is a parameter, not a measurement. Injected events carry
injected_anomaly=True.

Rows land in the same feature groups as the real sample (ess_sds_events,
ess_waveform_features), features computed by the same code as the feature pipeline.

    # print one event (five cycles) as JSON
    python generators/ess_sds_stream.py --n 1

    # 200 events, 30% with a 2% droop, to parquet
    python generators/ess_sds_stream.py --n 200 --hot 0.3 --out data/synthetic.parquet

    # stream into the feature store, real gaps compressed 3600x (one hour becomes one second)
    python generators/ess_sds_stream.py --insert --follow --hot 0.2
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# the generator shares load_sample, flat_top and build with the feature pipeline one folder up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_pipeline import KEYS, SAMPLE_URL, build, flat_top, load_sample  # noqa: E402


class Replay:
    def __init__(self, source: str, hot: float, droop: float, seed: int | None):
        self.sample = load_sample(source)
        _, feats = build(self.sample)
        by_event = feats.groupby("event_id")
        self.gain_sd = by_event.i_flat_level.std() / by_event.i_flat_level.mean()
        self.offset_sd = by_event.q_flat_mean.std()
        self.events = self.sample.groupby("event_id")
        self.rf_on = feats.groupby(self.sample.event_id).i_max.min().gt(1.0)  # droop needs RF
        ts = self.events.event_ts.first().sort_values()
        self.gaps = ts.diff().dt.total_seconds().dropna().to_numpy()
        self.hot, self.droop = hot, droop
        self.rng = np.random.default_rng(seed)
        self.last_id = 0

    def next(self) -> tuple[pd.DataFrame, float]:
        inject = self.rng.random() < self.hot
        pool = self.rf_on.index[self.rf_on] if inject else self.rf_on.index
        src_id = pool[self.rng.integers(len(pool))]
        src = self.events.get_group(src_id).sort_values("cycle_id")
        gain = self.rng.normal(1.0, self.gain_sd[src_id])
        offset = self.rng.normal(0.0, self.offset_sd[src_id])

        now = datetime.now(timezone.utc)
        event_id = self.last_id = max(int(now.timestamp() * 1000), self.last_id + 1)
        ev = src.assign(
            event_id=event_id,
            cycle_id=event_id + (src.cycle_id - src_id),
            event_ts=pd.Timestamp(now),
            I=[np.asarray(i) * gain for i in src.I],
            Q=[np.asarray(q) + offset for q in src.Q],
            synthetic=True,
            source_event_id=src_id,
            injected_anomaly=False,
        )
        if inject:
            last = ev.index[-1]
            i = ev.at[last, "I"].copy()
            first, end = flat_top(i)
            i[first : end + 1] *= 1.0 - self.droop * np.linspace(0.0, 1.0, end - first + 1)
            ev.at[last, "I"] = i
            ev["injected_anomaly"] = True
        return ev, float(self.rng.choice(self.gaps))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SAMPLE_URL, help="folder or URL holding ess_llrf_sample.parquet")
    ap.add_argument("--n", type=int, default=5, help="events per batch")
    ap.add_argument("--out", type=Path, help="write the batch (waveforms, metadata, features) to this parquet file")
    ap.add_argument("--insert", action="store_true", help="write to the feature store instead of stdout")
    ap.add_argument("--events-fg", default="ess_sds_events")
    ap.add_argument("--features-fg", default="ess_waveform_features")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--follow", action="store_true", help="keep streaming until Ctrl-C")
    ap.add_argument("--speedup", type=float, default=3600.0, help="compress real inter-event gaps by this factor")
    ap.add_argument("--max-wait", type=float, default=10.0, help="cap on seconds between two events")
    ap.add_argument("--hot", type=float, default=0.0, metavar="RATE", help="fraction of events with an injected droop")
    ap.add_argument("--droop", type=float, default=0.02, help="relative sag of I across the flat-top")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.out and args.follow:
        ap.error("--out writes one batch, drop --follow")

    replay = Replay(args.source, args.hot, args.droop, args.seed)

    if args.insert:
        import hopsworks

        fs = hopsworks.login().get_feature_store()
        events_fg = fs.get_feature_group(args.events_fg, version=args.version)
        features_fg = fs.get_feature_group(args.features_fg, version=args.version)
        # online only; the offline copy is materialised on the job schedule, not per batch
        online = {"start_offline_materialization": False}

    while True:
        batch = []
        for _ in range(args.n):
            event, gap = replay.next()
            batch.append(event)
            if args.follow:
                time.sleep(min(gap / args.speedup, args.max_wait))
        captures = pd.concat(batch, ignore_index=True)
        events, features = build(captures)
        if args.insert:
            # features first: inference_pipeline.py polls events, its rows must find their features
            features_fg.insert(features, write_options=online)
            events_fg.insert(events, write_options=online)
            print(f"{datetime.now():%H:%M:%S} {events.event_id.nunique()} events, last {events.event_id.iloc[-1]}", file=sys.stderr)
        elif args.out:
            captures.merge(features, on=[*KEYS, "event_ts"]).to_parquet(args.out, index=False)
            print(f"wrote {args.out}: {events.event_id.nunique()} events, {len(events)} cycles, "
                  f"{events.groupby('event_id').injected_anomaly.first().sum()} injected", file=sys.stderr)
        else:
            print(events.merge(features, on=[*KEYS, "event_ts"]).to_json(orient="records", date_format="iso", indent=1))
        if not args.follow:
            break


if __name__ == "__main__":
    main()
