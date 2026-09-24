"""Replay SPIRAL2 LLRF events as a live stream, resampled from the real release.

Each emitted event copies the header and physics features of a real event from the same
cavity and class, gets a fresh event_id and event_ts, and carries synthetic=True plus the
id of the real event it copies. Nothing is invented: timing, cavity activity and class mix
come from the 2019-2025 data.

Events land in the same feature groups as the historical data (llrf_events, llrf_physics),
so the feature view built on them serves live events with no extra plumbing. Labels are
not emitted: in operation the fault category arrives later, from the post-mortem analysis.

    # print 3 events as JSON
    python llrf_stream.py --data ../data/llrf --n 3

    # stream into the feature store, real gaps compressed 3600x (one hour becomes one second)
    python llrf_stream.py --data ../data/llrf --insert --follow

    # make one cavity misbehave: 5x its real fault rate
    python llrf_stream.py --data ../data/llrf --insert --follow --hot CMA07_CAV1
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class Replay:
    def __init__(self, data: Path, hot: str | None, hot_factor: float, seed: int | None):
        events = pd.read_parquet(data / "events.parquet")
        physics = pd.read_parquet(data / "physics.parquet").drop(columns=["cavity_id", "event_ts"])
        labels = pd.read_parquet(data / "labels.parquet")[["event_id", "is_fault"]]
        self.event_cols = list(events.columns)
        self.physics_cols = ["event_id", "cavity_id", "event_ts", *physics.columns.drop("event_id")]
        self.rows = events.merge(physics, on="event_id").merge(labels, on="event_id")
        self.rng = np.random.default_rng(seed)

        by_cavity = self.rows.sort_values("event_ts").groupby("cavity_id")
        self.activity = by_cavity.size() / len(self.rows)
        # real inter-event gaps per cavity, in seconds
        self.gaps = {c: g.event_ts.diff().dt.total_seconds().dropna().to_numpy() for c, g in by_cavity}
        self.fault_rate = by_cavity.is_fault.mean()
        if hot:
            if hot not in self.fault_rate:
                raise SystemExit(f"unknown cavity {hot}; known: {', '.join(self.fault_rate.index)}")
            self.fault_rate[hot] = min(1.0, self.fault_rate[hot] * hot_factor)

    def next(self) -> tuple[dict, float]:
        cavity = self.rng.choice(self.activity.index, p=self.activity.to_numpy())
        fault = int(self.rng.random() < self.fault_rate[cavity])
        pool = self.rows[(self.rows.cavity_id == cavity) & (self.rows.is_fault == fault)]
        if pool.empty:
            pool = self.rows[self.rows.cavity_id == cavity]
        src = pool.iloc[self.rng.integers(len(pool))]
        gaps = self.gaps[cavity]
        gap = float(self.rng.choice(gaps)) if len(gaps) else 3600.0
        now = datetime.now(timezone.utc).replace(microsecond=0)
        event = src.drop("is_fault").to_dict()
        event.update(
            event_id=f"syn_{cavity.lower()}_{now:%Y%m%dT%H%M%S}_{self.rng.integers(1_000_000):06d}",
            source_event_id=src.event_id,
            event_ts=now,
            synthetic=True,
        )
        return event, gap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="output folder of data/prepare_llrf.py")
    ap.add_argument("--n", type=int, default=10, help="events per batch")
    ap.add_argument("--insert", action="store_true", help="write to the feature store instead of stdout")
    ap.add_argument("--events-fg", default="llrf_events")
    ap.add_argument("--physics-fg", default="llrf_physics")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--follow", action="store_true", help="keep streaming until Ctrl-C")
    ap.add_argument("--speedup", type=float, default=3600.0, help="compress real inter-event gaps by this factor")
    ap.add_argument("--max-wait", type=float, default=10.0, help="cap on seconds between two events")
    ap.add_argument("--hot", metavar="CAVITY", help="raise the fault rate of one cavity")
    ap.add_argument("--hot-factor", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    replay = Replay(args.data, args.hot, args.hot_factor, args.seed)

    if args.insert:
        import hopsworks

        fs = hopsworks.login().get_feature_store()
        events_fg = fs.get_feature_group(args.events_fg, version=args.version)
        physics_fg = fs.get_feature_group(args.physics_fg, version=args.version)
        # online only; the offline copy is materialised on the job schedule, not per batch
        online = {"start_offline_materialization": False}

    while True:
        batch = []
        for _ in range(args.n):
            event, gap = replay.next()
            batch.append(event)
            if args.follow:
                time.sleep(min(gap / args.speedup, args.max_wait))
        df = pd.DataFrame(batch)
        if args.insert:
            events_fg.insert(df[replay.event_cols], write_options=online)
            physics_fg.insert(df[replay.physics_cols], write_options=online)
            print(f"{datetime.now():%H:%M:%S} {len(df)} events, last {df.event_id.iloc[-1]}", file=sys.stderr)
        else:
            print(df.to_json(orient="records", date_format="iso", indent=1))
        if not args.follow:
            break


if __name__ == "__main__":
    main()
