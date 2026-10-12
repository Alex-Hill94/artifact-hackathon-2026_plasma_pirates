"""Air quality feature pipeline: daily PM2.5 readings and weather into three feature groups.

One row is one station on one day. data/prepare_air_quality.py downloads the public
measurements (LCSQA, data.gouv.fr) and the weather (Open-Meteo) and writes two parquet files;
this pipeline computes the features and writes three feature groups, all keyed on station_id
with event_ts as event time, all online so a deployment can look a station up by key:

    pm25_daily      the day's mean PM2.5 in µg/m³: the label, known once the day is over
    pm25_rolling    what was known when the day started: yesterday's value, the mean and the
                    highest value of the 3 and 7 days before
    weather_daily   temperature, wind, rain, humidity, pressure and radiation of the day at the
                    station's city

The rolling features of a day never read that day's own value. pm25_rolling and weather_daily
therefore reach one day further than pm25_daily: the day whose PM2.5 is not measured yet, the
one the deployment predicts. Its label arrives with the next run.

    python ../../data/prepare_air_quality.py --out air
    python feature_pipeline.py --data air            # shapes only
    python feature_pipeline.py --data air --insert
"""

import argparse
import time
from pathlib import Path

import pandas as pd

KEYS = ["station_id", "event_ts"]
WINDOWS = [3, 7]
MIN_DAYS = {3: 2, 7: 4}  # a window with fewer measured days than this gives no value

DESCRIPTIONS = {
    "pm25_daily": "Daily mean PM2.5 per station, µg/m³ (LCSQA regulatory stations, Normandy)",
    "pm25_rolling": "PM2.5 of the days before each day, per station: lag and rolling windows",
    "weather_daily": "Daily weather at the station's city (Open-Meteo archive)",
}


def rolling(readings: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling features on a full calendar per station, so a window is a number of days, not of rows."""
    frames = []
    calendar = pd.date_range(readings.event_ts.min(), readings.event_ts.max() + pd.Timedelta(days=1), freq="D")
    for station_id, rows in readings.groupby("station_id"):
        pm25 = rows.set_index("event_ts").pm25.reindex(calendar).rename_axis("event_ts")
        before = pm25.shift(1)
        table = pd.DataFrame({"pm25_lag1": before})
        for days in WINDOWS:
            window = before.rolling(f"{days}D", min_periods=MIN_DAYS[days])
            table[f"pm25_{days}d_mean"] = window.mean().round(2)
            table[f"pm25_{days}d_max"] = window.max()
        frames.append(table.dropna().reset_index().assign(station_id=station_id))
    return pd.concat(frames, ignore_index=True)[[*KEYS, *table.columns]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="output folder of data/prepare_air_quality.py")
    ap.add_argument("--insert", action="store_true", help="insert into the feature store")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    readings = pd.read_parquet(args.data / "readings.parquet")
    frames = {
        "pm25_daily": readings.drop(columns="valid_hours"),
        "pm25_rolling": rolling(readings),
        "weather_daily": pd.read_parquet(args.data / "weather.parquet"),
    }
    for name, df in frames.items():
        print(f"{name:14s} {len(df):>5} rows x {df.shape[1]:>2} columns")
    if not args.insert:
        return

    import hopsworks

    fs = hopsworks.login().get_feature_store()
    for name, df in frames.items():
        fg = fs.get_or_create_feature_group(
            name=name, version=args.version, primary_key=["station_id"], event_time="event_ts",
            online_enabled=True, description=DESCRIPTIONS[name],
        )
        start = time.monotonic()
        fg.insert(df, wait=True)
        print(f"{name}: inserted {len(df)} rows in {time.monotonic() - start:.0f} s")


if __name__ == "__main__":
    main()
