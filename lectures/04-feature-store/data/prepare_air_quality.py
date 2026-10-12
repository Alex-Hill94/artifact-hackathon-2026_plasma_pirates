"""Turn public air quality and weather data into feature-group-ready parquet files.

Air quality: hourly measurements of the French regulatory stations, one CSV per day for the
whole country, published by LCSQA and Ineris on data.gouv.fr under Licence Ouverte 2.0
("Données temps réel de mesure des concentrations de polluants atmosphériques réglementés").
This script keeps PM2.5 at the stations of the Normandy cities in CITIES.

Weather: daily values from the Open-Meteo archive API (CC BY 4.0), at each city.

Writes to --out (default ./air):
    readings.parquet   one row per station and day: the mean of the valid hourly PM2.5 values,
                       kept when at least MIN_HOURS of the day's hours are valid
    weather.parquet    one row per station and day: temperature, wind, rain, humidity, pressure and
                       radiation at the station's city,
                       up to the day after the last reading (Open-Meteo answers that day with its forecast)

Every table carries station_id (primary key) and event_ts (event time, the day at 00:00 UTC).

    python prepare_air_quality.py --out ./air
    python prepare_air_quality.py --out ./air --start 2026-09-01     # one month, 0.5 GB to read
"""

import argparse
import io
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HOURLY = (
    "https://files.data.gouv.fr/ineris/lcsqa/concentrations-de-polluants-atmospheriques-reglementes"
    "/temps-reel/{day:%Y}/FR_E2_{day:%Y-%m-%d}.csv"
)
GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
CITIES = ["Caen", "Rouen", "Le Havre", "Cherbourg"]
POLLUTANT = "PM2.5"
MIN_HOURS = 18
WEATHER = {
    "temperature_2m_mean": "temperature_c",
    "temperature_2m_min": "temperature_min_c",
    "temperature_2m_max": "temperature_max_c",
    "wind_speed_10m_mean": "wind_speed_kmh",
    "wind_speed_10m_max": "wind_speed_max_kmh",
    "wind_gusts_10m_max": "wind_gusts_kmh",
    "wind_direction_10m_dominant": "wind_direction_deg",
    "precipitation_sum": "precipitation_mm",
    "precipitation_hours": "precipitation_hours",
    "relative_humidity_2m_mean": "humidity_pct",
    "surface_pressure_mean": "pressure_hpa",
    "shortwave_radiation_sum": "radiation_mj_m2",
}


def get_json(url: str, **params) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}") as r:
        return json.load(r)


def hourly(day: pd.Timestamp, cache: Path) -> pd.DataFrame:
    """The day's valid hourly PM2.5 rows of the stations in CITIES. The country file is 12 MB, the kept rows a few kB."""
    path = cache / f"{day:%Y-%m-%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    try:
        with urllib.request.urlopen(HOURLY.format(day=day)) as r:
            raw = pd.read_csv(io.BytesIO(r.read()), sep=";", encoding="utf-8-sig")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        print(f"{day:%Y-%m-%d}: no file published")
        return pd.DataFrame()
    city = raw["nom site"].str.extract(f"^({'|'.join(CITIES)})", expand=False)
    # validité 1 is a valid measurement; -1 is invalid and 4 has no value
    kept = raw[city.notna() & raw["Polluant"].eq(POLLUTANT) & raw["validité"].eq(1)]
    rows = pd.DataFrame({
        "station_id": kept["code site"],
        "station_name": kept["nom site"],
        "city": city[kept.index],
        "influence": kept["type d'influence"],
        "event_ts": day,
        "pm25": kept["valeur"].astype(float),
    })
    rows.to_parquet(path, index=False)
    return rows


def readings_table(days: pd.DatetimeIndex, cache: Path, workers: int) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        frames = list(pool.map(lambda d: hourly(d, cache), days))
    rows = pd.concat(frames, ignore_index=True)
    keys = ["station_id", "station_name", "city", "influence", "event_ts"]
    daily = rows.groupby(keys, as_index=False).agg(pm25=("pm25", "mean"), valid_hours=("pm25", "size"))
    daily = daily[daily.valid_hours >= MIN_HOURS]
    return daily.assign(pm25=daily.pm25.round(2)).sort_values(["station_id", "event_ts"], ignore_index=True)


def weather_table(stations: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for city, ids in stations.groupby("city").station_id:
        place = get_json(GEOCODING, name=city, country_code="FR", count=1)["results"][0]
        daily = get_json(
            ARCHIVE, latitude=place["latitude"], longitude=place["longitude"], timezone="UTC",
            start_date=f"{days[0]:%Y-%m-%d}", end_date=f"{days[-1]:%Y-%m-%d}", daily=",".join(WEATHER),
        )["daily"]
        table = pd.DataFrame(daily).rename(columns=WEATHER | {"time": "event_ts"})
        table["event_ts"] = pd.to_datetime(table.event_ts)
        frames += [table.assign(station_id=station_id) for station_id in ids]
    weather = pd.concat(frames, ignore_index=True).dropna()
    return weather[["station_id", "event_ts", *WEATHER.values()]].sort_values(["station_id", "event_ts"], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("air"))
    ap.add_argument("--cache", type=Path, default=Path(".lcsqa"))
    ap.add_argument("--start", default="2024-10-01", help="first day, the hourly files start on 2021-01-01")
    ap.add_argument("--end", default=f"{pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1):%Y-%m-%d}", help="last day, default yesterday")
    ap.add_argument("--workers", type=int, default=16, help="parallel downloads")
    args = ap.parse_args()

    days = pd.date_range(args.start, args.end, freq="D")
    readings = readings_table(days, args.cache, args.workers)
    weather = weather_table(readings[["station_id", "city"]].drop_duplicates(), days.union([days[-1] + pd.Timedelta(days=1)]))

    args.out.mkdir(parents=True, exist_ok=True)
    readings.to_parquet(args.out / "readings.parquet", index=False)
    weather.to_parquet(args.out / "weather.parquet", index=False)
    print(f"readings.parquet   {readings.shape}, {readings.station_id.nunique()} stations, {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}")
    print(f"weather.parquet    {weather.shape}")
    print(readings.groupby(["station_id", "station_name", "influence"]).pm25.agg(["size", "mean"]).round(1).to_string())


if __name__ == "__main__":
    main()
