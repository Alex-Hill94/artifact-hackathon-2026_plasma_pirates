# Air quality forecast

Feature, training and inference pipelines on public data: the day's PM2.5 at a Normandy air quality station, from the days before and the day's weather. The lecture's concept slides use this case.

## Data

`data/prepare_air_quality.py` downloads both sources and writes two parquet files.

- PM2.5: hourly measurements of the French regulatory stations (LCSQA and Ineris, data.gouv.fr, Licence Ouverte 2.0). Seven stations in Caen, Rouen, Le Havre and Cherbourg, from 1 October 2024. A day is kept when 18 of its hours are valid: 4,872 station-days.
- Weather: daily values at each city from the Open-Meteo archive (CC BY 4.0), up to the day after the last reading.
- Key: `station_id`. Event time: `event_ts`, the day at 00:00 UTC.

The default period reads 741 country files of 12 MB each and takes about 25 minutes. `--start 2026-09-01` reads one month.

## Run

In a project you own. The first two lines run anywhere, the insert and the deployment need the cluster:

```bash
cd usecases/air-quality
python ../../data/prepare_air_quality.py --out air
python training_pipeline.py --local air             # no cluster: train from the parquet files

hops files upload air/readings.parquet /Projects/<project>/Resources/air/
hops files upload air/weather.parquet /Projects/<project>/Resources/air/
hops job deploy air_quality_features feature_pipeline.py --args "--data /hopsfs/Resources/air --insert" --run --wait

python training_pipeline.py                                                # v1: trained on the first winter
python training_pipeline.py --train-end 2025-10-01 --test-end 2026-04-01   # v2: trained on a full year

hops files upload predictor.py /Projects/<project>/Resources/jobs/air_quality_inference/
hops job deploy air_quality_inference inference_pipeline.py --run --wait
hops deployment stop pm25forecast
```

From the cluster terminal the two jobs are `python feature_pipeline.py --data air --insert` and `python inference_pipeline.py`.

## What you get

| Object | Name |
|---|---|
| Feature groups, online | `pm25_daily` (the label), `pm25_rolling` (yesterday, 3 and 7 day mean and max), `weather_daily` (12 weather values) |
| Feature view | `pm25_forecast`: the label joined to both groups on `station_id`, as of the label's day |
| Model | `pm25_forecast`: gradient boosting, split on time |
| Deployment | `pm25forecast`: send station ids, get the day's predicted PM2.5 in µg/m³ |

## Results

Mean absolute error in µg/m³ on the test period, against persistence (today equals yesterday):

| Model | Trained on | Tested on | Model | Persistence |
|---|---|---|---|---|
| v1 | Oct 2024 to Mar 2025 | Apr to Sep 2025 | 2.49 | 2.52 |
| v2 | Oct 2024 to Sep 2025 | Oct 2025 to Mar 2026 | 3.06 | 3.41 |

```
$ python inference_pipeline.py
pm25forecast v2: 6 stations predicted by key in 19 ms
                              station         day  yesterday  predicted  measured
station_id
FR05089     Le Havre Lafaurie-montant  2026-10-12       3.55       4.39   not yet
FR21001              Caen Chemin-Vert  2026-10-12       3.12       4.30   not yet
FR25053     Rouen Proxi Quai de Paris  2026-10-12       7.03       6.84   not yet
```

The day predicted has no measurement yet. After the next `prepare_air_quality.py` and feature pipeline run, the same command prints the measured value next to it.

## Limits

- v1 saw one winter and does not beat persistence in summer. v2 does, by 10%.
- The weather of a past day is the archive's value, the weather of the predicted day is a forecast.
- `FR05074` stopped reporting in June 2026. The inference pipeline predicts the stations that have a row on the latest day.
