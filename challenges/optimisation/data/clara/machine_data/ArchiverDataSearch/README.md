# CLARA Archiver Image Search

Search and filter the CLARA image archive dataset by time, camera and PV value, over an [HSDS](https://github.com/HDFGroup/hsds) backend.

The archive is 30k+ HDF5 *metadata* files, one per capture. Each holds 182 scalar PV readings plus the camera name and the path of the capture file that holds the images. 
This project transposes those files into a single column-per-PV HSDS domain and puts a query API and a web UI in front of it.

**Images are referenced, not stored.** The capture files live on the CLARA storage; the index only records where each one sits *relative to the acquisition host's archive root*, plus the image count. The datasets inside are named `Capture000001` … `Capture{n_images:06d}`:

```python
import h5py

with h5py.File(f"{image_root}/{capture_path}") as f:
    image = f["Capture000001"][:]   # greyscale
```

Set `ARCHIVER_IMAGE_ROOT` to wherever this host has that archive mounted (say `/mnt/images/clara`) and the service will render `Capture000001` into the details panel. Leave it unset — the normal case for an index served away from the image store — and the UI simply shows the PV values.

## Quick start

```bash
bash scripts/bootstrap.sh                            # create .venv, install, write .hscfg
bash scripts/hsds_start.sh                           # HSDS on http://localhost:5101
.venv/bin/python -m archiver_search.ingest sample    # build the index
.venv/bin/python -m archiver_search                  # UI + API on http://localhost:8000
```

`scripts/hsds_stop.sh` shuts the server down again. HSDS state stays inside the
checkout: object data in `hsds_data/`, logs and the password file in `hsds_run/`.

Copy `.env.example` to `.env` to change the domain, endpoint, credentials, cache
sizes or limits; every setting is also an `ARCHIVER_*` environment variable.

### Without a server

For a quick look, or for CI, the same code can run against a plain local HDF5
file — `h5pyd` and `h5py` present the same interface:

```bash
ARCHIVER_BACKEND=local ARCHIVER_DOMAIN=./archiver.h5 \
  .venv/bin/python -m archiver_search.ingest sample
ARCHIVER_BACKEND=local ARCHIVER_DOMAIN=./archiver.h5 \
  .venv/bin/python -m archiver_search
```

This is how the test suite runs, so the query engine is covered without a
running HSDS.

## Ingest

```
python -m archiver_search.ingest <dir> [--domain D] [--append] [--workers N] [-v]
```

- Walks `<dir>` for `*.hdf` and takes the timestamp and camera from the
  **filename**, so 30k files can be sorted before any of them is opened. (The
  filename agrees with the file contents for every file in the sample.)
- Reads the files with a thread pool and writes rows in timestamp order, which
  is what later makes a time-range filter a `searchsorted` rather than a scan.
- **Skips and reports** files it cannot read or that lack `CamName`/`Capture`;
  one bad file does not abort a 30k-file run. PVs missing from a file are
  stored as `NaN`.
- `--append` adds files that are not already indexed and requires them to be
  newer than the newest indexed row, so the sort order holds. Unreadable files
  are exempt from that check — otherwise one corrupt file would block every
  future append. A re-ingest without `--append` rebuilds the domain.

On the 216-file sample this takes about 8 s against a local HSDS.

## Domain layout

```
/                       attrs: schema_version, created_utc, n_rows, source_root
/index/timestamp        int64  [N]  epoch microseconds UTC, ascending
/index/camera           int16  [N]  code into /index/camera_names
/index/camera_names     str    [C]
/index/n_images         int32  [N]
/index/capture_path     str    [N]  relative to the acquisition host's archive root
/index/source_file      str    [N]
/meta/pv_names          str    [P]  canonical EPICS names
/meta/pv_keys           str    [P]  sanitised link names, index-aligned
/pv/<pv_key>            f8     [N]  attrs: pv_name, area, family, device, number, field, min, max
```

One 1-D dataset per PV, rather than a wide table: a query mentioning three PVs
transfers three thin hyperslabs instead of 182 columns' worth of chunk, a PV
that appears later in the archive is a new dataset rather than a migration, and
per-PV metadata has somewhere to live. All datasets are `maxshape=(None,)` so
`--append` never rewrites.

`pv_key` replaces `-` with `_` and `:` with `__`, so
`CLA-S04-MAG-QUAD-01:SETI` becomes `CLA_S04_MAG_QUAD_01__SETI`. Queries accept
either spelling, any case, and a bare device name when it is unambiguous — so
`CLA_S04_MAG_QUAD_01` resolves to `CLA-S04-MAG-QUAD-01:SETI`.

## How a query runs

Every search goes through the same funnel in [store.py](archiver_search/store.py):

1. the time range becomes a contiguous row window by `searchsorted` on the
   cached timestamp column;
2. the camera filter becomes a mask over the cached camera codes;
3. each PV predicate reads one hyperslab of its column over that window, issued
   in parallel, and becomes a numpy mask;
4. the masks combine per the request's boolean tree;
5. sort and paginate to the concrete row indices of the page;
6. only that page is hydrated with capture paths and selected PV values.

Every response reports `scanned_rows` (how many rows the time window narrowed
to) and `pv_reads` (hyperslabs fetched), so it is obvious when a query is
touching more than it should.

## API

Interactive docs at `/docs`. All times are UTC; a bare date is accepted.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/meta` | row count, time bounds, camera list, PV count |
| `GET /api/cameras` | cameras with row counts and first/last capture |
| `GET /api/pvs?q=&area=&family=&field=&limit=` | PV catalogue with stored min/max |
| `POST /api/search` | full structured query, including nested boolean groups |
| `GET /api/search` | the same query as flat parameters — shareable and curl-able |
| `GET /api/rows/{id}` | one row, all 182 PVs, and whether its capture is mounted here |
| `GET /api/rows/{id}/image.png?index=` | that capture's image, rendered; `404` unless mounted |
| `POST /api/series`, `GET /api/series` | per-PV time series, decimated server-side |
| `GET /api/export.csv` | every matching row, streamed, no page limit |

### Flat parameters

```
?from=&to=&days=&camera=&area=&f=<pv>|<op>|<v1>[|<v2>]&fall=&combine=and|or
&select=&sort=timestamp|camera&order=asc|desc&limit=&offset=
```

`|` separates the parts of a filter because PV names contain `:` and `-`.
Repeat `f` for more filters; `combine` joins them. A PV name containing
`* ? [` is a glob over several PVs: with `f` the row matches if **any** of them
satisfies the predicate, with `fall` if **all** do.

Operators: `eq ne lt lte gt gte between outside in isnan notnan`. `between`
and `outside` are inclusive and accept their bounds in either order; `eq`/`ne`
use a tolerance (`tol`, default `1e-9`) because these are floats read back from
a control system.

### Structured body

```json
{
  "time": {"from": "2026-06-24T00:00:00Z", "days": 5},
  "cameras": {"areas": ["S07"]},
  "filters": {"op": "and", "clauses": [
    {"pv": "CLA-S04-MAG-QUAD-01:SETI", "op": "between", "value": [0, 15]},
    {"op": "or", "clauses": [
      {"pv_pattern": "*:getPhase", "match": "any", "op": "gt", "value": 80},
      {"pv": "CLA-L01-RF-CTRL-01:getPower", "op": "isnan"}
    ]}
  ]},
  "select": ["CLA-L01-RF-CTRL-01:getPhase"],
  "sort": {"by": "timestamp", "order": "desc"},
  "limit": 100, "offset": 0
}
```

`{"from": ..., "days": 5}` is the "N days following X" form. A `from`/`to` pair
given the wrong way round is swapped rather than rejected.

### Response

```json
{"total": 216, "limit": 100, "offset": 0, "elapsed_ms": 11.0,
 "scanned_rows": 216, "pv_reads": 1,
 "rows": [{"id": 0, "timestamp": "2026-04-24T13:11:04Z",
           "camera": "CLA-S01-DIA-CAM-01", "area": "S01",
           "capture_path": "2026/4/24/CLA-S01-DIA-CAM-01_2026-04-24_13-11-04UTC_10_images_full.hdf",
           "n_images": 10,
           "source_file": "2026-04-24_13-11-04_CLA-S01-DIA-CAM-01.hdf",
           "values": {"CLA-L01-RF-CTRL-01:getPhase": 82.9554}}]}
```

Unknown PVs, unknown cameras and malformed filters are `400` with a message
naming the problem — an unknown camera is an error, not an empty result.

## The example queries

```bash
# 1. S04 QUAD-01 set between 0 and 15 A
curl -sG localhost:8000/api/search \
     --data-urlencode 'f=CLA-S04-MAG-QUAD-01:SETI|between|0|15'

# 2. all S07 cameras, 1 April - 1 May 2026 (bounds either way round)
curl -sG localhost:8000/api/search --data-urlencode area=S07 \
     --data-urlencode from=2026-05-01 --data-urlencode to=2026-04-01

# 3. the RF phases for the 5 days following a date
curl -s localhost:8000/api/series -H 'content-type: application/json' \
     -d '{"pv_pattern":"*:getPhase","time":{"from":"2026-06-24","days":5}}'

# 4. L01 RF phase over a threshold and the LRG1 solenoid at a setpoint
curl -sG localhost:8000/api/search \
     --data-urlencode 'f=CLA-L01-RF-CTRL-01:getPhase|gt|80' \
     --data-urlencode 'f=CLA-LRG1-MAG-SOL-01:SETI|eq|-193.5'
```

What they return **on the 216-file sample**, which spans 2026-04-24 to
2026-07-24 and is not representative of the full archive:

| | result |
| --- | --- |
| 1 | 216 rows — QUAD-01 never leaves 0 … 11.4 A in the sample, so everything matches. One `pv_read`. |
| 2 | 0 rows: the sample's S07 captures are all from 2026-07-24, outside April. Widen the range to see them. |
| 3 | 143 points across 7 `:getPhase` PVs. February 2026 predates the sample, hence the 2026-06-24 start. |
| 4 | As literally worded (phase > 100, SETI == 0) this is 0 rows: phase peaks at ~83 and that solenoid is only ever −198 or −193.5. The thresholds above are the in-range equivalent and return 67 rows. |

## Web UI

`http://localhost:8000` — one page, no build step.

- **Left rail**: time range (all / from-to / from + N days), cameras grouped by
  area with per-area select-all, a filter builder whose value inputs follow the
  operator's arity and which shows each PV's stored range, and a picker for
  extra PV columns.
- **Results**: sortable table, pagination, per-row expansion to all 182 PVs —
  above them, the capture's first image rendered with `imshow`, when
  `ARCHIVER_IMAGE_ROOT` points at a mounted archive — and CSV export.
- **Plot**: the selected PVs, or a glob, over the same window and filters.
- Query state lives in the URL hash in the same encoding as `GET /api/search`,
  so **Copy link** hands someone else the exact view, and the back button works.

## Tests

```bash
.venv/bin/python -m pytest -q      # 119 tests, ~12 s, no server needed
```

The suite ingests `sample/` into a temporary local `.h5` (plus one deliberately
corrupt file, to keep the skip path honest) and drives `Store` and the API over
it. The query tests are cross-checked against brute force: every sample file is
read directly with `h5py`, the same predicate is evaluated in pure Python, and
the two sets of matched source files must be identical.

## Layout

```
archiver_search/
  schema.py      domain layout; PV, filename and capture-path naming
  config.py      settings
  backend.py     h5py / h5pyd selection and domain opening
  ingest.py      directory tree -> HSDS domain
  store.py       caches and the query funnel
  models.py      request/response models
  api.py         FastAPI app
static/          index.html, app.js, style.css
scripts/         bootstrap.sh, hsds_start.sh, hsds_stop.sh
tests/
```

## Scaling notes

The sample is 216 rows; the archive is 30k+, and the design assumes it keeps
growing.

- Index size is about 1.5 kB per row (182 float64 plus the string columns), so
  30k rows is ~45 MB and a million rows ~1.5 GB.
- The API process caches the timestamp and camera columns in full — 10 bytes a
  row, so ~300 kB at 30k rows and ~50 MB at 5M. PV windows go in an LRU bounded by
  `ARCHIVER_PV_CACHE_MIB`.
- A filter reads one column over the time window, so query cost scales with
  *rows in the window × PVs mentioned*, not with the archive size. Narrow the
  time range first and the rest is cheap.
- For much larger archives, shard the domain by year and fan out — the layout
  and the query funnel are per-domain and do not assume a single file.
