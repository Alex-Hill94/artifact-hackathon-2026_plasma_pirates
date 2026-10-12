"""HTTP API over the HSDS index.

Two ways in, one engine. ``POST /api/search`` takes the full structured query
including nested boolean groups; ``GET /api/search`` takes the same query as
flat repeatable parameters so that a result set is a shareable URL and a
``curl`` one-liner. Both build the same :class:`SearchRequest`.
"""

from __future__ import annotations

import csv
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import Settings, get_settings
from .images import ImageError, read_capture_image, render_png
from .models import (
    CameraFilter,
    CameraOut,
    Clause,
    Group,
    MetaOut,
    PVOut,
    RowOut,
    SearchRequest,
    SearchResponse,
    SeriesRequest,
    SeriesResponse,
    Sort,
    TimeRange,
)
from .store import PVInfo, QueryError, Store

log = logging.getLogger("archiver_search.api")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = Store(settings=settings)
    log.info("serving domain %s", app.state.store.domain)
    try:
        yield
    finally:
        app.state.store.close()


app = FastAPI(
    title="CLARA Archiver Image Search",
    description=__doc__,
    version="0.1.0",
    lifespan=lifespan,
)


def get_store(request: Request) -> Store:
    return request.app.state.store


StoreDep = Annotated[Store, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.exception_handler(QueryError)
async def _query_error(request: Request, exc: QueryError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})


# --- flat query-string form --------------------------------------------------


def parse_filter_token(token: str, match: str) -> Clause:
    """``CLA-S04-MAG-QUAD-01:SETI|between|0|15`` -> a :class:`Clause`.

    ``|`` separates the parts because PV names contain ``:`` and ``-``. A name
    containing a glob character is taken as a pattern over several PVs.
    """
    parts = token.split("|")
    if len(parts) < 2:
        raise QueryError(
            f"filter {token!r} must be '<pv>|<op>[|<value>...]', "
            "e.g. 'CLA-S04-MAG-QUAD-01:SETI|between|0|15'"
        )
    name, op, *raw = parts
    name = name.strip()
    if not name:
        raise QueryError(f"filter {token!r} is missing a PV name")

    values: Any
    if not raw:
        values = None
    elif len(raw) == 1:
        values = _number(raw[0], token)
    else:
        values = [_number(v, token) for v in raw]

    fields: dict[str, Any] = {"op": op.strip().lower(), "value": values}
    if any(ch in name for ch in "*?["):
        fields["pv_pattern"] = name
        fields["match"] = match
    else:
        fields["pv"] = name
    try:
        return Clause(**fields)
    except ValidationError as exc:
        reasons = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        raise QueryError(f"filter {token!r}: {reasons}") from exc


def _number(text: str, token: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise QueryError(f"filter {token!r}: {text!r} is not a number") from exc


def search_params(
    from_: Annotated[str | None, Query(alias="from")] = None,
    to: Annotated[str | None, Query()] = None,
    days: Annotated[float | None, Query()] = None,
    camera: Annotated[list[str], Query()] = [],
    area: Annotated[list[str], Query()] = [],
    f: Annotated[
        list[str],
        Query(description="filter as '<pv>|<op>[|<value>...]'; repeatable"),
    ] = [],
    fall: Annotated[
        list[str],
        Query(description="as 'f', but a glob must match on every PV"),
    ] = [],
    combine: Annotated[str, Query(pattern="^(and|or)$")] = "and",
    select: Annotated[list[str], Query()] = [],
    sort: Annotated[str, Query(pattern="^(timestamp|camera)$")] = "timestamp",
    order: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
    limit: Annotated[int, Query(ge=0)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SearchRequest:
    """Build a :class:`SearchRequest` from flat, repeatable query parameters."""
    clauses = [parse_filter_token(token, "any") for token in f]
    clauses += [parse_filter_token(token, "all") for token in fall]
    try:
        window = TimeRange.model_validate(
            {"from": from_, "to": to, "days": days}
        )
    except ValueError as exc:
        raise QueryError(str(exc)) from exc
    return SearchRequest(
        time=window,
        cameras=CameraFilter(names=camera, areas=area),
        filters=Group(op=combine, clauses=clauses) if clauses else None,
        select=select,
        sort=Sort(by=sort, order=order),
        limit=limit,
        offset=offset,
    )


SearchParams = Annotated[SearchRequest, Depends(search_params)]


# --- endpoints ---------------------------------------------------------------


@app.get("/api/meta", response_model=MetaOut, summary="Index extent and contents")
def meta(store: StoreDep) -> dict[str, Any]:
    return store.meta()


@app.get("/api/cameras", response_model=list[CameraOut], summary="Known cameras")
def cameras(store: StoreDep) -> list[dict[str, Any]]:
    return store.cameras()


@app.get("/api/pvs", response_model=list[PVOut], summary="Process variable catalogue")
def pvs(
    store: StoreDep,
    q: str | None = None,
    area: str | None = None,
    family: str | None = None,
    field: str | None = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> list[dict[str, Any]]:
    found = store.list_pvs(q=q, area=area, family=family, field_=field)
    return [_pv_out(info) for info in found[:limit]]


def _pv_out(info: PVInfo) -> dict[str, Any]:
    p = info.parsed
    return {
        "name": info.name,
        "key": info.key,
        "machine": p.machine,
        "area": p.area,
        "family": p.family,
        "device": p.device,
        "number": p.number,
        "field": p.field,
        "min": info.min,
        "max": info.max,
    }


@app.post("/api/search", response_model=SearchResponse, summary="Search (structured)")
def search_post(
    request: SearchRequest, store: StoreDep, settings: SettingsDep
) -> dict[str, Any]:
    _check_limit(request, settings)
    return store.search(request)


@app.get("/api/search", response_model=SearchResponse, summary="Search (query string)")
def search_get(
    request: SearchParams, store: StoreDep, settings: SettingsDep
) -> dict[str, Any]:
    _check_limit(request, settings)
    return store.search(request)


def _check_limit(request: SearchRequest, settings: Settings) -> None:
    if request.limit > settings.max_limit:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be <= {settings.max_limit}; use /api/export.csv "
            "for bulk retrieval",
        )


@app.get("/api/rows/{row_id}", response_model=RowOut, summary="One row, all PVs")
def row(row_id: int, store: StoreDep) -> dict[str, Any]:
    return store.row(row_id)


@app.get(
    "/api/rows/{row_id}/image.png",
    summary="Capture image, rendered",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "the capture file is not mounted on this host"},
    },
)
def row_image(
    row_id: int,
    store: StoreDep,
    index: Annotated[
        int, Query(ge=1, description="1-based image within the capture")
    ] = 1,
) -> Response:
    path = store.image_path(store.capture_path(row_id))
    if path is None:
        raise HTTPException(
            status_code=404, detail=f"no capture file for row {row_id} under image_root"
        )
    try:
        data = read_capture_image(path, index)
    except ImageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Capture files never change once written, so let the browser keep it.
    return Response(
        content=render_png(data),
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"},
    )


@app.post("/api/series", response_model=SeriesResponse, summary="PV time series")
def series(request: SeriesRequest, store: StoreDep) -> dict[str, Any]:
    return store.series(request)


@app.get("/api/series", response_model=SeriesResponse, summary="PV time series")
def series_get(
    request: SearchParams,
    store: StoreDep,
    pv: Annotated[list[str], Query()] = [],
    pv_pattern: str | None = None,
    max_points: Annotated[int, Query(ge=2, le=200_000)] = 4000,
) -> dict[str, Any]:
    if not pv and not pv_pattern:
        raise QueryError("give at least one 'pv' or a 'pv_pattern'")
    return store.series(
        SeriesRequest(
            pvs=pv,
            pv_pattern=pv_pattern,
            time=request.time,
            cameras=request.cameras,
            filters=request.filters,
            max_points=max_points,
        )
    )


@app.get("/api/export.csv", summary="Stream all matching rows as CSV")
def export_csv(request: SearchParams, store: StoreDep) -> StreamingResponse:
    # The page limit does not apply to an export; iter_rows is bounded by
    # settings.max_export_rows instead.
    request.limit = 0
    request.offset = 0
    selected = [store.resolve_pv(token).name for token in request.select]
    header = [
        "id",
        "timestamp",
        "camera",
        "area",
        "capture_path",
        "n_images",
        "source_file",
        *selected,
    ]

    def rows() -> Iterator[str]:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        yield _drain(buffer)
        for block in store.iter_rows(request):
            for item in block:
                writer.writerow(
                    [
                        item["id"],
                        item["timestamp"],
                        item["camera"],
                        item["area"] or "",
                        item["capture_path"],
                        item["n_images"],
                        item["source_file"],
                        *(item["values"].get(name, "") for name in selected),
                    ]
                )
            yield _drain(buffer)

    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="archiver-search.csv"'},
    )


def _drain(buffer: io.StringIO) -> str:
    text = buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)
    return text


# --- static UI ---------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
