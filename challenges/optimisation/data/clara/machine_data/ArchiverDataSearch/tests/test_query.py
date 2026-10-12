"""The query funnel, cross-checked against brute force over the source files."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from archiver_search.models import (
    CameraFilter,
    Clause,
    Group,
    SearchRequest,
    SeriesRequest,
    Sort,
    TimeRange,
)
from archiver_search.store import QueryError

Q1_PV = "CLA-S04-MAG-QUAD-01:SETI"
PHASE_PV = "CLA-L01-RF-CTRL-01:getPhase"
SOL_PV = "CLA-LRG1-MAG-SOL-01:SETI"


def all_rows(store, **kwargs) -> dict:
    return store.search(SearchRequest(limit=0, **kwargs))


def sources(store, request: SearchRequest) -> set[str]:
    """Every matching row's source filename, paging past the limit."""
    request = request.model_copy(update={"limit": 1000, "offset": 0})
    found: set[str] = set()
    while True:
        page = store.search(request)
        found.update(row["source_file"] for row in page["rows"])
        if len(found) >= page["total"] or not page["rows"]:
            return found
        request = request.model_copy(update={"offset": request.offset + request.limit})


def brute(reference_rows, predicate) -> set[str]:
    return {row["source_file"] for row in reference_rows if predicate(row)}


# --- resolution ---------------------------------------------------------------


def test_resolve_pv_accepts_every_spelling(store):
    canonical = store.resolve_pv(Q1_PV)
    for token in (
        Q1_PV,
        Q1_PV.lower(),
        "CLA_S04_MAG_QUAD_01__SETI",
        "CLA_S04_MAG_QUAD_01:SETI",
    ):
        assert store.resolve_pv(token).name == canonical.name


def test_resolve_pv_accepts_a_bare_device_name(store):
    """The user's example query spells the PV without its field."""
    assert store.resolve_pv("CLA_S04_MAG_QUAD_01").name.startswith("CLA-S04-MAG-QUAD-01")


def test_resolve_pv_reports_unknown_names(store):
    with pytest.raises(QueryError, match="CLA-XXX"):
        store.resolve_pv("CLA-XXX-MAG-QUAD-99:SETI")


def test_ambiguous_device_name_lists_the_candidates(store):
    """A device with several fields cannot be addressed by device name alone."""
    counts: dict[str, list[str]] = {}
    for info in store.list_pvs():
        device = info.parsed.device_name
        if device:
            counts.setdefault(device, []).append(info.name)
    ambiguous = next((d for d, names in counts.items() if len(names) > 1), None)
    if ambiguous is None:
        pytest.skip("no multi-field device in this sample")
    with pytest.raises(QueryError) as excinfo:
        store.resolve_pv(ambiguous)
    assert "ambiguous" in str(excinfo.value).lower()
    for name in counts[ambiguous][:2]:
        assert name in str(excinfo.value)


def test_match_pvs_glob(store):
    phases = store.match_pvs("*:getPhase")
    assert phases
    assert all(info.name.endswith(":getPhase") for info in phases)


# --- the four example queries -------------------------------------------------


def test_example_1_quad_between_0_and_15(store, reference_rows):
    """Find all images where CLA_S04_MAG_QUAD_01 is set between 0 and 15 amps."""
    request = SearchRequest(
        filters=Group(clauses=[Clause(pv="CLA_S04_MAG_QUAD_01", op="between", value=[0, 15])]),
    )
    assert sources(store, request) == brute(
        reference_rows, lambda r: 0 <= r["values"][Q1_PV] <= 15
    )


def test_example_2_s07_cameras_over_a_reversed_range(store, reference_rows):
    """S07 cameras between May 1st and April 1st -- bounds the wrong way round."""
    request = SearchRequest(
        time=TimeRange.model_validate({"from": "2026-05-01T00:00:00Z", "to": "2026-04-01T00:00:00Z"}),
        cameras=CameraFilter(areas=["S07"]),
    )
    lo = datetime(2026, 4, 1, tzinfo=timezone.utc)
    hi = datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert sources(store, request) == brute(
        reference_rows,
        lambda r: "-S07-" in r["camera"] and lo <= r["timestamp"] < hi,
    )


def test_example_3_rf_phases_for_five_days(store):
    """Get the RF phases for the 5 days following a date."""
    meta = store.meta()
    start = datetime.fromisoformat(meta["time_from"].replace("Z", "+00:00"))
    response = store.series(
        SeriesRequest(
            pv_pattern="*:getPhase",
            time=TimeRange.model_validate({"from": start.isoformat(), "days": 5}),
        )
    )
    assert response["series"]
    assert {s["pv"] for s in response["series"]} == {
        info.name for info in store.match_pvs("*:getPhase")
    }
    for s in response["series"]:
        assert len(s["t"]) == len(s["v"]) == response["total"]
        assert all(
            start <= datetime.fromisoformat(t.replace("Z", "+00:00")) < start + timedelta(days=5)
            for t in s["t"]
        )


def test_example_4_phase_and_solenoid(store, reference_rows):
    """L01 RF phase above a threshold AND the LRG1 solenoid at a setpoint.

    The literal example asks for phase > 100 and SETI == 0, neither of which
    occurs in this sample (phase peaks at ~83, SETI is only ever -198 or
    -193.5), so the thresholds are taken from the data. The point of the test is
    the AND path, which needs both clauses to match something.
    """
    threshold = 80.0
    setpoint = next(
        row["values"][SOL_PV]
        for row in reference_rows
        if row["values"][PHASE_PV] > threshold
    )
    request = SearchRequest(
        filters=Group(
            op="and",
            clauses=[
                Clause(pv=PHASE_PV, op="gt", value=threshold),
                Clause(pv=SOL_PV, op="eq", value=setpoint, tol=1e-6),
            ],
        )
    )
    expected = brute(
        reference_rows,
        lambda r: r["values"][PHASE_PV] > threshold
        and abs(r["values"][SOL_PV] - setpoint) <= 1e-6,
    )
    assert expected  # the fixture would be vacuous otherwise
    assert sources(store, request) == expected


def test_example_4_as_literally_written_is_empty_not_an_error(store):
    request = SearchRequest(
        filters=Group(
            clauses=[
                Clause(pv=PHASE_PV, op="gt", value=100),
                Clause(pv=SOL_PV, op="eq", value=0),
            ]
        )
    )
    assert store.search(request)["total"] == 0


# --- operators ----------------------------------------------------------------


OPS = [
    ("gt", 5.0, lambda v: v > 5.0),
    ("gte", 5.0, lambda v: v >= 5.0),
    ("lt", 5.0, lambda v: v < 5.0),
    ("lte", 5.0, lambda v: v <= 5.0),
    ("between", [0.0, 5.0], lambda v: 0.0 <= v <= 5.0),
    ("outside", [0.0, 5.0], lambda v: v < 0.0 or v > 5.0),
    ("notnan", None, lambda v: not math.isnan(v)),
    ("isnan", None, lambda v: math.isnan(v)),
]


@pytest.mark.parametrize("op,value,check", OPS, ids=[o[0] for o in OPS])
def test_operator_matches_brute_force(store, reference_rows, op, value, check):
    request = SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op=op, value=value)]))
    assert sources(store, request) == brute(
        reference_rows, lambda r: check(r["values"][Q1_PV])
    )


def test_between_is_inclusive_at_both_ends(store, reference_rows):
    values = sorted(row["values"][Q1_PV] for row in reference_rows)
    lo, hi = values[0], values[-1]
    inclusive = store.search(
        SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op="between", value=[lo, hi])]))
    )
    assert inclusive["total"] == len(values)
    open_lo = store.search(
        SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op="gt", value=lo)]))
    )
    assert open_lo["total"] == sum(1 for v in values if v > lo)


def test_between_tolerates_reversed_bounds(store):
    forward = store.search(
        SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op="between", value=[0, 15])]))
    )
    reverse = store.search(
        SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op="between", value=[15, 0])]))
    )
    assert forward["total"] == reverse["total"]


def test_in_matches_exact_stored_values(store, reference_rows):
    wanted = sorted({row["values"][Q1_PV] for row in reference_rows})[:3]
    request = SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op="in", value=wanted)]))
    assert sources(store, request) == brute(
        reference_rows, lambda r: any(r["values"][Q1_PV] == w for w in wanted)
    )


def test_eq_uses_a_tolerance(store, reference_rows):
    target = reference_rows[0]["values"][Q1_PV]
    tight = store.search(
        SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op="eq", value=target)]))
    )
    assert tight["total"] >= 1
    nudged = store.search(
        SearchRequest(
            filters=Group(clauses=[Clause(pv=Q1_PV, op="eq", value=target + 1e-7, tol=1e-6)])
        )
    )
    assert nudged["total"] >= tight["total"]


def test_ne_is_the_complement_of_eq(store, reference_rows):
    target = reference_rows[0]["values"][Q1_PV]
    eq = store.search(SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op="eq", value=target)])))
    ne = store.search(SearchRequest(filters=Group(clauses=[Clause(pv=Q1_PV, op="ne", value=target)])))
    assert eq["total"] + ne["total"] == all_rows(store)["total"]


# --- boolean structure --------------------------------------------------------


def test_or_of_two_clauses(store, reference_rows):
    request = SearchRequest(
        filters=Group(
            op="or",
            clauses=[
                Clause(pv=Q1_PV, op="lt", value=1),
                Clause(pv=PHASE_PV, op="gt", value=80),
            ],
        )
    )
    assert sources(store, request) == brute(
        reference_rows,
        lambda r: r["values"][Q1_PV] < 1 or r["values"][PHASE_PV] > 80,
    )


def test_nested_groups(store, reference_rows):
    request = SearchRequest(
        filters=Group(
            op="and",
            clauses=[
                Clause(pv=Q1_PV, op="between", value=[0, 15]),
                Group(
                    op="or",
                    clauses=[
                        Clause(pv=PHASE_PV, op="gt", value=80),
                        Clause(pv=PHASE_PV, op="lt", value=10),
                    ],
                ),
            ],
        )
    )
    assert sources(store, request) == brute(
        reference_rows,
        lambda r: 0 <= r["values"][Q1_PV] <= 15
        and (r["values"][PHASE_PV] > 80 or r["values"][PHASE_PV] < 10),
    )


def test_pattern_match_any_vs_all(store, reference_rows):
    phases = [info.name for info in store.match_pvs("*:getPhase")]
    assert len(phases) > 1
    threshold = 0.0
    any_req = SearchRequest(
        filters=Group(
            clauses=[Clause(pv_pattern="*:getPhase", match="any", op="gt", value=threshold)]
        )
    )
    all_req = SearchRequest(
        filters=Group(
            clauses=[Clause(pv_pattern="*:getPhase", match="all", op="gt", value=threshold)]
        )
    )
    assert sources(store, any_req) == brute(
        reference_rows, lambda r: any(r["values"][p] > threshold for p in phases)
    )
    assert sources(store, all_req) == brute(
        reference_rows, lambda r: all(r["values"][p] > threshold for p in phases)
    )


# --- cameras, time, paging ----------------------------------------------------


def test_camera_names_and_areas(store, reference_rows):
    cameras = store.cameras()
    assert cameras
    name = cameras[0]["name"]
    by_name = SearchRequest(cameras=CameraFilter(names=[name]))
    assert sources(store, by_name) == brute(reference_rows, lambda r: r["camera"] == name)

    area = cameras[0]["area"]
    by_area = SearchRequest(cameras=CameraFilter(areas=[area]))
    assert sources(store, by_area) == brute(
        reference_rows, lambda r: f"-{area}-" in r["camera"]
    )


def test_unknown_camera_is_an_error_not_an_empty_result(store):
    with pytest.raises(QueryError):
        store.search(SearchRequest(cameras=CameraFilter(names=["CLA-S99-DIA-CAM-01"])))
    with pytest.raises(QueryError):
        store.search(SearchRequest(cameras=CameraFilter(areas=["S99"])))


def test_time_window_narrows_the_scan(store):
    meta = store.meta()
    start = datetime.fromisoformat(meta["time_from"].replace("Z", "+00:00"))
    whole = store.search(SearchRequest(limit=1))
    narrow = store.search(
        SearchRequest(
            limit=1,
            time=TimeRange.model_validate({"from": start.isoformat(), "days": 1}),
        )
    )
    assert narrow["scanned_rows"] < whole["scanned_rows"]
    assert narrow["total"] <= whole["total"]


def test_empty_window_returns_nothing(store):
    request = SearchRequest(
        time=TimeRange.model_validate({"from": "1999-01-01T00:00:00Z", "to": "1999-02-01T00:00:00Z"})
    )
    response = store.search(request)
    assert response["total"] == 0
    assert response["rows"] == []


def test_pagination_covers_every_row_exactly_once(store):
    total = all_rows(store)["total"]
    seen: list[int] = []
    for offset in range(0, total, 50):
        page = store.search(SearchRequest(limit=50, offset=offset))
        assert page["total"] == total
        seen.extend(row["id"] for row in page["rows"])
    assert len(seen) == total
    assert len(set(seen)) == total


def test_sort_order(store):
    asc = store.search(SearchRequest(limit=20, sort=Sort(by="timestamp", order="asc")))
    desc = store.search(SearchRequest(limit=20, sort=Sort(by="timestamp", order="desc")))
    stamps = [row["timestamp"] for row in asc["rows"]]
    assert stamps == sorted(stamps)
    assert [row["timestamp"] for row in desc["rows"]] == sorted(
        [row["timestamp"] for row in desc["rows"]], reverse=True
    )
    assert asc["rows"][0]["id"] != desc["rows"][0]["id"]


def test_sort_by_camera_is_alphabetical(store):
    page = store.search(SearchRequest(limit=1000, sort=Sort(by="camera", order="asc")))
    names = [row["camera"] for row in page["rows"]]
    assert names == sorted(names)


# --- row shape ----------------------------------------------------------------


def test_rows_carry_capture_coordinates(store):
    page = store.search(SearchRequest(limit=5))
    for row in page["rows"]:
        assert row["n_images"] >= 1
        assert "images" not in row
        assert row["capture_path"].endswith(".hdf")


def test_select_adds_only_the_requested_pvs(store):
    page = store.search(SearchRequest(limit=3, select=[Q1_PV, "CLA_L01_RF_CTRL_01__getPhase"]))
    for row in page["rows"]:
        assert set(row["values"]) == {Q1_PV, PHASE_PV}


def test_row_endpoint_returns_every_pv(store, reference_rows):
    page = store.search(SearchRequest(limit=1))
    row = store.row(page["rows"][0]["id"])
    expected = next(r for r in reference_rows if r["source_file"] == row["source_file"])
    assert set(row["values"]) == set(expected["values"])
    for name, value in expected["values"].items():
        assert row["values"][name] == pytest.approx(value)


def test_unknown_row_id_is_an_error(store):
    with pytest.raises(QueryError):
        store.row(10**9)


# --- counters and series ------------------------------------------------------


def test_query_reads_only_the_pvs_it_mentions(store):
    """The columnar layout's whole point: one hyperslab per referenced PV."""
    response = store.search(
        SearchRequest(limit=10, filters=Group(clauses=[Clause(pv=Q1_PV, op="gte", value=0)]))
    )
    assert response["pv_reads"] <= 2  # the filter column, plus hydration at most


def test_series_respects_filters_and_decimation(store):
    unfiltered = store.series(SeriesRequest(pvs=[PHASE_PV]))
    assert unfiltered["total"] == all_rows(store)["total"]
    assert unfiltered["series"][0]["decimated"] is False

    filtered = store.series(
        SeriesRequest(
            pvs=[PHASE_PV],
            filters=Group(clauses=[Clause(pv=Q1_PV, op="lt", value=1)]),
        )
    )
    assert filtered["total"] <= unfiltered["total"]

    decimated = store.series(SeriesRequest(pvs=[PHASE_PV], max_points=10))
    series = decimated["series"][0]
    assert series["decimated"] is True
    assert len(series["t"]) <= 10


def test_series_needs_a_pv():
    with pytest.raises(ValueError):
        SeriesRequest()


def test_meta_matches_the_index(store, reference_rows):
    meta = store.meta()
    assert meta["n_rows"] == len(reference_rows)
    assert meta["pv_count"] == len({n for r in reference_rows for n in r["values"]})
    assert meta["schema_version"] >= 1
    assert sum(c["n_rows"] for c in meta["cameras"]) == meta["n_rows"]
