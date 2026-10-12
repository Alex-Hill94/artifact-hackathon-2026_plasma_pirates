"""HTTP surface: the flat query form, its parity with POST, and error handling."""

from __future__ import annotations

import csv
import io

import pytest

Q1_PV = "CLA-S04-MAG-QUAD-01:SETI"
PHASE_PV = "CLA-L01-RF-CTRL-01:getPhase"


def test_meta(client):
    meta = client.get("/api/meta").json()
    assert meta["n_rows"] > 0
    assert meta["pv_count"] > 0
    assert meta["time_from"] <= meta["time_to"]
    assert meta["cameras"]


def test_cameras(client):
    cameras = client.get("/api/cameras").json()
    assert cameras
    for camera in cameras:
        assert camera["n_rows"] > 0
        assert camera["first"] <= camera["last"]


def test_pv_catalogue_filters(client):
    everything = client.get("/api/pvs", params={"limit": 5000}).json()
    assert everything
    entry = next(pv for pv in everything if pv["name"] == Q1_PV)
    assert (entry["area"], entry["family"], entry["field"]) == ("S04", "MAG", "SETI")
    assert entry["min"] <= entry["max"]

    by_area = client.get("/api/pvs", params={"area": "S04"}).json()
    assert by_area and all(pv["area"] == "S04" for pv in by_area)

    by_query = client.get("/api/pvs", params={"q": "quad"}).json()
    assert by_query and all("QUAD" in pv["name"].upper() for pv in by_query)


def test_pv_catalogue_honours_limit(client):
    assert len(client.get("/api/pvs", params={"limit": 3}).json()) == 3


# --- the four example queries over HTTP ---------------------------------------


def test_example_1(client):
    response = client.get("/api/search", params={"f": f"{Q1_PV}|between|0|15"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] > 0
    # Only the mentioned PV is read, whatever the row count.
    assert body["pv_reads"] <= 2


def test_example_1_underscore_spelling_is_equivalent(client):
    canonical = client.get("/api/search", params={"f": f"{Q1_PV}|between|0|15"}).json()
    underscored = client.get(
        "/api/search", params={"f": "CLA_S04_MAG_QUAD_01|between|0|15"}
    ).json()
    assert canonical["total"] == underscored["total"]


def test_example_2_reversed_range(client):
    body = client.get(
        "/api/search",
        params={"area": "S07", "from": "2026-05-01", "to": "2026-04-01"},
    ).json()
    assert body["scanned_rows"] < client.get("/api/search").json()["scanned_rows"]
    for row in body["rows"]:
        assert "-S07-" in row["camera"]
        assert "2026-04-01" <= row["timestamp"] < "2026-05-01"


def test_example_3_series(client):
    start = client.get("/api/meta").json()["time_from"]
    body = client.post(
        "/api/series",
        json={"pv_pattern": "*:getPhase", "time": {"from": start, "days": 5}},
    ).json()
    assert body["series"]
    assert all(s["pv"].endswith(":getPhase") for s in body["series"])


def test_example_4(client):
    body = client.get(
        "/api/search",
        params=[
            ("f", f"{PHASE_PV}|gt|100"),
            ("f", "CLA-LRG1-MAG-SOL-01:SETI|eq|0"),
        ],
    ).json()
    # No row in the sample has that solenoid at 0; an empty result, not an error.
    assert body["total"] == 0


# --- GET/POST parity ----------------------------------------------------------


def test_get_and_post_agree(client):
    get_body = client.get(
        "/api/search",
        params=[
            ("f", f"{Q1_PV}|between|0|15"),
            ("f", f"{PHASE_PV}|gt|50"),
            ("select", Q1_PV),
            ("limit", "10"),
            ("order", "desc"),
        ],
    ).json()
    post_body = client.post(
        "/api/search",
        json={
            "filters": {
                "op": "and",
                "clauses": [
                    {"pv": Q1_PV, "op": "between", "value": [0, 15]},
                    {"pv": PHASE_PV, "op": "gt", "value": 50},
                ],
            },
            "select": [Q1_PV],
            "limit": 10,
            "sort": {"by": "timestamp", "order": "desc"},
        },
    ).json()
    assert get_body["total"] == post_body["total"]
    assert [r["id"] for r in get_body["rows"]] == [r["id"] for r in post_body["rows"]]


def test_combine_or(client):
    or_body = client.get(
        "/api/search",
        params=[("f", f"{Q1_PV}|lt|1"), ("f", f"{PHASE_PV}|gt|100"), ("combine", "or")],
    ).json()
    and_body = client.get(
        "/api/search",
        params=[("f", f"{Q1_PV}|lt|1"), ("f", f"{PHASE_PV}|gt|100")],
    ).json()
    assert or_body["total"] >= and_body["total"]


def test_fall_requires_every_matched_pv(client):
    any_body = client.get("/api/search", params={"f": "*:getPhase|gt|0"}).json()
    all_body = client.get("/api/search", params={"fall": "*:getPhase|gt|0"}).json()
    assert all_body["total"] <= any_body["total"]


def test_post_nested_group(client):
    body = client.post(
        "/api/search",
        json={
            "filters": {
                "op": "and",
                "clauses": [
                    {"pv": Q1_PV, "op": "gte", "value": 0},
                    {
                        "op": "or",
                        "clauses": [
                            {"pv": PHASE_PV, "op": "gt", "value": 100},
                            {"pv": PHASE_PV, "op": "lt", "value": 10},
                        ],
                    },
                ],
            },
            "limit": 5,
        },
    )
    assert body.status_code == 200
    assert body.json()["total"] >= 0


# --- rows and images ----------------------------------------------------------


def test_row_detail(client):
    first = client.get("/api/search", params={"limit": 1}).json()["rows"][0]
    row = client.get(f"/api/rows/{first['id']}").json()
    assert row["id"] == first["id"]
    assert row["capture_path"] == first["capture_path"]
    assert len(row["values"]) == client.get("/api/meta").json()["pv_count"]
    assert row["n_images"] >= 1
    assert "images" not in row


def test_missing_row_is_400(client):
    assert client.get("/api/rows/999999999").status_code == 400


# --- errors -------------------------------------------------------------------


@pytest.mark.parametrize(
    "params,fragment",
    [
        ({"f": "CLA-NOPE-MAG-QUAD-01:SETI|gt|1"}, "CLA-NOPE"),
        ({"f": f"{Q1_PV}|between|1"}, "two values"),
        ({"f": f"{Q1_PV}|gt"}, "requires a value"),
        ({"f": f"{Q1_PV}|gt|banana"}, "not a number"),
        ({"f": Q1_PV}, "<op>"),
        ({"f": f"{Q1_PV}|nonsense|1"}, "nonsense"),
        ({"camera": "CLA-S99-DIA-CAM-01"}, "S99"),
        ({"area": "S99"}, "S99"),
    ],
)
def test_bad_filters_are_400_with_a_readable_message(client, params, fragment):
    response = client.get("/api/search", params=params)
    assert response.status_code in (400, 422)
    assert fragment.lower() in response.text.lower()


def test_limit_cap(client):
    from archiver_search.config import get_settings

    over = get_settings().max_limit + 1
    response = client.get("/api/search", params={"limit": over})
    assert response.status_code == 400
    assert "export" in response.json()["detail"]


def test_series_needs_a_pv(client):
    assert client.get("/api/series").status_code == 400


def test_negative_offset_is_rejected(client):
    assert client.get("/api/search", params={"offset": -1}).status_code == 422


# --- export -------------------------------------------------------------------


def test_export_csv(client):
    total = client.get("/api/search", params={"limit": 0}).json()["total"]
    response = client.get("/api/export.csv", params={"select": Q1_PV})
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(response.text)))
    header, body = rows[0], rows[1:]
    assert header[:3] == ["id", "timestamp", "camera"]
    assert header[-1] == Q1_PV
    assert len(body) == total
    assert "image_datasets" not in header
    assert body[0][header.index("capture_path")].endswith(".hdf")


def test_export_respects_filters(client):
    params = {"f": f"{Q1_PV}|between|0|1"}
    total = client.get("/api/search", params={**params, "limit": 0}).json()["total"]
    text = client.get("/api/export.csv", params=params).text
    assert len(list(csv.reader(io.StringIO(text)))) == total + 1


# --- static UI ----------------------------------------------------------------


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Archiver" in response.text
    assert client.get("/static/app.js").status_code == 200
