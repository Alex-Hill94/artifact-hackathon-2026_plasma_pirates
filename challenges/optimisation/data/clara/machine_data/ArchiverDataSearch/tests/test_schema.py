"""Naming conventions: the part everything else is built on."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from archiver_search import schema


@pytest.mark.parametrize(
    "name,area,family,device,number,field",
    [
        ("CLA-S04-MAG-QUAD-01:SETI", "S04", "MAG", "QUAD", "01", "SETI"),
        ("CLA-L01-RF-CTRL-01:getPhase", "L01", "RF", "CTRL", "01", "getPhase"),
        ("CLA-LRG1-MAG-SOL-01:SETI", "LRG1", "MAG", "SOL", "01", "SETI"),
        ("CLA-S01-DIA-WCM-01:DQ", "S01", "DIA", "WCM", "01", "DQ"),
    ],
)
def test_parse_pv_name(name, area, family, device, number, field):
    parsed = schema.parse_pv_name(name)
    assert (parsed.machine, parsed.area, parsed.family) == ("CLA", area, family)
    assert (parsed.device, parsed.number, parsed.field) == (device, number, field)
    assert parsed.device_name == f"CLA-{area}-{family}-{device}-{number}"


def test_parse_pv_name_tolerates_junk():
    parsed = schema.parse_pv_name("not a pv")
    assert parsed.name == "not a pv"
    assert parsed.area is None
    assert parsed.device_name is None


def test_sanitise_round_trip():
    name = "CLA-S04-MAG-QUAD-01:SETI"
    key = schema.sanitise_pv(name)
    assert key == "CLA_S04_MAG_QUAD_01__SETI"
    assert schema.unsanitise_pv(key) == name


def test_fold_unifies_the_spellings_a_user_might_type():
    canonical = schema.fold_pv("CLA-S04-MAG-QUAD-01:SETI")
    for spelling in (
        "cla-s04-mag-quad-01:seti",
        "CLA_S04_MAG_QUAD_01:SETI",
        "CLA_S04_MAG_QUAD_01__SETI",
        "  CLA-S04-MAG-QUAD-01:SETI  ",
    ):
        assert schema.fold_pv(spelling) == canonical


def test_pattern_match_ignores_case_and_separator():
    assert schema.pv_matches_pattern("CLA-L01-RF-CTRL-01:getPhase", "*:getphase")
    assert schema.pv_matches_pattern("CLA-L01-RF-CTRL-01:getPhase", "CLA_L01_RF*")
    assert not schema.pv_matches_pattern("CLA-L01-RF-CTRL-01:getPower", "*:getPhase")


def test_parse_metadata_filename():
    parsed = schema.parse_metadata_filename("2026-04-24_13-11-04_CLA-S01-DIA-CAM-01.hdf")
    assert parsed is not None
    assert parsed.camera == "CLA-S01-DIA-CAM-01"
    assert parsed.timestamp == datetime(2026, 4, 24, 13, 11, 4, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "filename",
    ["notes.txt", "2026-13-99_13-11-04_CAM.hdf", "CLA-S01-DIA-CAM-01.hdf"],
)
def test_parse_metadata_filename_rejects(filename):
    assert schema.parse_metadata_filename(filename) is None


def test_parse_capture_path():
    path = (
        "/home/jovyan/work/shared/claraserv3/2026/4/24/"
        "CLA-S01-DIA-CAM-01_2026-04-24_13-11-04UTC_10_images_full.hdf"
    )
    assert schema.parse_capture_path(path) == 10
    # No token: a capture file always holds at least one image.
    assert schema.parse_capture_path("/tmp/whatever.hdf") == 1


def test_strip_capture_root():
    path = (
        "/home/jovyan/work/shared/claraserv3/2026/4/24/"
        "CLA-S01-DIA-CAM-01_2026-04-24_13-11-04UTC_10_images_full.hdf"
    )
    relative = "2026/4/24/CLA-S01-DIA-CAM-01_2026-04-24_13-11-04UTC_10_images_full.hdf"
    assert schema.strip_capture_root(path) == relative
    # Idempotent, so it can repair paths read out of an older index.
    assert schema.strip_capture_root(relative) == relative


@pytest.mark.parametrize(
    "path", ["/somewhere/else/x.hdf", "/x.hdf", "  /home/jovyan/work/shared/claraserv3/a.hdf  "]
)
def test_strip_capture_root_always_returns_a_relative_path(path):
    """Joining the result onto an image root must never escape it."""
    assert not schema.strip_capture_root(path).startswith("/")


def test_image_dataset_name_matches_the_documented_accessor():
    assert schema.image_dataset_name(1) == "Capture000001"
    assert schema.image_dataset_name(10) == "Capture000010"


def test_epoch_round_trip():
    when = datetime(2026, 2, 2, 3, 4, 5, tzinfo=timezone.utc)
    us = schema.to_epoch_us(when)
    assert schema.from_epoch_us(us) == when
    assert schema.isoformat_us(us) == "2026-02-02T03:04:05Z"


def test_naive_datetimes_are_read_as_utc():
    naive = datetime(2026, 2, 2, 3, 4, 5)
    aware = naive.replace(tzinfo=timezone.utc)
    assert schema.to_epoch_us(naive) == schema.to_epoch_us(aware)
