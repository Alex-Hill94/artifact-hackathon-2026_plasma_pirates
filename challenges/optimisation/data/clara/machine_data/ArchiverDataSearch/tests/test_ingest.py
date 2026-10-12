"""The index an ingest produces must be complete, sorted and self-describing."""

from __future__ import annotations

import numpy as np
import pytest

from archiver_search import schema
from archiver_search.backend import as_str
from archiver_search.config import Settings
from archiver_search.ingest import build_candidates, ingest, IngestReport, read_metadata_file


def test_every_readable_file_is_ingested(index, reference_rows):
    _, report = index
    assert report.ingested == len(reference_rows)
    # The one deliberately corrupt file is skipped, not fatal.
    assert report.scanned == report.ingested + report.skipped_count
    assert report.skipped_count == 1
    assert any(
        "CAM-99" in path for paths in report.skipped.values() for path in paths
    )


def test_pv_columns_cover_the_source_files(index, reference_rows):
    _, report = index
    expected = {name for row in reference_rows for name in row["values"]}
    assert report.pv_count == len(expected)


def test_rows_are_sorted_by_timestamp(index):
    import h5py

    domain, _ = index
    with h5py.File(domain, "r") as f:
        stamps = f[schema.TIMESTAMP][...]
    assert stamps.size > 0
    assert np.all(np.diff(stamps) >= 0)


def test_layout_and_root_attributes(index, reference_rows):
    import h5py

    domain, report = index
    with h5py.File(domain, "r") as f:
        assert f.attrs["schema_version"] == schema.SCHEMA_VERSION
        assert int(f.attrs["n_rows"]) == report.ingested
        assert "created_utc" in f.attrs

        n = f[schema.TIMESTAMP].shape[0]
        for path in (
            schema.CAMERA,
            schema.N_IMAGES,
            schema.CAPTURE_PATH,
            schema.SOURCE_FILE,
        ):
            assert f[path].shape == (n,), path
            # Appendable, so a later --append does not need a rewrite.
            assert f[path].maxshape == (None,), path

        names = [s.decode() if isinstance(s, bytes) else s for s in f[schema.PV_NAMES][...]]
        keys = [s.decode() if isinstance(s, bytes) else s for s in f[schema.PV_KEYS][...]]
        assert len(names) == len(keys) == report.pv_count
        for name, key in zip(names, keys):
            assert key == schema.sanitise_pv(name)
            column = f[schema.pv_dataset_path(key)]
            assert column.shape == (n,)
            assert column.attrs["pv_name"] == name


def test_pv_min_max_attributes_match_the_data(index):
    import h5py

    domain, _ = index
    with h5py.File(domain, "r") as f:
        keys = [s.decode() if isinstance(s, bytes) else s for s in f[schema.PV_KEYS][...]]
        for key in keys[:20]:
            column = f[schema.pv_dataset_path(key)]
            values = column[...]
            finite = values[np.isfinite(values)]
            if not finite.size:
                continue
            assert column.attrs["min"] == pytest.approx(float(finite.min()))
            assert column.attrs["max"] == pytest.approx(float(finite.max()))


def test_values_match_a_direct_read(index, reference_rows):
    """Spot-check the columnar transpose against the original files."""
    import h5py

    domain, _ = index
    by_source = {row["source_file"]: row for row in reference_rows}
    with h5py.File(domain, "r") as f:
        sources = [
            s.decode() if isinstance(s, bytes) else s for s in f[schema.SOURCE_FILE][...]
        ]
        cameras = [
            s.decode() if isinstance(s, bytes) else s for s in f[schema.CAMERA_NAMES][...]
        ]
        codes = f[schema.CAMERA][...]
        captures = f[schema.CAPTURE_PATH][...]
        n_images = f[schema.N_IMAGES][...]
        keys = [s.decode() if isinstance(s, bytes) else s for s in f[schema.PV_KEYS][...]]
        names = [s.decode() if isinstance(s, bytes) else s for s in f[schema.PV_NAMES][...]]

        for row_id in (0, len(sources) // 2, len(sources) - 1):
            expected = by_source[sources[row_id]]
            assert cameras[codes[row_id]] == expected["camera"]
            capture = captures[row_id]
            capture = capture.decode() if isinstance(capture, bytes) else str(capture)
            assert capture == expected["capture_path"]
            assert n_images[row_id] == schema.parse_capture_path(expected["capture_path"])
            for name, key in zip(names, keys):
                stored = float(f[schema.pv_dataset_path(key)][row_id])
                assert stored == pytest.approx(expected["values"][name])


def test_unreadable_file_is_reported_rather_than_raised(tmp_path):
    bad = tmp_path / "2026-05-01_00-00-00_CLA-S01-DIA-CAM-01.hdf"
    bad.write_bytes(b"definitely not HDF5")
    report = IngestReport()
    candidates = build_candidates([str(bad)], report)
    assert len(candidates) == 1
    result = read_metadata_file(candidates[0])
    assert isinstance(result, tuple)
    assert result[0] == "not-hdf5"


def test_candidates_are_sorted_without_opening_files(sample_dir):
    paths = sorted(str(p) for p in sample_dir.glob("*.hdf"))
    candidates = build_candidates(paths, IngestReport())
    assert len(candidates) == len(paths)
    stamps = [c.timestamp for c in candidates]
    assert stamps == sorted(stamps)


def test_append_is_idempotent(index, source_tree, tmp_path):
    """Re-ingesting the same tree with --append adds nothing."""
    import shutil

    import h5py

    domain = tmp_path / "append.h5"
    shutil.copy2(index[0], domain)
    settings = Settings(domain=str(domain), backend="local")
    report = ingest(str(source_tree), domain=str(domain), append=True, settings=settings)
    assert report.ingested == 0
    with h5py.File(domain, "r") as f:
        assert f[schema.TIMESTAMP].shape[0] == index[1].ingested


def test_append_adds_rows_and_grows_every_column(sample_dir, tmp_path):
    """An append of newer files extends the index and every PV column with it.

    Datasets are allocated as rows arrive rather than trimmed at the end, so an
    append has to grow columns that were sized for the first half alone.
    """
    import shutil

    import h5py

    paths = sorted(sample_dir.glob("*.hdf"))
    first, second = paths[: len(paths) // 2], paths[len(paths) // 2 :]
    early, late = tmp_path / "early", tmp_path / "late"
    for root, group in ((early, first), (late, second)):
        root.mkdir()
        for path in group:
            shutil.copy2(path, root / path.name)

    domain = tmp_path / "grown.h5"
    settings = Settings(domain=str(domain), backend="local")
    built = ingest(str(early), domain=str(domain), settings=settings)
    appended = ingest(
        str(late), domain=str(domain), append=True, settings=settings
    )
    assert appended.ingested > 0

    total = built.ingested + appended.ingested
    with h5py.File(domain, "r") as f:
        assert f.attrs["n_rows"] == total
        assert f[schema.TIMESTAMP].shape[0] == total
        stamps = f[schema.TIMESTAMP][...]
        assert np.all(np.diff(stamps) >= 0)
        # No column may be left short (or padded) by the growth.
        keys = [as_str(k) for k in f[schema.PV_KEYS][...]]
        assert keys
        for key in keys:
            assert f[schema.pv_dataset_path(key)].shape[0] == total
