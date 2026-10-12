"""Shared fixtures.

The whole stack is exercised without a server: ``ingest`` writes a real index
into a temporary local ``.h5`` with ``h5py``, and ``Store`` reads it back
through the same ``h5py``-shaped interface it uses for h5pyd. Tests marked
``hsds`` instead point at a live server via ``ARCHIVER_*`` in the environment.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from archiver_search import schema
from archiver_search.config import Settings
from archiver_search.ingest import ingest
from archiver_search.store import Store

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    if not SAMPLE_DIR.is_dir():
        pytest.skip(f"no sample data at {SAMPLE_DIR}")
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def source_tree(tmp_path_factory, sample_dir: Path) -> Path:
    """A copy of ``sample/`` with one deliberately corrupt file added.

    A 30k-file run will meet unreadable files; the ingest is supposed to skip
    them and carry on, so the fixture makes sure there is always one to skip.
    """
    root = tmp_path_factory.mktemp("source")
    for path in sorted(sample_dir.glob("*.hdf")):
        shutil.copy2(path, root / path.name)
    (root / "2026-05-01_00-00-00_CLA-S01-DIA-CAM-99.hdf").write_bytes(
        b"not an HDF5 file at all\n"
    )
    return root


@pytest.fixture(scope="session")
def index(tmp_path_factory, source_tree: Path):
    """``(path, report)`` for an index built from :func:`source_tree`."""
    domain = tmp_path_factory.mktemp("domain") / "archiver.h5"
    settings = Settings(domain=str(domain), backend="local")
    report = ingest(str(source_tree), domain=str(domain), workers=4, settings=settings)
    return domain, report


@pytest.fixture(scope="session")
def settings(index) -> Settings:
    domain, _ = index
    return Settings(domain=str(domain), backend="local")


@pytest.fixture(scope="session")
def store(settings: Settings) -> Store:
    with Store(settings=settings) as s:
        yield s


@pytest.fixture()
def api_env() -> dict[str, str]:
    """Extra ``ARCHIVER_*`` settings for :func:`client`; override to add some."""
    return {}


@pytest.fixture()
def client(settings: Settings, api_env: dict[str, str], monkeypatch):
    """A TestClient whose app is bound to the temporary local index."""
    from fastapi.testclient import TestClient

    from archiver_search import api
    from archiver_search.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("ARCHIVER_DOMAIN", settings.domain)
    monkeypatch.setenv("ARCHIVER_BACKEND", "local")
    for key, value in api_env.items():
        monkeypatch.setenv(key, value)
    try:
        # The store is created by the lifespan handler, so the app must be
        # entered as a context manager.
        with TestClient(api.app) as c:
            yield c
    finally:
        get_settings.cache_clear()


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@pytest.fixture(scope="session")
def reference_rows(source_tree: Path) -> list[dict]:
    """Every sample file read directly with h5py, for brute-force cross-checks."""
    import h5py

    rows = []
    for path in sorted(source_tree.glob("*.hdf")):
        try:
            with h5py.File(path, "r") as f:
                values = {
                    name: float(f[name][()])
                    for name in f
                    if name not in schema.NON_PV_DATASETS
                }
                camera = _text(f["CamName"][()])
                capture = _text(f["Capture"][()])
        except (OSError, KeyError):
            continue
        parsed = schema.parse_metadata_filename(os.path.basename(path))
        rows.append(
            {
                "source_file": os.path.basename(path),
                "timestamp": parsed.timestamp if parsed else None,
                "camera": camera,
                # The index stores capture paths relative to the acquisition
                # host's root, so compare against the same form.
                "capture_path": schema.strip_capture_root(capture),
                "values": values,
            }
        )
    return rows
