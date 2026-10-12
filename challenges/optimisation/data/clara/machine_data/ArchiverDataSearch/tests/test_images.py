"""Capture images: finding them under the host's root, and rendering them.

The index records capture paths relative to the acquisition host, so each
reader says where it has the archive mounted. Nothing here assumes the archive
is mounted at all -- the common case is that it is not.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from archiver_search import schema
from archiver_search.config import Settings
from archiver_search.images import ImageError, read_capture_image, render_png
from archiver_search.store import Store

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Row 0 gets a capture file; row 1 deliberately does not.
PRESENT, ABSENT = 0, 1


def _gradient(height: int = 12, width: int = 20) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    return (x + y).astype("uint16")


@pytest.fixture()
def image_root(tmp_path, store):
    """An archive holding the capture file for row 0, and nothing else."""
    root = tmp_path / "images"
    target = root / store.capture_path(PRESENT)
    target.parent.mkdir(parents=True)
    with h5py.File(target, "w") as f:
        f.create_dataset(schema.image_dataset_name(1), data=_gradient())
    return root


@pytest.fixture()
def api_env(image_root) -> dict[str, str]:
    return {"ARCHIVER_IMAGE_ROOT": str(image_root)}


@pytest.fixture()
def rooted_store(settings, image_root):
    with Store(
        settings=Settings(
            domain=settings.domain, backend="local", image_root=str(image_root)
        )
    ) as s:
        yield s


# --- locating ----------------------------------------------------------------


def test_capture_paths_are_stored_relative(store):
    """The acquisition host's own root must not leak into the index."""
    path = store.capture_path(PRESENT)
    assert not path.startswith("/")
    assert schema.CAPTURE_ROOT not in path


def test_image_path_finds_a_mounted_capture(rooted_store, image_root):
    found = rooted_store.image_path(rooted_store.capture_path(PRESENT))
    assert found == (image_root / rooted_store.capture_path(PRESENT)).resolve()


def test_image_path_is_none_when_the_capture_is_missing(rooted_store):
    assert rooted_store.image_path(rooted_store.capture_path(ABSENT)) is None


def test_image_path_is_none_without_a_configured_root(store):
    assert store.settings.image_root is None
    assert store.image_path(store.capture_path(PRESENT)) is None


def test_image_path_tolerates_an_absolute_stored_path(rooted_store):
    """An index built before paths went relative must still resolve."""
    relative = rooted_store.capture_path(PRESENT)
    legacy = f"{schema.CAPTURE_ROOT}/{relative}"
    assert rooted_store.image_path(legacy) == rooted_store.image_path(relative)


def test_image_path_cannot_escape_the_root(rooted_store):
    assert rooted_store.image_path("../../../etc/passwd") is None


# --- reading and rendering ----------------------------------------------------


def test_read_capture_image(image_root, store):
    path = image_root / store.capture_path(PRESENT)
    assert np.array_equal(read_capture_image(path, 1), _gradient())


def test_read_capture_image_rejects_a_missing_dataset(image_root, store):
    path = image_root / store.capture_path(PRESENT)
    with pytest.raises(ImageError, match="Capture000002"):
        read_capture_image(path, 2)


def test_read_capture_image_rejects_a_non_hdf5_file(tmp_path):
    junk = tmp_path / "junk.hdf"
    junk.write_bytes(b"certainly not HDF5\n")
    with pytest.raises(ImageError, match="not readable"):
        read_capture_image(junk)


def test_render_png_produces_a_png():
    assert render_png(_gradient()).startswith(PNG_MAGIC)


def test_render_png_keeps_the_image_aspect_ratio():
    from PIL import Image
    import io

    tall = render_png(_gradient(height=40, width=10))
    with Image.open(io.BytesIO(tall)) as rendered:
        width, height = rendered.size
    assert height > width


# --- over HTTP ----------------------------------------------------------------


def test_row_reports_an_available_image(client):
    row = client.get(f"/api/rows/{PRESENT}").json()
    assert row["has_image"] is True
    # The path itself is an artefact of the acquisition host, not something to
    # show, but it is still what the image endpoint resolves against.
    assert not row["capture_path"].startswith("/")


def test_row_without_a_capture_file_reports_no_image(client):
    assert client.get(f"/api/rows/{ABSENT}").json()["has_image"] is False


def test_image_endpoint_renders_the_capture(client):
    response = client.get(f"/api/rows/{PRESENT}/image.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(PNG_MAGIC)


def test_image_endpoint_404s_for_a_capture_that_is_not_mounted(client):
    response = client.get(f"/api/rows/{ABSENT}/image.png")
    assert response.status_code == 404
    assert "image_root" in response.json()["detail"]


def test_image_endpoint_404s_for_an_image_beyond_the_capture(client):
    response = client.get(f"/api/rows/{PRESENT}/image.png", params={"index": 7})
    assert response.status_code == 404
    assert "Capture000007" in response.json()["detail"]


def test_image_endpoint_rejects_an_unknown_row(client):
    assert client.get("/api/rows/999999/image.png").status_code == 400


class TestWithoutAnImageRoot:
    """The default deployment: an index served away from the image store."""

    @pytest.fixture()
    def api_env(self) -> dict[str, str]:
        return {}

    def test_no_row_claims_an_image(self, client):
        assert client.get(f"/api/rows/{PRESENT}").json()["has_image"] is False

    def test_the_image_endpoint_404s(self, client):
        assert client.get(f"/api/rows/{PRESENT}/image.png").status_code == 404
