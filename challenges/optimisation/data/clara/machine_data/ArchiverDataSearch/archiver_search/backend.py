"""Opening the index domain, against either HSDS or a local HDF5 file.

The whole point is that :mod:`ingest` and :mod:`store` never care which one
they got: both are ``h5py``-shaped. HSDS is the production backend; the local
file path exists so the test suite exercises the identical code with no server.
"""

from __future__ import annotations

import os
from typing import Any

import h5py

from .config import Settings, get_settings

_HDF5_SUFFIXES = (".h5", ".hdf5", ".hdf")


def resolve_backend(domain: str, backend: str = "auto") -> str:
    """Decide whether ``domain`` names a local file or an HSDS domain.

    ``auto`` calls it local when the name looks like an HDF5 file *and* its
    parent directory exists on this machine -- an HSDS domain such as
    ``/home/admin/clara/archiver.h5`` has no such directory here. Set
    ``ARCHIVER_BACKEND=hsds`` or ``=local`` to decide explicitly.
    """
    if backend in ("hsds", "local"):
        return backend
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}; use auto, hsds or local")
    if not domain.endswith(_HDF5_SUFFIXES):
        return "hsds"
    parent = os.path.dirname(os.path.abspath(domain))
    return "local" if os.path.isdir(parent) else "hsds"


def open_domain(
    domain: str | None = None,
    mode: str = "r",
    settings: Settings | None = None,
) -> Any:
    """Open the index domain for reading or writing.

    Returns an object satisfying the subset of the ``h5py.File`` API used by
    this package: ``__getitem__``, ``__contains__``, ``create_dataset``,
    ``require_group``, ``attrs`` and ``close``.
    """
    settings = settings or get_settings()
    domain = domain or settings.domain

    if resolve_backend(domain, settings.backend) == "local":
        return h5py.File(domain, mode)

    import h5pyd

    kwargs: dict[str, Any] = {}
    if settings.hsds_endpoint:
        kwargs["endpoint"] = settings.hsds_endpoint
    if settings.hsds_username:
        kwargs["username"] = settings.hsds_username
    if settings.hsds_password:
        kwargs["password"] = settings.hsds_password
    if settings.hsds_bucket:
        kwargs["bucket"] = settings.hsds_bucket
    return h5pyd.File(domain, mode, **kwargs)


def ensure_parent_folders(domain: str, settings: Settings | None = None) -> None:
    """Create the HSDS folders leading up to ``domain``.

    HSDS will not create a domain whose parent folder is missing, and a fresh
    server has only ``/home/<user>/``. No-op for the local-file backend, where
    the directory is expected to exist already.
    """
    settings = settings or get_settings()
    if resolve_backend(domain, settings.backend) == "local":
        return

    import h5pyd

    kwargs: dict[str, Any] = {}
    if settings.hsds_endpoint:
        kwargs["endpoint"] = settings.hsds_endpoint
    if settings.hsds_username:
        kwargs["username"] = settings.hsds_username
    if settings.hsds_password:
        kwargs["password"] = settings.hsds_password
    if settings.hsds_bucket:
        kwargs["bucket"] = settings.hsds_bucket

    parts = [p for p in os.path.dirname(domain).split("/") if p]
    path = ""
    for part in parts:
        path = f"{path}/{part}"
        try:
            h5pyd.Folder(path + "/", mode="x", **kwargs)
        except (OSError, IOError):
            # Already present, or not ours to create (e.g. /home itself).
            pass


def string_dtype(h5file: Any) -> Any:
    """Variable-length UTF-8 string dtype for whichever backend ``h5file`` is."""
    if isinstance(h5file, h5py.File):
        return h5py.string_dtype(encoding="utf-8")
    import h5pyd

    return h5pyd.special_dtype(vlen=str)


def as_str(value: Any) -> str:
    """Decode a string read back from HDF5/HSDS, which may be ``bytes``."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
