"""Runtime configuration, read from the environment or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ARCHIVER_", extra="ignore"
    )

    #: HSDS endpoint. Leave unset to fall back to ``~/.hscfg`` / ``HS_ENDPOINT``.
    hsds_endpoint: str | None = "http://localhost:5101"
    hsds_username: str | None = "admin"
    hsds_password: str | None = "admin"
    hsds_bucket: str | None = None

    #: Domain holding the index.
    domain: str = "/home/admin/clara/archiver.h5"
    #: ``auto`` opens ``domain`` with h5py when it looks like a local HDF5 file
    #: and with h5pyd otherwise; ``hsds``/``local`` force the choice.
    backend: str = "auto"

    #: Where this host has the capture archive mounted, e.g.
    #: ``/mnt/images/clara``. Stored capture paths are relative to it. Left
    #: unset, the API simply offers no image previews.
    image_root: str | None = None

    #: Seconds before the cached index is re-checked against the domain.
    cache_ttl_seconds: float = 30.0
    #: Upper bound on the PV column cache, in MiB.
    pv_cache_mib: int = 256
    #: Parallel hyperslab reads issued against HSDS for one query.
    read_workers: int = 8

    #: Rows a single /api/search page may return.
    max_limit: int = 1000
    #: Rows /api/export.csv will stream before stopping.
    max_export_rows: int = 1_000_000

    api_host: str = "127.0.0.1"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
