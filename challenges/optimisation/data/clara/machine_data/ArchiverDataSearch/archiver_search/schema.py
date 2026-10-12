"""Layout of the HSDS domain, plus the naming conventions of the CLARA archive.

Everything that knows how a PV name, a metadata filename or a capture path is
spelled lives here, so that :mod:`ingest`, :mod:`store` and the API all agree.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# --- domain layout -----------------------------------------------------------

INDEX_GROUP = "index"
META_GROUP = "meta"
PV_GROUP = "pv"

TIMESTAMP = f"/{INDEX_GROUP}/timestamp"
CAMERA = f"/{INDEX_GROUP}/camera"
CAMERA_NAMES = f"/{INDEX_GROUP}/camera_names"
N_IMAGES = f"/{INDEX_GROUP}/n_images"
CAPTURE_PATH = f"/{INDEX_GROUP}/capture_path"
SOURCE_FILE = f"/{INDEX_GROUP}/source_file"
PV_NAMES = f"/{META_GROUP}/pv_names"
PV_KEYS = f"/{META_GROUP}/pv_keys"

#: Rows per chunk for the 1-D columns. 262144 int64 == 2 MiB, which is in the
#: range HSDS is happiest with, and one chunk covers the whole sample.
CHUNK_ROWS = 262144

#: Names in a metadata file that are not process variables.
NON_PV_DATASETS = frozenset({"CamName", "Capture"})


def pv_dataset_path(pv_key: str) -> str:
    return f"/{PV_GROUP}/{pv_key}"


# --- PV names ----------------------------------------------------------------

# CLA-S04-MAG-QUAD-01:SETI  ->  machine-area-family-device-number:field
# Verified to match all 182 PVs in the archive sample.
_PV_RE = re.compile(
    r"^(?P<machine>[A-Z0-9]+)-(?P<area>[A-Z0-9]+)-(?P<family>[A-Z0-9]+)"
    r"-(?P<device>[A-Z0-9]+)-(?P<number>\d+):(?P<field>\w+)$"
)

# CLA-S07-DIA-CAM-04 -- same, without the field.
_DEVICE_RE = re.compile(
    r"^(?P<machine>[A-Z0-9]+)-(?P<area>[A-Z0-9]+)-(?P<family>[A-Z0-9]+)"
    r"-(?P<device>[A-Z0-9]+)-(?P<number>\d+)$"
)


@dataclass(frozen=True)
class PVName:
    """The decomposition of an EPICS PV name used for grouping and filtering."""

    name: str
    machine: str | None = None
    area: str | None = None
    family: str | None = None
    device: str | None = None
    number: str | None = None
    field: str | None = None

    @property
    def key(self) -> str:
        return sanitise_pv(self.name)

    @property
    def device_name(self) -> str | None:
        """``CLA-S04-MAG-QUAD-01`` -- the PV name without its field."""
        if self.machine is None:
            return None
        return f"{self.machine}-{self.area}-{self.family}-{self.device}-{self.number}"


def parse_pv_name(name: str) -> PVName:
    """Decompose a PV name. Unparseable names still yield a usable ``PVName``."""
    m = _PV_RE.match(name)
    if m is None:
        return PVName(name=name)
    return PVName(name=name, **m.groupdict())


def parse_device_name(name: str) -> PVName:
    """Decompose a device name such as a camera, which carries no field."""
    m = _DEVICE_RE.match(name)
    if m is None:
        return PVName(name=name)
    return PVName(name=name, **m.groupdict())


def sanitise_pv(name: str) -> str:
    """``CLA-S04-MAG-QUAD-01:SETI`` -> ``CLA_S04_MAG_QUAD_01__SETI``.

    Used as the HSDS link name. No PV in the archive contains an underscore, so
    this is reversible, but the canonical mapping is stored in the domain
    anyway and is what lookups actually use.
    """
    return name.replace(":", "__").replace("-", "_")


def unsanitise_pv(key: str) -> str:
    """Inverse of :func:`sanitise_pv`."""
    head, sep, tail = key.partition("__")
    base = head.replace("_", "-")
    return f"{base}:{tail}" if sep else base


def fold_pv(token: str) -> str:
    """Normalise a user-supplied PV spelling for matching.

    Case is dropped and ``-``/``_`` are unified, so ``CLA_S04_MAG_QUAD_01:seti``,
    ``CLA-S04-MAG-QUAD-01:SETI`` and ``CLA_S04_MAG_QUAD_01__SETI`` all fold
    together.
    """
    token = token.strip().replace("__", ":")
    return re.sub(r"[-_]", "-", token).upper()


def pv_matches_pattern(name: str, pattern: str) -> bool:
    """Glob match that is insensitive to case and to ``-``/``_`` spelling."""
    return fnmatch.fnmatchcase(fold_pv(name), fold_pv(pattern))


# --- filenames ---------------------------------------------------------------

# 2026-04-24_13-11-04_CLA-S01-DIA-CAM-01.hdf
_FILENAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<camera>.+)\.hdf$"
)

# ..._2026-04-24_13-11-04UTC_10_images_full.hdf
_CAPTURE_RE = re.compile(r"_(?P<n_images>\d+)_images_(?P<suffix>[^/]*)\.hdf$")


@dataclass(frozen=True)
class MetadataFilename:
    timestamp: datetime
    camera: str


def parse_metadata_filename(filename: str) -> MetadataFilename | None:
    """Pull the UTC timestamp and camera out of a metadata filename.

    Verified against the file contents for every file in the sample, so this is
    a trustworthy pre-index: ingest can sort 30k files without opening any.
    """
    m = _FILENAME_RE.match(filename)
    if m is None:
        return None
    stamp = f"{m.group('date')} {m.group('time').replace('-', ':')}"
    try:
        ts = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return MetadataFilename(
        timestamp=ts.replace(tzinfo=timezone.utc), camera=m.group("camera")
    )


def parse_capture_path(path: str) -> int:
    """Number of images in a capture file, from its ``_<n>_images_`` token.

    Returns 1 when the token is absent, which is the safe assumption: a capture
    file always holds at least one image.
    """
    m = _CAPTURE_RE.search(path)
    if m is None:
        return 1
    return int(m.group("n_images"))


#: Where capture files sit on the acquisition host. It is an artefact of that
#: one machine, so the index stores paths relative to it and each reader says
#: where it has the archive mounted (``ARCHIVER_IMAGE_ROOT``).
CAPTURE_ROOT = "/home/jovyan/work/shared/claraserv3"


def strip_capture_root(path: str) -> str:
    """``/home/jovyan/work/shared/claraserv3/2026/4/24/x.hdf`` -> ``2026/4/24/x.hdf``.

    Always returns a relative path, so joining it onto an image root can never
    escape that root. Idempotent, which lets it double as a read-side repair
    for indexes built before paths were stored relative.
    """
    text = path.strip()
    if text.startswith(CAPTURE_ROOT):
        text = text[len(CAPTURE_ROOT) :]
    return text.lstrip("/")


def image_dataset_name(index: int) -> str:
    """``1 -> 'Capture000001'`` -- how images are addressed inside a capture file."""
    return f"Capture{index:06d}"


# --- time --------------------------------------------------------------------

def to_epoch_us(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def from_epoch_us(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)


def isoformat_us(us: int) -> str:
    return from_epoch_us(us).isoformat().replace("+00:00", "Z")
