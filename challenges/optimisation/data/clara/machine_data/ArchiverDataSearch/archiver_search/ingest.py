"""Build the searchable index from a directory of CLARA metadata files.

Each source file is a flat HDF5 holding ~182 scalar float64 process variables
plus ``CamName`` and ``Capture``. Ingest turns that pile of files into one
column-oriented domain: a sorted timestamp index and one 1-D dataset per PV.

Usage::

    python -m archiver_search.ingest sample
    python -m archiver_search.ingest /data/archive --domain /home/admin/clara/archiver.h5
    python -m archiver_search.ingest /data/archive --append
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Sequence

import h5py
import numpy as np

from . import schema
from .backend import as_str, ensure_parent_folders, open_domain, string_dtype
from .config import Settings, get_settings

log = logging.getLogger("archiver_search.ingest")

#: Files read (and rows written) per batch. Bounds peak memory at roughly
#: BATCH_ROWS * n_pvs * 8 bytes, so ~3 MB at the default with 182 PVs.
BATCH_ROWS = 2048

# Fallback timestamp source: the capture path embeds ..._2026-04-24_13-11-04UTC_...
_CAPTURE_TIME_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})UTC_")


@dataclass
class Candidate:
    """A source file with the timestamp and camera parsed from its name."""

    path: str
    timestamp: datetime
    camera: str


@dataclass
class Row:
    """One fully read metadata file, ready to be written."""

    timestamp: int
    camera: str
    capture_path: str
    n_images: int
    source_file: str
    values: dict[str, float]


@dataclass
class IngestReport:
    scanned: int = 0
    ingested: int = 0
    skipped: dict[str, list[str]] = field(default_factory=dict)
    pv_count: int = 0
    cameras: Counter = field(default_factory=Counter)

    def skip(self, reason: str, path: str) -> None:
        self.skipped.setdefault(reason, []).append(path)

    @property
    def skipped_count(self) -> int:
        return sum(len(v) for v in self.skipped.values())

    def summary(self) -> str:
        lines = [
            f"scanned {self.scanned} file(s), ingested {self.ingested}, "
            f"skipped {self.skipped_count}",
            f"{self.pv_count} process variable(s), "
            f"{len(self.cameras)} camera(s): "
            + ", ".join(f"{c}={n}" for c, n in sorted(self.cameras.items())),
        ]
        for reason, paths in sorted(self.skipped.items()):
            shown = ", ".join(os.path.basename(p) for p in paths[:5])
            more = f" (+{len(paths) - 5} more)" if len(paths) > 5 else ""
            lines.append(f"  skipped [{reason}] {len(paths)}: {shown}{more}")
        return "\n".join(lines)


# --- reading source files ----------------------------------------------------


def find_source_files(root: str) -> list[str]:
    """Every ``*.hdf`` under ``root``, recursively, in a stable order."""
    if os.path.isfile(root):
        return [root]
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        found.extend(
            os.path.join(dirpath, f) for f in filenames if f.endswith(".hdf")
        )
    found.sort()
    return found


def _probe_timestamp(path: str) -> Candidate | None:
    """Recover timestamp and camera by opening a file whose name did not parse."""
    try:
        with h5py.File(path, "r") as f:
            camera = as_str(f["CamName"][()])
            capture = as_str(f["Capture"][()])
    except (OSError, KeyError):
        return None
    m = _CAPTURE_TIME_RE.search(capture)
    if m is None:
        return None
    ts = datetime.strptime(
        f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S"
    )
    return Candidate(path, ts.replace(tzinfo=timezone.utc), camera)


def build_candidates(paths: Sequence[str], report: IngestReport) -> list[Candidate]:
    """Resolve each path to a ``Candidate``, sorted by timestamp.

    The filename alone is enough for the overwhelming majority (it was verified
    to agree with the file contents for every file in the sample), so 30k files
    can be ordered without opening any of them.
    """
    candidates: list[Candidate] = []
    for path in paths:
        parsed = schema.parse_metadata_filename(os.path.basename(path))
        if parsed is not None:
            candidates.append(Candidate(path, parsed.timestamp, parsed.camera))
            continue
        probed = _probe_timestamp(path)
        if probed is None:
            report.skip("unparseable-name", path)
        else:
            log.debug("recovered timestamp for %s from its capture path", path)
            candidates.append(probed)
    candidates.sort(key=lambda c: (c.timestamp, c.path))
    return candidates


def read_metadata_file(candidate: Candidate) -> Row | tuple[str, str]:
    """Read one metadata file. Returns a ``Row``, or ``(reason, path)`` on failure.

    Failures are returned rather than raised: a 30k-file ingest must not die on
    one corrupt file. The sample directory contains exactly such a file -- a
    gzipped tar that was accidentally given a ``.hdf`` name.
    """
    try:
        with h5py.File(candidate.path, "r") as f:
            if "Capture" not in f:
                return ("missing-capture", candidate.path)
            capture = as_str(f["Capture"][()])
            camera = (
                as_str(f["CamName"][()]) if "CamName" in f else candidate.camera
            )
            values: dict[str, float] = {}
            for name, obj in f.items():
                if name in schema.NON_PV_DATASETS:
                    continue
                if not isinstance(obj, h5py.Dataset):
                    continue
                try:
                    values[name] = float(np.asarray(obj[()]).reshape(-1)[0])
                except (ValueError, TypeError, IndexError):
                    values[name] = float("nan")
    except OSError:
        return ("not-hdf5", candidate.path)
    except KeyError:
        return ("missing-dataset", candidate.path)

    return Row(
        timestamp=schema.to_epoch_us(candidate.timestamp),
        camera=camera,
        capture_path=schema.strip_capture_root(capture),
        n_images=schema.parse_capture_path(capture),
        source_file=os.path.basename(candidate.path),
        values=values,
    )


#: Reading is bound by h5py, which serialises on a global lock, so worker
#: *processes* are the only way to get any parallelism out of it: measured
#: across the sample, 8 threads run at 0.99x of serial while 8 processes run at
#: 6.3x. Start method is "spawn" rather than the platform default: the index
#: domain is already open by the time the pool starts, and a forked child that
#: closes its inherited HDF5 handle on exit can corrupt the parent's file.
_START_METHOD = "spawn"


@contextmanager
def _reader(
    workers: int,
) -> Iterator[Callable[[Sequence[Candidate]], Iterable[Any]]]:
    """Yield a mapper that runs ``read_metadata_file`` over ``workers`` processes."""
    if workers <= 1:
        yield lambda chunk: map(read_metadata_file, chunk)
        return
    context = multiprocessing.get_context(_START_METHOD)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:

        def read_many(chunk: Sequence[Candidate]) -> Iterable[Any]:
            # One task per file would spend more time in IPC than in HDF5.
            size = max(1, len(chunk) // (workers * 4))
            return pool.map(read_metadata_file, chunk, chunksize=size)

        yield read_many


def _read_batches(
    candidates: Sequence[Candidate], workers: int, report: IngestReport
) -> Iterator[list[Row]]:
    """Read files in timestamp order, in batches, dropping unreadable ones."""
    if not candidates:
        return
    with _reader(workers) as read_many:
        for start in range(0, len(candidates), BATCH_ROWS):
            chunk = candidates[start : start + BATCH_ROWS]
            rows: list[Row] = []
            for result in read_many(chunk):
                if isinstance(result, tuple):
                    report.skip(result[0], result[1])
                else:
                    rows.append(result)
            if rows:
                yield rows


def discover_pvs(candidates: Sequence[Candidate], sample_size: int = 16) -> list[str]:
    """Union of PV names across a spread-out sample of files.

    Only a seed: PVs first seen later in the run get their column created and
    back-filled on the fly, so a wrong guess here costs nothing but a few
    extra dataset creations.
    """
    if not candidates:
        return []
    step = max(1, len(candidates) // sample_size)
    names: set[str] = set()
    for candidate in candidates[::step][:sample_size]:
        result = read_metadata_file(candidate)
        if not isinstance(result, tuple):
            names.update(result.values)
    return sorted(names)


# --- writing the domain ------------------------------------------------------


class DomainWriter:
    """Creates and fills the column-oriented datasets in the index domain.

    Datasets grow to fit each batch and are never shrunk, so unreadable files
    simply never get space allocated and no trim is needed at the end. Growing
    is free on both backends; shrinking is emphatically not. HSDS flushes the
    whole domain across every data node before any decreasing resize (see
    ``dset_lib.updateShape``), which measured ~2 s per call -- trimming 187
    columns by a single row used to account for 374 s of a 380 s ingest.
    """

    def __init__(self, h5file: Any, expected: int, start_row: int = 0):
        self.f = h5file
        #: Rows we expect to write. Sizes the chunks only; allocation follows
        #: the data.
        self.expected = expected
        #: Rows currently allocated in every dataset.
        self.allocated = start_row
        self.cursor = start_row
        self.str_dtype = string_dtype(h5file)
        self.pv_keys: dict[str, str] = {}  # pv name -> dataset key
        self.pv_stats: dict[str, tuple[float, float]] = {}
        self.camera_codes: dict[str, int] = {}
        self._index: dict[str, Any] = {}

    # -- setup

    def create(self) -> None:
        self.f.require_group(schema.INDEX_GROUP)
        self.f.require_group(schema.META_GROUP)
        self.f.require_group(schema.PV_GROUP)
        self._index[schema.TIMESTAMP] = self._column(schema.TIMESTAMP, "i8")
        self._index[schema.CAMERA] = self._column(schema.CAMERA, "i2")
        self._index[schema.N_IMAGES] = self._column(schema.N_IMAGES, "i4")
        self._index[schema.CAPTURE_PATH] = self._column(
            schema.CAPTURE_PATH, self.str_dtype
        )
        self._index[schema.SOURCE_FILE] = self._column(
            schema.SOURCE_FILE, self.str_dtype
        )

    def adopt(self) -> None:
        """Attach to an existing domain so rows can be appended to it."""
        for path in (
            schema.TIMESTAMP,
            schema.CAMERA,
            schema.N_IMAGES,
            schema.CAPTURE_PATH,
            schema.SOURCE_FILE,
        ):
            self._index[path] = self.f[path]
        names = [as_str(n) for n in self.f[schema.PV_NAMES][...]]
        keys = [as_str(k) for k in self.f[schema.PV_KEYS][...]]
        self.pv_keys = dict(zip(names, keys))
        for name, key in self.pv_keys.items():
            attrs = self.f[schema.pv_dataset_path(key)].attrs
            self.pv_stats[name] = (
                float(attrs.get("min", np.inf)),
                float(attrs.get("max", -np.inf)),
            )
        self.camera_codes = {
            as_str(n): i for i, n in enumerate(self.f[schema.CAMERA_NAMES][...])
        }

    def _column(self, path: str, dtype: Any, fillvalue: Any = None) -> Any:
        kwargs: dict[str, Any] = {}
        if fillvalue is not None:
            kwargs["fillvalue"] = fillvalue
        return self.f.create_dataset(
            path,
            shape=(self.allocated,),
            maxshape=(None,),
            chunks=(min(schema.CHUNK_ROWS, max(self.expected, 1)),),
            dtype=dtype,
            **kwargs,
        )

    def ensure_pv(self, name: str) -> str:
        """Return the dataset key for ``name``, creating its column if new."""
        key = self.pv_keys.get(name)
        if key is not None:
            return key
        key = schema.sanitise_pv(name)
        path = schema.pv_dataset_path(key)
        if path not in self.f:
            # NaN fill value means rows before this PV appeared read as NaN
            # without ever being written.
            self._column(path, "f8", fillvalue=np.nan)
            parsed = schema.parse_pv_name(name)
            attrs = self.f[path].attrs
            attrs["pv_name"] = name
            for part in ("machine", "area", "family", "device", "number", "field"):
                value = getattr(parsed, part)
                if value:
                    attrs[part] = value
        self.pv_keys[name] = key
        return key

    def camera_code(self, name: str) -> int:
        code = self.camera_codes.get(name)
        if code is None:
            code = len(self.camera_codes)
            self.camera_codes[name] = code
        return code

    # -- writing

    def write_batch(self, rows: Sequence[Row]) -> None:
        lo, hi = self.cursor, self.cursor + len(rows)
        self._resize_all(hi)

        self._index[schema.TIMESTAMP][lo:hi] = np.array(
            [r.timestamp for r in rows], dtype="i8"
        )
        self._index[schema.CAMERA][lo:hi] = np.array(
            [self.camera_code(r.camera) for r in rows], dtype="i2"
        )
        self._index[schema.N_IMAGES][lo:hi] = np.array(
            [r.n_images for r in rows], dtype="i4"
        )
        self._index[schema.CAPTURE_PATH][lo:hi] = [r.capture_path for r in rows]
        self._index[schema.SOURCE_FILE][lo:hi] = [r.source_file for r in rows]

        names: set[str] = set()
        for row in rows:
            names.update(row.values)
        for name in sorted(names):
            key = self.ensure_pv(name)
            column = np.fromiter(
                (row.values.get(name, np.nan) for row in rows),
                dtype="f8",
                count=len(rows),
            )
            self.f[schema.pv_dataset_path(key)][lo:hi] = column
            finite = column[np.isfinite(column)]
            if finite.size:
                lo_stat, hi_stat = self.pv_stats.get(name, (np.inf, -np.inf))
                self.pv_stats[name] = (
                    min(lo_stat, float(finite.min())),
                    max(hi_stat, float(finite.max())),
                )
        self.cursor = hi

    def _resize_all(self, size: int) -> None:
        if size == self.allocated:
            return
        for dataset in self._index.values():
            dataset.resize((size,))
        for key in self.pv_keys.values():
            self.f[schema.pv_dataset_path(key)].resize((size,))
        self.allocated = size

    def finalise(self, source_root: str) -> None:
        """Record the lookup tables and the domain attributes.

        The resize is a no-op in the normal case -- allocation already tracks
        the cursor. It stays as a guard: leaving unwritten rows on the end
        would be a correctness bug, and being slow about it beats being wrong.
        """
        self._resize_all(self.cursor)

        names = sorted(self.pv_keys)
        self._write_string_table(schema.PV_NAMES, names)
        self._write_string_table(
            schema.PV_KEYS, [self.pv_keys[name] for name in names]
        )
        for name in names:
            stats = self.pv_stats.get(name)
            if stats and np.isfinite(stats[0]):
                attrs = self.f[schema.pv_dataset_path(self.pv_keys[name])].attrs
                attrs["min"], attrs["max"] = stats

        cameras = sorted(self.camera_codes, key=self.camera_codes.get)
        self._write_string_table(schema.CAMERA_NAMES, cameras)

        self.f.attrs["schema_version"] = schema.SCHEMA_VERSION
        self.f.attrs["created_utc"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        self.f.attrs["n_rows"] = self.cursor
        self.f.attrs["source_root"] = source_root

    def _write_string_table(self, path: str, values: Sequence[str]) -> None:
        """Create the lookup table, or resize and overwrite an existing one.

        Rewriting in place rather than deleting and recreating: h5pyd defers
        link updates to flush time, so dropping a link and recreating it under
        the same name within one session fails on close. Only --append, which
        adopts the tables a previous run wrote, ever takes the second path.
        """
        if path in self.f:
            dataset = self.f[path]
            if dataset.shape[0] != len(values):
                dataset.resize((len(values),))
        else:
            dataset = self.f.create_dataset(
                path, shape=(len(values),), maxshape=(None,), dtype=self.str_dtype
            )
        if values:
            dataset[...] = list(values)


# --- top level ---------------------------------------------------------------


def ingest(
    source: str,
    domain: str | None = None,
    append: bool = False,
    workers: int = 8,
    settings: Settings | None = None,
) -> IngestReport:
    """Ingest every metadata file under ``source`` into ``domain``."""
    settings = settings or get_settings()
    domain = domain or settings.domain
    report = IngestReport()

    paths = find_source_files(source)
    report.scanned = len(paths)
    log.info("found %d candidate file(s) under %s", len(paths), source)

    candidates = build_candidates(paths, report)

    existing_sources: set[str] = set()
    start_row = 0
    if append:
        with open_domain(domain, "r", settings) as f:
            existing_sources = {
                as_str(s) for s in f[schema.SOURCE_FILE][...]
            }
            start_row = int(f[schema.TIMESTAMP].shape[0])
            max_ts = int(f[schema.TIMESTAMP][-1]) if start_row else -1
        before = len(candidates)
        candidates = [
            c for c in candidates if os.path.basename(c.path) not in existing_sources
        ]
        if before != len(candidates):
            report.skip("already-ingested", f"{before - len(candidates)} file(s)")
        stale = [c for c in candidates if schema.to_epoch_us(c.timestamp) < max_ts]
        if stale:
            # A file that can never be read was skipped by the ingest that built
            # the index, so it is not really missing data -- and left to count as
            # stale it would block every future append. Pay the open only here,
            # on a path that is normally empty.
            unreadable = set()
            for candidate in stale:
                result = read_metadata_file(candidate)
                if isinstance(result, tuple):
                    report.skip(result[0], result[1])
                    unreadable.add(candidate.path)
            missed = [c for c in stale if c.path not in unreadable]
            if missed:
                raise SystemExit(
                    f"--append requires new rows to be newer than the existing index "
                    f"(newest row is {schema.isoformat_us(max_ts)}); "
                    f"{len(missed)} candidate(s) predate it. Re-run without --append."
                )
            candidates = [c for c in candidates if c.path not in unreadable]

    mode = "a" if append else "w"
    if not append:
        ensure_parent_folders(domain, settings)
    with open_domain(domain, mode, settings) as f:
        writer = DomainWriter(
            f, expected=start_row + len(candidates), start_row=start_row
        )
        if append:
            writer.adopt()
        else:
            writer.create()
            for name in discover_pvs(candidates):
                writer.ensure_pv(name)

        for batch in _read_batches(candidates, workers, report):
            writer.write_batch(batch)
            report.ingested += len(batch)
            for row in batch:
                report.cameras[row.camera] += 1
            log.info("wrote %d/%d rows", writer.cursor, writer.expected)

        writer.finalise(os.path.abspath(source))
        report.pv_count = len(writer.pv_keys)

    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="directory of metadata .hdf files")
    parser.add_argument("--domain", default=None, help="target HSDS domain or .h5 path")
    parser.add_argument(
        "--append",
        action="store_true",
        help="add to an existing index instead of rebuilding it",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    # h5pyd logs every HTTP request at INFO, which buries the ingest progress.
    logging.getLogger("h5pyd").setLevel(logging.WARNING)
    report = ingest(
        args.source, domain=args.domain, append=args.append, workers=args.workers
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
