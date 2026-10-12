"""Query execution against the HSDS index.

Every search runs the same funnel, cheapest stage first:

1. the time range becomes a contiguous row window by binary search, because
   rows are stored in timestamp order;
2. the camera filter is a mask over a small cached integer column;
3. each PV predicate costs exactly one hyperslab read of that window;
4. masks combine according to the request's boolean tree;
5. sort and paginate to concrete row indices;
6. only the rows on the page are hydrated with strings and selected values.

The effect is that the size of a query's network traffic tracks the *time
window* and the number of PVs mentioned, not the size of the archive.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import schema
from .backend import as_str, open_domain
from .config import Settings, get_settings
from .models import (
    CameraFilter,
    Clause,
    Group,
    Node,
    SearchRequest,
    SeriesRequest,
    Sort,
    TimeRange,
)

log = logging.getLogger("archiver_search.store")


class QueryError(ValueError):
    """A query that cannot be answered as written -- surfaced to the caller as 400."""


@dataclass
class PVInfo:
    name: str
    key: str
    parsed: schema.PVName
    min: float | None = None
    max: float | None = None


@dataclass
class Snapshot:
    """The small, hot parts of the domain, held in memory between queries."""

    n_rows: int
    timestamps: np.ndarray
    camera_codes: np.ndarray
    camera_names: list[str]
    #: Alphabetical rank of each camera code, so "sort by camera" sorts by name
    #: rather than by the order ingest happened to meet them.
    camera_rank: np.ndarray
    pvs: dict[str, PVInfo]  # canonical name -> info
    by_fold: dict[str, str]  # folded name -> canonical name
    by_device_fold: dict[str, list[str]]  # folded device name -> canonical names
    attrs: dict[str, Any]
    loaded_at: float = field(default_factory=time.monotonic)


@dataclass
class QueryStats:
    scanned_rows: int = 0
    pv_reads: int = 0


class Store:
    """Thread-safe reader over one index domain.

    A single :class:`Store` is shared by the whole API process. HSDS access is
    serialised through one file handle, but the read-side caches mean most
    queries never touch it.
    """

    def __init__(self, domain: str | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.domain = domain or self.settings.domain
        self._lock = threading.RLock()
        self._file: Any = None
        self._snapshot: Snapshot | None = None
        self._pv_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
        self._pv_cache_bytes = 0

    # -- lifecycle

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
            self._snapshot = None
            self._pv_cache.clear()
            self._pv_cache_bytes = 0

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def file(self) -> Any:
        with self._lock:
            if self._file is None:
                self._file = open_domain(self.domain, "r", self.settings)
            return self._file

    # -- snapshot

    def snapshot(self) -> Snapshot:
        """The cached index columns and PV catalogue, refreshed on a TTL."""
        with self._lock:
            snap = self._snapshot
            if snap is not None and (
                time.monotonic() - snap.loaded_at < self.settings.cache_ttl_seconds
            ):
                return snap
            if snap is not None and self._row_count() == snap.n_rows:
                snap.loaded_at = time.monotonic()
                return snap
            self._snapshot = self._load_snapshot()
            self._pv_cache.clear()
            self._pv_cache_bytes = 0
            return self._snapshot

    def _row_count(self) -> int:
        try:
            return int(self.file[schema.TIMESTAMP].shape[0])
        except (KeyError, OSError):
            return -1

    def _load_snapshot(self) -> Snapshot:
        f = self.file
        try:
            timestamps = np.asarray(f[schema.TIMESTAMP][...], dtype="i8")
        except KeyError as exc:  # pragma: no cover - misconfiguration
            raise QueryError(
                f"domain {self.domain!r} has no index; run archiver_search.ingest first"
            ) from exc
        n_rows = int(timestamps.shape[0])
        camera_codes = np.asarray(f[schema.CAMERA][...], dtype="i2")
        camera_names = [as_str(n) for n in f[schema.CAMERA_NAMES][...]]

        names = [as_str(n) for n in f[schema.PV_NAMES][...]]
        keys = [as_str(k) for k in f[schema.PV_KEYS][...]]
        pvs: dict[str, PVInfo] = {}
        by_fold: dict[str, str] = {}
        by_device_fold: dict[str, list[str]] = {}
        for name, key in zip(names, keys):
            attrs = f[schema.pv_dataset_path(key)].attrs
            parsed = schema.parse_pv_name(name)
            info = PVInfo(
                name=name,
                key=key,
                parsed=parsed,
                min=float(attrs["min"]) if "min" in attrs else None,
                max=float(attrs["max"]) if "max" in attrs else None,
            )
            pvs[name] = info
            by_fold[schema.fold_pv(name)] = name
            by_fold[schema.fold_pv(key)] = name
            device = parsed.device_name
            if device:
                by_device_fold.setdefault(schema.fold_pv(device), []).append(name)

        return Snapshot(
            n_rows=n_rows,
            timestamps=timestamps,
            camera_codes=camera_codes,
            camera_names=camera_names,
            camera_rank=np.argsort(np.argsort(camera_names, kind="stable")),
            pvs=pvs,
            by_fold=by_fold,
            by_device_fold=by_device_fold,
            attrs={k: _plain(v) for k, v in f.attrs.items()},
        )

    # -- PV lookup

    def resolve_pv(self, token: str) -> PVInfo:
        """Find the PV a user-supplied name refers to.

        Accepts the canonical ``CLA-S04-MAG-QUAD-01:SETI``, the sanitised
        ``CLA_S04_MAG_QUAD_01__SETI``, any mixture of ``-``/``_``, and -- when
        it is unambiguous -- a bare device name with the field left off, which
        is how people usually write them.
        """
        snap = self.snapshot()
        folded = schema.fold_pv(token)
        name = snap.by_fold.get(folded)
        if name is not None:
            return snap.pvs[name]
        candidates = snap.by_device_fold.get(folded)
        if candidates:
            if len(candidates) == 1:
                return snap.pvs[candidates[0]]
            raise QueryError(
                f"{token!r} is ambiguous; it matches "
                + ", ".join(sorted(candidates))
            )
        raise QueryError(f"unknown PV {token!r}")

    def match_pvs(self, pattern: str) -> list[PVInfo]:
        """PVs matching a glob, insensitive to case and ``-``/``_`` spelling."""
        snap = self.snapshot()
        folded = schema.fold_pv(pattern)
        matched = [
            info
            for name, info in snap.pvs.items()
            if fnmatch.fnmatchcase(schema.fold_pv(name), folded)
        ]
        if not matched:
            raise QueryError(f"no PV matches pattern {pattern!r}")
        return sorted(matched, key=lambda i: i.name)

    def list_pvs(
        self,
        q: str | None = None,
        area: str | None = None,
        family: str | None = None,
        field_: str | None = None,
    ) -> list[PVInfo]:
        snap = self.snapshot()
        out = []
        needle = schema.fold_pv(q) if q else None
        for info in snap.pvs.values():
            p = info.parsed
            if area and (p.area or "").upper() != area.upper():
                continue
            if family and (p.family or "").upper() != family.upper():
                continue
            if field_ and (p.field or "").lower() != field_.lower():
                continue
            if needle and needle not in schema.fold_pv(info.name):
                continue
            out.append(info)
        return sorted(out, key=lambda i: i.name)

    # -- column reads

    def read_pv_window(self, info: PVInfo, lo: int, hi: int, stats: QueryStats) -> np.ndarray:
        """Values of one PV over rows ``[lo, hi)``, cached."""
        key = (info.key, lo, hi)
        with self._lock:
            hit = self._pv_cache.get(key)
            if hit is not None:
                self._pv_cache.move_to_end(key)
                return hit
        stats.pv_reads += 1
        data = np.asarray(
            self.file[schema.pv_dataset_path(info.key)][lo:hi], dtype="f8"
        )
        self._cache_put(key, data)
        return data

    def read_pv_at(
        self, info: PVInfo, rows: np.ndarray, stats: QueryStats
    ) -> np.ndarray:
        """Values of one PV at scattered row indices.

        Prefers an already-cached window covering those rows -- after filtering,
        the window a clause read is usually still in the cache, so hydrating the
        page costs nothing.
        """
        lo, hi = int(rows.min()), int(rows.max()) + 1
        with self._lock:
            for (key, c_lo, c_hi), data in reversed(self._pv_cache.items()):
                if key == info.key and c_lo <= lo and c_hi >= hi:
                    self._pv_cache.move_to_end((key, c_lo, c_hi))
                    return data[rows - c_lo]
        return self.read_pv_window(info, lo, hi, stats)[rows - lo]

    def _cache_put(self, key: tuple[str, int, int], data: np.ndarray) -> None:
        budget = self.settings.pv_cache_mib * 1024 * 1024
        with self._lock:
            self._pv_cache[key] = data
            self._pv_cache_bytes += data.nbytes
            while self._pv_cache_bytes > budget and len(self._pv_cache) > 1:
                _, evicted = self._pv_cache.popitem(last=False)
                self._pv_cache_bytes -= evicted.nbytes

    def _read_rows(self, path: str, rows: np.ndarray) -> list[Any]:
        """Values of a column at scattered row indices.

        A contiguous slice beats many small reads until the rows get sparse, so
        pick whichever transfers less.
        """
        if rows.size == 0:
            return []
        dataset = self.file[path]
        lo, hi = int(rows.min()), int(rows.max()) + 1
        span = hi - lo
        if span <= max(rows.size * 4, 256):
            block = dataset[lo:hi]
            return [block[i - lo] for i in rows]
        return [dataset[int(i)] for i in rows]

    # -- the funnel

    def _window(self, snap: Snapshot, when: TimeRange) -> tuple[int, int]:
        """Rows falling in ``[from, to)``, as a contiguous slice."""
        lo = 0
        hi = snap.n_rows
        if when.from_ is not None:
            lo = int(np.searchsorted(snap.timestamps, _us(when.from_), side="left"))
        if when.to is not None:
            hi = int(np.searchsorted(snap.timestamps, _us(when.to), side="left"))
        return lo, max(lo, hi)

    def _camera_mask(
        self, snap: Snapshot, cameras: CameraFilter, lo: int, hi: int
    ) -> np.ndarray | None:
        if cameras.is_empty():
            return None
        wanted: set[int] = set()
        unknown: list[str] = []
        for name in cameras.names:
            folded = name.strip().upper()
            hits = [
                code
                for code, cam in enumerate(snap.camera_names)
                if cam.upper() == folded
            ]
            if hits:
                wanted.update(hits)
            else:
                unknown.append(name)
        for area in cameras.areas:
            folded = area.strip().upper()
            hits = [
                code
                for code, cam in enumerate(snap.camera_names)
                if (schema.parse_device_name(cam).area or "").upper() == folded
            ]
            if hits:
                wanted.update(hits)
            else:
                unknown.append(f"area {area}")
        if unknown:
            raise QueryError(
                "unknown camera(s): "
                + ", ".join(unknown)
                + "; known: "
                + ", ".join(sorted(snap.camera_names))
            )
        if not wanted:
            return np.zeros(hi - lo, dtype=bool)
        return np.isin(snap.camera_codes[lo:hi], np.fromiter(wanted, dtype="i2"))

    def _clause_mask(
        self, clause: Clause, lo: int, hi: int, stats: QueryStats
    ) -> np.ndarray:
        if clause.pv is not None:
            infos = [self.resolve_pv(clause.pv)]
        else:
            infos = self.match_pvs(clause.pv_pattern or "")

        masks = [
            _apply_op(self.read_pv_window(info, lo, hi, stats), clause)
            for info in infos
        ]
        if len(masks) == 1:
            return masks[0]
        combined = masks[0].copy()
        for mask in masks[1:]:
            if clause.match == "any":
                combined |= mask
            else:
                combined &= mask
        return combined

    def _node_mask(
        self, node: Node, lo: int, hi: int, stats: QueryStats
    ) -> np.ndarray:
        if isinstance(node, Clause):
            return self._clause_mask(node, lo, hi, stats)
        if not node.clauses:
            return np.ones(hi - lo, dtype=bool)
        masks = [self._node_mask(child, lo, hi, stats) for child in node.clauses]
        combined = masks[0].copy()
        for mask in masks[1:]:
            if node.op == "and":
                combined &= mask
            else:
                combined |= mask
        return combined

    def select_rows(
        self, request: SearchRequest | SeriesRequest, stats: QueryStats
    ) -> tuple[np.ndarray, Snapshot]:
        """Absolute indices of every row matching the request, in storage order."""
        snap = self.snapshot()
        lo, hi = self._window(snap, request.time)
        stats.scanned_rows = hi - lo
        if hi <= lo:
            return np.empty(0, dtype="i8"), snap

        mask = np.ones(hi - lo, dtype=bool)
        camera_mask = self._camera_mask(snap, request.cameras, lo, hi)
        if camera_mask is not None:
            mask &= camera_mask
        if request.filters is not None:
            # Prefetch the PV windows concurrently: over HSDS these are
            # independent HTTP round trips and latency dominates.
            self._prefetch(request.filters, lo, hi, stats)
            mask &= self._node_mask(request.filters, lo, hi, stats)

        return (np.flatnonzero(mask) + lo).astype("i8"), snap

    def _prefetch(self, node: Node, lo: int, hi: int, stats: QueryStats) -> None:
        infos: list[PVInfo] = []
        _collect_pvs(node, self, infos)
        with self._lock:
            pending = [i for i in infos if (i.key, lo, hi) not in self._pv_cache]
        if len(pending) < 2:
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.settings.read_workers) as pool:
            list(pool.map(lambda i: self.read_pv_window(i, lo, hi, stats), pending))

    # -- public queries

    def search(self, request: SearchRequest) -> dict[str, Any]:
        started = time.perf_counter()
        stats = QueryStats()
        rows, snap = self.select_rows(request, stats)
        total = int(rows.size)

        page = _paginate(rows, snap, request.sort, request.offset, request.limit)
        out = self._hydrate(page, snap, request.select, stats)

        return {
            "total": total,
            "limit": request.limit,
            "offset": request.offset,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "scanned_rows": stats.scanned_rows,
            "pv_reads": stats.pv_reads,
            "rows": out,
        }

    def iter_rows(
        self, request: SearchRequest, chunk: int = 2000
    ) -> Iterable[list[dict[str, Any]]]:
        """All matching rows in pages, for streaming exports."""
        stats = QueryStats()
        rows, snap = self.select_rows(request, stats)
        rows = _sorted_rows(rows, snap, request.sort)
        limit = min(rows.size, self.settings.max_export_rows)
        for start in range(0, limit, chunk):
            block = rows[start : start + chunk]
            yield self._hydrate(block, snap, request.select, stats)

    def row(self, row_id: int) -> dict[str, Any]:
        """One row with every PV value attached."""
        snap = self.snapshot()
        if not 0 <= row_id < snap.n_rows:
            raise QueryError(f"row {row_id} is out of range (0..{snap.n_rows - 1})")
        stats = QueryStats()
        rows = np.array([row_id], dtype="i8")
        out = self._hydrate(rows, snap, list(snap.pvs), stats)[0]
        out["has_image"] = self.image_path(out["capture_path"]) is not None
        return out

    def capture_path(self, row_id: int) -> str:
        """One row's capture path, without hydrating its 182 PV values."""
        snap = self.snapshot()
        if not 0 <= row_id < snap.n_rows:
            raise QueryError(f"row {row_id} is out of range (0..{snap.n_rows - 1})")
        stored = self._read_rows(schema.CAPTURE_PATH, np.array([row_id], dtype="i8"))
        return as_str(stored[0])

    def image_path(self, capture_path: str) -> Path | None:
        """Locate a capture file under ``image_root``, or ``None`` if it is not there.

        ``None`` covers every uninteresting case -- no root configured, the
        archive not mounted here, that capture not copied across -- because to
        a caller they all mean the same thing: no image to show.
        """
        root = self.settings.image_root
        if not root:
            return None
        base = Path(root)
        candidate = base / schema.strip_capture_root(capture_path)
        try:
            resolved = candidate.resolve()
            # strip_capture_root cannot produce a leading "/", but it can leave
            # a ".." in a path that came from somewhere unexpected.
            resolved.relative_to(base.resolve())
        except (OSError, ValueError):
            return None
        return resolved if resolved.is_file() else None

    def series(self, request: SeriesRequest) -> dict[str, Any]:
        started = time.perf_counter()
        stats = QueryStats()
        rows, snap = self.select_rows(request, stats)

        infos: dict[str, PVInfo] = {}
        for token in request.pvs:
            info = self.resolve_pv(token)
            infos[info.name] = info
        if request.pv_pattern:
            for info in self.match_pvs(request.pv_pattern):
                infos[info.name] = info

        out = []
        if rows.size:
            times = snap.timestamps[rows]
            keep, decimated = _decimate(times.size, request.max_points)
            stamps = [schema.isoformat_us(int(t)) for t in times[keep]]
            for name, info in sorted(infos.items()):
                values = self.read_pv_at(info, rows, stats)[keep]
                out.append(
                    {
                        "pv": name,
                        "t": stamps,
                        "v": [None if not np.isfinite(v) else float(v) for v in values],
                        "decimated": decimated,
                    }
                )
        else:
            out = [
                {"pv": name, "t": [], "v": [], "decimated": False}
                for name in sorted(infos)
            ]

        return {
            "total": int(rows.size),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "series": out,
        }

    def meta(self) -> dict[str, Any]:
        snap = self.snapshot()
        return {
            "domain": self.domain,
            "schema_version": int(snap.attrs.get("schema_version", 0)),
            "created_utc": snap.attrs.get("created_utc"),
            "n_rows": snap.n_rows,
            "pv_count": len(snap.pvs),
            "time_from": (
                schema.isoformat_us(int(snap.timestamps[0])) if snap.n_rows else None
            ),
            "time_to": (
                schema.isoformat_us(int(snap.timestamps[-1])) if snap.n_rows else None
            ),
            "cameras": self.cameras(),
        }

    def cameras(self) -> list[dict[str, Any]]:
        snap = self.snapshot()
        out = []
        for code, name in enumerate(snap.camera_names):
            where = np.flatnonzero(snap.camera_codes == code)
            out.append(
                {
                    "name": name,
                    "area": schema.parse_device_name(name).area,
                    "n_rows": int(where.size),
                    "first": (
                        schema.isoformat_us(int(snap.timestamps[where[0]]))
                        if where.size
                        else None
                    ),
                    "last": (
                        schema.isoformat_us(int(snap.timestamps[where[-1]]))
                        if where.size
                        else None
                    ),
                }
            )
        return sorted(out, key=lambda c: c["name"])

    # -- hydration

    def _hydrate(
        self,
        rows: np.ndarray,
        snap: Snapshot,
        select: Sequence[str],
        stats: QueryStats,
    ) -> list[dict[str, Any]]:
        if rows.size == 0:
            return []
        captures = self._read_rows(schema.CAPTURE_PATH, rows)
        sources = self._read_rows(schema.SOURCE_FILE, rows)
        n_images = self._read_rows(schema.N_IMAGES, rows)

        selected: dict[str, np.ndarray] = {}
        infos: dict[str, PVInfo] = {}
        for token in select:
            info = self.resolve_pv(token)
            infos[info.name] = info
        for name, info in infos.items():
            selected[name] = self.read_pv_at(info, rows, stats)

        out = []
        for position, row_id in enumerate(rows):
            camera = snap.camera_names[int(snap.camera_codes[row_id])]
            count = int(n_images[position])
            out.append(
                {
                    "id": int(row_id),
                    "timestamp": schema.isoformat_us(int(snap.timestamps[row_id])),
                    "camera": camera,
                    "area": schema.parse_device_name(camera).area,
                    "capture_path": as_str(captures[position]),
                    "n_images": count,
                    "source_file": as_str(sources[position]),
                    "values": {
                        name: (
                            None
                            if not np.isfinite(values[position])
                            else float(values[position])
                        )
                        for name, values in selected.items()
                    },
                }
            )
        return out


# --- helpers -----------------------------------------------------------------


def _us(when: datetime) -> int:
    return schema.to_epoch_us(when)


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_pvs(node: Node, store: Store, out: list[PVInfo]) -> None:
    if isinstance(node, Clause):
        if node.pv is not None:
            out.append(store.resolve_pv(node.pv))
        elif node.pv_pattern:
            out.extend(store.match_pvs(node.pv_pattern))
    else:
        for child in node.clauses:
            _collect_pvs(child, store, out)


def _apply_op(values: np.ndarray, clause: Clause) -> np.ndarray:
    op = clause.op
    if op == "isnan":
        return ~np.isfinite(values)
    if op == "notnan":
        return np.isfinite(values)

    with np.errstate(invalid="ignore"):
        if op == "eq":
            return np.abs(values - float(clause.value)) <= clause.tol
        if op == "ne":
            return ~(np.abs(values - float(clause.value)) <= clause.tol)
        if op == "lt":
            return values < float(clause.value)
        if op == "lte":
            return values <= float(clause.value)
        if op == "gt":
            return values > float(clause.value)
        if op == "gte":
            return values >= float(clause.value)
        if op == "between":
            low, high = sorted(float(v) for v in clause.value)
            return (values >= low) & (values <= high)
        if op == "outside":
            low, high = sorted(float(v) for v in clause.value)
            return (values < low) | (values > high)
        if op == "in":
            wanted = np.array([float(v) for v in clause.value], dtype="f8")
            return np.any(
                np.abs(values[:, None] - wanted[None, :]) <= clause.tol, axis=1
            )
    raise QueryError(f"unsupported operator {op!r}")


def _sorted_rows(rows: np.ndarray, snap: Snapshot, sort: Sort) -> np.ndarray:
    if rows.size == 0:
        return rows
    if sort.by == "camera":
        # Ties within a camera stay in timestamp order, since `rows` arrives in
        # storage order and the sort is stable.
        order = np.argsort(snap.camera_rank[snap.camera_codes[rows]], kind="stable")
    else:
        order = np.argsort(snap.timestamps[rows], kind="stable")
    ordered = rows[order]
    return ordered[::-1] if sort.order == "desc" else ordered


def _paginate(
    rows: np.ndarray, snap: Snapshot, sort: Sort, offset: int, limit: int
) -> np.ndarray:
    ordered = _sorted_rows(rows, snap, sort)
    return ordered[offset : offset + limit]


def _decimate(n: int, max_points: int) -> tuple[np.ndarray, bool]:
    """Indices to keep so a series is at most ``max_points`` long."""
    if n <= max_points:
        return np.arange(n), False
    return np.unique(np.linspace(0, n - 1, max_points).astype(int)), True
