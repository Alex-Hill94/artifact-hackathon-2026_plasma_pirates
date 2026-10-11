"""Real CLARA camera captures from the CC-IN2P3 HDF5 archive, one row per capture.

Each HDF5 file holds Capture000001..N: a uint16 camera frame with the machine's magnet
PVs as attributes. A row pairs the capture's SETI/READI currents with the beam spot read
from the frame (background-subtracted centroid and RMS size, in mm on the chip) and a
small false-colour thumbnail as a JPEG data URI, which the Hopsworks data preview shows
as an image. The frames are read in place from the archive mount; only the row is written.

    # one scan to parquet, no cluster
    python camera_pipeline.py --select "2026/3/19/S07 Quad 03 Scans" --out data/

    # straight into the clara_camera_captures feature group
    python camera_pipeline.py --select "2026/3/19/S07 Quad 03 Scans" --insert
"""

import argparse
import base64
import io
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage

FEATURE_GROUP = "clara_camera_captures"
ARCHIVE = "/m4cast/optimisation/extracted/var/lib/docker/volumes/shared-data/_data/claraserv3"
FILE_NAME = re.compile(r"^(?P<camera>[A-Z0-9-]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})UTC_.*\.hdf$")
CURRENTS = ("SETI", "READI")
THUMB_BIN = 6
# A pixel belongs to the spot when the 5x5-smoothed frame clears both this many noise sigmas
# above the pedestal and this fraction of the smoothed peak. The frames are 12-bit stored in
# 16 bits (steps of 16 counts), so the MAD of the dark area is often 0 and the sigma cut alone
# would take in the whole background.
THRESHOLD_SIGMA = 5.0
THRESHOLD_PEAK_FRACTION = 0.05
# Fewer lit pixels than this is a dark frame, not a beam.
MIN_LIT_PIXELS = 50
SATURATION = 65504


def column(pv: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", pv.lower()).strip("_")


def text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def event_ts(stamp: str, year: int) -> datetime:
    # The camera stamp ("Mar 19 15:50:09.970301") carries no year; the file name does.
    return datetime.strptime(f"{year} {stamp}", "%Y %b %d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


def spot(frame: np.ndarray, pix_mm: float) -> dict:
    """Pedestal, noise and the beam's first and second moments, centred on the chip."""
    pedestal = float(np.median(frame))
    noise = max(1.4826 * float(np.median(np.abs(frame - pedestal))), 1.0)
    signal = np.clip(frame.astype(np.float64) - pedestal, 0, None)
    smooth = ndimage.uniform_filter(signal, 5)
    threshold = max(THRESHOLD_SIGMA * noise, THRESHOLD_PEAK_FRACTION * float(smooth.max()))
    weights = np.where(smooth > threshold, signal, 0.0)
    lit = int(np.count_nonzero(weights))
    out = {
        "pedestal": pedestal,
        "noise": noise,
        "threshold": threshold,
        "lit_pixels": lit,
        "saturated_pixels": int(np.count_nonzero(frame >= SATURATION)),
        "peak": float(signal.max()),
        "signal_sum": float(weights.sum()),
    }
    if lit < MIN_LIT_PIXELS:
        return out | {"beam_present": False, "mu_x_mm": None, "mu_y_mm": None, "sigma_x_mm": None, "sigma_y_mm": None}
    h, w = frame.shape
    ys, xs = np.indices(frame.shape)
    total = weights.sum()
    mx, my = (weights * xs).sum() / total, (weights * ys).sum() / total
    sx = np.sqrt((weights * (xs - mx) ** 2).sum() / total)
    sy = np.sqrt((weights * (ys - my) ** 2).sum() / total)
    return out | {
        "beam_present": True,
        "mu_x_mm": float((mx - (w - 1) / 2) * pix_mm),
        "mu_y_mm": float((my - (h - 1) / 2) * pix_mm),
        "sigma_x_mm": float(sx * pix_mm),
        "sigma_y_mm": float(sy * pix_mm),
    }


def thumbnail(frame: np.ndarray, pedestal: float) -> str:
    """Block-mean downsample, square-root stretch to the 99.9th percentile, inferno, JPEG."""
    h, w = (s - s % THUMB_BIN for s in frame.shape)
    small = frame[:h, :w].reshape(h // THUMB_BIN, THUMB_BIN, w // THUMB_BIN, THUMB_BIN).mean(axis=(1, 3)) - pedestal
    top = max(float(np.percentile(small, 99.9)), 1.0)
    rgb = colormaps["inferno"](np.sqrt(np.clip(small / top, 0, 1)), bytes=True)[..., :3]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def read_file(path: str) -> list[dict]:
    name = FILE_NAME.match(os.path.basename(path))
    if not name:
        return []
    camera, year = name["camera"], int(name["date"][:4])
    rel = os.path.relpath(path, ARCHIVE)
    scan = os.path.dirname(rel).split(os.sep)[3:]  # past YYYY/M/D
    rows = []
    with h5py.File(path, "r") as f:
        for capture in sorted(f.keys()):
            ds = f[capture]
            attrs = ds.attrs
            frame = ds[()]
            pix_mm = float(attrs.get(f"{camera}:ANA:PixMM_RBV", np.nan))
            reading = spot(frame, pix_mm)
            row = {
                "capture_id": f"{os.path.splitext(os.path.basename(path))[0]}_{capture}",
                "event_ts": event_ts(text(attrs["Event Timestamp"]), year),
                "camera": camera,
                "scan": "/".join(scan),
                "source_file": rel,
                "capture": capture,
                "camera_model": text(attrs.get("Model", "")),
                "width": frame.shape[1],
                "height": frame.shape[0],
                "pix_mm": pix_mm,
            } | reading
            row["thumbnail"] = thumbnail(frame, reading["pedestal"])
            for pv, value in attrs.items():
                device, _, field = pv.partition(":")
                if field in CURRENTS and "-MAG-" in device:
                    row[column(pv)] = float(value)
            rows.append(row)
    return rows


def captures(select: str, limit: int | None, workers: int) -> pd.DataFrame:
    root = Path(ARCHIVE) / select
    files = sorted(str(p) for p in root.rglob("*.hdf")) if root.is_dir() else [str(root)]
    files = files[:limit] if limit else files
    print(f"{len(files)} files under {root}")
    with ProcessPoolExecutor(workers) as pool:
        rows = [row for file_rows in pool.map(read_file, files) for row in file_rows]
    return pd.DataFrame(rows)


def insert(df: pd.DataFrame, version: int) -> None:
    import hopsworks

    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP,
        version=version,
        primary_key=["capture_id"],
        event_time="event_ts",
        partition_key=["camera"],
        online_enabled=False,
        description="Real CLARA camera captures (CC-IN2P3 archive): magnet SETI/READI, beam spot moments, thumbnail",
        statistics_config={"columns": [c for c in df.columns if c != "thumbnail"]},
    )
    if fg.id is not None:
        # Magnet PVs differ between files; the group keeps the columns of its first insert.
        known = [f.name for f in fg.features]
        dropped = sorted(set(df.columns) - set(known))
        if dropped:
            print(f"dropping {len(dropped)} columns the group does not have: {dropped[:5]}...")
        df = df.reindex(columns=known)
    fg.insert(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Hopsworks splits job arguments on spaces and ignores quotes, and the operators' scan
    # folders have spaces in their names, so the words are joined back together here.
    ap.add_argument("--select", required=True, nargs="+", help=f"file or directory under {ARCHIVE}")
    ap.add_argument("--limit", type=int, help="first N files only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, help="directory to write captures.parquet into")
    ap.add_argument("--insert", action="store_true", help=f"insert into the {FEATURE_GROUP} feature group")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    df = captures(" ".join(args.select), args.limit, args.workers)
    print(f"{len(df)} captures, {int(df.beam_present.sum())} with a beam, {df.filter(regex='_seti$').shape[1]} SETI columns")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out / "captures.parquet", index=False)
    if args.insert:
        insert(df, args.version)
    if not (args.out or args.insert):
        print(df.drop(columns="thumbnail").head().to_string())


if __name__ == "__main__":
    main()
