#!/bin/bash
# CLARA camera archive to two small parquet files, run as a SLURM job on CC-IN2P3 next to the HDF5 files.
#
#   args: --archive <claraserv3 dir> --out <results dir> [--limit N] [--workers N]
#
# captures.parquet   one row per capture: spot moments in mm and pixels (same algorithm as
#                    ../camera_pipeline.py), frame and camera metadata, every SETI/READI/GETSETI/RPOWER PV
# files.parquet      one row per HDF5 file (one burst at fixed settings): moments averaged over the beam
#                    captures and their shot-to-shot spread, PV medians, thumbnail of the mean frame
#
# Written to <out>/.tmp first and renamed, so the ingestion job never sees a half-written file.
set -eo pipefail
source /cvmfs/sft.cern.ch/lcg/views/LCG_110a/x86_64-el9-gcc13-opt/setup.sh
set -u
exec python3 - "$@" <<'PYEOF'
import argparse, base64, io, json, os, re, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage

FILE_NAME = re.compile(r"^(?P<camera>[A-Z0-9-]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})UTC_.*\.hdf$")
PV_FIELDS = ("SETI", "READI", "GETSETI", "RPOWER")
THRESHOLD_SIGMA = 5.0
THRESHOLD_PEAK_FRACTION = 0.05
MIN_LIT_PIXELS = 50
# 12-bit frames are stored in the top bits of uint16; their full scale reads 65504.
SATURATION = {np.dtype("uint16"): 65504, np.dtype("uint8"): 255}
THUMB_BIN = 10
MOMENTS = ["mu_x_mm", "mu_y_mm", "sigma_x_mm", "sigma_y_mm", "mu_x_px", "mu_y_px", "sigma_x_px", "sigma_y_px"]


def column(pv):
    return re.sub(r"[^a-z0-9]+", "_", pv.lower()).strip("_")


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def stamp(attrs, year, fallback):
    try:
        # The camera stamp ("Mar 19 15:50:09.970301") carries no year; the file name does.
        return datetime.strptime(f"{year} {text(attrs['Event Timestamp'])}", "%Y %b %d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
        return fallback


def spot(frame, pix_mm):
    pedestal = float(np.median(frame))
    noise = max(1.4826 * float(np.median(np.abs(frame - pedestal))), 1.0)
    signal = np.clip(frame.astype(np.float64) - pedestal, 0, None)
    smooth = ndimage.uniform_filter(signal, 5)
    threshold = max(THRESHOLD_SIGMA * noise, THRESHOLD_PEAK_FRACTION * float(smooth.max()))
    weights = np.where(smooth > threshold, signal, 0.0)
    lit = int(np.count_nonzero(weights))
    out = {"pedestal": pedestal, "noise": noise, "threshold": threshold, "lit_pixels": lit,
           "saturated_pixels": int(np.count_nonzero(frame >= SATURATION.get(frame.dtype, np.iinfo(frame.dtype).max))),
           "peak": float(signal.max()), "signal_sum": float(weights.sum())}
    if lit < MIN_LIT_PIXELS:
        return out | {"beam_present": False} | {m: None for m in MOMENTS}
    h, w = frame.shape
    total = weights.sum()
    wx, wy = weights.sum(axis=0), weights.sum(axis=1)
    xs, ys = np.arange(w), np.arange(h)
    mx, my = (wx * xs).sum() / total, (wy * ys).sum() / total
    sx = np.sqrt((wx * (xs - mx) ** 2).sum() / total)
    sy = np.sqrt((wy * (ys - my) ** 2).sum() / total)
    cx, cy = mx - (w - 1) / 2, my - (h - 1) / 2
    mm = pix_mm if np.isfinite(pix_mm) else None
    return out | {"beam_present": True, "mu_x_px": float(cx), "mu_y_px": float(cy),
                  "sigma_x_px": float(sx), "sigma_y_px": float(sy),
                  "mu_x_mm": cx * mm if mm else None, "mu_y_mm": cy * mm if mm else None,
                  "sigma_x_mm": sx * mm if mm else None, "sigma_y_mm": sy * mm if mm else None}


def thumbnail(mean_frame):
    h, w = (s - s % THUMB_BIN for s in mean_frame.shape)
    small = mean_frame[:h, :w].reshape(h // THUMB_BIN, THUMB_BIN, w // THUMB_BIN, THUMB_BIN).mean(axis=(1, 3))
    small = small - np.median(small)
    top = max(float(np.percentile(small, 99.9)), 1.0)
    rgb = colormaps["inferno"](np.sqrt(np.clip(small / top, 0, 1)), bytes=True)[..., :3]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def read_file(path, archive):
    name = FILE_NAME.match(os.path.basename(path))
    camera, year = name["camera"], int(name["date"][:4])
    file_ts = datetime.strptime(f"{name['date']} {name['time']}", "%Y-%m-%d %H-%M-%S").replace(tzinfo=timezone.utc)
    file_id = os.path.splitext(os.path.basename(path))[0]
    rel = os.path.relpath(path, archive)
    scan = "/".join(os.path.dirname(rel).split(os.sep)[3:])  # past YYYY/M/D
    rows, acc = [], None
    with h5py.File(path, "r") as f:
        names = sorted(k for k in f.keys() if k.startswith("Capture"))
        for i, capture in enumerate(names, 1):
            ds = f[capture]
            attrs, frame = ds.attrs, ds[()]
            pix_mm = float(attrs.get(f"{camera}:ANA:PixMM_RBV", np.nan))
            row = {"capture_id": f"{file_id}_{capture}", "event_ts": stamp(attrs, year, file_ts),
                   "file_id": file_id, "camera": camera, "scan": scan, "source_file": rel, "capture": capture,
                   "capture_index": i, "n_captures": len(names), "camera_model": text(attrs.get("Model", "")),
                   "serial_number": text(attrs.get("Serial Number", "")), "bit_depth": frame.dtype.itemsize * 8,
                   "width": frame.shape[1], "height": frame.shape[0],
                   "pix_mm": pix_mm if np.isfinite(pix_mm) else None} | spot(frame, pix_mm)
            for pv, value in attrs.items():
                if pv.partition(":")[2] in PV_FIELDS:
                    try:
                        row[column(pv)] = float(value)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
            acc = frame.astype(np.float64) if acc is None else acc + frame if acc.shape == frame.shape else acc
    df = pd.DataFrame(rows)
    beam = df[df.beam_present]
    pv_cols = [c for c in df.columns if c.split("_")[-1] in ("seti", "readi", "getseti", "rpower")]
    file_row = {"file_id": file_id, "event_ts": df.event_ts.min(), "camera": camera, "scan": scan,
                "source_file": rel, "n_captures": len(df), "n_beam": len(beam), "pix_mm": df.pix_mm.iloc[0],
                "width": int(df.width.iloc[0]), "height": int(df.height.iloc[0]), "bit_depth": int(df.bit_depth.iloc[0])}
    for m in MOMENTS:
        values = pd.to_numeric(beam[m], errors="coerce")
        file_row[f"{m}_mean"] = values.mean() if len(beam) else None
        file_row[f"{m}_std"] = values.std() if len(beam) > 1 else None
    for c in pv_cols:
        file_row[c] = df[c].median()
    file_row["thumbnail"] = thumbnail(acc / len(df)) if acc is not None else None
    return rows, file_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="N files spread over the archive, for a test run")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()

    files = []
    for root, _, names in os.walk(args.archive):
        files += [os.path.join(root, n) for n in names if FILE_NAME.match(n)]
    files.sort()
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][: args.limit]
    sizes = {f: os.path.getsize(f) for f in files}
    files.sort(key=lambda f: -sizes[f])
    print(f"{len(files)} files, {sum(sizes.values()) / 1e12:.2f} TB, {args.workers} workers", flush=True)

    t0, captures, file_rows, failed = time.time(), [], [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(read_file, f, args.archive): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows, file_row = fut.result()
                captures += rows
                file_rows.append(file_row)
            except Exception as e:
                failed.append({"file": os.path.relpath(futures[fut], args.archive), "error": repr(e)[:300]})
            if i % 200 == 0 or i == len(files):
                print(f"{i}/{len(files)} files, {len(captures)} captures, {len(failed)} failed, "
                      f"{time.time() - t0:.0f} s", flush=True)

    tmp = os.path.join(args.out, ".tmp")
    os.makedirs(tmp, exist_ok=True)
    pd.DataFrame(captures).to_parquet(os.path.join(tmp, "captures.parquet"), index=False, compression="snappy")  # LCG pyarrow is built without zstd
    pd.DataFrame(file_rows).to_parquet(os.path.join(tmp, "files.parquet"), index=False, compression="snappy")  # LCG pyarrow is built without zstd
    manifest = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "files": len(files), "captures": len(captures),
                "file_rows": len(file_rows), "failed": failed, "seconds": round(time.time() - t0)}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    for name in ("captures.parquet", "files.parquet", "manifest.json"):
        os.replace(os.path.join(tmp, name), os.path.join(args.out, name))
    os.rmdir(tmp)
    print(json.dumps({k: v for k, v in manifest.items() if k != "failed"} | {"failed": len(failed)}), flush=True)


if __name__ == "__main__":
    main()
PYEOF
