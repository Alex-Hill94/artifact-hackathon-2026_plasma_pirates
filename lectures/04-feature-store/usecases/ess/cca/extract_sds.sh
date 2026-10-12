#!/bin/bash
# ESS SDS captures to two small parquet files, run as a SLURM job on CC-IN2P3 next to the raw data.
#
#   args: --raw <dir of SDS_Data_*.tar> --out <results dir> [--limit N] [--workers N]
#
# llrf_waveforms.parquet   the public sample's schema for every event: one row per (event, cycle, RF chain)
#                          on the DIG-101:Dwn0 chain of every MBL, HBL and spoke section, I = Cmp0 and
#                          Q = Cmp1 point-sampled to 256 values as the sample was, plus capture metadata
# channel_stats.parquet    every RF chain of every cycle at full resolution: per component mean, std, min,
#                          max, ptp, skew, kurtosis, derivative std and max, dominant FFT bin and magnitude
#
# Parquet members are read in place inside the tars, nothing is extracted. Written to <out>/.tmp first
# and renamed, so the ingestion job never sees a half-written file.
set -euo pipefail
PY=/pbs/software/redhat-9-x86_64/anaconda/3.11/bin/python3
exec "$PY" - "$@" <<'PYEOF'
import argparse, gc, io, json, multiprocessing, os, re, tarfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RF_MEMBERS = ("MBL.parquet", "HBL.parquet", "SPOKES.parquet")
WAVE_CHAIN = re.compile(r":RFS-DIG-101:Dwn0-Cmp[01]")
CMP = re.compile(r"-Cmp([01])")
POINTS = 256
META = ["trig_code", "trig_user_tag", "trig_name", "beam_mode", "beam_state", "beam_present",
        "beam_current", "beam_energy", "beam_dest", "ctype", "cycle_ts", "event_ts"]
COLUMNS = ["pv", "cycle_id", "event_id", "array_size", "data_array", *META]


class Member(io.RawIOBase):
    """A tar member as a seekable file, so pyarrow reads the parquet footer without extracting."""

    def __init__(self, path, offset, size):
        self.f, self.offset, self.size, self.pos = open(path, "rb"), offset, size, 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, pos, whence=0):
        self.pos = pos if whence == 0 else self.pos + pos if whence == 1 else self.size + pos
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        self.f.seek(self.offset + self.pos)
        data = self.f.read(n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def stats(x):
    m, s = x.mean(), x.std()
    z = (x - m) / s if s > 0 else np.zeros_like(x)
    d = np.diff(x)
    spec = np.abs(np.fft.rfft(x - m))
    k = int(spec[1:].argmax()) + 1 if len(spec) > 1 else 0
    return {"mean": m, "std": s, "min": x.min(), "max": x.max(), "ptp": np.ptp(x),
            "skew": (z ** 3).mean(), "kurt": (z ** 4).mean() - 3, "dwdt_std": d.std() if len(d) else 0.0,
            "dwdt_max": np.abs(d).max() if len(d) else 0.0, "fft_peak_bin": k,
            "fft_peak_mag": spec[k] / len(x) if k else 0.0}


def process(task):
    tar, member, offset, size = task
    stat_rows, wave_rows = [], []
    pf = pq.ParquetFile(Member(tar, offset, size))
    cols = [c for c in COLUMNS if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=500, columns=cols):
        df = batch.to_pandas()
        for r in df.itertuples(index=False):
            if not r.array_size:
                continue
            cmp = CMP.search(r.pv)
            if not cmp:
                continue
            x = np.asarray(r.data_array, dtype="float64")
            key = {"event_id": int(r.event_id), "cycle_id": int(r.cycle_id), "pv_base": CMP.sub("-Cmp", r.pv)}
            stat_rows.append({**key, "cmp": int(cmp.group(1)), "n_samples": len(x),
                              **{k: float(v) for k, v in stats(x).items()}})
            if WAVE_CHAIN.search(r.pv):
                wave = np.interp(np.linspace(0, len(x) - 1, POINTS), np.arange(len(x)), x).astype("float32")
                wave_rows.append({**key, "cmp": int(cmp.group(1)), "n_samples": len(x), "wave": wave,
                                  **{m: getattr(r, m) for m in META if hasattr(r, m)}})
    # One PM member decodes to more than 1 GB; give it back before the next task.
    del pf
    gc.collect()
    pa.default_memory_pool().release_unused()
    source = os.path.basename(tar)
    system = member.split(".")[0]
    for row in stat_rows + wave_rows:
        row["source_tar"], row["system"] = source, system
    return stat_rows, wave_rows


def pair(rows, value_cols, cmp_names):
    """One row per (event, cycle, chain) from one row per component."""
    df = pd.DataFrame(rows)
    keys = ["event_id", "cycle_id", "pv_base"]
    shared = [c for c in df.columns if c not in value_cols and c != "cmp"]
    base = df.drop_duplicates(keys)[shared].set_index(keys)
    for cmp, name in cmp_names.items():
        part = df[df.cmp == cmp].set_index(keys)[value_cols]
        base = base.join(part.rename(columns=lambda c: name(c)), how="left")
    return base.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N tars only, for a test run")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()

    tars = sorted(os.path.join(args.raw, f) for f in os.listdir(args.raw)
                  if f.startswith("SDS_Data_") and f.endswith(".tar"))
    if args.limit:
        tars = tars[:args.limit]
    tasks = []
    for tar in tars:
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.rsplit("/", 1)[-1] in RF_MEMBERS:
                    tasks.append((tar, m.name.rsplit("/", 1)[-1], m.offset_data, m.size))
    tasks.sort(key=lambda t: -t[3])
    print(f"{len(tars)} tars, {len(tasks)} RF members, {sum(t[3] for t in tasks) / 1e9:.1f} GB, "
          f"{args.workers} workers", flush=True)

    t0, stat_rows, wave_rows = time.time(), [], []
    # ProcessPoolExecutor raises when a worker is killed (out of memory); multiprocessing.Pool waits forever.
    # fork, because the program is read from stdin and cannot be re-imported by a spawned worker.
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for i, fut in enumerate(as_completed([pool.submit(process, t) for t in tasks]), 1):
            s, w = fut.result()
            stat_rows += s
            wave_rows += w
            if i % 10 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} members, {len(stat_rows)} channels, {time.time() - t0:.0f} s", flush=True)

    names = ["mean", "std", "min", "max", "ptp", "skew", "kurt", "dwdt_std", "dwdt_max", "fft_peak_bin", "fft_peak_mag"]
    channel = pair(stat_rows, names, {0: lambda c: f"cmp0_{c}", 1: lambda c: f"cmp1_{c}"})
    channel[[c for c in channel.columns if c.startswith("cmp") and not c.endswith("peak_bin")]] = \
        channel[[c for c in channel.columns if c.startswith("cmp") and not c.endswith("peak_bin")]].astype("float32")
    waves = pair(wave_rows, ["wave"], {0: lambda c: "I", 1: lambda c: "Q"})
    waves = waves.dropna(subset=["I", "Q"])
    waves["section"] = waves.pv_base.str.split(":").str[0]
    channel["section"] = channel.pv_base.str.split(":").str[0]

    tmp = os.path.join(args.out, ".tmp")
    os.makedirs(tmp, exist_ok=True)
    for name, df in (("llrf_waveforms", waves), ("channel_stats", channel)):
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), os.path.join(tmp, f"{name}.parquet"),
                       compression="zstd")
    manifest = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "tars": len(tars), "members": len(tasks),
                "events": int(channel.event_id.nunique()), "llrf_waveforms": len(waves),
                "channel_stats": len(channel), "seconds": round(time.time() - t0)}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for name in ("llrf_waveforms.parquet", "channel_stats.parquet", "manifest.json"):
        os.replace(os.path.join(tmp, name), os.path.join(args.out, name))
    os.rmdir(tmp)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
PYEOF
