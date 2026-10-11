# CLARA cameras: real captures from the CC-IN2P3 archive

One row per real camera capture from the CLARA HDF5 archive (`/sps/m4cast/artifact_hackathon_2026/optimisation/extracted` on CC-IN2P3, 17,885 files, 36 cameras). Unlike `../clara`, nothing here is simulated.

## Source

Each `<CAMERA>_<date>_<time>UTC_<n>_images_full.hdf` holds `Capture000001`...: a uint16 frame (12-bit values in 16 bits) and, as attributes on the same dataset, the machine state at that capture: `SETI`, `READI`, `GETSETI`, `RPOWER` per magnet, `<camera>:ANA:PixMM_RBV` per camera, `Event Timestamp`, camera model and serial.

On the hackathon cluster the archive is mounted read-only at `/m4cast` in every project pod. The frames are read in place; only the rows below are written.

## Feature group

`clara_camera_captures` v1, offline, key `capture_id`, event time `event_ts`, partition `camera`.

| column | what |
|---|---|
| `capture_id` | file name without extension plus capture, e.g. `CLA-S07-DIA-CAM-03_2026-03-19_13-16-33UTC_10_images_full_Capture000001` |
| `event_ts` | camera timestamp, UTC (year from the file name, the stamp has none) |
| `camera`, `scan`, `source_file`, `capture` | where the frame is: camera PV, sub-folder below the day (the operators' scan name, empty if none), path under the archive root, dataset |
| `pix_mm`, `width`, `height` | pixel size from `<camera>:ANA:PixMM_RBV`, frame size |
| `pedestal`, `noise`, `threshold` | frame median, 1.4826 x MAD (floor 1 count), spot threshold in counts |
| `mu_x_mm`, `mu_y_mm`, `sigma_x_mm`, `sigma_y_mm` | first and second moments of the spot, mm, relative to the chip centre; null when `beam_present` is false |
| `beam_present`, `lit_pixels`, `saturated_pixels`, `peak`, `signal_sum` | spot pixels, pixels at 65504, peak and summed signal above the pedestal |
| `thumbnail` | 1/6-scale frame, square-root stretch, inferno colour map, JPEG data URI (about 14 KB); the data preview shows it as an image |
| `<magnet>_seti`, `<magnet>_readi` | commanded and read-back current of every magnet PV in the file, e.g. `cla_s07_mag_quad_03_seti` |

Spot: the frame minus its median, smoothed 5x5; a pixel is in the spot when the smoothed value clears both 5 noise sigmas and 5 % of the smoothed peak. Moments are weighted by the unsmoothed signal. Below 50 lit pixels the frame counts as dark.

Magnet PVs differ between files. The group keeps the columns of its first insert; later inserts are reindexed to them and extra PVs are dropped with a message.

## Commands

From this folder, with `requirements.txt` here on top of `lectures/04-feature-store/requirements.txt`:

```bash
# one scan, 2 files, to parquet
python camera_pipeline.py --select "2026/3/19/S07 Quad 03 Scans" --limit 2 --out data/

# the same scan into the feature group
python camera_pipeline.py --select "2026/3/19/S07 Quad 03 Scans" --insert
```

`--select` is a file or a directory under the archive root (`claraserv3/`); directories are searched recursively. `--workers` sets the number of files read in parallel (default 8).

As a Hopsworks job, in an environment cloned from a base environment with `requirements.txt` installed (Hopsworks clones base environments only):

```bash
hops env clone clara-camera --from pandas-training-pipeline
hops env install clara-camera -f requirements.txt
hops job deploy clara_camera_captures camera_pipeline.py --env clara-camera \
  --args '--select "2026/3/19/S07 Quad 03 Scans" --insert' --run --wait
```

## Results

2 files of `2026/3/19/S07 Quad 03 Scans` (CLA-S07-DIA-CAM-03), read through the mount on the hackathon cluster: 20 captures in 2 s, all with a beam, 24 SETI columns. Spot about 0.29 mm (x) by 0.09 mm (y) RMS, centroid near (+0.17, -0.16) mm, 40 to 110 saturated pixels per frame.

Cluster run, project `aissai26_clara`, job `clara_camera_captures` in environment `clara-camera`: the whole scan, 161 files, 1,610 captures, all with a beam, inserted into `clara_camera_captures` v1 in about 5 minutes. The data preview shows the thumbnails.
