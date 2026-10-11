# Data — Forecasting (CLEAR, October 2026)

Shot-by-shot passive acquisitions from CLEAR, 6–9 October 2026: nine runs, ~118,000 shots,
572 GB raw. The run list is in [`../scientific_case.md`](../scientific_case.md).

**Where:**
- **Raw runs:** the CERNBox share <https://cernbox.cern.ch/s/yVioydShmcGKkto>, under
  `measured_data/`. It also holds the case owner's acquisition and analysis scripts
  (`acquire.py`, `check_data.py`, `plot_data.py`, `study_params.py`).
- **Download** a run, or its first few chunks, into `raw/<run>/` (e.g.
  `raw/acquisition_2026-10-09_07-46-07/chunk_00000.pkl`). One chunk is ~0.5 GB, so start
  with one or two.
- **Tables:** you reduce the raw chunks yourself to one row per shot. Section 2 of the
  kickoff notebook shows how and saves the result in `tables/<run>_<n>chunks.parquet`.
- Git ignores `raw/` and `tables/`.

## What the runs contain

The runs are **deliberately diverse**. Together they represent real operating conditions,
and handling that diversity is part of the challenge:

- Some runs were taken in **stable** conditions.
- Some were taken in **wrong conditions**, e.g. in saturation, where no clear correlation
  with the charge should be expected.
- Some are **single-bunch**, others **multi-bunch**.
- In some, the **laser position and/or intensity** was varied during the run.

The conditions of each run are not labelled. Recognising them from the data is part of
the Easy tier.

## Raw format

One folder per run, written by the case owner's `acquire.py`:

```
measured_data/acquisition_<YYYY-MM-DD_hh-mm-ss>/
├── parameters.pkl        {name: (device/property#field, selector)} -- what was read
└── chunk_00000.pkl ...   100 shots each (the last chunk may be shorter)
```

Each chunk is a pickled **list of shot dicts**:

| key | meaning |
|---|---|
| `shot` | shot number within the run, continuous from 0 |
| `pc_time` | acquisition PC clock (`datetime`) |
| `trigger_stamp` | `BCM_GUN`'s acquisition stamp, which marks the shot. stamps are datetimes or numbers in ns / µs / ms / s |
| `<NAME>` | the value: a float, an array, or a dict (a whole device property) |
| `<NAME>_stamp` | that value's **own** acquisition stamp |

**How a shot is recorded:**
- A new shot is detected when `BCM_GUN`'s stamp changes. All 23 parameters are then read
  in parallel.
- So `<NAME>_stamp` may be older than `trigger_stamp`, when the device had not updated
  yet. This is the basis of synchronisation.
- Slow devices (`TEMP*`, `HUMID*`, `SNH_110`, `SNI_120`, `DHG0130`) update much less
  often than once per shot.

The parameters (fields as seen in the case owner's `plots/structure.txt`):

| Parameter | Value | Data field used |
|---|---|---|
| `BCM_GUN` | float, **nC** | target |
| `clex_laser_power`, `lab_laser_power` | float | |
| `laser_scope` | dict: `signal` (2002,) V, `times` (2002,) ns, `saturated`, `acquired`, gain/offset | `signal` |
| `BTV0125` | dict: `image2D` (1216, 1936) int16, `isGoodExposure`, `cycleStampGlob`, `imageTimeStamp`, `acqImageNumberDiffFromLast`, calibrations, projections | `image2D` |
| `PGIA`, `PGRA`, `PGIP`, `PKR15A`, `PKI15P` | array (1, 960), arbitrary units | the array |
| `PGLA`, `PKI15A` | dict: `samples` (1, 960), arbitrary units (labelled mV), `hotSpotsSamples` (1, 2), timing | `samples` |
| `SNH_110`, `SNI_120`, `DHG0130` | float | |
| `TEMP1–4`, `HUMID1–4` | float | |

**Loading:**
- The chunks contain CERN control-system objects (e.g. enum values in the camera
  metadata), so a plain `pickle.load` needs `pyda` installed.
- `load_chunk` in the kickoff notebook (section 2a) reads them with a restricted unpickler
  instead. Only numpy, datetime and plain Python types are rebuilt, and anything else
  becomes an inert placeholder. No `pyda` is needed, and nothing in a file can run code.
- Status flags such as `isGoodExposure` or `saturated` are CERN `Boolean` objects. Their
  value is in the placeholder's `state['value']` (`flag_value` in the notebook).

## The table made by the kickoff notebook (`tables/<run>_<n>chunks.parquet`)

Section 2c of the kickoff notebook reduces raw chunks to one row per shot. The index is
`shot`, and every shot is kept. These features are one simple choice, a starting point:
improving them is part of the Easy tier.

| Column | Meaning |
|---|---|
| `t` | trigger time, s since epoch |
| `pc_time` | acquisition PC clock |
| `<NAME>` | scalar parameters |
| `<NAME>.peak`, `.peak_pos`, `.integral`, `.baseline` | waveform features, baseline-subtracted, same as `plot_data.py` |
| `<NAME>.w00 … .w47` | the waveform averaged in 48 equal windows (float32) |
| `<NAME>.intensity`, `.cx`, `.cy`, `.sx`, `.sy` | camera: background-subtracted total intensity, centroid and rms size in px |
| `<NAME>.stamp_offset` | that parameter's stamp − trigger stamp, in s. ≈ 0 means the same shot; ≈ −1 shot spacing means the previous shot |
| `BTV0125.isGoodExposure`, `.acqImageNumberDiffFromLast` | camera quality flags |
| `BTV0125.cycleStampGlob_offset`, `.imageTimeStamp_offset` | the camera's own timing, relative to the trigger, in s |
| `laser_scope.saturated`, `.acquired`, `.vertOffset_V`, `.vertGain_VperLSB`, `.horizOffset_ns` | scope status and settings |
| `PGLA.hotspot0/1`, `PKI15A.hotspot0/1` | the device's own two hot-spot samples |

**Not done in the table, on purpose:** synchronisation, dropout flagging and slow-device
forward-filling. Those come after, in section 3 of the kickoff notebook.

## Common rules

The same for every team, so that results can be compared. The kickoff notebook implements
them as functions (section 1b):

- **Test region:** the last 10 % of a run's rows, from shot `cut_shot(df)` on.
- **Evaluated shots:** test shots whose charge is real. A **dropout** is a charge below
  30 % of the local 201-shot median, or a missing charge, and is not evaluated.
- **Device flags:** shots with a bad camera exposure or a saturated scope *are* evaluated,
  because their charge is real.
- **Forecasting:** forecast `(shot, h)` predicts `shot` from information up to `shot − h`.
  The baseline is the mean of the last 9 non-dropout charges at `shot − h`.

## Ontology

There is no formal ontology file yet. What each parameter is:

| Parameter | What it measures | Unit |
|---|---|---|
| `BCM_GUN` | beam charge at the gun (**target**) | nC |
| `PGIA` | gun RF, incident power | a.u. |
| `PGRA` | gun RF, reflected power | a.u. |
| `PGLA` | gun RF, pickup loop in the gun | a.u. |
| `PGIP` | gun RF, phase | a.u. |
| `PKI15A` | klystron MKS15, incident power | a.u. |
| `PKR15A` | klystron MKS15, reflected power | a.u. |
| `PKI15P` | klystron MKS15, phase | a.u. |
| `lab_laser_power` | laser energy, measured in the laser lab | J |
| `clex_laser_power` | laser energy, measured in the accelerator | J |
| `laser_scope` | fast photodiode on the laser pulse | V |
| `BTV0125` | camera on the laser spot at the virtual cathode | px |
| `SNH_110`, `SNI_120`, `DHG0130` | magnet currents (solenoids and corrector near the gun) | A |
| `TEMP1–4` | hall temperature | °C |
| `HUMID1–4` | hall humidity | % |

RF signals are in arbitrary units, so compare them within a run, not across runs.

**Not every signal is guaranteed to carry useful information about the charge.** Finding
out which ones do, which are redundant, and which are just noise is part of the challenge
(Easy tier).
