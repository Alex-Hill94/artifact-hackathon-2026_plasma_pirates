# Scientific Case — Forecasting (CLEAR beam charge)

## Background

CLEAR (CERN Linear Electron Accelerator for Research) is a ~40 m electron linac at CERN,
operated as a user facility since 2017 (19 experiments in 2017, 32–39 per year since). Its
beam is unusually flexible:

| Beam parameter | Range |
|---|---|
| Energy | 20 – 220 MeV |
| Bunch charge | 0.01 – 1.5 nC |
| Normalised emittance | 3 µm (0.05 nC/bunch) – 20 µm (0.4 nC/bunch) |
| Bunch length | ~100 – 1200 µm |
| Relative energy spread | < 0.2 % rms |
| Repetition rate | 0.833 – 10 Hz |
| Micro-bunches per train | 1 – 150, spaced at 1.5 or 3.0 GHz |

The beam is produced by a laser-driven RF photoinjector (gun, virtual-cathode camera
`BTV0125`, charge monitors at the gun). The accelerating structures, fed by two klystrons,
`MKS11` and `MKS15`, then take it to the experimental areas (VESPER/EOS, in-vacuum,
plasma lens, in-air). The beam charge downstream is measured by other ICTs.

Experiments are mostly medical (43 %), followed by machine development (17 %),
instrumentation (16 %), ML and plasma (6 % each), impedance, training, detectors and R2E.

## Motivation

Many CLEAR users need **beam charge to stay stable** across a run. Medical-irradiation
and detector tests are the main examples, because the delivered dose or signal scales with
charge. Charge drifts and jitters shot to shot. The causes are spread across the
whole machine:

- the laser: power and spot on the cathode
- the RF: klystron amplitudes and phases
- the ambient conditions: hall temperature and humidity

There is also no single knob that fixes it.

Two data-driven tools would help operations:

1. **Virtual diagnostic:** infer the charge `Q_t` of a shot from the other variables
   measured on that same shot. This gives a charge estimate when the charge monitor is
   unavailable, and shows which variables actually drive charge.
2. **Forecasting:** predict `Q_{t+1}` (or `Q_{t+1..t+H}`) from everything known up to
   shot `t`. This gives operators, or a feedback loop, time to act before the charge
   drifts.

The longer-term goal, and the link to the parent optimisation challenge, is to **close the
loop**. Once forecasting works, a controller `g(x)` can adjust a controllable subset of the
inputs (`x ⊂ X`, e.g. laser power or RF phase) so that the forecast charge stays on target.
That turns forecasting into predictive optimisation and control.

## This challenge's data (October 2026)

A new passive acquisition (`acquire.py`) ran on CLEAR during 6–9 October 2026. It records
the gun charge `BCM_GUN` and everything that could drive it:

| Family | Parameters | Kind |
|---|---|---|
| Charge (target) | `BCM_GUN` | scalar, nC; it also triggers the acquisition |
| Laser | `lab_laser_power`, `clex_laser_power` | scalars: laser energy (J) measured in the laser lab and in the accelerator |
|  | `laser_scope` | fast photodiode on the laser pulse, 2,002 samples |
|  | `BTV0125` | camera image of the laser on the virtual cathode, 1216 × 1936 px, with exposure and stamp metadata |
| Gun RF | `PGIA`, `PGRA`, `PGLA`, `PGIP` | waveforms, 960 samples, a.u.: incident power, reflected power, gun pickup loop, phase |
| Klystron 15 | `PKI15A`, `PKR15A`, `PKI15P` | waveforms, 960 samples, a.u.: incident power, reflected power, phase |
| Magnets | `SNH_110`, `SNI_120`, `DHG0130` | scalars, A, slow |
| Ambient | `TEMP1–4`, `HUMID1–4` | scalars, °C and %, slow |

**How a shot is recorded:**
- A new shot is detected when `BCM_GUN`'s acquisition stamp changes. All other parameters
  are then read in parallel.
- Every value is stored with **its own acquisition stamp**, so a reading from a device
  that had not yet updated can be recognised and moved to the right shot.
- Data are saved as pickle chunks of 100 shots.

Nine runs were recorded:

| Run | Shots (≤) | Raw size |
|---|---|---|
| `acquisition_2026-10-06_21-57-43` | 22,700 | 110 GB |
| `acquisition_2026-10-07_20-55-58` | 500 | 2 GB |
| `acquisition_2026-10-07_21-42-34` | 100 | 0.2 GB |
| `acquisition_2026-10-07_21-43-26` | 7,200 | 35 GB |
| `acquisition_2026-10-08_00-08-19` | 25,500 | 124 GB |
| `acquisition_2026-10-08_15-42-27` | 12,100 | 59 GB |
| `acquisition_2026-10-08_21-11-47` | 20,000 | 97 GB |
| `acquisition_2026-10-09_07-46-07` | 11,000 | 53 GB |
| `acquisition_2026-10-09_11-48-33` | 18,600 | 91 GB |

The runs are deliberately diverse and represent the full range of operating conditions:
- stable operation
- runs in wrong conditions (e.g. saturation), with no clear correlation to find
- single-bunch and multi-bunch beams
- runs where the laser position and/or intensity was varied

Recognising which is which, from the data alone, is part of the challenge.

Shot counts are upper bounds (number of chunks × 100). In total there are about 118,000
shots and 572 GB, almost all of it camera images.

Reducing each run to a table with one row per shot is part of the challenge: images and
waveforms become features, and every acquisition stamp is kept for synchronisation. The
kickoff notebook shows how for one chunk.

## References

[1] D. Gamba et al., *The CLEAR user facility at CERN*, Nucl. Instrum. Methods Phys. Res. A
909 (2018) 480–483.

[2] B. Lim, S. Ö. Arık, N. Loeff, T. Pfister, *Temporal Fusion Transformers for interpretable
multi-horizon time series forecasting*, Int. J. Forecasting 37(4) (2021) 1748–1764.

[3] D. Salinas, V. Flunkert, J. Gasthaus, T. Januschowski, *DeepAR: Probabilistic forecasting
with autoregressive recurrent networks*, Int. J. Forecasting 36(3) (2020) 1181–1191.

[4] T. Chen, C. Guestrin, *XGBoost: A scalable tree boosting system*, KDD 2016.
