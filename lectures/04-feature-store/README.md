# Feature Store

**Speaker:** Lex Avstreikh (Hopsworks)
**Slot:** Monday, October 12 (concepts) + Tuesday, October 13 (recap + 2h hands-on)

## Abstract

Feature, training and inference pipelines: three independent programs sharing one data layer. We build all three on real accelerator data, the SPIRAL2 LLRF fault diagnosis release from GANIL, then point your own challenge data at the same pattern. Everything runs from a terminal, yours or the cluster's, by hand or through a coding agent.

## Slides

[slides.hops.io/2026-10/aissai26](https://slides.hops.io/2026-10/aissai26/). Arrow keys to move, `O` for the overview, `L` copies the link to the current slide. `?doc` renders the whole deck as one page.

## Tracks

| # | Track | Who | Time |
|---|---|---|---|
| 00 | [Setup](tracks/00-setup.md): `hops setup`, laptop and cluster terminal, agents | everyone | 10 min |
| 01 | [Ingest](tracks/01-ingest.md): LLRF events to feature groups | everyone | 20 min |
| 02 | [Share](tracks/02-share.md): read the organisers' store, publish yours | everyone | 15 min |
| 03 | [Train](tracks/03-train.md): feature view, time split, model registry | anomaly teams | 20 min |
| 04 | [Serve](tracks/04-serve.md): live events, online features, deployment | anomaly teams | 20 min |
| 05 | [Simulation](tracks/05-simulation.md): Cheetah runs to surrogate to optimiser | optimisation, surrogate teams | 30 min |

Monday: talk, then 00 and 01 together. Tuesday: 02 together, then 03 + 04 or 05 per challenge.

## Layout

```
04-feature-store/
├── data/prepare_llrf.py        Zenodo 22729653 → parquet, one file per feature group
├── generators/llrf_stream.py   live LLRF events resampled from the real release
├── generators/sim_campaign.py  Cheetah quadrupole scan, one row per run
├── tracks/                     00 to 05
└── requirements.txt
```

## Data

SPIRAL2 LLRF Fault Diagnosis: Structured Results Data (V7). A. Ghribi, P. Lassalle, M. Dalibard-Brun, M. Di Giacomo (GANIL), F. Bouly (CNRS). Zenodo [22729653](https://zenodo.org/records/22729653), MIT license. Pipeline: [aghribi/LLRF_SP2_pipeline](https://github.com/aghribi/LLRF_SP2_pipeline).

Synthetic events from `llrf_stream.py` are resampled from this release and flagged `synthetic=True`.

## Additional resources

- [Hopsworks docs](https://docs.hopsworks.ai/latest/concepts/hopsworks/)
- Docs MCP for agents: `https://mcp.hopsworks.ai/mcp`
- Book: *Building Machine Learning Systems with a Feature Store*, Jim Dowling, O'Reilly
- Paper: *The Hopsworks Feature Store for Machine Learning*, SIGMOD Companion 2024, [doi:10.1145/3626246.3653389](https://doi.org/10.1145/3626246.3653389)
