# Feature Store

**Speaker:** Lex Avstreikh (Hopsworks)
**Slot:** Monday, October 12 (concepts) + Tuesday, October 13 (recap + 2h hands-on)

## Abstract

Feature, training and inference pipelines: three independent programs sharing one data layer. We run all three on the real challenge data, ESS, CLARA and PALLAS, served as feature groups from the hackathon cluster, then you take the same pattern into your own project. Everything runs from a terminal, yours or the cluster's, by hand or through a coding agent.

## Slides

[slides.hops.io/2026-10/aissai26](https://slides.hops.io/2026-10/aissai26/). Arrow keys to move, `O` for the overview, `L` copies the link to the current slide. `?doc` renders the whole deck as one page.

## Tracks

The cluster is `https://aissai26.hops.live`.

| # | Track | Where | Time |
|---|---|---|---|
| 00 | [Setup](tracks/00-setup.md): Auth0 login, `hops setup`, laptop and cluster terminal, agents | laptop | 10 min |
| 01 | [Read](tracks/01-read.md): the real ESS, CLARA and PALLAS feature groups | laptop | 10 min |
| 02 | [Train](tracks/02-train.md): feature view, training data, model registry | laptop | 15 min |
| 03 | [Your project](tracks/03-your-project.md): your features next to the real ones | cluster terminal | 20 min |
| 04 | [Serve](tracks/04-serve.md): deploy and call your model | cluster terminal | 15 min |

Then your challenge: `usecases/` holds a feature, a training and an inference pipeline per challenge.

## Layout

```
04-feature-store/
├── tracks/                 00 to 04, and own_project.py (tracks 03 and 04 in one script)
├── usecases/air-quality/   the concept slides' case: public PM2.5 and weather, a forecast, lookup by station
├── usecases/ess/           anomaly detection: waveform features, IsolationForest, scoring by key
├── usecases/clara/         optimisation, Easy objective: Cheetah captures, surrogate, optimiser on the deployment
├── usecases/clara-camera/  optimisation: real camera captures, surrogate, optimiser checked on real bursts
├── usecases/pallas/        surrogate models: the Smilei pack, two surrogates, a search
├── data/, generators/      public air quality and weather, the SPIRAL2 LLRF release and its stream
└── requirements.txt        the Hopsworks client the cluster runs
```

## Data

The challenge datasets, as feature groups in the projects `m4cast_ess`, `m4cast_clara` and `m4cast_pallas`; row counts are in [track 01](tracks/01-read.md). The raw files stay at CC-IN2P3: the rows were computed there and only the rows came to the cluster. Cite each dataset as its challenge README asks.

The concept slides use public air quality data: hourly PM2.5 of the French regulatory stations, LCSQA and Ineris, [data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-temps-reel-de-mesure-des-concentrations-de-polluants-atmospheriques-reglementes-1/), Licence Ouverte 2.0, and weather from [Open-Meteo](https://open-meteo.com/), CC BY 4.0.

The SPIRAL2 track uses SPIRAL2 LLRF Fault Diagnosis: Structured Results Data (V7). A. Ghribi, P. Lassalle, M. Dalibard-Brun, M. Di Giacomo (GANIL), F. Bouly (CNRS). Zenodo [22729653](https://zenodo.org/records/22729653), MIT license. Pipeline: [aghribi/LLRF_SP2_pipeline](https://github.com/aghribi/LLRF_SP2_pipeline).

## Additional resources

- [Hopsworks docs](https://docs.hopsworks.ai/latest/concepts/hopsworks/)
- Docs MCP for agents: `https://mcp.hopsworks.ai/mcp`
- Book: *Building Machine Learning Systems with a Feature Store*, Jim Dowling, O'Reilly
- Paper: *The Hopsworks Feature Store for Machine Learning*, SIGMOD Companion 2024, [doi:10.1145/3626246.3653389](https://doi.org/10.1145/3626246.3653389)
