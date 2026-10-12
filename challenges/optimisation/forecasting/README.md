# Optimisation — Forecasting (CLEAR beam charge)

**Case owner:** Antonio Gilardi (CERN)
**Status:** ready for review. The kickoff notebook has been tested on a real raw chunk.

## Overview

The parent [optimisation challenge](../) asks *how do I find the best settings*. This
sub-challenge looks at the time dimension: *how will the beam behave next, and can we see
it coming?*

Using shot-by-shot data from CLEAR at CERN (October 2026), study how the accelerator's
exogenous variables behave and use them to infer the gun charge `BCM_GUN`. The exogenous
variables are:
- RF amplitudes and phases
- the laser: power, and the spot on the virtual cathode
- a few magnet currents
- ambient temperature and humidity

The work moves through three stages:
1. **Understand the data:** reduce the raw runs to tables, clean and synchronise them, and
   run a full correlation study.
2. **Virtual diagnostic:** infer the charge of a shot from the same shot's other variables.
3. **Forecasting:** predict the next shot's charge, or several shots ahead, from the past.

The long-term goal is to feed the forecast into a controller that keeps the charge stable.
That brings the problem back to optimisation.

See [`scientific_case.md`](scientific_case.md) for the background and
[`objectives.md`](objectives.md) for the Easy, Medium and Hard tiers.

## Structure

- `scientific_case.md`: background, motivation and the data
- `objectives.md`: easy / medium / hard goals
- `requirements.txt`: Python dependencies
- `setup_venv.sh`: creates a `venv/` with those dependencies and registers it in Jupyter as
  "Python (forecasting)". Run `./setup_venv.sh` once, before opening the notebook
- `kickoff_notebook.ipynb`: starter notebook. It opens a raw chunk safely, reduces it to a
  table, works through the Easy tier as an example, and holds the common rules (test split,
  dropouts, rolling-mean baseline). The Medium and Hard tasks are left open, with the
  references to beat
- `data/README.md`: where to download the raw runs, the raw and table formats, every
  column, and the common rules

## Getting help

Reach out to the case owner, or ask during the Tuesday hands-on session and the Wednesday
mid-challenge check-in.
