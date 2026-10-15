# Plasma Pirates presentation

This is a deliberately lightweight Quarto RevealJS scaffold based on
`index_inspiration.md`. It contains the visual language and a suggested story
arc, but leaves the scientific figures and final results as editable placeholders.

## Local requirements

Install the [Quarto CLI](https://quarto.org/docs/get-started/) separately. From
this directory, the optional Python figure-generation dependencies can be
installed with:

```bash
python -m pip install -r requirements.txt
```

The current deck does not require Python to display the existing interactive
scatter plot: it uses Plotly.js from the CDN. Python and the packages in
`requirements.txt` are useful if you want to generate additional Plotly figures
or replace the dummy data with a data-driven workflow.

## Render locally

From the repository root:

```bash
quarto preview presentation/index.md
```

To create a static HTML file:

```bash
quarto render presentation/index.md --to revealjs
```

The resulting HTML is self-contained enough to open or publish as a static deck.

## Rebuild the static outlier histogram

The outlier slide uses a local CSV/Plotly snapshot; it does not contact
Hopsworks when the presentation is viewed. To refresh that snapshot, run this
from the repository root with the usual Hopsworks credentials available:

```bash
.venv/bin/python presentation/build_outlier_histogram.py --max-rows 50000
```

This writes `presentation/data/trajectory_histogram_snapshot.csv` and
`presentation/data/trajectory_histograms.html`.
