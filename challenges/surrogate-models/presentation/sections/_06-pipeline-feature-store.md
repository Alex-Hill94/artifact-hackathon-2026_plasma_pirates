## 1 · Pull the data from the feature store {#pipe-feature-store}

::: {.columns}
::: {.column width="57%"}

<div class="viz-html" data-viz="fg-explorer"></div>

::: {.caption}
The four feature groups of the public pack in the team project. Click one to see its columns; hover a column for its type.
:::

:::
::: {.column width="40%"}

```python
project = pp.connect(pp.PROJECT)
fs = project.get_feature_store()

case = pp.CASES[CASE]   # inputs, targets, rows
fv = pp.get_feature_view(
    fs, f"{OWNER}_{CASE}_fv", FV_VERSION)
rows = pp.read_rows(fv, case)
```

::: {.compact}
- The **case** fixes the feature group, inputs, targets and row filter (e.g. injected bunches only).
- Nobody exports a CSV by hand: every experiment starts from the same versioned data.
- One row per simulation, per position for the trajectories.
:::

![](index_files/images/Screenshot 2026-10-14 at 10.11.57.png){.lightbox .thumb fig-alt="PALLAS Data Explorer showing the trajectory feature group and candidate outliers"}

<a class="ext-link" data-link="hopsworks" target="_blank" rel="noopener">Open the project in Hopsworks ↗</a>

:::
:::
