## 1 · Read the simulation data from Hopsworks

::: {.columns}
::: {.column width="53%"}

![](index_files/images/Screenshot 2026-10-14 at 10.11.57.png){fig-alt="PALLAS Data Explorer showing the trajectory feature group and candidate outliers" width="96%"}

::: {.caption}
The same feature groups can be inspected interactively before they become model inputs.
:::

:::
::: {.column width="43%"}

### The pipeline starts with a feature store connection

```python
project = pp.connect(pp.PROJECT)
fs = project.get_feature_store()

case = pp.CASES[CASE]
fv = pp.get_feature_view(
    fs, f"{OWNER}_{CASE}_fv", FV_VERSION
)
rows = pp.read_rows(fv, case)
```

The selected **case** determines the feature group, input columns, target columns, and row filter. The feature view then applies the shared input normalisations stored in Hopsworks.

This keeps the experiment tied to a reproducible data definition rather than a manually exported table.

:::
:::
