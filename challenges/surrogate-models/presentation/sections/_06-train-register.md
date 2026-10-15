## 2 · Transform, train, score, and register

::: {.columns}
::: {.column width="48%"}

```python
TD_VERSION = pp.prepare_statistics(fv)
train_configs, holdout_configs, holdout_groups = pp.split_configs(
    rows, case, SPLIT, HOLDOUT_FRACTION, SEED
)

train_rows = rows[rows.config.isin(train_configs)]
X_train = pp.transform(fv, train_rows, case)
model = pp.SurrogateModel(
    model_module.build_model(**params),
    case.targets, TARGET_SCALING
)
model.fit(X_train, train_rows)
```

The split is made on complete **configurations**, so rows from a single simulation do not leak into both training and holdout data. The official scorer evaluates predictions on configurations the model has not seen.

:::
::: {.column width="48%"}

![](index_files/images/Screenshot 2026-10-15 at 10.02.13.png){fig-alt="Hopsworks provenance graph linking a feature group, feature view, training dataset, and registered model" width="98%"}

::: {.caption}
Hopsworks records the lineage from source feature group through feature view and training dataset to the registered model.
:::

::: {.accent-card}
**Registry hand-off**

The upload bundles the fitted model, metadata, code, metrics, feature-view version, and split. A later inference run can recover the same inputs and provenance.
:::

:::
:::
