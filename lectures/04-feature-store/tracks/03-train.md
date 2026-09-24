# 03 · Train: feature view, time split, model registry

20 minutes. Select features across groups, split on time, train, register the model with its lineage.

## Feature view

A feature view selects features from one or more feature groups and names the label. It stores no data.

```python
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

events = fs.get_feature_group("llrf_events", version=1)
physics = fs.get_feature_group("llrf_physics", version=1)
labels = fs.get_feature_group("llrf_labels", version=1)

query = (
    labels.select(["is_fault"])
    .join(events.select(["kpi", "kii", "ampt", "phit"]), on=["event_id"])
    .join(physics.select_except(["event_id", "cavity_id", "event_ts"]), on=["event_id"])
)

fv = fs.get_or_create_feature_view(
    name="llrf_fault", version=1,
    query=query, labels=["is_fault"],
    description="Fault vs normal from header and physics features",
)
```

The join is point-in-time correct on `event_ts` of the label group: each label only sees feature values that existed when it was recorded.

## Split on time, not at random

Events from the same cavity in the same week look alike. A random split puts neighbours on both sides and inflates the score. Train on the past, test on the future.

```python
X_train, X_test, y_train, y_test = fv.train_test_split(
    train_start="2019-01-01", train_end="2024-01-01",
    test_start="2024-01-01", test_end="2026-01-01",
)
```

Or materialise it once, versioned, so every team member trains on the same rows:

```bash
hops td compute llrf_fault 1 --start-time 2019-01-01 --end-time 2026-01-01
hops td list llrf_fault
```

## Train

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib, os

model = HistGradientBoostingClassifier(max_iter=300).fit(X_train, y_train.values.ravel())
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"test AUC 2024-2025: {auc:.3f}")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
```

`HistGradientBoostingClassifier` takes the NaNs in this data as they are.

Expect a test AUC of 1.000. The physics features are computed over the whole buffer, trip included: they describe a fault that already happened. The stretch below asks the harder question.

## Register

```python
mr = project.get_model_registry()
entry = mr.python.create_model(
    name="llrf_fault", metrics={"auc": auc},
    feature_view=fv, input_example=X_train.head(1),
    description="HGB on header + physics, trained 2019-2023, tested 2024-2025",
)
entry.save("model")
```

```bash
hops model list
hops model lineage llrf_fault
```

The model now points at the feature view and the rows it trained on. Six months from now, "which data trained this" has an answer.

## Stretch

- Swap physics for `llrf_precursor` (from the shared store) and compare the AUC. The gap is the difference between describing a trip and seeing it coming.
- Root cause: a second view over faults only, label `fault_category`.
