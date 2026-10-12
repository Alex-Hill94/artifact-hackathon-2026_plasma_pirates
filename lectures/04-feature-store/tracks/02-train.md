# 02 · Train on the real data

15 minutes, from your laptop. A feature view picks the rows and columns, the model registry keeps the model with its view.

Each use case ships a training pipeline. The PALLAS one runs on the full campaigns as it is:

```bash
hops project use m4cast_pallas
cd usecases/pallas
python training_pipeline.py --only pallas_inverse_a
```

```
pallas_inverse_a: trained on 5766, held out 1442
                   p_1    a_0   c_n2
model            0.973  0.916  0.961
linear baseline  0.854  0.409  0.814
registered pallas_inverse_a v1
```

26 seconds on a laptop. `--only pallas_direct_b` trains the other objective, a Gaussian process on 2,284 rows, in about two and a half minutes (R² 0.996, 0.937, 0.999).

## What the pipeline does

```python
fg = fs.get_feature_group("pallas_campaign_a", version=1)
query = fg.select(features + labels).filter((fg.scan == "random") & (fg.injected == True))
fv = fs.get_or_create_feature_view(name="pallas_inverse_a", version=1, query=query, labels=labels)

X, y = fv.training_data()
...
entry = mr.python.create_model(name="pallas_inverse_a", metrics=metrics, feature_view=fv)
entry.save("models/pallas_inverse_a")
```

- The filter lives in the view, so everyone who reads the view gets the same rows.
- `feature_view=fv` on the model is the lineage: the registry shows which view and which feature groups the model came from.
- `get_or_create_feature_view` returns the view if a teammate already created it. A new idea is a new `version`, or a new name.

## Your turn

The view and the model are shared with everyone in the project. Use a name of your own for anything you add:

```python
fv = fs.get_or_create_feature_view(name="<you>_inverse_a", version=1, query=query, labels=labels)
```
