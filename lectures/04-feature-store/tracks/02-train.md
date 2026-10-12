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

26 seconds on a laptop. Every run registers a new model version, so yours may print `v2` or later. `--only pallas_direct_b` trains the other objective, a Gaussian process on 2,284 rows, in about two and a half minutes (R² 0.996, 0.937, 0.999).

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

## In a project of your own

In `m4cast_pallas` you can register a model and you cannot deploy it. Once you have your own project with the real stores shared in (track 03), run the same pipeline there and point it at the shared store:

```bash
hops project use <your_project>
python training_pipeline.py --store m4cast_pallas_featurestore
```

The feature groups are read from `m4cast_pallas`, the views and the models are created in your project, where track 04 can deploy them. Same scores, 3 min 29 s for both objectives from a laptop.

## Your turn

The view and the model are shared with everyone in the project. Use a name of your own for anything you add:

```python
fv = fs.get_or_create_feature_view(name="<you>_inverse_a", version=1, query=query, labels=labels)
```
