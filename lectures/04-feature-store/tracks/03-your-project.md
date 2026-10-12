# 03 · Your own project

20 minutes, in the cluster terminal. In the `m4cast_*` projects you read and train. To write features and to deploy, you need a project where you are the owner.

## Create it

Create a project in the UI, or from Python on your laptop:

```python
import hopsworks

hopsworks.login(project="m4cast_pallas")
hopsworks.create_project("<you>_aissai26")
```

Then give the organisers its id: the number after `/p/` in the project's URL, printed by `create_project` too. They share the three real feature stores into it, read-only. It takes a few seconds, and nothing has to be accepted on your side.

## Check

Open the terminal of your new project (your project, **Terminal**, **Start Terminal**):

```bash
git clone -b lecture-04-feature-store https://github.com/MagicLex/artifact-hackathon-2026
cd artifact-hackathon-2026/lectures/04-feature-store
python -c "import hopsworks; p = hopsworks.login(); print(p.get_feature_store(name='m4cast_pallas_featurestore').get_feature_group('pallas_campaign_a', version=1).read().shape)"
```

`(13073, 36)` means the share is in place. `hops fg list` shows only the feature groups your project owns; the shared stores are read through `get_feature_store(name=...)`.

## Real rows, your feature, one view

```python
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()                                    # yours
real = project.get_feature_store(name="m4cast_pallas_featurestore")  # shared, read-only

a = real.get_feature_group("pallas_campaign_a", version=1)
df = a.read()

mine = df[df.injected][["config", "e_med_mev", "q_pc"]].copy()
mine["charge_per_mev"] = mine.q_pc / mine.e_med_mev

fg = fs.get_or_create_feature_group(name="charge_per_mev", version=1, primary_key=["config"])
fg.insert(mine[["config", "charge_per_mev"]])

query = (
    a.select(["e_med_mev", "de_mad", "x_of", "p_1"])
    .filter(a.injected == True)
    .join(fg.select(["charge_per_mev"]), on=["config"])
)
fv = fs.get_or_create_feature_view(name="p1_from_beam", version=1, query=query, labels=["p_1"])
X, y = fv.training_data()      # (10751, 4)
```

The join is on the primary key, `config`. The real rows are not copied: your project holds one column and a view that reads both stores.

The whole thing, with the model and the deployment of track 04, is `tracks/own_project.py`:

```bash
python tracks/own_project.py --no-deploy
```

```
read (13073, 36)
inserted 10751 rows in 6 s
training data (10751, 4)
r2 0.776 features ['e_med_mev', 'de_mad', 'x_of', 'charge_per_mev']
```

## Why the terminal

An insert goes through Kafka and HopsFS, which are not reachable from outside the cluster. Reads, views and training data work from your laptop in this project too.
