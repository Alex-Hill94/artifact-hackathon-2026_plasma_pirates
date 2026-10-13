# PALLAS surrogates on the feature store

Feature, training and inference pipelines on the PALLAS Smilei simulations: one row per simulation, the settings in, the electron bunch out. Two surrogates and a search on the deployed one.

## Data

In `m4cast_pallas`, read-only. No event time, nothing online.

| Feature group | Key | Rows | What |
|---|---|---|---|
| `pallas_campaign_a` | `config` | 13,073 | `p_1, a_0, c_n2, x_of` and the bunch, both scans |
| `pallas_campaign_b` | `config` | 5,286 | `p_max, cn2_max, l_inj, dip_frac, x_of` and the bunch |
| `pallas_b_trajectories` | `config, z_um` | 322,615 | the bunch at 113 positions |
| `pallas_b_moments` | `config, plane_um, particle_group` | 211,360 | mean and covariance at 20 planes |

The feature pipeline applied the pack's data card once: `injected` is `q_pC >= 3`, `in_reference_train` is the organisers' frozen split, names are lower case. Campaign B holds 5,286 of the 6,000 simulations; the other 714 are the hidden test set.

## Run

From a laptop, in `m4cast_pallas`:

```bash
hops project use m4cast_pallas
cd usecases/pallas
python training_pipeline.py --only pallas_inverse_a     # 26 s
python training_pipeline.py --only pallas_direct_b      # 2 min 41 s
```

In a project you own, with the real store shared in ([track 03](../../tracks/03-your-project.md)). Train from a laptop: on the terminal's single CPU the Gaussian process takes 10 minutes instead of 3.

```bash
hops project use <your_project>
python training_pipeline.py --store m4cast_pallas_featurestore
```

Deploy and search from the cluster terminal:

```bash
uv pip install -r ../../requirements.txt
python inference_pipeline.py --target-mev 200 --min-charge-pc 20
hops deployment stop pallasdirectb
```

Without a cluster, on the organisers' public 10 % sample (`notebook.ipynb` walks the same steps):

```bash
uv pip install requests
python feature_pipeline.py --out pallas_fg
python training_pipeline.py --local pallas_fg --models models
python inference_pipeline.py --local models/pallas_direct_b/model.pkl --fg-dir pallas_fg --target-mev 200 --min-charge-pc 20
```

## What you get

| View and model | Rows | Features to labels | Model |
|---|---|---|---|
| `pallas_direct_b` | campaign B, injected, reference train: 2,855 | five knobs to `e_med_mev, de_mad, q_end` | Gaussian process per target |
| `pallas_inverse_a` | campaign A, random scan, injected: 7,208 | nine bunch features to `p_1, a_0, c_n2` | gradient boosting per target |

Deployment `pallasdirectb` serves the direct surrogate through `predictor.py`.

## Results

R² on a 20 % holdout:

```
                   p_1    a_0   c_n2
pallas_inverse_a 0.973  0.916  0.961
linear baseline  0.854  0.409  0.814

                 e_med_mev  de_mad  q_end
pallas_direct_b      0.996   0.937  0.999
linear baseline      0.946   0.244  0.829
```

The search, 60 calls to the deployment, about 10 minutes:

```
p_max       6999.7362
cn2_max        0.0565
l_inj          0.7427
dip_frac       0.3341
x_of         889.6666
predicted: E_med 204.6 MeV, dE_mad 0.034, q_end 66.9 pC (loss 0.0347)
```

- The proposal changes with the machine that runs the optimiser.
- Campaign B already holds real runs with a lower loss: config 5746, 204.4 MeV and 64.7 pC, 0.0318.
- The proposal is a setting to simulate: run Smilei there and insert the run as a new row of `pallas_campaign_b`.

## Limits

- In `m4cast_pallas`, creating a feature group or a deployment answers 403.
- `mr.get_model(name)` without a version returns version 1. The inference pipeline takes the best `r2_mean` unless `--model-version` is given.
- The six test input files and the `level5` and `level6` data are not in the feature store.

## Cite

- G. Kane et al., Machine Learning: Science and Technology 7, 030502 (2026). doi:10.1088/2632-2153/ae6603
- J. Derouillat et al., Comput. Phys. Commun. 222, 351 (2018). doi:10.1016/j.cpc.2017.09.024
