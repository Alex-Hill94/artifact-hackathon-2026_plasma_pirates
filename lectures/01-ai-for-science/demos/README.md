# Demos for "AI for Science: putting physics into learning"

Three small CPU demos, 25 s to 3 min each, one per part of the lecture.

| Notebook | Part | What it shows |
|---|---|---|
| `demo1_pinn_oscillator.ipynb` | I. Soft constraints | Data-only network vs PINN on a damped oscillator observed only at the start; PINN with wrong physics |
| `demo2_symmetry_cluster.ipynb` | II. Hard constraints | Raw coordinates vs augmentation vs canonicalisation vs invariant descriptors vs an equivariant graph network, on cluster energies |
| `demo3_vae_signals.ipynb` | III. Generative models | A standard autoencoder (VAE) learns the space of damped-oscillation signals: latent map, generation from N(0,1) checked against the law, uncertainty by conditioning, anomaly detection |

Each notebook is self-contained: it states the question, explains every step and every
figure, and ends with take-home messages and a few things to try. They can be read
without the slides, and are saved with their outputs, so they can also be read without
running anything.

## Run

```bash
uv sync                 # numpy, matplotlib, CPU torch, jupyter
uv run jupyter lab
```

Without uv, any Python >= 3.11 with `numpy`, `matplotlib`, `torch` and `jupyter` will do.

Figures are written to `figures/` (they are also embedded in the slides).

`fem_heat.py` (not a notebook) makes the finite-element figure of the "problem" slide in Part I:
a deliberately rough P1 solution of the 1D heat equation with scikit-fem
(`uv run python fem_heat.py`).

The `.py` files are the sources; regenerate the notebooks, then run them to store the outputs:

```bash
uv run python py2nb.py demo*.py
uv run python -m nbconvert --to notebook --execute --inplace demo*.ipynb
```
