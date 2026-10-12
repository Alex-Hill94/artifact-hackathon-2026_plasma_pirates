# %% [markdown]
# # Demo I: soft constraints, a PINN on a damped harmonic oscillator
#
# *Part I of the lecture "AI for Science: putting physics into learning". This notebook
# can be read and run on its own: it needs only `numpy`, `matplotlib` and `torch`, and
# takes about one minute on a laptop CPU.*
#
# ## The question
#
# A neural network trained on data knows nothing outside the region where it has seen
# data. In science we usually know something more: the equation the system obeys.
# A **soft constraint** puts that knowledge into the **loss function**: the network is
# penalised whenever it violates the equation, but nothing forces it to obey exactly.
# When the equation is a differential equation, this is called a
# **physics-informed neural network (PINN)**.
#
# ## The experiment
#
# We observe a damped oscillator only at the **beginning** of its motion, with noise:
#
# $$\ddot x + \mu \dot x + k x = 0, \qquad x(0)=1,\ \dot x(0)=0$$
#
# and we ask for $x(t)$ over a time window about three times longer than the observed
# part. The network $x_\theta(t)$ takes the time $t$ as input and returns the position.
# Three networks with the **same architecture and the same initial weights** are trained;
# only the loss differs:
#
# - **A, data only**: loss = mean squared error on the observed points.
# - **B, PINN**: same loss + the ODE residual evaluated (by automatic differentiation)
#   on collocation points spread over the **whole** time window.
# - **C, PINN with the wrong physics**: as B, but with a stiffness $k$ off by 45%.
#
# ## What to look for
#
# 1. A fits the points and is meaningless outside them; B follows the true solution
#    everywhere.
# 2. The model with the lowest training loss is the worst one.
# 3. A soft constraint is only as good as the physics you put in (model C).

# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)
t_start = time.time()

BLUE, RED, GREEN, ORANGE, GREY = "#5073B8", "#B85050", "#50B873", "#EBA05A", "#888888"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figures", exist_ok=True)      # each figure is also saved there as a PDF

# %% [markdown]
# ## The physical system and the data
#
# We write the oscillator with a damping rate $d$ and a natural frequency $\omega_0$, so
# that $\mu = 2d$ and $k = \omega_0^2$. Here $d = 2$ and $\omega_0 = 20$: the motion is
# under-damped, with a period of about $0.32$, and the exact solution is known in closed
# form. It is used **only** to generate the data and to judge the result, never for training.
#
# Three sets of points are defined below:
#
# - **observations**: 10 points on $[0, 0.36]$ (a bit more than one period), with Gaussian
#   noise of standard deviation $0.04$. This is all the data there is;
# - **collocation points**: 60 times spread over the whole window $[0, 1]$. They carry
#   **no measurement**: they are only the places where we will ask the network to satisfy
#   the equation, so they cost nothing;
# - a dense grid, for plotting and for the final error.

# %%
d, w0 = 2.0, 20.0          # damping rate and natural frequency
mu, k = 2 * d, w0**2       # ODE coefficients


def exact(t):
    """Analytic solution of the under-damped oscillator."""
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    return np.exp(-d * t) * 2 * A * np.cos(phi + w * t)


t_dense = np.linspace(0, 1, 500)
t_obs = np.linspace(0, 0.36, 10)
x_obs = exact(t_obs) + 0.04 * np.random.randn(len(t_obs))

t_obs_t = torch.tensor(t_obs, dtype=torch.float32).view(-1, 1)
x_obs_t = torch.tensor(x_obs, dtype=torch.float32).view(-1, 1)
t_col = torch.linspace(0, 1, 60).view(-1, 1).requires_grad_(True)   # collocation points
t_dense_t = torch.tensor(t_dense, dtype=torch.float32).view(-1, 1)

# %% [markdown]
# ## Model and training loop
#
# The network is a small multilayer perceptron (3 hidden layers of 32 `tanh` units).
# A smooth activation matters here: we need its second derivative with respect to the input.
#
# **The ODE residual.** For a given network, `ode_residual` computes
#
# $$r_\theta(t) = \ddot x_\theta(t) + \mu\, \dot x_\theta(t) + k\, x_\theta(t)$$
#
# The derivatives are obtained by `torch.autograd.grad`, applied twice: they are the exact
# derivatives of the network, there is no finite-difference step and no mesh. The option
# `create_graph=True` keeps these derivatives differentiable, so that the residual can
# itself be differentiated with respect to the weights during training. If the network
# were the exact solution, $r_\theta$ would be zero everywhere.
#
# **The loss.**
#
# $$\mathcal{L}(\theta) = \underbrace{\frac{1}{N_d}\sum_{i} \big(x_\theta(t_i) - x_i\big)^2}_{\text{data, on the 10 observations}}
# \; + \; \lambda \underbrace{\frac{1}{N_c}\sum_{j} r_\theta(t_j)^2}_{\text{physics, on the 60 collocation points}}$$
#
# Model A keeps only the first term. Two remarks:
#
# - $\lambda = 10^{-4}$ looks tiny, but the two terms do not have the same units: the
#   residual contains $k\,x$ with $k = 400$, so its square is of order $10^5$ for a wrong
#   function. $\lambda$ brings the two terms to a comparable scale. Choosing it is the
#   main practical difficulty of soft constraints.
# - There is no separate term for the initial conditions $x(0)$ and $\dot x(0)$. A
#   second-order ODE has a two-parameter family of solutions; here the observations
#   near $t=0$ select the right one.
#
# The three models are then trained with Adam for 20 000 steps (this is the slow cell,
# about one minute). The only difference between A, B and C is the loss.

# %%
def mlp():
    return torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )


def ode_residual(model, t, mu, k):
    x = model(t)
    dx = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    d2x = torch.autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
    return d2x + mu * dx + k * x


def train(physics=None, steps=20000, lam=1e-4):
    """physics=None: data only. physics=(mu, k): add the ODE residual to the loss."""
    torch.manual_seed(1)
    model = mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for step in range(steps):
        opt.zero_grad()
        loss_data = torch.mean((model(t_obs_t) - x_obs_t) ** 2)
        loss = loss_data
        if physics is not None:
            loss_phys = torch.mean(ode_residual(model, t_col, *physics) ** 2)
            loss = loss + lam * loss_phys
        loss.backward()
        opt.step()
        if step % 100 == 0:
            history.append((step, loss_data.item(),
                            loss_phys.item() * lam if physics is not None else np.nan))
    return model, np.array(history)


model_A, hist_A = train(physics=None)
model_B, hist_B = train(physics=(mu, k))
model_C, hist_C = train(physics=(mu, 0.55 * k))   # wrong stiffness: k off by 45%

with torch.no_grad():
    pred = {name: m(t_dense_t).numpy().ravel() for name, m in
            [("A", model_A), ("B", model_B), ("C", model_C)]}

for name in "ABC":
    rmse = np.sqrt(np.mean((pred[name] - exact(t_dense)) ** 2))
    print(f"model {name}: RMSE over the whole window = {rmse:.3f}")

# %% [markdown]
# The numbers above are root-mean-square errors against the exact solution over the whole
# window $[0, 1]$: about 0.66 for A, 0.03 for B and 0.28 for C, for an oscillation of
# amplitude 1.
#
# ## Results: with vs without the physics
#
# In the figure below, the grey band is the region where observations exist, the grey
# line is the exact solution and the coloured line is the network.
#
# - **A** goes through the observed points, then does something arbitrary: nothing in
#   its loss says anything about $t > 0.36$. This is plain extrapolation failure.
# - **B** continues the oscillation with the right period and the right decay, although
#   it has seen **no data** there. The green ticks are the collocation points: at each of
#   them the equation is (softly) enforced, and that is enough to propagate the
#   information contained in the first period to the rest of the window.

# %%
def panel(ax, y, color, title):
    ax.axvspan(0, t_obs.max(), color=GREY, alpha=0.08, lw=0)
    ax.plot(t_dense, exact(t_dense), color=GREY, lw=2, label="exact solution")
    ax.plot(t_dense, y, color=color, lw=2.2, label="network")
    ax.scatter(t_obs, x_obs, color=ORANGE, zorder=5, s=28, label="observations")
    ax.set_ylim(-1.3, 1.5)
    ax.set_xlabel("time $t$")
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
panel(axes[0], pred["A"], RED, "A: data only")
panel(axes[1], pred["B"], BLUE, "B: data + ODE residual (PINN)")
axes[1].scatter(t_col.detach().numpy().ravel(), np.full(60, -1.22), marker="|",
                color=GREEN, s=40, label="collocation points")
axes[0].set_ylabel("$x(t)$")
axes[1].legend(loc="upper right", fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig("figures/fig_I_oscillator.pdf")
plt.show()

# %% [markdown]
# ## What if the physics is wrong?
#
# Model C is trained exactly like B, but with $0.55\,k$ in the residual instead of $k$:
# the equation it is asked to satisfy oscillates at about $\sqrt{0.55} \approx 0.74$
# times the true frequency.
#
# The result is smooth and looks perfectly plausible, and it is wrong: in the observed
# region the network compromises between the data and the wrong equation (it misses
# several points), and beyond it follows the wrong equation, drifting out of phase with
# the truth. Nothing in the curve itself warns you. The only visible symptom is that data
# and physics cannot both be satisfied, which is the subject of the next figure.

# %%
fig, ax = plt.subplots(figsize=(5, 3.4))
panel(ax, pred["C"], RED, "C: PINN with the wrong $k$")
ax.set_ylabel("$x(t)$")
fig.tight_layout()
fig.savefig("figures/fig_I_wrong_physics.pdf")
plt.show()

# %% [markdown]
# ## Training curves: the lowest loss is not the best model
#
# The solid lines show the **data** term of the loss for A and B, the dashed line the
# (weighted) physics term of B.
#
# - A drives its data loss down to $10^{-8}$ or below. But the noise on the observations
#   has variance $0.04^2 \approx 1.6\times 10^{-3}$: a model that fits better than that is
#   fitting the **noise**. A has memorised its ten points.
# - B's data loss settles around $2\times 10^{-3}$, the noise level: the physics term
#   prevents it from chasing the noise. This is what a correct model should do.
#
# So the model with by far the lowest training loss gives by far the worst prediction.
# A training loss measures agreement with the training points, nothing else.

# %%
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.semilogy(hist_A[:, 0], hist_A[:, 1], color=RED, label="A: data loss")
ax.semilogy(hist_B[:, 0], hist_B[:, 1], color=BLUE, label="B: data loss")
ax.semilogy(hist_B[:, 0], hist_B[:, 2], color=BLUE, ls="--", label=r"B: $\lambda\,\times$ physics loss")
ax.set_xlabel("training step")
ax.set_ylabel("loss")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig_I_loss.pdf")
plt.show()

print(f"total runtime: {time.time() - t_start:.0f} s")

# %% [markdown]
# ## Take-home messages
#
# - A physics term in the loss can replace data that you do not have: here, 10 points on
#   one period plus the equation give the solution over three periods.
# - It is a **soft** constraint: the equation is satisfied approximately, at the
#   collocation points, and in competition with the data term through the weight $\lambda$.
# - The network is confident whether the physics is right or wrong. Checking the physics
#   is your job.
#
# ## Things to try
#
# Each of these needs a change of one line, then re-running from the training cell.
#
# 1. Change `lam` in `train` to `1e-2` and to `1e-6`. When does the PINN stop following the
#    data, and when does it stop following the physics?
# 2. Reduce the number of collocation points (`torch.linspace(0, 1, 60)`) to 10. What
#    happens between the points?
# 3. Extend the collocation points and the plotting grid to $t = 2$. Does the PINN still
#    extrapolate?
# 4. Change the wrong stiffness in model C from `0.55 * k` to `0.9 * k`. Could you still
#    tell from the figure that the physics is wrong?
# 5. Harder: make `k` a trainable parameter (a `torch.nn.Parameter` added to the
#    optimiser) and recover it from the data. This is the **inverse problem**, the case
#    where PINNs are most useful.
