# 00 · Setup

Everyone, 10 minutes. You leave with `hops` pointed at the hackathon cluster, from your laptop and from the cluster terminal.

The cluster is `https://aissai26.hops.live`. No VPN.

## Log in first

Open `https://aissai26.hops.live` and use **Login with Auth0**. The **Register** button on the same page does not work on this cluster.

On first login you are a member of three projects: `m4cast_ess`, `m4cast_clara` and `m4cast_pallas`. They hold the real challenge data.

## Laptop

```bash
git clone -b lecture-04-feature-store https://github.com/MagicLex/artifact-hackathon-2026
cd artifact-hackathon-2026/lectures/04-feature-store
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
hops setup --host https://aissai26.hops.live
```

`requirements.txt` pins the Hopsworks client the workshop was tested with. The client on PyPI ignores the key that `hops setup` stores, so `hopsworks.login()` would ask for one.

`hops setup` opens the browser. Pick a project, give the API key a name, confirm. The key lands in `~/.hops.toml` (mode 0600).

| If you see | Do |
|---|---|
| The login page instead of "Complete CLI Setup" | Log in, then open the URL that `hops setup` printed again |
| `Request failed with status code 400` in the browser | The key name is already used. Type another one |
| `Token flow failed: ... Read timed out` in the terminal | Run `hops setup` again |

No browser on the machine (SSH, a login node): add `--no-browser` and open the printed URL elsewhere.

```bash
hops project use m4cast_pallas
hops fg list
hops context
```

`hops context` prints what the project holds: feature groups, views, models, deployments, jobs. Run it whenever you lose track.

## Cluster terminal

In the Hopsworks UI: your project, then **Terminal**, then **Start Terminal**. The pod has `hops`, `git`, `python`, `uv`, `claude` and `codex`, and the same client as `requirements.txt`. No login: the terminal authenticates as you, in the project you opened it from.

```bash
git clone -b lecture-04-feature-store https://github.com/MagicLex/artifact-hackathon-2026
cd artifact-hackathon-2026/lectures/04-feature-store
uv pip install -r requirements.txt
hops fg list
```

The install takes a few seconds: the Hopsworks client is already there, it adds the libraries the use cases need (scikit-optimize for the PALLAS and CLARA searches). The ESS autoencoder also needs PyTorch, which is not in `requirements.txt`: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`, about 6 seconds in the terminal. A terminal session keeps what you install until it is stopped.

The challenge files are mounted read-only in every terminal: `/m4cast` (samples, the PALLAS pack) and `/m4cast-results` (the tables the feature groups were filled from).

## What runs where

| | Laptop | Cluster terminal |
|---|---|---|
| Read feature groups, `hops fg preview` | yes | yes |
| Feature views, training data | yes | yes |
| Train, register a model | yes | yes |
| Insert into a feature group | no | yes |
| Deploy a model, call a deployment | no | yes |

From outside, the cluster answers on the web port and on the query service only. Writes and model serving stay inside it.

## Coding agents

Any agent that can run a shell can drive Hopsworks through `hops`. Add `--json` to any command for machine-readable output.

```bash
hops skills list            # playbooks for feature groups, views, training, serving
hops skills show hops-fg    # print one; paste it into any agent
hops init --dir .           # Claude Code: installs the hops skill and allows Bash(hops *)
```

Docs for agents, over MCP: `https://mcp.hopsworks.ai/mcp`.

```bash
claude mcp add --transport http hopsworks-docs -s user https://mcp.hopsworks.ai/mcp
```

In the cluster terminal the skills are already installed for `claude` and `codex`.
