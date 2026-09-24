# 00 · Setup

Everyone, 10 minutes. You leave with `hops` pointed at your team project, from your laptop and from the cluster terminal.

## Laptop

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install "hopsworks[python]" -r lectures/04-feature-store/requirements.txt
hops setup --host <HOST>
```

`hops setup` opens the browser. Log in, pick your team project, confirm the API key. The key lands in `~/.hops.toml` (mode 0600). Re-running it is a no-op; `--force` issues a new key.

No browser on the machine (SSH, cluster login node): add `--no-browser` and open the printed URL elsewhere.

```bash
hops project list
hops project use <team_project>
hops context
```

`hops context` prints what the project holds: feature groups, views, models, deployments, jobs. Run it whenever you lose track.

## Cluster terminal

In the Hopsworks UI: your project, then **Terminal**. The pod already has `hops`, `git`, `python`, `claude` and `codex`. No login: the terminal authenticates as you, in the project you opened it from.

```bash
git clone https://github.com/aghribi/artifact-hackathon-2026.git
cd artifact-hackathon-2026/lectures/04-feature-store
hops context
```

Laptop or terminal, the commands are the same. Pick the terminal when the data is big or the laptop is slow; pick the laptop when you want your own editor.

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

## Check

```bash
hops fg list
```

An empty table and no error means you are ready.
