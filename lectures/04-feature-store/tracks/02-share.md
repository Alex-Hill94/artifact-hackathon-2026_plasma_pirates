# 02 · Share: one team writes, every team reads

15 minutes. Read features another project owns, then publish your own. No copies: a shared feature store is read-only, and the owner keeps control.

## Read the organisers' store

The organisers load the full LLRF release (all five tables) into the project `aissai_llrf` and share its feature store with every team project.

```bash
hops fg list        # the STORE column shows where each feature group lives
```

```python
import hopsworks

project = hopsworks.login()
shared = project.get_feature_store(name="aissai_llrf_featurestore")

precursor = shared.get_feature_group("llrf_precursor", version=1)
precursor.read().shape      # (2865, 504)
```

Feature groups from a shared store join with your own in a feature view like any other. Track 03 works the same whether you wrote the LLRF groups yourself or read them from here.

## Publish yours

Share a whole feature group, or only some columns. Primary key and event time always go with it. Sharing needs the Data Owner role in your project.

```python
fs = project.get_feature_store()
history = fs.get_feature_group("llrf_cavity_history", version=1)

history.share("<twin_team_project>")
history.share("<other_project>", features=["faults_7d", "hours_since_last_event"])
history.shared_with()
history.unshare("<other_project>")
```

Or everything at once: `fs.share("<twin_team_project>")`.

In the UI: Project settings, **Shared with other projects**, **Share feature store**.

## Who reads what

| Writer | Feature groups | Readers |
|---|---|---|
| organisers | `llrf_*` (SPIRAL2 release) | every team |
| anomaly teams | cavity history, precursor features you engineer | the other anomaly team |
| optimisation teams | `quad_scan_runs` (track 05) | surrogate teams |
| surrogate teams | surrogate predictions, residuals | optimisation teams |

Both optimisation teams and both surrogate teams run the same Cheetah lattice. Runs one team simulated are training data for the other.

## Check

```bash
hops fg lineage llrf_cavity_history --version 1
```
