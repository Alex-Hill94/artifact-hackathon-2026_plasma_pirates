# 01 · Read the real data

Everyone, 10 minutes, from your laptop. The full challenge datasets are in three projects, as feature groups.

| Project | Feature group | Rows | One row is |
|---|---|---|---|
| `m4cast_ess` | `ess_llrf_waveforms` | 52,842 | a cycle of an RF chain, I and Q at 256 points |
| | `ess_sds_channel_stats` | 497,659 | a channel of a cycle, full-resolution statistics |
| `m4cast_clara` | `clara_camera_captures` | 225,421 | a camera capture: spot moments, magnet settings |
| | `clara_camera_files` | 17,878 | a burst at fixed settings, with a thumbnail |
| `m4cast_pallas` | `pallas_campaign_a` | 13,073 | a Smilei simulation, scan A |
| | `pallas_campaign_b` | 5,286 | a Smilei simulation, scan B |
| | `pallas_b_trajectories` | 322,615 | the bunch at one position along the accelerator |
| | `pallas_b_moments` | 211,360 | the bunch's moments at one plane |

The raw files stay at CC-IN2P3. These rows were computed there, next to them, and only the rows came to the cluster.

## From the command line

```bash
hops project use m4cast_ess
hops fg list
hops fg preview ess_llrf_waveforms --n 3 --columns event_id,cycle_id,pv_base,beam_mode,in_sample
```

```
event_id     cycle_id     pv_base                                    beam_mode  in_sample
-----------  -----------  -----------------------------------------  ---------  ---------
15642663250  15642663248  MBL-010RFC:RFS-DIG-101:Dwn0-CmpSDS-PMData  Probe      False
```

## From Python

```python
import hopsworks

project = hopsworks.login(project="m4cast_ess")
fs = project.get_feature_store()
waves = fs.get_feature_group("ess_llrf_waveforms", version=1).read()
waves.shape     # (52842, 24), about 6 s
```

The same three lines read CLARA and PALLAS: change the project and the name.

| Read | Time from a laptop |
|---|---|
| `m4cast_pallas.pallas_b_trajectories` | 3 s |
| `m4cast_ess.ess_llrf_waveforms` | 5 to 6 s |
| `m4cast_clara.clara_camera_captures` (193 columns) | 11 to 23 s |
| `m4cast_clara.clara_camera_files` (a thumbnail per row) | 16 s to 2 min |

`clara_camera_files` is for the UI: its data preview shows the thumbnails. In code, read `clara_camera_captures`.

## What these projects let you do

In the three `m4cast_*` projects you are a Data scientist: you read, build feature views, train and register models. You cannot create a feature group or a deployment there. Track 03 gives you a project where you can.
