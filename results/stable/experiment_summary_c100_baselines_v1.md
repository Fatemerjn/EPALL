# Experiment Summary

## Setup
| Field | Value |
| --- | --- |
| experiment_tag | thesis_c100_baselines_v1 |
| dataset | cifar100 |
| method | derpp, er |
| arch | resnet18 |
| class_per_task | 5 |
| n_tasks | 10 |
| n_forget | 3 |
| seed | 0, 1 |
| num_runs | 4 |

## Key Metrics
| Method | Dataset | Seeds | Runs | Final Avg Acc | Avg Forgetting | Fu | WorstDrop | Au | Retrain Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| derpp | cifar100 | 0, 1 | 2 | 0.5461 +/- 0.0030 | 0.1090 +/- 0.0003 | 0.0027 +/- 0.0160 | 0.0610 +/- 0.0354 | 0.3800 +/- 0.0311 | 0.0000 +/- 0.0000 |
| er | cifar100 | 0, 1 | 2 | 0.4944 +/- 0.0309 | 0.1549 +/- 0.0202 | 0.0213 +/- 0.0297 | 0.0580 +/- 0.0396 | 0.3440 +/- 0.0085 | 0.0000 +/- 0.0000 |

## Automatic Observations
- No direct pall_modified vs pall_original comparison was available in the input CSV.
- These observations compare descriptive means only and do not imply statistical significance.
- Differences with absolute delta <= 0.0010 are reported as similar.
