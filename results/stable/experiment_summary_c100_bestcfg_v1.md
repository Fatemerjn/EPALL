# Experiment Summary

## Setup
| Field | Value |
| --- | --- |
| experiment_tag | thesis_c100_bestcfg_v1 |
| dataset | cifar100 |
| method | pall_modified |
| arch | subnet_resnet18 |
| class_per_task | 5 |
| n_tasks | 10 |
| n_forget | 3 |
| seed | 0, 1 |
| num_runs | 2 |

## Key Metrics
| Method | Dataset | Seeds | Runs | Final Avg Acc | Avg Forgetting | Fu | WorstDrop | Au | Retrain Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pall_modified | cifar100 | 0, 1 | 2 | 0.5456 +/- 0.0188 | 0.0859 +/- 0.0004 | -0.0063 +/- 0.0032 | 0.0140 +/- 0.0085 | 0.1990 +/- 0.0014 | 12.0668 +/- 0.5956 |

## Automatic Observations
- No direct pall_modified vs pall_original comparison was available in the input CSV.
- These observations compare descriptive means only and do not imply statistical significance.
- Differences with absolute delta <= 0.0010 are reported as similar.
