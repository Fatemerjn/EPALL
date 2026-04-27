# Experiment Summary

## Setup
| Field | Value |
| --- | --- |
| experiment_tag | ablation_c100_t10_e1_s0_v1 |
| dataset | cifar100 |
| method | pall_modified |
| arch | subnet_resnet18 |
| class_per_task | 5 |
| n_tasks | 10 |
| n_forget | 3 |
| seed | 0 |
| num_runs | 4 |

## Key Metrics
| Method | Dataset | Seeds | Runs | Final Avg Acc | Avg Forgetting | Fu | WorstDrop | Au | Retrain Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pall_modified | cifar100 | 0 | 4 | 0.4556 +/- 0.0035 | 0.0719 +/- 0.0046 | 0.0031 +/- 0.0036 | 0.0195 +/- 0.0044 | 0.2000 +/- 0.0000 | 10.2613 +/- 1.1523 |

## Automatic Observations
- No direct pall_modified vs pall_original comparison was available in the input CSV.
- These observations compare descriptive means only and do not imply statistical significance.
- Differences with absolute delta <= 0.0010 are reported as similar.
