# Experiment Summary

## Setup
| Field | Value |
| --- | --- |
| experiment_tag | thesis_fixed_schedule_c100_v1 |
| dataset | cifar100 |
| method | pall_modified, pall_original |
| arch | subnet_resnet18 |
| class_per_task | 5 |
| n_tasks | 10 |
| n_forget | 3 |
| seed | 0, 1 |
| num_runs | 4 |

## Key Metrics
| Method | Dataset | Seeds | Runs | Final Avg Acc | Avg Forgetting | Fu | WorstDrop | Au | Retrain Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pall_modified | cifar100 | 0, 1 | 2 | 0.5456 +/- 0.0188 | 0.0859 +/- 0.0004 | -0.0063 +/- 0.0032 | 0.0140 +/- 0.0085 | 0.1990 +/- 0.0014 | 11.3155 +/- 0.0357 |
| pall_original | cifar100 | 0, 1 | 2 | 0.4124 +/- 0.0143 | 0.1485 +/- 0.0191 | 0.0331 +/- 0.0057 | 0.0770 +/- 0.0127 | 0.2000 +/- 0.0000 | 0.1758 +/- 0.0001 |

## Automatic Observations
- cifar100: Final Avg Acc: pall_modified is better than pall_original (mean delta 0.1331).
- cifar100: Fu: pall_modified is better than pall_original (mean delta -0.0394).
- cifar100: WorstDrop: pall_modified is better than pall_original (mean delta -0.0630).
- cifar100: Retrain Time: pall_modified is worse than pall_original (mean delta 11.1397).
- These observations compare descriptive means only and do not imply statistical significance.
- Differences with absolute delta <= 0.0010 are reported as similar.
