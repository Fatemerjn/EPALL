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
| num_runs | 6 |

## Key Metrics
| Method | Dataset | Seeds | Runs | Final Avg Acc | Avg Forgetting | Fu | WorstDrop | Au | Retrain Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pall_modified | cifar100 | 0, 1 | 3 | 0.5169 +/- 0.0515 | 0.0803 +/- 0.0098 | -0.0044 +/- 0.0040 | 0.0140 +/- 0.0060 | 0.1993 +/- 0.0012 | 11.3027 +/- 0.0336 |
| pall_original | cifar100 | 0, 1 | 3 | 0.3869 +/- 0.0454 | 0.1343 +/- 0.0280 | 0.0288 +/- 0.0086 | 0.0767 +/- 0.0090 | 0.2000 +/- 0.0000 | 0.1733 +/- 0.0044 |

## Automatic Observations
- cifar100: Final Avg Acc: pall_modified is better than pall_original (mean delta 0.1300).
- cifar100: Fu: pall_modified is better than pall_original (mean delta -0.0331).
- cifar100: WorstDrop: pall_modified is better than pall_original (mean delta -0.0627).
- cifar100: Retrain Time: pall_modified is worse than pall_original (mean delta 11.1295).
- These observations compare descriptive means only and do not imply statistical significance.
- Differences with absolute delta <= 0.0010 are reported as similar.
