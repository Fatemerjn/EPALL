# Experiment Summary

## Setup
| Field | Value |
| --- | --- |
| experiment_tag | thesis_fixed_schedule_c10_v1 |
| dataset | cifar10 |
| method | pall_modified, pall_original |
| arch | subnet_resnet18 |
| class_per_task | 2 |
| n_tasks | 5 |
| n_forget | 3 |
| seed | 0, 1 |
| num_runs | 4 |

## Key Metrics
| Method | Dataset | Seeds | Runs | Final Avg Acc | Avg Forgetting | Fu | WorstDrop | Au | Retrain Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pall_modified | cifar10 | 0, 1 | 2 | 0.9212 +/- 0.0148 | 0.0810 +/- 0.0127 | -0.0003 +/- 0.0077 | 0.0060 +/- 0.0049 | 0.4922 +/- 0.0110 | 9.4219 +/- 0.0326 |
| pall_original | cifar10 | 0, 1 | 2 | 0.9174 +/- 0.0198 | 0.0829 +/- 0.0151 | 0.0035 +/- 0.0027 | 0.0130 +/- 0.0099 | 0.4980 +/- 0.0028 | 0.2483 +/- 0.0152 |

## Automatic Observations
- cifar10: Final Avg Acc: pall_modified is better than pall_original (mean delta 0.0038).
- cifar10: Fu: pall_modified is better than pall_original (mean delta -0.0038).
- cifar10: WorstDrop: pall_modified is better than pall_original (mean delta -0.0070).
- cifar10: Retrain Time: pall_modified is worse than pall_original (mean delta 9.1736).
- These observations compare descriptive means only and do not imply statistical significance.
- Differences with absolute delta <= 0.0010 are reported as similar.
