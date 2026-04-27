# Experiment Summary

## Setup
| Field | Value |
| --- | --- |
| experiment_tag | thesis_c100_baselines_v1, thesis_c100_bestcfg_v1, thesis_fixed_schedule_c100_v1 |
| dataset | cifar100 |
| method | derpp, er, pall_modified, pall_original |
| arch | resnet18, subnet_resnet18 |
| class_per_task | 5 |
| n_tasks | 10 |
| n_forget | 3 |
| seed | 0, 1 |
| num_runs | 12 |

## Key Metrics
| Method | Dataset | Seeds | Runs | Final Avg Acc | Avg Forgetting | Fu | WorstDrop | Au | Retrain Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| derpp | cifar100 | 0, 1 | 2 | 0.5461 +/- 0.0030 | 0.1090 +/- 0.0003 | 0.0027 +/- 0.0160 | 0.0610 +/- 0.0354 | 0.3800 +/- 0.0311 | 0.0000 +/- 0.0000 |
| er | cifar100 | 0, 1 | 2 | 0.4944 +/- 0.0309 | 0.1549 +/- 0.0202 | 0.0213 +/- 0.0297 | 0.0580 +/- 0.0396 | 0.3440 +/- 0.0085 | 0.0000 +/- 0.0000 |
| pall_modified | cifar100 | 0, 1 | 4 | 0.5456 +/- 0.0153 | 0.0859 +/- 0.0003 | -0.0063 +/- 0.0026 | 0.0140 +/- 0.0069 | 0.1990 +/- 0.0012 | 12.0668 +/- 0.4863 |
| pall_original | cifar100 | 0, 1 | 4 | 0.4124 +/- 0.0117 | 0.1485 +/- 0.0156 | 0.0331 +/- 0.0046 | 0.0770 +/- 0.0104 | 0.2000 +/- 0.0000 | 0.1758 +/- 0.0001 |

## Automatic Observations
- cifar100: Final Avg Acc: pall_modified is better than pall_original (mean delta 0.1331).
- cifar100: Fu: pall_modified is better than pall_original (mean delta -0.0394).
- cifar100: WorstDrop: pall_modified is better than pall_original (mean delta -0.0630).
- cifar100: Retrain Time: pall_modified is worse than pall_original (mean delta 11.8910).
- These observations compare descriptive means only and do not imply statistical significance.
- Differences with absolute delta <= 0.0010 are reported as similar.
