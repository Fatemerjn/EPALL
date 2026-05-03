Correlation summary:
| metric | pearson_r | n |
| --- | --- | --- |
| WorstDrop | 0.0435 | 10 |
| avg_forgetting | 0.6442 | 10 |
| final_avg_acc | 0.0524 | 10 |
| Au | -0.1625 | 10 |

| dataset | method | experiment_tag | seed | S_share_ratio | S_share_crit_ratio | final_avg_acc | avg_forgetting | WorstDrop | Au | updated_param_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar100 | pall_adapter | ablation_cifar100_adapter_no_shared_e3_v1 | 0 | 0.0000 | 0.0000 | 0.4009 | 0.0564 | 0.0020 | 0.1580 | 0.0117 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_no_shared_e3_v1 | 1 | 0.0000 | 0.0000 | 0.3854 | 0.0731 | 0.0020 | 0.2100 | 0.0100 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_shared_critical_p005_e3_v1 | 0 | 0.5000 | 0.0342 | 0.4177 | 0.0818 | 0.0140 | 0.1540 | 0.0131 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_shared_critical_p005_e3_v1 | 1 | 0.5000 | 0.0413 | 0.4194 | 0.0742 | 0.0040 | 0.1780 | 0.0114 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_shared_critical_p010_e3_v1 | 0 | 0.5000 | 0.0664 | 0.4240 | 0.0804 | 0.0100 | 0.1400 | 0.0131 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_shared_critical_p010_e3_v1 | 1 | 0.5000 | 0.0813 | 0.4149 | 0.0758 | 0.0040 | 0.1920 | 0.0114 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_shared_critical_p020_e3_v1 | 0 | 0.5000 | 0.1357 | 0.4291 | 0.0792 | 0.0080 | 0.1440 | 0.0131 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_shared_critical_p020_e3_v1 | 1 | 0.5000 | 0.1429 | 0.3877 | 0.0913 | 0.0020 | 0.1840 | 0.0114 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_shared_no_protection_e3_v1 | 0 | 0.5000 | 0.0000 | 0.4254 | 0.0796 | 0.0100 | 0.1440 | 0.0131 |
| cifar100 | pall_adapter | ablation_cifar100_adapter_shared_no_protection_e3_v1 | 1 | 0.5000 | 0.0000 | 0.4151 | 0.0707 | 0.0020 | 0.1980 | 0.0114 |
