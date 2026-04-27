# Adapter Prototype Notes

`pall_adapter` is a minimal selective-forgetting prototype with a frozen ResNet backbone and per-task bottleneck adapters.

Curated adapter outputs live under `results/adapter/`.

Known limitation:

- `pall_adapter uses adapter reset prototype, not full PALL overlap mask yet.`

Smoke test:

```bash
python3 main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 1 --n_epochs 1 --seed 0 --method pall_adapter
```

Fixed-schedule adapter run:

```bash
python3 main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --arch resnet18 --method pall_adapter --seed 0 --deterministic \
  --request_schedule_file schedules/cifar10_t5_f3_fixed_seed0.json \
  --adapter_bottleneck 16 --adapter_train_classifier
```

Bottleneck ablation examples:

```bash
python3 main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --arch resnet18 --method pall_adapter --seed 0 --deterministic \
  --request_schedule_file schedules/cifar10_t5_f3_fixed_seed0.json \
  --adapter_bottleneck 8 --adapter_train_classifier
```

```bash
python3 main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --arch resnet18 --method pall_adapter --seed 0 --deterministic \
  --request_schedule_file schedules/cifar10_t5_f3_fixed_seed0.json \
  --adapter_bottleneck 16 --adapter_train_classifier
```

Optional classifier training:

```bash
python3 main.py --dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 1 --n_epochs 1 --seed 0 --method pall_adapter --adapter_train_classifier
```
