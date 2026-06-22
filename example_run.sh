#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt


# Example script to run PALL for CIFAR-10 (5x2) simulations with # forgetting requests = 3
python -u main.py --dataset 'cifar10' --class_per_task 2 --n_tasks 5 --n_forget 3 --arch 'resnet18' \
                  --method 'pall' --sparsity 0.8 --k_shot 50 --alpha 0.5 --beta 1.0 --mem_budget 500 \
                  --optim sgd --momentum 0.9 --lr 1e-2 --weight_decay 5e-4 --batch_size 32 --n_epochs 20


# Example script to run PALL for CIFAR-100 (10x5) simulations with # forgetting requests = 3
# NOTE: CIFAR-100 is constrained to class_per_task == 5 (see main.py); the paper
# protocol is 10 tasks x 5 classes. The previous 10x10 config now errors out.
python -u main.py --dataset 'cifar100' --class_per_task 5 --n_tasks 10 --n_forget 3 --arch 'resnet34' \
                  --method 'pall' --sparsity 0.9 --k_shot 50 --alpha 0.5 --beta 1.0 --mem_budget 500 \
                  --optim sgd --momentum 0.9 --lr 1e-2 --weight_decay 5e-4 --batch_size 32 --n_epochs 20


# Example script to run PALL for TinyImageNet (20x10) simulations with # forgetting requests = 3
python -u main.py --dataset 'tinyimagenet' --class_per_task 10 --n_tasks 20 --n_forget 3 --arch 'vit_t_8' \
                  --method 'pall' --sparsity 0.95 --k_shot 50 --alpha 0.5 --beta 1.0 --mem_budget 1000 \
                  --optim adam --lr 1e-3 --weight_decay 0.0 --batch_size 256 --n_epochs 100
