import inspect
import sys
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import numpy as np
import models
import time


class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        model_kwargs = {
            "n_tasks": args.n_tasks,
            "sparsity": args.sparsity,
            "norm_params": args.norm_params,
        }
        if str(args.arch).startswith("adapter_"):
            model_kwargs.update(
                {
                    "adapter_bottleneck": args.adapter_bottleneck,
                    "adapter_shared_bottleneck": args.adapter_shared_bottleneck,
                    "adapter_location": args.adapter_location,
                }
            )
        self.net = models.__dict__[args.arch](args.class_per_task * args.n_tasks, **model_kwargs)
        self.device = args.device
        self.n_tasks = args.n_tasks
        self.cpt = args.class_per_task
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.task_status = {}
        self.prev_tasks = []
        self.n_iters = 1
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_reduction_none = nn.CrossEntropyLoss(reduction='none')
        self.opt = None
        self.scheduler = None
        self._is_macos = sys.platform == "darwin"
        self._lifecycle_start = time.perf_counter()

    def _elapsed_since(self, start_time):
        return time.perf_counter() - start_time

    def _format_elapsed(self, seconds):
        return f"{float(seconds):.2f}s"

    def log_progress(self, message):
        elapsed = self._format_elapsed(self._elapsed_since(self._lifecycle_start))
        print(f"[INFO] [{self.__class__.__name__}] +{elapsed} {message}", flush=True)

    def resolve_num_workers(self):
        if getattr(self.args, "num_workers", None) is not None:
            return int(self.args.num_workers)
        if self._is_macos:
            return 0
        return 2

    def resolve_pin_memory(self):
        if getattr(self.args, "pin_memory", None) is not None:
            return bool(self.args.pin_memory)
        if self._is_macos:
            return False
        return self.device.type == "cuda"

    def get_dataloader_settings(self, batch_size=None, shuffle=False):
        num_workers = self.resolve_num_workers()
        pin_memory = self.resolve_pin_memory()
        return {
            "batch_size": self.args.batch_size if batch_size is None else batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": False if num_workers > 0 else False,
        }

    def build_dataloader(self, dataset, batch_size=None, shuffle=False, context=""):
        settings = self.get_dataloader_settings(batch_size=batch_size, shuffle=shuffle)
        loader = DataLoader(dataset, **settings)
        self.log_progress(
            "dataloader built: context={context} dataset_size={size} batch_size={batch_size} "
            "shuffle={shuffle} num_workers={num_workers} pin_memory={pin_memory} "
            "persistent_workers={persistent_workers} device={device}".format(
                context=context or "default",
                size=len(dataset),
                batch_size=settings["batch_size"],
                shuffle=settings["shuffle"],
                num_workers=settings["num_workers"],
                pin_memory=settings["pin_memory"],
                persistent_workers=settings["persistent_workers"],
                device=self.device,
            )
        )
        return loader

    def log_epoch_start(self, task_id, epoch, total_epochs, phase="training"):
        self.log_progress(f"{phase} epoch start: task={task_id} epoch={epoch + 1}/{total_epochs}")

    def log_epoch_end(self, task_id, epoch, total_epochs, epoch_start_time, phase="training"):
        self.log_progress(
            f"{phase} epoch end: task={task_id} epoch={epoch + 1}/{total_epochs} "
            f"elapsed={self._format_elapsed(self._elapsed_since(epoch_start_time))}"
        )

    def _forward_net(self, net, x, task, **kwargs):
        supports_task = getattr(net, "_supports_task_arg", None)
        if supports_task is None:
            try:
                supports_task = "task" in inspect.signature(net.forward).parameters
            except (TypeError, ValueError):
                supports_task = False
            net._supports_task_arg = supports_task
        if supports_task:
            return net(x, task=task, **kwargs)
        return net(x, **kwargs)

    def init_optimizer(self):
        if self.args.optim == "sgd":
            return SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.args.optim == "adam":
            return Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def forward(self, x, task):
        out = self._forward_net(self.net, x, task)
        if task > 0:
            out[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            out[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return out

    def forward_with_features(self, x, task):
        out, features = self._forward_net(self.net, x, task, returnt='all')
        if task > 0:
            out[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            out[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return out, features

    def evaluate(self, x, task):
        return self.forward(x, task)  # default to the forward pass

    def eval_mode(self):
        self.eval()

    def train_mode(self):
        self.train()

    def learn(self, task_id, dataset):
        return  # default: do nothing when we want to learn a task

    def forget(self, task_id):
        return  # default: do nothing when we want to forget a task

    def privacy_aware_lifelong_learning(self, task_id, dataset, learn_type):
        t0 = time.perf_counter()
        if learn_type == "T":
            self.log_progress(f"task operation start: task={task_id} mode=train")
            if task_id not in self.task_status:  # first time learning the task
                self.task_status[task_id] = learn_type
                self.learn(task_id, dataset)
            else:  # second time consolidate - we do not explore the impact of repetition yet
                raise NotImplementedError
        else:  # learn type is "F" forget
            assert learn_type == "F", f"[ERROR] unknown learning type {learn_type}"
            assert task_id in self.task_status, f"[ERROR] {task_id} was not learned"
            self.log_progress(f"task operation start: task={task_id} mode=forget")
            self.task_status[task_id] = "F"
            self.forget(task_id)
        self.log_progress(
            f"task operation end: task={task_id} mode={learn_type} elapsed={self._format_elapsed(self._elapsed_since(t0))}"
        )
