import inspect
import sys
import random
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import numpy as np
import models
import time


def _seed_worker(worker_id):
    """Deterministically reseed NumPy/Python RNGs inside each DataLoader worker.

    PyTorch seeds ``torch`` in every worker from the parent generator, but it
    does NOT reseed NumPy or the ``random`` module. Any augmentation or sampling
    that relies on those (common in CL data pipelines) therefore becomes
    nondeterministic once ``num_workers > 0``. Deriving the per-worker seed from
    ``torch.initial_seed()`` restores full reproducibility across seeds/epochs.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def grad_l2_norm_ratio(deleted_grads, active_grads, eps=1e-12):
    """Gradient Norm Ratio metric (reviewer item): the L2 norm of the forget-task
    gradient over the retain-task gradient, both taken over the same set of shared
    parameters. ``deleted_grads``/``active_grads`` are dicts {param_name: tensor}
    (as produced by the PALL gradient-importance helpers). No new backward passes:
    it reuses tensors already computed during forgetting. Returns a float, or
    ``None`` if either gradient dict is empty/unavailable."""
    def _concat_norm(grads):
        if not grads:
            return None
        flats = [g.reshape(-1) for g in grads.values() if g is not None]
        if not flats:
            return None
        return float(torch.cat(flats).norm(p=2).item())

    deleted_norm = _concat_norm(deleted_grads)
    active_norm = _concat_norm(active_grads)
    if deleted_norm is None or active_norm is None:
        return None
    return deleted_norm / (active_norm + eps)


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

    def _loader_generator(self):
        """Lazily build a CPU generator seeded from the run seed.

        Passing an explicit generator makes shuffling order a deterministic
        function of the run seed, decoupled from global RNG consumption
        elsewhere in the pipeline (so adding/removing a forward pass cannot
        silently change which samples a loader yields).
        """
        if getattr(self, "_dataloader_generator", None) is None:
            seed = int(getattr(self.args, "seed", 0) or 0)
            self._dataloader_generator = torch.Generator()
            self._dataloader_generator.manual_seed(seed)
        return self._dataloader_generator

    def get_dataloader_settings(self, batch_size=None, shuffle=False):
        num_workers = self.resolve_num_workers()
        pin_memory = self.resolve_pin_memory()
        settings = {
            "batch_size": self.args.batch_size if batch_size is None else batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            # Tear workers down between loaders to bound resident memory on long,
            # heavy runs; the spawn cost is negligible relative to epoch time.
            "persistent_workers": False,
        }
        if num_workers > 0:
            settings["worker_init_fn"] = _seed_worker
        # A seeded generator only affects shuffling; attach it whenever we shuffle.
        if shuffle:
            settings["generator"] = self._loader_generator()
        return settings

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
