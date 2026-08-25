import time
import torch
import models
from .base import *
from .er import RehearsalMemory


class LoRA(Base):
    """Parameter-efficient LoRA baseline for continual learning + task unlearning.

    Mirrors ``PALLAdapter``'s train/forget skeleton on a frozen backbone, but with
    plain low-rank LoRA modules and a deliberately SIMPLE forget rule (no soft-mask
    or overlap protection -- this is a baseline):
      * ``learn()``  : train only the per-task LoRA module + the classifier;
      * ``forget()`` : reset the forgotten task's LoRA and its classifier rows,
        then a short ER-style replay retrain on the remaining tasks.
    """

    def __init__(self, args):
        super(LoRA, self).__init__(args)
        # Base builds self.net via the factory but cannot pass the method-specific
        # --lora_rank / --lora_alpha. Rebuild with the CLI values so the flags take
        # effect, without modifying the shared Base.
        self.net = models.__dict__[self.args.arch](
            self.args.class_per_task * self.args.n_tasks,
            n_tasks=self.args.n_tasks,
            sparsity=self.args.sparsity,
            norm_params=self.args.norm_params,
            lora_rank=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            pretrained_backbone=getattr(self.args, "pretrained_backbone", "none"),
            pretrained_weights=getattr(self.args, "pretrained_weights", None),
            input_dataset=self.args.dataset,
            pretrained_input_norm=self.args.pretrained_input_norm,
        ).to(self.device)

        self.memory = RehearsalMemory(
            buffer_size=self.args.mem_budget,
            n_tasks=self.args.n_tasks,
            cpt=self.args.class_per_task,
            dim_x=self.args.dim_input,
            device=self.device,
            mem_type=self.args.mem_type,
            save_logits=False,
        )
        self.k_shot = args.k_shot

        # A LoRA baseline trains the per-task LoRA modules AND the classifier.
        self.net.freeze_backbone(train_classifier=True)
        self.net.freeze_backbone_batchnorm()
        total = self.net.count_total_params()
        trainable = self.net.count_trainable_params()
        self.param_stats = {
            "total_params": int(total),
            "num_trainable_params": int(trainable),
            "num_adapter_params": int(self.net.count_adapter_params()),
            "lora_rank": int(self.args.lora_rank),
            "lora_alpha": float(self.args.lora_alpha),
            "trainable_param_ratio": float(trainable / total) if total else 0.0,
        }
        self.log_progress(
            "lora init: total_params={total} trainable_params={trainable} lora_params={lora} "
            "rank={rank} alpha={alpha} trainable_ratio={ratio:.6f}".format(
                total=self.param_stats["total_params"],
                trainable=self.param_stats["num_trainable_params"],
                lora=self.param_stats["num_adapter_params"],
                rank=self.args.lora_rank,
                alpha=self.args.lora_alpha,
                ratio=self.param_stats["trainable_param_ratio"],
            )
        )

    def init_optimizer(self):
        params = [param for param in self.net.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("lora has no trainable parameters after freezing the backbone.")
        if self.args.optim == "sgd":
            return SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        if self.args.optim == "adam":
            return Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        raise NotImplementedError

    def train_mode(self):
        self.train()
        self.net.freeze_backbone_batchnorm()

    def extract_logits_and_features(self, data_loader, task_id, norm_features=True):
        features, logits, targets = [], [], []
        with torch.no_grad():
            self.net.eval()
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred, feats = self.forward_with_features(x, task_id)
                if norm_features:
                    feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
                features.append(feats)
                logits.append(pred)
                targets.append(y)
            features, logits, targets = torch.cat(features), torch.cat(logits), torch.cat(targets)
        self.train_mode()
        return features.cpu(), logits.cpu(), targets.cpu()

    def fill_buffer(self, task_id, dataset):
        sel_loader = self.build_dataloader(dataset, shuffle=False, context=f"fill_buffer_task_{task_id}")
        _, _, targets = self.extract_logits_and_features(sel_loader, task_id)
        if self.args.mem_type == "random":
            sel_indices = self.memory.select_indices_by_random(targets)
        else:
            raise NotImplementedError
        x, y = zip(*(sel_loader.dataset[idx] for idx in sel_indices))
        self.memory.add((x, y), task_id)

    def learn(self, task_id, dataset):
        loader = self.build_dataloader(dataset, shuffle=True, context=f"learn_task_{task_id}")
        self.opt = self.init_optimizer()
        train_start = time.perf_counter()
        self.log_progress(f"task training start: task={task_id} epochs={self.args.n_epochs} steps_per_epoch={len(loader)}")
        self.train_mode()
        for epoch in range(self.args.n_epochs):
            epoch_start = time.perf_counter()
            self.log_epoch_start(task_id, epoch, self.args.n_epochs)
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.forward(x, task_id), y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.log_epoch_end(task_id, epoch, self.args.n_epochs, epoch_start)
        self.fill_buffer(task_id, dataset)
        self.prev_tasks.append(task_id)
        self.log_progress(
            f"task training end: task={task_id} elapsed={self._format_elapsed(self._elapsed_since(train_start))}"
        )

    def _compute_retrain_steps(self):
        steps = self.args.retrain_steps
        if self.args.retrain_epochs is not None:
            steps = self.args.retrain_epochs
        if steps is None:
            steps = self.k_shot
        return int(steps)

    def _count_active_trainable_params(self, active_tasks):
        lora_params = sum(self.net.count_adapter_params(task_id) for task_id in active_tasks)
        classifier_params = len(active_tasks) * self.cpt * self.net.feature_dim
        return int(lora_params + classifier_params)

    def forget(self, task_id):
        self._forget_impl(task_id)

    def forget_with_diagnostics(self, task_id, eval_fn=None, debug_context=None, remaining_tasks=None):
        del debug_context
        return self._forget_impl(task_id, eval_fn=eval_fn, remaining_tasks=remaining_tasks, return_info=True)

    def _forget_impl(self, task_id, eval_fn=None, remaining_tasks=None, return_info=False):
        if task_id not in self.prev_tasks:
            raise AssertionError(f"[ERROR] {task_id} is not learned yet")
        forget_start = time.perf_counter()
        self.log_progress(f"forget phase start: task={task_id}")
        active_tasks = (
            list(remaining_tasks)
            if remaining_tasks is not None
            else [t for t in self.prev_tasks if t != task_id]
        )
        reset_param_count = self.net.count_adapter_params(task_id)

        # (1) Reset: erase the forgotten task's LoRA module and its classifier rows.
        t_reset_start = time.perf_counter()
        self.net.reset_task_lora(task_id)
        self.net.reset_classifier_slice(task_id, self.cpt)
        t_reset = time.perf_counter() - t_reset_start

        info = {
            "t_reset": t_reset,
            "t_retrain": 0.0,
            "num_updated_params": 0,
            "reset_param_count": int(reset_param_count),
            "finetune_diag": {
                "active_tasks": active_tasks,
                "deleted_task_id": task_id,
                "retrain_steps": 0,
                "buffer_sizes": {},
            },
        }

        if eval_fn is not None:
            info["after_reset_eval"] = self.run_rng_neutral_diagnostic(eval_fn, "after_reset")

        if task_id in self.memory.buffer:
            self.memory.remove(task_id)
        if task_id in self.prev_tasks:
            self.prev_tasks.remove(task_id)

        # (2) Short ER-style replay retrain on the remaining tasks.
        retrain_steps = self._compute_retrain_steps()
        buffer_sizes = {}
        for active_task in active_tasks:
            entry = self.memory.buffer.get(active_task) if self.memory.buffer else None
            if entry is None:
                buffer_sizes[str(active_task)] = 0
                continue
            buffer_sizes[str(active_task)] = min(int(entry.get("num_seen", 0)), int(entry["X"].shape[0]))
        info["finetune_diag"]["buffer_sizes"] = buffer_sizes
        info["finetune_diag"]["retrain_steps"] = retrain_steps

        can_finetune = bool(active_tasks) and retrain_steps > 0 and any(size > 0 for size in buffer_sizes.values())
        if can_finetune:
            finetune_opt = self.init_optimizer()
            info["num_updated_params"] = self._count_active_trainable_params(active_tasks)
            t_retrain_start = time.perf_counter()
            self.log_progress(
                f"retrain phase start: task={task_id} steps={retrain_steps} active_tasks={active_tasks}"
            )
            self.train_mode()
            for _ in range(retrain_steps):
                finetune_opt.zero_grad()
                loss = 0.0
                for active_task in active_tasks:
                    batch_size = max(1, self.args.batch_size // len(active_tasks))
                    x_past, y_past = self.memory.sample_task(batch_size, active_task)
                    loss = loss + self.loss_fn(self.forward(x_past, active_task), y_past) / len(active_tasks)
                loss.backward()
                finetune_opt.step()
            info["t_retrain"] = time.perf_counter() - t_retrain_start
            self.log_progress(
                f"retrain phase end: task={task_id} elapsed={self._format_elapsed(info['t_retrain'])}"
            )
        else:
            self.log_progress(f"retrain phase skipped: task={task_id} active_tasks={active_tasks}")

        info["t_forget_total"] = time.perf_counter() - forget_start
        self.log_progress(
            f"forget phase end: task={task_id} elapsed={self._format_elapsed(info['t_forget_total'])}"
        )
        return info if return_info else None
