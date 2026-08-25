import time

import torch
import torch.nn.functional as F

from .base import Base
from .er import RehearsalMemory
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class SalUn(Base):
    """Saliency Unlearning baseline.

    SalUn builds a gradient-based saliency mask on the forgotten task, then
    performs a short masked update on only the most salient weights toward
    uniform or random labels for the forget data.
    """

    def __init__(self, args):
        super(SalUn, self).__init__(args)
        self.memory = RehearsalMemory(
            buffer_size=self.args.mem_budget,
            n_tasks=self.args.n_tasks,
            cpt=self.args.class_per_task,
            dim_x=self.args.dim_input,
            device=self.device,
            mem_type=self.args.mem_type,
            save_logits=False,
        )

    def _trainable_params(self):
        return [(name, param) for name, param in self.net.named_parameters() if param.requires_grad]

    def _buffer_tensors(self, task_id):
        if task_id not in self.memory.buffer:
            return None, None
        block = self.memory.buffer[task_id]
        count = min(int(block["num_seen"]), int(block["X"].shape[0]))
        return block["X"][:count], block["Y"][:count]

    def fill_buffer(self, task_id, dataset):
        sel_loader = self.build_dataloader(dataset, shuffle=False, context=f"fill_buffer_task_{task_id}")
        targets = []
        for _x, y in sel_loader:
            targets.append(y)
        targets = torch.cat(targets, dim=0)
        if self.args.mem_type != "random":
            raise NotImplementedError
        selected = self.memory.select_indices_by_random(targets.numpy())
        x, y = zip(*(sel_loader.dataset[idx] for idx in selected))
        self.memory.add((x, y), task_id)

    def learn(self, task_id, dataset):
        loader = self.build_dataloader(dataset, shuffle=True, context=f"learn_task_{task_id}")
        self.opt = self.init_optimizer()
        train_start = time.perf_counter()
        self.log_progress(f"task training start: task={task_id} epochs={self.args.n_epochs} steps_per_epoch={len(loader)}")

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        for epoch in range(self.args.n_epochs):
            epoch_start = time.perf_counter()
            self.log_epoch_start(task_id, epoch, self.args.n_epochs)
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.forward(x, task_id), y)

                n_prev_tasks = len(self.prev_tasks)
                if n_prev_tasks > 0:
                    replay_batch = max(1, self.args.batch_size // n_prev_tasks)
                    for prev_task in self.prev_tasks:
                        x_past, y_past = self.memory.sample_task(replay_batch, prev_task)
                        loss += self.loss_fn(self.forward(x_past, prev_task), y_past) / n_prev_tasks

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()
            self.log_epoch_end(task_id, epoch, self.args.n_epochs, epoch_start)

        self.fill_buffer(task_id, dataset)
        self.prev_tasks.append(task_id)
        self.log_progress(
            f"task training end: task={task_id} elapsed={self._format_elapsed(self._elapsed_since(train_start))}"
        )

    def _compute_saliency(self, task_id):
        params = self._trainable_params()
        saliency = {name: torch.zeros_like(param, device=param.device) for name, param in params}
        x_all, y_all = self._buffer_tensors(task_id)
        if x_all is None or x_all.numel() == 0:
            return saliency

        total = 0
        batch_size = max(1, int(self.args.batch_size))
        for start in range(0, x_all.shape[0], batch_size):
            x = x_all[start:start + batch_size].to(self.device)
            y = y_all[start:start + batch_size].to(self.device)
            self.net.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.forward(x, task_id), y)
            loss.backward()
            n_batch = int(x.shape[0])
            total += n_batch
            for name, param in params:
                if param.grad is not None:
                    saliency[name].add_((param.grad.detach() * param.detach()).abs(), alpha=n_batch)

        self.net.zero_grad(set_to_none=True)
        if total > 0:
            for name in saliency:
                saliency[name].div_(float(total))
        return saliency

    def _build_saliency_mask(self, saliency):
        ratio = float(getattr(self.args, "salun_mask_ratio", 0.1))
        flat = torch.cat([score.reshape(-1) for score in saliency.values()])
        if flat.numel() == 0 or ratio <= 0.0:
            return {name: torch.zeros_like(score, dtype=torch.bool) for name, score in saliency.items()}, 0
        k = min(flat.numel(), max(1, int(flat.numel() * ratio)))
        threshold = torch.topk(flat, k, largest=True).values[-1]
        masks = {name: score >= threshold for name, score in saliency.items()}
        mask_count = sum(int(mask.sum().item()) for mask in masks.values())
        return masks, mask_count

    def _init_forget_optimizer(self):
        if self.args.optim == "sgd":
            return torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0.0)
        if self.args.optim == "adam":
            return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.0)
        raise NotImplementedError

    def _uniform_forget_loss(self, x, task_id):
        logits = self.forward(x, task_id)
        start = self.cpt * task_id
        end = self.cpt * (task_id + 1)
        task_logits = logits[:, start:end]
        if getattr(self.args, "salun_target", "uniform") == "random":
            target = torch.randint(0, self.cpt, (task_logits.shape[0],), device=task_logits.device)
            return F.cross_entropy(task_logits, target)
        log_probs = F.log_softmax(task_logits, dim=1)
        uniform = torch.full_like(task_logits, 1.0 / self.cpt)
        return -(uniform * log_probs).sum(dim=1).mean()

    def _forget_steps(self):
        if self.args.forget_iters is not None:
            return max(0, int(self.args.forget_iters))
        if self.args.retrain_steps is not None:
            return max(0, int(self.args.retrain_steps))
        return max(1, int(self.args.k_shot))

    def forget(self, task_id):
        self._forget_impl(task_id)

    def forget_with_diagnostics(self, task_id, eval_fn=None, debug_context=None, remaining_tasks=None):
        return self._forget_impl(task_id, remaining_tasks=remaining_tasks, return_info=True)

    def _forget_impl(self, task_id, remaining_tasks=None, return_info=False):
        forget_start = time.perf_counter()
        self.log_progress(f"forget phase start: task={task_id}")

        saliency = self._compute_saliency(task_id)
        masks, mask_count = self._build_saliency_mask(saliency)
        active_tasks = list(remaining_tasks) if remaining_tasks is not None else [t for t in self.prev_tasks if t != task_id]
        steps = self._forget_steps()
        opt = self._init_forget_optimizer()

        for _step in range(steps):
            x_forget, _y_forget = self.memory.sample_task(self.args.batch_size, task_id)
            loss = self._uniform_forget_loss(x_forget, task_id)

            n_active = len(active_tasks)
            if n_active > 0:
                replay_batch = max(1, self.args.batch_size // n_active)
                for active_task in active_tasks:
                    x_past, y_past = self.memory.sample_task(replay_batch, active_task)
                    loss += self.loss_fn(self.forward(x_past, active_task), y_past) / n_active

            opt.zero_grad()
            loss.backward()
            for name, param in self._trainable_params():
                if param.grad is not None:
                    param.grad.mul_(masks[name].to(param.grad.device))
            opt.step()

        if task_id in self.memory.buffer:
            self.memory.remove(task_id)
        if task_id in self.prev_tasks:
            self.prev_tasks.remove(task_id)

        info = {
            "t_reset": None,
            "t_retrain": None,
            "t_forget_total": time.perf_counter() - forget_start,
            "num_updated_params": mask_count,
            "salun_mask_ratio": float(getattr(self.args, "salun_mask_ratio", 0.1)),
            "salun_target": getattr(self.args, "salun_target", "uniform"),
            "salun_forget_steps": steps,
        }
        self.log_progress(
            f"forget phase end: task={task_id} mask_params={mask_count} steps={steps} "
            f"elapsed={self._format_elapsed(info['t_forget_total'])}"
        )
        if return_info:
            return info
        return None
