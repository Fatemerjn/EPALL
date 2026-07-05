import time

import torch

from .base import Base
from .er import RehearsalMemory
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class SSD(Base):
    """Selective Synaptic Dampening baseline.

    The method first learns tasks with a small replay buffer. On a forget request,
    it estimates diagonal Fisher importance on the forgotten task and on retained
    tasks, then multiplicatively dampens parameters whose forget-task Fisher is
    larger than retained-task Fisher.
    """

    def __init__(self, args):
        super(SSD, self).__init__(args)
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

    def _estimate_fisher(self, task_ids):
        params = self._trainable_params()
        fisher = {name: torch.zeros_like(param, device=param.device) for name, param in params}
        total = 0
        batch_size = max(1, int(self.args.batch_size))

        for task_id in task_ids:
            x_all, y_all = self._buffer_tensors(task_id)
            if x_all is None or x_all.numel() == 0:
                continue
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
                        fisher[name].add_(param.grad.detach().pow(2), alpha=n_batch)

        self.net.zero_grad(set_to_none=True)
        if total > 0:
            for name in fisher:
                fisher[name].div_(float(total))
        return fisher

    def forget(self, task_id):
        self._forget_impl(task_id)

    def forget_with_diagnostics(self, task_id, eval_fn=None, debug_context=None, remaining_tasks=None):
        return self._forget_impl(task_id, remaining_tasks=remaining_tasks, return_info=True)

    def _forget_impl(self, task_id, remaining_tasks=None, return_info=False):
        forget_start = time.perf_counter()
        self.log_progress(f"forget phase start: task={task_id}")

        active_tasks = list(remaining_tasks) if remaining_tasks is not None else [t for t in self.prev_tasks if t != task_id]
        forget_fisher = self._estimate_fisher([task_id])
        retain_fisher = self._estimate_fisher(active_tasks)

        alpha = float(getattr(self.args, "ssd_alpha", 1.0))
        dampening_lambda = float(getattr(self.args, "ssd_lambda", 1.0))
        eps = 1e-8
        updated_params = 0
        total_params = 0

        with torch.no_grad():
            for name, param in self._trainable_params():
                f_forget = forget_fisher[name]
                f_retain = retain_fisher[name]
                ratio = f_forget / (f_retain + eps)
                selected = ratio > alpha
                total_params += int(param.numel())
                updated_params += int(selected.sum().item())
                if torch.any(selected):
                    dampening = torch.ones_like(param)
                    dampening[selected] = torch.clamp(
                        dampening_lambda / (ratio[selected] + eps),
                        min=0.0,
                        max=1.0,
                    )
                    param.mul_(dampening)

        if task_id in self.memory.buffer:
            self.memory.remove(task_id)
        if task_id in self.prev_tasks:
            self.prev_tasks.remove(task_id)

        info = {
            "t_reset": None,
            "t_retrain": None,
            "t_forget_total": time.perf_counter() - forget_start,
            "num_updated_params": updated_params,
            "ssd_alpha": alpha,
            "ssd_lambda": dampening_lambda,
            "ssd_total_params": total_params,
        }
        self.log_progress(
            f"forget phase end: task={task_id} updated_params={updated_params} "
            f"elapsed={self._format_elapsed(info['t_forget_total'])}"
        )
        if return_info:
            return info
        return None
