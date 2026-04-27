import math
import time
import torch
from .base import *
from .er import RehearsalMemory


class PALLAdapter(Base):
    def __init__(self, args):
        super(PALLAdapter, self).__init__(args)
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
        self.prototype_warning = "pall_adapter uses adapter reset prototype, not full PALL overlap mask yet."
        self._prototype_warning_logged = False
        self.archived_task_ids = set()

        self.net.freeze_backbone(train_classifier=self.args.adapter_train_classifier)
        self.net.freeze_backbone_batchnorm()
        self.param_stats = self._build_param_stats()
        self._log_param_stats()
        if not self.args.adapter_train_classifier:
            print("[WARN] pall_adapter classifier training is disabled; only adapters will be optimized.", flush=True)
        self._log_prototype_warning()

    def _build_param_stats(self):
        total_params = self.net.count_total_params()
        trainable_params = self.net.count_trainable_params()
        adapter_params = self.net.count_adapter_params()
        shared_adapter_params, total_shared_adapter_params = self.net.count_shared_adapter_params()
        return {
            "total_params": int(total_params),
            "num_trainable_params": int(trainable_params),
            "num_adapter_params": int(adapter_params),
            "shared_adapter_params": int(shared_adapter_params),
            "trainable_param_ratio": float(trainable_params / total_params) if total_params else 0.0,
            "adapter_mode": "per_task",
            "adapter_bottleneck": int(self.args.adapter_bottleneck),
            "adapter_shared_bottleneck": int(self.args.adapter_shared_bottleneck),
            "adapter_location": self.args.adapter_location,
            "adapter_train_classifier": bool(self.args.adapter_train_classifier),
            "shared_adapter_enabled": bool(self.args.adapter_shared_bottleneck > 0),
            "total_shared_adapter_params": int(total_shared_adapter_params),
        }

    def _log_param_stats(self):
        stats = self.param_stats
        self.log_progress(
            "adapter init: total_params={total} trainable_params={trainable} adapter_params={adapter} "
            "shared_adapter_params={shared_adapter} trainable_ratio={ratio:.6f} adapter_location={location} "
            "train_classifier={train_classifier}".format(
                total=stats["total_params"],
                trainable=stats["num_trainable_params"],
                adapter=stats["num_adapter_params"],
                shared_adapter=stats["shared_adapter_params"],
                ratio=stats["trainable_param_ratio"],
                location=stats["adapter_location"],
                train_classifier=stats["adapter_train_classifier"],
            )
        )
        if stats["shared_adapter_enabled"]:
            self.log_progress(
                "shared adapter enabled: bottleneck={bottleneck} shared_params={shared}".format(
                    bottleneck=stats["adapter_shared_bottleneck"],
                    shared=stats["shared_adapter_params"],
                )
            )

    def _log_prototype_warning(self):
        if not self._prototype_warning_logged:
            print(f"[WARN] {self.prototype_warning}", flush=True)
            self._prototype_warning_logged = True

    def _shared_adapter_param_items(self):
        if getattr(self.net, "shared_adapter", None) is None:
            return []
        return list(self.net.shared_adapter.named_parameters())

    def _zero_shared_adapter_grads(self):
        for _, param in self._shared_adapter_param_items():
            param.grad = None

    def _zeros_like_shared_adapter(self):
        return {name: torch.zeros_like(param) for name, param in self._shared_adapter_param_items()}

    def _compute_shared_importance(self, task_ids):
        param_items = self._shared_adapter_param_items()
        if not param_items or not task_ids:
            return self._zeros_like_shared_adapter(), self._zeros_like_shared_adapter(), 0

        sampled = 0
        loss = 0.0
        batch_size = max(1, self.args.batch_size // max(1, len(task_ids)))
        for task_id in task_ids:
            if not hasattr(self.memory, "buffer") or task_id not in self.memory.buffer:
                continue
            x_task, y_task = self.memory.sample_task(batch_size, task_id)
            loss = loss + self.loss_fn(self.forward(x_task, task_id), y_task)
            sampled += 1

        if sampled == 0:
            return self._zeros_like_shared_adapter(), self._zeros_like_shared_adapter(), 0

        self._zero_shared_adapter_grads()
        loss = loss / sampled
        loss.backward()
        grads = {}
        importance = {}
        for name, param in param_items:
            grad = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
            grads[name] = grad
            importance[name] = grad.abs()
        self._zero_shared_adapter_grads()
        return importance, grads, sampled

    def _topk_shared_masks(self, importance_map, ratio):
        if ratio <= 0.0 or not importance_map:
            return self._zeros_like_shared_adapter(), 0, 0

        flat_chunks = []
        meta = []
        total_params = 0
        for name, tensor in importance_map.items():
            flat = tensor.reshape(-1)
            flat_chunks.append(flat)
            meta.append((name, tensor.shape, flat.numel(), tensor.device))
            total_params += flat.numel()

        if total_params == 0:
            return self._zeros_like_shared_adapter(), 0, 0

        k = min(int(math.ceil(ratio * total_params)), total_params)
        if k <= 0:
            return self._zeros_like_shared_adapter(), 0, total_params

        flat_all = torch.cat(flat_chunks)
        top_indices = torch.topk(flat_all, k, largest=True, sorted=False).indices
        selected = torch.zeros_like(flat_all, dtype=torch.bool)
        selected[top_indices] = True

        masks = {}
        offset = 0
        count = 0
        for name, shape, numel, device in meta:
            curr = selected[offset:offset + numel].view(shape).to(device=device)
            masks[name] = curr
            count += int(curr.sum().item())
            offset += numel
        return masks, count, total_params

    def _combine_shared_masks(self, candidate_masks, protected_masks):
        crit_masks = {}
        candidate_count = 0
        protected_count = 0
        crit_count = 0
        for name, candidate in candidate_masks.items():
            protected = protected_masks.get(name)
            if protected is None:
                protected = torch.zeros_like(candidate, dtype=torch.bool)
            crit = torch.logical_and(candidate, torch.logical_not(protected))
            crit_masks[name] = crit
            candidate_count += int(candidate.sum().item())
            protected_count += int(protected.sum().item())
            crit_count += int(crit.sum().item())
        return crit_masks, candidate_count, protected_count, crit_count

    def _apply_shared_forgetting_update(self, grad_map, crit_masks, lr):
        lr = self.lr if lr is None else lr
        updated = 0
        with torch.no_grad():
            for name, param in self._shared_adapter_param_items():
                grad = grad_map.get(name)
                mask = crit_masks.get(name)
                if grad is None or mask is None or not torch.any(mask):
                    continue
                param.add_(grad * mask.to(dtype=grad.dtype), alpha=lr)
                updated += int(mask.sum().item())
        return updated

    def train_mode(self):
        self.train()
        self.net.freeze_backbone_batchnorm()

    def init_optimizer(self):
        params = [param for param in self.net.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("pall_adapter has no trainable parameters after freezing the backbone.")
        if self.args.optim == "sgd":
            return SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        if self.args.optim == "adam":
            return Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        raise NotImplementedError

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
        adapter_params = sum(self.net.count_adapter_params(task_id) for task_id in active_tasks)
        shared_adapter_params = self.net.count_shared_adapter_params()[0]
        if not self.args.adapter_train_classifier:
            return int(adapter_params + shared_adapter_params)
        classifier_rows = len(active_tasks) * self.cpt
        classifier_params = classifier_rows * self.net.feature_dim
        return int(adapter_params + classifier_params + shared_adapter_params)

    def forget(self, task_id):
        self._forget_impl(task_id)

    def forget_with_diagnostics(self, task_id, eval_fn=None, debug_context=None, remaining_tasks=None):
        del debug_context
        return self._forget_impl(task_id, eval_fn=eval_fn, remaining_tasks=remaining_tasks, return_info=True)

    def _forget_impl(self, task_id, eval_fn=None, remaining_tasks=None, return_info=False):
        if task_id not in self.prev_tasks:
            raise AssertionError(f"[ERROR] {task_id} is not learned yet")

        self._log_prototype_warning()
        forget_start = time.perf_counter()
        self.log_progress(f"forget phase start: task={task_id}")

        active_tasks = list(remaining_tasks) if remaining_tasks is not None else [t for t in self.prev_tasks if t != task_id]
        reset_param_count = self.net.count_adapter_params(task_id)
        shared_param_count, total_shared_params = self.net.count_shared_adapter_params()
        shared_candidate_masks = self._zeros_like_shared_adapter()
        shared_protected_masks = self._zeros_like_shared_adapter()
        shared_crit_masks = self._zeros_like_shared_adapter()
        shared_forget_candidates = 0
        shared_protected_params = 0
        shared_s_share_crit = 0
        deleted_grads = self._zeros_like_shared_adapter()
        if total_shared_params > 0 and self.args.adapter_shared_forget_ratio > 0.0:
            deleted_importance, deleted_grads, _ = self._compute_shared_importance([task_id])
            shared_candidate_masks, _, _ = self._topk_shared_masks(
                deleted_importance,
                self.args.adapter_shared_forget_ratio,
            )
        if (
            total_shared_params > 0
            and self.args.adapter_shared_forget_ratio > 0.0
            and self.args.adapter_shared_protect_ratio > 0.0
        ):
            active_importance, _, _ = self._compute_shared_importance(active_tasks)
            shared_protected_masks, _, _ = self._topk_shared_masks(
                active_importance,
                self.args.adapter_shared_protect_ratio,
            )
        (
            shared_crit_masks,
            shared_forget_candidates,
            shared_protected_params,
            shared_s_share_crit,
        ) = self._combine_shared_masks(shared_candidate_masks, shared_protected_masks)
        total_overlap_scope = reset_param_count + total_shared_params
        if total_overlap_scope <= 0:
            total_overlap_scope = max(reset_param_count, total_shared_params, 1)
        share_ratio = shared_param_count / total_overlap_scope if total_overlap_scope else 0.0
        s_share_crit_ratio = shared_s_share_crit / shared_param_count if shared_param_count else 0.0

        info = {
            "s_t": int(total_overlap_scope),
            "s_share": int(shared_param_count),
            "s_share_crit": int(shared_s_share_crit),
            "s_share_ratio": float(share_ratio),
            "s_share_crit_ratio": float(s_share_crit_ratio),
            "num_updated_params": 0,
            "protection": {
                "active": False,
                "method_variant": "adapter_prototype",
                "warning": self.prototype_warning,
            },
            "adapter_shared_forget_ratio": float(self.args.adapter_shared_forget_ratio),
            "adapter_shared_protect_ratio": float(self.args.adapter_shared_protect_ratio),
            "shared_adapter_params": int(shared_param_count),
            "shared_forget_candidates": int(shared_forget_candidates),
            "shared_protected_params": int(shared_protected_params),
            "shared_s_share_crit": int(shared_s_share_crit),
            "finetune_diag": {
                "active_tasks": active_tasks,
                "deleted_task_id": task_id,
                "retrain_steps": 0,
                "buffer_sizes": {},
            },
        }
        if total_shared_params > 0:
            print(
                "[PALLAdapter] shared critical overlap: candidates={cand} protected={prot} s_share_crit={crit}".format(
                    cand=shared_forget_candidates,
                    prot=shared_protected_params,
                    crit=shared_s_share_crit,
                ),
                flush=True,
            )

        t_reset_start = time.perf_counter()
        self.net.reset_task_adapter(task_id)
        self.net.reset_classifier_slice(task_id, self.cpt)
        if shared_s_share_crit > 0:
            self._apply_shared_forgetting_update(
                deleted_grads,
                shared_crit_masks,
                self.args.adapter_shared_forget_lr,
            )
        info["t_reset"] = time.perf_counter() - t_reset_start
        self.archived_task_ids.add(task_id)

        if eval_fn is not None:
            info["after_reset_eval"] = eval_fn("after_reset")

        if task_id in self.memory.buffer:
            self.memory.remove(task_id)
        if task_id in self.prev_tasks:
            self.prev_tasks.remove(task_id)

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
            info["t_retrain"] = 0.0
            self.log_progress(f"retrain phase skipped: task={task_id} active_tasks={active_tasks}")

        info["t_forget_total"] = time.perf_counter() - forget_start
        self.log_progress(
            f"forget phase end: task={task_id} elapsed={self._format_elapsed(info['t_forget_total'])}"
        )
        return info if return_info else None

    def compute_overlap_matrix(self, include_forgotten=True):
        task_ids = set(self.prev_tasks)
        if include_forgotten:
            task_ids.update(self.archived_task_ids)
        task_ids = sorted(task_ids)
        shared_param_count, total_shared_params = self.net.count_shared_adapter_params()
        matrix = []
        for task_i in task_ids:
            row = []
            for task_j in task_ids:
                if task_i == task_j:
                    row.append(1.0)
                    continue
                union = total_shared_params + self.net.count_adapter_params(task_i) + self.net.count_adapter_params(task_j)
                row.append((shared_param_count / union) if union else 0.0)
            matrix.append(row)
        return {"task_ids": task_ids, "matrix": matrix}
