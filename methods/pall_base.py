import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .base import *
from .er import RehearsalMemory
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class PALLBase(Base):
    """Subnet-mask PALL with overlap-aware selective forgetting (shared base).

    This base class holds ALL the machinery. The concrete entry points are thin
    subclasses that map 1:1 to the paper methods and to ``--method``:

      * ``methods/pall_original.py`` -> ``PALLOriginal`` (variant ``"original"``):
        PALL-Original baseline. Forgetting resets the forget-task subnet and
        replays to recover, with NO protection of the shared overlap.
      * ``methods/pall_modified.py`` -> ``PALLModified`` (variant ``"modified"``):
        the MAIN method. Identifies the critical shared overlap (forget-subnet
        INTERSECT retained-subnets) and protects it during forgetting via an L2
        anchor on a ranked subset. The ranking criterion (``--protect_importance``)
        is one of: ``conflict`` (relu(-g_forget*g_retain); best under HIGH overlap),
        ``gradient`` (|grad L_retain|; default), or ``weight`` (|w|; legacy
        ablation). See ``_compute_conflict_importance``,
        ``_compute_retain_grad_importance`` and ``_select_critical_shared_masks``.

    Behaviour is selected purely by ``args.method_variant`` (set by
    ``normalize_method`` in main.py), so the subclasses add only documentation and
    a stable import name. ``methods/pall.py`` remains a backward-compatible shim
    exposing ``PALL = PALLBase``. ``--method pall`` is a deprecated alias for
    ``pall_modified``. The adapter variant lives in ``methods/pall_adapter.py``.
    """

    def __init__(self, args):
        super(PALLBase, self).__init__(args)
        self.memory = RehearsalMemory(buffer_size=self.args.mem_budget,
                                      n_tasks=self.args.n_tasks,
                                      cpt=self.args.class_per_task,
                                      dim_x=self.args.dim_input,
                                      device=self.device,
                                      mem_type=self.args.mem_type,
                                      save_logits=True)
        self.alpha = args.alpha
        self.beta = args.beta
        self.k_shot = args.k_shot
        self.per_task_masks = {}
        self.combined_masks = {}
        self.finetuning_hist = {}
        self.method_variant = getattr(args, "method_variant", "modified") or "modified"
        self.archived_task_masks = {}

    def init_optimizer(self):
        if self.args.optim == "sgd":
            return SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.args.optim == "adam":
            return Adam(self.net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

    def der_loss(self, a, b, task_id):
        a_ = a[..., task_id*self.cpt:(task_id+1)*self.cpt]
        b_ = b[..., task_id*self.cpt:(task_id+1)*self.cpt]
        return F.mse_loss(a_, b_)

    def forward(self, x, task, mask=None, mode="train"):
        pred = self.net.forward(x, task, mask, mode)
        if task > 0:
            pred[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            pred[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return pred

    def forward_with_features(self, x, task, mask=None, mode="train"):
        pred, features = self.net.forward(x, task, mask, mode, returnt='all')
        if task > 0:
            pred[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            pred[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return pred, features

    def evaluate(self, x, task, mask=None, mode="test"):
        if task in self.per_task_masks:
            return self.forward(x, task, mask=self.per_task_masks[task], mode="test")
        else:
            return self.forward(x, task, mask=None, mode="no_mask")

    def evaluate_features(self, x, task):
        """Penultimate (masked-backbone) features on the SAME forward path as
        ``evaluate``: the task's subnet mask while that mask still exists, else the
        mask-free full-network features (mode='no_mask') that the deployed model
        actually exposes once the task's subnet has been removed by unlearning.
        This is exactly the representation the linear-probe leakage audit reads
        before vs after forgetting -- before: task-masked subnet; after: no_mask."""
        if task in self.per_task_masks:
            _out, features = self.forward_with_features(x, task, mask=self.per_task_masks[task], mode="test")
        else:
            _out, features = self.forward_with_features(x, task, mask=None, mode="no_mask")
        return features

    def extract_logits_and_features(self, data_loader, task_id, norm_features=True):
        features, logits, targets = [], [], []
        with torch.no_grad():
            self.net.eval()
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred, feats = self.forward_with_features(x, task_id, mask=self.per_task_masks[task_id], mode="test")
                if norm_features:
                    feats = feats / feats.norm(dim=1).view(-1, 1)
                features.append(feats)
                logits.append(pred)
                targets.append(y)
            features, logits, targets = torch.cat(features), torch.cat(logits), torch.cat(targets)
        self.net.train()
        return features.cpu(), logits.cpu(), targets.cpu()

    def fill_buffer(self, task_id, dataset):
        sel_loader = self.build_dataloader(dataset, shuffle=False, context=f"fill_buffer_task_{task_id}")
        features, logits, targets = self.extract_logits_and_features(sel_loader, task_id)
        if self.args.mem_type == "random":
            sel_indices = self.memory.select_indices_by_random(targets)
        else:
            raise NotImplementedError
        x, y = zip(*(sel_loader.dataset[idx] for idx in sel_indices))
        if self.memory.save_logits:
            self.memory.add((x, y, logits[sel_indices].detach()), task_id)
        else:
            self.memory.add((x, y), task_id)

    def learn(self, task_id, dataset):
        assert task_id not in self.per_task_masks, f"[ERROR] {task_id} already present in learned subnet masks"
        assert self.args.weight_decay >= 0.0, f"[ERROR] Invalid weight_decay value: {self.args.weight_decay}"

        loader = self.build_dataloader(dataset, shuffle=True, context=f"learn_task_{task_id}")
        self.opt = self.init_optimizer()
        train_start = time.perf_counter()
        self.log_progress(f"task training start: task={task_id} epochs={self.args.n_epochs} steps_per_epoch={len(loader)}")

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        for epoch in range(self.args.n_epochs):
            epoch_start = time.perf_counter()
            self.log_epoch_start(task_id, epoch, self.args.n_epochs)
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.forward(x, task_id, mask=None, mode="train"), y)
                self.opt.zero_grad()
                loss.backward()

                # Set gradients to zero (no backprop) for the active combined subnets
                if self.combined_masks != {}:  # Only do this for tasks 1 and beyond
                    for key in self.combined_masks.keys():
                        key_split = key.split('.')
                        module_attr = key_split[-1]
                        if 'classifier' in key_split or len(key_split) == 2:  # e.g., conv1.weight or classifier.weight
                            module_name = key_split[0]
                            if hasattr(getattr(self.net, module_name), module_attr):
                                if getattr(getattr(self.net, module_name), module_attr) is not None:
                                    getattr(getattr(self.net, module_name), module_attr).grad[self.combined_masks[key] == 1] = 0
                        elif len(key_split) == 4:  # e.g., layer1.0.conv1.weight or encoder_layers.0.mlp_lin1.weight
                            curr_module = getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2])
                            if hasattr(curr_module, module_attr):
                                if getattr(curr_module, module_attr) is not None:
                                    getattr(curr_module, module_attr).grad[self.combined_masks[key] == 1] = 0
                        elif len(key_split) == 5:  # e.g., encoder_layers.0.mha.W_V.weight
                            curr_module = getattr(getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2]), key_split[3])
                            if hasattr(curr_module, module_attr):
                                if getattr(curr_module, module_attr) is not None:
                                    getattr(curr_module, module_attr).grad[self.combined_masks[key] == 1] = 0
                        else:
                            raise NotImplementedError('This should not happen with the currently implemented models!')

                    if self.args.weight_decay != 0.0:
                        for n, p in self.net.named_parameters():    # no weight decay for the scores
                            if n.endswith("weight") or n.endswith("bias"):
                                if self.combined_masks[n] is not None:
                                    p.grad += self.args.weight_decay * p * (1 - self.combined_masks[n])
                else:
                    if self.args.weight_decay != 0.0:
                        for n, p in self.net.named_parameters():    # no weight decay for the scores
                            if n.endswith("weight") or n.endswith("bias"):
                                p.grad += self.args.weight_decay * p

                self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()
            self.log_epoch_end(task_id, epoch, self.args.n_epochs, epoch_start)

        # Then save the per-task-dependent masks
        self.per_task_masks[task_id] = self.net.get_masks(task_id)

        # Fill up rehearsal memory using the final architecture
        self.fill_buffer(task_id, dataset)

        # Combine task masks to keep track of parameters to-update or not
        if self.combined_masks == {}:
            self.combined_masks = deepcopy(self.per_task_masks[task_id])
        else:
            for key in self.per_task_masks[task_id].keys():
                if self.combined_masks[key] is not None and self.per_task_masks[task_id][key] is not None:
                    self.combined_masks[key] = 1 - ((1 - self.combined_masks[key]) * (1 - self.per_task_masks[task_id][key]))

        # Print sparsity metrics
        connectivity_per_layer = self.print_connectivity(self.combined_masks)
        all_connectivity = self.global_connectivity(self.combined_masks)
        print("Connectivity per Layer: {}".format(connectivity_per_layer))
        print("Global Connectivity: {}".format(all_connectivity))

        # Reinitialize the scores and unused weights
        self.net.reinit_scores()
        self.net.reinit_weights(self.combined_masks)
        self.log_progress(
            f"task training end: task={task_id} elapsed={self._format_elapsed(self._elapsed_since(train_start))}"
        )

    def _backup_shared_weights(self, shared_masks, removed_task_id):
        if not self.per_task_masks:
            return
        if not shared_masks:
            return
        modules = dict(self.net.named_modules())
        for other_task_id, other_masks in self.per_task_masks.items():
            if other_task_id == removed_task_id:
                continue
            for key, shared_mask in shared_masks.items():
                if shared_mask is None:
                    continue
                other_mask = other_masks.get(key)
                if other_mask is None:
                    continue
                shared_mask = torch.logical_and(shared_mask == 1, other_mask == 1)
                if not torch.any(shared_mask):
                    continue
                module_name, param_name = key.rsplit('.', 1)
                module = modules.get(module_name)
                if module is None or not hasattr(module, "store_backup"):
                    continue
                if param_name == "weight":
                    module.store_backup(other_task_id, weight_mask=shared_mask)
                elif param_name == "bias":
                    module.store_backup(other_task_id, bias_mask=shared_mask)

    def _clear_task_backups(self, task_id):
        for _, module in self.net.named_modules():
            if hasattr(module, "backup_weights") and task_id in module.backup_weights:
                del module.backup_weights[task_id]

    def _count_mask_entries(self, masks):
        count = 0
        for mask in masks.values():
            if mask is None:
                continue
            count += int(torch.sum(mask != 0).item())
        return count

    def _masked_l2_norm(self, param_map, masks):
        total_sq = 0.0
        for name, mask in masks.items():
            if mask is None:
                continue
            param = param_map.get(name)
            if param is None:
                continue
            mask = mask.to(dtype=torch.bool, device=param.device)
            if not torch.any(mask):
                continue
            vals = param.detach()[mask]
            total_sq += (vals * vals).sum().item()
        return math.sqrt(total_sq)

    def _diff_norm_from_refs(self, param_map, refs):
        total_sq = 0.0
        for name, ref in refs.items():
            param = param_map.get(name)
            if param is None:
                continue
            mask = ref["mask"]
            ref_vals = ref["values"]
            if ref_vals.numel() == 0:
                continue
            diff = param.detach()[mask] - ref_vals
            total_sq += (diff * diff).sum().item()
        return math.sqrt(total_sq)

    def _select_critical_shared_masks(self, shared_masks, score_map=None):
        """Select the *critical* subset of the forget/retain shared parameters.

        The candidate set ``shared_masks`` (S_share) is the structural
        intersection of the per-task subnet masks. Within it we keep the top
        ``protect_ratio`` (or those >= ``protect_threshold``) most important
        coordinates, ranked by:

        * ``score_map`` -- the **absolute gradient** of the retained-task loss on
          the rehearsal buffer, ``|grad L_retain|``, when provided. This is the
          default PALL-Modified "gradient" importance (the paper's S_active) and
          is produced by ``_compute_retain_grad_importance``.
        * otherwise the current **absolute weight magnitude** ``|w|`` -- the
          legacy criterion, kept as a selectable ablation
          (``--protect_importance weight``).

        ``score_map`` is keyed by parameter name with tensors shaped like the
        parameters. Both paths share identical top-k / threshold logic, so the
        weight ablation is bit-for-bit the previous behaviour.
        """
        if not shared_masks:
            return {}
        protect_ratio = self.args.protect_ratio
        protect_threshold = self.args.protect_threshold
        param_map = dict(self.net.named_parameters())

        def _score(key, param):
            # Per-parameter importance used to rank the shared-overlap set.
            if score_map is not None:
                s = score_map.get(key)
                if s is None:
                    return torch.zeros_like(param)
                return s.to(device=param.device, dtype=param.dtype)
            return param.detach().abs()

        if protect_ratio is None and protect_threshold is None:
            return {
                key: (mask.clone() if mask is not None else None)
                for key, mask in shared_masks.items()
            }

        values = []
        for key, mask in shared_masks.items():
            if mask is None:
                continue
            param = param_map.get(key)
            if param is None:
                continue
            mask = mask.to(dtype=torch.bool, device=param.device)
            if torch.any(mask):
                values.append(_score(key, param)[mask].view(-1))

        if not values:
            return {
                key: (torch.zeros_like(mask, dtype=torch.bool) if mask is not None else None)
                for key, mask in shared_masks.items()
            }

        all_values = torch.cat(values)
        if protect_ratio is not None:
            if protect_ratio <= 0:
                threshold = float("inf")
            elif protect_ratio >= 1:
                threshold = all_values.min().item()
            else:
                k = max(1, int(math.ceil(protect_ratio * all_values.numel())))
                if k >= all_values.numel():
                    threshold = all_values.min().item()
                else:
                    threshold = torch.topk(all_values, k, largest=True, sorted=False).values.min().item()
        else:
            threshold = protect_threshold

        crit_masks = {}
        for key, mask in shared_masks.items():
            if mask is None:
                crit_masks[key] = None
                continue
            param = param_map.get(key)
            if param is None:
                crit_masks[key] = torch.zeros_like(mask, dtype=torch.bool)
                continue
            mask = mask.to(dtype=torch.bool, device=param.device)
            if protect_ratio is not None and protect_ratio <= 0:
                crit_masks[key] = torch.zeros_like(mask, dtype=torch.bool)
            else:
                crit_masks[key] = torch.logical_and(mask, _score(key, param) >= threshold)
        return crit_masks

    def _reinit_anchor_values(self, masks, param_map):
        """Fresh, data-independent reinit sample for the masked positions.

        Used by ``--protect_anchor reinit``. Reviewer point: anchoring the L2
        penalty to the pre-forget weights ``w_old`` keeps the protected critical
        weights close to a state that still ENCODES the forgotten task; anchoring
        instead to a fresh reinitialization removes that residual leakage while
        still pinning the weights (so retained tasks can repair them). The init
        scheme mirrors ``SubnetResNet.reinit_weights``: kaiming_uniform_(a=sqrt 5)
        for weights, uniform(+/- 1/sqrt(fan_in)) for biases.
        """
        out = {}
        for key, mask in masks.items():
            if mask is None:
                continue
            param = param_map.get(key)
            if param is None:
                continue
            m = mask.to(dtype=torch.bool, device=param.device)
            if not torch.any(m):
                continue
            sample = torch.zeros_like(param)
            if key.endswith(".bias"):
                weight = param_map.get(key[: -len("bias")] + "weight")
                if weight is not None and weight.dim() >= 2:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                    if fan_in > 0:
                        nn.init.uniform_(sample, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))
            elif param.dim() >= 2:
                nn.init.kaiming_uniform_(sample, a=math.sqrt(5))
            out[key] = sample[m].clone()
        return out

    def _compute_signed_grads(self, task_ids):
        """Per-parameter SIGNED gradient of the summed CE loss over the given
        tasks' rehearsal buffers.

        Gradients are isolated: every parameter's ``.grad`` is cleared before and
        after, so the surrounding reset/retrain phases start from a clean state.
        Returns ``{}`` when no buffer samples are available. The caller must ensure
        each task's mask is still in ``per_task_masks`` and its buffer is still in
        memory (important for the forget task, whose buffer is removed mid-forget).
        """
        grads = {}
        if not task_ids:
            return grads
        for param in self.net.parameters():
            param.grad = None
        total_loss = None
        sampled = 0
        batch_size = max(1, self.args.batch_size // max(1, len(task_ids)))
        for t_id in task_ids:
            if t_id not in self.per_task_masks:
                continue
            if not hasattr(self.memory, "buffer") or t_id not in self.memory.buffer:
                continue
            sample = self.memory.sample_task(batch_size, t_id)
            x_past, y_past = sample[0], sample[1]
            h = self.forward(x_past, t_id, mask=self.per_task_masks[t_id], mode="test")
            task_loss = self.loss_fn(h, y_past)
            total_loss = task_loss if total_loss is None else total_loss + task_loss
            sampled += 1
        if sampled == 0 or total_loss is None:
            for param in self.net.parameters():
                param.grad = None
            return grads
        (total_loss / sampled).backward()
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
        for param in self.net.parameters():
            param.grad = None
        return grads

    def _compute_retain_grad_importance(self, active_tasks):
        """Per-parameter |grad L_retain| over the rehearsal buffer (S_active).

        Absolute retained-task gradient magnitude -- the retained-task sensitivity
        used to rank which shared-overlap parameters to protect (the paper's
        |grad L| / S_active importance). The forget task has already been removed
        from memory before this is called, so no forget data leaks in.
        """
        grads = self._compute_signed_grads(active_tasks)
        importance = {name: g.abs() for name, g in grads.items()}
        self.log_progress(
            f"retain-grad importance computed: active_tasks={active_tasks} "
            f"params_scored={len(importance)}"
        )
        return importance

    def _compute_conflict_importance(self, active_tasks, forget_grads):
        """Per-parameter gradient-CONFLICT energy on the rehearsal buffer.

        For each coordinate the score is ``relu(-g_forget * g_retain)``: large
        exactly where the forget-task gradient and the retained-task gradient have
        OPPOSITE signs and both are sizeable -- i.e. the parameters where forgetting
        and retention directly compete. Protecting these is what bounds WorstDrop
        when the forget/retain parameter overlap is high. ``forget_grads`` is the
        SIGNED forget-task gradient captured BEFORE the forget buffer was removed.
        """
        retain_grads = self._compute_signed_grads(active_tasks)
        conflict = {}
        for name, g_r in retain_grads.items():
            g_f = forget_grads.get(name)
            if g_f is None:
                conflict[name] = torch.zeros_like(g_r)
            else:
                conflict[name] = torch.clamp(-(g_f.to(device=g_r.device, dtype=g_r.dtype) * g_r), min=0.0)
        self.log_progress(
            f"conflict importance computed: active_tasks={active_tasks} "
            f"params_scored={len(conflict)}"
        )
        return conflict

    def _compute_retrain_steps(self, overlap_ratio):
        steps = self.args.retrain_steps
        if self.args.retrain_epochs is not None:
            steps = self.args.retrain_epochs
        if steps is None:
            steps = self.k_shot
        if self.args.adaptive_retrain and steps > 0:
            steps = max(1, int(math.ceil(steps * overlap_ratio)))
        return steps

    def _mask_indices(self, masks):
        indices = {}
        for key, mask in masks.items():
            if mask is None:
                continue
            mask_cpu = mask.detach().to(dtype=torch.bool, device="cpu")
            if torch.any(mask_cpu):
                indices[key] = torch.nonzero(mask_cpu, as_tuple=False).tolist()
        return indices

    def _dump_unlearning_debug(
        self,
        debug_dir,
        masks,
        indices,
        norms,
        diff_norms,
    ):
        path = Path(debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        cpu_masks = {k: (v.detach().cpu() if v is not None else None) for k, v in masks.items()}
        torch.save(cpu_masks, path / "masks.pt")
        with open(path / "indices.json", "w", encoding="utf-8") as handle:
            json.dump(indices, handle, indent=2)
        payload = {"norms": norms, "diff_norms": diff_norms}
        with open(path / "norms.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def forget(self, task_id):
        self._forget_impl(task_id)

    def forget_with_diagnostics(self, task_id, eval_fn=None, debug_context=None, remaining_tasks=None):
        return self._forget_impl(
            task_id,
            eval_fn=eval_fn,
            debug_context=debug_context,
            remaining_tasks=remaining_tasks,
            return_info=True,
        )

    def _forget_impl(self, task_id, eval_fn=None, debug_context=None, remaining_tasks=None, return_info=False):
        assert task_id in self.per_task_masks, f"[ERROR] {task_id} is not learned yet (no per_task_mask found)"
        assert hasattr(self.memory, "buffer") and task_id in self.memory.buffer, (
            "[PALLBase] Forget-data access contract violated: the forget task's "
            "rehearsal buffer must be present until S_forget/forget-gradient "
            "estimation finishes. No raw-train or held-out fallback is used."
        )
        forget_start = time.perf_counter()
        self.log_progress(f"forget phase start: task={task_id}")

        info = {
            "t_reset": 0.0,
            "t_retrain": 0.0,
            "num_updated_params": 0,
            "grad_norm_ratio": None,
        }

        # (1) Backup the masks of the task to be forgotten from the list
        deleted_masks = deepcopy(self.per_task_masks[task_id])
        self.archived_task_masks[task_id] = deepcopy(deleted_masks)
        self._clear_task_backups(task_id)

        s_t_masks = {
            key: (mask == 1 if mask is not None else None)
            for key, mask in deleted_masks.items()
        }
        s_share_masks = {
            key: (torch.zeros_like(mask, dtype=torch.bool) if mask is not None else None)
            for key, mask in s_t_masks.items()
        }
        s_share_crit_masks = {
            key: (torch.zeros_like(mask, dtype=torch.bool) if mask is not None else None)
            for key, mask in s_t_masks.items()
        }

        # Data-access contract: any forget-task gradient used by this unlearning
        # request is computed from the task's rehearsal-buffer exemplars while
        # the buffer is still present. The buffer is deleted immediately below;
        # this path never falls back to a raw training set or a held-out set.
        forget_grads = {}
        if self.method_variant == "modified" and self.args.protect_importance == "conflict":
            forget_grads = self._compute_signed_grads([task_id])

        # Gradient Norm Ratio (reviewer metric): capture the SIGNED forget-task
        # gradient on the subnet while the forget buffer is still present. Reuse
        # the conflict grads when already computed, else compute once here. This
        # runs for pall_original and pall_modified alike, independent of
        # protect_importance / protect_ratio; the retain-side grads are taken
        # below (after active_tasks is known) and the ratio stored in info.
        metric_forget_grads = forget_grads if forget_grads else self._compute_signed_grads([task_id])

        # (2) Remove the masks of the task to be forgotten from the list as well as the rehearsal samples from memory
        self.memory.remove(task_id)
        assert task_id not in self.memory.buffer, (
            "[PALLBase] Forget-task rehearsal buffer must be deleted immediately "
            "after the pre-delete forget-gradient access window."
        )
        del self.per_task_masks[task_id]
        if isinstance(self.net, SubnetVisionTransformer):
            nn.init.zeros_(self.net.class_tokens[task_id])
            nn.init.normal_(self.net.pos_embeddings[task_id], std=0.02)

        crit_refs = {}
        before_norms = {}
        after_reset_norms = {}
        after_retrain_norms = {}
        diff_norms = {}

        if self.per_task_masks != {}:
            active_tasks = list(remaining_tasks) if remaining_tasks is not None else list(self.per_task_masks.keys())

            # Gradient Norm Ratio: retain-task signed grads over the same subnet,
            # combined with the forget grads captured above. No dependence on
            # protect settings, so pall_original/pall_modified both populate it.
            metric_active_grads = self._compute_signed_grads(active_tasks)
            info["grad_norm_ratio"] = grad_l2_norm_ratio(metric_forget_grads, metric_active_grads)

            # (3) Create combined masks for until task_id
            prev_tasks = [t for t in self.per_task_masks.keys() if task_id > t]
            combined_prev_task_id = {}
            for prev_task in prev_tasks:
                if combined_prev_task_id == {}:
                    combined_prev_task_id = deepcopy(self.per_task_masks[prev_task])
                else:
                    for key in self.per_task_masks[prev_task].keys():
                        if combined_prev_task_id[key] is not None and self.per_task_masks[prev_task][key] is not None:
                            combined_prev_task_id[key] = 1 - ((1 - combined_prev_task_id[key]) * (1 - self.per_task_masks[prev_task][key]))

            # (4) Create combined masks for after task_id
            after_tasks = [t for t in self.per_task_masks.keys() if task_id < t]
            combined_after_task_id = {}
            for after_task in after_tasks:
                if combined_after_task_id == {}:
                    combined_after_task_id = deepcopy(self.per_task_masks[after_task])
                else:
                    for key in self.per_task_masks[after_task].keys():
                        if combined_after_task_id[key] is not None and self.per_task_masks[after_task][key] is not None:
                            combined_after_task_id[key] = 1 - ((1 - combined_after_task_id[key]) * (1 - self.per_task_masks[after_task][key]))

            combined_remaining = {}
            for remaining_task in active_tasks:
                if combined_remaining == {}:
                    combined_remaining = deepcopy(self.per_task_masks[remaining_task])
                else:
                    for key in self.per_task_masks[remaining_task].keys():
                        if combined_remaining[key] is not None and self.per_task_masks[remaining_task][key] is not None:
                            combined_remaining[key] = 1 - ((1 - combined_remaining[key]) * (1 - self.per_task_masks[remaining_task][key]))
                        else:
                            combined_remaining[key] = None

            for key, mask in s_t_masks.items():
                if mask is None:
                    s_share_masks[key] = None
                    continue
                if combined_remaining and combined_remaining.get(key) is not None:
                    s_share_masks[key] = torch.logical_and(mask, combined_remaining[key] == 1)
                else:
                    s_share_masks[key] = torch.zeros_like(mask, dtype=torch.bool)

            # PALL-Modified ranks the critical shared subset by one of three
            # criteria (see --protect_importance). All are computed HERE, before
            # the reset, so they reflect the trained representations:
            #   conflict : relu(-g_forget * g_retain) -- best under HIGH overlap,
            #              targets params where forgetting and retention fight;
            #   gradient : |g_retain|                 -- default main method;
            #   weight   : |w| (score_map=None)       -- legacy ablation.
            crit_score_map = None
            if self.method_variant == "modified":
                if self.args.protect_importance == "gradient":
                    crit_score_map = self._compute_retain_grad_importance(active_tasks)
                elif self.args.protect_importance == "conflict":
                    crit_score_map = self._compute_conflict_importance(active_tasks, forget_grads)
            s_share_crit_masks = self._select_critical_shared_masks(
                s_share_masks, score_map=crit_score_map
            )

            s_t_count = self._count_mask_entries(s_t_masks)
            s_share_count = self._count_mask_entries(s_share_masks)
            s_share_crit_count = self._count_mask_entries(s_share_crit_masks)
            s_share_ratio = s_share_count / s_t_count if s_t_count else 0.0
            s_share_crit_ratio = s_share_crit_count / s_share_count if s_share_count else 0.0

            info.update(
                {
                    "s_t": s_t_count,
                    "s_share": s_share_count,
                    "s_share_crit": s_share_crit_count,
                    "s_share_ratio": s_share_ratio,
                    "s_share_crit_ratio": s_share_crit_ratio,
                }
            )

            use_protection = self.method_variant == "modified"
            param_map = dict(self.net.named_parameters())
            if s_share_crit_count > 0 and ((use_protection and self.args.lambda_protect > 0.0) or debug_context is not None):
                # Anchor target for the critical-shared L2 penalty: pre-forget
                # weights w_old (default), or a fresh reinit sample when
                # --protect_anchor reinit (w_old still encodes forget info).
                reinit_anchor = {}
                if getattr(self.args, "protect_anchor", "old") == "reinit":
                    reinit_anchor = self._reinit_anchor_values(s_share_crit_masks, param_map)
                for key, mask in s_share_crit_masks.items():
                    if mask is None:
                        continue
                    param = param_map.get(key)
                    if param is None:
                        continue
                    mask = mask.to(dtype=torch.bool, device=param.device)
                    if not torch.any(mask):
                        continue
                    crit_refs[key] = {
                        "mask": mask,
                        "values": reinit_anchor.get(key, param.detach()[mask].clone()),
                    }

            if use_protection and s_share_crit_count > 0:
                self._backup_shared_weights(s_share_crit_masks, task_id)

            # (5) Store indices of what to reset and what to finetune to recover KT
            to_finetune, to_reset, buffer_leak = {}, {}, {}
            if combined_prev_task_id == {}:
                assert combined_after_task_id != {}
                for key in deleted_masks.keys():
                    if deleted_masks[key] is not None and combined_after_task_id[key] is not None:
                        to_reset[key] = torch.logical_not(deleted_masks[key] == 1)
                        if combined_remaining.get(key) is not None:
                            to_finetune[key] = torch.logical_and(deleted_masks[key] == 1, combined_remaining[key] == 1)
                        else:
                            to_finetune[key] = torch.zeros_like(deleted_masks[key], dtype=torch.bool)
            else:
                if combined_after_task_id == {}:
                    for key in deleted_masks.keys():
                        if deleted_masks[key] is not None and combined_prev_task_id[key] is not None:
                            to_reset[key] = torch.logical_and(deleted_masks[key] == 1, combined_prev_task_id[key] == 0)
                            if combined_remaining.get(key) is not None:
                                to_finetune[key] = torch.logical_and(deleted_masks[key] == 1, combined_remaining[key] == 1)
                            else:
                                to_finetune[key] = torch.zeros_like(deleted_masks[key], dtype=torch.bool)
                            buffer_leak_ids = [k for k in self.finetuning_hist.keys() if task_id in self.finetuning_hist[k][1]]
                            buffer_leak[key] = torch.zeros_like(deleted_masks[key], dtype=torch.bool)
                            if buffer_leak_ids:
                                for k in buffer_leak_ids:
                                    if self.finetuning_hist[k][0][key] is not None:
                                        buffer_leak[key] = torch.logical_or(buffer_leak[key], torch.logical_and(torch.logical_and(deleted_masks[key] == 1, combined_prev_task_id[key] == 1), self.finetuning_hist[k][0][key] == 1))
                                to_reset[key] = torch.logical_not(torch.logical_or(to_reset[key], buffer_leak[key]))
                                to_finetune[key] = torch.logical_or(to_finetune[key], buffer_leak[key])
                            else:
                                to_reset[key] = torch.logical_not(to_reset[key])
                else:
                    for key in deleted_masks.keys():
                        if (deleted_masks[key] is not None and combined_prev_task_id[key] is not None and combined_after_task_id[key] is not None):
                            to_reset[key] = torch.logical_and(deleted_masks[key] == 1, combined_prev_task_id[key] == 0)
                            if combined_remaining.get(key) is not None:
                                to_finetune[key] = torch.logical_and(deleted_masks[key] == 1, combined_remaining[key] == 1)
                            else:
                                to_finetune[key] = torch.zeros_like(deleted_masks[key], dtype=torch.bool)
                            buffer_leak_ids = [k for k in self.finetuning_hist.keys() if task_id in self.finetuning_hist[k][1]]
                            buffer_leak[key] = torch.zeros_like(deleted_masks[key], dtype=torch.bool)
                            if buffer_leak_ids:
                                for k in buffer_leak_ids:
                                    if self.finetuning_hist[k][0][key] is not None:
                                        buffer_leak[key] = torch.logical_or(buffer_leak[key], torch.logical_and(torch.logical_and(deleted_masks[key] == 1, combined_prev_task_id[key] == 1), self.finetuning_hist[k][0][key] == 1))
                                to_reset[key] = torch.logical_not(torch.logical_or(to_reset[key], buffer_leak[key]))
                                to_finetune[key] = torch.logical_or(to_finetune[key], buffer_leak[key])
                            else:
                                to_reset[key] = torch.logical_not(to_reset[key])
                                to_finetune[key] = to_finetune[key]

            if debug_context is not None:
                before_norms = {
                    "S_t": self._masked_l2_norm(param_map, s_t_masks),
                    "S_share": self._masked_l2_norm(param_map, s_share_masks),
                    "S_share_crit": self._masked_l2_norm(param_map, s_share_crit_masks),
                }

            # (6) Reinitialize the weights that were specific to this task
            t_reset_start = time.perf_counter()
            self.log_progress(f"reset phase start: task={task_id}")
            self.net.reinit_weights(to_reset)
            info["t_reset"] = time.perf_counter() - t_reset_start
            self.log_progress(
                f"reset phase end: task={task_id} elapsed={self._format_elapsed(info['t_reset'])}"
            )

            if eval_fn is not None:
                info["after_reset_eval"] = eval_fn("after_reset")

            if debug_context is not None:
                after_reset_norms = {
                    "S_t": self._masked_l2_norm(param_map, s_t_masks),
                    "S_share": self._masked_l2_norm(param_map, s_share_masks),
                    "S_share_crit": self._masked_l2_norm(param_map, s_share_crit_masks),
                }
                if crit_refs:
                    diff_norms["S_share_crit_after_reset"] = self._diff_norm_from_refs(param_map, crit_refs)

            num_updated_params = 0
            for key in to_finetune.keys():
                if to_finetune[key] is not None:
                    num_updated_params += int(torch.sum(to_finetune[key] != 0).item())
            do_finetune = num_updated_params > 0
            info["num_updated_params"] = num_updated_params

            retrain_steps = self._compute_retrain_steps(s_share_ratio)
            if self.args.retrain_epochs is not None:
                retrain_source = "retrain_epochs"
            elif self.args.retrain_steps is not None:
                retrain_source = "retrain_steps"
            else:
                retrain_source = "k_shot"
            retrain_from_cli = retrain_source in ("retrain_epochs", "retrain_steps")

            buffer_sizes = {}
            if hasattr(self.memory, "buffer"):
                for t_id in active_tasks:
                    entry = self.memory.buffer.get(t_id) if self.memory.buffer else None
                    if entry is None:
                        buffer_sizes[str(t_id)] = 0
                        continue
                    num_seen = int(entry.get("num_seen", 0))
                    total = int(entry["X"].shape[0]) if "X" in entry else num_seen
                    buffer_sizes[str(t_id)] = min(num_seen, total)

            info["finetune_diag"] = {
                "active_tasks": active_tasks,
                "deleted_task_id": task_id,
                "num_updated_params": num_updated_params,
                "do_finetune": do_finetune,
                "retrain_steps": retrain_steps,
                "retrain_steps_source": retrain_source,
                "retrain_steps_from_cli": retrain_from_cli,
                "allow_zero_retrain": self.args.allow_zero_retrain,
                "retrain_steps_fallback": None,
                "buffer_sizes": buffer_sizes,
            }
            print(
                "[INFO] finetune diag: deleted_task_id={task_id} active_tasks={active} "
                "num_updated_params={updated} do_finetune={do_ft} "
                "retrain_steps={steps} source={source} from_cli={from_cli} "
                "buffer_sizes={buffers}".format(
                    task_id=task_id,
                    active=active_tasks,
                    updated=num_updated_params,
                    do_ft=do_finetune,
                    steps=retrain_steps,
                    source=retrain_source,
                    from_cli=retrain_from_cli,
                    buffers=buffer_sizes,
                )
            )

            if do_finetune and self.args.allow_zero_retrain and self.args.retrain_steps == 0 and retrain_steps == 0:
                info["finetune_diag"]["retrain_steps"] = 0
                info["finetune_diag"]["retrain_steps_source"] = "retrain_steps"
                info["finetune_diag"]["retrain_steps_from_cli"] = True
                info["finetune_diag"]["skipped_due_to_zero_retrain"] = True
                info["finetune_diag"]["do_finetune"] = False
                print("[INFO] finetune skipped: allow_zero_retrain enabled and retrain_steps=0")
                do_finetune = False

            # (7) Then finetune a subset of these weights which were used in other tasks
            if do_finetune:
                self.finetuning_hist[task_id] = (to_finetune, active_tasks)

                finetune_opt = self.init_optimizer()
                if retrain_steps == 0 and not self.args.allow_zero_retrain:
                    fallback_steps = 50
                    print(
                        "[WARN] retrain_steps resolved to 0 with overlap; defaulting to {steps}".format(
                            steps=fallback_steps
                        )
                    )
                    retrain_steps = fallback_steps
                    info["finetune_diag"]["retrain_steps"] = retrain_steps
                    info["finetune_diag"]["retrain_steps_fallback"] = fallback_steps
                info["protection"] = {
                    "active": use_protection,
                    "method_variant": self.method_variant,
                    "protect_ratio": self.args.protect_ratio,
                    "protect_threshold": self.args.protect_threshold,
                    "lambda_protect": self.args.lambda_protect,
                    "retrain_steps": retrain_steps,
                    "adaptive_retrain": self.args.adaptive_retrain,
                }

                t_retrain_start = time.perf_counter()
                self.log_progress(
                    f"retrain phase start: task={task_id} steps={retrain_steps} active_tasks={active_tasks}"
                )
                for _ in range(retrain_steps):
                    finetune_opt.zero_grad()
                    loss = 0.0
                    for t_id in active_tasks:
                        if self.alpha > 0.0:
                            x_past, y_past, h_past = self.memory.sample_task(self.args.batch_size // len(active_tasks), t_id)
                            h = self.forward(x_past, t_id, mask=self.per_task_masks[t_id], mode="test")
                            loss += self.alpha * self.der_loss(h, h_past, t_id) / len(active_tasks)
                        if self.beta > 0.0:
                            x_past, y_past, h_past = self.memory.sample_task(self.args.batch_size // len(active_tasks), t_id)
                            h = self.forward(x_past, t_id, mask=self.per_task_masks[t_id], mode="test")
                            loss += self.beta * self.loss_fn(h, y_past) / len(active_tasks)
                    if use_protection and self.args.lambda_protect > 0.0 and crit_refs:
                        reg = 0.0
                        for key, ref in crit_refs.items():
                            param = param_map.get(key)
                            if param is None:
                                continue
                            reg += (param[ref["mask"]] - ref["values"]).pow(2).sum()
                        # Adaptive protection: scale the penalty by the measured
                        # critical-overlap ratio so protection strengthens as the
                        # forget/retain overlap grows (1x at zero overlap, up to 2x).
                        lambda_eff = self.args.lambda_protect
                        if getattr(self.args, "adaptive_protect", False):
                            lambda_eff = lambda_eff * (1.0 + s_share_crit_ratio)
                        loss += lambda_eff * reg
                    loss.backward()

                    for key in to_finetune.keys():
                        key_split = key.split('.')
                        module_attr = key_split[-1]
                        if 'classifier' in key_split or len(key_split) == 2:  # e.g., conv1.weight or classifier.weight
                            module_name = key_split[0]
                            if hasattr(getattr(self.net, module_name), module_attr):
                                if getattr(getattr(self.net, module_name), module_attr) is not None:
                                    getattr(getattr(self.net, module_name), module_attr).grad[to_finetune[key] == 0] = 0
                        elif len(key_split) == 4:  # e.g., layer1.0.conv1.weight or encoder_layers.0.mlp_lin1.weight
                            curr_module = getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2])
                            if hasattr(curr_module, module_attr):
                                if getattr(curr_module, module_attr) is not None:
                                    getattr(curr_module, module_attr).grad[to_finetune[key] == 0] = 0
                        elif len(key_split) == 5:  # e.g., encoder_layers.0.mha.W_V.weight
                            curr_module = getattr(getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2]), key_split[3])
                            if hasattr(curr_module, module_attr):
                                if getattr(curr_module, module_attr) is not None:
                                    getattr(curr_module, module_attr).grad[to_finetune[key] == 0] = 0
                        else:
                            raise NotImplementedError('This should not happen with the currently implemented models!')

                    if self.args.weight_decay != 0.0:
                        for n, p in self.net.named_parameters():    # make sure no weight decay for the scores
                            if n.endswith("weight") or n.endswith("bias"):
                                if to_finetune[n] is not None:
                                    p.grad += self.args.weight_decay * p * to_finetune[n]

                    finetune_opt.step()
                info["t_retrain"] = time.perf_counter() - t_retrain_start
                self.log_progress(
                    f"retrain phase end: task={task_id} elapsed={self._format_elapsed(info['t_retrain'])}"
                )
            else:
                info["protection"] = {
                    "active": use_protection,
                    "method_variant": self.method_variant,
                    "protect_ratio": self.args.protect_ratio,
                    "protect_threshold": self.args.protect_threshold,
                    "lambda_protect": self.args.lambda_protect,
                    "retrain_steps": 0,
                    "adaptive_retrain": self.args.adaptive_retrain,
                }
                self.log_progress(f"retrain phase skipped: task={task_id} updated_params=0")

            if debug_context is not None:
                after_retrain_norms = {
                    "S_t": self._masked_l2_norm(param_map, s_t_masks),
                    "S_share": self._masked_l2_norm(param_map, s_share_masks),
                    "S_share_crit": self._masked_l2_norm(param_map, s_share_crit_masks),
                }
                if crit_refs:
                    diff_norms["S_share_crit_after_retrain"] = self._diff_norm_from_refs(param_map, crit_refs)

            # (8) Recreate the combined masks
            self.combined_masks = {}
            for task in self.per_task_masks.keys():
                if self.combined_masks == {}:
                    self.combined_masks = deepcopy(self.per_task_masks[task])
                else:
                    for key in self.per_task_masks[task].keys():
                        if self.combined_masks[key] is not None and self.per_task_masks[task][key] is not None:
                            self.combined_masks[key] = 1 - ((1 - self.combined_masks[key]) * (1 - self.per_task_masks[task][key]))

            # Print sparsity metrics
            connectivity_per_layer = self.print_connectivity(self.combined_masks)
            all_connectivity = self.global_connectivity(self.combined_masks)
            print("Connectivity per Layer: {}".format(connectivity_per_layer))
            print("Global Connectivity: {}".format(all_connectivity))

        else:   # If task_id was the only task remaining so far, then reset all.
            s_t_count = self._count_mask_entries(s_t_masks)
            info.update(
                {
                    "s_t": s_t_count,
                    "s_share": 0,
                    "s_share_crit": 0,
                    "s_share_ratio": 0.0,
                    "s_share_crit_ratio": 0.0,
                    "protection": {
                        "active": self.method_variant == "modified",
                        "method_variant": self.method_variant,
                        "protect_ratio": self.args.protect_ratio,
                        "protect_threshold": self.args.protect_threshold,
                        "lambda_protect": self.args.lambda_protect,
                        "retrain_steps": 0,
                        "adaptive_retrain": self.args.adaptive_retrain,
                    },
                }
            )
            param_map = dict(self.net.named_parameters())
            if debug_context is not None:
                before_norms = {
                    "S_t": self._masked_l2_norm(param_map, s_t_masks),
                    "S_share": 0.0,
                    "S_share_crit": 0.0,
                }
            self.combined_masks = {}
            t_reset_start = time.perf_counter()
            self.log_progress(f"reset phase start: task={task_id}")
            self.net.reinit_weights(self.combined_masks)
            info["t_reset"] = time.perf_counter() - t_reset_start
            self.log_progress(
                f"reset phase end: task={task_id} elapsed={self._format_elapsed(info['t_reset'])}"
            )
            if eval_fn is not None:
                info["after_reset_eval"] = eval_fn("after_reset")
            if debug_context is not None:
                after_reset_norms = {
                    "S_t": self._masked_l2_norm(param_map, s_t_masks),
                    "S_share": 0.0,
                    "S_share_crit": 0.0,
                }

        if debug_context is not None:
            debug_path = Path(debug_context["debug_dir"]) / f"unlearn_{debug_context['unlearning_step']}_task{task_id}"
            debug_masks = {
                "S_t": s_t_masks,
                "S_share": s_share_masks,
                "S_share_crit": s_share_crit_masks,
            }
            if "to_reset" in locals():
                debug_masks["to_reset"] = to_reset
            if "to_finetune" in locals():
                debug_masks["to_finetune"] = to_finetune
            indices = {
                "S_t": self._mask_indices(s_t_masks),
                "S_share": self._mask_indices(s_share_masks),
                "S_share_crit": self._mask_indices(s_share_crit_masks),
            }
            norms = {
                "before_reset": before_norms,
                "after_reset": after_reset_norms,
                "after_retrain": after_retrain_norms,
            }
            self._dump_unlearning_debug(debug_path, debug_masks, indices, norms, diff_norms)

        self.log_progress(
            f"forget phase end: task={task_id} elapsed={self._format_elapsed(self._elapsed_since(forget_start))}"
        )
        return info if return_info else None

    def compute_overlap_matrix(self, include_forgotten=True):
        mask_map = {}
        if include_forgotten:
            mask_map.update(self.archived_task_masks)
        mask_map.update(self.per_task_masks)

        task_ids = sorted(mask_map.keys())
        matrix = []
        for task_i in task_ids:
            row = []
            masks_i = mask_map[task_i]
            for task_j in task_ids:
                masks_j = mask_map[task_j]
                inter, union = 0, 0
                for key, mask_i in masks_i.items():
                    mask_j = masks_j.get(key)
                    if mask_i is None or mask_j is None:
                        continue
                    mi = mask_i.detach().to(dtype=torch.bool, device="cpu")
                    mj = mask_j.detach().to(dtype=torch.bool, device="cpu")
                    inter += torch.logical_and(mi, mj).sum().item()
                    union += torch.logical_or(mi, mj).sum().item()
                row.append(inter / union if union else 0.0)
            matrix.append(row)
        return {"task_ids": task_ids, "matrix": matrix}

    def print_connectivity(self, combined_masks, percent=1.0):
        connectivity_dict = {}
        for key in combined_masks.keys():
            mask = combined_masks[key]
            if mask is not None:
                connectivity = torch.sum(mask == 1) / np.prod(mask.shape)
                connectivity_dict[key] = connectivity * percent
        return connectivity_dict

    def global_connectivity(self, combined_masks):
        denum, num = 0, 0
        for key in combined_masks.keys():
            mask = combined_masks[key]
            if mask is not None:
                num += torch.sum(mask == 1).item()
                denum += np.prod(mask.shape)
        return num / denum
