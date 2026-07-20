import math
import time
import torch
import models
from .base import *
from .er import RehearsalMemory


class PALLAdapter(Base):
    """Parameter-efficient, overlap-aware adapter forgetting.

    Architecture: frozen ResNet backbone (+ frozen BN), one task-specific
    bottleneck adapter per task, an optional *shared* bottleneck adapter, and a
    single classifier. Forgetting task ``t`` performs:
      1. reset of adapter[t] and of classifier rows for task ``t``;
      2. (shared adapter only) a soft-masked gradient-ASCENT step on the shared
         adapter that erodes forget-task knowledge while protecting parameters
         important to the retained tasks.

    Overlap masks (only meaningful when a shared adapter exists):
      * ``shared_forget_mask``  : top-``adapter_shared_forget_ratio`` shared params
        by |grad L| on the FORGET task's buffer  -> S_forget.
      * ``shared_active_mask``  : top-``adapter_shared_protect_ratio`` shared params
        by |grad L| on the ACTIVE tasks' buffers -> S_active.
      * ``shared_critical_mask``: S_forget AND S_active                -> S_share_crit.
      * hard-protected subset (frozen) + soft-scaled remainder inside the critical set.

    CAVEATS (read before citing results):
      * If ``adapter_shared_bottleneck == 0`` (the DEFAULT) there is no shared
        adapter, so steps involving overlap are no-ops and forgetting degenerates
        to adapter + classifier RESET only. Likewise if
        ``adapter_shared_forget_ratio`` / ``adapter_shared_protect_ratio`` are 0
        (defaults). The overlap-aware behaviour only activates when these flags
        are set explicitly (paper config: bottleneck 16, alpha_f 0.3, alpha_p 0.2).
      * Phase 3 is the ITERATIVE uniform-target soft-masked loop of Algorithm 1
        (``_run_phase3_shared_forgetting``, ``--adapter_forget_steps`` iterations):
        each step minimises the uniform-target loss on replayed forget samples and
        applies ``w -= lr * m_soft * grad`` with the masks held fixed -- matching
        the WorstDrop theorem. (Set ``--adapter_forget_steps 1`` for the old
        single-step behaviour.)
      * The hard-protected subset is ranked by retained-task gradient importance,
        or by gradient-CONFLICT energy ``relu(-g_forget*g_retain)`` under
        ``--protect_importance conflict`` (recommended for HIGH overlap), mirroring
        ``pall_modified``.
    """

    def __init__(self, args):
        super(PALLAdapter, self).__init__(args)
        # Optionally swap the from-scratch feature extractor for a frozen ImageNet
        # backbone. Base already built the default net; rebuild only when requested
        # so default (pretrained_backbone=none) runs are byte-for-byte unchanged.
        if getattr(args, "pretrained_backbone", "none") != "none":
            self.net = models.__dict__[args.arch](
                args.class_per_task * args.n_tasks,
                n_tasks=args.n_tasks,
                sparsity=args.sparsity,
                norm_params=args.norm_params,
                adapter_bottleneck=args.adapter_bottleneck,
                adapter_shared_bottleneck=args.adapter_shared_bottleneck,
                adapter_location=args.adapter_location,
                pretrained_backbone=args.pretrained_backbone,
                pretrained_weights=args.pretrained_weights,
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
        self.method_note = (
            "pall_adapter uses parameter-efficient overlap-aware adapter forgetting "
            "with partial shared-adapter protection."
        )
        self._method_note_logged = False
        self.archived_task_ids = set()

        self.net.freeze_backbone(train_classifier=self.args.adapter_train_classifier)
        self.net.freeze_backbone_batchnorm()
        self.param_stats = self._build_param_stats()
        self._log_param_stats()
        if not self.args.adapter_train_classifier:
            print("[WARN] pall_adapter classifier training is disabled; only adapters will be optimized.", flush=True)
        self._log_method_note()

    def _build_param_stats(self):
        total_params = self.net.count_total_params()
        trainable_params = self.net.count_trainable_params()
        task_adapter_params = (
            self.net.count_task_adapter_params() if hasattr(self.net, "count_task_adapter_params") else self.net.count_adapter_params()
        )
        shared_adapter_params, total_shared_adapter_params = self.net.count_shared_adapter_params()
        adapter_params = (
            self.net.count_total_adapter_params() if hasattr(self.net, "count_total_adapter_params") else task_adapter_params + shared_adapter_params
        )
        shared_adapter_ratio = float(shared_adapter_params / adapter_params) if adapter_params else 0.0
        return {
            "total_params": int(total_params),
            "num_trainable_params": int(trainable_params),
            "num_adapter_params": int(task_adapter_params),
            "task_adapter_params": int(task_adapter_params),
            "adapter_params": int(adapter_params),
            "shared_adapter_params": int(shared_adapter_params),
            "shared_adapter_ratio": shared_adapter_ratio,
            "classifier_param_count": int(sum(param.numel() for _, param in self._classifier_param_items())),
            "trainable_param_ratio": float(trainable_params / total_params) if total_params else 0.0,
            "adapter_mode": "per_task",
            "adapter_bottleneck": int(self.args.adapter_bottleneck),
            "adapter_shared_bottleneck": int(self.args.adapter_shared_bottleneck),
            "adapter_location": self.args.adapter_location,
            "adapter_train_classifier": bool(self.args.adapter_train_classifier),
            "shared_adapter_enabled": bool(getattr(self.net, "shared_adapter", None) is not None),
            "total_shared_adapter_params": int(total_shared_adapter_params),
        }

    def _log_param_stats(self):
        stats = self.param_stats
        self.log_progress(
            "adapter init: total_params={total} trainable_params={trainable} task_adapter_params={task_adapter} "
            "shared_adapter_params={shared_adapter} shared_adapter_ratio={shared_ratio:.6f} "
            "adapter_params_total={adapter_total} classifier_param_count={classifier} "
            "trainable_ratio={ratio:.6f} adapter_location={location} "
            "train_classifier={train_classifier}".format(
                total=stats["total_params"],
                trainable=stats["num_trainable_params"],
                task_adapter=stats["task_adapter_params"],
                shared_adapter=stats["shared_adapter_params"],
                shared_ratio=stats["shared_adapter_ratio"],
                adapter_total=stats["adapter_params"],
                classifier=stats["classifier_param_count"],
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
        else:
            self.log_progress(
                "shared adapter disabled: shared_adapter_params={shared} task_adapter_params={task_adapter} "
                "shared_adapter_ratio={shared_ratio:.6f} adapter_shared_bottleneck={bottleneck}".format(
                    shared=stats["shared_adapter_params"],
                    task_adapter=stats["task_adapter_params"],
                    shared_ratio=stats["shared_adapter_ratio"],
                    bottleneck=stats["adapter_shared_bottleneck"],
                )
            )

    def _log_method_note(self):
        if not self._method_note_logged:
            print(f"[WARN] {self.method_note}", flush=True)
            self._method_note_logged = True

    def _shared_adapter_param_items(self):
        if getattr(self.net, "shared_adapter", None) is None:
            return []
        return list(self.net.shared_adapter.named_parameters())

    def _classifier_param_items(self):
        return list(self.net.classifier.named_parameters())

    def _zero_shared_adapter_grads(self):
        for _, param in self._shared_adapter_param_items():
            param.grad = None

    def _zero_classifier_grads(self):
        for _, param in self._classifier_param_items():
            param.grad = None

    def _zeros_like_shared_adapter(self):
        return {name: torch.zeros_like(param) for name, param in self._shared_adapter_param_items()}

    def _zeros_like_classifier(self):
        return {name: torch.zeros_like(param) for name, param in self._classifier_param_items()}

    def _compute_shared_importance(self, task_ids, include_classifier=False):
        param_items = self._shared_adapter_param_items()
        classifier_items = self._classifier_param_items() if include_classifier else []
        if (not param_items and not classifier_items) or not task_ids:
            return (
                self._zeros_like_shared_adapter(),
                self._zeros_like_shared_adapter(),
                self._zeros_like_classifier(),
                0,
            )

        sampled = 0
        loss = 0.0
        batch_size = max(1, self.args.batch_size // max(1, len(task_ids)))
        classifier_requires_grad = [param.requires_grad for _, param in classifier_items]
        if include_classifier:
            for _, param in classifier_items:
                param.requires_grad = True
        for task_id in task_ids:
            if not hasattr(self.memory, "buffer") or task_id not in self.memory.buffer:
                continue
            x_task, y_task = self.memory.sample_task(batch_size, task_id)
            loss = loss + self.loss_fn(self.forward(x_task, task_id), y_task)
            sampled += 1

        if sampled == 0:
            if include_classifier:
                for (_, param), flag in zip(classifier_items, classifier_requires_grad):
                    param.requires_grad = flag
            return (
                self._zeros_like_shared_adapter(),
                self._zeros_like_shared_adapter(),
                self._zeros_like_classifier(),
                0,
            )

        self._zero_shared_adapter_grads()
        self._zero_classifier_grads()
        loss = loss / sampled
        loss.backward()
        grads = {}
        importance = {}
        classifier_grads = {}
        for name, param in param_items:
            grad = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
            grads[name] = grad
            importance[name] = grad.abs()
        for name, param in classifier_items:
            classifier_grads[name] = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
        self._zero_shared_adapter_grads()
        self._zero_classifier_grads()
        if include_classifier:
            for (_, param), flag in zip(classifier_items, classifier_requires_grad):
                param.requires_grad = flag
        return importance, grads, classifier_grads, sampled

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

    def _build_shared_critical_mask(self, shared_forget_masks, shared_active_masks):
        # The shared critical mask captures shared adapter parameters that are
        # both targeted for forgetting and simultaneously important for the
        # remaining active tasks.
        shared_critical_masks = {}
        forget_count = 0
        active_count = 0
        critical_count = 0
        for name, forget_mask in shared_forget_masks.items():
            active_mask = shared_active_masks.get(name)
            if active_mask is None:
                active_mask = torch.zeros_like(forget_mask, dtype=torch.bool)
            critical_mask = torch.logical_and(forget_mask, active_mask)
            shared_critical_masks[name] = critical_mask
            forget_count += int(forget_mask.sum().item())
            active_count += int(active_mask.sum().item())
            critical_count += int(critical_mask.sum().item())
        return shared_critical_masks, forget_count, active_count, critical_count

    def _resolve_shared_protect_strength(self, overlap_count, forget_count):
        if self.args.adapter_shared_protect_strength is None:
            if forget_count <= 0:
                return 0.0
            return float(overlap_count / max(1, forget_count))
        return float(max(0.0, min(1.0, self.args.adapter_shared_protect_strength)))

    def _select_hard_protected_mask(self, importance_map, critical_masks, ratio):
        if ratio <= 0.0 or not critical_masks:
            return self._zeros_like_shared_adapter(), 0

        score_chunks = []
        meta = []
        total_critical = 0
        for name, critical_mask in critical_masks.items():
            if critical_mask is None:
                continue
            flat_critical = critical_mask.reshape(-1)
            critical_indices = torch.nonzero(flat_critical, as_tuple=False).reshape(-1)
            if critical_indices.numel() == 0:
                continue
            importance = importance_map.get(name)
            if importance is None:
                importance = torch.zeros_like(critical_mask, dtype=torch.float32)
            flat_scores = importance.reshape(-1)[critical_indices]
            score_chunks.append(flat_scores)
            meta.append((name, critical_mask.shape, critical_indices, flat_critical.numel(), critical_mask.device))
            total_critical += int(critical_indices.numel())

        if total_critical == 0:
            return self._zeros_like_shared_adapter(), 0

        k = min(int(math.ceil(ratio * total_critical)), total_critical)
        if k <= 0:
            return self._zeros_like_shared_adapter(), 0

        all_scores = torch.cat(score_chunks)
        top_indices = torch.topk(all_scores, k, largest=True, sorted=False).indices
        selected = torch.zeros_like(all_scores, dtype=torch.bool)
        selected[top_indices] = True

        hard_protected_masks = self._zeros_like_shared_adapter()
        offset = 0
        protected_count = 0
        for name, shape, critical_indices, numel, device in meta:
            local_selected = selected[offset:offset + critical_indices.numel()]
            flat_mask = torch.zeros(numel, dtype=torch.bool, device=device)
            flat_mask[critical_indices[local_selected]] = True
            hard_protected_masks[name] = flat_mask.view(shape)
            protected_count += int(local_selected.sum().item())
            offset += critical_indices.numel()
        return hard_protected_masks, protected_count

    def _uniform_forget_loss(self, x, task_id):
        """Cross-entropy of the forget task's logits against a UNIFORM target.

        Implements the paper's Phase-3 objective L_unlearn = E[-(1/|C|) sum_c log p_c]:
        it pushes the forget task's class distribution toward uniform (maximum
        entropy), i.e. erases the ability to discriminate the forgotten classes.
        Only the forget task's own class slice is scored; the rest is already
        masked to -inf inside ``forward``.
        """
        logits = self.forward(x, task_id)
        start = task_id * self.cpt
        end = start + self.cpt
        log_probs = torch.log_softmax(logits[:, start:end], dim=1)
        return -log_probs.mean()

    def _compute_shared_conflict(self, deleted_grads, active_grads):
        """Per-parameter gradient-CONFLICT energy on the shared adapter.

        ``relu(-g_forget * g_retain)``: large exactly where the forget-task and
        retained-task gradients have OPPOSITE signs and both are sizeable -- the
        shared-adapter weights where forgetting and retention genuinely compete.
        Used to pick the hard-protected subset when ``--protect_importance conflict``.
        """
        conflict = {}
        for name, param in self._shared_adapter_param_items():
            g_f = deleted_grads.get(name)
            g_r = active_grads.get(name)
            if g_f is None or g_r is None:
                conflict[name] = torch.zeros_like(param)
            else:
                conflict[name] = torch.clamp(
                    -(g_f.to(device=param.device, dtype=param.dtype)
                      * g_r.to(device=param.device, dtype=param.dtype)),
                    min=0.0,
                )
        return conflict

    def _compute_bound_check(self, task_id, active_tasks, deleted_grads,
                             shared_forget_mask, shared_critical_mask,
                             hard_protected_shared_mask, protect_strength):
        """WorstDrop first-order bound verification (Theorem thm:worstdrop).

        Phase 3 applies the soft-masked forget update ``w -= eta * m_i * g_forget_i``
        (see ``_run_phase3_shared_forgetting``). To first order, the drop the theorem
        predicts on a retained task ``t`` is ``Delta_t <= E_crit_t + eps_C_t`` with,
        per shared-adapter coordinate ``i``,
            E_crit_t = sum_{i in S_share_crit}       eta * N * m_i * |g_forget_i| * |grad_retain_t_i|
            eps_C_t  = sum_{i in S_forget\\S_active}  eta * N *  1  * |g_forget_i| * |grad_retain_t_i|
        where ``m_i`` is the EXACT Phase-3 soft mask (``1-protect_strength`` on
        soft-critical, ``0`` on hard-protected, ``1`` on forget-exclusive), ``eta`` =
        ``adapter_shared_forget_lr`` and ``N`` = ``adapter_forget_steps``. The total
        forget displacement ``Gamma_i = |sum_k g_i^(k)|`` is approximated by
        ``N*|g_forget_i|`` (the Phase-2 forget gradient reused as the per-step
        magnitude). ``grad_retain_t_i`` is task ``t``'s own retained gradient from ONE
        replay batch (``_compute_shared_importance([t])``). This is a FIRST-ORDER
        predictor in loss units; any measured accuracy drop that exceeds it reflects
        the ``O(eta^2 N^2 G^2)`` second-order term of Eq. (soft-bound) and/or the
        accuracy-vs-risk Lipschitz constant. The MEASURED per-task drop
        (pre-forget minus post-forget accuracy) is merged in by main.py.
        """
        eta = self.args.adapter_shared_forget_lr
        eta = float(self.lr if eta is None else eta)
        n_steps = int(self.args.adapter_forget_steps or 0)
        soft_scale = max(0.0, 1.0 - float(protect_strength))
        names = [name for name, _ in self._shared_adapter_param_items()]

        def _bool(mask_dict, name, like):
            mask = mask_dict.get(name)
            if mask is None:
                return torch.zeros_like(like, dtype=torch.bool)
            return mask.to(dtype=torch.bool, device=like.device)

        per_task = {}
        for t in active_tasks:
            _imp, retain_grads_t, _cg, sampled = self._compute_shared_importance([t])
            e_crit = 0.0
            eps_c = 0.0
            for name in names:
                g_f = deleted_grads.get(name)
                g_r = retain_grads_t.get(name)
                if g_f is None or g_r is None:
                    continue
                forget_mask = shared_forget_mask.get(name)
                if forget_mask is None:
                    continue
                forget_mask = forget_mask.to(dtype=torch.bool, device=g_f.device)
                crit_mask = _bool(shared_critical_mask, name, forget_mask)
                hard_mask = _bool(hard_protected_shared_mask, name, forget_mask)
                soft_crit = torch.logical_and(crit_mask, torch.logical_not(hard_mask))   # m = 1-p
                forget_only = torch.logical_and(forget_mask, torch.logical_not(crit_mask))  # m = 1
                energy = (eta * n_steps) * g_f.abs() * g_r.abs()  # eta*N*|g_forget|*|grad_retain_t|
                e_crit += float((energy * soft_scale)[soft_crit].sum().item())
                eps_c += float(energy[forget_only].sum().item())
            per_task[str(int(t))] = {
                "E_crit": e_crit,
                "eps_C": eps_c,
                "predicted_bound": e_crit + eps_c,
            }
        predicted_worstdrop = max((v["predicted_bound"] for v in per_task.values()), default=0.0)
        return {
            "method": "pall_adapter",
            "eta": eta,
            "n_steps": n_steps,
            "protect_strength": float(protect_strength),
            "soft_scale": soft_scale,
            "gamma_approx": "n_steps * |g_forget_i| (Phase-2 forget gradient)",
            "units": "predicted_bound is in loss/gradient-energy units; measured_drop is an accuracy drop",
            "per_task": per_task,
            "predicted_worstdrop": predicted_worstdrop,
        }

    def _apply_ascent_step_forgetting(self, grad_map, forget_masks, critical_masks,
                                      hard_protected_masks, lr, protect_strength):
        """Single gradient-ASCENT step on the forget task's true-label loss.

        The legacy ``--adapter_forget_mode ascent_step`` rule: one soft-masked
        step ``w += lr * m_soft * g_forget`` using the precomputed forget-task
        gradient (gradient ascent erodes forget-task accuracy). Same soft mask as
        the iterative loop: full on ``S_forget_only``, scaled by
        ``1-protect_strength`` on ``S_share_crit``, frozen on the hard-protected
        subset and outside ``S_forget``. Returns (updated, full, soft) with
        updated = full + soft so the protected+updated == shared_forget invariant
        holds structurally.
        """
        lr = self.lr if lr is None else lr
        soft_scale = max(0.0, 1.0 - protect_strength)
        full_count = 0
        soft_count = 0
        with torch.no_grad():
            for name, param in self._shared_adapter_param_items():
                grad = grad_map.get(name)
                forget_mask = forget_masks.get(name)
                if grad is None or forget_mask is None or not torch.any(forget_mask):
                    continue
                critical_mask = critical_masks.get(name)
                if critical_mask is None:
                    critical_mask = torch.zeros_like(forget_mask, dtype=torch.bool)
                hard_protected_mask = hard_protected_masks.get(name)
                if hard_protected_mask is None:
                    hard_protected_mask = torch.zeros_like(forget_mask, dtype=torch.bool)
                full_mask = torch.logical_and(forget_mask, torch.logical_not(critical_mask))
                soft_mask = torch.logical_and(critical_mask, torch.logical_not(hard_protected_mask))
                if torch.any(full_mask):
                    param.add_(grad * full_mask.to(dtype=grad.dtype), alpha=lr)
                if soft_scale > 0.0 and torch.any(soft_mask):
                    param.add_(grad * soft_mask.to(dtype=grad.dtype), alpha=lr * soft_scale)
                full_count += int(full_mask.sum().item())
                soft_count += int(soft_mask.sum().item())
        self.log_progress(
            f"adapter ascent-step forgetting: full={full_count} soft={soft_count} "
            f"soft_scale={soft_scale:.3f}"
        )
        return full_count + soft_count, full_count, soft_count

    def _run_phase3_shared_forgetting(self, task_id, forget_masks, critical_masks,
                                      hard_protected_masks, lr, protect_strength, n_steps):
        """Phase 3: ITERATIVE uniform-target soft-masked forgetting on the shared adapter.

        Matches Algorithm 1 / the WorstDrop theorem: for ``n_steps`` iterations we
        minimise the uniform-target loss on replayed forget samples and apply a
        per-coordinate soft-masked gradient DESCENT step ``w -= lr * m_soft * grad``,
        with the masks held FIXED across steps:
          * full   (forget \\ critical)      : m=1   (surgical erasure);
          * soft   (critical \\ hard-protect): m=1-protect_strength;
          * hard-protected + everything else : m=0   (frozen).
        Returns (updated, full_count, soft_count) where updated = full + soft, so the
        invariant hard_protected + updated == shared_forget_count holds structurally.
        """
        lr = self.lr if lr is None else lr
        soft_scale = max(0.0, 1.0 - protect_strength)
        full_masks, soft_masks = {}, {}
        full_count = 0
        soft_count = 0
        for name, _ in self._shared_adapter_param_items():
            forget_mask = forget_masks.get(name)
            if forget_mask is None or not torch.any(forget_mask):
                continue
            critical_mask = critical_masks.get(name)
            if critical_mask is None:
                critical_mask = torch.zeros_like(forget_mask, dtype=torch.bool)
            hard_protected_mask = hard_protected_masks.get(name)
            if hard_protected_mask is None:
                hard_protected_mask = torch.zeros_like(forget_mask, dtype=torch.bool)
            full_masks[name] = torch.logical_and(forget_mask, torch.logical_not(critical_mask))
            soft_masks[name] = torch.logical_and(critical_mask, torch.logical_not(hard_protected_mask))
            full_count += int(full_masks[name].sum().item())
            soft_count += int(soft_masks[name].sum().item())

        steps_run = 0
        batch_size = max(1, self.args.batch_size)
        for _ in range(max(0, int(n_steps))):
            if not hasattr(self.memory, "buffer") or task_id not in self.memory.buffer:
                break
            x, _y = self.memory.sample_task(batch_size, task_id)
            for param in self.net.parameters():
                param.grad = None
            loss = self._uniform_forget_loss(x, task_id)
            loss.backward()
            with torch.no_grad():
                for name, param in self._shared_adapter_param_items():
                    grad = param.grad
                    if grad is None:
                        continue
                    full_mask = full_masks.get(name)
                    if full_mask is not None and torch.any(full_mask):
                        param.add_(grad * full_mask.to(dtype=grad.dtype), alpha=-lr)
                    soft_mask = soft_masks.get(name)
                    if soft_scale > 0.0 and soft_mask is not None and torch.any(soft_mask):
                        param.add_(grad * soft_mask.to(dtype=grad.dtype), alpha=-lr * soft_scale)
            steps_run += 1
        for param in self.net.parameters():
            param.grad = None

        self.log_progress(
            f"adapter phase-3 iterative forgetting: steps={steps_run} full={full_count} "
            f"soft={soft_count} soft_scale={soft_scale:.3f}"
        )
        return full_count + soft_count, full_count, soft_count

    def _compute_continuous_conflict_mask(self, deleted_grads, active_grads, forget_masks, gamma):
        """Continuous conflict-weighted soft mask (``--adapter_mask_mode continuous``).

        Instead of the discrete full/soft/frozen partition, every coordinate in
        ``S_forget`` gets a per-coordinate multiplier
        ``m_i = clamp(1 - gamma * c_hat_i, 0, 1)`` where ``c_i = relu(-g_forget_i *
        g_retain_i)`` is the SAME gradient-conflict energy used by the discrete
        conflict path (signed forget gradient captured before the forget buffer is
        deleted) and ``c_hat = c / max(c)`` is its global max-normalization.
        Coordinates outside ``S_forget`` stay frozen (``m=0``), exactly as before.
        Returns ``(m_cont, stats)``; ``m_cont`` is a dict of float multiplier tensors
        held FIXED across the Phase-3 iterations. ``stats`` partitions S_forget into
        full (m==1) / soft (0<m<1) / frozen (m==0) so the structural invariant
        ``frozen + (full+soft) == shared_forget`` still holds, and records the
        conflict diagnostics (mean/max of c_hat, selectivity fractions)."""
        conflict = self._compute_shared_conflict(deleted_grads, active_grads)  # relu(-g_f*g_r) >= 0
        max_c = 0.0
        for name, _ in self._shared_adapter_param_items():
            c = conflict.get(name)
            if c is not None and c.numel() > 0:
                max_c = max(max_c, float(c.max().item()))

        m_cont = {}
        sum_c_hat = 0.0
        n_shared = 0
        n_nonzero_conflict_all = 0
        n_forget = 0
        n_nonzero_conflict_forget = 0
        full_count = soft_count = frozen_count = 0
        m_below_half_count = 0
        for name, param in self._shared_adapter_param_items():
            c = conflict.get(name)
            forget_mask = forget_masks.get(name)
            if c is None:
                c = torch.zeros_like(param)
            c_hat = c / max_c if max_c > 0.0 else torch.zeros_like(c)
            m = torch.clamp(1.0 - gamma * c_hat, min=0.0, max=1.0)
            if forget_mask is None:
                fmask = torch.zeros_like(param, dtype=torch.bool)
            else:
                fmask = forget_mask.to(dtype=torch.bool, device=param.device)
            m = m * fmask.to(dtype=m.dtype)  # freeze (m=0) outside S_forget
            m_cont[name] = m
            # diagnostics over ALL shared coordinates
            sum_c_hat += float(c_hat.sum().item())
            n_shared += int(c_hat.numel())
            n_nonzero_conflict_all += int((c > 0).sum().item())
            # partition of S_forget by the fixed multiplier
            in_forget = fmask
            n_forget += int(in_forget.sum().item())
            n_nonzero_conflict_forget += int(((c > 0) & in_forget).sum().item())
            full_count += int(((m >= 1.0) & in_forget).sum().item())
            soft_count += int(((m > 0.0) & (m < 1.0) & in_forget).sum().item())
            frozen_count += int(((m <= 0.0) & in_forget).sum().item())
            m_below_half_count += int(((m < 0.5) & in_forget).sum().item())

        updated_count = full_count + soft_count
        stats = {
            "gamma": float(gamma),
            "max_c_raw": float(max_c),
            "mean_c_hat": float(sum_c_hat / n_shared) if n_shared else 0.0,
            "max_c_hat": 1.0 if max_c > 0.0 else 0.0,
            "n_shared": int(n_shared),
            "n_forget": int(n_forget),
            "full_count": int(full_count),
            "soft_count": int(soft_count),
            "frozen_count": int(frozen_count),
            "updated_count": int(updated_count),
            "pct_m_below_half": float(100.0 * m_below_half_count / n_forget) if n_forget else 0.0,
            "pct_nonzero_conflict_all": float(100.0 * n_nonzero_conflict_all / n_shared) if n_shared else 0.0,
            "pct_nonzero_conflict_forget": float(100.0 * n_nonzero_conflict_forget / n_forget) if n_forget else 0.0,
        }
        return m_cont, stats

    def _run_phase3_continuous_forgetting(self, task_id, m_cont, lr, n_steps):
        """Phase 3 with the CONTINUOUS conflict-weighted multiplier ``m_cont`` (fixed
        across steps): ``w -= lr * m_cont * grad`` of the uniform-target loss."""
        lr = self.lr if lr is None else lr
        steps_run = 0
        batch_size = max(1, self.args.batch_size)
        for _ in range(max(0, int(n_steps))):
            if not hasattr(self.memory, "buffer") or task_id not in self.memory.buffer:
                break
            x, _y = self.memory.sample_task(batch_size, task_id)
            for param in self.net.parameters():
                param.grad = None
            loss = self._uniform_forget_loss(x, task_id)
            loss.backward()
            with torch.no_grad():
                for name, param in self._shared_adapter_param_items():
                    grad = param.grad
                    m = m_cont.get(name)
                    if grad is None or m is None:
                        continue
                    param.add_(grad * m.to(dtype=grad.dtype), alpha=-lr)
            steps_run += 1
        for param in self.net.parameters():
            param.grad = None
        return steps_run

    def _apply_classifier_forgetting_update(self, grad_map, lr):
        lr = self.lr if lr is None else lr
        updated = 0
        with torch.no_grad():
            for name, param in self._classifier_param_items():
                grad = grad_map.get(name)
                if grad is None:
                    continue
                param.add_(grad, alpha=lr)
                updated += int((grad.abs() > 0).sum().item())
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
        assert hasattr(self.memory, "buffer") and task_id in self.memory.buffer, (
            "[PALLAdapter] Forget-data access contract violated: S_forget "
            "estimation and Phase-3 forgetting consume only the forget task's "
            "rehearsal buffer before deletion. No raw-train or held-out fallback "
            "is used."
        )

        self._log_method_note()
        forget_start = time.perf_counter()
        self.log_progress(f"forget phase start: task={task_id}")

        active_tasks = list(remaining_tasks) if remaining_tasks is not None else [t for t in self.prev_tasks if t != task_id]
        component_mode = getattr(self.args, "adapter_component_mode", "full")
        run_shared_update = component_mode in {"full", "uniform_unprotected", "mask_no_ascent"}
        run_classifier_ascent = component_mode in {"full", "uniform_unprotected"}
        run_retained_repair = component_mode != "reset_only"
        eval_component_stages = bool(getattr(self.args, "eval_component_stages", False))
        reset_param_count = self.net.count_adapter_params(task_id)
        shared_param_count, total_shared_params = self.net.count_shared_adapter_params()
        shared_forget_mask = self._zeros_like_shared_adapter()
        shared_active_mask = self._zeros_like_shared_adapter()
        shared_critical_mask = self._zeros_like_shared_adapter()
        hard_protected_shared_mask = self._zeros_like_shared_adapter()
        shared_forget_count = 0
        shared_active_count = 0
        shared_critical_count = 0
        shared_protect_strength = 0.0
        updated_adapter_params = 0
        shared_full_update_params = 0
        shared_soft_update_params = 0
        protected_adapter_params = 0
        hard_protected_adapter_params = 0
        classifier_param_count = int(sum(param.numel() for _, param in self._classifier_param_items()))
        classifier_forget_param_count = 0
        deleted_grads = self._zeros_like_shared_adapter()
        active_importance = self._zeros_like_shared_adapter()
        active_grads = self._zeros_like_shared_adapter()
        classifier_deleted_grads = self._zeros_like_classifier()
        # Data-access contract: the forget-task rehearsal buffer is still present
        # here. S_forget, classifier forget gradients, and the default iterative
        # uniform-target Phase 3 sample from that buffer only; after those
        # forget-data-dependent steps, the buffer is deleted before retained repair.
        if run_shared_update and total_shared_params > 0 and self.args.adapter_shared_forget_ratio > 0.0:
            deleted_importance, deleted_grads, classifier_deleted_grads, _ = self._compute_shared_importance(
                [task_id],
                include_classifier=run_classifier_ascent,
            )
            shared_forget_mask, _, _ = self._topk_shared_masks(
                deleted_importance,
                self.args.adapter_shared_forget_ratio,
            )
        elif run_classifier_ascent and classifier_param_count > 0:
            _, _, classifier_deleted_grads, _ = self._compute_shared_importance(
                [task_id],
                include_classifier=True,
            )
        if (
            run_shared_update
            and total_shared_params > 0
            and self.args.adapter_shared_forget_ratio > 0.0
            and self.args.adapter_shared_protect_ratio > 0.0
        ):
            active_importance, active_grads, _, _ = self._compute_shared_importance(active_tasks)
            shared_active_mask, _, _ = self._topk_shared_masks(
                active_importance,
                self.args.adapter_shared_protect_ratio,
            )
        # Build the explicit critical mask as the overlap between the shared
        # forget-target mask and the active-task protection mask.
        (
            shared_critical_mask,
            shared_forget_count,
            shared_active_count,
            shared_critical_count,
        ) = self._build_shared_critical_mask(shared_forget_mask, shared_active_mask)
        # Within the critical shared region, protect a top-ranked subset
        # according to adapter_shared_protect_ratio. These parameters are held
        # fixed during forgetting, while the remaining critical parameters can
        # still receive a softened forgetting update. The ranking score is the
        # gradient-CONFLICT energy when --protect_importance conflict (protect the
        # weights where forgetting and retention fight hardest), otherwise the
        # plain retained-task gradient importance.
        if getattr(self.args, "protect_importance", "gradient") == "conflict":
            hard_protect_score = self._compute_shared_conflict(deleted_grads, active_grads)
        else:
            hard_protect_score = active_importance
        hard_protected_shared_mask, hard_protected_adapter_params = self._select_hard_protected_mask(
            hard_protect_score,
            shared_critical_mask,
            self.args.adapter_shared_protect_ratio,
        )
        shared_protect_strength = self._resolve_shared_protect_strength(
            shared_critical_count,
            shared_forget_count,
        )
        total_overlap_scope = reset_param_count + total_shared_params
        if total_overlap_scope <= 0:
            total_overlap_scope = max(reset_param_count, total_shared_params, 1)
        share_ratio = shared_param_count / total_overlap_scope if total_overlap_scope else 0.0
        s_share_crit_ratio = shared_critical_count / shared_param_count if shared_param_count else 0.0
        shared_critical_ratio = shared_critical_count / max(1, shared_forget_count) if shared_forget_count else 0.0
        if shared_param_count <= 0:
            self.log_progress(
                "shared overlap inactive: shared_adapter_params=0 task_adapter_params={task_adapter} "
                "shared_adapter_ratio=0.000000 adapter_shared_bottleneck={bottleneck}".format(
                    task_adapter=self.net.count_adapter_params(),
                    bottleneck=self.args.adapter_shared_bottleneck,
                )
            )

        # Gradient Norm Ratio (reviewer metric): forget-vs-retain shared-adapter
        # gradient L2 norms, reusing deleted_grads/active_grads already computed
        # above. Only defined when both shared forget and protect ratios are > 0
        # (otherwise active_grads is not populated); None otherwise.
        grad_norm_ratio = None
        if (
            total_shared_params > 0
            and self.args.adapter_shared_forget_ratio > 0.0
            and self.args.adapter_shared_protect_ratio > 0.0
        ):
            grad_norm_ratio = grad_l2_norm_ratio(deleted_grads, active_grads)

        # WorstDrop first-order bound verification (Theorem thm:worstdrop). Computed
        # here on the PRE-forget model (before reset/Phase-3) from the Phase-2 forget
        # gradient + one-batch per-task retained gradients. Gated by --eval_bound
        # because it costs one extra retained backward pass per active task.
        bound_check = None
        if (
            component_mode == "full"
            and getattr(self.args, "eval_bound", False)
            and shared_forget_count > 0
            and active_tasks
        ):
            bound_check = self._compute_bound_check(
                task_id, active_tasks, deleted_grads,
                shared_forget_mask, shared_critical_mask,
                hard_protected_shared_mask, shared_protect_strength,
            )

        info = {
            "adapter_component_mode": component_mode,
            "component_stages": {
                "target_reset": True,
                "shared_update": bool(run_shared_update),
                "classifier_ascent": bool(run_classifier_ascent),
                "retained_repair": bool(run_retained_repair),
            },
            "stage_evals": {},
            "grad_norm_ratio": grad_norm_ratio,
            "bound_check": bound_check,
            "s_t": int(total_overlap_scope),
            "s_share": int(shared_param_count),
            "s_share_crit": int(shared_critical_count),
            "s_share_ratio": float(share_ratio),
            "s_share_crit_ratio": float(s_share_crit_ratio),
            "S_share": int(shared_param_count),
            "S_share_crit": int(shared_critical_count),
            "S_share_ratio": float(share_ratio),
            "S_share_crit_ratio": float(s_share_crit_ratio),
            "num_updated_params": 0,
            "protection": {
                "active": bool(shared_critical_count > 0),
                "method_variant": "pall_adapter_hard_mask",
                "method_note": self.method_note,
                "shared_forget_count": int(shared_forget_count),
                "shared_active_count": int(shared_active_count),
                "shared_critical_count": int(shared_critical_count),
                "protected_adapter_params": int(protected_adapter_params),
                "hard_protected_adapter_params": int(hard_protected_adapter_params),
                "updated_adapter_params": int(updated_adapter_params),
            },
            "adapter_shared_forget_ratio": float(self.args.adapter_shared_forget_ratio),
            "adapter_shared_protect_ratio": float(self.args.adapter_shared_protect_ratio),
            "shared_protect_strength": float(shared_protect_strength),
            "shared_adapter_params": int(shared_param_count),
            "classifier_param_count": int(classifier_param_count),
            "classifier_forget_param_count": int(classifier_forget_param_count),
            "shared_forget_count": int(shared_forget_count),
            "shared_active_count": int(shared_active_count),
            "shared_critical_count": int(shared_critical_count),
            "shared_critical_ratio": float(shared_critical_ratio),
            "protected_adapter_params": int(protected_adapter_params),
            "hard_protected_adapter_params": int(hard_protected_adapter_params),
            "updated_adapter_params": int(updated_adapter_params),
            "shared_forget_candidates": int(shared_forget_count),
            "shared_active_critical": int(shared_active_count),
            "shared_overlap_critical": int(shared_critical_count),
            "shared_effective_forget_params": int(updated_adapter_params),
            "shared_full_update_params": int(shared_full_update_params),
            "shared_soft_update_params": int(shared_soft_update_params),
            "shared_protected_params": int(shared_critical_count),
            "shared_s_share_crit": int(shared_critical_count),
            "finetune_diag": {
                "active_tasks": active_tasks,
                "deleted_task_id": task_id,
                "retrain_steps": 0,
                "buffer_sizes": {},
            },
        }
        component_eval_time = 0.0

        def record_component_stage(stage):
            nonlocal component_eval_time
            if not eval_component_stages or eval_fn is None:
                return
            eval_start = time.perf_counter()
            info["stage_evals"][stage] = eval_fn(stage)
            component_eval_time += time.perf_counter() - eval_start

        t_target_reset_start = time.perf_counter()
        self.net.reset_task_adapter(task_id)
        self.net.reset_classifier_slice(task_id, self.cpt)
        info["t_target_reset"] = time.perf_counter() - t_target_reset_start
        record_component_stage("after_target_reset")

        t_shared_update_start = time.perf_counter()
        if shared_forget_count > 0:
            if component_mode == "uniform_unprotected":
                zero_critical = {
                    name: torch.zeros_like(mask, dtype=torch.bool)
                    for name, mask in shared_forget_mask.items()
                }
                (
                    updated_adapter_params,
                    shared_full_update_params,
                    shared_soft_update_params,
                ) = self._run_phase3_shared_forgetting(
                    task_id,
                    shared_forget_mask,
                    zero_critical,
                    zero_critical,
                    self.args.adapter_shared_forget_lr,
                    0.0,
                    self.args.adapter_forget_steps,
                )
                hard_protected_adapter_params = 0
                self.log_progress(
                    "adapter component ablation: uniform_unprotected uses the "
                    "same S_forget support with multiplier one"
                )
            elif getattr(self.args, "adapter_mask_mode", "discrete") == "continuous":
                # CONTINUOUS conflict-weighted soft mask: replaces the discrete
                # full/soft/frozen partition with a per-coordinate multiplier over
                # S_forget. The frozen (m==0) subset plays the "hard-protected" role
                # so the structural invariant still holds.
                gamma = float(getattr(self.args, "adapter_conflict_gamma", 1.0))
                m_cont, conflict_mask_stats = self._compute_continuous_conflict_mask(
                    deleted_grads, active_grads, shared_forget_mask, gamma,
                )
                self._run_phase3_continuous_forgetting(
                    task_id, m_cont, self.args.adapter_shared_forget_lr,
                    self.args.adapter_forget_steps,
                )
                updated_adapter_params = int(conflict_mask_stats["updated_count"])
                shared_full_update_params = int(conflict_mask_stats["full_count"])
                shared_soft_update_params = int(conflict_mask_stats["soft_count"])
                hard_protected_adapter_params = int(conflict_mask_stats["frozen_count"])
                info["conflict_mask_stats"] = conflict_mask_stats
                self.log_progress(
                    "adapter phase-3 continuous forgetting: gamma={g} mean_c_hat={mc:.4f} "
                    "full={f} soft={s} frozen={fr} pct_m_below_half={pb:.1f} "
                    "pct_nonzero_conflict_forget={pc:.1f}".format(
                        g=gamma, mc=conflict_mask_stats["mean_c_hat"],
                        f=shared_full_update_params, s=shared_soft_update_params,
                        fr=hard_protected_adapter_params,
                        pb=conflict_mask_stats["pct_m_below_half"],
                        pc=conflict_mask_stats["pct_nonzero_conflict_forget"],
                    )
                )
            elif getattr(self.args, "adapter_forget_mode", "uniform_loop") == "ascent_step":
                (
                    updated_adapter_params,
                    shared_full_update_params,
                    shared_soft_update_params,
                ) = self._apply_ascent_step_forgetting(
                    deleted_grads,
                    shared_forget_mask,
                    shared_critical_mask,
                    hard_protected_shared_mask,
                    self.args.adapter_shared_forget_lr,
                    shared_protect_strength,
                )
            else:
                (
                    updated_adapter_params,
                    shared_full_update_params,
                    shared_soft_update_params,
                ) = self._run_phase3_shared_forgetting(
                    task_id,
                    shared_forget_mask,
                    shared_critical_mask,
                    hard_protected_shared_mask,
                    self.args.adapter_shared_forget_lr,
                    shared_protect_strength,
                    self.args.adapter_forget_steps,
                )
            protected_adapter_params = int(hard_protected_adapter_params)
        info["t_shared_update"] = time.perf_counter() - t_shared_update_start
        record_component_stage("after_shared_update")
        protected_adapter_params = int(protected_adapter_params)
        updated_adapter_params = int(updated_adapter_params)
        hard_protected_adapter_params = int(hard_protected_adapter_params)
        if protected_adapter_params > shared_forget_count:
            raise AssertionError(
                "protected_adapter_params exceeds shared_forget_count: "
                f"{protected_adapter_params} > {shared_forget_count}"
            )
        if updated_adapter_params > shared_forget_count:
            raise AssertionError(
                "updated_adapter_params exceeds shared_forget_count: "
                f"{updated_adapter_params} > {shared_forget_count}"
            )
        if protected_adapter_params + updated_adapter_params != shared_forget_count:
            raise AssertionError(
                "protected_adapter_params + updated_adapter_params must equal shared_forget_count: "
                f"{protected_adapter_params} + {updated_adapter_params} != {shared_forget_count}"
            )
        overlap_analysis = {
            "shared_total": int(shared_param_count),
            "shared_forget": int(shared_forget_count),
            "shared_active": int(shared_active_count),
            "shared_critical": int(shared_critical_count),
            "protected_params": int(protected_adapter_params),
            "hard_protected_params": int(hard_protected_adapter_params),
            "updated_params": int(updated_adapter_params),
            "critical_ratio": float(shared_critical_count / max(shared_forget_count, 1)),
            "protected_ratio": float(protected_adapter_params / max(shared_forget_count, 1)),
            "updated_ratio": float(updated_adapter_params / max(shared_forget_count, 1)),
        }
        info["updated_adapter_params"] = int(updated_adapter_params)
        info["protected_adapter_params"] = int(protected_adapter_params)
        info["hard_protected_adapter_params"] = int(hard_protected_adapter_params)
        info["shared_effective_forget_params"] = int(updated_adapter_params)
        info["shared_full_update_params"] = int(shared_full_update_params)
        info["shared_soft_update_params"] = int(shared_soft_update_params)
        info["shared_protected_params"] = int(protected_adapter_params)
        info["shared_critical_ratio"] = float(overlap_analysis["critical_ratio"])
        info["overlap_analysis"] = overlap_analysis
        info["protection"]["protected_adapter_params"] = int(protected_adapter_params)
        info["protection"]["hard_protected_adapter_params"] = int(hard_protected_adapter_params)
        info["protection"]["updated_adapter_params"] = int(updated_adapter_params)
        info["protection"]["overlap_analysis"] = overlap_analysis
        info["protection"]["active"] = bool(
            run_shared_update
            and component_mode != "uniform_unprotected"
            and shared_critical_count > 0
        )
        t_classifier_ascent_start = time.perf_counter()
        if run_classifier_ascent and classifier_param_count > 0:
            classifier_forget_param_count = self._apply_classifier_forgetting_update(
                classifier_deleted_grads,
                self.args.adapter_shared_forget_lr,
            )
            info["classifier_forget_param_count"] = int(classifier_forget_param_count)
        info["t_classifier_ascent"] = time.perf_counter() - t_classifier_ascent_start
        record_component_stage("after_classifier_ascent")
        if total_shared_params > 0:
            print(
                "[PALLAdapter] critical mask stats: shared_forget={forget_count} shared_active={active_count} "
                "shared_critical={critical_count} protected={protected} updated={updated}".format(
                    forget_count=shared_forget_count,
                    active_count=shared_active_count,
                    critical_count=shared_critical_count,
                    protected=protected_adapter_params,
                    updated=updated_adapter_params,
                ),
                flush=True,
            )
        print(
            "[PALLAdapter] classifier forgetting: classifier_param_count={total} classifier_forget_param_count={updated}".format(
                total=classifier_param_count,
                updated=classifier_forget_param_count,
            ),
            flush=True,
        )
        info["t_reset"] = (
            info["t_target_reset"]
            + info["t_shared_update"]
            + info["t_classifier_ascent"]
        )
        self.archived_task_ids.add(task_id)

        if eval_fn is not None:
            # Backward-compatible field: historically this means the complete
            # pre-repair intervention (target reset + shared update + classifier
            # ascent), not target reset alone. New component experiments use the
            # precise names under stage_evals.
            backward_eval_start = time.perf_counter()
            info["after_reset_eval"] = eval_fn("after_reset")
            if eval_component_stages:
                component_eval_time += time.perf_counter() - backward_eval_start

        if task_id in self.memory.buffer:
            self.memory.remove(task_id)
        assert task_id not in self.memory.buffer, (
            "[PALLAdapter] Forget-task rehearsal buffer must be deleted after "
            "S_forget/Phase-3 access and before retained-task repair."
        )
        if task_id in self.prev_tasks:
            self.prev_tasks.remove(task_id)

        requested_retrain_steps = self._compute_retrain_steps()
        retrain_steps = requested_retrain_steps if run_retained_repair else 0
        buffer_sizes = {}
        for active_task in active_tasks:
            entry = self.memory.buffer.get(active_task) if self.memory.buffer else None
            if entry is None:
                buffer_sizes[str(active_task)] = 0
                continue
            buffer_sizes[str(active_task)] = min(int(entry.get("num_seen", 0)), int(entry["X"].shape[0]))
        info["finetune_diag"]["buffer_sizes"] = buffer_sizes
        info["finetune_diag"]["retrain_steps"] = retrain_steps
        info["finetune_diag"]["requested_retrain_steps"] = requested_retrain_steps

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

        info["t_retained_repair"] = info["t_retrain"]
        record_component_stage("after_retained_repair")
        info["t_component_eval"] = component_eval_time
        info["t_forget_total_raw"] = time.perf_counter() - forget_start
        info["t_forget_total"] = max(0.0, info["t_forget_total_raw"] - component_eval_time)
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
