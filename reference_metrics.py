"""Numerically stable paired metrics for retraining-reference audits."""

import torch
import torch.nn.functional as F


def paired_reference_batch_sums(logits_a, logits_b, features_a=None, features_b=None):
    """Return additive metric sums for one sample-aligned batch.

    ``logits_a`` and ``logits_b`` must contain only the task-local class slice.
    Feature cosine similarity is omitted when feature shapes are incompatible.
    The caller divides each sum by its corresponding count across batches.
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(
            f"paired logits must have identical shapes, got {tuple(logits_a.shape)} "
            f"and {tuple(logits_b.shape)}"
        )
    if logits_a.ndim != 2:
        raise ValueError(f"paired logits must be rank 2, got rank {logits_a.ndim}")

    n = int(logits_a.shape[0])
    if n == 0:
        return {
            "n": 0,
            "agreement_sum": 0.0,
            "js_sum": 0.0,
            "logit_l2_sum": 0.0,
            "feature_n": 0,
            "feature_cosine_sum": 0.0,
        }

    log_p = F.log_softmax(logits_a, dim=1)
    log_q = F.log_softmax(logits_b, dim=1)
    p = log_p.exp()
    q = log_q.exp()
    mixture = 0.5 * (p + q)
    log_mixture = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    js_per_sample = 0.5 * (
        (p * (log_p - log_mixture)).sum(dim=1)
        + (q * (log_q - log_mixture)).sum(dim=1)
    )

    feature_n = 0
    feature_cosine_sum = 0.0
    if (
        features_a is not None
        and features_b is not None
        and features_a.shape == features_b.shape
        and features_a.ndim == 2
        and int(features_a.shape[0]) == n
    ):
        feature_n = n
        feature_cosine_sum = float(
            F.cosine_similarity(features_a, features_b, dim=1, eps=1e-8).sum().item()
        )

    return {
        "n": n,
        "agreement_sum": float(
            (logits_a.argmax(dim=1) == logits_b.argmax(dim=1)).sum().item()
        ),
        "js_sum": float(js_per_sample.sum().item()),
        "logit_l2_sum": float(torch.linalg.vector_norm(logits_a - logits_b, dim=1).sum().item()),
        "feature_n": feature_n,
        "feature_cosine_sum": feature_cosine_sum,
    }
