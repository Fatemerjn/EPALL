"""Small, dependency-free metrics used by the privacy diagnostics."""

import numpy as np


def roc_auc_from_scores(member_scores, nonmember_scores):
    """Mann--Whitney ROC-AUC with average ranks for tied scores."""
    member = np.asarray(member_scores, dtype=np.float64)
    nonmember = np.asarray(nonmember_scores, dtype=np.float64)
    if member.size == 0 or nonmember.size == 0:
        return None
    all_values = np.concatenate([member, nonmember])
    order = np.argsort(all_values, kind="mergesort")
    sorted_values = all_values[order]
    ranks = np.empty(all_values.size, dtype=np.float64)
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = ranks[: member.size].sum()
    auc = (rank_sum - member.size * (member.size + 1) / 2.0) / (
        member.size * nonmember.size
    )
    return float(auc)
