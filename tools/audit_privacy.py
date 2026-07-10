#!/usr/bin/env python3
"""
Empirical privacy audit over dumped per-sample MIA scores.

Companion to ``tools/audit_seed_completeness.py`` (same run-directory walking
style). It scans ``runs/`` for run directories that contain a ``mia_scores/``
folder -- written by main.py's ``compute_mia`` when ``--eval_mia`` is set -- and
turns the raw per-sample member/non-member scores into an empirical epsilon
lower bound in the spirit of Steinke, Nasr & Jagielski, "Privacy Auditing with
One (1) Training Run" (NeurIPS 2024).

For one before/after MIA dump we treat "member" as the positive class and, over
a grid of score thresholds (predict member when score >= threshold; higher
stored score == more member-like), form a one-sided 95% Clopper-Pearson lower
bound on the TPR and upper bound on the FPR. The audited privacy lower bound is

    eps_hat = max_threshold  log( TPR_lower_95 / FPR_upper_95 ),

clipped at 0 (a negative max, or no threshold with a usable positive FPR upper
bound, yields eps_hat = 0). Clopper-Pearson upper bounds are strictly positive
even with zero observed false positives, so division by zero does not occur.

NOTE: runs completed before the per-sample dump was added have no ``mia_scores/``
folder -- only the aggregated auc/acc survived in metrics.json -- so this tool
reports 0 rows for them until those ``--eval_mia`` configs are re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse the shared regime heuristic instead of reimplementing it.
try:
    from make_report_table import derive_regime
except ImportError:  # pragma: no cover - support package-style invocation
    from tools.make_report_table import derive_regime


# --------------------------------------------------------------------------- #
# Clopper-Pearson bounds (scipy if available, else a self-contained fallback)  #
# --------------------------------------------------------------------------- #
try:
    from scipy.stats import beta as _scipy_beta

    def _beta_ppf(q: float, a: float, b: float) -> float:
        return float(_scipy_beta.ppf(q, a, b))

    _CP_BACKEND = "scipy"
except ImportError:  # pragma: no cover - exercised only without scipy
    def _betacf(x: float, a: float, b: float) -> float:
        """Continued fraction for the incomplete beta function (Numerical Recipes)."""
        MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < EPS:
                break
        return h

    def _betai(x: float, a: float, b: float) -> float:
        """Regularized incomplete beta function I_x(a, b)."""
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
        if x < (a + 1.0) / (a + b + 2.0):
            return bt * _betacf(x, a, b) / a
        return 1.0 - bt * _betacf(1.0 - x, b, a) / b

    def _beta_ppf(q: float, a: float, b: float) -> float:
        """Invert I_x(a, b) = q by bisection (fallback beta quantile)."""
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _betai(mid, a, b) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    _CP_BACKEND = "fallback"


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) Clopper-Pearson lower bound on a proportion."""
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    return _beta_ppf(alpha, k, n - k + 1)


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) Clopper-Pearson upper bound on a proportion."""
    if n <= 0:
        return 0.0
    if k >= n:
        return 1.0
    return _beta_ppf(1.0 - alpha, k + 1, n - k)


def eps_hat(member: np.ndarray, nonmember: np.ndarray, alpha: float = 0.05) -> float:
    """Empirical epsilon lower bound from one before/after MIA dump."""
    member = np.asarray(member, dtype=np.float64).ravel()
    nonmember = np.asarray(nonmember, dtype=np.float64).ravel()
    if member.size == 0 or nonmember.size == 0:
        return 0.0
    nm, nn = int(member.size), int(nonmember.size)
    thresholds = np.unique(np.concatenate([member, nonmember]))
    best = 0.0  # clip at 0
    for thr in thresholds:
        tp = int((member >= thr).sum())
        fp = int((nonmember >= thr).sum())
        tpr_lower = cp_lower(tp, nm, alpha)
        fpr_upper = cp_upper(fp, nn, alpha)
        if tpr_lower <= 0.0 or fpr_upper <= 0.0:
            continue
        cand = math.log(tpr_lower / fpr_upper)
        if cand > best:
            best = cand
    return float(best)


# --------------------------------------------------------------------------- #
# Run-directory scanning                                                       #
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _run_meta(run_dir: Path) -> Dict[str, str]:
    config = _load_json(run_dir / "config.json") or {}
    row = {
        "dataset": str(config.get("dataset", "")),
        "method": str(config.get("method", "")),
        "experiment_tag": str(config.get("experiment_tag", "")),
        "seed": str(config.get("seed", "")),
        "regime": "",
    }
    row["regime"] = derive_regime(row)
    return row


def find_score_dirs(root: Path) -> List[Path]:
    """Run directories (parents of a ``mia_scores`` folder) sorted for stability."""
    seen = set()
    out: List[Path] = []
    for scores_dir in sorted(root.rglob("mia_scores")):
        if scores_dir.is_dir() and scores_dir.parent not in seen:
            seen.add(scores_dir.parent)
            out.append(scores_dir.parent)
    return out


def audit_run(run_dir: Path, alpha: float = 0.05) -> List[Dict[str, Any]]:
    meta = _run_meta(run_dir)
    # Group the per-sample dumps by (unlearning_step, task_id) so before/after pair up.
    pairs: Dict[Tuple[Any, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for score_path in sorted((run_dir / "mia_scores").glob("*.json")):
        payload = _load_json(score_path)
        if not payload:
            continue
        phase = payload.get("phase")
        if phase not in ("before", "after"):
            continue
        key = (payload.get("unlearning_step"), payload.get("task_id"))
        pairs[key][phase] = payload

    rows: List[Dict[str, Any]] = []
    for (step, task_id), phases in sorted(pairs.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        before = phases.get("before")
        after = phases.get("after")

        def _eps(payload: Optional[Dict[str, Any]]) -> Optional[float]:
            if not payload:
                return None
            return eps_hat(payload.get("member_scores", []), payload.get("nonmember_scores", []), alpha)

        ref = after or before or {}
        rows.append(
            {
                "dataset": meta["dataset"],
                "method": meta["method"],
                "regime": meta["regime"],
                "experiment_tag": meta["experiment_tag"],
                "seed": meta["seed"],
                "task_id": task_id,
                "eps_hat_before": _eps(before),
                "eps_hat_after": _eps(after),
                "n_members": ref.get("n_members"),
                "n_nonmembers": ref.get("n_nonmembers"),
                "unlearning_step": step,
                "run_dir": str(run_dir),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
CSV_COLUMNS = [
    "dataset",
    "method",
    "regime",
    "experiment_tag",
    "seed",
    "task_id",
    "eps_hat_before",
    "eps_hat_after",
    "n_members",
    "n_nonmembers",
]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _fmt(row.get(col)) for col in CSV_COLUMNS})


def write_markdown(path: Path, rows: List[Dict[str, Any]], summary: List[Tuple[str, str, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Empirical Privacy Audit",
        "",
        f"- Rows (before/after MIA dumps): {len(rows)}",
        "",
        "| " + " | ".join(CSV_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in CSV_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in CSV_COLUMNS) + " |")
    lines.extend(["", "## Mean eps_hat_after by dataset x method", ""])
    if summary:
        lines.append("| dataset | method | mean_eps_hat_after | n |")
        lines.append("| --- | --- | --- | --- |")
        for dataset, method, mean_eps, count in summary:
            lines.append(f"| {dataset} | {method} | {mean_eps:.4f} | {count} |")
    else:
        lines.append("_no rows_")
    path.write_text("\n".join(lines) + "\n")


def summarize(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, float, int]]:
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        eps = row.get("eps_hat_after")
        if eps is None:
            continue
        buckets[(row["dataset"], row["method"])].append(float(eps))
    out = []
    for (dataset, method), values in sorted(buckets.items()):
        out.append((dataset, method, float(np.mean(values)), len(values)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical privacy audit from dumped per-sample MIA scores.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Run artifact root to scan.")
    parser.add_argument("--out-csv", type=Path, default=Path("results/aggregates/privacy_audit.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("results/aggregates/privacy_audit.md"))
    parser.add_argument("--alpha", type=float, default=0.05, help="One-sided CP tail (95%% bounds -> 0.05).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"[ERROR] Run root does not exist: {args.root}", file=sys.stderr)
        return 1
    run_dirs = find_score_dirs(args.root)
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(audit_run(run_dir, alpha=args.alpha))

    write_csv(args.out_csv, rows)
    summary = summarize(rows)
    write_markdown(args.out_md, rows, summary)

    print(f"Empirical privacy audit ({_CP_BACKEND} CP backend)")
    print(f"  Run dirs with mia_scores/: {len(run_dirs)}")
    print(f"  before/after audit rows:   {len(rows)}")
    if not rows:
        print("  (no mia_scores/ dumps found -- re-run the --eval_mia configs to populate this audit.)")
    print(f"  Wrote: {args.out_csv}")
    print(f"  Wrote: {args.out_md}")
    if summary:
        print("\nMean eps_hat_after by dataset x method:")
        for dataset, method, mean_eps, count in summary:
            print(f"  {dataset:10s} {method:16s} mean_eps_hat_after={mean_eps:.4f}  (n={count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
