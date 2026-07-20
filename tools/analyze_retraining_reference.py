#!/usr/bin/env python3
"""Extract the strict same-method retraining-reference audit from completed runs."""

import argparse
import csv
import json
from pathlib import Path


TAG = "adapter_retraining_reference_v2"


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def select_latest(root, tag):
    selected = {}
    for config_path in root.glob("**/config.json"):
        config = load_json(config_path)
        if not isinstance(config, dict):
            continue
        if config.get("experiment_tag") != tag or config.get("method") != "pall_adapter":
            continue
        metrics = load_json(config_path.with_name("metrics.json"))
        if not isinstance(metrics, dict):
            continue
        events = metrics.get("unlearning_events") or []
        agreement = next(
            (
                event.get("agreement")
                for event in reversed(events)
                if isinstance(event.get("agreement"), dict)
            ),
            None,
        )
        if not isinstance(agreement, dict):
            continue
        key = (config.get("dataset"), int(config.get("seed")))
        candidate = (config_path.parent.name, config_path, config, agreement)
        if key not in selected or candidate[0] > selected[key][0]:
            selected[key] = candidate
    return [selected[key] for key in sorted(selected)]


def build_rows(selected):
    rows = []
    for _timestamp, config_path, config, audit in selected:
        rows.append({
            "dataset": config.get("dataset"),
            "seed": int(config.get("seed")),
            "reference": audit.get("reference"),
            "reference_method_class": audit.get("reference_method_class"),
            "forgotten_task_id": audit.get("forgotten_task_id"),
            "agreement_forget": audit.get("agreement_forget"),
            "agreement_retained_mean": audit.get("agreement_retained_mean"),
            "js_forget": audit.get("js_forget"),
            "js_retained_mean": audit.get("js_retained_mean"),
            "logit_l2_forget": audit.get("logit_l2_forget"),
            "logit_l2_retained_mean": audit.get("logit_l2_retained_mean"),
            "feature_cosine_forget": audit.get("feature_cosine_forget"),
            "feature_cosine_retained_mean": audit.get("feature_cosine_retained_mean"),
            "source_run": str(config_path.parent),
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset", "seed", "source_run"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def write_markdown(path, rows, tag):
    lines = [
        "# Same-Method Retraining-Reference Audit",
        "",
        f"Strict experiment tag: `{tag}`. Each reference is a fresh instance of the same method and architecture, trained without the forgotten task.",
        "All output metrics use the task-local class slice. This is a diagnostic comparison, not proof of exact unlearning.",
        "",
        "| Dataset | Seed | Agreement F/R | JS F/R | Logit L2 F/R | Feature cosine F/R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | "
            f"{fmt(row['agreement_forget'])}/{fmt(row['agreement_retained_mean'])} | "
            f"{fmt(row['js_forget'])}/{fmt(row['js_retained_mean'])} | "
            f"{fmt(row['logit_l2_forget'])}/{fmt(row['logit_l2_retained_mean'])} | "
            f"{fmt(row['feature_cosine_forget'])}/{fmt(row['feature_cosine_retained_mean'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--tag", default=TAG)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("results/aggregates/retraining_reference"),
    )
    args = parser.parse_args()

    rows = build_rows(select_latest(args.root, args.tag))
    if not rows:
        raise SystemExit(f"No completed retraining-reference runs for tag={args.tag!r}")
    write_csv(args.out_prefix.with_suffix(".csv"), rows)
    write_markdown(args.out_prefix.with_suffix(".md"), rows, args.tag)
    print(f"Selected {len(rows)} latest run(s); wrote {args.out_prefix}.csv/.md")


if __name__ == "__main__":
    main()
