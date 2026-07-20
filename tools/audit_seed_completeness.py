#!/usr/bin/env python3
"""
Audit intended thesis/reviewer experiment seeds under runs/.

The audit is intentionally read-only: it reports missing, partial, and duplicate
run artifacts without modifying raw runs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from make_thesis_table import (
    CONFIG_GROUP_COLUMNS,
    config_group_value,
    extract_run_row,
    group_key,
    load_json,
    seed_key,
)


SEEDS_01 = ("0", "1")
SEEDS_012 = ("0", "1", "2")
METHODS_STANDARD = (
    "pall_original",
    "pall_modified",
    "pall_adapter",
    "lora",
    "er",
    "derpp",
    "ewc",
    "lwf",
    "clpu",
)


@dataclass(frozen=True)
class ExpectedSpec:
    result_set: str
    dataset: str
    method: str
    experiment_tag: str
    regime: str
    expected_seeds: Tuple[str, ...]
    config: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, ...]:
        effective_config: Dict[str, Any] = {
            "dataset": self.dataset,
            "method": self.method,
            "experiment_tag": self.experiment_tag,
            "protect_importance": "gradient",
            "protect_ratio": "",
            "lambda_protect": 0.0,
            "protect_anchor": "old",  # main.py argparse default, written into every run config
            "adaptive_protect": False,  # main.py argparse default (store_true), written into every run config
            "adapter_bottleneck": 16,
            "adapter_shared_bottleneck": 0,
            "adapter_shared_forget_ratio": 0.0,
            "adapter_shared_protect_ratio": 0.0,
            "adapter_forget_steps": 10,
            "adapter_shared_forget_lr": "",
            "adapter_shared_protect_strength": "",
            "retrain_steps": "",
            "adapter_train_classifier": False,
            "adapter_component_mode": "full",
            "adapter_mask_mode": "discrete",  # main.py argparse default, written into every run config
            "adapter_conflict_gamma": 1.0,  # main.py argparse default, written into every run config
            "pretrained_backbone": (
                "imagenet_resnet18" if self.regime == "pretrained_frozen" else "none"
            ),
            "pretrained_input_norm": "imagenet",
        }
        effective_config.update(self.config)
        row: Dict[str, Any] = {"dataset": self.dataset, "method": self.method}
        for column in CONFIG_GROUP_COLUMNS:
            row[column] = config_group_value(effective_config, column)
        return group_key(row, group_by_config=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit thesis run seed completeness.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Run artifact root.")
    parser.add_argument(
        "--schedules-dir",
        type=Path,
        default=Path("schedules"),
        help="Schedule directory used to infer optional TinyImageNet seed 1.",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional CSV report path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional Markdown report path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when missing seeds, duplicate completed runs, or partial directories are found.",
    )
    return parser.parse_args()


def adapter_config(
    *,
    bottleneck: int = 16,
    shared_bottleneck: int = 16,
    forget_ratio: float = 0.3,
    protect_ratio: float = 0.2,
    forget_steps: int = 10,
) -> Dict[str, Any]:
    return {
        "protect_importance": "gradient",
        "lambda_protect": 0.0,
        "adapter_bottleneck": bottleneck,
        "adapter_shared_bottleneck": shared_bottleneck,
        "adapter_shared_forget_ratio": forget_ratio,
        "adapter_shared_protect_ratio": protect_ratio,
        "adapter_forget_steps": forget_steps,
        "retrain_steps": 50,
        "adapter_train_classifier": True,
    }


def retrain_config() -> Dict[str, Any]:
    return {"protect_importance": "gradient", "lambda_protect": 0.0, "retrain_steps": 50}


def protected_retrain_config() -> Dict[str, Any]:
    return {
        "protect_importance": "gradient",
        "protect_ratio": 0.2,
        "lambda_protect": 1.0,
        "retrain_steps": 50,
    }


def expected_specs(schedules_dir: Path) -> List[ExpectedSpec]:
    specs: List[ExpectedSpec] = []

    def add(
        result_set: str,
        dataset: str,
        methods: Sequence[str],
        tag: str,
        regime: str,
        seeds: Sequence[str] = SEEDS_01,
        configs: Dict[str, Dict[str, Any]] | None = None,
    ) -> None:
        configs = configs or {}
        for method in methods:
            specs.append(
                ExpectedSpec(
                    result_set=result_set,
                    dataset=dataset,
                    method=method,
                    experiment_tag=tag,
                    regime=regime,
                    expected_seeds=tuple(str(seed) for seed in seeds),
                    config=configs.get(method, {}),
                )
            )

    for dataset in ("cifar10", "cifar100"):
        prefix = "cifar10" if dataset == "cifar10" else "cifar100"
        add(
            "overlap-heavy baselines",
            dataset,
            ("ewc", "lwf", "clpu"),
            f"{prefix}_baselines_v2",
            "from_scratch",
        )
        add(
            "overlap-heavy proposed",
            dataset,
            ("pall_original", "pall_modified", "pall_adapter", "lora"),
            f"{prefix}_main",
            "from_scratch",
            configs={
                "pall_original": retrain_config(),
                "pall_modified": protected_retrain_config(),
                "pall_adapter": adapter_config(),
            },
        )
        add(
            "extra unlearning baselines",
            dataset,
            ("ssd", "salun"),
            f"{prefix}_extra_baselines",
            "from_scratch",
        )
        add(
            "standard benchmark",
            dataset,
            METHODS_STANDARD,
            f"{prefix}_standard",
            "standard_split",
            seeds=SEEDS_012,
            configs={
                "pall_original": retrain_config(),
                "pall_modified": protected_retrain_config(),
                "pall_adapter": adapter_config(),
            },
        )
        add(
            "standard unlearning baselines",
            dataset,
            ("ssd", "salun"),
            "standard_unlearning_ssd_salun_v1",
            "standard_split",
            seeds=SEEDS_012,
        )
        add(
            "MIA from-scratch",
            dataset,
            ("pall_modified", "clpu"),
            f"{prefix}_mia",
            "from_scratch",
            seeds=SEEDS_012,
            configs={"pall_modified": protected_retrain_config()},
        )
        add(
            "MIA pretrained PEFT",
            dataset,
            ("pall_adapter", "lora"),
            f"{prefix}_pretrained_mia",
            "pretrained_frozen",
            seeds=SEEDS_012,
            configs={"pall_adapter": adapter_config()},
        )
        add(
            "pretrained PEFT",
            dataset,
            ("pall_adapter", "lora"),
            f"{prefix}_pretrained",
            "pretrained_frozen",
            seeds=SEEDS_012,
            configs={"pall_adapter": adapter_config()},
        )

        for component_mode in (
            "reset_only",
            "reset_repair",
            "uniform_unprotected",
            "mask_no_ascent",
            "full",
        ):
            specs.append(
                ExpectedSpec(
                    result_set="adapter component attribution",
                    dataset=dataset,
                    method="pall_adapter",
                    experiment_tag="adapter_components_pretrained_imagenetnorm_v2",
                    regime="pretrained_frozen",
                    expected_seeds=SEEDS_012,
                    config={
                        **adapter_config(),
                        "adapter_component_mode": component_mode,
                    },
                )
            )

    for forget_ratio in (0.3, 0.5):
        for protect_ratio in (0.2, 0.4):
            for forget_steps in (10, 30):
                specs.append(
                    ExpectedSpec(
                        result_set="pretrained adapter tuning",
                        dataset="cifar100",
                        method="pall_adapter",
                        experiment_tag="adapter_tune_pretrained_v1",
                        regime="pretrained_frozen",
                        expected_seeds=SEEDS_01,
                        config=adapter_config(
                            forget_ratio=forget_ratio,
                            protect_ratio=protect_ratio,
                            forget_steps=forget_steps,
                        ),
                    )
                )

    for bottleneck in (4, 8, 16, 32, 64, 128):
        specs.append(
            ExpectedSpec(
                result_set="adapter bottleneck ablation",
                dataset="cifar10",
                method="pall_adapter",
                experiment_tag="adapter_bottleneck_ablation_v1",
                regime="ablation",
                expected_seeds=("0",),
                config=adapter_config(bottleneck=bottleneck),
            )
        )

    tiny_main_seeds = ["0"]
    if (schedules_dir / "tinyimagenet_t20_f3_seed1.json").exists():
        tiny_main_seeds.append("1")
    add(
        "TinyImageNet main",
        "tinyimagenet",
        ("pall_original", "pall_modified", "clpu"),
        "tiny_main",
        "from_scratch",
        seeds=tiny_main_seeds,
        configs={"pall_original": retrain_config(), "pall_modified": protected_retrain_config()},
    )
    add(
        "TinyImageNet pretrained PEFT",
        "tinyimagenet",
        ("pall_adapter", "lora"),
        "tiny_pretrained",
        "pretrained_frozen",
        seeds=SEEDS_012,
        configs={"pall_adapter": adapter_config()},
    )

    specs.extend(
        [
            ExpectedSpec(
                result_set="TinyImageNet fixed-schedule from-scratch",
                dataset="tinyimagenet",
                method="pall_original",
                experiment_tag="tiny_e3_original_v1",
                regime="from_scratch_fixed_schedule",
                expected_seeds=SEEDS_012,
                config={"adapter_forget_steps": 1},
            ),
            ExpectedSpec(
                result_set="TinyImageNet fixed-schedule from-scratch",
                dataset="tinyimagenet",
                method="pall_modified",
                experiment_tag="tiny_e3_modified_v1",
                regime="from_scratch_fixed_schedule",
                expected_seeds=SEEDS_012,
                config={"adapter_forget_steps": 1},
            ),
            ExpectedSpec(
                result_set="TinyImageNet fixed-schedule from-scratch",
                dataset="tinyimagenet",
                method="pall_adapter",
                experiment_tag="tiny_e3_adapter_v1",
                regime="from_scratch_fixed_schedule",
                expected_seeds=SEEDS_012,
                config={
                    **adapter_config(forget_steps=1),
                    "retrain_steps": "",
                },
            ),
        ]
    )

    for dataset in ("cifar10", "cifar100"):
        for anchor in ("old", "reinit"):
            specs.append(
                ExpectedSpec(
                    result_set="anchor ablation",
                    dataset=dataset,
                    method="pall_modified",
                    experiment_tag="anchor_ablation_v1",
                    regime="ablation_mia",
                    expected_seeds=SEEDS_012,
                    config={**protected_retrain_config(), "protect_anchor": anchor},
                )
            )
    return specs


def find_partial_dirs(root: Path) -> List[Path]:
    partial: List[Path] = []
    for config_path in sorted(root.rglob("config.json")):
        if not config_path.with_name("metrics.json").exists():
            partial.append(config_path.parent)
    return partial


def load_actual_runs(root: Path) -> Dict[Tuple[Tuple[str, ...], str], List[Path]]:
    actual: Dict[Tuple[Tuple[str, ...], str], List[Path]] = {}
    for metrics_path in sorted(root.rglob("metrics.json")):
        row = extract_run_row(metrics_path, group_by_config=True)
        if row is None:
            continue
        actual.setdefault((group_key(row, group_by_config=True), seed_key(row)), []).append(metrics_path.parent)
    return actual


def config_label(spec: ExpectedSpec) -> str:
    parts = []
    for column in CONFIG_GROUP_COLUMNS:
        if column == "experiment_tag":
            continue
        value = spec.config.get(column)
        if value in (None, ""):
            continue
        parts.append(f"{column}={value}")
    return ", ".join(parts) if parts else "-"


def audit(root: Path, schedules_dir: Path) -> Tuple[List[Dict[str, str]], List[Path]]:
    specs = expected_specs(schedules_dir)
    actual = load_actual_runs(root)
    rows: List[Dict[str, str]] = []
    for spec in specs:
        key = spec.key()
        present_seeds: List[str] = []
        duplicate_seeds: List[str] = []
        duplicate_paths: List[str] = []
        for seed in spec.expected_seeds:
            paths = actual.get((key, seed), [])
            if paths:
                present_seeds.append(seed)
            if len(paths) > 1:
                duplicate_seeds.append(seed)
                duplicate_paths.extend(str(path) for path in paths)
        missing_seeds = [seed for seed in spec.expected_seeds if seed not in present_seeds]
        status = "complete"
        if missing_seeds:
            status = "missing"
        if duplicate_seeds:
            status = "duplicate" if status == "complete" else "missing+duplicate"
        rows.append(
            {
                "status": status,
                "result_set": spec.result_set,
                "dataset": spec.dataset,
                "regime": spec.regime,
                "method": spec.method,
                "experiment_tag": spec.experiment_tag,
                "config": config_label(spec),
                "expected_seeds": " ".join(spec.expected_seeds),
                "present_seeds": " ".join(present_seeds),
                "missing_seeds": " ".join(missing_seeds),
                "duplicate_seeds": " ".join(duplicate_seeds),
                "duplicate_paths": "; ".join(duplicate_paths),
            }
        )
    return rows, find_partial_dirs(root)


def print_report(rows: Sequence[Dict[str, str]], partial_dirs: Sequence[Path]) -> None:
    missing = [row for row in rows if row["missing_seeds"]]
    duplicates = [row for row in rows if row["duplicate_seeds"]]
    complete_count = sum(1 for row in rows if row["status"] == "complete")
    print("Seed completeness audit")
    print(f"  Expected method/dataset/regime/config specs: {len(rows)}")
    print(f"  Complete specs: {complete_count}")
    print(f"  Specs with missing seeds: {len(missing)}")
    print(f"  Specs with duplicate completed seeds: {len(duplicates)}")
    print(f"  Partial run directories: {len(partial_dirs)}")
    if missing:
        print("\nMissing seeds:")
        for row in missing:
            print(
                "  - "
                f"{row['result_set']} | {row['dataset']} | {row['regime']} | {row['method']} | "
                f"{row['experiment_tag']} | config: {row['config']} | missing: {row['missing_seeds']}"
            )
    if duplicates:
        print("\nDuplicate completed seeds:")
        for row in duplicates:
            print(
                "  - "
                f"{row['result_set']} | {row['dataset']} | {row['regime']} | {row['method']} | "
                f"{row['experiment_tag']} | config: {row['config']} | duplicate seeds: {row['duplicate_seeds']}"
            )
            print(f"    paths: {row['duplicate_paths']}")
    if partial_dirs:
        print("\nPartial run directories (config.json without metrics.json):")
        for path in partial_dirs:
            print(f"  - {path}")


def write_csv_report(path: Path, rows: Sequence[Dict[str, str]], partial_dirs: Sequence[Path]) -> None:
    columns = [
        "status",
        "result_set",
        "dataset",
        "regime",
        "method",
        "experiment_tag",
        "config",
        "expected_seeds",
        "present_seeds",
        "missing_seeds",
        "duplicate_seeds",
        "duplicate_paths",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        for partial in partial_dirs:
            writer.writerow(
                {
                    "status": "partial",
                    "result_set": "partial run directory",
                    "config": str(partial),
                }
            )


def write_markdown_report(path: Path, rows: Sequence[Dict[str, str]], partial_dirs: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Seed Completeness Audit",
        "",
        f"- Expected specs: {len(rows)}",
        f"- Complete specs: {sum(1 for row in rows if row['status'] == 'complete')}",
        f"- Specs with missing seeds: {sum(1 for row in rows if row['missing_seeds'])}",
        f"- Specs with duplicate completed seeds: {sum(1 for row in rows if row['duplicate_seeds'])}",
        f"- Partial run directories: {len(partial_dirs)}",
        "",
        "| status | result_set | dataset | regime | method | experiment_tag | config | expected_seeds | present_seeds | missing_seeds | duplicate_seeds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                row.get(column, "") or "NA"
                for column in (
                    "status",
                    "result_set",
                    "dataset",
                    "regime",
                    "method",
                    "experiment_tag",
                    "config",
                    "expected_seeds",
                    "present_seeds",
                    "missing_seeds",
                    "duplicate_seeds",
                )
            )
            + " |"
        )
    if partial_dirs:
        lines.extend(["", "## Partial Run Directories", ""])
        lines.extend(f"- `{path}`" for path in partial_dirs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"[ERROR] Run root does not exist: {args.root}", file=sys.stderr)
        return 1
    rows, partial_dirs = audit(args.root, args.schedules_dir)
    print_report(rows, partial_dirs)
    if args.out_csv:
        write_csv_report(args.out_csv, rows, partial_dirs)
        print(f"[INFO] Wrote CSV audit: {args.out_csv}")
    if args.out_md:
        write_markdown_report(args.out_md, rows, partial_dirs)
        print(f"[INFO] Wrote Markdown audit: {args.out_md}")
    has_gap = any(row["missing_seeds"] or row["duplicate_seeds"] for row in rows) or bool(partial_dirs)
    return 2 if args.strict and has_gap else 0


if __name__ == "__main__":
    raise SystemExit(main())
