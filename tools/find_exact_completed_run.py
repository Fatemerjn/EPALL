#!/usr/bin/env python3
"""Find a completed run whose effective ``main.py`` config matches exactly.

The expected configuration is obtained from ``main.py``'s argparse declarations
without importing its ML dependencies. Runtime-only placement fields are
ignored, while every experiment argument (including defaults) must match. Exit
status 0 means a completed match was found, 1 means no match, and 2 means the
check itself failed.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ONLY_KEYS = {
    "device",
    "gpu",
    "is_macos",
    "resolved_num_workers",
    "resolved_pin_memory",
    "resolved_persistent_workers",
    "run_dir",
    "timestamp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the newest exact completed match for a main.py command."
    )
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Run artifact root.")
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="Arguments intended for main.py (place after --).",
    )
    args = parser.parse_args()
    if args.main_args[:1] == ["--"]:
        args.main_args = args.main_args[1:]
    if not args.main_args:
        parser.error("main.py arguments are required after --")
    return args


def literal_keyword(call: ast.Call, name: str, default: Any = None) -> Any:
    for keyword in call.keywords:
        if keyword.arg == name:
            try:
                return ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                if isinstance(keyword.value, ast.Name):
                    return keyword.value.id
                raise RuntimeError(f"unsupported parser keyword {name} in main.py")
    return default


def parser_schema() -> Tuple[Dict[str, Any], Dict[str, Tuple[str, str, Optional[str]]]]:
    """Read argparse defaults/options from main.py without importing ML dependencies."""

    source = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults: Dict[str, Any] = {}
    options: Dict[str, Tuple[str, str, Optional[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "parser":
            continue
        if node.func.attr == "set_defaults":
            for keyword in node.keywords:
                if keyword.arg is not None:
                    defaults[keyword.arg] = ast.literal_eval(keyword.value)
            continue
        if node.func.attr != "add_argument":
            continue
        flags = [
            ast.literal_eval(argument)
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("-")
        ]
        if not flags:
            continue
        dest = literal_keyword(node, "dest")
        if dest is None:
            dest = next(flag for flag in flags if flag.startswith("--")).lstrip("-").replace("-", "_")
        action = str(literal_keyword(node, "action", "store"))
        type_name = literal_keyword(node, "type")
        if action == "store_true":
            default = False
        elif action == "store_false":
            default = True
        else:
            default = literal_keyword(node, "default")
        defaults.setdefault(str(dest), default)
        for flag in flags:
            options[flag] = (str(dest), action, str(type_name) if type_name is not None else None)
    return defaults, options


def convert_value(raw: str, type_name: Optional[str]) -> Any:
    if type_name == "int":
        return int(raw)
    if type_name == "float":
        return float(raw)
    return raw


def derive_variant(config: Dict[str, Any]) -> str:
    method = config["method"]
    if method == "pall_modified":
        has_target = config["protect_ratio"] is not None or config["protect_threshold"] is not None
        protecting = has_target and float(config["lambda_protect"] or 0.0) > 0.0
        if not protecting:
            return "pall_modified_noprotect"
        label = {
            "gradient": "pall_modified_grad",
            "weight": "pall_modified_weight",
            "conflict": "pall_modified_conflict",
        }.get(str(config["protect_importance"]), "pall_modified_grad")
        if config["adaptive_protect"]:
            label += "_adapt"
        return label
    if method == "pall_adapter":
        if int(config["adapter_shared_bottleneck"] or 0) <= 0 or float(
            config["adapter_shared_forget_ratio"] or 0.0
        ) <= 0.0:
            return "adapter_reset"
        if float(config["adapter_shared_protect_ratio"] or 0.0) <= 0.0:
            return "adapter_shared"
        label = "adapter_protected"
        if config["protect_importance"] == "conflict":
            label += "_conflict"
        return label
    if method == "ssd":
        return f"ssd_a{float(config['ssd_alpha']):g}_l{float(config['ssd_lambda']):g}"
    if method == "salun":
        return f"salun_{config['salun_target']}_m{float(config['salun_mask_ratio']):g}"
    if method == "lora":
        return f"lora_r{config['lora_rank']}"
    return str(method)


def load_expected_config(main_args: list[str]) -> Dict[str, Any]:
    defaults, options = parser_schema()
    expected = dict(defaults)
    index = 0
    while index < len(main_args):
        option = main_args[index]
        if option not in options:
            raise ValueError(f"unknown main.py option in resume check: {option}")
        dest, action, type_name = options[option]
        if action == "store_true":
            expected[dest] = True
            index += 1
        elif action == "store_false":
            expected[dest] = False
            index += 1
        else:
            if index + 1 >= len(main_args):
                raise ValueError(f"missing value for main.py option: {option}")
            expected[dest] = convert_value(main_args[index + 1], type_name)
            index += 2

    method = str(expected["method"])
    if method == "pall":
        method = "pall_modified"
        expected["method"] = method
    expected["method_variant"] = {
        "pall_modified": "modified",
        "pall_original": "original",
        "pall_adapter": "adapter",
        "ssd": "ssd",
        "salun": "salun",
    }.get(method)
    expected["variant"] = derive_variant(expected)
    metadata = {
        "cifar10": (10, 32, 3),
        "cifar100": (100, 32, 3),
        "tinyimagenet": (200, 64, 3),
    }[str(expected["dataset"])]
    expected["num_classes"], expected["image_size"], expected["channels"] = metadata
    arch = str(expected["arch"]).lower()
    if method in {"pall", "pall_original", "pall_modified"}:
        arch = "subnet_" + arch
    elif method == "pall_adapter":
        arch = "adapter_" + arch
    elif method == "lora":
        arch = "lora_" + arch
    expected["arch"] = arch
    expected["dim_input"] = [metadata[2], metadata[1], metadata[1]]
    return expected


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            values_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def schedule_matches(config: Dict[str, Any], expected_path: Any) -> bool:
    if expected_path in (None, ""):
        return config.get("request_schedule_file") in (None, "")

    expected = Path(str(expected_path))
    if not expected.is_absolute():
        expected = REPO_ROOT / expected
    actual_path = config.get("request_schedule_file")
    if actual_path in (None, "") or Path(str(actual_path)).name != expected.name:
        return False

    expected_payload = load_json(expected)
    actual_payload = config.get("loaded_request_schedule")
    return expected_payload is not None and actual_payload == expected_payload


def config_matches(config: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key in RUNTIME_ONLY_KEYS:
            continue
        if key == "request_schedule_file":
            if not schedule_matches(config, expected_value):
                return False
            continue
        if key not in config or not values_equal(config.get(key), expected_value):
            return False
    return True


def is_completed(run_dir: Path, metrics: Dict[str, Any], n_forget: int) -> bool:
    if not (run_dir / "results.pth").is_file():
        return False
    events = metrics.get("unlearning_events")
    if not isinstance(events, list) or len(events) != n_forget:
        return False
    normalized = metrics.get("normalized_results")
    final = normalized.get("final") if isinstance(normalized, dict) else None
    return isinstance(final, dict) and final.get("final_avg_accuracy") is not None


def matching_runs(root: Path, expected: Dict[str, Any]) -> Iterable[Path]:
    if not root.is_dir():
        return []
    matches = []
    n_forget = int(expected["n_forget"])
    for config_path in root.rglob("config.json"):
        config = load_json(config_path)
        if config is None or not config_matches(config, expected):
            continue
        metrics = load_json(config_path.with_name("metrics.json"))
        if metrics is not None and is_completed(config_path.parent, metrics, n_forget):
            matches.append(config_path.parent)
    return matches


def main() -> int:
    args = parse_args()
    try:
        expected = load_expected_config(args.main_args)
        matches = sorted(matching_runs(args.root, expected), key=lambda path: (path.name, str(path)))
    except Exception as exc:
        print(f"[ERROR] exact-resume check failed: {exc}", file=sys.stderr)
        return 2
    if not matches:
        return 1
    print(matches[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
