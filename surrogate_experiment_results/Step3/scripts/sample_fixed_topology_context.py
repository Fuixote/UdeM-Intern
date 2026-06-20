#!/usr/bin/env python3
"""Sample mutable context fields on a fixed Step3 topology."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fixed_topology_xy_common as common  # noqa: E402


REQUIRED_CONFIG_SECTIONS = ("recipient_cpra", "utility")


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the limited YAML shape used by Step3 config examples.

    This intentionally avoids a PyYAML dependency in the project test
    environment. It supports top-level scalars and one level of nested scalar
    mappings, which is enough for `context_generator.example.yaml`.
    """
    root: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        if ":" not in text:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line!r}")
        key, raw_value = text.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if indent == 0:
            if raw_value == "":
                root[key] = {}
                current_section = key
            else:
                root[key] = _parse_scalar(raw_value)
                current_section = None
        elif indent == 2 and current_section:
            root.setdefault(current_section, {})
            root[current_section][key] = _parse_scalar(raw_value)
        else:
            raise ValueError(f"Unsupported YAML indentation in {path}: {raw_line!r}")
    return root


def load_generator_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    return load_simple_yaml(config_path)


def validate_generator_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("generator_config must be a mapping")
    if not config.get("generator_version"):
        raise ValueError("generator_config requires generator_version")
    if config.get("method") != "bounded_additive_uniform":
        raise ValueError("Only method='bounded_additive_uniform' is currently implemented")
    if config.get("status") not in {"pilot_not_locked", "locked"}:
        raise ValueError("generator_config status must be pilot_not_locked or locked")
    for section in REQUIRED_CONFIG_SECTIONS:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"generator_config requires {section} settings")
        for key in ("lower", "upper", "half_width"):
            if key not in config[section]:
                raise ValueError(f"generator_config {section} requires {key}")
        lower = float(config[section]["lower"])
        upper = float(config[section]["upper"])
        half_width = float(config[section]["half_width"])
        if lower > upper:
            raise ValueError(f"{section} lower must be <= upper")
        if half_width < 0:
            raise ValueError(f"{section} half_width must be non-negative")


def bounded_additive_uniform(base_value: float, rng: np.random.Generator, spec: dict[str, Any]) -> float:
    lower = float(spec["lower"])
    upper = float(spec["upper"])
    half_width = float(spec["half_width"])
    sampled = float(base_value) + float(rng.uniform(-half_width, half_width))
    return float(np.clip(sampled, lower, upper))


def sample_context(
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    context_seed: int,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    validate_generator_config(generator_config)
    common.assert_structure_matches_template(base_payload, topology_template)

    out = copy.deepcopy(base_payload)
    rng = np.random.Generator(np.random.PCG64(int(context_seed)))
    cpra_by_recipient: dict[str, float] = {}

    for vertex_id, node in out.get("data", {}).items():
        if node.get("type") == "Pair":
            vertex_key = str(vertex_id)
            patient = node.setdefault("patient", {})
            base_cpra = float(patient.get("cPRA", 0.0))
            new_cpra = bounded_additive_uniform(
                base_cpra,
                rng,
                generator_config["recipient_cpra"],
            )
            patient["cPRA"] = new_cpra
            cpra_by_recipient[vertex_key] = new_cpra

    for source_id, node in out.get("data", {}).items():
        for match in node.get("matches", []) or []:
            target_id = str(match["recipient"])
            if target_id not in cpra_by_recipient:
                raise ValueError(f"Arc {source_id}->{target_id} does not target a Pair recipient")
            match["utility"] = bounded_additive_uniform(
                float(match.get("utility", 0.0)),
                rng,
                generator_config["utility"],
            )
            match["recipient_cpra"] = cpra_by_recipient[target_id]

    common.assert_structure_matches_template(out, topology_template)
    common.assert_context_label_consistency(out, topology_template)
    out.setdefault("metadata", {})
    out["metadata"]["step3_fixed_topology_context"] = {
        "context_seed": int(context_seed),
        "generator_version": str(generator_config["generator_version"]),
        "generator_config_hash": common.generator_config_hash(generator_config),
        "method": str(generator_config["method"]),
    }
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--base-payload", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = json.loads(args.template.read_text(encoding="utf-8"))
    base_payload = json.loads(args.base_payload.read_text(encoding="utf-8"))
    config = load_generator_config(args.config)
    sampled = sample_context(template, base_payload, args.context_seed, config)
    common.atomic_write_json(args.output, sampled)
    print(json.dumps(sampled.get("metadata", {}).get("step3_fixed_topology_context", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
