#!/usr/bin/env python3
"""Build topology-only candidate/vertex incidence graphs from the formal bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import gnn_data_common as common


def build_dataset(
    summary_rows: list[dict[str, str]],
    *,
    target_rows: list[dict[str, str]] | None = None,
    require_formal_targets: bool = False,
) -> tuple[list[dict], dict]:
    records = []
    failures = []
    target_by_id = None if target_rows is None else {row["topology_id"]: row for row in target_rows}
    topology_ids = set()
    topology_hashes = set()
    feasible_hashes = set()
    for row in summary_rows:
        template_path = common.resolve_project_path(row["template_path"])
        template = json.loads(template_path.read_text(encoding="utf-8"))
        formal_target = None if target_by_id is None else target_by_id.get(row["topology_id"])
        if require_formal_targets and (
            formal_target is None or not common.truthy(formal_target.get("formal_label_ready"))
        ):
            failures.append(f"{row['topology_id']}:formal_target_missing")
        record = common.build_graph_record(row, template, formal_target_row=formal_target)
        record_failures = common.validate_no_target_leakage(record)
        if record_failures:
            failures.append(f"{row['topology_id']}:{','.join(record_failures)}")
        records.append(record)
        topology_ids.add(record["topology_id"])
        topology_hashes.add(record["topology_hash"])
        feasible_hashes.add(record["feasible_set_hash"])
    if len(records) != 1000:
        failures.append(f"record_count_mismatch:{len(records)}!=1000")
    if len(topology_ids) != len(records):
        failures.append("topology_ids_not_unique")
    if len(topology_hashes) != len(records):
        failures.append("topology_hashes_not_unique")
    if len(feasible_hashes) != len(records):
        failures.append("feasible_set_hashes_not_unique")
    audit = {
        "passed": not failures,
        "record_count": len(records),
        "unique_topology_ids": len(topology_ids),
        "unique_topology_hashes": len(topology_hashes),
        "unique_feasible_set_hashes": len(feasible_hashes),
        "node_feature_names": common.NODE_FEATURE_NAMES,
        "relation_types": common.RELATION_TYPES,
        "target": "formal_label_mean_pp" if target_rows is not None else "normalized_improvement_pp",
        "target_status": (
            "formal_three_seed_mean"
            if target_rows is not None and not any(record["target"]["value"] is None for record in records)
            else "incomplete_or_seed42_provisional"
        ),
        "formal_target_count": sum(record["target"].get("formal") is True for record in records),
        "require_formal_targets": require_formal_targets,
        "target_is_input_feature": False,
        "failures": failures,
    }
    return records, audit


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summary", type=Path, default=common.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "data" / "topology_incidence_graphs.jsonl")
    parser.add_argument("--audit-output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "data" / "topology_incidence_graphs.audit.json")
    parser.add_argument("--target-table", type=Path, default=None)
    parser.add_argument("--require-formal-targets", action="store_true")
    args = parser.parse_args()
    target_rows = None if args.target_table is None else common.read_csv(args.target_table)
    records, audit = build_dataset(
        common.read_csv(args.formal_summary),
        target_rows=target_rows,
        require_formal_targets=args.require_formal_targets,
    )
    if audit["passed"]:
        write_jsonl(args.output, records)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "output": str(args.output), "audit_output": str(args.audit_output)}, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
