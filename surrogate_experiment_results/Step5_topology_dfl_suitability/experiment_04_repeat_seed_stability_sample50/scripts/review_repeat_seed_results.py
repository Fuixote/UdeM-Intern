#!/usr/bin/env python3
"""Review the 120 jobs and quantify target stability over seeds 42/43/44."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any

import repeat_seed_common as common


reviewer = common.base_reviewer()


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            output[ordered[position]] = rank
        start = end
    return output


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right))
    return None if denominator == 0 else numerator / denominator


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def sign_class(value: float, material_threshold: float = 0.1) -> str:
    if value > material_threshold:
        return "positive"
    if value < -material_threshold:
        return "negative"
    if value == 0:
        return "zero"
    return "small"


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def review_all(selected_rows: list[dict[str, str]], formal_rows: list[dict[str, str]], output_root: Path) -> dict[str, Any]:
    formal_by_id = {row["topology_id"]: row for row in formal_rows}
    long_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for selected_index, selected in enumerate(selected_rows):
        topology_id = selected["topology_id"]
        formal = formal_by_id[topology_id]
        long_rows.append({
            "topology_id": topology_id,
            "selection_category": selected["selection_category"],
            "train_seed": common.REFERENCE_SEED,
            "normalized_improvement_pp": float(formal["normalized_improvement_pp"]),
            "test_hash": formal["test_hash"],
            "source": "experiment_03_formal",
        })
        for seed in common.TRAIN_SEEDS:
            job_row, label_row, job_failures = reviewer.review_one_job(
                selected,
                manifest_index=selected_index * len(common.TRAIN_SEEDS) + common.TRAIN_SEEDS.index(seed),
                output_root=output_root,
                regime=common.DEFAULT_REGIME,
                protocol=common.DEFAULT_PROTOCOL,
                data_seed=seed,
                sample_size=common.SAMPLE_SIZE,
                test_size=common.TEST_SIZE,
                theta_seed=common.THETA_SEED,
                gurobi_seed=common.GUROBI_SEED,
                max_epochs=common.MAX_EPOCHS,
                metric_stride=common.METRIC_STRIDE,
                early_stop_patience=common.EARLY_STOP_PATIENCE,
                early_stop_min_delta=common.EARLY_STOP_MIN_DELTA,
                threshold=0.1,
                require_early_stop=True,
            )
            job_rows.append(job_row)
            if job_failures or label_row is None:
                failures.append(f"{topology_id}@{seed}:{','.join(job_failures) or 'label_missing'}")
                continue
            if str(label_row["test_hash"]) != str(formal["test_hash"]):
                failures.append(f"{topology_id}@{seed}:fixed_test_hash_mismatch")
            long_rows.append({
                "topology_id": topology_id,
                "selection_category": selected["selection_category"],
                "train_seed": seed,
                "normalized_improvement_pp": float(label_row["normalized_improvement_pp"]),
                "test_hash": label_row["test_hash"],
                "source": "experiment_04_repeat",
            })

    stability_rows: list[dict[str, Any]] = []
    complete_by_seed = {seed: {} for seed in common.ALL_LABEL_SEEDS}
    for row in long_rows:
        complete_by_seed[int(row["train_seed"])][row["topology_id"]] = float(row["normalized_improvement_pp"])
    for selected in selected_rows:
        topology_id = selected["topology_id"]
        if any(topology_id not in complete_by_seed[seed] for seed in common.ALL_LABEL_SEEDS):
            continue
        values = [complete_by_seed[seed][topology_id] for seed in common.ALL_LABEL_SEEDS]
        classes = [sign_class(value) for value in values]
        stability_rows.append({
            "topology_id": topology_id,
            "selection_category": selected["selection_category"],
            **{f"seed{seed}_normalized_improvement_pp": complete_by_seed[seed][topology_id] for seed in common.ALL_LABEL_SEEDS},
            "mean_normalized_improvement_pp": statistics.fmean(values),
            "std_normalized_improvement_pp": statistics.pstdev(values),
            "range_normalized_improvement_pp": max(values) - min(values),
            "all_exact_zero": all(value == 0 for value in values),
            "sign_class_consistent": len(set(classes)) == 1,
            "material_sign_consistent": not ({"positive", "negative"} <= set(classes)),
            "test_hash": selected["test_hash"],
        })

    pairwise = []
    common_ids = [row["topology_id"] for row in selected_rows if all(row["topology_id"] in complete_by_seed[seed] for seed in common.ALL_LABEL_SEEDS)]
    for left_index, left_seed in enumerate(common.ALL_LABEL_SEEDS):
        for right_seed in common.ALL_LABEL_SEEDS[left_index + 1 :]:
            left = [complete_by_seed[left_seed][topology_id] for topology_id in common_ids]
            right = [complete_by_seed[right_seed][topology_id] for topology_id in common_ids]
            top_count = min(10, len(common_ids))
            left_top = {topology_id for _, topology_id in sorted(zip(left, common_ids), reverse=True)[:top_count]}
            right_top = {topology_id for _, topology_id in sorted(zip(right, common_ids), reverse=True)[:top_count]}
            pairwise.append({
                "left_seed": left_seed,
                "right_seed": right_seed,
                "count": len(common_ids),
                "pearson": pearson(left, right),
                "spearman": spearman(left, right),
                "top10_overlap": len(left_top & right_top) / top_count if top_count else None,
            })
    correlations = [row["spearman"] for row in pairwise if row["spearman"] is not None]
    zero_selected = [row for row in stability_rows if row["selection_category"] in {"ceiling_zero", "nonzero_gap_plateau"}]
    material_selected = [row for row in stability_rows if row["selection_category"] in {"material_positive", "material_negative", "extreme_positive", "extreme_negative"}]
    stability_gate = {
        "minimum_pairwise_spearman": min(correlations) if correlations else None,
        "required_minimum_pairwise_spearman": 0.7,
        "zero_plateau_consistency": sum(row["all_exact_zero"] for row in zero_selected) / len(zero_selected) if zero_selected else None,
        "required_zero_plateau_consistency": 0.8,
        "material_sign_consistency": sum(row["material_sign_consistent"] for row in material_selected) / len(material_selected) if material_selected else None,
        "required_material_sign_consistency": 0.8,
    }
    stability_gate["passed"] = bool(
        stability_gate["minimum_pairwise_spearman"] is not None
        and stability_gate["minimum_pairwise_spearman"] >= stability_gate["required_minimum_pairwise_spearman"]
        and stability_gate["zero_plateau_consistency"] is not None
        and stability_gate["zero_plateau_consistency"] >= stability_gate["required_zero_plateau_consistency"]
        and stability_gate["material_sign_consistency"] is not None
        and stability_gate["material_sign_consistency"] >= stability_gate["required_material_sign_consistency"]
    )
    return {
        "integrity_passed": not failures and len(long_rows) == 180 and len(stability_rows) == 60,
        "stability_gate": stability_gate,
        "expected_new_job_count": 120,
        "successful_new_job_count": sum(not row.get("failure") for row in job_rows),
        "label_row_count": len(long_rows),
        "stability_topology_count": len(stability_rows),
        "pairwise_seed_metrics": pairwise,
        "failures": failures,
        "job_rows": job_rows,
        "long_rows": long_rows,
        "stability_rows": stability_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=common.DEFAULT_SELECTED_TOPOLOGIES)
    parser.add_argument("--formal-summary", type=Path, default=common.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = review_all(common.read_csv(args.topologies_csv), common.read_csv(args.formal_summary), args.output_root)
    output_dir = args.output_dir or args.output_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "repeat_seed_job_metrics.csv", result.pop("job_rows"))
    atomic_csv(output_dir / "repeat_seed_labels_long.csv", result.pop("long_rows"))
    atomic_csv(output_dir / "repeat_seed_stability_summary.csv", result.pop("stability_rows"))
    reviewer.atomic_write_json(output_dir / "repeat_seed_review_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
