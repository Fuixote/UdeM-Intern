#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import re
import statistics
from datetime import datetime
from pathlib import Path

from experiment_config import PROCESSED_DATA_DIR, RAW_DATA_DIR, resolve_path


WORKSPACE = Path(__file__).resolve().parent
BATCH_NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(?:__.+)?$")
RAW_FILE_PATTERN = re.compile(r"^genjson-(\d+)\.json$")
PROCESSED_FILE_PATTERN = re.compile(r"^G-(\d+)\.json$")
RAW_ARTIFACT_NAMES = (
    "config.json",
    "effective_config.json",
    "run_info.json",
    "batch_summary.json",
    "batch_report.md",
)
PROCESSED_METADATA_FILES = (
    "run_info.json",
    "batch_summary.json",
    "batch_report.md",
)
GROUND_TRUTH_NOISE_SIGMA = 0.15
GROUND_TRUTH_NOISE_MODE = "deterministic_per_edge"


def timestamp_now(now=None):
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")


def round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def parse_numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def compute_quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def summarize_numeric(values, digits=4):
    cleaned = sorted(float(v) for v in values if v is not None)
    if not cleaned:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "std": None,
        }
    std_dev = statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0
    return {
        "count": len(cleaned),
        "min": round_or_none(cleaned[0], digits),
        "max": round_or_none(cleaned[-1], digits),
        "mean": round_or_none(statistics.mean(cleaned), digits),
        "median": round_or_none(statistics.median(cleaned), digits),
        "p25": round_or_none(compute_quantile(cleaned, 0.25), digits),
        "p75": round_or_none(compute_quantile(cleaned, 0.75), digits),
        "std": round_or_none(std_dev, digits),
    }


def format_value(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_percent(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def is_batch_name(name):
    return bool(BATCH_NAME_PATTERN.match(name))


def collect_raw_files(raw_batch_dir):
    return sorted(
        (
            path
            for path in raw_batch_dir.glob("genjson-*.json")
            if path.is_file() and RAW_FILE_PATTERN.match(path.name)
        ),
        key=lambda path: int(RAW_FILE_PATTERN.match(path.name).group(1)),
    )


def processed_output_sort_key(output_name):
    match = PROCESSED_FILE_PATTERN.match(output_name)
    if match:
        return int(match.group(1))
    return output_name


def collect_processed_files(processed_batch_dir):
    if not processed_batch_dir.exists() or not processed_batch_dir.is_dir():
        return {}
    return {
        path.name: path
        for path in sorted(processed_batch_dir.glob("G-*.json"))
        if path.is_file() and PROCESSED_FILE_PATTERN.match(path.name)
    }


def expected_output_name(raw_file):
    match = RAW_FILE_PATTERN.match(raw_file.name)
    if not match:
        raise ValueError(f"Unsupported raw filename: {raw_file.name}")
    return f"G-{match.group(1)}.json"


def build_processed_batch_name(raw_batch_dir, started_at):
    if is_batch_name(raw_batch_dir.name):
        return raw_batch_dir.name
    return f"{timestamp_now(started_at)}__processed"


def resolve_processed_batch_dir(raw_batch_dir, output_dir_arg, started_at):
    if output_dir_arg:
        requested_path = resolve_path(output_dir_arg)
        if is_batch_name(requested_path.name):
            return requested_path
        return requested_path / build_processed_batch_name(raw_batch_dir, started_at)
    return resolve_path(PROCESSED_DATA_DIR) / build_processed_batch_name(raw_batch_dir, started_at)


def load_json_if_exists(path):
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text_if_exists(path):
    if not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def get_survival_time(age):
    if age is None or age == "Unknown":
        return None
    val = 25 - 0.3 * (float(age) - 20)
    return round(max(5.0, val), 2)


def get_qaly(utility, survival_time):
    if survival_time is None:
        return None
    u_norm = float(utility) / 91.0
    multiplier = math.sqrt(1.0 + u_norm ** 2) / math.sqrt(2.0)
    return round(multiplier * survival_time, 2)


def get_success_prob(utility, cpra, alpha=0.7, beta=0.3):
    if utility is None or cpra is None or cpra == "Unknown":
        return None
    score_part = float(utility) / 91.0
    cpra_part = 1.0 - float(cpra)
    return round(alpha * score_part + beta * cpra_part, 4)


def get_deterministic_epsilon(source_key, sigma=GROUND_TRUTH_NOISE_SIGMA):
    digest = hashlib.sha256(source_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = random.Random(seed)
    return rng.gauss(0, sigma)


def get_ground_truth(success_prob, qaly, source_key, sigma=GROUND_TRUTH_NOISE_SIGMA):
    if success_prob is None or qaly is None:
        return None
    epsilon = get_deterministic_epsilon(source_key, sigma)
    w_true = success_prob * qaly * (1 + epsilon)
    return round(max(0.0, w_true), 4)


def build_processed_payload(input_file):
    with input_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_data = data.get("data", {})
    recipients_data = data.get("recipients", {})
    vertices = {}

    for node_id, attributes in raw_data.items():
        is_altruistic = attributes.get("altruistic", False)

        if is_altruistic:
            vertices[str(node_id)] = {
                "type": "NDD",
                "id": str(node_id),
                "donor": {
                    "dage": attributes.get("dage", "Unknown"),
                    "bloodtype": attributes.get("bloodtype", "Unknown"),
                },
                "matches": [],
            }
            for match in attributes.get("matches", []):
                d_age = match.get("donor_age")
                survival_time = get_survival_time(d_age)
                utility = match.get("utility", 0)
                cpra = match.get("recipient_cpra")
                source_key = "|".join(
                    [
                        input_file.name,
                        str(node_id),
                        str(match.get("recipient")),
                        str(utility),
                    ]
                )
                vertices[str(node_id)]["matches"].append(
                    {
                        "donor_node_id": str(node_id),
                        "recipient": str(match["recipient"]),
                        "utility": utility,
                        "graft_survival_time": survival_time,
                        "qaly": get_qaly(utility, survival_time),
                        "success_prob": get_success_prob(utility, cpra),
                        "ground_truth_label": get_ground_truth(
                            get_success_prob(utility, cpra),
                            get_qaly(utility, survival_time),
                            source_key,
                        ),
                        "donor_age": d_age,
                        "donor_bt": match.get("donor_bt"),
                        "recipient_age": match.get("recipient_age"),
                        "recipient_cpra": match.get("recipient_cpra"),
                        "recipient_bt": match.get("recipient_bt"),
                    }
                )
        else:
            sources = attributes.get("sources", [])
            patient_id = str(sources[0]) if sources else str(node_id)

            if patient_id not in vertices:
                patient_info = recipients_data.get(patient_id, {})
                vertices[patient_id] = {
                    "type": "Pair",
                    "id": patient_id,
                    "patient": {
                        "age": patient_info.get("age", "Unknown"),
                        "bloodtype": patient_info.get("bloodtype", "Unknown"),
                        "cPRA": patient_info.get("cPRA", "Unknown"),
                        "hasBloodCompatibleDonor": patient_info.get("hasBloodCompatibleDonor", False),
                    },
                    "donors": [],
                    "matches": [],
                }

            vertices[patient_id]["donors"].append(
                {
                    "original_node_id": str(node_id),
                    "dage": attributes.get("dage", "Unknown"),
                    "bloodtype": attributes.get("bloodtype", "Unknown"),
                }
            )

            for match in attributes.get("matches", []):
                d_age = match.get("donor_age")
                survival_time = get_survival_time(d_age)
                utility = match.get("utility", 0)
                cpra = match.get("recipient_cpra")
                source_key = "|".join(
                    [
                        input_file.name,
                        str(node_id),
                        str(match.get("recipient")),
                        str(utility),
                    ]
                )
                vertices[patient_id]["matches"].append(
                    {
                        "donor_node_id": str(node_id),
                        "recipient": str(match["recipient"]),
                        "utility": utility,
                        "graft_survival_time": survival_time,
                        "qaly": get_qaly(utility, survival_time),
                        "success_prob": get_success_prob(utility, cpra),
                        "ground_truth_label": get_ground_truth(
                            get_success_prob(utility, cpra),
                            get_qaly(utility, survival_time),
                            source_key,
                        ),
                        "donor_age": d_age,
                        "donor_bt": match.get("donor_bt"),
                        "recipient_age": match.get("recipient_age"),
                        "recipient_cpra": match.get("recipient_cpra"),
                        "recipient_bt": match.get("recipient_bt"),
                    }
                )

    final_vertices = {}
    for vertex_id, vertex_data in vertices.items():
        best_matches = {}
        for match in vertex_data.get("matches", []):
            target = match["recipient"]
            utility = match["utility"]
            if target in vertices and (
                target not in best_matches or utility > best_matches[target]["utility"]
            ):
                best_matches[target] = {
                    "recipient": target,
                    "utility": utility,
                    "graft_survival_time": match.get("graft_survival_time"),
                    "qaly": match.get("qaly"),
                    "success_prob": match.get("success_prob"),
                    "ground_truth_label": match.get("ground_truth_label"),
                    "donor_age": match.get("donor_age"),
                    "donor_bt": match.get("donor_bt"),
                    "recipient_age": match.get("recipient_age"),
                    "recipient_cpra": match.get("recipient_cpra"),
                    "recipient_bt": match.get("recipient_bt"),
                }
                if "donor_node_id" in match:
                    best_matches[target]["winning_donor_id"] = match["donor_node_id"]

        vertex_data["matches"] = list(best_matches.values())
        final_vertices[vertex_id] = vertex_data

    return {
        "metadata": {
            "original_file": input_file.name,
            "total_vertices": len(final_vertices),
            "structure": "Unified Pair/NDD Graph",
            "ground_truth_noise_sigma": GROUND_TRUTH_NOISE_SIGMA,
            "ground_truth_noise_mode": GROUND_TRUTH_NOISE_MODE,
        },
        "data": final_vertices,
    }


def write_processed_file(input_file, output_file):
    print(f"Processing {input_file} -> {output_file}")
    payload = build_processed_payload(input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(
        f"Successfully wrote {payload['metadata']['total_vertices']} vertices to {output_file}"
    )


def collect_processed_payload_metrics(payload, file_name):
    data = payload.get("data", {})
    vertex_count = len(data)
    pair_count = 0
    ndd_count = 0
    edge_count = 0
    donors_per_pair = []
    outgoing_matches_per_vertex = []
    utility_values = []
    qaly_values = []
    success_prob_values = []
    ground_truth_values = []

    for vertex in data.values():
        vertex_type = vertex.get("type", "Unknown")
        matches = vertex.get("matches", []) or []
        outgoing_matches_per_vertex.append(len(matches))
        edge_count += len(matches)

        if vertex_type == "Pair":
            pair_count += 1
            donors_per_pair.append(len(vertex.get("donors", []) or []))
        elif vertex_type == "NDD":
            ndd_count += 1

        for match in matches:
            utility_values.append(parse_numeric(match.get("utility")))
            qaly_values.append(parse_numeric(match.get("qaly")))
            success_prob_values.append(parse_numeric(match.get("success_prob")))
            ground_truth_values.append(parse_numeric(match.get("ground_truth_label")))

    return {
        "file": file_name,
        "vertex_count": vertex_count,
        "pair_count": pair_count,
        "ndd_count": ndd_count,
        "edge_count": edge_count,
        "avg_outgoing_matches_per_vertex": round_or_none(edge_count / vertex_count, 4)
        if vertex_count
        else None,
        "avg_donors_per_pair": round_or_none(sum(donors_per_pair) / len(donors_per_pair), 4)
        if donors_per_pair
        else None,
        "donors_per_pair_values": donors_per_pair,
        "outgoing_matches_per_vertex_values": outgoing_matches_per_vertex,
        "utility_values": utility_values,
        "qaly_values": qaly_values,
        "success_prob_values": success_prob_values,
        "ground_truth_values": ground_truth_values,
    }


def load_processed_metrics(processed_file):
    with processed_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return collect_processed_payload_metrics(payload, processed_file.name)


def collect_raw_artifacts(raw_batch_dir):
    artifacts = {}
    for artifact_name in RAW_ARTIFACT_NAMES:
        path = raw_batch_dir / artifact_name
        artifacts[artifact_name] = str(path) if path.exists() else None
    return artifacts


def assess_batch(raw_batch_dir, processed_batch_dir):
    raw_files = collect_raw_files(raw_batch_dir)
    expected_outputs = {expected_output_name(path): path for path in raw_files}
    existing_outputs = collect_processed_files(processed_batch_dir)
    missing_outputs = [
        output_name
        for output_name in expected_outputs
        if output_name not in existing_outputs
    ]
    missing_metadata = [
        metadata_name
        for metadata_name in PROCESSED_METADATA_FILES
        if not (processed_batch_dir / metadata_name).exists()
    ]
    extra_outputs = sorted(
        name for name in existing_outputs if name not in expected_outputs
    )

    if not processed_batch_dir.exists():
        status = "unprocessed"
    elif missing_outputs or missing_metadata:
        status = "partial"
    else:
        status = "complete"

    return {
        "status": status,
        "raw_files": raw_files,
        "expected_outputs": expected_outputs,
        "missing_outputs": missing_outputs,
        "missing_metadata": missing_metadata,
        "existing_outputs": existing_outputs,
        "extra_outputs": extra_outputs,
    }


def build_run_info(
    args,
    raw_batch_dir,
    processed_batch_dir,
    started_at,
    finished_at,
    raw_files,
    expected_output_names,
    selected_raw_files,
    processed_output_names,
    newly_processed_output_names,
    skipped_existing_output_names,
    failed_outputs,
    processing_mode,
):
    raw_artifacts = collect_raw_artifacts(raw_batch_dir)
    return {
        "generated_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "processor_script": "1-data-processing.py",
        "workspace": str(WORKSPACE),
        "raw_batch_name": raw_batch_dir.name,
        "raw_input_dir": str(raw_batch_dir),
        "processed_batch_name": processed_batch_dir.name,
        "output_dir": str(processed_batch_dir),
        "processing_mode": processing_mode,
        "cli_args": vars(args),
        "selection": {
            "requested_file": args.file,
            "selected_raw_files": [path.name for path in selected_raw_files],
        },
        "counts": {
            "raw_file_count": len(raw_files),
            "expected_output_count": len(expected_output_names),
            "selected_raw_file_count": len(selected_raw_files),
            "processed_file_count": len(processed_output_names),
            "newly_processed_count": len(newly_processed_output_names),
            "skipped_existing_count": len(skipped_existing_output_names),
            "failed_count": len(failed_outputs),
        },
        "processing_config": {
            "ground_truth_noise_sigma": GROUND_TRUTH_NOISE_SIGMA,
            "ground_truth_noise_mode": GROUND_TRUTH_NOISE_MODE,
        },
        "raw_artifacts": raw_artifacts,
        "artifacts": {
            "run_info": str(processed_batch_dir / "run_info.json"),
            "batch_summary": str(processed_batch_dir / "batch_summary.json"),
            "batch_report": str(processed_batch_dir / "batch_report.md"),
        },
    }


def build_batch_summary(
    args,
    raw_batch_dir,
    processed_batch_dir,
    started_at,
    finished_at,
    raw_files,
    expected_output_names,
    selected_raw_files,
    processed_output_names,
    newly_processed_output_names,
    skipped_existing_output_names,
    failed_outputs,
    processing_mode,
):
    warnings = []
    raw_artifact_paths = collect_raw_artifacts(raw_batch_dir)
    raw_artifact_presence = {}
    for artifact_name, artifact_path in raw_artifact_paths.items():
        exists = artifact_path is not None
        raw_artifact_presence[artifact_name] = {
            "exists": exists,
            "path": artifact_path,
        }
        if not exists:
            warnings.append(f"Missing raw artifact: {artifact_name}")

    raw_config = load_json_if_exists(raw_batch_dir / "config.json")
    raw_effective_config = load_json_if_exists(raw_batch_dir / "effective_config.json")
    raw_run_info = load_json_if_exists(raw_batch_dir / "run_info.json")

    per_file_metrics = []
    vertices_per_file = []
    pairs_per_file = []
    ndds_per_file = []
    edges_per_file = []
    donors_per_pair = []
    outgoing_matches_per_vertex = []
    utility_values = []
    qaly_values = []
    success_prob_values = []
    ground_truth_values = []

    for output_name in processed_output_names:
        processed_file = processed_batch_dir / output_name
        try:
            metrics = load_processed_metrics(processed_file)
        except Exception as exc:
            warnings.append(f"Failed to read processed file {output_name}: {exc}")
            continue

        if metrics["edge_count"] == 0:
            warnings.append(f"Processed file {output_name} contains zero edges.")

        per_file_metrics.append(
            {
                "file": metrics["file"],
                "vertex_count": metrics["vertex_count"],
                "pair_count": metrics["pair_count"],
                "ndd_count": metrics["ndd_count"],
                "edge_count": metrics["edge_count"],
                "avg_outgoing_matches_per_vertex": metrics["avg_outgoing_matches_per_vertex"],
                "avg_donors_per_pair": metrics["avg_donors_per_pair"],
            }
        )
        vertices_per_file.append(metrics["vertex_count"])
        pairs_per_file.append(metrics["pair_count"])
        ndds_per_file.append(metrics["ndd_count"])
        edges_per_file.append(metrics["edge_count"])
        donors_per_pair.extend(metrics["donors_per_pair_values"])
        outgoing_matches_per_vertex.extend(metrics["outgoing_matches_per_vertex_values"])
        utility_values.extend(metrics["utility_values"])
        qaly_values.extend(metrics["qaly_values"])
        success_prob_values.extend(metrics["success_prob_values"])
        ground_truth_values.extend(metrics["ground_truth_values"])

    if failed_outputs:
        warnings.append(
            f"{len(failed_outputs)} outputs failed during processing: "
            + ", ".join(sorted(failed_outputs))
        )
    missing_processed_outputs = [
        output_name for output_name in expected_output_names if output_name not in processed_output_names
    ]
    if missing_processed_outputs:
        warnings.append(
            f"Missing processed outputs: {', '.join(sorted(missing_processed_outputs, key=processed_output_sort_key))}"
        )

    duration_seconds = (finished_at - started_at).total_seconds()
    return {
        "batch": {
            "raw_batch_name": raw_batch_dir.name,
            "processed_batch_name": processed_batch_dir.name,
            "raw_input_dir": str(raw_batch_dir),
            "output_dir": str(processed_batch_dir),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round_or_none(duration_seconds, 3),
            "processed_file_count": len(processed_output_names),
            "processed_files": processed_output_names,
        },
        "source_raw_batch": {
            "raw_file_count": len(raw_files),
            "raw_files": [path.name for path in raw_files],
            "selected_raw_files": [path.name for path in selected_raw_files],
            "expected_output_count": len(expected_output_names),
            "artifact_presence": raw_artifact_presence,
        },
        "processing_status": {
            "mode": processing_mode,
            "newly_processed_files": newly_processed_output_names,
            "newly_processed_count": len(newly_processed_output_names),
            "skipped_existing_files": skipped_existing_output_names,
            "skipped_existing_count": len(skipped_existing_output_names),
            "failed_outputs": sorted(failed_outputs),
            "failed_count": len(failed_outputs),
        },
        "parameters": {
            "cli_args": vars(args),
            "processing_config": {
                "ground_truth_noise_sigma": GROUND_TRUTH_NOISE_SIGMA,
                "ground_truth_noise_mode": GROUND_TRUTH_NOISE_MODE,
            },
            "raw_config": raw_config,
            "raw_effective_config": raw_effective_config,
            "raw_run_info_summary": {
                "generator_script": raw_run_info.get("generator_script"),
                "seed": raw_run_info.get("seed"),
                "generated_at": raw_run_info.get("generated_at"),
                "finished_at": raw_run_info.get("finished_at"),
            }
            if raw_run_info
            else None,
        },
        "aggregate": {
            "file_level": {
                "vertices_per_file": summarize_numeric(vertices_per_file, 4),
                "pairs_per_file": summarize_numeric(pairs_per_file, 4),
                "ndds_per_file": summarize_numeric(ndds_per_file, 4),
                "edges_per_file": summarize_numeric(edges_per_file, 4),
            },
            "population_level": {
                "donors_per_pair": summarize_numeric(donors_per_pair, 4),
                "outgoing_matches_per_vertex": summarize_numeric(outgoing_matches_per_vertex, 4),
                "utility": summarize_numeric(utility_values, 4),
                "qaly": summarize_numeric(qaly_values, 4),
                "success_prob": summarize_numeric(success_prob_values, 4),
                "ground_truth_label": summarize_numeric(ground_truth_values, 4),
            },
        },
        "per_file_metrics": per_file_metrics,
        "warnings": warnings,
    }


def render_batch_report(summary):
    batch = summary["batch"]
    processing_status = summary["processing_status"]
    source_raw_batch = summary["source_raw_batch"]
    params = summary["parameters"]
    aggregate = summary["aggregate"]

    lines = [
        f"# Processed Data Batch Report: `{batch['processed_batch_name']}`",
        "",
        "## Batch Overview",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Raw batch | `{batch['raw_batch_name']}` |",
        f"| Processed batch | `{batch['processed_batch_name']}` |",
        f"| Raw input directory | `{batch['raw_input_dir']}` |",
        f"| Output directory | `{batch['output_dir']}` |",
        f"| Started at | `{batch['started_at']}` |",
        f"| Finished at | `{batch['finished_at']}` |",
        f"| Duration | `{format_value(batch['duration_seconds'], 3)} s` |",
        f"| Processing mode | `{processing_status['mode']}` |",
        f"| Raw files in source batch | `{source_raw_batch['raw_file_count']}` |",
        f"| Processed files present | `{batch['processed_file_count']}` |",
        f"| Newly processed files | `{processing_status['newly_processed_count']}` |",
        f"| Skipped existing files | `{processing_status['skipped_existing_count']}` |",
        f"| Failed outputs | `{processing_status['failed_count']}` |",
        "",
    ]

    if summary["warnings"]:
        lines.extend(["## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Output Artifacts",
            "",
            "- `G-*.json`: processed pair / NDD graph instances",
            "- `run_info.json`: processing metadata, CLI parameters, and source raw artifact paths",
            "- `batch_summary.json`: machine-readable processed batch statistics",
            "- `batch_report.md`: this report",
            "",
            "## Source Raw Batch Snapshot",
            "",
            "| Artifact | Exists | Path |",
            "| --- | --- | --- |",
        ]
    )

    for artifact_name, artifact_payload in source_raw_batch["artifact_presence"].items():
        lines.append(
            f"| `{artifact_name}` | `{artifact_payload['exists']}` | `{artifact_payload['path'] or 'n/a'}` |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Statistics",
            "",
            f"- Vertices per file: mean `{format_value(aggregate['file_level']['vertices_per_file']['mean'])}`, min `{format_value(aggregate['file_level']['vertices_per_file']['min'])}`, max `{format_value(aggregate['file_level']['vertices_per_file']['max'])}`",
            f"- Pair vertices per file: mean `{format_value(aggregate['file_level']['pairs_per_file']['mean'])}`, min `{format_value(aggregate['file_level']['pairs_per_file']['min'])}`, max `{format_value(aggregate['file_level']['pairs_per_file']['max'])}`",
            f"- NDD vertices per file: mean `{format_value(aggregate['file_level']['ndds_per_file']['mean'])}`, min `{format_value(aggregate['file_level']['ndds_per_file']['min'])}`, max `{format_value(aggregate['file_level']['ndds_per_file']['max'])}`",
            f"- Edges per file: mean `{format_value(aggregate['file_level']['edges_per_file']['mean'])}`, min `{format_value(aggregate['file_level']['edges_per_file']['min'])}`, max `{format_value(aggregate['file_level']['edges_per_file']['max'])}`",
            f"- Donors per pair: mean `{format_value(aggregate['population_level']['donors_per_pair']['mean'])}`, median `{format_value(aggregate['population_level']['donors_per_pair']['median'])}`",
            f"- Outgoing matches per vertex: mean `{format_value(aggregate['population_level']['outgoing_matches_per_vertex']['mean'])}`, median `{format_value(aggregate['population_level']['outgoing_matches_per_vertex']['median'])}`",
            f"- Utility: mean `{format_value(aggregate['population_level']['utility']['mean'])}`, p25 `{format_value(aggregate['population_level']['utility']['p25'])}`, median `{format_value(aggregate['population_level']['utility']['median'])}`, p75 `{format_value(aggregate['population_level']['utility']['p75'])}`",
            f"- QALY: mean `{format_value(aggregate['population_level']['qaly']['mean'])}`, p25 `{format_value(aggregate['population_level']['qaly']['p25'])}`, median `{format_value(aggregate['population_level']['qaly']['median'])}`, p75 `{format_value(aggregate['population_level']['qaly']['p75'])}`",
            f"- Success probability: mean `{format_value(aggregate['population_level']['success_prob']['mean'])}`, p25 `{format_value(aggregate['population_level']['success_prob']['p25'])}`, median `{format_value(aggregate['population_level']['success_prob']['median'])}`, p75 `{format_value(aggregate['population_level']['success_prob']['p75'])}`",
            f"- Ground-truth label: mean `{format_value(aggregate['population_level']['ground_truth_label']['mean'])}`, p25 `{format_value(aggregate['population_level']['ground_truth_label']['p25'])}`, median `{format_value(aggregate['population_level']['ground_truth_label']['median'])}`, p75 `{format_value(aggregate['population_level']['ground_truth_label']['p75'])}`",
            "",
            "## Processing Outcome",
            "",
            f"- Mode: `{processing_status['mode']}`",
            f"- Newly processed count: `{processing_status['newly_processed_count']}`",
            f"- Skipped existing count: `{processing_status['skipped_existing_count']}`",
            f"- Failed count: `{processing_status['failed_count']}`",
            "",
        ]
    )

    if processing_status["newly_processed_files"]:
        lines.extend(
            [
                "### Newly Processed Files",
                "",
                ", ".join(f"`{name}`" for name in processing_status["newly_processed_files"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Parameter Snapshot",
            "",
            "### CLI Arguments",
            "",
            "```json",
            json.dumps(params["cli_args"], indent=2, ensure_ascii=False),
            "```",
            "",
            "### Processing Config",
            "",
            "```json",
            json.dumps(params["processing_config"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )

    if params["raw_config"] is not None:
        lines.extend(
            [
                "### Source Raw Config",
                "",
                "```json",
                json.dumps(params["raw_config"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )

    if params["raw_effective_config"] is not None:
        lines.extend(
            [
                "### Source Raw Effective Config",
                "",
                "```json",
                json.dumps(params["raw_effective_config"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## File-Level Consistency",
            "",
            "| File | Vertices | Pairs | NDDs | Edges | Avg Outgoing Matches / Vertex | Avg Donors / Pair |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in summary["per_file_metrics"]:
        lines.append(
            f"| `{metric['file']}` | {metric['vertex_count']} | {metric['pair_count']} | "
            f"{metric['ndd_count']} | {metric['edge_count']} | "
            f"{format_value(metric['avg_outgoing_matches_per_vertex'])} | "
            f"{format_value(metric['avg_donors_per_pair'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_batch_artifacts(
    args,
    raw_batch_dir,
    processed_batch_dir,
    started_at,
    finished_at,
    raw_files,
    expected_output_names,
    selected_raw_files,
    processed_output_names,
    newly_processed_output_names,
    skipped_existing_output_names,
    failed_outputs,
    processing_mode,
):
    run_info = build_run_info(
        args=args,
        raw_batch_dir=raw_batch_dir,
        processed_batch_dir=processed_batch_dir,
        started_at=started_at,
        finished_at=finished_at,
        raw_files=raw_files,
        expected_output_names=expected_output_names,
        selected_raw_files=selected_raw_files,
        processed_output_names=processed_output_names,
        newly_processed_output_names=newly_processed_output_names,
        skipped_existing_output_names=skipped_existing_output_names,
        failed_outputs=failed_outputs,
        processing_mode=processing_mode,
    )
    summary = build_batch_summary(
        args=args,
        raw_batch_dir=raw_batch_dir,
        processed_batch_dir=processed_batch_dir,
        started_at=started_at,
        finished_at=finished_at,
        raw_files=raw_files,
        expected_output_names=expected_output_names,
        selected_raw_files=selected_raw_files,
        processed_output_names=processed_output_names,
        newly_processed_output_names=newly_processed_output_names,
        skipped_existing_output_names=skipped_existing_output_names,
        failed_outputs=failed_outputs,
        processing_mode=processing_mode,
    )
    report = render_batch_report(summary)

    run_info_path = processed_batch_dir / "run_info.json"
    batch_summary_path = processed_batch_dir / "batch_summary.json"
    batch_report_path = processed_batch_dir / "batch_report.md"
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    batch_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    batch_report_path.write_text(report, encoding="utf-8")

    print(f"Run metadata saved to {run_info_path}")
    print(f"Batch summary saved to {batch_summary_path}")
    print(f"Batch report saved to {batch_report_path}")


def process_explicit_batch(args, raw_batch_dir):
    started_at = datetime.now()
    processed_batch_dir = resolve_processed_batch_dir(raw_batch_dir, args.output_dir, started_at)
    assessment = assess_batch(raw_batch_dir, processed_batch_dir)
    raw_files = assessment["raw_files"]

    if not raw_files:
        print(f"No files matching genjson-*.json found in {raw_batch_dir}")
        return 1

    if args.file:
        raw_input_path = raw_batch_dir / args.file
        if not raw_input_path.exists():
            print(f"File not found: {raw_input_path}")
            return 1
        if not RAW_FILE_PATTERN.match(args.file):
            print(f"Could not parse number from filename: {args.file}.")
            return 1
        selected_raw_files = [raw_input_path]
        processing_mode = "single_file"
    elif args.all:
        selected_raw_files = raw_files
        if args.force:
            processing_mode = "full_batch"
        elif assessment["status"] == "complete":
            processing_mode = "full_batch"
        elif processed_batch_dir.exists():
            processing_mode = "partial_repair"
        else:
            processing_mode = "full_batch"
    else:
        print("Please specify a target: use --all to process all files, or --file <filename> for a single file.")
        print("Example: python 1-data-processing.py dataset/raw/<batch_name> --file genjson-1523.json")
        return 1

    processed_batch_dir.mkdir(parents=True, exist_ok=True)

    newly_processed_output_names = []
    skipped_existing_output_names = []
    failed_outputs = set()

    for raw_file in selected_raw_files:
        output_name = expected_output_name(raw_file)
        output_path = processed_batch_dir / output_name
        should_process = args.force or not output_path.exists()

        if not should_process:
            skipped_existing_output_names.append(output_name)
            continue

        try:
            write_processed_file(raw_file, output_path)
            newly_processed_output_names.append(output_name)
        except Exception as exc:
            failed_outputs.add(output_name)
            print(f"Error processing {raw_file}: {exc}")

    processed_output_names = sorted(
        (
            output_name
            for output_name in assessment["expected_outputs"]
            if (processed_batch_dir / output_name).exists()
        ),
        key=processed_output_sort_key,
    )

    if not args.file and not args.force and assessment["status"] == "complete" and not newly_processed_output_names:
        print(f"Skipping complete processed batch: {processed_batch_dir}")
        return 0

    finished_at = datetime.now()
    write_batch_artifacts(
        args=args,
        raw_batch_dir=raw_batch_dir,
        processed_batch_dir=processed_batch_dir,
        started_at=started_at,
        finished_at=finished_at,
        raw_files=raw_files,
        expected_output_names=sorted(assessment["expected_outputs"], key=processed_output_sort_key),
        selected_raw_files=selected_raw_files,
        processed_output_names=processed_output_names,
        newly_processed_output_names=sorted(newly_processed_output_names),
        skipped_existing_output_names=sorted(skipped_existing_output_names),
        failed_outputs=failed_outputs,
        processing_mode=processing_mode,
    )
    return 1 if failed_outputs else 0


def find_raw_batches(raw_root):
    if not raw_root.exists():
        return []
    batches = []
    for child in sorted(raw_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or not is_batch_name(child.name):
            continue
        if collect_raw_files(child):
            batches.append(child)
    return batches


def process_scan_mode(args, raw_root):
    if args.file:
        print("Error: --file is only supported when an explicit raw batch directory is provided.")
        return 1
    if args.force:
        print("Error: --force is only supported when an explicit raw batch directory is provided.")
        return 1

    raw_batches = find_raw_batches(raw_root)
    if not raw_batches:
        print(f"No timestamped raw batches found under {raw_root}")
        return 0

    processed_root = resolve_path(args.output_dir) if args.output_dir else resolve_path(PROCESSED_DATA_DIR)
    processed_root.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    repaired_count = 0
    skipped_count = 0
    failure_count = 0

    for raw_batch_dir in raw_batches:
        started_at = datetime.now()
        processed_batch_dir = processed_root / raw_batch_dir.name
        assessment = assess_batch(raw_batch_dir, processed_batch_dir)
        raw_files = assessment["raw_files"]

        if assessment["status"] == "complete":
            skipped_count += 1
            print(f"Skipping already complete batch: {raw_batch_dir.name}")
            continue

        processing_mode = "full_batch" if assessment["status"] == "unprocessed" else "partial_repair"
        selected_raw_files = raw_files
        processed_batch_dir.mkdir(parents=True, exist_ok=True)

        if assessment["status"] == "unprocessed":
            raw_files_to_process = raw_files
            skipped_existing_output_names = []
            processed_count += 1
        else:
            raw_files_to_process = [
                assessment["expected_outputs"][output_name]
                for output_name in assessment["missing_outputs"]
            ]
            skipped_existing_output_names = sorted(
                output_name
                for output_name in assessment["expected_outputs"]
                if output_name not in assessment["missing_outputs"]
                and (processed_batch_dir / output_name).exists()
            )
            repaired_count += 1

        newly_processed_output_names = []
        failed_outputs = set()
        for raw_file in raw_files_to_process:
            output_name = expected_output_name(raw_file)
            output_path = processed_batch_dir / output_name
            try:
                write_processed_file(raw_file, output_path)
                newly_processed_output_names.append(output_name)
            except Exception as exc:
                failed_outputs.add(output_name)
                print(f"Error processing {raw_file}: {exc}")

        processed_output_names = sorted(
            (
                output_name
                for output_name in assessment["expected_outputs"]
                if (processed_batch_dir / output_name).exists()
            ),
            key=processed_output_sort_key,
        )
        finished_at = datetime.now()
        write_batch_artifacts(
            args=args,
            raw_batch_dir=raw_batch_dir,
            processed_batch_dir=processed_batch_dir,
            started_at=started_at,
            finished_at=finished_at,
            raw_files=raw_files,
            expected_output_names=sorted(assessment["expected_outputs"], key=processed_output_sort_key),
            selected_raw_files=selected_raw_files,
            processed_output_names=processed_output_names,
            newly_processed_output_names=sorted(newly_processed_output_names),
            skipped_existing_output_names=sorted(skipped_existing_output_names),
            failed_outputs=failed_outputs,
            processing_mode=processing_mode,
        )
        if failed_outputs:
            failure_count += 1

    print(
        "Scan summary: "
        f"new={processed_count}, repaired={repaired_count}, skipped={skipped_count}, failed={failure_count}"
    )
    return 1 if failure_count else 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert donor-based KEP JSON graphs into batch-organized Pair/NDD G-JSON graphs."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Explicit raw batch directory. If omitted, scan dataset/raw for unprocessed timestamped batches.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Processed root directory or explicit processed batch directory (default: dataset/processed).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all files in the explicit raw batch directory.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process a specific raw file (e.g., genjson-1494.json). Only valid with an explicit raw batch directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected outputs even if they already exist. Only valid with an explicit raw batch directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    explicit_input_dir = resolve_path(args.input_dir) if args.input_dir else None
    raw_root = resolve_path(RAW_DATA_DIR)

    if explicit_input_dir is None or explicit_input_dir == raw_root:
        return process_scan_mode(args, raw_root)

    if not explicit_input_dir.exists() or not explicit_input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {explicit_input_dir}")
        return 1

    return process_explicit_batch(args, explicit_input_dir)


if __name__ == "__main__":
    raise SystemExit(main())
