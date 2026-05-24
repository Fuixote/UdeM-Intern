#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PROCESSED_FILE_PATTERN = "G-*.json"
SUMMARY_JSON_NAME = "label_diagnostics.json"
GRAPH_CSV_NAME = "label_graph_diagnostics.csv"


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


def round_or_none(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def numeric_summary(values):
    cleaned = sorted(v for v in (parse_numeric(value) for value in values) if v is not None)
    if not cleaned:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "fraction_zero": None,
        }
    return {
        "count": len(cleaned),
        "min": round_or_none(cleaned[0]),
        "max": round_or_none(cleaned[-1]),
        "mean": round_or_none(statistics.mean(cleaned)),
        "median": round_or_none(statistics.median(cleaned)),
        "std": round_or_none(statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0),
        "fraction_zero": sum(1 for value in cleaned if value == 0.0) / len(cleaned),
    }


def pearson_correlation(xs, ys):
    pairs = [
        (parse_numeric(x), parse_numeric(y))
        for x, y in zip(xs, ys)
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_values = [x for x, _ in pairs]
    y_values = [y for _, y in pairs]
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_denom == 0.0 or y_denom == 0.0:
        return None
    return round_or_none(numerator / (x_denom * y_denom))


def processed_sort_key(path):
    stem = path.stem
    try:
        return int(stem.split("-", 1)[1])
    except (IndexError, ValueError):
        return stem


def collect_processed_files(dataset_dir):
    return sorted(
        (path for path in Path(dataset_dir).glob(PROCESSED_FILE_PATTERN) if path.is_file()),
        key=processed_sort_key,
    )


def iter_matches(payload):
    for vertex in payload.get("data", {}).values():
        for match in vertex.get("matches", []) or []:
            yield match


def collect_graph_row(path, payload):
    matches = list(iter_matches(payload))
    labels = [match.get("ground_truth_label") for match in matches]
    latent = [match.get("latent_clean_linear_label") for match in matches]
    label_stats = numeric_summary(labels)
    latent_stats = numeric_summary(latent)
    label_mean = label_stats["mean"]
    latent_mean = latent_stats["mean"]
    return {
        "file": path.name,
        "label_mode": payload.get("metadata", {}).get("ground_truth_label_mode"),
        "vertex_count": len(payload.get("data", {})),
        "edge_count": len(matches),
        "ground_truth_label_mean": label_mean,
        "latent_clean_linear_label_mean": latent_mean,
        "label_to_clean_mean_ratio": round_or_none(label_mean / latent_mean)
        if label_mean is not None and latent_mean not in (None, 0.0)
        else None,
        "fraction_zero_label": label_stats["fraction_zero"],
    }


def summarize_processed_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    files = collect_processed_files(dataset_dir)
    field_values = {}
    graph_rows = []
    label_modes = set()
    vertex_count = 0
    edge_count = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        label_mode = metadata.get("ground_truth_label_mode")
        if label_mode:
            label_modes.add(label_mode)
        vertex_count += len(payload.get("data", {}))
        graph_row = collect_graph_row(path, payload)
        graph_rows.append(graph_row)

        for match in iter_matches(payload):
            edge_count += 1
            for key, value in match.items():
                parsed = parse_numeric(value)
                if parsed is not None:
                    field_values.setdefault(key, []).append(parsed)

    labels = {
        "ground_truth_label": numeric_summary(field_values.get("ground_truth_label", [])),
    }
    if "latent_clean_linear_label" in field_values:
        labels["latent_clean_linear_label"] = numeric_summary(field_values["latent_clean_linear_label"])

    correlations = {
        "latent_clean_linear_label_vs_ground_truth_label": pearson_correlation(
            field_values.get("latent_clean_linear_label", []),
            field_values.get("ground_truth_label", []),
        ),
        "step2b_polynomial_label_vs_ground_truth_label": pearson_correlation(
            field_values.get("step2b_polynomial_label", []),
            field_values.get("ground_truth_label", []),
        ),
        "step2c_polynomial_label_vs_ground_truth_label": pearson_correlation(
            field_values.get("step2c_polynomial_label", []),
            field_values.get("ground_truth_label", []),
        ),
    }

    summary = {
        "dataset": {
            "path": str(dataset_dir),
            "name": dataset_dir.name,
            "graph_count": len(files),
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "label_modes": sorted(label_modes),
        },
        "labels": labels,
        "correlations": correlations,
        "step2a": {
            "label_noise_value": numeric_summary(field_values.get("label_noise_value", [])),
            "step2a_noise_sigma": numeric_summary(field_values.get("step2a_noise_sigma", [])),
        },
        "step2b": {
            "step2b_polynomial_label": numeric_summary(field_values.get("step2b_polynomial_label", [])),
            "step2b_polynomial_score": numeric_summary(field_values.get("step2b_polynomial_score", [])),
        },
        "step2c": {
            "step2c_polynomial_label": numeric_summary(field_values.get("step2c_polynomial_label", [])),
            "step2c_multiplier": numeric_summary(field_values.get("step2c_multiplier", [])),
            "step2c_noisy_polynomial_label": numeric_summary(
                field_values.get("step2c_noisy_polynomial_label", [])
            ),
        },
    }
    return summary, graph_rows


def write_diagnostics(dataset_dir, summary, graph_rows, output_dir=None):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_JSON_NAME
    graph_csv_path = output_dir / GRAPH_CSV_NAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = [
        "file",
        "label_mode",
        "vertex_count",
        "edge_count",
        "ground_truth_label_mean",
        "latent_clean_linear_label_mean",
        "label_to_clean_mean_ratio",
        "fraction_zero_label",
    ]
    with graph_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in graph_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    return {
        "summary_json": str(summary_path),
        "graph_csv": str(graph_csv_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Step2 processed dataset label statistics.")
    parser.add_argument("processed_dataset_dir", help="Path to a processed dataset directory containing G-*.json.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory for label_diagnostics artifacts. Defaults to the processed dataset directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.processed_dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"Processed dataset directory does not exist: {dataset_dir}")
        return 1
    summary, graph_rows = summarize_processed_dataset(dataset_dir)
    artifacts = write_diagnostics(dataset_dir, summary, graph_rows, args.output_dir)
    print(json.dumps({"dataset": summary["dataset"], "artifacts": artifacts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
