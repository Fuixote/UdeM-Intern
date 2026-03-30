#!/usr/bin/env python3
import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from experiment_config import RAW_DATA_DIR, resolve_path

# This script automates the usage of the modified kidney-webapp generator.
# It builds a headless Node.js wrapper, validates generation parameters,
# injects a deterministic RNG, and writes each run into its own output folder.

WORKSPACE = Path(__file__).resolve().parent
WEBAPP_DIR = WORKSPACE / "generator_webapp"
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(?:__.+)?$")

JS_FILES = [
    "js/kidney/blood-type.js",
    "js/kidney/donor-patient.js",
    "js/kidney/generated-dataset.js",
    "js/kidney/generator.js",
    "js/kidney/pra-band.js",
    "js/kidney/tuning.js",
    "js/kidney/compat-band.js",
]


def timestamp_now(now=None):
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")


def validate_probability(name, value):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate_probability_sum(name, values, upper_bound=1.0, tol=1e-9):
    total = sum(values)
    if total > upper_bound + tol:
        raise ValueError(f"{name} must sum to at most {upper_bound}, got {total:.6f}")


def validate_args(args):
    if args.instances <= 0:
        raise ValueError(f"--instances must be > 0, got {args.instances}")
    if args.patients <= 0:
        raise ValueError(f"--patients must be > 0, got {args.patients}")

    probability_args = {
        "--prob_ndd": args.prob_ndd,
        "--prob_o": args.prob_o,
        "--prob_a": args.prob_a,
        "--prob_b": args.prob_b,
        "--donor_prob_o": args.donor_prob_o,
        "--donor_prob_a": args.donor_prob_a,
        "--donor_prob_b": args.donor_prob_b,
        "--donors1": args.donors1,
        "--donors2": args.donors2,
        "--donors3": args.donors3,
        "--prob_spousal": args.prob_spousal,
        "--prob_female": args.prob_female,
        "--prob_spousal_pra_compat": args.prob_spousal_pra_compat,
    }
    for name, value in probability_args.items():
        validate_probability(name, value)

    validate_probability_sum(
        "donor count probabilities",
        [args.donors1, args.donors2, args.donors3],
    )
    validate_probability_sum(
        "patient blood type probabilities",
        [args.prob_o, args.prob_a, args.prob_b],
    )
    validate_probability_sum(
        "donor blood type probabilities",
        [args.donor_prob_o, args.donor_prob_a, args.donor_prob_b],
    )


def sanitize_run_name(run_name):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name.strip())
    return sanitized.strip("._-")


def build_batch_dir_name(batch_timestamp, run_name=None):
    if run_name:
        sanitized = sanitize_run_name(run_name)
        if sanitized:
            return f"{batch_timestamp}__{sanitized}"
    return batch_timestamp


def resolve_output_dir(args, batch_timestamp):
    batch_dir_name = build_batch_dir_name(batch_timestamp, args.run_name)
    if args.output_dir:
        requested_path = resolve_path(args.output_dir)
        if TIMESTAMP_PATTERN.match(requested_path.name):
            return requested_path
        return requested_path / batch_dir_name
    output_root = resolve_path(args.output_root)
    return output_root / batch_dir_name


def prepare_output_dir(output_dir, force=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            if not force:
                raise ValueError(
                    f"Output directory already exists and is not empty: {output_dir}. "
                    "Use --force to overwrite it or choose a new --run_name/--output_dir."
                )
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


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


def make_distribution(counter, ordered_keys=None, digits=4):
    total = sum(counter.values())
    keys = ordered_keys or sorted(counter.keys())
    distribution = {}
    for key in keys:
        count = counter.get(key, 0)
        distribution[str(key)] = {
            "count": int(count),
            "share": round_or_none(count / total, digits) if total else None,
        }
    return {"total": int(total), "distribution": distribution}


def compare_distribution(target_map, actual_counter, ordered_keys, digits=4):
    total = sum(actual_counter.values())
    rows = {}
    max_abs_diff = 0.0
    for key in ordered_keys:
        target = target_map.get(key)
        actual = (actual_counter.get(key, 0) / total) if total else None
        abs_diff = abs(actual - target) if actual is not None and target is not None else None
        if abs_diff is not None:
            max_abs_diff = max(max_abs_diff, abs_diff)
        rows[str(key)] = {
            "target": round_or_none(target, digits),
            "actual": round_or_none(actual, digits),
            "count": int(actual_counter.get(key, 0)),
            "abs_diff": round_or_none(abs_diff, digits),
        }
    return {
        "total": int(total),
        "max_abs_diff": round_or_none(max_abs_diff, digits) if total else None,
        "rows": rows,
    }


def summarize_boolean_counts(true_count, false_count, digits=4):
    total = true_count + false_count
    return {
        "total": total,
        "true_count": true_count,
        "false_count": false_count,
        "true_share": round_or_none(true_count / total, digits) if total else None,
        "false_share": round_or_none(false_count / total, digits) if total else None,
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


def format_distribution_table(title, comparison, ordered_keys):
    lines = [
        f"### {title}",
        "",
        "| Category | Target | Actual | Count | Abs Diff |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ordered_keys:
        row = comparison["rows"][str(key)]
        lines.append(
            f"| `{key}` | {format_percent(row['target'])} | {format_percent(row['actual'])} | "
            f"{row['count']} | {format_percent(row['abs_diff'])} |"
        )
    lines.append("")
    lines.append(f"Max absolute deviation: {format_percent(comparison['max_abs_diff'])}")
    lines.append("")
    return "\n".join(lines)


def build_effective_config_snapshot(effective_config):
    if effective_config is None:
        return None
    snapshot = {"raw": effective_config}
    patient_bt = effective_config.get("patientBtDistribution")
    donor_bt = effective_config.get("donorBtDistribution")
    donor_counts = effective_config.get("donorCountProbabilities")
    if isinstance(patient_bt, dict):
        snapshot["patient_blood_distribution"] = patient_bt
    if isinstance(donor_bt, dict):
        snapshot["donor_blood_distribution"] = donor_bt
    if isinstance(donor_counts, list):
        snapshot["donor_count_probabilities"] = donor_counts
    if effective_config.get("tune"):
        snapshot["tune"] = effective_config.get("tune")
    for key in (
        "donorBtDistributionByPatientO",
        "donorBtDistributionByPatientA",
        "donorBtDistributionByPatientB",
        "donorBtDistributionByPatientAB",
        "donorBtDistributionByPatientNDD",
        "praBands",
        "compatPraBands",
        "incompatPraBands",
    ):
        if key in effective_config:
            snapshot[key] = effective_config[key]
    return snapshot


def summarize_generated_batch(output_dir, args, requested_config, effective_config, batch_timestamp, started_at, finished_at):
    raw_files = sorted(output_dir.glob("genjson-*.json"))
    recipient_bt_counter = Counter()
    donor_bt_counter = Counter()
    paired_donor_bt_counter = Counter()
    ndd_donor_bt_counter = Counter()
    donor_bt_by_source_counter = defaultdict(Counter)
    donor_count_counter = Counter()
    cpra_band_counter = Counter()
    recipient_has_compatible_true = 0
    recipient_has_compatible_false = 0
    recipient_ages = []
    recipient_cpra_values = []
    donor_ages = []
    utility_values = []
    recipients_per_file = []
    donor_nodes_per_file = []
    paired_donors_per_file = []
    ndd_donors_per_file = []
    matches_per_file = []
    outgoing_matches_per_donor = []
    incoming_matches_per_recipient = []
    donors_per_recipient = []
    per_file_metrics = []

    for raw_file in raw_files:
        with raw_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        donor_nodes = payload.get("data", {})
        recipients = payload.get("recipients", {})
        donors_for_recipient = Counter()
        recipient_incoming = Counter()
        paired_donor_count = 0
        ndd_donor_count = 0
        match_count = 0

        for donor_id, donor_data in donor_nodes.items():
            donor_bt = donor_data.get("bloodtype", "Unknown")
            donor_bt_counter[donor_bt] += 1
            donor_age = parse_numeric(donor_data.get("dage"))
            if donor_age is not None:
                donor_ages.append(donor_age)
            is_altruistic = bool(donor_data.get("altruistic", False))
            if is_altruistic:
                ndd_donor_count += 1
                ndd_donor_bt_counter[donor_bt] += 1
                donor_bt_by_source_counter["NDD"][donor_bt] += 1
            else:
                paired_donor_count += 1
                paired_donor_bt_counter[donor_bt] += 1
                for source_id in donor_data.get("sources", []):
                    source_key = str(source_id)
                    donors_for_recipient[source_key] += 1
                    recipient_info = recipients.get(source_key, {})
                    donor_bt_by_source_counter[recipient_info.get("bloodtype", "Unknown")][donor_bt] += 1

            matches = donor_data.get("matches", []) or []
            outgoing_matches_per_donor.append(len(matches))
            match_count += len(matches)
            for match in matches:
                utility = parse_numeric(match.get("utility"))
                if utility is not None:
                    utility_values.append(utility)
                recipient_id = match.get("recipient")
                if recipient_id is not None:
                    recipient_incoming[str(recipient_id)] += 1

        for recipient_id, recipient_data in recipients.items():
            recipient_bt = recipient_data.get("bloodtype", "Unknown")
            recipient_bt_counter[recipient_bt] += 1
            age = parse_numeric(recipient_data.get("age"))
            if age is not None:
                recipient_ages.append(age)
            cpra = parse_numeric(recipient_data.get("cPRA"))
            if cpra is not None:
                recipient_cpra_values.append(cpra)
                if cpra < 0.2:
                    cpra_band_counter["[0.00, 0.20)"] += 1
                elif cpra < 0.8:
                    cpra_band_counter["[0.20, 0.80)"] += 1
                else:
                    cpra_band_counter["[0.80, 1.00]"] += 1
            else:
                cpra_band_counter["Unknown"] += 1

            if recipient_data.get("hasBloodCompatibleDonor", False):
                recipient_has_compatible_true += 1
            else:
                recipient_has_compatible_false += 1

            donors_per_recipient.append(donors_for_recipient.get(str(recipient_id), 0))
            donor_count_counter[str(donors_for_recipient.get(str(recipient_id), 0))] += 1
            incoming_matches_per_recipient.append(recipient_incoming.get(str(recipient_id), 0))

        recipients_per_file.append(len(recipients))
        donor_nodes_per_file.append(len(donor_nodes))
        paired_donors_per_file.append(paired_donor_count)
        ndd_donors_per_file.append(ndd_donor_count)
        matches_per_file.append(match_count)
        per_file_metrics.append({
            "file": raw_file.name,
            "recipient_count": len(recipients),
            "donor_node_count": len(donor_nodes),
            "paired_donor_count": paired_donor_count,
            "ndd_donor_count": ndd_donor_count,
            "match_count": match_count,
            "avg_outgoing_matches_per_donor": round_or_none(match_count / len(donor_nodes), 4) if donor_nodes else None,
        })

    requested_recipient_bt = {
        "O": requested_config["patientBtDistribution"]["probO"],
        "A": requested_config["patientBtDistribution"]["probA"],
        "B": requested_config["patientBtDistribution"]["probB"],
        "AB": requested_config["patientBtDistribution"]["probAB"],
    }
    requested_donor_bt = {
        "O": requested_config["donorBtDistribution"]["probO"],
        "A": requested_config["donorBtDistribution"]["probA"],
        "B": requested_config["donorBtDistribution"]["probB"],
        "AB": requested_config["donorBtDistribution"]["probAB"],
    }
    requested_donor_counts = {
        "1": args.donors1,
        "2": args.donors2,
        "3": args.donors3,
        "4": round(1.0 - (args.donors1 + args.donors2 + args.donors3), 10),
    }

    target_vs_actual = {
        "recipient_bloodtype_distribution": compare_distribution(
            requested_recipient_bt,
            recipient_bt_counter,
            ["O", "A", "B", "AB"],
        ),
        "donor_count_distribution_per_recipient": compare_distribution(
            requested_donor_counts,
            donor_count_counter,
            ["1", "2", "3", "4"],
        ),
        "altruistic_donor_share": {
            "target": round_or_none(args.prob_ndd, 4),
            "actual": round_or_none((sum(ndd_donor_bt_counter.values()) / sum(donor_bt_counter.values())), 4) if donor_bt_counter else None,
            "count": int(sum(ndd_donor_bt_counter.values())),
            "total": int(sum(donor_bt_counter.values())),
            "abs_diff": round_or_none(
                abs((sum(ndd_donor_bt_counter.values()) / sum(donor_bt_counter.values())) - args.prob_ndd),
                4,
            ) if donor_bt_counter else None,
        },
    }
    if args.split_donor_blood:
        split_targets = {
            "O": requested_config.get("donorBtDistributionByPatientO", {}),
            "A": requested_config.get("donorBtDistributionByPatientA", {}),
            "B": requested_config.get("donorBtDistributionByPatientB", {}),
            "AB": requested_config.get("donorBtDistributionByPatientAB", {}),
            "NDD": requested_config.get("donorBtDistributionByPatientNDD", {}),
        }
        target_vs_actual["donor_bloodtype_distribution_by_source"] = {
            source: compare_distribution(split_targets[source], donor_bt_by_source_counter.get(source, Counter()), ["O", "A", "B", "AB"])
            for source in ("O", "A", "B", "AB", "NDD")
        }
    else:
        target_vs_actual["donor_bloodtype_distribution"] = compare_distribution(
            requested_donor_bt,
            donor_bt_counter,
            ["O", "A", "B", "AB"],
        )

    warnings = []
    if len(raw_files) != args.instances:
        warnings.append(
            f"Expected {args.instances} raw files, but found {len(raw_files)}."
        )
    inconsistent_patients = [metric["file"] for metric in per_file_metrics if metric["recipient_count"] != args.patients]
    if inconsistent_patients:
        warnings.append(
            f"{len(inconsistent_patients)} files do not contain the requested {args.patients} recipients."
        )
    zero_match_files = [metric["file"] for metric in per_file_metrics if metric["match_count"] == 0]
    if zero_match_files:
        warnings.append(f"{len(zero_match_files)} files contain zero matches.")

    recipient_bt_deviation = target_vs_actual["recipient_bloodtype_distribution"]["max_abs_diff"]
    donor_bt_deviation = None
    if "donor_bloodtype_distribution" in target_vs_actual:
        donor_bt_deviation = target_vs_actual["donor_bloodtype_distribution"]["max_abs_diff"]
    elif "donor_bloodtype_distribution_by_source" in target_vs_actual:
        donor_bt_deviation = max(
            (
                comparison["max_abs_diff"]
                for comparison in target_vs_actual["donor_bloodtype_distribution_by_source"].values()
                if comparison["max_abs_diff"] is not None
            ),
            default=None,
        )
    donor_count_deviation = target_vs_actual["donor_count_distribution_per_recipient"]["max_abs_diff"]
    altruistic_deviation = target_vs_actual["altruistic_donor_share"]["abs_diff"]
    if recipient_bt_deviation is not None and recipient_bt_deviation > 0.05:
        warnings.append(
            f"Recipient blood type distribution deviates by up to {format_percent(recipient_bt_deviation)} from the requested target."
        )
    if donor_bt_deviation is not None and donor_bt_deviation > 0.05:
        warnings.append(
            f"Donor blood type distribution deviates by up to {format_percent(donor_bt_deviation)} from the requested target."
        )
    if donor_count_deviation is not None and donor_count_deviation > 0.05:
        warnings.append(
            f"Donor count distribution deviates by up to {format_percent(donor_count_deviation)} from the requested target."
        )
    if altruistic_deviation is not None and altruistic_deviation > 0.05:
        warnings.append(
            f"Altruistic donor share deviates by {format_percent(altruistic_deviation)} from the requested target."
        )

    effective_config_snapshot = build_effective_config_snapshot(effective_config)
    duration_seconds = (finished_at - started_at).total_seconds()

    return {
        "batch": {
            "batch_timestamp": batch_timestamp,
            "batch_name": output_dir.name,
            "output_dir": str(output_dir),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round_or_none(duration_seconds, 3),
            "generated_file_count": len(raw_files),
            "generated_files": [path.name for path in raw_files],
        },
        "parameters": {
            "cli_args": vars(args),
            "requested_generator_config": requested_config,
            "effective_generator_config": effective_config_snapshot,
        },
        "aggregate": {
            "file_level": {
                "recipients_per_file": summarize_numeric(recipients_per_file, 4),
                "donor_nodes_per_file": summarize_numeric(donor_nodes_per_file, 4),
                "paired_donors_per_file": summarize_numeric(paired_donors_per_file, 4),
                "ndd_donors_per_file": summarize_numeric(ndd_donors_per_file, 4),
                "matches_per_file": summarize_numeric(matches_per_file, 4),
            },
            "population_level": {
                "donors_per_recipient": summarize_numeric(donors_per_recipient, 4),
                "outgoing_matches_per_donor": summarize_numeric(outgoing_matches_per_donor, 4),
                "incoming_matches_per_recipient": summarize_numeric(incoming_matches_per_recipient, 4),
                "recipient_age": summarize_numeric(recipient_ages, 4),
                "donor_age": summarize_numeric(donor_ages, 4),
                "recipient_cpra": summarize_numeric(recipient_cpra_values, 4),
                "utility": summarize_numeric(utility_values, 4),
            },
            "recipient_bloodtypes": make_distribution(recipient_bt_counter, ["O", "A", "B", "AB", "Unknown"]),
            "donor_bloodtypes_all": make_distribution(donor_bt_counter, ["O", "A", "B", "AB", "Unknown"]),
            "donor_bloodtypes_paired": make_distribution(paired_donor_bt_counter, ["O", "A", "B", "AB", "Unknown"]),
            "donor_bloodtypes_ndd": make_distribution(ndd_donor_bt_counter, ["O", "A", "B", "AB", "Unknown"]),
            "recipient_cpra_bands": make_distribution(cpra_band_counter, ["[0.00, 0.20)", "[0.20, 0.80)", "[0.80, 1.00]", "Unknown"]),
            "recipient_has_blood_compatible_donor": summarize_boolean_counts(
                recipient_has_compatible_true,
                recipient_has_compatible_false,
            ),
        },
        "target_vs_actual": target_vs_actual,
        "per_file_metrics": per_file_metrics,
        "warnings": warnings,
    }


def render_batch_report(summary):
    batch = summary["batch"]
    params = summary["parameters"]
    aggregate = summary["aggregate"]
    target_vs_actual = summary["target_vs_actual"]
    lines = [
        f"# Raw Data Batch Report: `{batch['batch_name']}`",
        "",
        "## Batch Overview",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Output directory | `{batch['output_dir']}` |",
        f"| Batch timestamp | `{batch['batch_timestamp']}` |",
        f"| Started at | `{batch['started_at']}` |",
        f"| Finished at | `{batch['finished_at']}` |",
        f"| Duration | `{format_value(batch['duration_seconds'], 3)} s` |",
        f"| Raw files generated | `{batch['generated_file_count']}` |",
        f"| Seed | `{params['cli_args']['seed']}` |",
        f"| Tuning enabled | `{not params['cli_args']['no_tune']}` |",
        f"| Split donor blood | `{params['cli_args']['split_donor_blood']}` |",
        "",
    ]

    if summary["warnings"]:
        lines.extend([
            "## Warnings",
            "",
        ])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend([
        "## Output Artifacts",
        "",
        "- `genjson-*.json`: raw donor-based graph instances",
        "- `config.json`: requested generator configuration",
        "- `effective_config.json`: final effective configuration actually used for generation after tuning",
        "- `run_info.json`: batch metadata, CLI parameters, and artifact paths",
        "- `batch_summary.json`: machine-readable aggregate statistics",
        "- `batch_report.md`: this report",
        "",
        "## Aggregate Statistics",
        "",
        f"- Recipients per file: mean `{format_value(aggregate['file_level']['recipients_per_file']['mean'])}`, min `{format_value(aggregate['file_level']['recipients_per_file']['min'])}`, max `{format_value(aggregate['file_level']['recipients_per_file']['max'])}`",
        f"- Donor nodes per file: mean `{format_value(aggregate['file_level']['donor_nodes_per_file']['mean'])}`, min `{format_value(aggregate['file_level']['donor_nodes_per_file']['min'])}`, max `{format_value(aggregate['file_level']['donor_nodes_per_file']['max'])}`",
        f"- NDD donors per file: mean `{format_value(aggregate['file_level']['ndd_donors_per_file']['mean'])}`, min `{format_value(aggregate['file_level']['ndd_donors_per_file']['min'])}`, max `{format_value(aggregate['file_level']['ndd_donors_per_file']['max'])}`",
        f"- Matches per file: mean `{format_value(aggregate['file_level']['matches_per_file']['mean'])}`, min `{format_value(aggregate['file_level']['matches_per_file']['min'])}`, max `{format_value(aggregate['file_level']['matches_per_file']['max'])}`",
        f"- Donors per recipient: mean `{format_value(aggregate['population_level']['donors_per_recipient']['mean'])}`, median `{format_value(aggregate['population_level']['donors_per_recipient']['median'])}`",
        f"- Outgoing matches per donor: mean `{format_value(aggregate['population_level']['outgoing_matches_per_donor']['mean'])}`, median `{format_value(aggregate['population_level']['outgoing_matches_per_donor']['median'])}`",
        f"- Incoming matches per recipient: mean `{format_value(aggregate['population_level']['incoming_matches_per_recipient']['mean'])}`, median `{format_value(aggregate['population_level']['incoming_matches_per_recipient']['median'])}`",
        f"- Recipient age: mean `{format_value(aggregate['population_level']['recipient_age']['mean'])}`, range `[{format_value(aggregate['population_level']['recipient_age']['min'])}, {format_value(aggregate['population_level']['recipient_age']['max'])}]`",
        f"- Donor age: mean `{format_value(aggregate['population_level']['donor_age']['mean'])}`, range `[{format_value(aggregate['population_level']['donor_age']['min'])}, {format_value(aggregate['population_level']['donor_age']['max'])}]`",
        f"- Recipient cPRA: mean `{format_value(aggregate['population_level']['recipient_cpra']['mean'])}`, p25 `{format_value(aggregate['population_level']['recipient_cpra']['p25'])}`, median `{format_value(aggregate['population_level']['recipient_cpra']['median'])}`, p75 `{format_value(aggregate['population_level']['recipient_cpra']['p75'])}`",
        f"- Edge utility: mean `{format_value(aggregate['population_level']['utility']['mean'])}`, p25 `{format_value(aggregate['population_level']['utility']['p25'])}`, median `{format_value(aggregate['population_level']['utility']['median'])}`, p75 `{format_value(aggregate['population_level']['utility']['p75'])}`",
        "",
        "## Recipient Compatibility",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Has blood-compatible donor | {aggregate['recipient_has_blood_compatible_donor']['true_count']} ({format_percent(aggregate['recipient_has_blood_compatible_donor']['true_share'])}) |",
        f"| No blood-compatible donor | {aggregate['recipient_has_blood_compatible_donor']['false_count']} ({format_percent(aggregate['recipient_has_blood_compatible_donor']['false_share'])}) |",
        "",
    ])

    lines.append(format_distribution_table(
        "Recipient Blood Type Distribution",
        target_vs_actual["recipient_bloodtype_distribution"],
        ["O", "A", "B", "AB"],
    ))

    if "donor_bloodtype_distribution" in target_vs_actual:
        lines.append(format_distribution_table(
            "Donor Blood Type Distribution",
            target_vs_actual["donor_bloodtype_distribution"],
            ["O", "A", "B", "AB"],
        ))
    else:
        lines.extend(["## Donor Blood Type Distribution By Source", ""])
        for source in ("O", "A", "B", "AB", "NDD"):
            lines.append(format_distribution_table(
                f"Donor Blood Type Distribution Given Source `{source}`",
                target_vs_actual["donor_bloodtype_distribution_by_source"][source],
                ["O", "A", "B", "AB"],
            ))

    lines.append(format_distribution_table(
        "Donor Count Distribution Per Recipient",
        target_vs_actual["donor_count_distribution_per_recipient"],
        ["1", "2", "3", "4"],
    ))

    altruistic = target_vs_actual["altruistic_donor_share"]
    lines.extend([
        "## Altruistic Donor Share",
        "",
        "| Target | Actual | Count | Total Donor Nodes | Abs Diff |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {format_percent(altruistic['target'])} | {format_percent(altruistic['actual'])} | {altruistic['count']} | {altruistic['total']} | {format_percent(altruistic['abs_diff'])} |",
        "",
        "## cPRA Bands",
        "",
        "| Band | Count | Share |",
        "| --- | ---: | ---: |",
    ])
    for band, payload in aggregate["recipient_cpra_bands"]["distribution"].items():
        lines.append(f"| `{band}` | {payload['count']} | {format_percent(payload['share'])} |")

    lines.extend([
        "",
        "## Parameter Snapshot",
        "",
        "### CLI Arguments",
        "",
        "```json",
        json.dumps(params["cli_args"], indent=2, ensure_ascii=False),
        "```",
        "",
        "### Requested Generator Config",
        "",
        "```json",
        json.dumps(params["requested_generator_config"], indent=2, ensure_ascii=False),
        "```",
        "",
    ])

    if params["effective_generator_config"] is not None:
        lines.extend([
            "### Effective Generator Config Used For Generation",
            "",
            "```json",
            json.dumps(params["effective_generator_config"], indent=2, ensure_ascii=False),
            "```",
            "",
        ])

    lines.extend([
        "## File-Level Consistency",
        "",
        "| File | Recipients | Donor Nodes | Paired Donors | NDD Donors | Matches | Avg Outgoing Matches / Donor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for metric in summary["per_file_metrics"]:
        lines.append(
            f"| `{metric['file']}` | {metric['recipient_count']} | {metric['donor_node_count']} | "
            f"{metric['paired_donor_count']} | {metric['ndd_donor_count']} | {metric['match_count']} | "
            f"{format_value(metric['avg_outgoing_matches_per_donor'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_node_script(temp_dir):
    """Concatenate the JS generator files into a single runnable Node.js script."""
    combined_js_path = Path(temp_dir) / "combined-kidney.js"

    with combined_js_path.open("w", encoding="utf-8") as outfile:
        for js_file in JS_FILES:
            file_path = WEBAPP_DIR / js_file
            if not file_path.exists():
                raise FileNotFoundError(f"Missing required generator file: {file_path}")
            outfile.write(file_path.read_text(encoding="utf-8"))
            outfile.write("\n")

        runner_logic = """
'use strict';
const fs = require('fs');
const path = require('path');
var args = process.argv.slice(2);
var config = JSON.parse(fs.readFileSync(args[0], 'utf8'));
var outputDir = args[1];
var effectiveConfigPath = args[2];

function mulberry32(a) {
  return function() {
    var t = a += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const seed = (Number.isFinite(config.seed) ? config.seed : 0) >>> 0;
const seededRandom = mulberry32(seed);
Math.random = seededRandom;

// Enforce full details for machine learning features
config.fullDetails = true;
config.outputFormat = "json";
config.outputName = path.join(outputDir, "genjson");
config.testing = false;

// Helper to draw a random age (consistent with drawDage: 18-68)
const drawAge = () => 18 + Math.floor(Math.random() * 51);

// Monkey-patch toJsonString to include recipient age and utility naming
const originalToJsonString = GeneratedDataset.prototype.toJsonString;
GeneratedDataset.prototype.toJsonString = function(fullDetails) {
  var serializedObj = JSON.parse(originalToJsonString.call(this, fullDetails));

  if (fullDetails && this.recipients) {
    this.recipients.forEach(r => {
      if (serializedObj.recipients && serializedObj.recipients["" + r.id]) {
        serializedObj.recipients["" + r.id].age = r.age;
      }
    });
  }

  if (fullDetails && serializedObj.data) {
    for (const donorId in serializedObj.data) {
      const donorObj = serializedObj.data[donorId];
      if (donorObj.matches) {
        donorObj.matches.forEach(m => {
          if (m.hasOwnProperty('score')) {
            m.utility = m.score;
            delete m.score;
          }
          const recip = this.recipients.find(r => r.id == m.recipient);
          if (recip) m.recipient_age = recip.age;
        });
      }
    }
  }
  return JSON.stringify(serializedObj, undefined, 2);
};

config.patientBtDistribution = new BloodTypeDistribution(
  config.patientBtDistribution.probO,
  config.patientBtDistribution.probA,
  config.patientBtDistribution.probB,
  config.patientBtDistribution.probAB
);

if (config.donorBtDistributionByPatientO) {
  config.donorBtDistributionByPatientO = new BloodTypeDistribution(
    config.donorBtDistributionByPatientO.probO,
    config.donorBtDistributionByPatientO.probA,
    config.donorBtDistributionByPatientO.probB,
    config.donorBtDistributionByPatientO.probAB
  );
  config.donorBtDistributionByPatientA = new BloodTypeDistribution(
    config.donorBtDistributionByPatientA.probO,
    config.donorBtDistributionByPatientA.probA,
    config.donorBtDistributionByPatientA.probB,
    config.donorBtDistributionByPatientA.probAB
  );
  config.donorBtDistributionByPatientB = new BloodTypeDistribution(
    config.donorBtDistributionByPatientB.probO,
    config.donorBtDistributionByPatientB.probA,
    config.donorBtDistributionByPatientB.probB,
    config.donorBtDistributionByPatientB.probAB
  );
  config.donorBtDistributionByPatientAB = new BloodTypeDistribution(
    config.donorBtDistributionByPatientAB.probO,
    config.donorBtDistributionByPatientAB.probA,
    config.donorBtDistributionByPatientAB.probB,
    config.donorBtDistributionByPatientAB.probAB
  );
  config.donorBtDistributionByPatientNDD = new BloodTypeDistribution(
    config.donorBtDistributionByPatientNDD.probO,
    config.donorBtDistributionByPatientNDD.probA,
    config.donorBtDistributionByPatientNDD.probB,
    config.donorBtDistributionByPatientNDD.probAB
  );
} else {
  config.donorBtDistribution = new BloodTypeDistribution(
    config.donorBtDistribution.probO,
    config.donorBtDistribution.probA,
    config.donorBtDistribution.probB,
    config.donorBtDistribution.probAB
  );
}

if (config.tune) {
  var tuneIter = config.tune.iters || 100;
  var tuneError = config.tune.error || 0.05;
  var tuneSize = config.tune.size || 1000;
  config = TuneConfig(config, tuneIter, tuneError, tuneSize,
    true /* tuneBloodTypes */,
    true /* tuneDonors */,
    true /* tunePRA */);
}

fs.mkdirSync(outputDir, { recursive: true });
if (effectiveConfigPath) {
  fs.writeFileSync(effectiveConfigPath, JSON.stringify(config, null, 2));
}
var gen = new KidneyGenerator(config);

console.log(`Generating ${config.numberOfInstances} JSON instances with seed ${seed}...`);
for (var i = 0; i < config.numberOfInstances; i++) {
  var generatedDataset = gen.generateDataset(config.patientsPerInstance, config.proportionAltruistic);
  generatedDataset.recipients.forEach(r => {
    r.age = drawAge();
  });
  var filename = config.outputName + "-" + i + "." + config.outputFormat;
  fs.writeFileSync(filename, generatedDataset.toJsonString(true));
}
console.log(`Successfully generated ${config.numberOfInstances} files in ${outputDir}`);
"""
        outfile.write(runner_logic)

    return combined_js_path


def create_config(args):
    """Create the Node generator configuration based on CLI arguments."""
    config = {
        "seed": args.seed,
        "donorCountProbabilities": [
            args.donors1,
            args.donors2,
            args.donors3,
            round(1.0 - (args.donors1 + args.donors2 + args.donors3), 10),
        ],
        "donorBtDistribution": {
            "probO": args.donor_prob_o,
            "probA": args.donor_prob_a,
            "probB": args.donor_prob_b,
            "probAB": round(1.0 - (args.donor_prob_o + args.donor_prob_a + args.donor_prob_b), 10),
        },
        "patientBtDistribution": {
            "probO": args.prob_o,
            "probA": args.prob_a,
            "probB": args.prob_b,
            "probAB": round(1.0 - (args.prob_o + args.prob_a + args.prob_b), 10),
        },
        "probSpousal": args.prob_spousal,
        "probFemale": args.prob_female,
        "probSpousalPraCompatibility": args.prob_spousal_pra_compat,
        "numberOfInstances": args.instances,
        "patientsPerInstance": args.patients,
        "proportionAltruistic": args.prob_ndd,
        "fileFormat": "json",
        "praBandsString": "0.2 0.11\n0.8 0.89",
        "compatBandsString": "0 101 0 1",
    }

    if args.tune:
        config["tune"] = {
            "iters": 100,
            "size": 1000,
            "error": 0.05,
        }

    if args.split_donor_blood:
        config["donorBtDistributionByPatientO"] = {"probO": 0.3721, "probA": 0.4899, "probB": 0.1219, "probAB": 0.0161}
        config["donorBtDistributionByPatientA"] = {"probO": 0.2783, "probA": 0.6039, "probB": 0.0907, "probAB": 0.0271}
        config["donorBtDistributionByPatientB"] = {"probO": 0.2910, "probA": 0.2719, "probB": 0.3689, "probAB": 0.0682}
        config["donorBtDistributionByPatientAB"] = {"probO": 0.3166, "probA": 0.4271, "probB": 0.1910, "probAB": 0.0653}
        config["donorBtDistributionByPatientNDD"] = {"probO": 0.4930, "probA": 0.3990, "probB": 0.0939, "probAB": 0.0141}

    return config


def build_run_metadata(args, output_dir, config_dict, batch_timestamp, started_at):
    return {
        "generated_at": started_at.isoformat(timespec="seconds"),
        "generator_script": "0-data-generation.py",
        "workspace": str(WORKSPACE),
        "batch_timestamp": batch_timestamp,
        "batch_name": output_dir.name,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "cli_args": vars(args),
        "generator_config": config_dict,
        "artifacts": {},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Headless KEP Instance Generator")
    parser.add_argument("--instances", type=int, default=1000, help="Number of graph instances to generate")
    parser.add_argument("--patients", type=int, default=50, help="Number of patients (pairs) per instance")
    parser.add_argument("--prob_ndd", type=float, default=0.05, help="Proportion of Non-Directed Donors (Altruistic)")
    parser.add_argument("--prob_o", type=float, default=0.4, help="Probability of patient blood type O")
    parser.add_argument("--prob_a", type=float, default=0.4, help="Probability of patient blood type A")
    parser.add_argument("--prob_b", type=float, default=0.1, help="Probability of patient blood type B")

    parser.add_argument("--donor_prob_o", type=float, default=0.4, help="Probability of donor blood type O")
    parser.add_argument("--donor_prob_a", type=float, default=0.4, help="Probability of donor blood type A")
    parser.add_argument("--donor_prob_b", type=float, default=0.1, help="Probability of donor blood type B")

    parser.add_argument("--donors1", type=float, default=1.0, help="Proportion of patients with 1 donor")
    parser.add_argument("--donors2", type=float, default=0.0, help="Proportion of patients with 2 donors")
    parser.add_argument("--donors3", type=float, default=0.0, help="Proportion of patients with 3 donors")
    parser.add_argument("--prob_spousal", type=float, default=0.0, help="Probability of spousal donor")
    parser.add_argument("--prob_female", type=float, default=0.0, help="Probability of female donor")
    parser.add_argument("--prob_spousal_pra_compat", type=float, default=0.0, help="Probability of spousal PRA compatibility")

    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic dataset generation")
    parser.add_argument("--output_root", type=str, default=str(RAW_DATA_DIR),
                        help="Root directory under which a per-run output folder will be created")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional label appended after the mandatory timestamped batch folder name")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional parent directory for the timestamped batch folder; if the path already ends in a timestamped batch name, it is used directly")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the target output directory if it already exists and is non-empty")

    parser.add_argument("--no_tune", action="store_true", help="Disable tuning (enabled by default)")
    parser.add_argument(
        "--split_donor_blood",
        action="store_true",
        help="Use different donor blood group distributions based on recipient blood group",
    )
    args = parser.parse_args()
    args.tune = not args.no_tune
    return args


def main():
    args = parse_args()
    started_at = datetime.now()
    batch_timestamp = timestamp_now(started_at)
    print("=== Step 0: CLI-Driven KEP Data Generation ===")

    try:
        validate_args(args)
        output_dir = prepare_output_dir(resolve_output_dir(args, batch_timestamp), force=args.force)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if not WEBAPP_DIR.exists():
        print("Error: The generator_webapp directory is missing.")
        return 1

    if shutil.which("node") is None:
        print("Error: Node.js is not installed. Please install node.js to run the generator headless.")
        print("Run: conda install -c conda-forge nodejs")
        return 1

    config_dict = create_config(args)
    run_metadata = build_run_metadata(args, output_dir, config_dict, batch_timestamp, started_at)

    print(f"Output directory: {output_dir}")
    print(f"Generating {args.instances} graphs, {args.patients} patients/graph, NDD ratio: {args.prob_ndd}")
    print(f"Seed: {args.seed}")
    if args.tune:
        print(" -> Tuning Enabled")
    if args.split_donor_blood:
        print(" -> Split Donor Blood Distribution Enabled")

    with tempfile.TemporaryDirectory(prefix="kep_gen_") as temp_dir:
        temp_dir = Path(temp_dir)
        temp_config_path = temp_dir / "config.json"
        temp_effective_config_path = temp_dir / "effective_config.json"

        try:
            print("\nBuilding Node.js generator script...")
            combined_js = build_node_script(temp_dir)
            temp_config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

            print("\nExecuting JSON generation via Node.js...")
            subprocess.run(
                ["node", str(combined_js), str(temp_config_path), str(output_dir), str(temp_effective_config_path)],
                check=True,
                text=True,
                cwd=str(WORKSPACE),
            )

            config_path = output_dir / "config.json"
            effective_config_path = output_dir / "effective_config.json"
            run_info_path = output_dir / "run_info.json"
            batch_summary_path = output_dir / "batch_summary.json"
            batch_report_path = output_dir / "batch_report.md"
            config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
            effective_config = None
            if temp_effective_config_path.exists():
                effective_config = json.loads(temp_effective_config_path.read_text(encoding="utf-8"))
                effective_config_path.write_text(json.dumps(effective_config, indent=2), encoding="utf-8")

            finished_at = datetime.now()
            batch_summary = summarize_generated_batch(
                output_dir=output_dir,
                args=args,
                requested_config=config_dict,
                effective_config=effective_config,
                batch_timestamp=batch_timestamp,
                started_at=started_at,
                finished_at=finished_at,
            )
            batch_summary_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
            batch_report_path.write_text(render_batch_report(batch_summary), encoding="utf-8")

            run_metadata["finished_at"] = finished_at.isoformat(timespec="seconds")
            run_metadata["effective_generator_config"] = build_effective_config_snapshot(effective_config)
            run_metadata["summary_highlights"] = {
                "generated_file_count": batch_summary["batch"]["generated_file_count"],
                "warnings": batch_summary["warnings"],
                "recipient_bloodtype_max_abs_diff": batch_summary["target_vs_actual"]["recipient_bloodtype_distribution"]["max_abs_diff"],
                "donor_count_max_abs_diff": batch_summary["target_vs_actual"]["donor_count_distribution_per_recipient"]["max_abs_diff"],
                "altruistic_share_abs_diff": batch_summary["target_vs_actual"]["altruistic_donor_share"]["abs_diff"],
            }
            run_metadata["artifacts"] = {
                "config": str(config_path),
                "effective_config": str(effective_config_path) if effective_config is not None else None,
                "run_info": str(run_info_path),
                "batch_summary": str(batch_summary_path),
                "batch_report": str(batch_report_path),
            }
            run_info_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
            print(f"Generator config saved to {config_path}")
            if effective_config is not None:
                print(f"Effective generator config saved to {effective_config_path}")
            print(f"Run metadata saved to {run_info_path}")
            print(f"Batch summary saved to {batch_summary_path}")
            print(f"Batch report saved to {batch_report_path}")

        except subprocess.CalledProcessError as e:
            print(f"\nGeneration failed with error code: {e.returncode}")
            return e.returncode
        except Exception as e:
            print(f"\nPost-generation reporting failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
