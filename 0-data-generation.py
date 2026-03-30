#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from experiment_config import RAW_DATA_DIR, resolve_path

# This script automates the usage of the modified kidney-webapp generator.
# It builds a headless Node.js wrapper, validates generation parameters,
# injects a deterministic RNG, and writes each run into its own output folder.

WORKSPACE = Path(__file__).resolve().parent
WEBAPP_DIR = WORKSPACE / "generator_webapp"

JS_FILES = [
    "js/kidney/blood-type.js",
    "js/kidney/donor-patient.js",
    "js/kidney/generated-dataset.js",
    "js/kidney/generator.js",
    "js/kidney/pra-band.js",
    "js/kidney/tuning.js",
    "js/kidney/compat-band.js",
]


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


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


def resolve_output_dir(args):
    if args.output_dir:
        return resolve_path(args.output_dir)
    output_root = resolve_path(args.output_root)
    run_name = args.run_name or f"gen_{timestamp_now()}"
    return output_root / run_name


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


def build_run_metadata(args, output_dir, config_dict):
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator_script": "0-data-generation.py",
        "workspace": str(WORKSPACE),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "cli_args": vars(args),
        "generator_config": config_dict,
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
                        help="Optional output subdirectory name; defaults to a timestamped folder")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional exact output directory. Overrides --output_root/--run_name")
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
    print("=== Step 0: CLI-Driven KEP Data Generation ===")

    try:
        validate_args(args)
        output_dir = prepare_output_dir(resolve_output_dir(args), force=args.force)
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
    run_metadata = build_run_metadata(args, output_dir, config_dict)

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

        try:
            print("\nBuilding Node.js generator script...")
            combined_js = build_node_script(temp_dir)
            temp_config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

            print("\nExecuting JSON generation via Node.js...")
            subprocess.run(
                ["node", str(combined_js), str(temp_config_path), str(output_dir)],
                check=True,
                text=True,
                cwd=str(WORKSPACE),
            )

            config_path = output_dir / "config.json"
            run_info_path = output_dir / "run_info.json"
            config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
            run_info_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
            print(f"Generator config saved to {config_path}")
            print(f"Run metadata saved to {run_info_path}")

        except subprocess.CalledProcessError as e:
            print(f"\nGeneration failed with error code: {e.returncode}")
            return e.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
