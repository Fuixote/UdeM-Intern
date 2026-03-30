#!/usr/bin/env python3
import os
import json
import subprocess
import shutil
import argparse
import tempfile

# This script automates the usage of the modified kidney-webapp generator
# It takes hyperparameters directly from Python command line arguments,
# dynamically generates the config JSON, and runs the Node.js KEP builder headless.

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(WORKSPACE, "generator_webapp")
DATASET_DIR = os.path.join(WORKSPACE, "dataset", "raw")

def build_node_script():
    """Concatenates the JS files into a single runnable Node.js script"""
    js_files = [
        "js/kidney/blood-type.js",
        "js/kidney/donor-patient.js",
        "js/kidney/generated-dataset.js",
        "js/kidney/generator.js",
        "js/kidney/pra-band.js",
        "js/kidney/tuning.js",
        "js/kidney/compat-band.js"
    ]
    
    combined_js_path = os.path.join(WORKSPACE, "combined-kidney.js")
    
    with open(combined_js_path, "w", encoding="utf-8") as outfile:
        # 1. First append all the core kidney-webapp library logic
        for js_file in js_files:
            file_path = os.path.join(WEBAPP_DIR, js_file)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
            else:
                print(f"Error: Missing required generator file: {file_path}")
                return None
                
        # 2. Append the Node.js execution wrapper logic
        runner_logic = """
'use strict';
const fs = require('fs');
var args = process.argv.slice(2);
var config = JSON.parse(fs.readFileSync(args[0]));

// Enforce full details for machine learning features
config.fullDetails = "true";
config.outputFormat = "json";
config.outputName = args[1] + "/genjson";
config.testing = false;

// --- INJECT PATIENT AGE LOGIC ---
// Helper to draw a random age (consistent with drawDage: 18-68)
const drawAge = () => 18 + (Math.floor(Math.random() * 51));

// Monkey-patch toJsonString to include recipient age
const originalToJsonString = GeneratedDataset.prototype.toJsonString;
GeneratedDataset.prototype.toJsonString = function(fullDetails) {
  var serializedObj = JSON.parse(originalToJsonString.call(this, fullDetails));
  
  // 1. Add age to top-level recipients data
  if (fullDetails && this.recipients) {
    this.recipients.forEach(r => {
      if (serializedObj.recipients && serializedObj.recipients[""+r.id]) {
        serializedObj.recipients[""+r.id].age = r.age;
      }
    });
  }

  // 2. Add recipient_age to individual matches
  if (fullDetails && serializedObj.data) {
    for (const donorId in serializedObj.data) {
      const donorObj = serializedObj.data[donorId];
      if (donorObj.matches) {
        donorObj.matches.forEach(m => {
          // Rename score to utility
          if (m.hasOwnProperty('score')) {
            m.utility = m.score;
            delete m.score;
          }
          // Find the recipient object to get its age
          const recip = this.recipients.find(r => r.id == m.recipient);
          if (recip) m.recipient_age = recip.age;
        });
      }
    }
  }
  return JSON.stringify(serializedObj, undefined, 2);
};
// --------------------------------

config.patientBtDistribution = new BloodTypeDistribution(
    config.patientBtDistribution.probO,
    config.patientBtDistribution.probA,
    config.patientBtDistribution.probB,
    config.patientBtDistribution.probAB);

if (config.donorBtDistributionByPatientO) {
  config.donorBtDistributionByPatientO = new BloodTypeDistribution(
    config.donorBtDistributionByPatientO.probO,
    config.donorBtDistributionByPatientO.probA,
    config.donorBtDistributionByPatientO.probB,
    config.donorBtDistributionByPatientO.probAB)
  config.donorBtDistributionByPatientA = new BloodTypeDistribution(
    config.donorBtDistributionByPatientA.probO,
    config.donorBtDistributionByPatientA.probA,
    config.donorBtDistributionByPatientA.probB,
    config.donorBtDistributionByPatientA.probAB)
  config.donorBtDistributionByPatientB = new BloodTypeDistribution(
    config.donorBtDistributionByPatientB.probO,
    config.donorBtDistributionByPatientB.probA,
    config.donorBtDistributionByPatientB.probB,
    config.donorBtDistributionByPatientB.probAB)
  config.donorBtDistributionByPatientAB = new BloodTypeDistribution(
    config.donorBtDistributionByPatientAB.probO,
    config.donorBtDistributionByPatientAB.probA,
    config.donorBtDistributionByPatientAB.probB,
    config.donorBtDistributionByPatientAB.probAB)
  config.donorBtDistributionByPatientNDD = new BloodTypeDistribution(
    config.donorBtDistributionByPatientNDD.probO,
    config.donorBtDistributionByPatientNDD.probA,
    config.donorBtDistributionByPatientNDD.probB,
    config.donorBtDistributionByPatientNDD.probAB)
} else {
  config.donorBtDistribution = new BloodTypeDistribution(
    config.donorBtDistribution.probO,
    config.donorBtDistribution.probA,
    config.donorBtDistribution.probB,
    config.donorBtDistribution.probAB);
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

var gen = new KidneyGenerator(config);

console.log(`Generating ${config.numberOfInstances} JSON instances...`);
for (var i=0; i < config.numberOfInstances; i++) {
  var generatedDataset = gen.generateDataset(config.patientsPerInstance, config.proportionAltruistic);
  
  // Assign ages to all recipients in this dataset
  generatedDataset.recipients.forEach(r => {
    r.age = drawAge();
  });

  // Format specific to your new naming: dataset/raw/genjson-0.json
  var filename = config.outputName + "-" + i + "." + config.outputFormat;
  
  var nullfn = function() {};
  // Write with the injected medical features enabled (fullDetails=true)
  fs.writeFileSync(filename, generatedDataset.toJsonString(true));
}
console.log(`Successfully generated ${config.numberOfInstances} files in ${args[1]}`);
"""
        outfile.write(runner_logic)
        
    return combined_js_path

def create_config(args):
    """Creates a configuration dictionary based on CLI arguments and Saidman defaults."""
    
    # 1. Base Configuration (Saidman Defaults)
    config = {
        "donorCountProbabilities": [
            args.donors1,
            args.donors2,
            args.donors3,
            max(0, 1.0 - (args.donors1 + args.donors2 + args.donors3))
        ],
        "donorBtDistribution": {
            "probO": args.donor_prob_o,
            "probA": args.donor_prob_a,
            "probB": args.donor_prob_b,
            "probAB": round(max(0, 1.0 - (args.donor_prob_o + args.donor_prob_a + args.donor_prob_b)), 4)
        },
        "patientBtDistribution": {
            "probO": args.prob_o,
            "probA": args.prob_a,
            "probB": args.prob_b,
            "probAB": round(max(0, 1.0 - (args.prob_o + args.prob_a + args.prob_b)), 4)
        },
        "probSpousal": args.prob_spousal,
        "probFemale": args.prob_female,
        "probSpousalPraCompatibility": args.prob_spousal_pra_compat,
        "numberOfInstances": args.instances,
        "patientsPerInstance": args.patients,
        "proportionAltruistic": args.prob_ndd,
        "fileFormat": "json",
        
        # 2. Calculated panel reactive antibodies (cPRA)
        "praBandsString": "0.2 0.11\n0.8 0.89",
        
        # 3. Compatibility calculations
        "compatBandsString": "0 101 0 1"
    }
    
    # 4. Tuning (Optional, enabled by default in Saidman generator to ensure output matches input distributions)
    if args.tune:
        config["tune"] = {
            "iters": 100,
            "size": 1000,
            "error": 0.05
        }
    
    # 5. Advanced: Split Donor Blood Types based on Recipient Type
    if args.split_donor_blood:
        config["donorBtDistributionByPatientO"] = {"probO": 0.3721, "probA": 0.4899, "probB": 0.1219, "probAB": 0.0161}
        config["donorBtDistributionByPatientA"] = {"probO": 0.2783, "probA": 0.6039, "probB": 0.0907, "probAB": 0.0271}
        config["donorBtDistributionByPatientB"] = {"probO": 0.2910, "probA": 0.2719, "probB": 0.3689, "probAB": 0.0682}
        config["donorBtDistributionByPatientAB"] = {"probO": 0.3166, "probA": 0.4271, "probB": 0.1910, "probAB": 0.0653}
        config["donorBtDistributionByPatientNDD"] = {"probO": 0.493, "probA": 0.399, "probB": 0.0939, "probAB": 0.0141}
        
    return config

def main():
    parser = argparse.ArgumentParser(description="Headless KEP Instance Generator")
    parser.add_argument("--instances", type=int, default=1000, help="Number of graph instances to generate")
    parser.add_argument("--patients", type=int, default=50, help="Number of patients (pairs) per instance")
    parser.add_argument("--prob_ndd", type=float, default=0.05, help="Proportion of Non-Directed Donors (Altruistic)")
    parser.add_argument("--prob_o", type=float, default=0.4, help="Probability of Blood Type O")
    parser.add_argument("--prob_a", type=float, default=0.4, help="Probability of Patient Blood Type A")
    parser.add_argument("--prob_b", type=float, default=0.1, help="Probability of Patient Blood Type B")
    
    # Donor specific blood distributions 
    parser.add_argument("--donor_prob_o", type=float, default=0.4, help="Probability of Donor Blood Type O")
    parser.add_argument("--donor_prob_a", type=float, default=0.4, help="Probability of Donor Blood Type A")
    parser.add_argument("--donor_prob_b", type=float, default=0.1, help="Probability of Donor Blood Type B")
    
    # Base characteristics
    parser.add_argument("--donors1", type=float, default=1.0, help="Prop of patients with 1 donor")
    parser.add_argument("--donors2", type=float, default=0.0, help="Prop of patients with 2 donors")
    parser.add_argument("--donors3", type=float, default=0.0, help="Prop of patients with 3 donors")
    parser.add_argument("--prob_spousal", type=float, default=0.0, help="Probability of Spousal Donor")
    parser.add_argument("--prob_female", type=float, default=0.0, help="Probability of Female Donor")
    parser.add_argument("--prob_spousal_pra_compat", type=float, default=0.0, help="Prob spousal PRA compat")
    
    # Advanced features matches web UI
    parser.add_argument("--no_tune", action="store_true", help="Disable Tuning (enabled by default)")
    parser.add_argument("--split_donor_blood", action="store_true", help="Use different donor blood group distributions based on recipient blood group")
    
    args = parser.parse_args()
    args.tune = not args.no_tune
    
    print("=== Step 0: CLI-Driven KEP Data Generation ===")
    
    if not os.path.exists(WEBAPP_DIR):
        print("Error: The generator_webapp directory is missing.")
        return

    # Check for node
    if shutil.which("node") is None:
        print("Error: Node.js is not installed. Please install node.js to run the generator headless.")
        print("Run: conda install -c conda-forge nodejs")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    
    config_dict = create_config(args)
    
    print("\nBuilding Node.js generator script...")
    combined_js = build_node_script()
    if not combined_js:
        return
        
    print(f"Generating {args.instances} graphs, {args.patients} patients/graph, NDD ratio: {args.prob_ndd}")
    if args.tune: print(" -> Tuning Enabled")
    if args.split_donor_blood: print(" -> Split Donor Blood Distribution Enabled")
    
    # Save transient config
    fd, temp_config_path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print("\\nExecuting JSON Generation via Node.js...")
        result = subprocess.run(["node", combined_js, temp_config_path, DATASET_DIR], check=True, text=True)
        
        # Save the master config to the dataset directory for future reference
        master_config_path = os.path.join(DATASET_DIR, "config.json")
        with open(master_config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Hyperparameters saved to {master_config_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"\\nGeneration failed with error code: {e.returncode}")
        
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        if os.path.exists(combined_js):
            os.remove(combined_js)

if __name__ == "__main__":
    main()
