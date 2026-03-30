import json
import os
import glob
import re
import argparse
import math
import random
import hashlib

def process_file(input_file, output_file):
    print(f"Processing {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        raw_data = data.get('data', {})
        recipients_data = data.get('recipients', {})
        
        # We will collect everything into a new dictionary where the key is the Pair ID / NDD ID
        vertices = {}
        
        # Helper for clinical metric calculation
        def get_survival_time(age):
            if age is None or age == 'Unknown': return None
            # Formula: max(5, 25 - 0.3 * (dage - 20))
            val = 25 - 0.3 * (float(age) - 20)
            return round(max(5.0, val), 2)
        
        def get_qaly(utility, survival_time):
            if survival_time is None: return None
            # Formula: (sqrt(1^2 + (Utility/100)^2) / sqrt(2)) * Time
            u_norm = float(utility) / 91.0
            multiplier = math.sqrt(1.0 + u_norm**2) / math.sqrt(2.0)
            return round(multiplier * survival_time, 2)
        
        def get_success_prob(utility, cpra, alpha=0.7, beta=0.3):
            if utility is None or cpra is None or cpra == 'Unknown': return None
            # Formula: alpha * (utility/91) + beta * (1 - cPRA)
            score_part = float(utility) / 91.0
            cpra_part = 1.0 - float(cpra)
            return round(alpha * score_part + beta * cpra_part, 4)
        
        def get_deterministic_epsilon(source_key, sigma=0.15):
            digest = hashlib.sha256(source_key.encode('utf-8')).digest()
            seed = int.from_bytes(digest[:8], byteorder='big', signed=False)
            rng = random.Random(seed)
            return rng.gauss(0, sigma)

        def get_ground_truth(success_prob, qaly, source_key, sigma=0.15):
            if success_prob is None or qaly is None: return None
            # Formula: P(success) * QALY * (1 + epsilon), where epsilon ~ N(0, sigma^2)
            # Use a deterministic pseudo-random draw derived from edge identity so
            # repeated preprocessing produces identical labels.
            epsilon = get_deterministic_epsilon(source_key, sigma)
            w_true = success_prob * qaly * (1 + epsilon)
            return round(max(0.0, w_true), 4)
        
        for node_id, attributes in raw_data.items():
            is_altruistic = attributes.get('altruistic', False)
            
            if is_altruistic:
                # For NDD, the ID remains the node_id
                vertices[str(node_id)] = {
                    'type': 'NDD',
                    'id': str(node_id),
                    'donor': {
                        'dage': attributes.get('dage', 'Unknown'),
                        'bloodtype': attributes.get('bloodtype', 'Unknown'),
                    },
                    'matches': []
                }
                # Normalize NDD matches
                if 'matches' in attributes:
                    for match in attributes['matches']:
                        d_age = match.get('donor_age')
                        s_time = get_survival_time(d_age)
                        u_val = match.get('utility', 0)
                        cpra_val = match.get('recipient_cpra')
                        source_key = "|".join([
                            os.path.basename(input_file),
                            str(node_id),
                            str(match.get('recipient')),
                            str(u_val),
                        ])
                        vertices[str(node_id)]['matches'].append({
                            'donor_node_id': str(node_id),
                            'recipient': str(match['recipient']),
                            'utility': u_val,
                            'graft_survival_time': s_time,
                            'qaly': get_qaly(u_val, s_time),
                            'success_prob': get_success_prob(u_val, cpra_val),
                            'ground_truth_label': get_ground_truth(
                                get_success_prob(u_val, cpra_val),
                                get_qaly(u_val, s_time),
                                source_key
                            ),
                            'donor_age': d_age,
                            'donor_bt': match.get('donor_bt'),
                            'recipient_age': match.get('recipient_age'),
                            'recipient_cpra': match.get('recipient_cpra'),
                            'recipient_bt': match.get('recipient_bt')
                        })
            else:
                # For Pair, the ID becomes the patient_id (source)
                sources = attributes.get('sources', [])
                patient_id = str(sources[0]) if sources else str(node_id)
                
                if patient_id not in vertices:
                    # Initialize the Pair if it doesn't exist
                    p_info = recipients_data.get(patient_id, {})
                    vertices[patient_id] = {
                        'type': 'Pair',
                        'id': patient_id,
                        'patient': {
                            'age': p_info.get('age', 'Unknown'),
                            'bloodtype': p_info.get('bloodtype', 'Unknown'),
                            'cPRA': p_info.get('cPRA', 'Unknown'),
                            'hasBloodCompatibleDonor': p_info.get('hasBloodCompatibleDonor', False)
                        },
                        'donors': [],
                        'matches': []
                    }
                
                # Append this specific donor to the Pair
                vertices[patient_id]['donors'].append({
                    'original_node_id': str(node_id),
                    'dage': attributes.get('dage', 'Unknown'),
                    'bloodtype': attributes.get('bloodtype', 'Unknown')
                })
                
                # Collect all outgoing matches from this donor
                if 'matches' in attributes:
                    for match in attributes['matches']:
                        d_age = match.get('donor_age')
                        s_time = get_survival_time(d_age)
                        u_val = match.get('utility', 0)
                        cpra_val = match.get('recipient_cpra')
                        source_key = "|".join([
                            os.path.basename(input_file),
                            str(node_id),
                            str(match.get('recipient')),
                            str(u_val),
                        ])
                        vertices[patient_id]['matches'].append({
                            'donor_node_id': str(node_id),
                            'recipient': str(match['recipient']),
                            'utility': u_val,
                            'graft_survival_time': s_time,
                            'qaly': get_qaly(u_val, s_time),
                            'success_prob': get_success_prob(u_val, cpra_val),
                            'ground_truth_label': get_ground_truth(
                                get_success_prob(u_val, cpra_val),
                                get_qaly(u_val, s_time),
                                source_key
                            ),
                            'donor_age': d_age,
                            'donor_bt': match.get('donor_bt'),
                            'recipient_age': match.get('recipient_age'),
                            'recipient_cpra': match.get('recipient_cpra'),
                            'recipient_bt': match.get('recipient_bt')
                        })

        # Process matches to keep only the highest utility edge between two Vertices
        final_vertices = {}
        
        for vid, v_data in vertices.items():
            best_matches = {}
            for match in v_data.get('matches', []):
                target = match['recipient']
                utility = match['utility']
                
                # We only keep edges if the target patient actually stabilized into our vertices graph
                if target in vertices:
                    # Update if target not seen yet, OR if this new utility is strictly greater
                    if target not in best_matches or utility > best_matches[target]['utility']:
                        # Save the match specifically tailored to the Pair-to-Pair format
                        best_matches[target] = {
                            'recipient': target,
                            'utility': utility,
                            'graft_survival_time': match.get('graft_survival_time'),
                            'qaly': match.get('qaly'),
                            'success_prob': match.get('success_prob'),
                            'ground_truth_label': match.get('ground_truth_label'),
                            # Keep full medical context for the "winning" donor edge
                            'donor_age': match.get('donor_age'),
                            'donor_bt': match.get('donor_bt'),
                            'recipient_age': match.get('recipient_age'),
                            'recipient_cpra': match.get('recipient_cpra'),
                            'recipient_bt': match.get('recipient_bt')
                        }
                        # For Pairs, keep track of which donor won the edge
                        if 'donor_node_id' in match:
                            best_matches[target]['winning_donor_id'] = match['donor_node_id']

            # Overwrite the raw accumulated matches with our filtered optimal matches list
            v_data['matches'] = list(best_matches.values())
            final_vertices[vid] = v_data

        # Construct final JSON structure
        output_data = {
            'metadata': {
                'original_file': os.path.basename(input_file),
                'total_vertices': len(final_vertices),
                'structure': 'Unified Pair/NDD Graph',
                'ground_truth_noise_sigma': 0.15,
                'ground_truth_noise_mode': 'deterministic_per_edge'
            },
            'data': final_vertices
        }

        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(output_data, out_f, indent=2)
            
        print(f"Successfully wrote {len(final_vertices)} vertices to {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert Donor-based KEP JSON graphs into Pair/NDD-based G-JSON graphs.')
    parser.add_argument('input_dir', nargs='?', default='dataset/raw', help='Directory containing the original JSON files (default: dataset/raw)')
    parser.add_argument('output_dir', nargs='?', default='dataset/processed', help='Directory to output G-X.json files (default: dataset/processed)')
    parser.add_argument('--all', action='store_true', help='Process all genjson-*.json files in the input directory')
    parser.add_argument('--file', type=str, help='Process a specific file (e.g., genjson-1494.json)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.file:
        input_path = os.path.join(args.input_dir, args.file)
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return
            
        # Extract the number from genjson-X.json
        match = re.search(r'genjson-(\d+)\.json', args.file)
        if match:
            file_num = match.group(1)
            output_file = os.path.join(args.output_dir, f'G-{file_num}.json')
            process_file(input_path, output_file)
        else:
            print(f"Could not parse number from filename: {args.file}. Skipping.")
            
    elif args.all:
        pattern = os.path.join(args.input_dir, 'genjson-*.json')
        files = glob.glob(pattern)
        
        if not files:
            print(f"No files matching genjson-*.json found in {args.input_dir}")
            return
            
        print(f"Found {len(files)} files to process.")
        for f in files:
            basename = os.path.basename(f)
            match = re.search(r'genjson-(\d+)\.json', basename)
            if match:
                file_num = match.group(1)
                output_file = os.path.join(args.output_dir, f'G-{file_num}.json')
                process_file(f, output_file)
    else:
        print("Please specify a target: use --all to process all files, or --file <filename> for a single file.")
        print("Example: python generate_pair_graph.py --file genjson-1523.json")

if __name__ == "__main__":
    main()
