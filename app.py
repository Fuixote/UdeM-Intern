from flask import Flask, render_template, jsonify, request
import json
import os

app = Flask(__name__)

# Basic Configuration
DATASET_DIR = os.path.join(app.root_path, 'dataset', 'processed')
SOLUTIONS_DIR = os.path.join(app.root_path, 'solutions')


def solution_sort_key(rel_path):
    rel_dir = os.path.dirname(rel_path)
    base = os.path.basename(rel_path)
    try:
        graph_num = int(base.split('-')[1].split('_')[0])
    except (IndexError, ValueError):
        graph_num = float('inf')
    return (rel_dir, graph_num, base)


def list_solution_files():
    rel_paths = []
    for root, _, files in os.walk(SOLUTIONS_DIR):
        for filename in files:
            if filename.endswith('_sol.json'):
                abs_path = os.path.join(root, filename)
                rel_paths.append(os.path.relpath(abs_path, SOLUTIONS_DIR))
    return sorted(rel_paths, key=solution_sort_key)


def build_edge_tooltip(match):
    lines = []

    survival = match.get('graft_survival_time')
    if survival is not None:
        lines.append(f"&bull; Estimated Graft Survival: {survival} years")

    qaly = match.get('qaly')
    if qaly is not None:
        lines.append(f"&bull; Estimated QALY: {qaly}")

    success_prob = match.get('success_prob')
    if success_prob is not None:
        lines.append(f"&bull; Success Probability: {success_prob * 100:.2f}%")

    gt_score = match.get('ground_truth_label')
    if gt_score is not None:
        lines.append(f"<b>Ground-Truth Score: {gt_score}</b>")

    donor_info = (
        f"&bull; Donor: Age {match.get('donor_age', '?')}, "
        f"BT {match.get('donor_bt', '?')}"
    )
    recipient_info = (
        f"&bull; Recipient: Age {match.get('recipient_age', '?')}, "
        f"BT {match.get('recipient_bt', '?')}, "
        f"cPRA {match.get('recipient_cpra', '?')}"
    )

    details = [
        "<u>Match Details</u>",
        donor_info,
        recipient_info,
    ]

    return "<br>".join(lines + [""] + details)


@app.route('/')
def index():
    available_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
    available_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]) if '-' in x else 0)
    default_file = 'G-0.json' if 'G-0.json' in available_files else (available_files[0] if available_files else None)
    return render_template('index.html', available_files=available_files, default_file=default_file)

@app.route('/api/data')
def get_data():
    filename = request.args.get('file', 'G-0.json')
    filepath = os.path.join(DATASET_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(filepath, 'r') as f:
            json_file = json.load(f)
            vertices = json_file.get('data', {})

        nodes = []
        edges = []
        out_degrees = {vid: 0 for vid in vertices.keys()}
        patient_in_degrees = {vid: 0 for vid in vertices.keys()}
        
        # Build edges
        for vid, v_data in vertices.items():
            for match in v_data.get('matches', []):
                target = match['recipient']
                # Only add edge if the target patient actually exists in our graph
                if target in vertices:
                    tooltip = build_edge_tooltip(match)

                    edges.append({
                        "from": vid,
                        "to": target,
                        "value": match['utility'],
                        "title": tooltip,
                        "arrows": "to"
                    })
                    out_degrees[vid] += 1
                    patient_in_degrees[target] += 1

        # Build nodes
        for vid, v_data in vertices.items():
            out_deg = out_degrees.get(vid, 0)
            in_deg = patient_in_degrees.get(vid, 0)
            
            if v_data['type'] == 'NDD':
                d_info = v_data.get('donor', {})
                dage = d_info.get('dage', 'Unknown')
                dbt = d_info.get('bloodtype', 'Unknown')
                
                tooltip = f"<b>Altruistic Donor (NDD) {vid}</b><br><br>"
                tooltip += f"<u>Donor Information</u><br>"
                tooltip += f"Age: {dage} | Blood: {dbt}<br>"
                tooltip += f"<i>Out-Degree: {out_deg} (can initiate {out_deg} chains)</i><br>"
                tooltip += f"<i>In-Degree: N/A (Does not receive)</i>"
                
                nodes.append({
                    "id": vid,
                    "label": f"A{vid}",
                    "value": out_deg,
                    "title": tooltip,
                    "group": out_deg,
                    "is_altruistic": True
                })
            else:
                # Pair
                p_info = v_data.get('patient', {})
                p_bloodtype = p_info.get('bloodtype', 'Unknown')
                cpra = p_info.get('cPRA', 'Unknown')
                p_age = p_info.get('age', 'Unknown')
                
                tooltip = f"<b>Patient-Donor Pair {vid}</b><br><br>"
                tooltip += f"<u>Patient {vid}</u><br>"
                tooltip += f"Age: {p_age} | Blood: {p_bloodtype} | cPRA: {cpra}<br>"
                tooltip += f"<br><u>Associated Donors ({len(v_data.get('donors', []))})</u><br>"
                
                for d in v_data.get('donors', []):
                    tooltip += f"&bull; Donor {d.get('original_node_id', 'Unknown')}: Age {d.get('dage', 'Unknown')}, Blood {d.get('bloodtype', 'Unknown')}<br>"
                
                tooltip += f"<br><i>Out-Degree: {out_deg} (can donate to {out_deg} pairs)</i><br>"
                tooltip += f"<i>In-Degree: {in_deg} ({in_deg} pairs/NDDs can donate to them)</i>"
                
                nodes.append({
                    "id": vid,
                    "label": f"P{vid}",
                    "value": out_deg,
                    "title": tooltip,
                    "group": out_deg,
                    "is_altruistic": False
                })
                    
        return jsonify({
            "nodes": nodes,
            "edges": edges
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/solutions')
def solutions_page():
    available_files = list_solution_files()
    default_file = available_files[0] if available_files else None
    return render_template('solutions.html', available_files=available_files, default_file=default_file)

@app.route('/api/solution_data')
def get_solution_data():
    sol_filename = request.args.get('file')
    if not sol_filename:
        return jsonify({"error": "No file specified"}), 400

    sol_path = os.path.abspath(os.path.join(SOLUTIONS_DIR, sol_filename))
    solutions_root = os.path.abspath(SOLUTIONS_DIR)
    if os.path.commonpath([solutions_root, sol_path]) != solutions_root:
        return jsonify({"error": "Invalid solution path"}), 400

    if not os.path.exists(sol_path):
        return jsonify({"error": "Solution file not found"}), 404

    try:
        with open(sol_path, 'r') as f:
            sol_data = json.load(f)
        
        graph_filename = sol_data.get('graph')
        graph_path = os.path.join(DATASET_DIR, graph_filename)
        
        if not os.path.exists(graph_path):
            return jsonify({"error": f"Graph file {graph_filename} not found"}), 404
            
        with open(graph_path, 'r') as f:
            graph_json = json.load(f)
            vertices = graph_json.get('data', {})

        # 1. First Pass: Calculate all out-degrees from original graph
        out_degrees = {vid: 0 for vid in vertices.keys()}
        patient_in_degrees = {vid: 0 for vid in vertices.keys()}
        for vid, v_data in vertices.items():
            for match in v_data.get('matches', []):
                target = match['recipient']
                if target in vertices:
                    out_degrees[vid] += 1
                    patient_in_degrees[target] += 1

        # 2. Identify nodes and edges in the solution
        sol_nodes = set()
        sol_edges = {} # (from, to) -> weight
        for match in sol_data.get('matches', []):
            ids = match['node_ids']
            e_weights = match.get('edge_weights', [])
            
            if match['type'] == 'cycle':
                for i in range(len(ids)):
                    src = ids[i]; dst = ids[(i + 1) % len(ids)]
                    sol_nodes.add(src)
                    # Use provided edge weight if available, else fallback
                    w = e_weights[i] if i < len(e_weights) else 0.0
                    sol_edges[(src, dst)] = w
            else: # chain
                for i in range(len(ids) - 1):
                    src = ids[i]; dst = ids[i+1]
                    sol_nodes.add(src)
                    w = e_weights[i] if i < len(e_weights) else 0.0
                    sol_edges[(src, dst)] = w
            for nid in ids:
                sol_nodes.add(nid)

        # 3. Build Edges (Only include solution edges, and color them white)
        edges = []
        for vid, v_data in vertices.items():
            for match in v_data.get('matches', []):
                target = match['recipient']
                if target in vertices:
                    if (vid, target) in sol_edges:
                        w_pred = sol_edges[(vid, target)]
                        edges.append({
                            "from": vid,
                            "to": target,
                            "value": match['utility'],
                            "arrows": "to",
                            "color": "#ffffff",
                            "width": 3
                        })

        # 4. Build Nodes (Style consistent with explorer)
        nodes = []
        for vid, v_data in vertices.items():
            out_deg = out_degrees.get(vid, 0)
            in_deg = patient_in_degrees.get(vid, 0)
            in_sol = vid in sol_nodes
            
            # Use same tooltip logic as Explorer
            if v_data['type'] == 'NDD':
                node_obj = {
                    "id": vid,
                    "label": f"A{vid}",
                    "value": out_deg,
                    "group": out_deg,
                    "is_altruistic": True
                }
            else:
                node_obj = {
                    "id": vid,
                    "label": f"P{vid}",
                    "value": out_deg,
                    "group": out_deg,
                    "is_altruistic": False
                }
            
            # Solution specific styling: highlight borders, dim non-solution nodes
            if in_sol:
                node_obj["borderWidth"] = 3
                node_obj["color"] = {"border": "#ffffff"}
            else:
                node_obj["opacity"] = 0.3 # Keep them but dim them to maintain layout context
            
            nodes.append(node_obj)
                    
        return jsonify({
            "nodes": nodes,
            "edges": edges,
            "solution_meta": {
                "total_w": sol_data.get('total_predicted_w'),
                "num_matches": sol_data.get('num_matches'),
                "model": sol_data.get('model_used'),
                "matches": sol_data.get('matches')
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 监听所有IP并开启 Debug 模式方便开发
    app.run(host='0.0.0.0', port=5674, debug=True)
