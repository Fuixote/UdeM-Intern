import glob
import json
import os

import torch
from torch_geometric.data import Data

from model.model_structure import DEFAULT_Y_SCALE

BT_MAP = {"O": 0, "A": 1, "B": 2, "AB": 3}


def get_one_hot_bt(bt_str):
    vec = [0.0, 0.0, 0.0, 0.0]
    if bt_str in BT_MAP:
        vec[BT_MAP[bt_str]] = 1.0
    return vec


def get_pair_representative_donor(node):
    donors = node.get('donors', [])
    if not donors:
        raise ValueError(f"Pair node {node.get('id', '<unknown>')} has no donors")
    return donors[0]


def build_node_feature(node):
    if node['type'] == 'Pair':
        patient = node['patient']
        donor = get_pair_representative_donor(node)
        return [
            patient['age'] / 100.0,
            patient['cPRA'],
            1.0 if patient['hasBloodCompatibleDonor'] else 0.0,
        ] + get_one_hot_bt(patient['bloodtype']) + [
            donor['dage'] / 100.0,
        ] + get_one_hot_bt(donor['bloodtype']) + [
            0.0
        ]

    donor = node['donor']
    return [
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        donor['dage'] / 100.0,
    ] + get_one_hot_bt(donor['bloodtype']) + [
        1.0
    ]


def build_graph_components(json_path, label_scale=1.0):
    with open(json_path, 'r') as f:
        content = json.load(f)

    nodes_data = content['data']
    node_ids = sorted(nodes_data.keys(), key=lambda x: int(x))
    id_map = {old_id: i for i, old_id in enumerate(node_ids)}
    id_map_rev = {i: old_id for i, old_id in enumerate(node_ids)}

    x_list = []
    adj = {}
    for nid in node_ids:
        x_list.append(build_node_feature(nodes_data[nid]))
        adj[id_map[nid]] = []

    x = torch.tensor(x_list, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    y_labels = []
    gt_labels = []
    edge_count = 0

    for src_id, node in nodes_data.items():
        for match in node.get('matches', []):
            dst_id = match['recipient']
            src_idx = id_map[src_id]
            dst_idx = id_map[dst_id]
            edge_indices.append([src_idx, dst_idx])
            edge_attrs.append([match['utility'] / 100.0])
            gt_label = match.get('ground_truth_label', 0.0)
            gt_labels.append(gt_label)
            y_labels.append(gt_label / label_scale)
            adj[src_idx].append({'dst': dst_idx, 'edge_idx': edge_count})
            edge_count += 1

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        y = torch.tensor(y_labels, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        y = torch.empty((0,), dtype=torch.float)

    return {
        'content': content,
        'nodes_data': nodes_data,
        'node_ids': node_ids,
        'id_map': id_map,
        'id_map_rev': id_map_rev,
        'num_nodes': len(node_ids),
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': y,
        'gt_labels': gt_labels,
        'adj': adj,
        'filename': os.path.basename(json_path),
    }


def find_all_cycles_and_chains(adj, nodes_data, id_map_rev, max_cycle=3, max_chain=5):
    cycles = []
    chains = []
    num_nodes = len(nodes_data)

    ndd_indices = [i for i in range(num_nodes) if nodes_data[id_map_rev[i]]['type'] != 'Pair']
    pair_indices = [i for i in range(num_nodes) if nodes_data[id_map_rev[i]]['type'] == 'Pair']

    def dfs_chain(u, current_path, current_edges):
        if len(current_edges) >= max_chain:
            return
        for edge in adj[u]:
            v = edge['dst']
            if v not in current_path:
                chains.append({'nodes': current_path + [v], 'edges': current_edges + [edge['edge_idx']]})
                dfs_chain(v, current_path + [v], current_edges + [edge['edge_idx']])

    for ndd in ndd_indices:
        dfs_chain(ndd, [ndd], [])

    def dfs_cycle(start_node, u, current_path, current_edges):
        if len(current_path) > max_cycle:
            return
        for edge in adj[u]:
            v = edge['dst']
            if v == start_node:
                if start_node == min(current_path):
                    cycles.append({'nodes': current_path, 'edges': current_edges + [edge['edge_idx']]})
            elif v not in current_path and v > start_node:
                if nodes_data[id_map_rev[v]]['type'] == 'Pair':
                    dfs_cycle(start_node, v, current_path + [v], current_edges + [edge['edge_idx']])

    for p_idx in pair_indices:
        dfs_cycle(p_idx, p_idx, [p_idx], [])

    return cycles, chains


def enumerate_candidates(adj, nodes_data, id_map_rev, max_cycle=3, max_chain=5):
    cycles, chains = find_all_cycles_and_chains(
        adj, nodes_data, id_map_rev, max_cycle=max_cycle, max_chain=max_chain
    )
    typed_cycles = [{'type': 'cycle', **candidate} for candidate in cycles]
    typed_chains = [{'type': 'chain', **candidate} for candidate in chains]
    return typed_cycles + typed_chains


def parse_json_to_pyg_data(json_path, label_scale=1.0):
    graph = build_graph_components(json_path, label_scale=label_scale)
    return Data(
        x=graph['x'],
        edge_index=graph['edge_index'],
        edge_attr=graph['edge_attr'],
        y=graph['y'],
        filename=graph['filename'],
    )


def parse_json_to_graph_info(json_path):
    graph = build_graph_components(json_path)
    return {
        'x': graph['x'],
        'edge_index': graph['edge_index'],
        'edge_attr': graph['edge_attr'],
        'adj': graph['adj'],
        'nodes_data': graph['nodes_data'],
        'id_map_rev': graph['id_map_rev'],
        'num_nodes': graph['num_nodes'],
        'gt_labels': graph['gt_labels'],
        'filename': graph['filename'],
    }


def parse_json_to_dfl_data(json_path, max_cycle=3, max_chain=5, label_scale=DEFAULT_Y_SCALE):
    graph = build_graph_components(json_path, label_scale=label_scale)
    candidates = enumerate_candidates(
        graph['adj'], graph['nodes_data'], graph['id_map_rev'],
        max_cycle=max_cycle, max_chain=max_chain
    )

    data = Data(
        x=graph['x'],
        edge_index=graph['edge_index'],
        edge_attr=graph['edge_attr'],
        y=graph['y'],
        filename=graph['filename'],
    )
    data.candidates = candidates
    data.num_nodes_custom = torch.tensor([graph['num_nodes']], dtype=torch.long)
    data.id_map_rev = graph['id_map_rev']
    return data


def load_graph_dataset(directory, parser_fn, pattern="G-*.json", log_prefix="Loading"):
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    dataset = []
    print(f"{log_prefix} {len(files)} graph files from {directory}...")
    for graph_file in files:
        try:
            dataset.append(parser_fn(graph_file))
        except Exception as e:
            print(f"⚠️ Skipping invalid file {graph_file}: {e}")
    return dataset
