from collections import deque

import numpy as np


def to_numpy_edge_index(edge_index):
    if hasattr(edge_index, "detach"):
        return edge_index.detach().cpu().numpy()
    return np.asarray(edge_index)


def to_numpy_weights(weights):
    if hasattr(weights, "detach"):
        return weights.detach().cpu().numpy().flatten()
    return np.asarray(weights, dtype=np.float32).flatten()


def infer_ndd_mask(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x[:, -1] > 0.5


def build_edge_lists(edge_index, num_nodes):
    edge_index_np = to_numpy_edge_index(edge_index)
    src = edge_index_np[0].astype(int)
    dst = edge_index_np[1].astype(int)
    outgoing = [[] for _ in range(num_nodes)]
    incoming = [[] for _ in range(num_nodes)]
    for edge_idx, (u, v) in enumerate(zip(src, dst)):
        outgoing[u].append(edge_idx)
        incoming[v].append(edge_idx)
    return src, dst, outgoing, incoming


def pair_nodes_by_desc_degree(src, dst, is_ndd_mask, num_nodes):
    degrees = np.zeros(num_nodes, dtype=np.int32)
    for edge_src, edge_dst in zip(src, dst):
        if bool(is_ndd_mask[edge_src]) or bool(is_ndd_mask[edge_dst]):
            continue
        degrees[edge_src] += 1
        degrees[edge_dst] += 1

    pair_nodes = [node_idx for node_idx in range(num_nodes) if not bool(is_ndd_mask[node_idx])]
    pair_nodes.sort(key=lambda node_idx: (-int(degrees[node_idx]), int(node_idx)))
    order_rank = {node_idx: rank for rank, node_idx in enumerate(pair_nodes)}
    return pair_nodes, order_rank


def shortest_distances_to_start(src, dst, is_ndd_mask, order_rank, start_node, num_nodes):
    max_dist = num_nodes + 1
    start_rank = order_rank[start_node]
    reverse_adj = [[] for _ in range(num_nodes)]

    for edge_src, edge_dst in zip(src, dst):
        if bool(is_ndd_mask[edge_src]) or bool(is_ndd_mask[edge_dst]):
            continue
        if order_rank.get(int(edge_src), max_dist) < start_rank:
            continue
        if order_rank.get(int(edge_dst), max_dist) < start_rank:
            continue
        reverse_adj[int(edge_dst)].append(int(edge_src))

    dist = np.full(num_nodes, max_dist, dtype=np.int32)
    queue = deque([start_node])
    dist[start_node] = 0

    while queue:
        node_idx = queue.popleft()
        for prev_node in reverse_adj[node_idx]:
            if dist[prev_node] != max_dist:
                continue
            dist[prev_node] = dist[node_idx] + 1
            queue.append(prev_node)

    return dist


def build_pief_chain_keys(src, dst, outgoing, is_ndd_mask, max_chain):
    valid_chain_keys = []
    active_donors = {node_idx for node_idx in range(len(outgoing)) if bool(is_ndd_mask[node_idx])}

    for position in range(1, max_chain + 1):
        next_active = set()
        for donor_idx in active_donors:
            for edge_idx in outgoing[donor_idx]:
                recipient_idx = int(dst[edge_idx])
                if bool(is_ndd_mask[recipient_idx]):
                    continue
                valid_chain_keys.append((edge_idx, position))
                next_active.add(recipient_idx)
        active_donors = next_active
        if not active_donors:
            break

    return valid_chain_keys


def build_pief_cycle_keys(src, dst, outgoing, is_ndd_mask, num_nodes, max_cycle):
    pair_nodes, order_rank = pair_nodes_by_desc_degree(src, dst, is_ndd_mask, num_nodes)
    valid_cycle_keys = []

    for start_node in pair_nodes:
        dist_to_start = shortest_distances_to_start(
            src=src,
            dst=dst,
            is_ndd_mask=is_ndd_mask,
            order_rank=order_rank,
            start_node=start_node,
            num_nodes=num_nodes,
        )
        start_rank = order_rank[start_node]
        active_donors = {start_node}

        for position in range(1, max_cycle + 1):
            next_active = set()
            for donor_idx in active_donors:
                for edge_idx in outgoing[donor_idx]:
                    recipient_idx = int(dst[edge_idx])
                    if bool(is_ndd_mask[recipient_idx]):
                        continue
                    recipient_rank = order_rank.get(recipient_idx)
                    if recipient_rank is None or recipient_rank < start_rank:
                        continue
                    if recipient_idx == start_node:
                        valid_cycle_keys.append((start_node, edge_idx, position))
                    elif int(dist_to_start[recipient_idx]) <= max_cycle - position:
                        valid_cycle_keys.append((start_node, edge_idx, position))
                        next_active.add(recipient_idx)
            active_donors = next_active
            if not active_donors and position < max_cycle:
                break

    return pair_nodes, order_rank, valid_cycle_keys


def edge_selection_array(num_edges, edge_indices):
    selected = np.zeros(num_edges, dtype=np.float32)
    for edge_idx in edge_indices:
        selected[edge_idx] = 1.0
    return selected


def decode_chain_matches(selected_chain_keys, src, dst, max_chain):
    if not selected_chain_keys:
        return []

    by_position_source = {}
    for edge_idx, position in selected_chain_keys:
        by_position_source[(position, int(src[edge_idx]))] = edge_idx

    start_edges = sorted(
        [(edge_idx, position) for edge_idx, position in selected_chain_keys if position == 1],
        key=lambda item: (int(src[item[0]]), int(dst[item[0]])),
    )

    matches = []
    for edge_idx, _ in start_edges:
        nodes = [int(src[edge_idx]), int(dst[edge_idx])]
        edges = [edge_idx]
        current = int(dst[edge_idx])
        for position in range(2, max_chain + 1):
            next_edge = by_position_source.get((position, current))
            if next_edge is None:
                break
            edges.append(next_edge)
            current = int(dst[next_edge])
            nodes.append(current)
        matches.append({"type": "chain", "nodes": nodes, "edges": edges})
    return matches


def decode_pief_cycle_matches(selected_cycle_keys, src, dst, max_cycle):
    if not selected_cycle_keys:
        return []

    by_start = {}
    for start_node, edge_idx, position in selected_cycle_keys:
        bucket = by_start.setdefault(int(start_node), {})
        bucket[position] = edge_idx

    matches = []
    for start_node in sorted(by_start):
        position_to_edge = by_start[start_node]
        first_edge = position_to_edge.get(1)
        if first_edge is None:
            continue

        nodes = [start_node, int(dst[first_edge])]
        edges = [first_edge]
        current = int(dst[first_edge])

        for position in range(2, max_cycle + 1):
            edge_idx = position_to_edge.get(position)
            if edge_idx is None:
                break
            edges.append(edge_idx)
            next_node = int(dst[edge_idx])
            if next_node == start_node:
                matches.append({"type": "cycle", "nodes": nodes, "edges": edges})
                break
            nodes.append(next_node)
            current = next_node
        else:
            continue

    return matches


def format_matches(matches, id_map_rev, weights):
    formatted = []
    for match in matches:
        formatted.append(
            {
                "type": match["type"],
                "node_ids": [id_map_rev[node_idx] for node_idx in match["nodes"]],
                "predicted_w": float(sum(weights[edge_idx] for edge_idx in match["edges"])),
                "edge_weights": [float(weights[edge_idx]) for edge_idx in match["edges"]],
            }
        )
    return formatted
