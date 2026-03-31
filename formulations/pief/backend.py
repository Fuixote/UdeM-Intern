import gurobipy as gp
from gurobipy import GRB

from formulations.common.backend_utils import (
    build_edge_lists,
    build_pief_chain_keys,
    build_pief_cycle_keys,
    decode_chain_matches,
    decode_pief_cycle_matches,
    edge_selection_array,
    format_matches,
    to_numpy_edge_index,
    to_numpy_weights,
)


def solve_pief(
    weights,
    edge_index,
    is_ndd_mask,
    num_nodes,
    max_cycle=3,
    max_chain=4,
    env=None,
    time_limit=None,
    id_map_rev=None,
):
    weights = to_numpy_weights(weights)
    edge_index_np = to_numpy_edge_index(edge_index)
    src, dst, outgoing, incoming = build_edge_lists(edge_index_np, num_nodes)
    pair_nodes = [node_idx for node_idx in range(num_nodes) if not bool(is_ndd_mask[node_idx])]
    ndd_nodes = [node_idx for node_idx in range(num_nodes) if bool(is_ndd_mask[node_idx])]

    valid_chain_keys = build_pief_chain_keys(
        src=src,
        dst=dst,
        outgoing=outgoing,
        is_ndd_mask=is_ndd_mask,
        max_chain=max_chain,
    )
    cycle_start_nodes, cycle_order_rank, valid_cycle_keys = build_pief_cycle_keys(
        src=src,
        dst=dst,
        outgoing=outgoing,
        is_ndd_mask=is_ndd_mask,
        num_nodes=num_nodes,
        max_cycle=max_cycle,
    )

    chain_incoming = {}
    chain_outgoing = {}
    for edge_idx, position in valid_chain_keys:
        chain_incoming.setdefault((int(dst[edge_idx]), position), []).append((edge_idx, position))
        chain_outgoing.setdefault((int(src[edge_idx]), position), []).append((edge_idx, position))

    cycle_incoming = {}
    cycle_outgoing = {}
    for start_node, edge_idx, position in valid_cycle_keys:
        cycle_incoming.setdefault((start_node, int(dst[edge_idx]), position), []).append(
            (start_node, edge_idx, position)
        )
        cycle_outgoing.setdefault((start_node, int(src[edge_idx]), position), []).append(
            (start_node, edge_idx, position)
        )

    model = gp.Model("KEP_PIEF", env=env)
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    chain_vars = model.addVars(valid_chain_keys, vtype=GRB.BINARY, name="chain")
    cycle_vars = model.addVars(valid_cycle_keys, vtype=GRB.BINARY, name="cycle")

    model.setObjective(
        gp.quicksum(weights[edge_idx] * chain_vars[edge_idx, position] for edge_idx, position in valid_chain_keys)
        + gp.quicksum(
            weights[edge_idx] * cycle_vars[start_node, edge_idx, position]
            for start_node, edge_idx, position in valid_cycle_keys
        ),
        GRB.MAXIMIZE,
    )

    for pair_node in pair_nodes:
        cycle_usage = gp.quicksum(
            cycle_vars[start_node, edge_idx, position]
            for start_node, edge_idx, position in valid_cycle_keys
            if int(dst[edge_idx]) == pair_node
        )
        chain_usage = gp.quicksum(
            chain_vars[edge_idx, position]
            for edge_idx, position in valid_chain_keys
            if int(dst[edge_idx]) == pair_node
        )
        model.addConstr(cycle_usage + chain_usage <= 1, name=f"pair_once_{pair_node}")

    for ndd_node in ndd_nodes:
        model.addConstr(
            gp.quicksum(
                chain_vars[edge_idx, 1]
                for edge_idx, position in valid_chain_keys
                if position == 1 and int(src[edge_idx]) == ndd_node
            )
            <= 1,
            name=f"ndd_once_{ndd_node}",
        )

    for pair_node in pair_nodes:
        for position in range(2, max_chain + 1):
            outgoing_at_position = gp.quicksum(
                chain_vars[edge_idx, position]
                for edge_idx, _ in chain_outgoing.get((pair_node, position), [])
            )
            incoming_previous = gp.quicksum(
                chain_vars[edge_idx, position - 1]
                for edge_idx, _ in chain_incoming.get((pair_node, position - 1), [])
            )
            model.addConstr(
                outgoing_at_position <= incoming_previous,
                name=f"chain_flow_{pair_node}_{position}",
            )

    for start_node in cycle_start_nodes:
        start_rank = cycle_order_rank[start_node]
        for pair_node in cycle_start_nodes:
            if cycle_order_rank[pair_node] <= start_rank:
                continue
            for position in range(1, max_cycle):
                outgoing_at_position = gp.quicksum(
                    cycle_vars[key]
                    for key in cycle_outgoing.get((start_node, pair_node, position + 1), [])
                )
                incoming_previous = gp.quicksum(
                    cycle_vars[key]
                    for key in cycle_incoming.get((start_node, pair_node, position), [])
                )
                model.addConstr(
                    incoming_previous == outgoing_at_position,
                    name=f"cycle_flow_{start_node}_{pair_node}_{position}",
                )

    model.optimize()

    selected_chain_keys = []
    selected_cycle_keys = []
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        selected_chain_keys = [
            (edge_idx, position)
            for edge_idx, position in valid_chain_keys
            if chain_vars[edge_idx, position].X > 0.5
        ]
        selected_cycle_keys = [
            (start_node, edge_idx, position)
            for start_node, edge_idx, position in valid_cycle_keys
            if cycle_vars[start_node, edge_idx, position].X > 0.5
        ]

    selected_matches = []
    selected_edges = []
    cycle_matches = decode_pief_cycle_matches(selected_cycle_keys, src, dst, max_cycle)
    chain_matches = decode_chain_matches(selected_chain_keys, src, dst, max_chain)
    selected_matches.extend(cycle_matches)
    selected_matches.extend(chain_matches)
    selected_edges.extend(edge_idx for _, edge_idx, _ in selected_cycle_keys)
    selected_edges.extend(edge_idx for edge_idx, _ in selected_chain_keys)

    result = {
        "status": model.status,
        "objective": float(model.ObjVal) if model.SolCount > 0 else 0.0,
        "edge_selection": edge_selection_array(len(src), selected_edges),
        "matches": selected_matches,
    }
    if id_map_rev is not None:
        result["formatted_matches"] = format_matches(selected_matches, id_map_rev, weights)
    return result
