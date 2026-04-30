import gurobipy as gp
from gurobipy import GRB

from formulations.common.backend_utils import (
    build_edge_lists,
    build_pief_chain_keys,
    decode_chain_matches,
    edge_selection_array,
    format_matches,
    to_numpy_edge_index,
    to_numpy_weights,
)


class CachedHybridKepModel:
    """Reusable hybrid KEP model for one fixed graph structure.

    The feasible set is fixed for a graph, so repeated solves only need to
    update the linear objective coefficients before optimize().
    """

    def __init__(
        self,
        edge_index,
        is_ndd_mask,
        num_nodes,
        cycle_candidates,
        max_chain=4,
        env=None,
        time_limit=None,
        name="KEP_CF_CYCLE_PIEF_CHAIN_CACHED",
    ):
        self.max_chain = max_chain
        self.cycle_candidates = cycle_candidates
        self.edge_index_np = to_numpy_edge_index(edge_index)
        self.src, self.dst, self.outgoing, self.incoming = build_edge_lists(self.edge_index_np, num_nodes)
        self.num_edges = len(self.src)
        self.pair_nodes = [node_idx for node_idx in range(num_nodes) if not bool(is_ndd_mask[node_idx])]
        self.ndd_nodes = [node_idx for node_idx in range(num_nodes) if bool(is_ndd_mask[node_idx])]

        self.valid_chain_keys = build_pief_chain_keys(
            src=self.src,
            dst=self.dst,
            outgoing=self.outgoing,
            is_ndd_mask=is_ndd_mask,
            max_chain=max_chain,
        )

        self.chain_incoming = {}
        self.chain_outgoing = {}
        for edge_idx, position in self.valid_chain_keys:
            self.chain_incoming.setdefault((int(self.dst[edge_idx]), position), []).append((edge_idx, position))
            self.chain_outgoing.setdefault((int(self.src[edge_idx]), position), []).append((edge_idx, position))

        self.model = gp.Model(name, env=env)
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = 1
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit

        self.cycle_vars = self.model.addVars(len(self.cycle_candidates), vtype=GRB.BINARY, name="cycle")
        self.chain_vars = self.model.addVars(self.valid_chain_keys, vtype=GRB.BINARY, name="chain")

        self.model.setObjective(
            gp.quicksum(0.0 * self.cycle_vars[idx] for idx in range(len(self.cycle_candidates)))
            + gp.quicksum(0.0 * self.chain_vars[key] for key in self.valid_chain_keys),
            GRB.MAXIMIZE,
        )

        for pair_node in self.pair_nodes:
            cycle_usage = gp.quicksum(
                self.cycle_vars[idx]
                for idx, candidate in enumerate(self.cycle_candidates)
                if pair_node in candidate["nodes"]
            )
            chain_usage = gp.quicksum(
                self.chain_vars[edge_idx, position]
                for edge_idx, position in self.valid_chain_keys
                if int(self.dst[edge_idx]) == pair_node
            )
            self.model.addConstr(cycle_usage + chain_usage <= 1, name=f"pair_once_{pair_node}")

        for ndd_node in self.ndd_nodes:
            self.model.addConstr(
                gp.quicksum(
                    self.chain_vars[edge_idx, 1]
                    for edge_idx, position in self.valid_chain_keys
                    if position == 1 and int(self.src[edge_idx]) == ndd_node
                )
                <= 1,
                name=f"ndd_once_{ndd_node}",
            )

        for pair_node in self.pair_nodes:
            for position in range(2, max_chain + 1):
                outgoing_at_position = gp.quicksum(
                    self.chain_vars[edge_idx, position]
                    for edge_idx, _ in self.chain_outgoing.get((pair_node, position), [])
                )
                incoming_previous = gp.quicksum(
                    self.chain_vars[edge_idx, position - 1]
                    for edge_idx, _ in self.chain_incoming.get((pair_node, position - 1), [])
                )
                self.model.addConstr(
                    outgoing_at_position <= incoming_previous,
                    name=f"chain_flow_{pair_node}_{position}",
                )
        self.model.update()

    def solve(self, weights, id_map_rev=None):
        weights = to_numpy_weights(weights)
        for idx, candidate in enumerate(self.cycle_candidates):
            self.cycle_vars[idx].Obj = float(sum(weights[edge_idx] for edge_idx in candidate["edges"]))
        for edge_idx, position in self.valid_chain_keys:
            self.chain_vars[edge_idx, position].Obj = float(weights[edge_idx])

        self.model.optimize()

        selected_cycle_indices = []
        selected_chain_keys = []
        if self.model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and self.model.SolCount > 0:
            selected_cycle_indices = [
                idx for idx in range(len(self.cycle_candidates)) if self.cycle_vars[idx].X > 0.5
            ]
            selected_chain_keys = [
                (edge_idx, position)
                for edge_idx, position in self.valid_chain_keys
                if self.chain_vars[edge_idx, position].X > 0.5
            ]

        selected_matches = []
        selected_edges = []
        for idx in selected_cycle_indices:
            candidate = self.cycle_candidates[idx]
            selected_matches.append({"type": "cycle", "nodes": candidate["nodes"], "edges": candidate["edges"]})
            selected_edges.extend(candidate["edges"])
        selected_matches.extend(decode_chain_matches(selected_chain_keys, self.src, self.dst, self.max_chain))
        selected_edges.extend(edge_idx for edge_idx, _ in selected_chain_keys)

        result = {
            "status": self.model.status,
            "objective": float(self.model.ObjVal) if self.model.SolCount > 0 else 0.0,
            "edge_selection": edge_selection_array(self.num_edges, selected_edges),
            "matches": selected_matches,
        }
        if id_map_rev is not None:
            result["formatted_matches"] = format_matches(selected_matches, id_map_rev, weights)
        return result

    def dispose(self):
        self.model.dispose()


def solve_cf_cycle_pief_chain(
    weights,
    edge_index,
    is_ndd_mask,
    num_nodes,
    cycle_candidates,
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

    chain_incoming = {}
    chain_outgoing = {}
    for edge_idx, position in valid_chain_keys:
        chain_incoming.setdefault((int(dst[edge_idx]), position), []).append((edge_idx, position))
        chain_outgoing.setdefault((int(src[edge_idx]), position), []).append((edge_idx, position))

    model = gp.Model("KEP_CF_CYCLE_PIEF_CHAIN", env=env)
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    cycle_vars = model.addVars(len(cycle_candidates), vtype=GRB.BINARY, name="cycle")
    chain_vars = model.addVars(valid_chain_keys, vtype=GRB.BINARY, name="chain")

    cycle_weights = [sum(weights[edge_idx] for edge_idx in candidate["edges"]) for candidate in cycle_candidates]
    model.setObjective(
        gp.quicksum(cycle_weights[idx] * cycle_vars[idx] for idx in range(len(cycle_candidates)))
        + gp.quicksum(weights[edge_idx] * chain_vars[edge_idx, position] for edge_idx, position in valid_chain_keys),
        GRB.MAXIMIZE,
    )

    for pair_node in pair_nodes:
        cycle_usage = gp.quicksum(
            cycle_vars[idx]
            for idx, candidate in enumerate(cycle_candidates)
            if pair_node in candidate["nodes"]
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

    model.optimize()

    selected_cycle_indices = []
    selected_chain_keys = []
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        selected_cycle_indices = [
            idx for idx in range(len(cycle_candidates)) if cycle_vars[idx].X > 0.5
        ]
        selected_chain_keys = [
            (edge_idx, position)
            for edge_idx, position in valid_chain_keys
            if chain_vars[edge_idx, position].X > 0.5
        ]

    selected_matches = []
    selected_edges = []
    for idx in selected_cycle_indices:
        candidate = cycle_candidates[idx]
        selected_matches.append({"type": "cycle", "nodes": candidate["nodes"], "edges": candidate["edges"]})
        selected_edges.extend(candidate["edges"])
    selected_matches.extend(decode_chain_matches(selected_chain_keys, src, dst, max_chain))
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
