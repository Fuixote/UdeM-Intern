# -*- coding: utf-8 -*-
"""Compare rebuilding a Gurobi LP model with reusing fixed constraints.

This file was converted from a Colab notebook export into a regular Python
script. Install dependencies outside the script, for example:

    python3 -m pip install gurobipy matplotlib
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: gurobipy. Install it with "
        "`python3 -m pip install gurobipy`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve random LP instances twice: rebuilding the model each time "
            "and reusing one model while changing objective coefficients."
        )
    )
    parser.add_argument("--constraints", type=int, default=200)
    parser.add_argument("--variables", type=int, default=500)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("untitled56_times.png"),
        help="Path for the runtime comparison plot. Use --no-plot to skip.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the computation-time plot.",
    )
    parser.add_argument(
        "--wls-access-id",
        default=os.environ.get("GRB_WLSACCESSID"),
        help="Optional Gurobi WLS access ID. Defaults to GRB_WLSACCESSID.",
    )
    parser.add_argument(
        "--wls-secret",
        default=os.environ.get("GRB_WLSSECRET"),
        help="Optional Gurobi WLS secret. Defaults to GRB_WLSSECRET.",
    )
    parser.add_argument(
        "--wls-license-id",
        type=int,
        default=(
            int(os.environ["GRB_LICENSEID"])
            if os.environ.get("GRB_LICENSEID")
            else None
        ),
        help="Optional Gurobi WLS license ID. Defaults to GRB_LICENSEID.",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> gp.Env | None:
    wls_values = [args.wls_access_id, args.wls_secret, args.wls_license_id]
    if not any(wls_values):
        return None

    if not all(wls_values):
        raise ValueError(
            "WLS credentials are incomplete. Provide --wls-access-id, "
            "--wls-secret, and --wls-license-id together."
        )

    params = {
        "WLSACCESSID": args.wls_access_id,
        "WLSSECRET": args.wls_secret,
        "LICENSEID": args.wls_license_id,
    }
    return gp.Env(params=params)


def create_random_instances(
    m_constraints: int,
    n_variables: int,
    n_instances: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rng = np.random.RandomState(seed)

    a_matrix = rng.rand(m_constraints, n_variables)
    rhs = 50 + 50 * rng.rand(m_constraints)
    cost_vectors = [rng.rand(n_variables) for _ in range(n_instances)]

    return a_matrix, rhs, cost_vectors


def new_model(env: gp.Env | None) -> gp.Model:
    model = gp.Model(env=env) if env is not None else gp.Model()
    model.Params.OutputFlag = 0
    return model


def add_fixed_region(
    model: gp.Model,
    a_matrix: np.ndarray,
    rhs: np.ndarray,
) -> gp.tupledict:
    m_constraints, n_variables = a_matrix.shape
    variables = model.addVars(n_variables, lb=0, name="x")

    for i in range(m_constraints):
        model.addConstr(
            gp.quicksum(a_matrix[i, j] * variables[j] for j in range(n_variables))
            <= rhs[i],
            name=f"constr_{i}",
        )

    return variables


def solve_by_rebuilding(
    a_matrix: np.ndarray,
    rhs: np.ndarray,
    cost_vectors: Iterable[np.ndarray],
    env: gp.Env | None = None,
) -> tuple[list[float], list[float | None]]:
    times = []
    obj_values = []
    _, n_variables = a_matrix.shape

    for costs in cost_vectors:
        start = time.perf_counter()

        model = new_model(env)
        variables = add_fixed_region(model, a_matrix, rhs)
        model.setObjective(
            gp.quicksum(costs[j] * variables[j] for j in range(n_variables)),
            GRB.MINIMIZE,
        )
        model.optimize()

        times.append(time.perf_counter() - start)
        obj_values.append(model.objVal if model.status == GRB.OPTIMAL else None)
        model.dispose()

    return times, obj_values


def solve_by_reusing(
    a_matrix: np.ndarray,
    rhs: np.ndarray,
    cost_vectors: Iterable[np.ndarray],
    env: gp.Env | None = None,
) -> tuple[list[float], list[float | None]]:
    times = []
    obj_values = []
    _, n_variables = a_matrix.shape

    model = new_model(env)
    variables = add_fixed_region(model, a_matrix, rhs)
    model.ModelSense = GRB.MINIMIZE
    model.update()

    for costs in cost_vectors:
        start = time.perf_counter()

        for j in range(n_variables):
            variables[j].Obj = costs[j]

        model.optimize()

        times.append(time.perf_counter() - start)
        obj_values.append(model.objVal if model.status == GRB.OPTIMAL else None)

    model.dispose()
    return times, obj_values


def max_objective_difference(
    left_values: Iterable[float | None],
    right_values: Iterable[float | None],
) -> float:
    return max(
        (
            abs(left - right)
            for left, right in zip(left_values, right_values)
            if left is not None and right is not None
        ),
        default=0.0,
    )


def print_results(
    rebuild_times: list[float],
    reuse_times: list[float],
    rebuild_objs: list[float | None],
    reuse_objs: list[float | None],
) -> None:
    total_rebuild_time = sum(rebuild_times)
    total_reuse_time = sum(reuse_times)
    avg_rebuild_time = np.mean(rebuild_times)
    avg_reuse_time = np.mean(reuse_times)
    speedup = total_rebuild_time / total_reuse_time if total_reuse_time else float("inf")

    print("===== Computation Time Comparison =====")
    print(f"Instances    : {len(rebuild_times)}")
    print()
    print("Approach 1: Rebuild model each time")
    print(f"Total time   : {total_rebuild_time:.4f} seconds")
    print(f"Average time : {avg_rebuild_time:.4f} seconds")
    print()
    print("Approach 2: Reuse fixed constraints")
    print(f"Total time   : {total_reuse_time:.4f} seconds")
    print(f"Average time : {avg_reuse_time:.4f} seconds")
    print()
    print("Speedup")
    print(f"Reusing is approximately {speedup:.2f}x faster")
    print(
        "Maximum objective difference: "
        f"{max_objective_difference(rebuild_objs, reuse_objs):.6e}"
    )


def write_plot(
    rebuild_times: list[float],
    reuse_times: list[float],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(rebuild_times, marker="o", label="Rebuild model")
    plt.plot(reuse_times, marker="s", label="Reuse model")
    plt.xlabel("Instance number")
    plt.ylabel("Time in seconds")
    plt.title("Computation time comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Plot saved to: {output_path}")


def main() -> int:
    args = parse_args()
    env = make_env(args)

    a_matrix, rhs, cost_vectors = create_random_instances(
        args.constraints,
        args.variables,
        args.instances,
        args.seed,
    )

    rebuild_times, rebuild_objs = solve_by_rebuilding(
        a_matrix,
        rhs,
        cost_vectors,
        env=env,
    )
    reuse_times, reuse_objs = solve_by_reusing(
        a_matrix,
        rhs,
        cost_vectors,
        env=env,
    )

    print_results(rebuild_times, reuse_times, rebuild_objs, reuse_objs)

    if not args.no_plot:
        write_plot(rebuild_times, reuse_times, args.plot)

    if env is not None:
        env.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
