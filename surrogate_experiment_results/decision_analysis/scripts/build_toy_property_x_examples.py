#!/usr/bin/env python3
"""Build small toy examples for the KEP-vs-shortest-path intuition.

Purpose
-------
This script creates paper-ready toy examples illustrating the current hypothesis:

  For packing-style combinatorial problems with close substitute solutions,
  2stage can select a solution with different identity from the oracle while
  still having small objective regret. Therefore SPO+/FY may have limited room
  to improve.

The script contrasts this with a shortest-path toy example where a prediction
error redirects the whole path and creates much larger regret.

Outputs
-------
surrogate_experiment_results/decision_analysis/results/toy_examples/
  toy_kep_packing_solutions.csv
  toy_stable_set_solutions.csv
  toy_weighted_matching_solutions.csv
  toy_knapsack_capacity_solutions.csv
  toy_partition_matroid_solutions.csv
  toy_shortest_path_solutions.csv
  toy_serial_path_solutions.csv
  toy_parametric_epsilon_solutions.csv
  toy_policy_summary.csv
  toy_summary_for_paper.tex
  toy_explanation.md

surrogate_experiment_results/decision_analysis/plots/toy_examples/
  toy_regret_comparison.png
  toy_parametric_epsilon_curve.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "toy_examples"
)

DEFAULT_PLOT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "plots"
    / "toy_examples"
)


@dataclass(frozen=True)
class PackingItem:
    """An item/component in a maximization packing problem.

    For KEP, each item represents a cycle and resources are patient-donor pair
    vertices used by that cycle.

    For stable set, each item represents a graph vertex; feasibility is handled
    separately through a conflict graph.
    """

    name: str
    true_value: float
    pred_2stage: float
    pred_dfl: float
    resources: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PathOption:
    """A candidate s-t path in a shortest-path minimization problem."""

    name: str
    true_cost: float
    pred_2stage_cost: float
    pred_dfl_cost: float
    arcs: tuple[str, ...]


MAX_DETAIL_FIELDS = [
    "example",
    "problem_family",
    "objective_sense",
    "solution",
    "true_value",
    "predicted_value_2stage",
    "predicted_value_dfl",
    "rank_by_true_value",
    "rank_by_2stage_prediction",
    "rank_by_dfl_prediction",
    "normalized_gap_to_oracle",
]

PATH_DETAIL_FIELDS = [
    "example",
    "problem_family",
    "objective_sense",
    "solution",
    "true_cost",
    "predicted_cost_2stage",
    "predicted_cost_dfl",
    "rank_by_true_cost",
    "rank_by_2stage_prediction",
    "rank_by_dfl_prediction",
    "normalized_gap_to_oracle",
]

SUMMARY_FIELDS = [
    "example",
    "problem_family",
    "objective_sense",
    "selected_by",
    "solution",
    "oracle_true_objective_or_cost",
    "selected_true_objective_or_cost",
    "selected_predicted_2stage_score_or_cost",
    "selected_predicted_dfl_score_or_cost",
    "normalized_gap_to_oracle",
    "normalized_gap_percent",
    "paper_takeaway",
]

REGRET_PLOT_EXAMPLES = [
    ("toy_kep_packing", "KEP /\npacking"),
    ("toy_stable_set", "Stable\nset"),
    ("toy_weighted_matching", "Weighted\nmatching"),
    ("toy_knapsack_capacity", "Knapsack /\ncapacity"),
    ("toy_partition_matroid", "Partition\nmatroid"),
    ("toy_parametric_epsilon_0p001", "Parametric\nepsilon"),
    ("toy_shortest_path", "Shortest\npath"),
    ("toy_serial_path", "Serial\npath"),
]


def ensure_dirs(result_dir: Path, plot_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def solution_name(items: Iterable[PackingItem]) -> str:
    names = sorted(item.name for item in items)
    return " + ".join(names) if names else "empty"


def resource_feasible(items: tuple[PackingItem, ...]) -> bool:
    used: set[str] = set()
    for item in items:
        if used.intersection(item.resources):
            return False
        used.update(item.resources)
    return True


def enumerate_resource_packings(items: list[PackingItem]) -> list[tuple[PackingItem, ...]]:
    """Enumerate non-empty resource-disjoint packings."""
    solutions: list[tuple[PackingItem, ...]] = []
    for r in range(1, len(items) + 1):
        for candidate in combinations(items, r):
            if resource_feasible(candidate):
                solutions.append(candidate)
    return solutions


def enumerate_cardinality_selections(
    items: list[PackingItem],
    max_items: int,
) -> list[tuple[PackingItem, ...]]:
    """Enumerate non-empty selections with a cardinality/capacity limit."""
    solutions: list[tuple[PackingItem, ...]] = []
    for r in range(1, max_items + 1):
        for candidate in combinations(items, r):
            solutions.append(candidate)
    return solutions


def stable_set_feasible(
    items: tuple[PackingItem, ...],
    conflicts: set[tuple[str, str]],
) -> bool:
    selected = sorted(item.name for item in items)
    for left, right in combinations(selected, 2):
        edge = tuple(sorted((left, right)))
        if edge in conflicts:
            return False
    return True


def enumerate_stable_sets(
    items: list[PackingItem],
    conflicts: set[tuple[str, str]],
) -> list[tuple[PackingItem, ...]]:
    """Enumerate non-empty stable sets."""
    normalized_conflicts = {tuple(sorted(edge)) for edge in conflicts}
    solutions: list[tuple[PackingItem, ...]] = []
    for r in range(1, len(items) + 1):
        for candidate in combinations(items, r):
            if stable_set_feasible(candidate, normalized_conflicts):
                solutions.append(candidate)
    return solutions


def sum_true_value(solution: tuple[PackingItem, ...]) -> float:
    return float(sum(item.true_value for item in solution))


def sum_pred_2stage(solution: tuple[PackingItem, ...]) -> float:
    return float(sum(item.pred_2stage for item in solution))


def sum_pred_dfl(solution: tuple[PackingItem, ...]) -> float:
    return float(sum(item.pred_dfl for item in solution))


def normalized_gap_maximization(oracle_value: float, selected_value: float) -> float:
    if abs(oracle_value) < 1e-12:
        return 0.0
    return float((oracle_value - selected_value) / abs(oracle_value))


def normalized_gap_minimization(oracle_cost: float, selected_cost: float) -> float:
    if abs(oracle_cost) < 1e-12:
        return 0.0
    return float((selected_cost - oracle_cost) / abs(oracle_cost))


def sort_solutions_max(
    solutions: list[tuple[PackingItem, ...]],
    score_fn: Callable[[tuple[PackingItem, ...]], float],
) -> list[tuple[PackingItem, ...]]:
    return sorted(
        solutions,
        key=lambda sol: (score_fn(sol), solution_name(sol)),
        reverse=True,
    )


def build_maximization_rows(
    *,
    example: str,
    problem_family: str,
    solutions: list[tuple[PackingItem, ...]],
    takeaway: str,
) -> tuple[list[dict], list[dict]]:
    """Return detailed solution rows and policy summary rows for maximization."""
    ranked_true = sort_solutions_max(solutions, sum_true_value)
    ranked_2stage = sort_solutions_max(solutions, sum_pred_2stage)
    ranked_dfl = sort_solutions_max(solutions, sum_pred_dfl)

    oracle_solution = ranked_true[0]
    true_second_best = ranked_true[1]
    two_stage_solution = ranked_2stage[0]
    two_stage_second = ranked_2stage[1]
    dfl_solution = ranked_dfl[0]
    dfl_second = ranked_dfl[1]

    oracle_value = sum_true_value(oracle_solution)

    detailed_rows: list[dict] = []
    for solution in solutions:
        detailed_rows.append(
            {
                "example": example,
                "problem_family": problem_family,
                "objective_sense": "maximize",
                "solution": solution_name(solution),
                "true_value": sum_true_value(solution),
                "predicted_value_2stage": sum_pred_2stage(solution),
                "predicted_value_dfl": sum_pred_dfl(solution),
                "rank_by_true_value": ranked_true.index(solution) + 1,
                "rank_by_2stage_prediction": ranked_2stage.index(solution) + 1,
                "rank_by_dfl_prediction": ranked_dfl.index(solution) + 1,
                "normalized_gap_to_oracle": normalized_gap_maximization(
                    oracle_value, sum_true_value(solution)
                ),
            }
        )

    summary_specs = [
        ("oracle_by_true", oracle_solution),
        ("true_second_best", true_second_best),
        ("2stage_rank1_by_prediction", two_stage_solution),
        ("2stage_rank2_by_prediction", two_stage_second),
        ("dfl_rank1_by_prediction", dfl_solution),
        ("dfl_rank2_by_prediction", dfl_second),
    ]

    summary_rows: list[dict] = []
    for selected_by, solution in summary_specs:
        gap = normalized_gap_maximization(oracle_value, sum_true_value(solution))
        summary_rows.append(
            {
                "example": example,
                "problem_family": problem_family,
                "objective_sense": "maximize",
                "selected_by": selected_by,
                "solution": solution_name(solution),
                "oracle_true_objective_or_cost": oracle_value,
                "selected_true_objective_or_cost": sum_true_value(solution),
                "selected_predicted_2stage_score_or_cost": sum_pred_2stage(solution),
                "selected_predicted_dfl_score_or_cost": sum_pred_dfl(solution),
                "normalized_gap_to_oracle": gap,
                "normalized_gap_percent": 100.0 * gap,
                "paper_takeaway": takeaway,
            }
        )

    return detailed_rows, summary_rows


def build_kep_packing_example() -> tuple[list[dict], list[dict]]:
    """Tiny KEP-like set-packing example.

    Four patient-donor pair vertices: 1,2,3,4.

    Oracle uses cycles C12 and C34, total true value 20.0.
    A close alternative uses C13 and C24, total true value 19.8.

    2stage slightly overestimates C13 and C24, so it selects the alternative.
    The solution identity changes, but regret is only 1%.
    """
    cycles = [
        PackingItem(
            name="C12",
            resources=frozenset({"1", "2"}),
            true_value=10.0,
            pred_2stage=10.0,
            pred_dfl=10.0,
        ),
        PackingItem(
            name="C34",
            resources=frozenset({"3", "4"}),
            true_value=10.0,
            pred_2stage=10.0,
            pred_dfl=10.0,
        ),
        PackingItem(
            name="C13",
            resources=frozenset({"1", "3"}),
            true_value=9.9,
            pred_2stage=10.1,
            pred_dfl=9.9,
        ),
        PackingItem(
            name="C24",
            resources=frozenset({"2", "4"}),
            true_value=9.9,
            pred_2stage=10.1,
            pred_dfl=9.9,
        ),
    ]
    solutions = enumerate_resource_packings(cycles)
    return build_maximization_rows(
        example="toy_kep_packing",
        problem_family="KEP / set packing",
        solutions=solutions,
        takeaway=(
            "2stage selects a different packing from the oracle, but the true "
            "objective gap is only 1% because the alternative packing is a close substitute."
        ),
    )


def build_stable_set_example() -> tuple[list[dict], list[dict]]:
    """Tiny stable-set example with the same packing-style phenomenon.

    Conflict graph:
      A and B are mutually compatible.
      C and D are mutually compatible.
      Every item in {A,B} conflicts with every item in {C,D}.

    Hence the two relevant stable sets are {A,B} and {C,D}.
    """
    items = [
        PackingItem(name="A", true_value=10.0, pred_2stage=10.0, pred_dfl=10.0),
        PackingItem(name="B", true_value=10.0, pred_2stage=10.0, pred_dfl=10.0),
        PackingItem(name="C", true_value=9.9, pred_2stage=10.1, pred_dfl=9.9),
        PackingItem(name="D", true_value=9.9, pred_2stage=10.1, pred_dfl=9.9),
    ]

    conflicts = {
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
    }

    solutions = enumerate_stable_sets(items, conflicts)
    return build_maximization_rows(
        example="toy_stable_set",
        problem_family="stable set",
        solutions=solutions,
        takeaway=(
            "The same close-substitute behavior can occur outside KEP in another "
            "packing-style combinatorial problem."
        ),
    )


def build_weighted_matching_example() -> tuple[list[dict], list[dict]]:
    """Weighted matching example with close substitute perfect matchings."""
    edges = [
        PackingItem(
            "M12",
            true_value=10.0,
            pred_2stage=10.0,
            pred_dfl=10.0,
            resources=frozenset({"1", "2"}),
        ),
        PackingItem(
            "M34",
            true_value=10.0,
            pred_2stage=10.0,
            pred_dfl=10.0,
            resources=frozenset({"3", "4"}),
        ),
        PackingItem(
            "M13",
            true_value=9.8,
            pred_2stage=10.2,
            pred_dfl=9.8,
            resources=frozenset({"1", "3"}),
        ),
        PackingItem(
            "M24",
            true_value=9.8,
            pred_2stage=10.2,
            pred_dfl=9.8,
            resources=frozenset({"2", "4"}),
        ),
    ]
    return build_maximization_rows(
        example="toy_weighted_matching",
        problem_family="weighted matching",
        solutions=enumerate_resource_packings(edges),
        takeaway=(
            "A matching can switch to a different perfect matching while losing only "
            "2% of the true objective when the alternative edges are close substitutes."
        ),
    )


def build_knapsack_capacity_example() -> tuple[list[dict], list[dict]]:
    """Cardinality-knapsack example where capacity allows two close items."""
    items = [
        PackingItem("A", true_value=10.0, pred_2stage=10.0, pred_dfl=10.0),
        PackingItem("B", true_value=10.0, pred_2stage=10.0, pred_dfl=10.0),
        PackingItem("C", true_value=9.75, pred_2stage=10.2, pred_dfl=9.75),
        PackingItem("D", true_value=9.75, pred_2stage=10.2, pred_dfl=9.75),
    ]
    return build_maximization_rows(
        example="toy_knapsack_capacity",
        problem_family="cardinality knapsack",
        solutions=enumerate_cardinality_selections(items, max_items=2),
        takeaway=(
            "Under a small capacity, 2stage can replace the oracle bundle with a "
            "nearby bundle and still incur only 2.5% normalized regret."
        ),
    )


def build_partition_matroid_example() -> tuple[list[dict], list[dict]]:
    """Partition-matroid example: select one item from each independent block."""
    items = [
        PackingItem(
            "A1",
            true_value=10.0,
            pred_2stage=10.0,
            pred_dfl=10.0,
            resources=frozenset({"block_a"}),
        ),
        PackingItem(
            "B1",
            true_value=10.0,
            pred_2stage=10.0,
            pred_dfl=10.0,
            resources=frozenset({"block_b"}),
        ),
        PackingItem(
            "A2",
            true_value=9.9,
            pred_2stage=10.15,
            pred_dfl=9.9,
            resources=frozenset({"block_a"}),
        ),
        PackingItem(
            "B2",
            true_value=9.9,
            pred_2stage=10.15,
            pred_dfl=9.9,
            resources=frozenset({"block_b"}),
        ),
    ]
    return build_maximization_rows(
        example="toy_partition_matroid",
        problem_family="partition matroid",
        solutions=enumerate_resource_packings(items),
        takeaway=(
            "A decomposable matroid base can change one item in each block while "
            "remaining within 1% of the oracle objective."
        ),
    )


def build_parametric_epsilon_family(
    epsilons: tuple[float, ...] = (0.05, 0.02, 0.01, 0.001),
) -> tuple[list[dict], list[dict]]:
    """Parametric close-substitute construction with arbitrarily small regret."""
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    for epsilon in epsilons:
        suffix = f"{epsilon:.3f}".replace(".", "p")
        example = f"toy_parametric_epsilon_{suffix}"
        solutions = [
            PackingItem(
                "S_star",
                true_value=1.0,
                pred_2stage=1.0,
                pred_dfl=1.0,
                resources=frozenset({"slot"}),
            ),
            PackingItem(
                "S_epsilon",
                true_value=1.0 - epsilon,
                pred_2stage=1.0 + epsilon,
                pred_dfl=1.0 - epsilon,
                resources=frozenset({"slot"}),
            ),
        ]
        rows, rows_summary = build_maximization_rows(
            example=example,
            problem_family="parametric packing family",
            solutions=enumerate_resource_packings(solutions),
            takeaway=(
                "For any small epsilon, a prediction-order flip can change the selected "
                "solution while keeping regret equal to epsilon."
            ),
        )
        for row in rows_summary:
            row["paper_takeaway"] = (
                "The normalized regret can be made arbitrarily small even though "
                "2stage selects S_epsilon instead of S_star."
            )
        detail_rows.extend(rows)
        summary_rows.extend(rows_summary)
    return detail_rows, summary_rows


def build_shortest_path_example() -> tuple[list[dict], list[dict]]:
    """Tiny shortest-path contrast.

    Minimization problem.

    Path A is the oracle path with true cost 5.
    Path B is the wrong path with true cost 8.

    2stage underestimates Path B, so it selects Path B.
    The regret is (8 - 5) / 5 = 60%.
    """
    paths = [
        PathOption(
            name="Path A: s-a1-a2-t",
            arcs=("s->a1", "a1->a2", "a2->t"),
            true_cost=5.0,
            pred_2stage_cost=5.0,
            pred_dfl_cost=5.0,
        ),
        PathOption(
            name="Path B: s-b1-b2-t",
            arcs=("s->b1", "b1->b2", "b2->t"),
            true_cost=8.0,
            pred_2stage_cost=4.8,
            pred_dfl_cost=8.0,
        ),
    ]

    ranked_true = sorted(paths, key=lambda path: (path.true_cost, path.name))
    ranked_2stage = sorted(paths, key=lambda path: (path.pred_2stage_cost, path.name))
    ranked_dfl = sorted(paths, key=lambda path: (path.pred_dfl_cost, path.name))

    oracle = ranked_true[0]
    true_second_best = ranked_true[1]
    two_stage_rank1 = ranked_2stage[0]
    two_stage_rank2 = ranked_2stage[1]
    dfl_rank1 = ranked_dfl[0]
    dfl_rank2 = ranked_dfl[1]

    oracle_cost = oracle.true_cost

    detailed_rows: list[dict] = []
    for path in paths:
        detailed_rows.append(
            {
                "example": "toy_shortest_path",
                "problem_family": "shortest path",
                "objective_sense": "minimize",
                "solution": path.name,
                "true_cost": path.true_cost,
                "predicted_cost_2stage": path.pred_2stage_cost,
                "predicted_cost_dfl": path.pred_dfl_cost,
                "rank_by_true_cost": ranked_true.index(path) + 1,
                "rank_by_2stage_prediction": ranked_2stage.index(path) + 1,
                "rank_by_dfl_prediction": ranked_dfl.index(path) + 1,
                "normalized_gap_to_oracle": normalized_gap_minimization(
                    oracle_cost, path.true_cost
                ),
            }
        )

    summary_specs = [
        ("oracle_by_true", oracle),
        ("true_second_best", true_second_best),
        ("2stage_rank1_by_prediction", two_stage_rank1),
        ("2stage_rank2_by_prediction", two_stage_rank2),
        ("dfl_rank1_by_prediction", dfl_rank1),
        ("dfl_rank2_by_prediction", dfl_rank2),
    ]

    takeaway = (
        "A prediction error redirects the whole path. The selected solution is not a "
        "local close substitute, so 2stage regret can be large."
    )

    summary_rows: list[dict] = []
    for selected_by, path in summary_specs:
        gap = normalized_gap_minimization(oracle_cost, path.true_cost)
        summary_rows.append(
            {
                "example": "toy_shortest_path",
                "problem_family": "shortest path",
                "objective_sense": "minimize",
                "selected_by": selected_by,
                "solution": path.name,
                "oracle_true_objective_or_cost": oracle_cost,
                "selected_true_objective_or_cost": path.true_cost,
                "selected_predicted_2stage_score_or_cost": path.pred_2stage_cost,
                "selected_predicted_dfl_score_or_cost": path.pred_dfl_cost,
                "normalized_gap_to_oracle": gap,
                "normalized_gap_percent": 100.0 * gap,
                "paper_takeaway": takeaway,
            }
        )

    return detailed_rows, summary_rows


def build_serial_path_example() -> tuple[list[dict], list[dict]]:
    """Second negative control: a coupled serial route can create large regret."""
    paths = [
        PathOption(
            name="Serial route A: low-cost chain",
            arcs=("s->a", "a->b", "b->t"),
            true_cost=10.0,
            pred_2stage_cost=10.0,
            pred_dfl_cost=10.0,
        ),
        PathOption(
            name="Serial route B: high-cost chain",
            arcs=("s->c", "c->d", "d->t"),
            true_cost=15.0,
            pred_2stage_cost=9.8,
            pred_dfl_cost=15.0,
        ),
    ]

    ranked_true = sorted(paths, key=lambda path: (path.true_cost, path.name))
    ranked_2stage = sorted(paths, key=lambda path: (path.pred_2stage_cost, path.name))
    ranked_dfl = sorted(paths, key=lambda path: (path.pred_dfl_cost, path.name))

    oracle = ranked_true[0]
    true_second_best = ranked_true[1]
    two_stage_rank1 = ranked_2stage[0]
    two_stage_rank2 = ranked_2stage[1]
    dfl_rank1 = ranked_dfl[0]
    dfl_rank2 = ranked_dfl[1]

    oracle_cost = oracle.true_cost

    detailed_rows: list[dict] = []
    for path in paths:
        detailed_rows.append(
            {
                "example": "toy_serial_path",
                "problem_family": "serial path control",
                "objective_sense": "minimize",
                "solution": path.name,
                "true_cost": path.true_cost,
                "predicted_cost_2stage": path.pred_2stage_cost,
                "predicted_cost_dfl": path.pred_dfl_cost,
                "rank_by_true_cost": ranked_true.index(path) + 1,
                "rank_by_2stage_prediction": ranked_2stage.index(path) + 1,
                "rank_by_dfl_prediction": ranked_dfl.index(path) + 1,
                "normalized_gap_to_oracle": normalized_gap_minimization(
                    oracle_cost, path.true_cost
                ),
            }
        )

    summary_specs = [
        ("oracle_by_true", oracle),
        ("true_second_best", true_second_best),
        ("2stage_rank1_by_prediction", two_stage_rank1),
        ("2stage_rank2_by_prediction", two_stage_rank2),
        ("dfl_rank1_by_prediction", dfl_rank1),
        ("dfl_rank2_by_prediction", dfl_rank2),
    ]

    takeaway = (
        "When feasibility couples all arcs into one serial decision, a small "
        "ranking error can redirect the whole route and create 50% regret."
    )

    summary_rows: list[dict] = []
    for selected_by, path in summary_specs:
        gap = normalized_gap_minimization(oracle_cost, path.true_cost)
        summary_rows.append(
            {
                "example": "toy_serial_path",
                "problem_family": "serial path control",
                "objective_sense": "minimize",
                "selected_by": selected_by,
                "solution": path.name,
                "oracle_true_objective_or_cost": oracle_cost,
                "selected_true_objective_or_cost": path.true_cost,
                "selected_predicted_2stage_score_or_cost": path.pred_2stage_cost,
                "selected_predicted_dfl_score_or_cost": path.pred_dfl_cost,
                "normalized_gap_to_oracle": gap,
                "normalized_gap_percent": 100.0 * gap,
                "paper_takeaway": takeaway,
            }
        )

    return detailed_rows, summary_rows


def build_all_toy_examples() -> tuple[dict[str, list[dict]], list[dict]]:
    """Build all toy examples and return detail tables plus one summary table."""
    detail_tables: dict[str, list[dict]] = {}
    summary_rows: list[dict] = []

    builders = [
        ("toy_kep_packing", build_kep_packing_example),
        ("toy_stable_set", build_stable_set_example),
        ("toy_weighted_matching", build_weighted_matching_example),
        ("toy_knapsack_capacity", build_knapsack_capacity_example),
        ("toy_partition_matroid", build_partition_matroid_example),
        ("toy_shortest_path", build_shortest_path_example),
        ("toy_serial_path", build_serial_path_example),
    ]
    for table_name, builder in builders:
        detail_rows, builder_summary = builder()
        detail_tables[table_name] = detail_rows
        summary_rows.extend(builder_summary)

    parametric_rows, parametric_summary = build_parametric_epsilon_family()
    detail_tables["toy_parametric_epsilon_solutions"] = parametric_rows
    summary_rows.extend(parametric_summary)

    return detail_tables, summary_rows


def latex_escape_text(value: object) -> str:
    """Escape plain text inserted into a LaTeX table cell."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex_table(path: Path, summary_rows: list[dict]) -> None:
    """Write a compact paper-ready LaTeX table."""
    table_specs = [
        ("toy_kep_packing", "KEP / set packing"),
        ("toy_stable_set", "Stable set"),
        ("toy_weighted_matching", "Weighted matching"),
        ("toy_knapsack_capacity", "Cardinality knapsack"),
        ("toy_partition_matroid", "Partition matroid"),
        ("toy_parametric_epsilon_0p001", "Parametric packing family"),
        ("toy_shortest_path", "Shortest path"),
        ("toy_serial_path", "Serial path control"),
    ]

    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Problem & Oracle / true best & 2stage selected & 2stage normalized gap \\",
        r"\midrule",
    ]
    for example, label in table_specs:
        oracle = next(
            row
            for row in summary_rows
            if row["example"] == example and row["selected_by"] == "oracle_by_true"
        )
        two_stage = next(
            row
            for row in summary_rows
            if row["example"] == example and row["selected_by"] == "2stage_rank1_by_prediction"
        )
        objective_label = "cost" if two_stage["objective_sense"] == "minimize" else "value"
        oracle_text = (
            f"{latex_escape_text(oracle['solution'])}, {objective_label} "
            f"{oracle['selected_true_objective_or_cost']:.3g}"
        )
        selected_text = (
            f"{latex_escape_text(two_stage['solution'])}, {objective_label} "
            f"{two_stage['selected_true_objective_or_cost']:.3g}"
        )
        lines.append(
            f"{latex_escape_text(label)} & {oracle_text} & {selected_text} "
            f"& {two_stage['normalized_gap_percent']:.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")


def write_markdown_explanation(path: Path) -> None:
    content = """# Toy examples for Property X

## Hypothesis

Decision-focused learning is most useful when prediction errors change the
objective value of the selected solution, not merely its identity.

For packing-style combinatorial problems with close substitute solutions,
2stage can select a different feasible solution from the oracle while still
having small objective regret. In such cases, SPO+/FY has limited room to
improve.

## Property X

Close substitute solutions in a decomposable packing structure.

A problem has this property when feasible solutions are composed of several
mostly independent components, and replacing one component with another feasible
component often changes the true objective only slightly.

## Positive packing-style examples

The oracle packing is C12 + C34 with true value 20.0. The 2stage prediction
selects C13 + C24 with true value 19.8. The solution identity changes, but the
normalized oracle gap is only 1%.

The same phenomenon appears in a small stable-set problem. The oracle stable set
is {A,B}, while 2stage selects {C,D}. The normalized oracle gap is again only 1%.

The expanded toy family adds weighted matching, cardinality-knapsack, and
partition-matroid examples. In each case, 2stage selects a different feasible
solution from the oracle, but the selected alternative is a close substitute and
the normalized regret remains small.

## Parametric epsilon family

The parametric construction has two complete feasible solutions: S_star with
true value 1 and S_epsilon with true value 1 - epsilon. The 2stage prediction
slightly overestimates S_epsilon, so it selects the wrong solution. The identity
changes, but the normalized regret is exactly epsilon and can be made
arbitrarily small.

This supports a structural, not merely numerical, version of Property X. The
claim should still be framed as a mechanism and not as a theorem covering every
packing instance.

## Path-like negative controls

The shortest-path example has two paths. The oracle path has true cost 5.0, but
2stage underestimates the wrong path and selects a path with true cost 8.0. The
normalized regret is 60%.

A second serial-path negative control has true costs 10.0 and 15.0, yielding 50%
regret when 2stage underestimates the high-cost chain.

These negative controls illustrate why shortest path can behave differently: a
prediction error can redirect the entire connected path, producing a much larger
regret.
"""
    path.write_text(content, encoding="utf-8")


def make_regret_plot(summary_rows: list[dict], plot_dir: Path) -> None:
    """Make a simple bar chart comparing 2stage and DFL selected regrets."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate the plot. "
            "Install it or rerun with --skip-plots."
        ) from exc

    two_stage_gaps = []
    dfl_gaps = []

    for example_key, _ in REGRET_PLOT_EXAMPLES:
        two_stage_row = next(
            row
            for row in summary_rows
            if row["example"] == example_key
            and row["selected_by"] == "2stage_rank1_by_prediction"
        )
        dfl_row = next(
            row
            for row in summary_rows
            if row["example"] == example_key
            and row["selected_by"] == "dfl_rank1_by_prediction"
        )
        two_stage_gaps.append(two_stage_row["normalized_gap_percent"])
        dfl_gaps.append(dfl_row["normalized_gap_percent"])

    labels = [label for _, label in REGRET_PLOT_EXAMPLES]
    x_positions = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.0))

    left_positions = [x - width / 2 for x in x_positions]
    right_positions = [x + width / 2 for x in x_positions]

    ax.bar(left_positions, two_stage_gaps, width, label="2stage selected")
    ax.bar(right_positions, dfl_gaps, width, label="DFL/SPO+ selected")

    ax.axhline(5.0, linestyle="--", linewidth=1, label="5% gap reference")

    ax.set_ylabel("Normalized oracle gap (%)")
    ax.set_title("Toy family: identity changes can have small or large regret")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.legend()

    for x, value in zip(left_positions, two_stage_gaps):
        ax.text(x, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    for x, value in zip(right_positions, dfl_gaps):
        ax.text(x, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(plot_dir / "toy_regret_comparison.png", dpi=200)
    plt.close(fig)


def make_epsilon_plot(summary_rows: list[dict], plot_dir: Path) -> None:
    """Plot the parametric construction where regret equals epsilon."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate the plot. "
            "Install it or rerun with --skip-plots."
        ) from exc

    rows = [
        row
        for row in summary_rows
        if row["example"].startswith("toy_parametric_epsilon_")
        and row["selected_by"] == "2stage_rank1_by_prediction"
    ]
    points = sorted(
        (
            row["normalized_gap_to_oracle"],
            row["normalized_gap_percent"],
        )
        for row in rows
    )
    epsilons = [point[0] for point in points]
    gap_percents = [point[1] for point in points]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epsilons, gap_percents, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("2stage normalized oracle gap (%)")
    ax.set_title("Parametric close-substitute family")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    for epsilon, gap in zip(epsilons, gap_percents):
        ax.text(epsilon, gap, f"{gap:.1f}%", ha="left", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(plot_dir / "toy_parametric_epsilon_curve.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build toy examples for the KEP / shortest-path DFL story."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs(args.result_dir, args.plot_dir)

    detail_tables, summary_rows = build_all_toy_examples()
    for table_name, detail_rows in detail_tables.items():
        output_name = (
            f"{table_name}.csv"
            if table_name.endswith("_solutions")
            else f"{table_name}_solutions.csv"
        )
        fieldnames = PATH_DETAIL_FIELDS if "true_cost" in detail_rows[0] else MAX_DETAIL_FIELDS
        write_csv(args.result_dir / output_name, detail_rows, fieldnames)

    write_csv(
        args.result_dir / "toy_policy_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )

    write_latex_table(args.result_dir / "toy_summary_for_paper.tex", summary_rows)
    write_markdown_explanation(args.result_dir / "toy_explanation.md")

    if not args.skip_plots:
        make_regret_plot(summary_rows, args.plot_dir)
        make_epsilon_plot(summary_rows, args.plot_dir)

    print(f"Saved toy example CSVs to: {args.result_dir}")
    if not args.skip_plots:
        print(f"Saved toy example plots to: {args.plot_dir}")

    key_rows = [
        row
        for row in summary_rows
        if row["selected_by"] == "2stage_rank1_by_prediction"
    ]
    print("\n2stage selected normalized gaps:")
    for row in key_rows:
        print(
            f"  {row['example']}: "
            f"{row['normalized_gap_percent']:.1f}% "
            f"({row['solution']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
