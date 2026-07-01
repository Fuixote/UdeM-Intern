#!/usr/bin/env python3
"""Build pure-topology structural atlas artifacts for K18-E1 sentinels."""

from __future__ import annotations

import argparse
from pathlib import Path

import step4_topology_common as common


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-ids", nargs="+")
    parser.add_argument("--use-k18-all", action="store_true")
    parser.add_argument("--template-root", type=Path, default=common.DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--k18-topologies", type=Path, default=common.DEFAULT_K18_TOPOLOGIES)
    parser.add_argument("--results-dir", type=Path, default=common.STRUCTURAL_DIR / "results")
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=common.STRUCTURAL_DIR / "visualizations",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    topology_ids = common.selected_topology_ids(args.topology_ids, use_k18_all=args.use_k18_all)
    k18_rows = common.topology_rows_by_id(args.k18_topologies)

    summary_rows = []
    arc_rows = []
    candidate_rows = []
    conflict_rows = []

    for topology_id in topology_ids:
        template = common.load_template(args.template_root, topology_id)
        summary_rows.append(
            common.topology_summary_row(template, k18_row=k18_rows.get(topology_id, {}))
        )
        current_arcs = common.arc_rows_from_template(template)
        current_candidates = common.candidate_rows_from_template(template)
        current_conflicts = common.conflict_rows_from_candidates(current_candidates)
        arc_rows.extend(current_arcs)
        candidate_rows.extend(current_candidates)
        conflict_rows.extend(current_conflicts)

        viz_dir = args.visualization_dir / topology_id
        viz_dir.mkdir(parents=True, exist_ok=True)
        (viz_dir / "compatibility_graph.svg").write_text(
            common.compatibility_graph_svg(
                topology_id,
                template.get("vertices", []),
                current_arcs,
            ),
            encoding="utf-8",
        )
        (viz_dir / "candidate_conflict_graph.svg").write_text(
            common.candidate_conflict_svg(topology_id, current_candidates, current_conflicts),
            encoding="utf-8",
        )

    common.write_csv(args.results_dir / "topology_summary.csv", summary_rows, common.TOPOLOGY_SUMMARY_FIELDS)
    common.write_csv(args.results_dir / "compatibility_arcs.csv", arc_rows, common.ARC_FIELDS)
    common.write_csv(args.results_dir / "feasible_candidates.csv", candidate_rows, common.CANDIDATE_FIELDS)
    common.write_csv(args.results_dir / "candidate_conflicts.csv", conflict_rows, common.CONFLICT_FIELDS)

    print(f"Saved topology summary rows: {len(summary_rows)}")
    print(f"Saved compatibility arc rows: {len(arc_rows)}")
    print(f"Saved feasible candidate rows: {len(candidate_rows)}")
    print(f"Saved candidate conflict rows: {len(conflict_rows)}")
    print(f"Visualizations: {args.visualization_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
