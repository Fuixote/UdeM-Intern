import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import benchmark_parallel_cached_samples as bench


class ParallelCachedSamplesBenchmarkTest(unittest.TestCase):
    def test_split_indexed_items_assigns_round_robin_chunks(self):
        chunks = bench.split_indexed_items(list(range(10)), num_chunks=4)

        self.assertEqual(chunks[0], [(0, 0), (4, 4), (8, 8)])
        self.assertEqual(chunks[1], [(1, 1), (5, 5), (9, 9)])
        self.assertEqual(chunks[2], [(2, 2), (6, 6)])
        self.assertEqual(chunks[3], [(3, 3), (7, 7)])

    def test_compare_results_tracks_objective_and_selection_mismatches(self):
        serial = [
            {"objective": 1.0, "edge_selection": np.array([1, 0, 0])},
            {"objective": 2.0, "edge_selection": np.array([0, 1, 0])},
        ]
        parallel = [
            {"objective": 1.0, "edge_selection": np.array([1, 0, 0])},
            {"objective": 2.5, "edge_selection": np.array([0, 0, 1])},
        ]

        comparison = bench.compare_results(serial, parallel, tolerance=1e-6)

        self.assertEqual(comparison["max_obj_diff"], 0.5)
        self.assertEqual(comparison["obj_mismatches"], 1)
        self.assertEqual(comparison["selection_mismatches"], 1)

    def test_format_graph_header_uses_short_graph_name(self):
        row = {
            "graph": "G-0.json",
            "num_nodes": 53,
            "num_edges": 235,
            "num_cycles": 24,
            "m_samples": 16,
            "serial_build_time": 0.01234,
            "serial_solve_time": 0.45678,
            "serial_total_time": 0.46912,
        }

        line = bench.format_graph_header(row)

        self.assertEqual(
            line,
            "Graph G-0.json | samples=16 | nodes=53 | edges=235 | cycles=24",
        )

    def test_format_serial_baseline_line_highlights_baseline_times(self):
        row = {
            "serial_build_time": 0.01234,
            "serial_solve_time": 0.45678,
            "serial_total_time": 0.46912,
        }

        line = bench.format_serial_baseline_line(row)

        self.assertEqual(
            line,
            "  serial baseline | total=0.4691s | solve=0.4568s | build=0.0123s",
        )

    def test_format_speedup_table_header_names_live_multiplier_columns(self):
        self.assertEqual(
            bench.format_speedup_table_header(),
            "  copies | solve x | total x | parallel solve | build    | check",
        )

    def test_format_speedup_table_row_puts_multipliers_first(self):
        row = {
            "parallel_copies": 4,
            "parallel_build_time": 0.2,
            "parallel_solve_wall_time": 0.25,
            "speedup_solve_only": 3.2,
            "speedup_including_build": 2.0,
            "obj_mismatches": 0,
            "selection_mismatches": 0,
        }

        line = bench.format_speedup_table_row(row)

        self.assertEqual(
            line,
            "       4 |   3.20x |   2.00x |        0.2500s | 0.2000s | ok",
        )

    def test_format_speedup_table_row_marks_mismatches(self):
        row = {
            "parallel_copies": 8,
            "parallel_build_time": 0.0918,
            "parallel_solve_wall_time": 0.0258,
            "speedup_solve_only": 3.17,
            "speedup_including_build": 0.76,
            "obj_mismatches": 1,
            "selection_mismatches": 2,
        }

        line = bench.format_speedup_table_row(row)

        self.assertTrue(line.endswith("mismatch obj=1 sel=2"))

    def test_format_speedup_line_remains_machine_readable(self):
        row = {
            "graph": "G-0.json",
            "m_samples": 16,
            "parallel_copies": 4,
            "serial_solve_time": 0.8,
            "serial_total_time": 0.9,
            "parallel_build_time": 0.2,
            "parallel_solve_wall_time": 0.25,
            "parallel_total_time": 0.45,
            "speedup_solve_only": 3.2,
            "speedup_including_build": 2.0,
            "max_obj_diff": 0.0,
            "obj_mismatches": 0,
            "selection_mismatches": 0,
        }

        line = bench.format_speedup_line(row)

        self.assertTrue(
            line.startswith("SPEEDUP | graph=G-0.json | samples=16 | copies=4 | solve=3.20x | total=2.00x")
        )
        self.assertIn("samples=16", line)
        self.assertIn("parallel_solve=0.2500s", line)
        self.assertIn("parallel_build=0.2000s", line)
        self.assertIn("mismatches(obj=0, selection=0)", line)

    def test_format_speedup_line_handles_missing_sample_count(self):
        row = {
            "graph": "G-0.json",
            "parallel_copies": 4,
            "serial_solve_time": 0.8,
            "serial_total_time": 0.9,
            "parallel_build_time": 0.2,
            "parallel_solve_wall_time": 0.25,
            "parallel_total_time": 0.45,
            "speedup_solve_only": 3.2,
            "speedup_including_build": 2.0,
            "max_obj_diff": 0.0,
            "obj_mismatches": 0,
            "selection_mismatches": 0,
        }

        line = bench.format_speedup_line(row)

        self.assertTrue(line.startswith("SPEEDUP | graph=G-0.json | copies=4 | solve=3.20x | total=2.00x"))
        self.assertNotIn("samples=", line)

    def test_select_graph_paths_defaults_to_multiple_graphs_from_start_index(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["G-0.json", "G-1.json", "G-2.json", "G-3.json"]:
                (root / name).write_text("{}", encoding="utf-8")

            paths = bench.select_graph_paths(
                root,
                graph_name=None,
                graph_index=1,
                num_graphs=2,
            )

            self.assertEqual([path.name for path in paths], ["G-1.json", "G-2.json"])

    def test_select_graph_paths_graph_name_overrides_num_graphs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["G-0.json", "G-1.json", "G-2.json"]:
                (root / name).write_text("{}", encoding="utf-8")

            paths = bench.select_graph_paths(
                root,
                graph_name="G-2.json",
                graph_index=0,
                num_graphs=2,
            )

            self.assertEqual([path.name for path in paths], ["G-2.json"])


if __name__ == "__main__":
    unittest.main()
