import unittest

from scripts.run_affected_unit_tests import select_affected_unit_tests, unittest_command


class TestRunAffectedUnitTests(unittest.TestCase):
    def test_source_directory_changes_map_to_expected_test_files(self):
        selection = select_affected_unit_tests(
            [
                "algorithms/pairwise_matching.py",
                "data_structures/contour.py",
                "utils/geometry.py",
                "visualizations.py",
            ]
        )

        self.assertFalse(selection.run_full_suite)
        self.assertEqual(
            selection.test_paths,
            (
                "unit_tests/test_algorithms.py",
                "unit_tests/test_data_structures.py",
                "unit_tests/test_utils.py",
                "unit_tests/test_visualizations.py",
            ),
        )

    def test_multiple_changed_files_dedupe_test_paths(self):
        selection = select_affected_unit_tests(
            [
                "algorithms/pairwise_matching.py",
                "algorithms/feature_extraction.py",
                "cython_libs.pyx",
            ]
        )

        self.assertFalse(selection.run_full_suite)
        self.assertEqual(
            selection.test_paths,
            (
                "unit_tests/test_algorithms.py",
                "unit_tests/test_find_neighbors_within_radius.py",
            ),
        )

    def test_test_file_change_runs_that_exact_test_file(self):
        selection = select_affected_unit_tests(["unit_tests/test_visualizations.py"])

        self.assertFalse(selection.run_full_suite)
        self.assertEqual(selection.test_paths, ("unit_tests/test_visualizations.py",))

    def test_infrastructure_change_requests_full_suite(self):
        selection = select_affected_unit_tests(["scripts/run_affected_unit_tests.py"])

        self.assertTrue(selection.run_full_suite)
        self.assertEqual(selection.test_paths, ())
        self.assertEqual(
            unittest_command(selection)[-4:],
            ["unittest", "discover", "-s", "unit_tests"],
        )

    def test_unmapped_changes_return_no_affected_tests(self):
        selection = select_affected_unit_tests(["README.md", "configs/puzzle.yaml"])

        self.assertFalse(selection.run_full_suite)
        self.assertEqual(selection.test_paths, ())


if __name__ == "__main__":
    unittest.main()
