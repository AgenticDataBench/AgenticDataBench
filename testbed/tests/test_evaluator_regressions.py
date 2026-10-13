"""Regression coverage for numeric tolerance and evaluator path binding.

Windows path tests simulate path construction; they do not replace a native
Windows run. CSV comparisons and the native-path integration test use the
actual evaluator and metric implementations.
"""
import csv
import json
import ntpath
from pathlib import Path, PureWindowsPath
import sys
import tempfile
import types
import unittest
from unittest import mock


TESTBED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTBED_ROOT.parent))
sys.path.insert(0, str(TESTBED_ROOT))

from da_agent.evaluators import evaluation  # noqa: E402
from da_agent.evaluators.metrics.table import compare_csv  # noqa: E402


class CsvToleranceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def score(self, gold, output, **options):
        for name, values in [("gold", gold), ("output", output)]:
            with (self.root / f"{name}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["value"])
                writer.writerows([[value] for value in values])
        return compare_csv(str(self.root / "output.csv"), str(self.root / "gold.csv"), **options)["score"]

    def test_absolute_tolerance_across_rounding_boundary(self):
        self.assertEqual(self.score([1.004], [1.006]), 1)

    def test_relative_tolerance_across_rounding_boundary(self):
        for sign in [1, -1]:
            for thresholds in [{0: None}, {"value": 0.01}]:
                with self.subTest(sign=sign, thresholds=thresholds):
                    self.assertEqual(self.score([sign * 100], [sign * 100.8], thresholds=thresholds), 1)

    def test_values_outside_tolerance_are_rejected(self):
        self.assertEqual(self.score([1], [1.02]), 0)
        self.assertEqual(self.score([100], [102], thresholds={0: 0.01}), 0)

    def test_ordered_relative_comparison_rejects_one_sided_nan(self):
        for gold, output in [([1, 2], [None, 2]), ([None, 2], [1, 2])]:
            with self.subTest(gold=gold, output=output):
                self.assertEqual(self.score(gold, output, ignore_order=False, thresholds={0: 0.01}), 0)

    def test_matching_missing_values_are_preserved(self):
        for ignore_order in [False, True]:
            with self.subTest(ignore_order=ignore_order):
                self.assertEqual(self.score([None, 2], [None, 2], ignore_order=ignore_order,
                                            thresholds={0: 0.01}), 1)

    def test_ordered_and_unordered_comparisons_remain_distinct(self):
        self.assertEqual(self.score([1, 2], [2, 1], ignore_order=True), 1)
        self.assertEqual(self.score([1, 2], [2, 1], ignore_order=False), 0)

    def test_string_normalization_is_preserved(self):
        self.assertEqual(self.score([" A ", "B"], ["b", "a"]), 1)

    def test_zero_relative_tolerance_requires_equal_values(self):
        self.assertEqual(self.score([100], [100], thresholds={0: 0}), 1)
        self.assertEqual(self.score([100], [100.0001], thresholds={0: 0}), 0)


class EvaluatorPathTests(unittest.TestCase):
    def bind_windows(self, expression, output_name="output.csv"):
        evaluator = evaluation.Evaluator(r"C:\test\new\output", r"C:\gold\tables")
        path_api = types.SimpleNamespace(join=ntpath.join, basename=ntpath.basename, exists=lambda _: True)
        task = {"id": "demo", "domain": "tests", "gold_file_name": "result.csv",
                "output_file_name": output_name, "eval_func": expression}
        with mock.patch.object(evaluation, "Path", PureWindowsPath), \
                mock.patch.object(evaluation, "os", types.SimpleNamespace(path=path_api)), \
                mock.patch.object(evaluator, "_get_trajectory_info_from_json", return_value={}):
            _, exists, _, (_, expressions) = evaluator._get_eval_config_info(task)
        self.assertTrue(exists)
        return eval(expressions[0], {"capture": lambda *args: args})

    def test_windows_backslashes_do_not_become_template_or_literal_escapes(self):
        self.assertEqual(self.bind_windows("capture('output.csv', 'result.csv')"),
                         ("C:/test/new/output/demo/output.csv", "C:/gold/tables/demo/result.csv"))

    def test_filenames_are_matched_literally(self):
        self.assertEqual(self.bind_windows("capture('output[1].csv', 'resultXcsv')", "output[1].csv"),
                         ("C:/test/new/output/demo/output[1].csv", "resultXcsv"))

    def test_repeated_placeholders_still_raise(self):
        with self.assertRaisesRegex(ValueError, "Found multiple matches for output.csv"):
            self.bind_windows("capture('output.csv', 'output.csv')")

    def test_native_paths_execute_the_csv_metric_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "path with spaces"
            output, gold = root / "outputs/demo", root / "gold/demo"
            (output / "dabench").mkdir(parents=True)
            gold.mkdir(parents=True)
            (output / "output.csv").write_text("value\n1.006\n", encoding="utf-8")
            (gold / "result.csv").write_text("value\n1.004\n", encoding="utf-8")
            result = {"trajectory": [], "finished": True, "steps": 0, "result": "output.csv",
                      "result_files": {"added_files": ["output.csv"], "changed_files": []}}
            (output / "dabench/result.json").write_text(json.dumps(result), encoding="utf-8")
            task = {"id": "demo", "domain": "tests", "gold_file_name": "result.csv",
                    "output_file_name": "output.csv", "eval_func": "compare_csv('output.csv', 'result.csv')"}
            config = root / "task.json"
            config.write_text(json.dumps([task]), encoding="utf-8")
            evaluator = evaluation.Evaluator(str(output.parent), str(gold.parent))
            self.assertEqual(evaluator.evaluate(str(config))[0]["total_score"], 1)


if __name__ == "__main__":
    unittest.main()
