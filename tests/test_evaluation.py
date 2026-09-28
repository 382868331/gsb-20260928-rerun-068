import unittest

from mleval import (
    PredictionBatch,
    SampleSet,
    bootstrap_interval,
    evaluate_classification,
)


def samples():
    return SampleSet.from_records(
        [
            {"sample_id": "s1", "y_true": "A", "weight": 1, "groups": {"region": "north"}},
            {"sample_id": "s2", "y_true": "A", "weight": 1, "groups": {"region": "north"}},
            {"sample_id": "s3", "y_true": "B", "weight": 2, "groups": {"region": "south"}},
            {"sample_id": "s4", "y_true": "B", "weight": 3, "groups": {"region": "south"}},
            {"sample_id": "s5", "y_true": "A", "weight": 4, "groups": {"region": "south"}},
        ]
    )


class EvaluationTest(unittest.TestCase):
    def test_weighted_overall_and_unweighted_group_average(self):
        predictions = PredictionBatch.from_records(
            [
                {"sample_id": "s1", "y_pred": "A"},
                {"sample_id": "s2", "y_pred": "B"},
                {"sample_id": "s3", "y_pred": "B"},
                {"sample_id": "s4", "y_pred": "A"},
                {"sample_id": "s4", "y_pred": "B"},
            ]
        )
        result = evaluate_classification(samples(), predictions)

        self.assertEqual(result.scored_sample_ids, ("s1", "s2", "s3"))
        self.assertAlmostEqual(result.overall["accuracy"], 0.75)
        self.assertAlmostEqual(result.overall["error_rate"], 0.25)
        self.assertAlmostEqual(result.overall["weight"], 4)

        north = result.groups["region"]["north"]
        south = result.groups["region"]["south"]
        self.assertAlmostEqual(north["accuracy"], 0.5)
        self.assertAlmostEqual(south["accuracy"], 1.0)
        self.assertAlmostEqual(result.group_macro_accuracy["region"], 0.75)

        self.assertEqual(result.data_quality["missing_sample_ids"], ["s5"])
        self.assertEqual(result.data_quality["duplicate_sample_ids"], ["s4"])
        self.assertEqual(result.data_quality["missing_weight"], 4)
        self.assertEqual(result.data_quality["duplicate_weight"], 3)

    def test_unknown_label_stays_in_denominator_and_counts_wrong(self):
        predictions = PredictionBatch.from_records(
            [
                {"sample_id": "s1", "y_pred": "A"},
                {"sample_id": "s2", "y_pred": "C"},
                {"sample_id": "s3", "y_pred": "B"},
                {"sample_id": "s4", "y_pred": "B"},
                {"sample_id": "s5", "y_pred": "A"},
            ]
        )
        result = evaluate_classification(samples(), predictions)

        self.assertEqual(result.data_quality["scored_count"], 5)
        self.assertAlmostEqual(result.overall["accuracy"], 10 / 11)
        self.assertEqual(result.overall["unknown_label_count"], 1)
        self.assertEqual(result.overall["unknown_label_weight"], 1)
        self.assertEqual(result.data_quality["unknown_label_sample_ids"], ["s2"])

    def test_versions_are_stable_and_bootstrap_is_reproducible(self):
        first = SampleSet(samples().samples)
        second = SampleSet(samples().samples)
        self.assertEqual(first.version, second.version)
        predictions = PredictionBatch.from_records(
            [
                {"sample_id": "s1", "y_pred": "A"},
                {"sample_id": "s2", "y_pred": "B"},
                {"sample_id": "s3", "y_pred": "B"},
                {"sample_id": "s4", "y_pred": "B"},
                {"sample_id": "s5", "y_pred": "A"},
            ]
        )
        evaluation = evaluate_classification(samples(), predictions)
        from mleval.alignment import align_predictions

        aligned = align_predictions(samples(), predictions)
        interval_one = bootstrap_interval(
            aligned.pairs, samples().classes, iterations=100, seed=7
        )
        interval_two = bootstrap_interval(
            aligned.pairs, samples().classes, iterations=100, seed=7
        )
        self.assertEqual(interval_one.replicate_values, interval_two.replicate_values)
        self.assertEqual(interval_one.lower, interval_two.lower)
        self.assertEqual(interval_one.upper, interval_two.upper)
        self.assertEqual(interval_one.sample_size, evaluation.data_quality["scored_count"])
        self.assertEqual(
            interval_one.result_version,
            bootstrap_interval(aligned.pairs, samples().classes, iterations=100, seed=7).result_version,
        )


if __name__ == "__main__":
    unittest.main()
