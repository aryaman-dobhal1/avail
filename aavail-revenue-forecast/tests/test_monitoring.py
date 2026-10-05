import unittest

import numpy as np
import pandas as pd

from monitoring import OODDetector, check_degradation, evaluate_logged_predictions, feature_drift


class OODTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.train = pd.DataFrame(rng.normal(0, 1, (500, 4)), columns=list("abcd"))
        self.detector = OODDetector().fit(self.train)

    def test_typical_row_is_not_novel(self):
        self.assertFalse(self.detector.is_novel(pd.DataFrame([[0, 0, 0, 0]], columns=list("abcd")))[0])

    def test_extreme_row_is_novel(self):
        self.assertTrue(self.detector.is_novel(pd.DataFrame([[40, -40, 40, 40]], columns=list("abcd")))[0])

    def test_extreme_row_scores_lower(self):
        normal = self.detector.score(pd.DataFrame([[0, 0, 0, 0]]))[0]
        odd = self.detector.score(pd.DataFrame([[40, 40, 40, 40]]))[0]
        self.assertLess(odd, normal)


class DriftTests(unittest.TestCase):
    def test_shifted_column_is_flagged(self):
        rng = np.random.default_rng(1)
        ref = pd.DataFrame({"same": rng.normal(0, 1, 400), "moved": rng.normal(0, 1, 400)})
        cur = pd.DataFrame({"same": rng.normal(0, 1, 400), "moved": rng.normal(2, 1, 400)})
        out = feature_drift(ref, cur).set_index("feature")
        self.assertTrue(out.loc["moved", "drifted"])
        self.assertFalse(out.loc["same", "drifted"])


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        hist = pd.DataFrame({"date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-03"]),
                             "target": [100.0, 200.0, np.nan]})
        self.bundles = {"all": {"history": hist, "cv_rmse": 10.0}}

    def record(self, as_of, y_pred, endpoint="predict", status="ok"):
        return {"endpoint": endpoint, "status": status,
                "prediction": {"country": "all", "as_of_date": as_of, "y_pred": y_pred}}

    def test_scores_only_realised_predictions(self):
        records = [self.record("2018-01-01", 110.0), self.record("2018-01-02", 190.0),
                   self.record("2018-01-03", 999.0),
                   self.record("2018-01-01", 0.0, status="error"),
                   self.record("2018-01-01", 0.0, endpoint="train")]
        report = evaluate_logged_predictions(records, self.bundles)
        self.assertEqual(int(report["n"].iloc[0]), 2)
        self.assertAlmostEqual(report["rmse"].iloc[0], 10.0)
        self.assertAlmostEqual(report["mae"].iloc[0], 10.0)

    def test_empty_log_gives_empty_report(self):
        self.assertTrue(evaluate_logged_predictions([], self.bundles).empty)

    def test_degradation_flag(self):
        report = pd.DataFrame({"country": ["all"], "n": [2], "rmse": [20.0], "mae": [20.0]})
        self.assertTrue(check_degradation(report, self.bundles, 0.25)["degraded"].iloc[0])
        report["rmse"] = 11.0
        self.assertFalse(check_degradation(report, self.bundles, 0.25)["degraded"].iloc[0])


if __name__ == "__main__":
    unittest.main()
