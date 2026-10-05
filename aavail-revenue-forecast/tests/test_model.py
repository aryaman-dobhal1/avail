import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_ingestion import clean_transactions, daily_series
from model import (FEATURE_COLUMNS, HORIZON, TrailingMeanBaseline, compare_models,
                   engineer_features, get_candidates, mae, model_load, model_predict,
                   model_train, rmse)
from tests.helpers import make_transactions, write_json_files


class MetricTests(unittest.TestCase):
    def test_rmse_known_value(self):
        self.assertAlmostEqual(rmse([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), np.sqrt(0.375))

    def test_mae_known_value(self):
        self.assertAlmostEqual(mae([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), 0.5)

    def test_rmse_punishes_one_big_miss_more_than_mae(self):
        y, pred = [0, 0, 0, 0], [0, 0, 0, 4]
        self.assertAlmostEqual(rmse(y, pred), 2.0)
        self.assertAlmostEqual(mae(y, pred), 1.0)

    def test_perfect_prediction_is_zero(self):
        self.assertEqual(rmse([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertEqual(mae([1, 2, 3], [1, 2, 3]), 0.0)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clean, _ = clean_transactions(make_transactions(n_days=300))
        cls.daily = daily_series(clean)
        cls.feats = engineer_features(cls.daily)

    def test_target_is_sum_of_next_30_days(self):
        t = 100
        expected = self.daily["revenue"].iloc[t + 1: t + 1 + HORIZON].sum()
        self.assertAlmostEqual(self.feats["target"].iloc[t], expected)

    def test_last_30_targets_are_unknown(self):
        self.assertTrue(self.feats["target"].tail(HORIZON).isna().all())
        self.assertFalse(self.feats["target"].iloc[:-HORIZON].isna().any())

    def test_features_do_not_look_ahead(self):
        changed = self.daily.copy()
        changed.loc[250:, ["revenue", "purchases", "total_views"]] *= 10
        other = engineer_features(changed)
        pd.testing.assert_frame_equal(self.feats[FEATURE_COLUMNS].iloc[:250],
                                      other[FEATURE_COLUMNS].iloc[:250])
        self.assertNotAlmostEqual(self.feats["target"].iloc[240], other["target"].iloc[240])

    def test_baseline_repeats_last_30_days(self):
        row = self.feats.dropna(subset=FEATURE_COLUMNS).iloc[[50]]
        pred = TrailingMeanBaseline().fit(row).predict(row)
        self.assertAlmostEqual(pred[0], row["rev_sum_30"].iloc[0])

    def test_compare_models_reports_every_candidate(self):
        rows = self.feats.dropna(subset=FEATURE_COLUMNS + ["target"])
        table = compare_models(rows[FEATURE_COLUMNS], rows["target"], get_candidates(quick=True))
        self.assertEqual(set(table["model"]), set(get_candidates(quick=True)))
        self.assertTrue((table["rmse"] >= table["mae"] - 1e-9).all())
        self.assertTrue(table["rmse"].is_monotonic_increasing)

    def test_too_little_data_raises(self):
        rows = self.feats.dropna(subset=FEATURE_COLUMNS + ["target"]).head(40)
        with self.assertRaises(ValueError):
            compare_models(rows[FEATURE_COLUMNS], rows["target"], get_candidates(quick=True))


class TrainPredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = tempfile.TemporaryDirectory()
        cls.models = tempfile.TemporaryDirectory()
        write_json_files(cls.data.name, make_transactions(n_days=450))
        cls.summary = model_train(cls.data.name, cls.models.name, top_n=2, quick=True)
        cls.bundles = model_load(cls.models.name)

    @classmethod
    def tearDownClass(cls):
        cls.data.cleanup()
        cls.models.cleanup()

    def test_trains_all_plus_top_countries(self):
        self.assertEqual(len(self.summary["models"]), 3)
        self.assertIn("all", self.bundles)
        self.assertIn("united_kingdom", self.bundles)
        self.assertTrue((Path(self.models.name) / "model-all.joblib").exists())

    def test_selected_model_is_not_worse_than_baseline(self):
        for bundle in self.bundles.values():
            self.assertLessEqual(bundle["cv_rmse"], bundle["baseline_rmse"] + 1e-9)

    def test_predict_returns_sensible_number(self):
        out = model_predict(self.bundles, "all", "2018-06-01")
        self.assertGreater(out["y_pred"], 0)
        self.assertEqual(out["as_of_date"], "2018-06-01")
        self.assertFalse(out["beyond_history"])
        self.assertIsInstance(out["out_of_distribution"], bool)

    def test_country_name_spelling_is_forgiving(self):
        a = model_predict(self.bundles, "United Kingdom", "2018-06-01")
        b = model_predict(self.bundles, "united_kingdom", "2018-06-01")
        self.assertEqual(a["y_pred"], b["y_pred"])

    def test_uk_forecast_is_below_all_countries(self):
        uk = model_predict(self.bundles, "united_kingdom", "2018-06-01")["y_pred"]
        everything = model_predict(self.bundles, "all", "2018-06-01")["y_pred"]
        self.assertLess(uk, everything)

    def test_unknown_country_raises(self):
        with self.assertRaises(ValueError):
            model_predict(self.bundles, "narnia", "2018-06-01")

    def test_date_before_history_raises(self):
        with self.assertRaises(ValueError):
            model_predict(self.bundles, "all", "2017-01-01")

    def test_date_after_history_uses_last_row_and_says_so(self):
        out = model_predict(self.bundles, "all", "2030-01-01")
        self.assertTrue(out["beyond_history"])
        self.assertLess(pd.Timestamp(out["as_of_date"]), pd.Timestamp("2030-01-01"))

    def test_obviously_foreign_features_are_flagged(self):
        bundle = self.bundles["all"]
        X = bundle["history"].dropna(subset=bundle["features"])[bundle["features"]].tail(1) * 100
        self.assertTrue(bundle["detector"].is_novel(X)[0])


if __name__ == "__main__":
    unittest.main()
