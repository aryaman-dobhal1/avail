import tempfile
import unittest

import numpy as np
import pandas as pd

from data_ingestion import (canonical_column, clean_transactions, country_key, daily_series,
                            ingest, load_json_files, pivot_revenue_by_country, top_countries)
from tests.helpers import make_transactions, write_json_files


class ColumnNameTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(canonical_column("StreamID"), "stream_id")
        self.assertEqual(canonical_column("TimesViewed"), "times_viewed")
        self.assertEqual(canonical_column("total_price"), "price")
        self.assertEqual(canonical_column("invoice_id"), "invoice")
        self.assertEqual(canonical_column("Country"), "country")

    def test_country_key(self):
        self.assertEqual(country_key(" United Kingdom "), "united_kingdom")
        self.assertEqual(country_key("united_kingdom"), "united_kingdom")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_df = make_transactions(n_days=120)
        cls.tmp = tempfile.TemporaryDirectory()
        write_json_files(cls.tmp.name, cls.raw_df)
        cls.raw = load_json_files(cls.tmp.name)
        cls.clean, cls.report = clean_transactions(cls.raw)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_column_spellings_are_merged(self):
        for col in ("price", "stream_id", "times_viewed", "invoice"):
            self.assertIn(col, self.raw.columns)
        self.assertFalse(self.raw["price"].isna().any())
        self.assertEqual(len(self.raw), len(self.raw_df))

    def test_missing_values_are_handled_and_reported(self):
        self.assertGreater(self.report["missing_times_viewed_imputed"], 0)
        self.assertGreater(self.report["missing_customer_id_flagged"], 0)
        self.assertFalse(self.clean["times_viewed"].isna().any())
        self.assertFalse(self.clean["customer_id"].isna().any())
        self.assertIn(-1, set(self.clean["customer_id"]))

    def test_invoice_numbers_are_digits_only(self):
        self.assertTrue(self.clean["invoice"].str.fullmatch(r"\d+").all())

    def test_rows_without_a_date_or_price_are_dropped(self):
        broken = self.raw.copy()
        broken.loc[0, "price"] = None
        broken.loc[1, "day"] = None
        _, report = clean_transactions(broken)
        self.assertEqual(report["rows_dropped_unusable"], 2)

    def test_missing_required_column_raises(self):
        with self.assertRaises(ValueError):
            clean_transactions(self.raw.drop(columns=["country"]))

    def test_top_countries_skips_unspecified(self):
        names = top_countries(self.clean, 3)
        self.assertEqual(names[0], "United Kingdom")
        self.assertNotIn("unspecified", names)
        self.assertEqual(len(names), 3)

    def test_daily_series_is_contiguous_and_keeps_revenue(self):
        daily = daily_series(self.clean)
        self.assertTrue((daily["date"].diff().dropna() == pd.Timedelta(days=1)).all())
        self.assertAlmostEqual(daily["revenue"].sum(), self.clean["price"].sum(), places=4)

    def test_country_series_matches_country_revenue(self):
        uk = daily_series(self.clean, "united kingdom")
        expected = self.clean.loc[self.clean["country"] == "United Kingdom", "price"].sum()
        self.assertAlmostEqual(uk["revenue"].sum(), expected, places=4)
        self.assertEqual(len(uk), len(daily_series(self.clean)))

    def test_pivot_has_one_column_per_country(self):
        wide = pivot_revenue_by_country(self.clean, ["United Kingdom", "Germany"])
        self.assertEqual(set(wide.columns), {"United Kingdom", "Germany"})

    def test_ingest_returns_all_plus_top_n(self):
        series, report = ingest(self.tmp.name, top_n=2)
        self.assertEqual(len(series), 3)
        self.assertIn("all", series)
        self.assertEqual(report["date_min"], str(self.clean["date"].min().date()))


class GapTests(unittest.TestCase):
    def setUp(self):
        days = [1, 2, 5]
        raw = pd.DataFrame({
            "country": "Germany", "year": 2018, "month": 1, "day": days,
            "invoice": ["1", "2", "3"], "price": [10.0, 20.0, 50.0],
            "stream_id": "a", "times_viewed": 1, "customer_id": 1,
        })
        self.clean, _ = clean_transactions(raw)

    def test_zero_strategy(self):
        daily = daily_series(self.clean, gap_strategy="zero")
        self.assertEqual(list(daily["revenue"]), [10.0, 20.0, 0.0, 0.0, 50.0])

    def test_interpolate_strategy(self):
        daily = daily_series(self.clean, gap_strategy="interpolate")
        self.assertTrue(np.allclose(daily["revenue"], [10, 20, 30, 40, 50]))

    def test_bad_strategy(self):
        with self.assertRaises(ValueError):
            daily_series(self.clean, gap_strategy="nope")


if __name__ == "__main__":
    unittest.main()
