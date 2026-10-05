import json
import tempfile
import unittest
from pathlib import Path

from logger import DEFAULT_LOG_PATH, RuntimeLogger, read_records


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nested" / "test.jsonl"
        self.logger = RuntimeLogger(self.path)
        prod = Path(DEFAULT_LOG_PATH)
        self.prod_before = prod.read_bytes() if prod.exists() else None

    def tearDown(self):
        prod = Path(DEFAULT_LOG_PATH)
        after = prod.read_bytes() if prod.exists() else None
        self.assertEqual(self.prod_before, after, "test touched the real runtime log")
        self.tmp.cleanup()

    def test_log_path_is_inside_scratch_dir(self):
        self.assertTrue(str(self.logger.path).startswith(self.tmp.name))
        self.assertNotEqual(self.logger.path.resolve(), Path(DEFAULT_LOG_PATH).resolve())

    def test_write_then_read_roundtrip(self):
        rid = self.logger.log("predict", inputs={"country": "all", "target_date": "2018-01-05"},
                              prediction={"y_pred": 123.4}, runtime=0.0123, model_version="1.0")
        (rec,) = self.logger.read()
        self.assertEqual(rec["unique_id"], rid)
        self.assertEqual(rec["endpoint"], "predict")
        self.assertEqual(rec["input"]["country"], "all")
        self.assertEqual(rec["prediction"]["y_pred"], 123.4)
        self.assertAlmostEqual(rec["runtime_seconds"], 0.0123)
        self.assertIn("T", rec["timestamp"])

    def test_ids_are_unique(self):
        ids = {self.logger.log("predict") for _ in range(25)}
        self.assertEqual(len(ids), 25)

    def test_read_last_n(self):
        for i in range(10):
            self.logger.log("predict", inputs={"i": i})
        last = self.logger.read(3)
        self.assertEqual([r["input"]["i"] for r in last], [7, 8, 9])

    def test_appends_instead_of_overwriting(self):
        self.logger.log("train")
        RuntimeLogger(self.path).log("predict")
        self.assertEqual([r["endpoint"] for r in self.logger.read()], ["train", "predict"])

    def test_damaged_line_is_skipped(self):
        self.logger.log("predict")
        with open(self.path, "a") as fh:
            fh.write("{not json\n")
        self.logger.log("predict")
        self.assertEqual(len(self.logger.read()), 2)

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(read_records(Path(self.tmp.name) / "nope.jsonl"), [])

    def test_tail_text_is_valid_jsonl(self):
        self.logger.log("predict")
        self.logger.log("train")
        lines = self.logger.tail_text(5).splitlines()
        self.assertEqual([json.loads(x)["endpoint"] for x in lines], ["predict", "train"])


if __name__ == "__main__":
    unittest.main()
