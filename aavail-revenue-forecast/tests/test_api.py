import json
import tempfile
import unittest
from pathlib import Path

from app import create_app, validate_predict_payload
from tests.helpers import make_transactions, write_json_files


def make_client(data_dir, model_dir, log_path):
    app = create_app({"DATA_DIR": data_dir, "MODEL_DIR": model_dir, "LOG_PATH": log_path,
                      "TOP_N": 2, "QUICK_TRAIN": True, "TESTING": True})
    return app.test_client()


class PayloadValidationTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate_predict_payload({"country": " all ", "target_date": "2018-03-01"}),
                         ("all", "2018-03-01"))

    def test_rejects_bad_input(self):
        bad = [None, [], {}, {"country": "all"}, {"target_date": "2018-03-01"},
               {"country": "", "target_date": "2018-03-01"},
               {"country": 5, "target_date": "2018-03-01"},
               {"country": "all", "target_date": "03/01/2018"},
               {"country": "all", "target_date": "2018-13-45"},
               {"country": "all", "target_date": 20180301}]
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_predict_payload(payload)


class UntrainedApiTests(unittest.TestCase):
    def test_predict_before_training_is_503(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = make_client(f"{tmp}/data", f"{tmp}/models", f"{tmp}/log.jsonl")
            resp = client.post("/predict", json={"country": "all", "target_date": "2018-06-01"})
            self.assertEqual(resp.status_code, 503)

    def test_train_without_data_is_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = make_client(f"{tmp}/data", f"{tmp}/models", f"{tmp}/log.jsonl")
            self.assertEqual(client.post("/train").status_code, 404)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "data").mkdir()
        write_json_files(root / "data", make_transactions(n_days=450))
        cls.log_path = root / "logs" / "log.jsonl"
        cls.client = make_client(str(root / "data"), str(root / "models"), str(cls.log_path))
        cls.train_resp = cls.client.post("/train")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def predict(self, **payload):
        return self.client.post("/predict", json=payload)

    def test_train_succeeds(self):
        self.assertEqual(self.train_resp.status_code, 200)
        body = self.train_resp.get_json()
        self.assertEqual(body["status"], "trained")
        self.assertIn("all", [m["country"] for m in body["models"]])

    def test_predict_single_country(self):
        resp = self.predict(country="United Kingdom", target_date="2018-06-01")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["country"], "united_kingdom")
        self.assertGreater(body["y_pred"], 0)
        self.assertIn("request_id", body)

    def test_predict_all_countries(self):
        resp = self.predict(country="all", target_date="2018-06-01")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["country"], "all")

    def test_missing_fields_are_400(self):
        self.assertEqual(self.predict(country="all").status_code, 400)
        self.assertEqual(self.predict(target_date="2018-06-01").status_code, 400)

    def test_bad_date_format_is_400(self):
        resp = self.predict(country="all", target_date="June 1st")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("YYYY-MM-DD", resp.get_json()["error"])

    def test_unknown_country_is_400(self):
        resp = self.predict(country="narnia", target_date="2018-06-01")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("narnia", resp.get_json()["error"])

    def test_non_json_body_is_400(self):
        resp = self.client.post("/predict", data="country=all", content_type="text/plain")
        self.assertEqual(resp.status_code, 400)

    def test_wrong_method_is_405(self):
        self.assertEqual(self.client.get("/predict").status_code, 405)

    def test_logfile_records_requests(self):
        self.predict(country="all", target_date="2018-07-01")
        resp = self.client.get("/logfile?n=1")
        self.assertEqual(resp.status_code, 200)
        rec = json.loads(resp.get_data(as_text=True).splitlines()[-1])
        self.assertEqual(rec["endpoint"], "predict")
        self.assertEqual(rec["input"], {"country": "all", "target_date": "2018-07-01"})
        self.assertIsNotNone(rec["runtime_seconds"])
        self.assertIn("unique_id", rec)

    def test_logfile_rejects_bad_n(self):
        self.assertEqual(self.client.get("/logfile?n=abc").status_code, 400)
        self.assertEqual(self.client.get("/logfile?n=0").status_code, 400)

    def test_failed_requests_are_logged_as_errors(self):
        self.predict(country="narnia", target_date="2018-06-01")
        rec = json.loads(self.client.get("/logfile?n=1").get_data(as_text=True))
        self.assertEqual(rec["status"], "error")


if __name__ == "__main__":
    unittest.main()
