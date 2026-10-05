import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from logger import DEFAULT_LOG_PATH, read_records
from model import DEFAULT_MODEL_DIR, mae, model_load, rmse


class OODDetector:

    def __init__(self, contamination=0.02, n_estimators=200, random_state=42):
        self.scaler = StandardScaler()
        self.forest = IsolationForest(n_estimators=n_estimators,
                                      contamination=contamination,
                                      random_state=random_state)

    def fit(self, X):
        values = np.asarray(X, dtype=float)
        self.forest.fit(self.scaler.fit_transform(values))
        return self

    def score(self, X):
        return self.forest.score_samples(self.scaler.transform(np.asarray(X, dtype=float)))

    def is_novel(self, X):
        scaled = self.scaler.transform(np.asarray(X, dtype=float))
        return self.forest.predict(scaled) == -1


def feature_drift(reference, current, alpha=0.01):
    rows = []
    for col in reference.columns:
        stat, p_value = ks_2samp(reference[col].dropna(), current[col].dropna())
        rows.append({"feature": col, "ks_stat": float(stat),
                     "p_value": float(p_value), "drifted": bool(p_value < alpha)})
    return pd.DataFrame(rows)


def evaluate_logged_predictions(records, bundles):
    rows = []
    for rec in records:
        if rec.get("endpoint") != "predict" or rec.get("status") != "ok":
            continue
        pred = rec.get("prediction") or {}
        bundle = bundles.get(pred.get("country"))
        if bundle is None or pred.get("y_pred") is None:
            continue
        hist = bundle["history"]
        actual = hist.loc[hist["date"] == pd.Timestamp(pred["as_of_date"]), "target"]
        if actual.empty or actual.isna().all():
            continue
        rows.append({"country": pred["country"], "actual": float(actual.iloc[0]),
                     "predicted": float(pred["y_pred"])})

    cols = ["country", "n", "rmse", "mae"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    out = [{"country": c, "n": len(g),
            "rmse": rmse(g["actual"], g["predicted"]),
            "mae": mae(g["actual"], g["predicted"])} for c, g in df.groupby("country")]
    return pd.DataFrame(out, columns=cols)


def check_degradation(report, bundles, tolerance=0.25):
    report = report.copy()
    report["reference_rmse"] = [bundles[c]["cv_rmse"] for c in report["country"]]
    report["degraded"] = report["rmse"] > report["reference_rmse"] * (1 + tolerance)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check live prediction error against training error")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--models", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tolerance", type=float, default=0.25)
    args = parser.parse_args(argv)

    bundles = model_load(args.models)
    report = evaluate_logged_predictions(read_records(args.log), bundles)
    if report.empty:
        print("no scorable predictions in the log yet")
        return 0
    report = check_degradation(report, bundles, args.tolerance)
    print(report.to_string(index=False))
    return 1 if report["degraded"].any() else 0


if __name__ == "__main__":
    sys.exit(main())
