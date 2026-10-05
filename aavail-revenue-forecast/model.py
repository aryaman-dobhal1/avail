import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_ingestion import DEFAULT_DATA_DIR, country_key, ingest

MODEL_VERSION = "1.0"
DEFAULT_MODEL_DIR = os.environ.get("AAVAIL_MODEL_DIR", "models")
HORIZON = 30
WINDOWS = (7, 14, 30, 60)
MIN_ROWS = 150
SEED = 42

FEATURE_COLUMNS = (
    [f"rev_sum_{w}" for w in WINDOWS]
    + ["purchases_sum_30", "invoices_sum_30", "views_sum_30", "doy_sin", "doy_cos"]
)


def rmse(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def mae(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(err)))


def engineer_features(daily, horizon=HORIZON):
    d = daily.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": d["date"]})

    for w in WINDOWS:
        out[f"rev_sum_{w}"] = d["revenue"].rolling(w, min_periods=w).sum()
    out["purchases_sum_30"] = d["purchases"].rolling(30, min_periods=30).sum()
    out["invoices_sum_30"] = d["unique_invoices"].rolling(30, min_periods=30).sum()
    out["views_sum_30"] = d["total_views"].rolling(30, min_periods=30).sum()

    angle = 2 * np.pi * d["date"].dt.dayofyear / 365.25
    out["doy_sin"] = np.sin(angle)
    out["doy_cos"] = np.cos(angle)

    forward = d["revenue"][::-1].rolling(horizon, min_periods=horizon).sum()[::-1]
    out["target"] = forward.shift(-1)
    return out


class TrailingMeanBaseline(BaseEstimator, RegressorMixin):

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X["rev_sum_30"], dtype=float)


def get_candidates(quick=False):
    n_rf, n_gb = (20, 30) if quick else (300, 200)
    return {
        "baseline_moving_average": TrailingMeanBaseline(),
        "linear_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=n_rf, min_samples_leaf=3,
                                               random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=n_gb, learning_rate=0.05,
                                                       max_depth=3, subsample=0.8,
                                                       random_state=SEED),
    }


def compare_models(X, y, candidates, n_splits=3, gap=HORIZON):
    if len(X) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} training rows, got {len(X)}")
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    rows = []
    for name, estimator in candidates.items():
        rmses, maes = [], []
        for train_idx, test_idx in splitter.split(X):
            fitted = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = fitted.predict(X.iloc[test_idx])
            rmses.append(rmse(y.iloc[test_idx], pred))
            maes.append(mae(y.iloc[test_idx], pred))
        rows.append({"model": name, "rmse": float(np.mean(rmses)),
                     "mae": float(np.mean(maes)), "rmse_std": float(np.std(rmses))})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def train_country(key, daily, quick=False):
    from monitoring import OODDetector

    feats = engineer_features(daily)
    rows = feats.dropna(subset=FEATURE_COLUMNS + ["target"])
    X, y = rows[FEATURE_COLUMNS], rows["target"]

    candidates = get_candidates(quick)
    comparison = compare_models(X, y, candidates)
    best = comparison.iloc[0]
    baseline = comparison.loc[comparison["model"] == "baseline_moving_average", "rmse"].iloc[0]

    model = clone(candidates[best["model"]]).fit(X, y)
    return {
        "key": key,
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon": HORIZON,
        "features": list(FEATURE_COLUMNS),
        "model_name": best["model"],
        "model": model,
        "comparison": comparison,
        "cv_rmse": float(best["rmse"]),
        "cv_mae": float(best["mae"]),
        "baseline_rmse": float(baseline),
        "n_rows": int(len(rows)),
        "detector": OODDetector().fit(X),
        "history": feats,
    }


def save_bundle(bundle, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    final = model_dir / f"model-{bundle['key']}.joblib"
    tmp = final.with_suffix(".joblib.tmp")
    joblib.dump(bundle, tmp)
    os.replace(tmp, final)
    return final


def model_train(data_dir=DEFAULT_DATA_DIR, model_dir=DEFAULT_MODEL_DIR, top_n=10, quick=False):
    series, report = ingest(data_dir, top_n=top_n)
    trained, skipped = [], {}
    for key, daily in series.items():
        try:
            bundle = train_country(key, daily, quick=quick)
        except ValueError as exc:
            skipped[key] = str(exc)
            continue
        save_bundle(bundle, model_dir)
        trained.append({"country": key, "best_model": bundle["model_name"],
                        "cv_rmse": round(bundle["cv_rmse"], 2),
                        "cv_mae": round(bundle["cv_mae"], 2),
                        "baseline_rmse": round(bundle["baseline_rmse"], 2),
                        "n_rows": bundle["n_rows"]})
    if not trained:
        raise ValueError("no model could be trained: " + "; ".join(skipped.values()))
    return {"models": trained, "skipped": skipped, "data_report": report,
            "model_version": MODEL_VERSION}


def model_load(model_dir=DEFAULT_MODEL_DIR):
    bundles = {}
    for path in sorted(Path(model_dir).glob("model-*.joblib")):
        bundle = joblib.load(path)
        bundles[bundle["key"]] = bundle
    return bundles


def model_predict(bundles, country, target_date):
    key = country_key(country)
    if key not in bundles:
        raise ValueError(f"no model for country {country!r}; available: {sorted(bundles)}")
    bundle = bundles[key]

    when = pd.Timestamp(target_date).normalize()
    hist = bundle["history"]
    usable = hist.dropna(subset=bundle["features"])
    if when < usable["date"].min():
        raise ValueError(f"target_date is too early; earliest usable date is "
                         f"{usable['date'].min().date()}")

    row = usable[usable["date"] <= when].iloc[[-1]]
    X = row[bundle["features"]]
    y_hat = max(0.0, float(bundle["model"].predict(X)[0]))

    return {
        "country": key,
        "target_date": str(when.date()),
        "as_of_date": str(row["date"].iloc[0].date()),
        "horizon_days": bundle["horizon"],
        "y_pred": round(y_hat, 2),
        "model": bundle["model_name"],
        "model_version": bundle["version"],
        "out_of_distribution": bool(bundle["detector"].is_novel(X)[0]),
        "beyond_history": bool(when > hist["date"].max()),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(model_train(), indent=2, default=str))
