import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = os.environ.get("AAVAIL_DATA_DIR", os.path.join("data", "cs-train"))
EXCLUDED_COUNTRIES = ("unspecified",)
REQUIRED_COLUMNS = ["country", "year", "month", "day", "invoice", "price"]

_ALIASES = {
    "streamid": "stream_id",
    "timesviewed": "times_viewed",
    "totalprice": "price",
    "invoiceid": "invoice",
    "invoiceno": "invoice",
    "customerid": "customer_id",
}


def canonical_column(name):
    key = re.sub(r"\s+", "_", str(name).strip().lower())
    return _ALIASES.get(key.replace("_", ""), key)


def country_key(name):
    return re.sub(r"[\s_]+", "_", str(name).strip().lower())


def load_json_files(data_dir=DEFAULT_DATA_DIR):
    files = sorted(Path(data_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no .json files found in {data_dir!r}")

    frames = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            frame = pd.DataFrame(json.load(fh))
        frame.columns = [canonical_column(c) for c in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean_transactions(raw):
    df = raw.copy()
    absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if absent:
        raise ValueError(f"input is missing required columns: {absent}")
    for col in ("customer_id", "stream_id", "times_viewed"):
        if col not in df.columns:
            df[col] = np.nan

    report = {"rows_raw": int(len(df))}

    parts = df[["year", "month", "day"]].apply(pd.to_numeric, errors="coerce")
    df["date"] = pd.to_datetime(parts, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["country"] = df["country"].astype("string").str.strip()

    unusable = df["date"].isna() | df["price"].isna() | df["country"].isna() | (df["country"] == "")
    report["rows_dropped_unusable"] = int(unusable.sum())
    df = df.loc[~unusable].copy()
    df["country"] = df["country"].astype(str)

    df["invoice"] = df["invoice"].astype(str).str.replace(r"\D", "", regex=True)
    df["invoice"] = df["invoice"].replace("", np.nan)

    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce")
    report["missing_times_viewed_imputed"] = int(df["times_viewed"].isna().sum())
    by_country = df.groupby("country")["times_viewed"].transform("median")
    df["times_viewed"] = df["times_viewed"].fillna(by_country)
    df["times_viewed"] = df["times_viewed"].fillna(df["times_viewed"].median()).fillna(0.0)

    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    report["missing_customer_id_flagged"] = int(df["customer_id"].isna().sum())
    df["customer_id"] = df["customer_id"].fillna(-1).astype("int64")

    report["missing_stream_id_filled"] = int(df["stream_id"].isna().sum())
    df["stream_id"] = df["stream_id"].fillna("unknown").astype(str)

    df = df.sort_values("date").reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    return df, report


def top_countries(df, n=10, exclude=EXCLUDED_COUNTRIES):
    totals = df.groupby("country")["price"].sum()
    totals = totals[~totals.index.str.lower().isin(exclude)]
    return list(totals.nlargest(n).index)


def pivot_revenue_by_country(df, countries=None):
    data = df if countries is None else df[df["country"].isin(countries)]
    return data.pivot_table(index="date", columns="country", values="price",
                            aggfunc="sum", fill_value=0.0)


def daily_series(df, country=None, gap_strategy="zero"):
    if gap_strategy not in ("zero", "interpolate"):
        raise ValueError("gap_strategy must be 'zero' or 'interpolate'")

    data = df if country is None else df[df["country"].str.lower() == str(country).lower()]
    daily = data.groupby("date").agg(
        purchases=("invoice", "size"),
        unique_invoices=("invoice", "nunique"),
        unique_streams=("stream_id", "nunique"),
        total_views=("times_viewed", "sum"),
        revenue=("price", "sum"),
    )

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D", name="date")
    daily = daily.reindex(full_range).astype(float)

    if gap_strategy == "interpolate":
        feed_gap = ~full_range.isin(pd.DatetimeIndex(df["date"].unique()))
        daily.loc[~feed_gap] = daily.loc[~feed_gap].fillna(0.0)
        daily = daily.interpolate(limit_direction="both")
    else:
        daily = daily.fillna(0.0)
    return daily.reset_index()


def build_series(df, top_n=10, gap_strategy="zero"):
    series = {"all": daily_series(df, None, gap_strategy)}
    for name in top_countries(df, top_n):
        series[country_key(name)] = daily_series(df, name, gap_strategy)
    return series


def ingest(data_dir=DEFAULT_DATA_DIR, top_n=10, gap_strategy="zero"):
    clean, report = clean_transactions(load_json_files(data_dir))
    series = build_series(clean, top_n=top_n, gap_strategy=gap_strategy)
    report["countries"] = list(series)
    report["date_min"] = str(clean["date"].min().date())
    report["date_max"] = str(clean["date"].max().date())
    report["unique_invoice_dates"] = int(clean["date"].nunique())
    return series, report


def main():
    parser = argparse.ArgumentParser(description="Compile the AAVAIL JSON files into daily series")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out", help="optional folder to write one CSV per series")
    args = parser.parse_args()

    series, report = ingest(args.data_dir, args.top_n)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for key, frame in series.items():
            frame.to_csv(Path(args.out) / f"{key}.csv", index=False)


if __name__ == "__main__":
    main()
