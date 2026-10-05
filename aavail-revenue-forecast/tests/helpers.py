import numpy as np
import pandas as pd

COUNTRY_WEIGHTS = {"United Kingdom": 6.0, "Germany": 3.0, "France": 2.0,
                   "EIRE": 1.5, "unspecified": 0.5}


def make_transactions(n_days=450, seed=7):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2017-11-01")
    rows, invoice = [], 10000
    for offset in range(n_days):
        if offset % 7 == 6:
            continue
        day = start + pd.Timedelta(days=offset)
        level = (1 + offset / 600) * (1 + 0.3 * np.sin(2 * np.pi * offset / 365))
        for country, weight in COUNTRY_WEIGHTS.items():
            for _ in range(rng.poisson(weight * level)):
                invoice += 1
                prefix = "A" if rng.random() < 0.05 else ""
                rows.append({
                    "country": country,
                    "customer_id": int(rng.integers(12000, 13000)),
                    "invoice": f"{prefix}{invoice}",
                    "price": round(float(rng.lognormal(3.0, 0.5)), 2),
                    "stream_id": f"S{int(rng.integers(100, 130))}",
                    "times_viewed": int(rng.integers(1, 12)),
                    "year": day.year, "month": day.month, "day": day.day,
                })
    return pd.DataFrame(rows)


def write_json_files(directory, df, seed=1):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df.loc[rng.random(len(df)) < 0.03, "times_viewed"] = np.nan
    df.loc[rng.random(len(df)) < 0.05, "customer_id"] = np.nan

    for i, ((year, month), part) in enumerate(df.groupby(["year", "month"])):
        if i % 2:
            part = part.rename(columns={"price": "total_price", "stream_id": "StreamID",
                                        "times_viewed": "TimesViewed", "invoice": "invoice_id"})
        part.to_json(f"{directory}/invoices-{year}-{month:02d}.json", orient="records")
