"""Generate sample CSVs for the DataQ dashboard.

Writes three files into ``sample_data/``:

* ``customers_clean.csv``   - well-formed customer records
* ``customers_dirty.csv``   - same shape with injected nulls, duplicate rows,
  an out-of-range ``age`` value, and a renamed column vs. the clean version
* ``orders.csv``            - order records with a ``order_date`` column for
  freshness testing (most recent row is today)

Run with ``python make_sample_data.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

N_ROWS = 200
SEED = 20260709

OUT_DIR = Path(__file__).resolve().parent / "sample_data"

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kara", "Liam", "Maya", "Noah", "Olivia", "Priya",
    "Quinn", "Ravi", "Sara", "Tom", "Uma", "Vik", "Wren", "Xander",
    "Yara", "Zane",
]
LAST_NAMES = [
    "Adams", "Baker", "Chen", "Diaz", "Evans", "Fisher", "Gupta", "Hill",
    "Ito", "Jones", "Khan", "Lopez", "Martin", "Nguyen", "Owen", "Patel",
    "Reyes", "Singh", "Tan", "Ueda", "Vargas", "Wong", "Yamada", "Zhao",
]
CITIES = ["Austin", "Boston", "Chicago", "Denver", "Eugene", "Fresno",
          "Galveston", "Hartford", "Irvine", "Jackson"]
SEGMENTS = ["retail", "smb", "enterprise"]


def _build_customers_clean(rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(1, N_ROWS + 1)
    first = rng.choice(FIRST_NAMES, size=N_ROWS)
    last = rng.choice(LAST_NAMES, size=N_ROWS)
    names = [f"{f} {l}" for f, l in zip(first, last)]
    emails = [
        f"{f.lower()}.{l.lower()}{i}@example.com"
        for i, (f, l) in enumerate(zip(first, last))
    ]
    ages = rng.integers(low=18, high=85, size=N_ROWS)
    cities = rng.choice(CITIES, size=N_ROWS)
    segments = rng.choice(SEGMENTS, size=N_ROWS, p=[0.6, 0.3, 0.1])
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 900, size=N_ROWS), unit="D"
    )
    return pd.DataFrame(
        {
            "customer_id": ids,
            "name": names,
            "email": emails,
            "age": ages,
            "city": cities,
            "segment": segments,
            "signup_date": signup.strftime("%Y-%m-%d"),
        }
    )


def _dirty_up(clean: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    dirty = clean.copy()

    # Inject nulls (~15% in email, ~25% in city).
    email_null_idx = rng.choice(
        dirty.index, size=int(len(dirty) * 0.15), replace=False
    )
    dirty.loc[email_null_idx, "email"] = np.nan

    city_null_idx = rng.choice(
        dirty.index, size=int(len(dirty) * 0.25), replace=False
    )
    dirty.loc[city_null_idx, "city"] = np.nan

    # Inject a small number of nulls in age too.
    age_null_idx = rng.choice(
        dirty.index, size=int(len(dirty) * 0.05), replace=False
    )
    dirty.loc[age_null_idx, "age"] = np.nan

    # Out-of-range ages.
    dirty.loc[3, "age"] = 250
    dirty.loc[7, "age"] = -5

    # Duplicate rows: append copies of a handful of existing rows.
    dupes = dirty.sample(n=5, random_state=int(rng.integers(0, 10_000)))
    dirty = pd.concat([dirty, dupes], ignore_index=True)

    # Renamed column vs. clean: signup_date -> registered_on.
    dirty = dirty.rename(columns={"signup_date": "registered_on"})

    return dirty


def _build_orders(rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(1, N_ROWS + 1)
    customer_ids = rng.integers(1, N_ROWS + 1, size=N_ROWS)
    quantities = rng.integers(1, 10, size=N_ROWS)
    prices = np.round(rng.uniform(5.0, 500.0, size=N_ROWS), 2)
    totals = np.round(quantities * prices, 2)

    # Dates: spread from ~1 year ago up to today so freshness passes by default.
    today = datetime.now().date()
    offsets = rng.integers(0, 365, size=N_ROWS - 1)
    dates = [today - timedelta(days=int(o)) for o in offsets]
    dates.append(today)  # ensure at least one "fresh" row
    rng.shuffle(dates)

    return pd.DataFrame(
        {
            "order_id": ids,
            "customer_id": customer_ids,
            "order_date": [d.isoformat() for d in dates],
            "quantity": quantities,
            "unit_price": prices,
            "total": totals,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    clean = _build_customers_clean(rng)
    dirty = _dirty_up(clean, rng)
    orders = _build_orders(rng)

    clean_path = OUT_DIR / "customers_clean.csv"
    dirty_path = OUT_DIR / "customers_dirty.csv"
    orders_path = OUT_DIR / "orders.csv"

    clean.to_csv(clean_path, index=False)
    dirty.to_csv(dirty_path, index=False)
    orders.to_csv(orders_path, index=False)

    for path, df in [(clean_path, clean), (dirty_path, dirty), (orders_path, orders)]:
        print(f"wrote {path.relative_to(OUT_DIR.parent)}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
