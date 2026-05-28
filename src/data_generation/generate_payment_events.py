"""
Generate payment event data for CreditScope Pro.

This module converts monthly account snapshots into payment-level events:
due dates, payment dates, missed payments, partial payments, late payments,
and payment channels.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def generate_payment_events(snapshots: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic payment events from account monthly snapshots.

    Parameters
    ----------
    snapshots:
        Account monthly snapshot data from DuckDB.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Payment event dataset.
    """

    rng = np.random.default_rng(seed)

    n = len(snapshots)

    due_dates = pd.to_datetime(snapshots["month"]) + pd.to_timedelta(
        rng.integers(15, 26, size=n), unit="D"
    )

    missed_payment_flag = snapshots["actual_payment_amount"] <= (
        snapshots["minimum_payment_due"] * 0.10
    )

    partial_payment_flag = (
        snapshots["actual_payment_amount"] > 0
    ) & (
        snapshots["actual_payment_amount"] < snapshots["minimum_payment_due"] * 0.95
    )

    risk_score = snapshots["behavioral_risk_score"].to_numpy()
    days_past_due = snapshots["days_past_due"].to_numpy()

    on_time_days = rng.choice(
    [0, 1, 2, 3, 4, 5],
    size=n,
    p=[0.72, 0.12, 0.07, 0.04, 0.03, 0.02],
)

    on_time_days = rng.choice(
        [0, 1, 2, 3, 4, 5],
        size=n,
        p=[0.72, 0.12, 0.07, 0.04, 0.03, 0.02],
    )

    days_late = np.where(
        missed_payment_flag,
        rng.integers(30, 75, size=n),
        np.where(
            partial_payment_flag,
            rng.integers(3, 30, size=n),
            on_time_days,
        ),
    )

    days_late = np.where(days_past_due >= 90, rng.integers(65, 110, size=n), days_late)
    days_late = np.where(days_past_due == 60, rng.integers(35, 70, size=n), days_late)
    days_late = np.where(days_past_due == 30, rng.integers(10, 40, size=n), days_late)

    late_payment_flag = days_late >= 3

    payment_dates = due_dates + pd.to_timedelta(days_late, unit="D")

    payment_channel = np.where(
        risk_score < 0.30,
        rng.choice(["autopay", "online", "mobile"], size=n, p=[0.62, 0.25, 0.13]),
        np.where(
            risk_score < 0.60,
            rng.choice(
                ["autopay", "online", "mobile", "branch"],
                size=n,
                p=[0.38, 0.34, 0.22, 0.06],
            ),
            rng.choice(
                ["online", "mobile", "branch", "failed_autopay"],
                size=n,
                p=[0.36, 0.30, 0.12, 0.22],
            ),
        ),
    )

    payment_events = pd.DataFrame(
        {
            "payment_id": [f"PAY_{i:010d}" for i in range(1, n + 1)],
            "account_id": snapshots["account_id"],
            "account_month_id": snapshots["account_month_id"],
            "customer_id": snapshots["customer_id"],
            "scenario_id": snapshots["scenario_id"],
            "payment_date": payment_dates.dt.date,
            "due_date": due_dates.dt.date,
            "scheduled_payment_amount": snapshots["minimum_payment_due"].round(2),
            "actual_payment_amount": snapshots["actual_payment_amount"].round(2),
            "payment_channel": payment_channel,
            "missed_payment_flag": missed_payment_flag.astype(bool),
            "partial_payment_flag": partial_payment_flag.astype(bool),
            "late_payment_flag": late_payment_flag.astype(bool),
            "days_late": days_late.astype(int),
        }
    )

    return payment_events


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        snapshots = conn.execute("""
    SELECT
        account_month_id,
        account_id,
        customer_id,
        scenario_id,
        month,
        minimum_payment_due,
        actual_payment_amount,
        days_past_due,
        behavioral_risk_score
    FROM account_monthly_snapshots
""").fetchdf()

        payment_events = generate_payment_events(snapshots, seed=42)

        conn.execute("DELETE FROM payment_events")
        conn.register("payment_events_df", payment_events)

        conn.execute("""
            INSERT INTO payment_events (
                payment_id,
                account_id,
                customer_id,
                payment_date,
                due_date,
                scheduled_payment_amount,
                actual_payment_amount,
                payment_channel,
                missed_payment_flag,
                partial_payment_flag,
                late_payment_flag,
                account_month_id,
                scenario_id,
                days_late
            )
            SELECT
                payment_id,
                account_id,
                customer_id,
                payment_date,
                due_date,
                scheduled_payment_amount,
                actual_payment_amount,
                payment_channel,
                missed_payment_flag,
                partial_payment_flag,
                late_payment_flag,
                account_month_id,
                scenario_id,
                days_late
            FROM payment_events_df
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM payment_events").fetchone()[0]

    print("Payment events generated and loaded into DuckDB.")
    print(f"Rows inserted: {row_count}")
    print()
    print(payment_events.head())
    print()
    print(payment_events["payment_channel"].value_counts())
    print()
    print(
        payment_events[
            ["missed_payment_flag", "partial_payment_flag", "late_payment_flag", "days_late"]
        ].mean()
    )