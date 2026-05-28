"""
Generate monthly bureau-level customer credit profiles for CreditScope Pro.

This module creates customer-month bureau snapshots using account balances,
credit limits, delinquency behavior, and default/chargeoff signals.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def generate_bureau_monthly(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    snapshots: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate monthly bureau snapshots at the customer-month level.

    Parameters
    ----------
    customers:
        Customer-level data.
    accounts:
        Account-level data.
    snapshots:
        Account monthly snapshot data.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Bureau monthly customer-level dataset.
    """

    rng = np.random.default_rng(seed)

    customer_month = (
        snapshots.groupby(["customer_id", "month"], as_index=False)
        .agg(
            total_debt_balance=("current_balance", "sum"),
            total_credit_limit=("current_balance", "size"),
            max_days_past_due=("days_past_due", "max"),
            delinquencies_current_month=("days_past_due", lambda x: (x > 0).sum()),
            default_count=("default_flag", "sum"),
            chargeoff_count=("chargeoff_flag", "sum"),
        )
    )

    credit_limits = (
        accounts.groupby("customer_id", as_index=False)
        .agg(
            total_credit_limit_value=("credit_limit", "sum"),
            num_open_accounts=("account_id", "count"),
        )
    )

    customer_month = customer_month.merge(
        credit_limits,
        on="customer_id",
        how="left",
    )

    customer_month = customer_month.merge(
        customers[
            [
                "customer_id",
                "baseline_fico",
                "baseline_dti",
                "baseline_financial_resilience",
                "employment_stability_score",
            ]
        ],
        on="customer_id",
        how="left",
    )

    customer_month["credit_utilization"] = (
        customer_month["total_debt_balance"]
        / customer_month["total_credit_limit_value"]
    ).clip(0, 1.5)

    customer_month = customer_month.sort_values(["customer_id", "month"])

    customer_month["delinquencies_12m"] = (
        customer_month.groupby("customer_id")["delinquencies_current_month"]
        .rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    utilization_penalty = customer_month["credit_utilization"] * 75
    delinquency_penalty = customer_month["delinquencies_12m"] * 7
    severe_dpd_penalty = np.where(customer_month["max_days_past_due"] >= 90, 55, 0)
    default_penalty = customer_month["default_count"] * 35
    chargeoff_penalty = customer_month["chargeoff_count"] * 65

    resilience_bonus = customer_month["baseline_financial_resilience"] * 25
    employment_bonus = customer_month["employment_stability_score"] * 15

    fico_score = (
        customer_month["baseline_fico"]
        - utilization_penalty
        - delinquency_penalty
        - severe_dpd_penalty
        - default_penalty
        - chargeoff_penalty
        + resilience_bonus
        + employment_bonus
        + rng.normal(0, 12, size=len(customer_month))
    )

    fico_score = np.clip(fico_score, 300, 850).round().astype(int)

    inquiry_probability = (
        0.04
        + customer_month["credit_utilization"] * 0.05
        + (customer_month["delinquencies_current_month"] > 0).astype(int) * 0.04
    ).clip(0, 0.25)

    num_recent_inquiries = rng.binomial(
        n=3,
        p=inquiry_probability.to_numpy(),
        size=len(customer_month),
    )

    bankruptcy_probability = (
        0.0002
        + (customer_month["max_days_past_due"] >= 90).astype(int) * 0.002
        + (customer_month["chargeoff_count"] > 0).astype(int) * 0.006
        + (fico_score < 500).astype(int) * 0.003
    ).clip(0, 0.03)

    bankruptcy_flag = rng.random(len(customer_month)) < bankruptcy_probability

    bureau = pd.DataFrame(
        {
            "bureau_snapshot_id": [
                f"BUREAU_{i:010d}" for i in range(1, len(customer_month) + 1)
            ],
            "customer_id": customer_month["customer_id"],
            "month": customer_month["month"],
            "fico_score": fico_score,
            "total_debt_balance": customer_month["total_debt_balance"].round(2),
            "total_credit_limit": customer_month["total_credit_limit_value"].round(2),
            "credit_utilization": customer_month["credit_utilization"].round(4),
            "num_open_accounts": customer_month["num_open_accounts"].astype(int),
            "num_recent_inquiries": num_recent_inquiries.astype(int),
            "delinquencies_12m": customer_month["delinquencies_12m"].astype(int),
            "bankruptcy_flag": bankruptcy_flag.astype(bool),
        }
    )

    return bureau


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        customers = conn.execute("""
            SELECT
                customer_id,
                baseline_fico,
                baseline_dti,
                baseline_financial_resilience,
                employment_stability_score
            FROM customers
        """).fetchdf()

        accounts = conn.execute("""
            SELECT
                account_id,
                customer_id,
                credit_limit
            FROM accounts
        """).fetchdf()

        snapshots = conn.execute("""
            SELECT
                account_id,
                customer_id,
                month,
                current_balance,
                days_past_due,
                default_flag,
                chargeoff_flag
            FROM account_monthly_snapshots
            WHERE scenario_id = 'baseline'
        """).fetchdf()

        bureau = generate_bureau_monthly(
            customers=customers,
            accounts=accounts,
            snapshots=snapshots,
            seed=42,
        )

        conn.execute("DELETE FROM bureau_monthly")
        conn.register("bureau_df", bureau)

        conn.execute("""
            INSERT INTO bureau_monthly (
                bureau_snapshot_id,
                customer_id,
                month,
                fico_score,
                total_debt_balance,
                total_credit_limit,
                credit_utilization,
                num_open_accounts,
                num_recent_inquiries,
                delinquencies_12m,
                bankruptcy_flag
            )
            SELECT
                bureau_snapshot_id,
                customer_id,
                month,
                fico_score,
                total_debt_balance,
                total_credit_limit,
                credit_utilization,
                num_open_accounts,
                num_recent_inquiries,
                delinquencies_12m,
                bankruptcy_flag
            FROM bureau_df
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM bureau_monthly").fetchone()[0]

    print("Bureau monthly data generated and loaded into DuckDB.")
    print(f"Rows inserted: {row_count}")
    print()
    print(bureau.head())
    print()
    print(
        bureau[
            [
                "fico_score",
                "total_debt_balance",
                "credit_utilization",
                "num_open_accounts",
                "num_recent_inquiries",
                "delinquencies_12m",
            ]
        ].describe()
    )