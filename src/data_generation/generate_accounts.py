"""
Generate synthetic account-level data for CreditScope Pro.

This module creates credit products linked to customers:
credit cards, personal loans, auto loans, and student refinance loans.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def assign_risk_tier(fico: int) -> str:
    """Assign a simple credit risk tier based on FICO."""
    if fico >= 740:
        return "prime"
    if fico >= 670:
        return "near_prime"
    if fico >= 600:
        return "subprime"
    return "deep_subprime"


def generate_accounts(customers: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic credit accounts for each customer.

    Parameters
    ----------
    customers:
        Customer DataFrame from the customers table.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Account-level synthetic dataset.
    """

    rng = np.random.default_rng(seed)

    account_rows = []
    account_counter = 1

    for row in customers.itertuples(index=False):
        fico = row.baseline_fico
        income = row.annual_income
        dti = row.baseline_dti

        if fico >= 700 and income >= 60_000:
            account_count_probs = [0.10, 0.45, 0.30, 0.15]
        elif fico >= 620:
            account_count_probs = [0.30, 0.45, 0.20, 0.05]
        else:
            account_count_probs = [0.55, 0.32, 0.10, 0.03]

        n_accounts = rng.choice([1, 2, 3, 4], p=account_count_probs)

        for _ in range(n_accounts):
            product_type = rng.choice(
                ["credit_card", "personal_loan", "auto_loan", "student_refi"],
                p=[0.62, 0.18, 0.15, 0.05],
            )

            risk_tier = assign_risk_tier(fico)

            if product_type == "credit_card":
                base_limit = income * rng.uniform(0.10, 0.35)
                credit_limit = np.clip(base_limit, 500, 35_000)
                term_months = None
                base_rate = 0.18

            elif product_type == "personal_loan":
                base_limit = income * rng.uniform(0.08, 0.28)
                credit_limit = np.clip(base_limit, 1_000, 45_000)
                term_months = int(rng.choice([24, 36, 48, 60]))
                base_rate = 0.13

            elif product_type == "auto_loan":
                base_limit = income * rng.uniform(0.20, 0.55)
                credit_limit = np.clip(base_limit, 5_000, 60_000)
                term_months = int(rng.choice([36, 48, 60, 72]))
                base_rate = 0.08

            else:
                base_limit = income * rng.uniform(0.12, 0.40)
                credit_limit = np.clip(base_limit, 3_000, 80_000)
                term_months = int(rng.choice([60, 84, 120]))
                base_rate = 0.09

            risk_spread = {
                "prime": 0.00,
                "near_prime": 0.025,
                "subprime": 0.065,
                "deep_subprime": 0.11,
            }[risk_tier]

            interest_rate = base_rate + risk_spread + rng.normal(0, 0.012)
            interest_rate = float(np.clip(interest_rate, 0.04, 0.34))

            open_month_offset = int(rng.integers(1, 72))
            open_date = pd.Timestamp("2026-01-01") - pd.DateOffset(months=open_month_offset)

            account_rows.append(
                {
                    "account_id": f"ACCT_{account_counter:08d}",
                    "customer_id": row.customer_id,
                    "product_type": product_type,
                    "open_date": open_date.date(),
                    "credit_limit": round(float(credit_limit), 2),
                    "interest_rate": round(interest_rate, 4),
                    "term_months": term_months,
                    "origination_fico": int(fico),
                    "origination_income": round(float(income), 2),
                    "origination_dti": round(float(dti), 4),
                    "risk_tier_at_origination": risk_tier,
                }
            )

            account_counter += 1

    accounts = pd.DataFrame(account_rows)

    return accounts


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        customers = conn.execute("SELECT * FROM customers").fetchdf()

        accounts = generate_accounts(customers, seed=42)

        conn.execute("DELETE FROM accounts")
        conn.register("accounts_df", accounts)
        conn.execute("""
    INSERT INTO accounts (
        account_id,
        customer_id,
        product_type,
        open_date,
        credit_limit,
        interest_rate,
        term_months,
        origination_fico,
        origination_income,
        origination_dti,
        risk_tier_at_origination
    )
    SELECT
        account_id,
        customer_id,
        product_type,
        open_date,
        credit_limit,
        interest_rate,
        term_months,
        origination_fico,
        origination_income,
        origination_dti,
        risk_tier_at_origination
    FROM accounts_df
""")

        row_count = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]

    print("Accounts generated and loaded into DuckDB.")
    print(f"Rows inserted: {row_count}")
    print()
    print(accounts.head())
    print()
    print(accounts["product_type"].value_counts())