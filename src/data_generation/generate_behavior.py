"""
Generate monthly account behavior snapshots for CreditScope Pro.

This module creates account-month records that connect:
customer profile, account profile, macro conditions, scenario shocks,
payment capacity, delinquency risk, and default behavior.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def generate_account_monthly_snapshots(seed: int = 42) -> pd.DataFrame:
    """
    Generate account-month synthetic behavior records.

    Returns
    -------
    pd.DataFrame
        Account monthly snapshot dataset.
    """

    rng = np.random.default_rng(seed)

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        accounts = conn.execute("""
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
            FROM accounts
        """).fetchdf()

        customers = conn.execute("""
            SELECT
                customer_id,
                birth_year,
                age,
                state,
                zip3,
                employment_type,
                annual_income,
                monthly_income,
                income_band,
                education_level,
                student_loan_flag,
                baseline_fico,
                baseline_dti,
                baseline_financial_resilience,
                employment_stability_score
            FROM customers
        """).fetchdf()

        macro = conn.execute("""
            SELECT
                month,
                state,
                zip3,
                unemployment_rate,
                inflation_rate,
                interest_rate_index,
                rent_index,
                wage_growth_rate,
                consumer_sentiment_index,
                recession_flag
            FROM macro_monthly
        """).fetchdf()

        scenarios = conn.execute("""
            SELECT
                scenario_id,
                scenario_name,
                month,
                student_loan_restart_flag,
                student_loan_payment_multiplier,
                interest_rate_shock_bps,
                unemployment_shock,
                inflation_shock,
                policy_stress_index
            FROM policy_scenarios
            WHERE scenario_id IN ('baseline', 'severe_credit_stress')
        """).fetchdf()

    base = accounts.merge(customers, on="customer_id", how="left")

    months = pd.DataFrame(
        {"month": pd.date_range("2025-01-01", periods=24, freq="MS").date}
    )

    base["join_key"] = 1
    months["join_key"] = 1

    account_months = base.merge(months, on="join_key").drop(columns=["join_key"])

    macro["month"] = pd.to_datetime(macro["month"]).dt.date
    scenarios["month"] = pd.to_datetime(scenarios["month"]).dt.date

    account_months = account_months.merge(
        macro,
        on=["month", "state", "zip3"],
        how="left",
    )

    account_months = account_months.merge(
        scenarios,
        on="month",
        how="left",
    )

    n = len(account_months)

    product_base_util = account_months["product_type"].map(
        {
            "credit_card": 0.42,
            "personal_loan": 0.72,
            "auto_loan": 0.82,
            "student_refi": 0.76,
        }
    ).to_numpy()

    risk_util_adjustment = account_months["risk_tier_at_origination"].map(
        {
            "prime": -0.08,
            "near_prime": 0.00,
            "subprime": 0.08,
            "deep_subprime": 0.14,
        }
    ).to_numpy()

    scenario_util_pressure = (
        account_months["policy_stress_index"].to_numpy() * 0.08
    )

    utilization_rate = (
        product_base_util
        + risk_util_adjustment
        + scenario_util_pressure
        + rng.normal(0, 0.08, n)
    )

    utilization_rate = np.clip(utilization_rate, 0.02, 1.20)

    credit_limit = account_months["credit_limit"].to_numpy()
    current_balance = credit_limit * utilization_rate
    available_credit = np.maximum(credit_limit - current_balance, 0)

    product_min_payment_rate = account_months["product_type"].map(
        {
            "credit_card": 0.025,
            "personal_loan": 0.035,
            "auto_loan": 0.030,
            "student_refi": 0.025,
        }
    ).to_numpy()

    interest_rate = account_months["interest_rate"].to_numpy()
    rate_shock_decimal = account_months["interest_rate_shock_bps"].to_numpy() / 10000

    minimum_payment_due = (
        current_balance * product_min_payment_rate
        + current_balance * ((interest_rate + rate_shock_decimal) / 12)
    )

    monthly_income = account_months["monthly_income"].to_numpy()
    baseline_dti = account_months["baseline_dti"].to_numpy()
    resilience = account_months["baseline_financial_resilience"].to_numpy()
    employment_stability = account_months["employment_stability_score"].to_numpy()

    unemployment = account_months["unemployment_rate"].to_numpy()
    inflation = account_months["inflation_rate"].to_numpy()
    rent_index = account_months["rent_index"].to_numpy()
    wage_growth = account_months["wage_growth_rate"].to_numpy()
    sentiment = account_months["consumer_sentiment_index"].to_numpy()
    policy_stress = account_months["policy_stress_index"].to_numpy()
    unemployment_shock = account_months["unemployment_shock"].to_numpy()
    inflation_shock = account_months["inflation_shock"].to_numpy()

    macro_stress_score = (
        unemployment * 5.0
        + inflation * 4.0
        + rent_index * 0.08
        + unemployment_shock * 8.0
        + inflation_shock * 5.0
        - wage_growth * 2.5
        - (sentiment - 90) * 0.004
    )

    macro_stress_score = np.clip(macro_stress_score, 0, 1)

    cashflow_stress_score = (
        baseline_dti * 0.55
        + utilization_rate * 0.22
        + macro_stress_score * 0.35
        + policy_stress * 0.30
        - resilience * 0.35
        - employment_stability * 0.18
        + rng.normal(0, 0.04, n)
    )

    cashflow_stress_score = np.clip(cashflow_stress_score, 0, 1)

    behavioral_risk_score = (
        cashflow_stress_score * 0.55
        + utilization_rate * 0.20
        + (1 - employment_stability) * 0.15
        + policy_stress * 0.10
        + rng.normal(0, 0.04, n)
    )

    behavioral_risk_score = np.clip(behavioral_risk_score, 0, 1)

    missed_payment_probability = sigmoid(
        -3.1
        + 4.0 * cashflow_stress_score
        + 1.4 * utilization_rate
        + 1.2 * macro_stress_score
        + 0.8 * policy_stress
        - 1.1 * resilience
    )

    missed_payment_flag = rng.random(n) < missed_payment_probability

    payment_ratio = np.where(
        missed_payment_flag,
        rng.uniform(0.00, 0.55, n),
        rng.uniform(0.85, 1.40, n),
    )

    actual_payment_amount = minimum_payment_due * payment_ratio

    days_past_due = np.select(
        [
            missed_payment_flag & (behavioral_risk_score < 0.45),
            missed_payment_flag & (behavioral_risk_score >= 0.45) & (behavioral_risk_score < 0.70),
            missed_payment_flag & (behavioral_risk_score >= 0.70),
        ],
        [30, 60, 90],
        default=0,
    )

    delinquency_status = np.select(
        [
            days_past_due == 0,
            days_past_due == 30,
            days_past_due == 60,
            days_past_due >= 90,
        ],
        ["current", "30_dpd", "60_dpd", "90_plus_dpd"],
        default="current",
    )

    default_probability = sigmoid(
        -7.2
        + 3.2 * cashflow_stress_score
        + 1.4 * behavioral_risk_score
        + 0.8 * utilization_rate
        + 0.7 * macro_stress_score
        + 0.8 * policy_stress
        + 0.018 * days_past_due
        - 1.3 * resilience
    )

    default_flag = rng.random(n) < default_probability
    chargeoff_flag = default_flag & (days_past_due >= 90)

    snapshots = pd.DataFrame(
        {
            "account_month_id": [
                f"ACCTMO_{i:010d}" for i in range(1, n + 1)
            ],
            "account_id": account_months["account_id"],
            "customer_id": account_months["customer_id"],
            "scenario_id": account_months["scenario_id"],
            "month": account_months["month"],
            "current_balance": np.round(current_balance, 2),
            "available_credit": np.round(available_credit, 2),
            "utilization_rate": np.round(utilization_rate, 4),
            "minimum_payment_due": np.round(minimum_payment_due, 2),
            "actual_payment_amount": np.round(actual_payment_amount, 2),
            "days_past_due": days_past_due.astype(int),
            "delinquency_status": delinquency_status,
            "chargeoff_flag": chargeoff_flag.astype(bool),
            "default_flag": default_flag.astype(bool),
            "cashflow_stress_score": np.round(cashflow_stress_score, 4),
            "behavioral_risk_score": np.round(behavioral_risk_score, 4),
            "macro_stress_score": np.round(macro_stress_score, 4),
        }
    )

    return snapshots


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    snapshots = generate_account_monthly_snapshots(seed=42)

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM account_monthly_snapshots")
        conn.register("snapshots_df", snapshots)

        conn.execute("""
            INSERT INTO account_monthly_snapshots (
                account_month_id,
                account_id,
                customer_id,
                scenario_id,
                month,
                current_balance,
                available_credit,
                utilization_rate,
                minimum_payment_due,
                actual_payment_amount,
                days_past_due,
                delinquency_status,
                chargeoff_flag,
                default_flag,
                cashflow_stress_score,
                behavioral_risk_score,
                macro_stress_score
            )
            SELECT
                account_month_id,
                account_id,
                customer_id,
                scenario_id,
                month,
                current_balance,
                available_credit,
                utilization_rate,
                minimum_payment_due,
                actual_payment_amount,
                days_past_due,
                delinquency_status,
                chargeoff_flag,
                default_flag,
                cashflow_stress_score,
                behavioral_risk_score,
                macro_stress_score
            FROM snapshots_df
        """)

        row_count = conn.execute(
            "SELECT COUNT(*) FROM account_monthly_snapshots"
        ).fetchone()[0]

    print("Account monthly snapshots generated and loaded into DuckDB.")
    print(f"Rows inserted: {row_count}")
    print()
    print(snapshots.head())
    print()
    print(
        snapshots.groupby("scenario_id")[[
            "cashflow_stress_score",
            "behavioral_risk_score",
            "macro_stress_score",
            "default_flag",
            "chargeoff_flag",
        ]].mean()
    )