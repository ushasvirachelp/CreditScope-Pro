"""
Generate synthetic customer-level data for CreditScope Pro.

This module creates borrower profiles with demographic, income,
credit, student loan, and financial resilience characteristics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import duckdb


def generate_customers(n_customers: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer records.

    Parameters
    ----------
    n_customers:
        Number of customers to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Customer-level synthetic dataset.
    """

    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST_{i:07d}" for i in range(1, n_customers + 1)]

    ages = rng.integers(21, 71, size=n_customers)
    birth_years = 2026 - ages

    states = rng.choice(
        ["NY", "NJ", "MA", "CT", "PA", "RI", "CA", "TX", "FL", "IL"],
        size=n_customers,
        p=[0.16, 0.10, 0.09, 0.06, 0.08, 0.03, 0.16, 0.12, 0.12, 0.08],
    )

    zip3_by_state = {
        "NY": ["100", "101", "112", "113", "146"],
        "NJ": ["070", "071", "073", "085"],
        "MA": ["021", "022", "027", "010"],
        "CT": ["061", "065", "068"],
        "PA": ["191", "152", "171"],
        "RI": ["028", "029"],
        "CA": ["900", "902", "941", "958"],
        "TX": ["733", "750", "770", "787"],
        "FL": ["320", "331", "336", "347"],
        "IL": ["600", "606", "627"],
    }

    zip3 = [rng.choice(zip3_by_state[state]) for state in states]

    employment_types = rng.choice(
        ["salaried", "hourly", "self_employed", "student", "unemployed"],
        size=n_customers,
        p=[0.52, 0.25, 0.10, 0.08, 0.05],
    )

    education_levels = rng.choice(
        ["high_school", "associate", "bachelor", "graduate"],
        size=n_customers,
        p=[0.28, 0.18, 0.36, 0.18],
    )

    income_mu_by_employment = {
        "salaried": 11.05,
        "hourly": 10.55,
        "self_employed": 10.85,
        "student": 10.10,
        "unemployed": 9.65,
    }

    annual_income = np.array(
        [
            rng.lognormal(mean=income_mu_by_employment[emp], sigma=0.45)
            for emp in employment_types
        ]
    )

    annual_income = np.clip(annual_income, 18_000, 240_000)
    monthly_income = annual_income / 12

    income_band = pd.cut(
        annual_income,
        bins=[0, 40_000, 80_000, 130_000, np.inf],
        labels=["low", "middle", "upper_middle", "high"],
    ).astype(str)

    student_loan_probability = np.where(
        np.isin(education_levels, ["bachelor", "graduate"]),
        0.42,
        0.18,
    )

    student_loan_flag = rng.random(n_customers) < student_loan_probability

    baseline_fico = (
        650
        + (annual_income - annual_income.mean()) / annual_income.std() * 35
        + rng.normal(0, 55, size=n_customers)
    )

    baseline_fico = np.clip(baseline_fico, 520, 830).round().astype(int)

    baseline_dti = rng.beta(2.5, 5.5, size=n_customers)
    baseline_dti += np.where(annual_income < 40_000, 0.10, 0)
    baseline_dti += np.where(student_loan_flag, 0.06, 0)
    baseline_dti = np.clip(baseline_dti, 0.05, 0.75)

    baseline_financial_resilience = (
        0.45
        + (baseline_fico - 650) / 300
        + (annual_income - annual_income.mean()) / annual_income.std() * 0.10
        - baseline_dti * 0.35
        + rng.normal(0, 0.08, size=n_customers)
    )

    baseline_financial_resilience = np.clip(baseline_financial_resilience, 0, 1)

    employment_stability_score = np.select(
        [
            employment_types == "salaried",
            employment_types == "hourly",
            employment_types == "self_employed",
            employment_types == "student",
            employment_types == "unemployed",
        ],
        [0.82, 0.62, 0.55, 0.48, 0.18],
        default=0.50,
    )

    employment_stability_score = np.clip(
        employment_stability_score + rng.normal(0, 0.08, size=n_customers),
        0,
        1,
    )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "birth_year": birth_years,
            "age": ages,
            "state": states,
            "zip3": zip3,
            "employment_type": employment_types,
            "annual_income": annual_income.round(2),
            "monthly_income": monthly_income.round(2),
            "income_band": income_band,
            "education_level": education_levels,
            "student_loan_flag": student_loan_flag,
            "baseline_fico": baseline_fico,
            "baseline_dti": baseline_dti.round(4),
            "baseline_financial_resilience": baseline_financial_resilience.round(4),
            "employment_stability_score": employment_stability_score.round(4),
        }
    )

    return customers


if __name__ == "__main__":
    df = generate_customers(n_customers=10_000, seed=42)

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM customers")
        conn.register("customers_df", df)
        conn.execute("""
            INSERT INTO customers (
                customer_id, birth_year, age, state, zip3, employment_type,
                annual_income, monthly_income, income_band, education_level,
                student_loan_flag, baseline_fico, baseline_dti,
                baseline_financial_resilience, employment_stability_score
            )
            SELECT * FROM customers_df
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]

    print("Customers generated and loaded into DuckDB.")
    print(f"Rows inserted: {row_count}")