"""
Create modeling dataset for CreditScope Pro.

This script joins account behavior, customer profile, account attributes,
and bureau-level features into one machine-learning-ready dataset.
"""

from pathlib import Path

import duckdb


def create_modeling_dataset() -> None:
    """Create the modeling_dataset table inside DuckDB."""

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("DROP TABLE IF EXISTS modeling_dataset")

        conn.execute("""
            CREATE TABLE modeling_dataset AS
            SELECT
                s.account_month_id,
                s.account_id,
                s.customer_id,
                s.month,
                s.scenario_id,

                -- Target
                s.default_flag,

                -- Account monthly behavior
                s.current_balance,
                s.available_credit,
                s.utilization_rate,
                s.minimum_payment_due,
                s.actual_payment_amount,
                s.days_past_due,
                s.cashflow_stress_score,
                s.behavioral_risk_score,
                s.macro_stress_score,

                -- Account attributes
                a.product_type,
                a.credit_limit,
                a.interest_rate,
                a.term_months,
                a.origination_fico,
                a.origination_income,
                a.origination_dti,
                a.risk_tier_at_origination,

                -- Customer attributes
                c.age,
                c.state,
                c.employment_type,
                c.annual_income,
                c.monthly_income,
                c.income_band,
                c.education_level,
                c.baseline_fico,
                c.baseline_dti,
                c.baseline_financial_resilience,
                c.employment_stability_score,

                -- Bureau features
                b.fico_score,
                b.total_debt_balance,
                b.total_credit_limit,
                b.credit_utilization,
                b.num_open_accounts,
                b.num_recent_inquiries,
                b.delinquencies_12m,
                b.bankruptcy_flag

            FROM account_monthly_snapshots s

            JOIN accounts a
                ON s.account_id = a.account_id

            JOIN customers c
                ON s.customer_id = c.customer_id

            LEFT JOIN bureau_monthly b
                ON s.customer_id = b.customer_id
                AND s.month = b.month

            WHERE s.scenario_id = 'baseline'
        """)

        row_count = conn.execute(
            "SELECT COUNT(*) FROM modeling_dataset"
        ).fetchone()[0]

        default_rate = conn.execute("""
            SELECT ROUND(AVG(default_flag::INT), 4)
            FROM modeling_dataset
        """).fetchone()[0]

        print("Modeling dataset created successfully.")
        print(f"Rows: {row_count}")
        print(f"Default rate: {default_rate}")

        print()
        print(conn.execute("""
            SELECT
                product_type,
                COUNT(*) AS rows,
                ROUND(AVG(default_flag::INT), 4) AS default_rate,
                ROUND(AVG(utilization_rate), 4) AS avg_utilization,
                ROUND(AVG(fico_score), 1) AS avg_fico
            FROM modeling_dataset
            GROUP BY product_type
            ORDER BY default_rate DESC
        """).fetchdf())


if __name__ == "__main__":
    create_modeling_dataset()