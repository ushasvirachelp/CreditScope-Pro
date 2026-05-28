"""
Portfolio summary metrics for CreditScope Pro.

This script prints executive-level credit risk, stress scenario,
payment behavior, and model governance metrics from DuckDB.
"""

from pathlib import Path

import duckdb


def run_portfolio_summary() -> None:
    """Run executive summary queries for CreditScope Pro."""

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        print("\n=== CreditScope Pro Portfolio Summary ===\n")

        print("1. Table Row Counts")
        print(conn.execute("""
            SELECT 'customers' AS table_name, COUNT(*) AS rows FROM customers
            UNION ALL
            SELECT 'accounts', COUNT(*) FROM accounts
            UNION ALL
            SELECT 'macro_monthly', COUNT(*) FROM macro_monthly
            UNION ALL
            SELECT 'policy_scenarios', COUNT(*) FROM policy_scenarios
            UNION ALL
            SELECT 'account_monthly_snapshots', COUNT(*) FROM account_monthly_snapshots
            UNION ALL
            SELECT 'payment_events', COUNT(*) FROM payment_events
            UNION ALL
            SELECT 'bureau_monthly', COUNT(*) FROM bureau_monthly
            UNION ALL
            SELECT 'modeling_dataset', COUNT(*) FROM modeling_dataset
            UNION ALL
            SELECT 'model_scoring_audit', COUNT(*) FROM model_scoring_audit
        """).fetchdf())

        print("\n2. Scenario Risk Comparison")
        print(conn.execute("""
            SELECT
                scenario_id,
                COUNT(*) AS rows,
                ROUND(AVG(default_flag::INT), 4) AS default_rate,
                ROUND(AVG(chargeoff_flag::INT), 4) AS chargeoff_rate,
                ROUND(AVG(cashflow_stress_score), 4) AS avg_cashflow_stress,
                ROUND(AVG(behavioral_risk_score), 4) AS avg_behavioral_risk,
                ROUND(AVG(macro_stress_score), 4) AS avg_macro_stress
            FROM account_monthly_snapshots
            GROUP BY scenario_id
            ORDER BY default_rate DESC
        """).fetchdf())

        print("\n3. Payment Behavior by Scenario")
        print(conn.execute("""
            SELECT
                scenario_id,
                COUNT(*) AS payment_events,
                ROUND(AVG(missed_payment_flag::INT), 4) AS missed_payment_rate,
                ROUND(AVG(partial_payment_flag::INT), 4) AS partial_payment_rate,
                ROUND(AVG(late_payment_flag::INT), 4) AS late_payment_rate,
                ROUND(AVG(days_late), 2) AS avg_days_late
            FROM payment_events
            GROUP BY scenario_id
            ORDER BY late_payment_rate DESC
        """).fetchdf())

        print("\n4. Model Risk Band Validation")
        print(conn.execute("""
            SELECT
                a.risk_band,
                COUNT(*) AS accounts,
                ROUND(AVG(a.predicted_pd), 4) AS avg_predicted_pd,
                ROUND(AVG(m.default_flag::INT), 4) AS actual_default_rate,
                ROUND(AVG(m.utilization_rate), 4) AS avg_utilization,
                ROUND(AVG(m.fico_score), 1) AS avg_fico
            FROM model_scoring_audit a
            JOIN modeling_dataset m
                ON a.account_month_id = m.account_month_id
            WHERE a.model_version = 'logit_v1'
            GROUP BY a.risk_band
            ORDER BY avg_predicted_pd DESC
        """).fetchdf())

        print("\n5. Product-Level Baseline Risk")
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
    run_portfolio_summary()