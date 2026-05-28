"""
Generate configurable stress scenarios for CreditScope Pro.

These scenarios represent future market or policy conditions that can
change borrower stress, payment capacity, and default risk.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def generate_policy_scenarios(
    start_month: str = "2025-01-01",
    periods: int = 24,
) -> pd.DataFrame:
    """
    Generate scenario-month records.

    Returns
    -------
    pd.DataFrame
        Scenario data by month.
    """

    months = pd.date_range(start=start_month, periods=periods, freq="MS")

    scenario_definitions = [
        {
            "scenario_id": "baseline",
            "scenario_name": "Baseline Economy",
            "student_loan_restart_flag": False,
            "student_loan_payment_multiplier": 1.00,
            "interest_rate_shock_bps": 0,
            "unemployment_shock": 0.000,
            "inflation_shock": 0.000,
            "policy_stress_index": 0.10,
        },
        {
            "scenario_id": "mild_rate_shock",
            "scenario_name": "Mild Interest Rate Shock",
            "student_loan_restart_flag": False,
            "student_loan_payment_multiplier": 1.00,
            "interest_rate_shock_bps": 75,
            "unemployment_shock": 0.003,
            "inflation_shock": 0.004,
            "policy_stress_index": 0.30,
        },
        {
            "scenario_id": "cost_of_living_stress",
            "scenario_name": "Cost of Living Stress",
            "student_loan_restart_flag": False,
            "student_loan_payment_multiplier": 1.00,
            "interest_rate_shock_bps": 50,
            "unemployment_shock": 0.005,
            "inflation_shock": 0.012,
            "policy_stress_index": 0.50,
        },
        {
            "scenario_id": "regional_recession",
            "scenario_name": "Regional Recession",
            "student_loan_restart_flag": False,
            "student_loan_payment_multiplier": 1.00,
            "interest_rate_shock_bps": 100,
            "unemployment_shock": 0.020,
            "inflation_shock": 0.006,
            "policy_stress_index": 0.70,
        },
        {
            "scenario_id": "severe_credit_stress",
            "scenario_name": "Severe Credit Stress",
            "student_loan_restart_flag": False,
            "student_loan_payment_multiplier": 1.00,
            "interest_rate_shock_bps": 150,
            "unemployment_shock": 0.030,
            "inflation_shock": 0.015,
            "policy_stress_index": 0.90,
        },
    ]

    rows = []
    counter = 1

    for month_index, month in enumerate(months):
        # Scenarios activate more strongly in the second half of the simulation.
        activation_factor = 0.35 if month_index < 12 else 1.00

        for scenario in scenario_definitions:
            rows.append(
                {
                    "scenario_month_id": f"SCENARIO_{counter:08d}",
                    "scenario_id": scenario["scenario_id"],
                    "scenario_name": scenario["scenario_name"],
                    "month": month.date(),
                    "student_loan_restart_flag": scenario["student_loan_restart_flag"],
                    "student_loan_payment_multiplier": scenario[
                        "student_loan_payment_multiplier"
                    ],
                    "interest_rate_shock_bps": round(
                        scenario["interest_rate_shock_bps"] * activation_factor, 2
                    ),
                    "unemployment_shock": round(
                        scenario["unemployment_shock"] * activation_factor, 4
                    ),
                    "inflation_shock": round(
                        scenario["inflation_shock"] * activation_factor, 4
                    ),
                    "policy_stress_index": round(
                        scenario["policy_stress_index"] * activation_factor, 4
                    ),
                }
            )

            counter += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        scenarios = generate_policy_scenarios(start_month="2025-01-01", periods=24)

        conn.execute("DELETE FROM policy_scenarios")
        conn.register("scenarios_df", scenarios)

        conn.execute("""
            INSERT INTO policy_scenarios (
                scenario_month_id,
                scenario_id,
                scenario_name,
                month,
                student_loan_restart_flag,
                student_loan_payment_multiplier,
                interest_rate_shock_bps,
                unemployment_shock,
                inflation_shock,
                policy_stress_index
            )
            SELECT
                scenario_month_id,
                scenario_id,
                scenario_name,
                month,
                student_loan_restart_flag,
                student_loan_payment_multiplier,
                interest_rate_shock_bps,
                unemployment_shock,
                inflation_shock,
                policy_stress_index
            FROM scenarios_df
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM policy_scenarios").fetchone()[0]

    print("Policy scenarios generated and loaded into DuckDB.")
    print(f"Rows inserted: {row_count}")
    print()
    print(scenarios.head(10))
    print()
    print(
        scenarios.groupby("scenario_id")[
            [
                "interest_rate_shock_bps",
                "unemployment_shock",
                "inflation_shock",
                "policy_stress_index",
            ]
        ].max()
    )