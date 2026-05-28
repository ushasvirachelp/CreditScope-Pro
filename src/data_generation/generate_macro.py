"""
Generate monthly macroeconomic data for CreditScope Pro.

This module creates regional macroeconomic conditions by month,
state, and ZIP3. These variables will later influence borrower stress,
payment behavior, delinquency, and default risk.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def generate_macro_monthly(
    start_month: str = "2025-01-01",
    periods: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic monthly macroeconomic data.

    Parameters
    ----------
    start_month:
        First month in the macro simulation.
    periods:
        Number of monthly periods to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Monthly macroeconomic dataset by state and ZIP3.
    """

    rng = np.random.default_rng(seed)

    months = pd.date_range(start=start_month, periods=periods, freq="MS")

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

    state_base = {
        "NY": {"unemp": 0.043, "rent": 1.18, "sentiment": 96},
        "NJ": {"unemp": 0.041, "rent": 1.12, "sentiment": 98},
        "MA": {"unemp": 0.038, "rent": 1.14, "sentiment": 99},
        "CT": {"unemp": 0.040, "rent": 1.08, "sentiment": 97},
        "PA": {"unemp": 0.044, "rent": 0.96, "sentiment": 95},
        "RI": {"unemp": 0.045, "rent": 1.02, "sentiment": 94},
        "CA": {"unemp": 0.047, "rent": 1.25, "sentiment": 94},
        "TX": {"unemp": 0.039, "rent": 0.94, "sentiment": 101},
        "FL": {"unemp": 0.036, "rent": 1.04, "sentiment": 100},
        "IL": {"unemp": 0.046, "rent": 0.98, "sentiment": 93},
    }

    rows = []
    macro_counter = 1

    for month_index, month in enumerate(months):
        # Gentle economy-wide trend over time
        inflation_trend = 0.028 + 0.004 * np.sin(month_index / 3)
        interest_rate_trend = 0.045 + 0.003 * np.sin(month_index / 4)

        # Mild stress period in later months to make the data dynamic
        stress_period = month_index >= 14

        for state, zip3_list in zip3_by_state.items():
            base = state_base[state]

            for zip3 in zip3_list:
                local_noise = rng.normal(0, 0.003)

                unemployment_rate = base["unemp"] + local_noise
                inflation_rate = inflation_trend + rng.normal(0, 0.002)
                interest_rate_index = interest_rate_trend + rng.normal(0, 0.0015)
                rent_index = base["rent"] + rng.normal(0, 0.04)
                wage_growth_rate = 0.032 + rng.normal(0, 0.006)
                consumer_sentiment_index = base["sentiment"] + rng.normal(0, 4)

                if stress_period:
                    unemployment_rate += rng.uniform(0.006, 0.018)
                    inflation_rate += rng.uniform(0.003, 0.012)
                    interest_rate_index += rng.uniform(0.002, 0.009)
                    rent_index += rng.uniform(0.02, 0.08)
                    wage_growth_rate -= rng.uniform(0.004, 0.012)
                    consumer_sentiment_index -= rng.uniform(4, 12)

                recession_flag = (
                    unemployment_rate > 0.058
                    or inflation_rate > 0.042
                    or consumer_sentiment_index < 88
                )

                rows.append(
                    {
                        "macro_id": f"MACRO_{macro_counter:08d}",
                        "month": month.date(),
                        "state": state,
                        "zip3": zip3,
                        "unemployment_rate": round(float(unemployment_rate), 4),
                        "inflation_rate": round(float(inflation_rate), 4),
                        "interest_rate_index": round(float(interest_rate_index), 4),
                        "rent_index": round(float(rent_index), 4),
                        "wage_growth_rate": round(float(wage_growth_rate), 4),
                        "consumer_sentiment_index": round(float(consumer_sentiment_index), 2),
                        "recession_flag": bool(recession_flag),
                    }
                )

                macro_counter += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        macro = generate_macro_monthly(start_month="2025-01-01", periods=24, seed=42)

        conn.execute("DELETE FROM macro_monthly")
        conn.register("macro_df", macro)

        conn.execute("""
            INSERT INTO macro_monthly (
                macro_id,
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
            )
            SELECT
                macro_id,
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
            FROM macro_df
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM macro_monthly").fetchone()[0]

    print("Macro monthly data generated and loaded into DuckDB.")
    print(f"Rows inserted: {row_count}")
    print()
    print(macro.head())
    print()
    print(
        macro[
            [
                "unemployment_rate",
                "inflation_rate",
                "interest_rate_index",
                "rent_index",
                "wage_growth_rate",
                "consumer_sentiment_index",
            ]
        ].describe()
    )