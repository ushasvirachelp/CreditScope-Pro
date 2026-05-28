"""
Calculate Population Stability Index (PSI) for CreditScope Pro.

This script compares earlier baseline portfolio records against later baseline
records to detect whether important model features have shifted over time.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def calculate_psi(
    reference_values: pd.Series,
    current_values: pd.Series,
    buckets: int = 10,
) -> float:
    """
    Calculate Population Stability Index for one numeric feature.

    PSI interpretation:
    - < 0.10: stable
    - 0.10 to 0.25: moderate drift
    - > 0.25: significant drift
    """

    reference_values = reference_values.dropna()
    current_values = current_values.dropna()

    if reference_values.empty or current_values.empty:
        return np.nan

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(reference_values.quantile(quantiles).to_numpy())

    if len(breakpoints) < 3:
        return np.nan

    reference_counts = pd.cut(
        reference_values,
        bins=breakpoints,
        include_lowest=True,
        duplicates="drop",
    ).value_counts(normalize=True, sort=False)

    current_counts = pd.cut(
        current_values,
        bins=breakpoints,
        include_lowest=True,
        duplicates="drop",
    ).value_counts(normalize=True, sort=False)

    reference_pct = reference_counts.to_numpy()
    current_pct = current_counts.to_numpy()

    epsilon = 0.0001
    reference_pct = np.where(reference_pct == 0, epsilon, reference_pct)
    current_pct = np.where(current_pct == 0, epsilon, current_pct)

    psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))

    return round(float(psi), 6)


def assign_drift_level(psi_score: float) -> str:
    """Assign a drift level based on PSI score."""
    if pd.isna(psi_score):
        return "not_available"
    if psi_score >= 0.25:
        return "significant_drift"
    if psi_score >= 0.10:
        return "moderate_drift"
    return "stable"


def calculate_feature_drift() -> None:
    """Calculate PSI for selected modeling features and store results in DuckDB."""

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    features = [
        "fico_score",
        "credit_utilization",
        "utilization_rate",
        "annual_income",
        "cashflow_stress_score",
        "behavioral_risk_score",
        "delinquencies_12m",
    ]

    with duckdb.connect(str(db_path)) as conn:
        df = conn.execute("""
            SELECT
                account_month_id,
                month,
                fico_score,
                credit_utilization,
                utilization_rate,
                annual_income,
                cashflow_stress_score,
                behavioral_risk_score,
                delinquencies_12m
            FROM modeling_dataset
        """).fetchdf()

    df["month"] = pd.to_datetime(df["month"])

    midpoint_month = df["month"].sort_values().unique()[len(df["month"].unique()) // 2]

    reference_df = df[df["month"] < midpoint_month].copy()
    current_df = df[df["month"] >= midpoint_month].copy()

    results = []

    for feature in features:
        psi_score = calculate_psi(
            reference_values=reference_df[feature],
            current_values=current_df[feature],
            buckets=10,
        )

        results.append(
            {
                "feature_name": feature,
                "reference_period": "early_baseline_months",
                "current_period": "late_baseline_months",
                "psi_score": psi_score,
                "drift_level": assign_drift_level(psi_score),
            }
        )

    psi_df = pd.DataFrame(results)

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_feature_psi (
                feature_name VARCHAR,
                reference_period VARCHAR,
                current_period VARCHAR,
                psi_score DOUBLE,
                drift_level VARCHAR,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("DELETE FROM model_feature_psi")

        conn.register("psi_df", psi_df)

        conn.execute("""
            INSERT INTO model_feature_psi (
                feature_name,
                reference_period,
                current_period,
                psi_score,
                drift_level
            )
            SELECT
                feature_name,
                reference_period,
                current_period,
                psi_score,
                drift_level
            FROM psi_df
        """)

        print("PSI drift monitoring table created.")
        print()
        print(conn.execute("""
            SELECT
                feature_name,
                psi_score,
                drift_level
            FROM model_feature_psi
            ORDER BY psi_score DESC
        """).fetchdf())


if __name__ == "__main__":
    calculate_feature_drift()