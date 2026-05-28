"""
Calibrate Logistic Regression predicted probabilities for CreditScope Pro.

This script trains the baseline Logistic Regression model, calibrates
predicted probabilities using isotonic calibration, compares raw vs
calibrated PDs, and writes calibrated PD values back into DuckDB.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sklearn import pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def assign_calibrated_risk_band(calibrated_pd: float) -> str:
    """Assign calibrated risk band based on calibrated probability of default."""
    if calibrated_pd >= 0.10:
        return "critical"
    if calibrated_pd >= 0.04:
        return "high"
    if calibrated_pd >= 0.015:
        return "medium"
    return "low"


def calibrate_model() -> None:
    """Train, calibrate, evaluate, and store calibrated default probabilities."""

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        df = conn.execute("""
            SELECT
                account_month_id,
                default_flag,

                current_balance,
                available_credit,
                utilization_rate,
                minimum_payment_due,
                actual_payment_amount,
                days_past_due,
                cashflow_stress_score,
                behavioral_risk_score,
                macro_stress_score,

                product_type,
                credit_limit,
                interest_rate,
                origination_fico,
                origination_income,
                origination_dti,
                risk_tier_at_origination,

                age,
                employment_type,
                annual_income,
                monthly_income,
                income_band,
                education_level,
                baseline_fico,
                baseline_dti,
                baseline_financial_resilience,
                employment_stability_score,

                fico_score,
                total_debt_balance,
                total_credit_limit,
                credit_utilization,
                num_open_accounts,
                num_recent_inquiries,
                delinquencies_12m,
                bankruptcy_flag
            FROM modeling_dataset
        """).fetchdf()

    numeric_features = [
        "current_balance",
        "available_credit",
        "utilization_rate",
        "minimum_payment_due",
        "actual_payment_amount",
        "days_past_due",
        "cashflow_stress_score",
        "behavioral_risk_score",
        "macro_stress_score",
        "credit_limit",
        "interest_rate",
        "origination_fico",
        "origination_income",
        "origination_dti",
        "age",
        "annual_income",
        "monthly_income",
        "baseline_fico",
        "baseline_dti",
        "baseline_financial_resilience",
        "employment_stability_score",
        "fico_score",
        "total_debt_balance",
        "total_credit_limit",
        "credit_utilization",
        "num_open_accounts",
        "num_recent_inquiries",
        "delinquencies_12m",
    ]

    categorical_features = [
        "product_type",
        "risk_tier_at_origination",
        "employment_type",
        "income_band",
        "education_level",
        "bankruptcy_flag",
    ]

    X = df[numeric_features + categorical_features]
    y = df["default_flag"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_calibration, X_test, y_calibration, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    base_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ]
    )

    pipeline.fit(X_train, y_train)

    raw_calibration_pd = pipeline.predict_proba(X_calibration)[:, 1]
    raw_test_pd = pipeline.predict_proba(X_test)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calibration_pd, y_calibration)

    calibrated_test_pd = calibrator.transform(raw_test_pd)
    raw_auc = roc_auc_score(y_test, raw_test_pd)
    calibrated_auc = roc_auc_score(y_test, calibrated_test_pd)

    raw_brier = brier_score_loss(y_test, raw_test_pd)
    calibrated_brier = brier_score_loss(y_test, calibrated_test_pd)

    print("Calibration evaluation complete.")
    print(f"Rows used: {len(df):,}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Calibration rows: {len(X_calibration):,}")
    print(f"Test rows: {len(X_test):,}")
    print()
    print(f"Raw ROC-AUC: {raw_auc:.4f}")
    print(f"Calibrated ROC-AUC: {calibrated_auc:.4f}")
    print()
    print(f"Raw Brier Score: {raw_brier:.6f}")
    print(f"Calibrated Brier Score: {calibrated_brier:.6f}")

    full_raw_pd = pipeline.predict_proba(X)[:, 1]
    full_calibrated_pd = calibrator.transform(full_raw_pd)

    calibration_df = pd.DataFrame(
        {
            "account_month_id": df["account_month_id"],
            "calibrated_pd": np.round(full_calibrated_pd, 6),
            "calibrated_risk_band": [
                assign_calibrated_risk_band(pd_) for pd_ in full_calibrated_pd
            ],
        }
    )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("""
            ALTER TABLE model_scoring_audit
            ADD COLUMN IF NOT EXISTS calibrated_pd DOUBLE
        """)

        conn.execute("""
            ALTER TABLE model_scoring_audit
            ADD COLUMN IF NOT EXISTS calibrated_risk_band VARCHAR
        """)

        conn.register("calibration_df", calibration_df)

        conn.execute("""
            UPDATE model_scoring_audit AS audit
            SET
                calibrated_pd = c.calibrated_pd,
                calibrated_risk_band = c.calibrated_risk_band
            FROM calibration_df AS c
            WHERE audit.account_month_id = c.account_month_id
              AND audit.model_version = 'logit_v1'
        """)

        print()
        print("Calibration values written to model_scoring_audit.")

        print()
        print(conn.execute("""
            SELECT
                calibrated_risk_band,
                COUNT(*) AS accounts,
                ROUND(AVG(calibrated_pd), 4) AS avg_calibrated_pd
            FROM model_scoring_audit
            WHERE model_version = 'logit_v1'
            GROUP BY calibrated_risk_band
            ORDER BY avg_calibrated_pd DESC
        """).fetchdf())

        print()
        print(conn.execute("""
            SELECT
                a.calibrated_risk_band,
                COUNT(*) AS accounts,
                ROUND(AVG(a.calibrated_pd), 4) AS avg_calibrated_pd,
                ROUND(AVG(m.default_flag::INT), 4) AS actual_default_rate
            FROM model_scoring_audit a
            JOIN modeling_dataset m
                ON a.account_month_id = m.account_month_id
            WHERE a.model_version = 'logit_v1'
            GROUP BY a.calibrated_risk_band
            ORDER BY avg_calibrated_pd DESC
        """).fetchdf())


if __name__ == "__main__":
    calibrate_model()