"""
Score modeling dataset and populate model_scoring_audit table.

This script trains the baseline Logistic Regression model, generates
predicted probabilities of default, assigns risk bands, and stores
model audit records in DuckDB.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def assign_risk_band(predicted_pd: float) -> str:
    """Assign risk band based on predicted probability of default."""
    if predicted_pd >= 0.50:
        return "critical"
    if predicted_pd >= 0.20:
        return "high"
    if predicted_pd >= 0.05:
        return "medium"
    return "low"


def score_model_audit() -> None:
    """Train model, score baseline records, and load model audit table."""

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        df = conn.execute("""
            SELECT
                account_month_id,
                account_id,
                customer_id,
                month,
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

    target = "default_flag"

    id_columns = [
        "account_month_id",
        "account_id",
        "customer_id",
        "month",
    ]

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
    y = df[target].astype(int)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predicted_pd = pipeline.predict_proba(X)[:, 1]

    audit = pd.DataFrame(
        {
            "score_id": [f"SCORE_{i:010d}" for i in range(1, len(df) + 1)],
            "account_month_id": df["account_month_id"],
            "account_id": df["account_id"],
            "customer_id": df["customer_id"],
            "month": df["month"],
            "model_version": "logit_v1",
            "predicted_pd": np.round(predicted_pd, 6),
            "risk_band": [assign_risk_band(pd_) for pd_ in predicted_pd],
            "top_shap_feature_1": None,
            "top_shap_value_1": None,
            "top_shap_feature_2": None,
            "top_shap_value_2": None,
            "psi_score": 0.0,
            "drift_alert_flag": False,
            "fairness_alert_flag": False,
            "override_flag": False,
        }
    )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM model_scoring_audit WHERE model_version = 'logit_v1'")
        conn.register("audit_df", audit)

        conn.execute("""
            INSERT INTO model_scoring_audit (
                score_id,
                account_month_id,
                account_id,
                customer_id,
                month,
                model_version,
                predicted_pd,
                risk_band,
                top_shap_feature_1,
                top_shap_value_1,
                top_shap_feature_2,
                top_shap_value_2,
                psi_score,
                drift_alert_flag,
                fairness_alert_flag,
                override_flag
            )
            SELECT
                score_id,
                account_month_id,
                account_id,
                customer_id,
                month,
                model_version,
                predicted_pd,
                risk_band,
                top_shap_feature_1,
                top_shap_value_1,
                top_shap_feature_2,
                top_shap_value_2,
                psi_score,
                drift_alert_flag,
                fairness_alert_flag,
                override_flag
            FROM audit_df
        """)

        row_count = conn.execute("""
            SELECT COUNT(*)
            FROM model_scoring_audit
            WHERE model_version = 'logit_v1'
        """).fetchone()[0]

        print("Model scoring audit table populated.")
        print(f"Rows inserted: {row_count}")

        print()
        print(conn.execute("""
            SELECT
                risk_band,
                COUNT(*) AS accounts,
                ROUND(AVG(predicted_pd), 4) AS avg_predicted_pd
            FROM model_scoring_audit
            WHERE model_version = 'logit_v1'
            GROUP BY risk_band
            ORDER BY avg_predicted_pd DESC
        """).fetchdf())


if __name__ == "__main__":
    score_model_audit()