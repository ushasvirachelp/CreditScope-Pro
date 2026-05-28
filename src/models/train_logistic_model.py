"""
Train baseline Logistic Regression model for CreditScope Pro.

This model predicts account-level default risk using the baseline
modeling_dataset created from DuckDB.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_logistic_model() -> None:
    """Train and evaluate a Logistic Regression default prediction model."""

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        df = conn.execute("""
            SELECT
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

    X_train, X_test, y_train, y_test = train_test_split(
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

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)

    print("Logistic Regression model trained successfully.")
    print(f"Rows used: {len(df):,}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Default rate: {y.mean():.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Risk decile check
    results = pd.DataFrame(
        {
            "actual_default": y_test.to_numpy(),
            "predicted_pd": y_proba,
        }
    )

    results["risk_decile"] = pd.qcut(
        results["predicted_pd"],
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1

    decile_summary = (
        results.groupby("risk_decile", as_index=False)
        .agg(
            accounts=("actual_default", "count"),
            default_rate=("actual_default", "mean"),
            avg_predicted_pd=("predicted_pd", "mean"),
        )
        .sort_values("risk_decile", ascending=False)
    )

    print()
    print("Risk Decile Summary:")
    print(decile_summary)


if __name__ == "__main__":
    train_logistic_model()