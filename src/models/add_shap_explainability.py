"""
Add SHAP explainability to CreditScope Pro model audit records.

This script trains the baseline Logistic Regression model again,
selects the highest-risk scored accounts, computes SHAP values for
those records, and updates model_scoring_audit with the top drivers.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_shap_explainability(sample_size: int = 500) -> None:
    """Compute SHAP explanations for top-risk accounts and update DuckDB."""

    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "duckdb" / "creditscope.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        df = conn.execute("""
            SELECT
                m.account_month_id,
                m.default_flag,

                m.current_balance,
                m.available_credit,
                m.utilization_rate,
                m.minimum_payment_due,
                m.actual_payment_amount,
                m.days_past_due,
                m.cashflow_stress_score,
                m.behavioral_risk_score,
                m.macro_stress_score,

                m.product_type,
                m.credit_limit,
                m.interest_rate,
                m.origination_fico,
                m.origination_income,
                m.origination_dti,
                m.risk_tier_at_origination,

                m.age,
                m.employment_type,
                m.annual_income,
                m.monthly_income,
                m.income_band,
                m.education_level,
                m.baseline_fico,
                m.baseline_dti,
                m.baseline_financial_resilience,
                m.employment_stability_score,

                m.fico_score,
                m.total_debt_balance,
                m.total_credit_limit,
                m.credit_utilization,
                m.num_open_accounts,
                m.num_recent_inquiries,
                m.delinquencies_12m,
                m.bankruptcy_flag,

                a.predicted_pd
            FROM modeling_dataset m
            JOIN model_scoring_audit a
                ON m.account_month_id = a.account_month_id
            WHERE a.model_version = 'logit_v1'
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

    # Select highest-risk records only for fast explainability.
    explain_df = df.sort_values("predicted_pd", ascending=False).head(sample_size).copy()
    X_explain = explain_df[numeric_features + categorical_features]

    transformed_train = pipeline.named_steps["preprocessor"].transform(X_train)
    transformed_explain = pipeline.named_steps["preprocessor"].transform(X_explain)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Convert sparse matrices if needed.
    if hasattr(transformed_train, "toarray"):
        transformed_train = transformed_train.toarray()

    if hasattr(transformed_explain, "toarray"):
        transformed_explain = transformed_explain.toarray()

    logistic_model = pipeline.named_steps["model"]

    # Use a small background sample to keep SHAP fast.
    background_size = min(1000, transformed_train.shape[0])
    background_idx = np.random.default_rng(42).choice(
        transformed_train.shape[0],
        size=background_size,
        replace=False,
    )

    explainer = shap.LinearExplainer(
        logistic_model,
        transformed_train[background_idx],
        feature_names=feature_names,
    )

    shap_values = explainer.shap_values(transformed_explain)

    updates = []

    for row_index, account_month_id in enumerate(explain_df["account_month_id"]):
        row_shap = shap_values[row_index]

        top_indices = np.argsort(np.abs(row_shap))[-2:][::-1]

        top_feature_1 = feature_names[top_indices[0]]
        top_value_1 = float(row_shap[top_indices[0]])

        top_feature_2 = feature_names[top_indices[1]]
        top_value_2 = float(row_shap[top_indices[1]])

        updates.append(
            {
                "account_month_id": account_month_id,
                "top_shap_feature_1": top_feature_1,
                "top_shap_value_1": round(top_value_1, 6),
                "top_shap_feature_2": top_feature_2,
                "top_shap_value_2": round(top_value_2, 6),
            }
        )

    updates_df = pd.DataFrame(updates)

    with duckdb.connect(str(db_path)) as conn:
        conn.register("updates_df", updates_df)

        conn.execute("""
            UPDATE model_scoring_audit AS audit
            SET
                top_shap_feature_1 = updates.top_shap_feature_1,
                top_shap_value_1 = updates.top_shap_value_1,
                top_shap_feature_2 = updates.top_shap_feature_2,
                top_shap_value_2 = updates.top_shap_value_2
            FROM updates_df AS updates
            WHERE audit.account_month_id = updates.account_month_id
              AND audit.model_version = 'logit_v1'
        """)

        updated_count = conn.execute("""
            SELECT COUNT(*)
            FROM model_scoring_audit
            WHERE model_version = 'logit_v1'
              AND top_shap_feature_1 IS NOT NULL
        """).fetchone()[0]

        print("SHAP explanations added to model_scoring_audit.")
        print(f"Records explained: {updated_count}")

        print()
        print(conn.execute("""
            SELECT
                account_id,
                customer_id,
                risk_band,
                ROUND(predicted_pd, 4) AS predicted_pd,
                top_shap_feature_1,
                ROUND(top_shap_value_1, 4) AS top_shap_value_1,
                top_shap_feature_2,
                ROUND(top_shap_value_2, 4) AS top_shap_value_2
            FROM model_scoring_audit
            WHERE model_version = 'logit_v1'
              AND top_shap_feature_1 IS NOT NULL
            ORDER BY predicted_pd DESC
            LIMIT 10
        """).fetchdf())


if __name__ == "__main__":
    add_shap_explainability(sample_size=500)