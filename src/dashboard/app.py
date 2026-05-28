"""
CreditScope Pro Executive Dashboard.

Streamlit dashboard for portfolio risk, stress scenarios,
payment behavior, and model governance metrics.
"""

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "creditscope.duckdb"


@st.cache_data
def run_query(query: str) -> pd.DataFrame:
    """Run a DuckDB query and return a DataFrame."""
    with duckdb.connect(str(DB_PATH)) as conn:
        return conn.execute(query).fetchdf()


st.set_page_config(
    page_title="CreditScope Pro",
    page_icon="💳",
    layout="wide",
)

st.title("CreditScope Pro")
st.caption("AI-driven credit risk intelligence and model governance platform")

st.divider()

# -----------------------------
# Sidebar Controls
# -----------------------------

available_scenarios = run_query("""
    SELECT DISTINCT scenario_id
    FROM account_monthly_snapshots
    ORDER BY scenario_id
""")

selected_scenario = st.sidebar.selectbox(
    "Select Scenario",
    available_scenarios["scenario_id"].tolist(),
    index=0,
    key="scenario_filter",
)

available_risk_bands = run_query("""
    SELECT DISTINCT risk_band
    FROM model_scoring_audit
    WHERE model_version = 'logit_v1'
    ORDER BY risk_band
""")

risk_band_options = ["all"] + available_risk_bands["risk_band"].tolist()

selected_risk_band = st.sidebar.selectbox(
    "Select Risk Band",
    risk_band_options,
    index=0,
    key="risk_band_filter",
)

available_products = run_query("""
    SELECT DISTINCT product_type
    FROM accounts
    ORDER BY product_type
""")

product_options = ["all"] + available_products["product_type"].tolist()

selected_product = st.sidebar.selectbox(
    "Select Product",
    product_options,
    index=0,
    key="product_filter",
)

st.sidebar.markdown("### Dashboard Controls")
st.sidebar.write(f"Current scenario: `{selected_scenario}`")
st.sidebar.write(f"Current risk band: `{selected_risk_band}`")
st.sidebar.write(f"Current product: `{selected_product}`")

st.sidebar.caption(
    "Scenario controls stress and payment sections. "
    "Risk band controls model governance. "
    "Product controls portfolio, scenario, payment, and product views."
)

# Reusable filter strings
product_account_filter = ""
product_snapshot_filter = ""
product_payment_filter = ""
product_modeling_filter = ""

if selected_product != "all":
    product_account_filter = f"WHERE product_type = '{selected_product}'"
    product_snapshot_filter = f"AND a.product_type = '{selected_product}'"
    product_payment_filter = f"AND a.product_type = '{selected_product}'"
    product_modeling_filter = f"WHERE product_type = '{selected_product}'"

# -----------------------------
# Executive KPI Cards
# -----------------------------

row_counts = run_query(f"""
    SELECT
        COUNT(DISTINCT customer_id) AS customers,
        COUNT(*) AS accounts
    FROM accounts
    {product_account_filter}
""")

selected_metrics = run_query(f"""
    SELECT
        ROUND(AVG(s.default_flag::INT), 4) AS selected_default_rate,
        ROUND(AVG(s.chargeoff_flag::INT), 4) AS selected_chargeoff_rate,
        ROUND(AVG(s.cashflow_stress_score), 4) AS selected_cashflow_stress
    FROM account_monthly_snapshots s
    JOIN accounts a
        ON s.account_id = a.account_id
    WHERE s.scenario_id = '{selected_scenario}'
    {product_snapshot_filter}
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    customers = row_counts["customers"].iloc[0]
    label = "Unique Customers" if selected_product == "all" else "Customers with Selected Product"
    st.metric(label, f"{customers:,.0f}")

with col2:
    accounts = row_counts["accounts"].iloc[0]
    label = "Total Accounts" if selected_product == "all" else "Selected Product Accounts"
    st.metric(label, f"{accounts:,.0f}")

with col3:
    selected_default = selected_metrics["selected_default_rate"].iloc[0]
    st.metric("Selected Scenario Default Rate", f"{selected_default:.2%}")

with col4:
    selected_stress = selected_metrics["selected_cashflow_stress"].iloc[0]
    st.metric("Avg Cashflow Stress", f"{selected_stress:.2%}")

st.info(
    f"Currently viewing `{selected_scenario}` scenario"
    + (
        f" for `{selected_product}` accounts."
        if selected_product != "all"
        else " across the full account portfolio."
    )
    + " Customer counts are distinct borrowers; account counts represent credit products being analyzed."
)

st.divider()

st.subheader("Executive Takeaways")

takeaway_default_rate = selected_metrics["selected_default_rate"].iloc[0]
takeaway_cashflow_stress = selected_metrics["selected_cashflow_stress"].iloc[0]

if selected_scenario == "severe_credit_stress":
    scenario_message = "The selected stress scenario shows elevated borrower pressure compared with baseline conditions."
else:
    scenario_message = "The selected baseline scenario represents normal portfolio behavior."

if selected_product == "all":
    product_message = "The view covers the full account portfolio."
else:
    product_message = f"The view is focused on `{selected_product}` accounts."

st.markdown(
    f"""
    - **Scenario view:** {scenario_message}
    - **Product scope:** {product_message}
    - **Default risk:** The selected view has a default rate of **{takeaway_default_rate:.2%}**.
    - **Cashflow stress:** The average borrower cashflow stress score is **{takeaway_cashflow_stress:.2%}**.
    - **Interpretation:** Higher stress, utilization, and lower FICO scores are associated with higher model-predicted default risk.
    """
)

st.subheader("Portfolio Composition")

portfolio_composition = run_query(f"""
    SELECT
        product_type,
        COUNT(*) AS accounts,
        COUNT(DISTINCT customer_id) AS customers,
        ROUND(AVG(credit_limit), 2) AS avg_credit_limit,
        ROUND(AVG(interest_rate), 4) AS avg_interest_rate
    FROM accounts
    {product_account_filter}
    GROUP BY product_type
    ORDER BY accounts DESC
""")

st.dataframe(portfolio_composition, use_container_width=True)

if not portfolio_composition.empty:
    composition_chart = portfolio_composition.set_index("product_type")[["accounts"]]
    st.bar_chart(composition_chart)

st.divider()

# -----------------------------
# Selected Scenario Overview
# -----------------------------

st.subheader(f"Selected Scenario Overview: {selected_scenario}")

selected_scenario_summary = run_query(f"""
    SELECT
        s.scenario_id,
        COUNT(*) AS rows,
        ROUND(AVG(s.default_flag::INT), 4) AS default_rate,
        ROUND(AVG(s.chargeoff_flag::INT), 4) AS chargeoff_rate,
        ROUND(AVG(s.cashflow_stress_score), 4) AS avg_cashflow_stress,
        ROUND(AVG(s.behavioral_risk_score), 4) AS avg_behavioral_risk,
        ROUND(AVG(s.macro_stress_score), 4) AS avg_macro_stress
    FROM account_monthly_snapshots s
    JOIN accounts a
        ON s.account_id = a.account_id
    WHERE s.scenario_id = '{selected_scenario}'
    {product_snapshot_filter}
    GROUP BY s.scenario_id
""")

st.dataframe(selected_scenario_summary, use_container_width=True)

st.divider()

# -----------------------------
# Scenario Risk Comparison
# -----------------------------

st.subheader("Scenario Risk Comparison")

scenario_risk = run_query(f"""
    SELECT
        s.scenario_id,
        COUNT(*) AS rows,
        ROUND(AVG(s.default_flag::INT), 4) AS default_rate,
        ROUND(AVG(s.chargeoff_flag::INT), 4) AS chargeoff_rate,
        ROUND(AVG(s.cashflow_stress_score), 4) AS avg_cashflow_stress,
        ROUND(AVG(s.behavioral_risk_score), 4) AS avg_behavioral_risk,
        ROUND(AVG(s.macro_stress_score), 4) AS avg_macro_stress
    FROM account_monthly_snapshots s
    JOIN accounts a
        ON s.account_id = a.account_id
    WHERE 1 = 1
    {product_snapshot_filter}
    GROUP BY s.scenario_id
    ORDER BY default_rate DESC
""")

st.dataframe(scenario_risk, use_container_width=True)

chart_data = scenario_risk.set_index("scenario_id")[
    ["default_rate", "chargeoff_rate", "avg_cashflow_stress", "avg_behavioral_risk"]
]

st.bar_chart(chart_data)

st.divider()

st.subheader("Monthly Default Trend")

monthly_default_trend = run_query(f"""
    SELECT
        s.month,
        s.scenario_id,
        ROUND(AVG(s.default_flag::INT), 4) AS default_rate,
        ROUND(AVG(s.chargeoff_flag::INT), 4) AS chargeoff_rate,
        ROUND(AVG(s.cashflow_stress_score), 4) AS avg_cashflow_stress
    FROM account_monthly_snapshots s
    JOIN accounts a
        ON s.account_id = a.account_id
    WHERE 1 = 1
    {product_snapshot_filter}
    GROUP BY s.month, s.scenario_id
    ORDER BY s.month, s.scenario_id
""")

st.dataframe(monthly_default_trend, use_container_width=True)

if not monthly_default_trend.empty:
    trend_chart = monthly_default_trend.pivot(
        index="month",
        columns="scenario_id",
        values="default_rate",
    )
    st.line_chart(trend_chart)

# -----------------------------
# Payment Behavior
# -----------------------------

st.subheader(f"Payment Behavior: {selected_scenario}")

payment_behavior_selected = run_query(f"""
    SELECT
        p.scenario_id,
        COUNT(*) AS payment_events,
        ROUND(AVG(p.missed_payment_flag::INT), 4) AS missed_payment_rate,
        ROUND(AVG(p.partial_payment_flag::INT), 4) AS partial_payment_rate,
        ROUND(AVG(p.late_payment_flag::INT), 4) AS late_payment_rate,
        ROUND(AVG(p.days_late), 2) AS avg_days_late
    FROM payment_events p
    JOIN accounts a
        ON p.account_id = a.account_id
    WHERE p.scenario_id = '{selected_scenario}'
    {product_payment_filter}
    GROUP BY p.scenario_id
""")

st.dataframe(payment_behavior_selected, use_container_width=True)

st.subheader("Payment Behavior by Scenario")

payment_behavior = run_query(f"""
    SELECT
        p.scenario_id,
        COUNT(*) AS payment_events,
        ROUND(AVG(p.missed_payment_flag::INT), 4) AS missed_payment_rate,
        ROUND(AVG(p.partial_payment_flag::INT), 4) AS partial_payment_rate,
        ROUND(AVG(p.late_payment_flag::INT), 4) AS late_payment_rate,
        ROUND(AVG(p.days_late), 2) AS avg_days_late
    FROM payment_events p
    JOIN accounts a
        ON p.account_id = a.account_id
    WHERE 1 = 1
    {product_payment_filter}
    GROUP BY p.scenario_id
    ORDER BY late_payment_rate DESC
""")

st.dataframe(payment_behavior, use_container_width=True)

payment_chart = payment_behavior.set_index("scenario_id")[
    ["missed_payment_rate", "partial_payment_rate", "late_payment_rate"]
]

st.bar_chart(payment_chart)

st.divider()

# -----------------------------
# Model Governance
# -----------------------------

st.subheader("Model Risk Band Validation")

risk_band_filter = ""
if selected_risk_band != "all":
    risk_band_filter = f"AND msa.risk_band = '{selected_risk_band}'"

risk_product_filter = ""
if selected_product != "all":
    risk_product_filter = f"AND m.product_type = '{selected_product}'"

risk_band_validation = run_query(f"""
    SELECT
        msa.risk_band,
        COUNT(*) AS accounts,
        ROUND(AVG(msa.predicted_pd), 4) AS avg_predicted_pd,
        ROUND(AVG(m.default_flag::INT), 4) AS actual_default_rate,
        ROUND(AVG(m.utilization_rate), 4) AS avg_utilization,
        ROUND(AVG(m.fico_score), 1) AS avg_fico
    FROM model_scoring_audit msa
    JOIN modeling_dataset m
        ON msa.account_month_id = m.account_month_id
    WHERE msa.model_version = 'logit_v1'
    {risk_band_filter}
    {risk_product_filter}
    GROUP BY msa.risk_band
    ORDER BY avg_predicted_pd DESC
""")

st.dataframe(risk_band_validation, use_container_width=True)

if not risk_band_validation.empty:
    risk_chart = risk_band_validation.set_index("risk_band")[
        ["avg_predicted_pd", "actual_default_rate"]
    ]
    st.bar_chart(risk_chart)

st.subheader("Risk Band Distribution")

risk_distribution_product_filter = ""
if selected_product != "all":
    risk_distribution_product_filter = f"AND m.product_type = '{selected_product}'"

risk_band_distribution = run_query(f"""
    SELECT
        msa.risk_band,
        COUNT(*) AS accounts
    FROM model_scoring_audit msa
    JOIN modeling_dataset m
        ON msa.account_month_id = m.account_month_id
    WHERE msa.model_version = 'logit_v1'
    {risk_distribution_product_filter}
    GROUP BY msa.risk_band
    ORDER BY
        CASE msa.risk_band
            WHEN 'critical' THEN 1
            WHEN 'high' THEN 2
            WHEN 'medium' THEN 3
            WHEN 'low' THEN 4
            ELSE 5
        END
""")

st.dataframe(risk_band_distribution, use_container_width=True)

if not risk_band_distribution.empty:
    st.bar_chart(risk_band_distribution.set_index("risk_band"))

st.divider()

st.subheader("High-Risk Account Spotlight")

spotlight_product_filter = ""
if selected_product != "all":
    spotlight_product_filter = f"AND m.product_type = '{selected_product}'"

spotlight_risk_filter = ""
if selected_risk_band != "all":
    spotlight_risk_filter = f"AND msa.risk_band = '{selected_risk_band}'"

high_risk_accounts = run_query(f"""
    SELECT
        msa.account_id,
        msa.customer_id,
        m.product_type,
        msa.risk_band,
        ROUND(msa.predicted_pd, 4) AS predicted_pd,
        m.default_flag,
        ROUND(m.utilization_rate, 4) AS utilization_rate,
        ROUND(m.fico_score, 1) AS fico_score,
        ROUND(m.cashflow_stress_score, 4) AS cashflow_stress_score,
        ROUND(m.behavioral_risk_score, 4) AS behavioral_risk_score,
        msa.top_shap_feature_1,
        ROUND(msa.top_shap_value_1, 4) AS top_shap_value_1,
        msa.top_shap_feature_2,
        ROUND(msa.top_shap_value_2, 4) AS top_shap_value_2
    FROM model_scoring_audit msa
    JOIN modeling_dataset m
        ON msa.account_month_id = m.account_month_id
    WHERE msa.model_version = 'logit_v1'
    {spotlight_product_filter}
    {spotlight_risk_filter}
    ORDER BY msa.predicted_pd DESC
    LIMIT 25
""")

st.dataframe(high_risk_accounts, use_container_width=True)

st.subheader("Product-Level Baseline Risk")

product_risk = run_query(f"""
    SELECT
        product_type,
        COUNT(*) AS rows,
        ROUND(AVG(default_flag::INT), 4) AS default_rate,
        ROUND(AVG(utilization_rate), 4) AS avg_utilization,
        ROUND(AVG(fico_score), 1) AS avg_fico
    FROM modeling_dataset
    {product_modeling_filter}
    GROUP BY product_type
    ORDER BY default_rate DESC
""")

st.dataframe(product_risk, use_container_width=True)

if not product_risk.empty:
    product_chart = product_risk.set_index("product_type")[
        ["default_rate", "avg_utilization"]
    ]
    st.bar_chart(product_chart)

st.divider()

st.caption(
    "CreditScope Pro simulates borrower-level credit behavior under macroeconomic stress, "
    "predicts default risk, and stores auditable model governance outputs."
)