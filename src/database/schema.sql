-- CreditScope Pro Database Schema
-- Phase 1: Synthetic Credit Risk Intelligence Platform

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR PRIMARY KEY,
    birth_year INTEGER,
    age INTEGER,
    state VARCHAR,
    zip3 VARCHAR,
    employment_type VARCHAR,
    annual_income DOUBLE,
    monthly_income DOUBLE,
    income_band VARCHAR,
    education_level VARCHAR,
    student_loan_flag BOOLEAN,
    baseline_fico INTEGER,
    baseline_dti DOUBLE,
    baseline_financial_resilience DOUBLE,
    employment_stability_score DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR,
    product_type VARCHAR,
    open_date DATE,
    credit_limit DOUBLE,
    interest_rate DOUBLE,
    term_months INTEGER,
    origination_fico INTEGER,
    origination_income DOUBLE,
    origination_dti DOUBLE,
    risk_tier_at_origination VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS student_loan_exposure (
    student_loan_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR,
    student_loan_balance DOUBLE,
    monthly_student_payment DOUBLE,
    repayment_plan VARCHAR,
    forbearance_flag BOOLEAN,
    restart_month DATE,
    payment_to_income_ratio DOUBLE,
    student_loan_stress_score DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS macro_monthly (
    macro_id VARCHAR PRIMARY KEY,
    month DATE,
    state VARCHAR,
    zip3 VARCHAR,
    unemployment_rate DOUBLE,
    inflation_rate DOUBLE,
    interest_rate_index DOUBLE,
    rent_index DOUBLE,
    wage_growth_rate DOUBLE,
    consumer_sentiment_index DOUBLE,
    recession_flag BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS policy_scenarios (
    scenario_month_id VARCHAR PRIMARY KEY,
    scenario_id VARCHAR,
    scenario_name VARCHAR,
    month DATE,
    student_loan_restart_flag BOOLEAN,
    student_loan_payment_multiplier DOUBLE,
    interest_rate_shock_bps DOUBLE,
    unemployment_shock DOUBLE,
    inflation_shock DOUBLE,
    policy_stress_index DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bureau_monthly (
    bureau_snapshot_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR,
    month DATE,
    fico_score INTEGER,
    total_debt_balance DOUBLE,
    total_credit_limit DOUBLE,
    credit_utilization DOUBLE,
    num_open_accounts INTEGER,
    num_recent_inquiries INTEGER,
    delinquencies_12m INTEGER,
    bankruptcy_flag BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS account_monthly_snapshots (
    account_month_id VARCHAR PRIMARY KEY,
    account_id VARCHAR,
    customer_id VARCHAR,
    scenario_id VARCHAR,
    month DATE,
    current_balance DOUBLE,
    available_credit DOUBLE,
    utilization_rate DOUBLE,
    minimum_payment_due DOUBLE,
    actual_payment_amount DOUBLE,
    days_past_due INTEGER,
    delinquency_status VARCHAR,
    chargeoff_flag BOOLEAN,
    default_flag BOOLEAN,
    cashflow_stress_score DOUBLE,
    behavioral_risk_score DOUBLE,
    macro_stress_score DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS payment_events (
    payment_id VARCHAR PRIMARY KEY,
    account_id VARCHAR,
    customer_id VARCHAR,
    payment_date DATE,
    due_date DATE,
    scheduled_payment_amount DOUBLE,
    actual_payment_amount DOUBLE,
    payment_channel VARCHAR,
    missed_payment_flag BOOLEAN,
    partial_payment_flag BOOLEAN,
    late_payment_flag BOOLEAN,
    days_late INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS model_scoring_audit (
    score_id VARCHAR PRIMARY KEY,
    account_month_id VARCHAR,
    account_id VARCHAR,
    customer_id VARCHAR,
    month DATE,
    model_version VARCHAR,
    predicted_pd DOUBLE,
    risk_band VARCHAR,
    top_shap_feature_1 VARCHAR,
    top_shap_value_1 DOUBLE,
    top_shap_feature_2 VARCHAR,
    top_shap_value_2 DOUBLE,
    psi_score DOUBLE,
    drift_alert_flag BOOLEAN,
    fairness_alert_flag BOOLEAN,
    override_flag BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_month_id) REFERENCES account_monthly_snapshots(account_month_id),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
