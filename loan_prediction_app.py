"""
LOAN APPROVAL PREDICTOR — Streamlit Web App
---------------------------------------------
A browser-based app for entering new customer details and getting a live
Approved/Rejected prediction from your trained logistic regression model,
plus a loan structuring calculator (interest, repayment schedule) adapted
from the loan business rules in your FEDHA-SYSTEM project.

HOW TO RUN:
1. Install streamlit and plotly once (if you don't already have them):
       pip install streamlit plotly
2. Place this file in the SAME FOLDER as 'Loan_Prediction.csv'
   (your LOGISTIC REGRESSION project folder).
3. From that folder, run:
       streamlit run loan_prediction_app.py
   Or double-click run_app.bat.

LAYOUT:
The sidebar is navigation ONLY (page switcher). All forms — applicant
details, loan calculator inputs — live in the main content area for
each page.

NOTE ON THE LOG FILE:
Writes to 'customer_loan_predictions_log.csv' using the same column
schema as loan_prediction_gui.py, so both apps share one combined log
if you ever use both in the same folder.

ICONS:
Uses Google's Material Symbols webfont (the same icon set used by
Material UI) loaded from Google Fonts' CDN — no emoji anywhere in the UI.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Always work relative to THIS file's own folder, regardless of where the
# app is launched from (terminal, VS Code Run button, double-click, etc.)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LOG_FILE = 'customer_loan_predictions_log.csv'

# Must match loan_prediction_gui.py's LOG_COLUMNS exactly.
LOG_COLUMNS = [
    'Timestamp', 'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area',
    'Prediction', 'P_Rejected', 'P_Approved'
]

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)


# ----------------------------------------------------------------------
# ICONS — Material Symbols (Google's official icon set, used by Material UI)
# ----------------------------------------------------------------------
def icon(name: str, size: int = 20, color: str = "currentColor", valign: str = "middle") -> str:
    return (
        f'<span class="material-symbols-outlined" '
        f'style="font-size:{size}px; color:{color}; vertical-align:{valign}; '
        f'line-height:1;">{name}</span>'
    )


# ----------------------------------------------------------------------
# GLOBAL STYLE
# ----------------------------------------------------------------------
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/icon?family=Material+Symbols+Outlined"
          rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet">
    <style>
        :root {
            --brand-primary:  #2FBF8F;   /* brighter teal — pops on dark bg */
            --brand-primary-light: rgba(47, 191, 143, 0.14);
            --brand-accent:   #E0B75C;
            --brand-approve:  #3FD08A;
            --brand-reject:   #F2685C;
            --brand-bg-card:  rgba(47, 191, 143, 0.07);
            --brand-border:   rgba(47, 191, 143, 0.22);
            --brand-text:     #EAEFEC;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--brand-text); }
        .block-container { padding-top: 1.6rem; max-width: 1200px; }

        h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; }
        h1 { font-size: 1.85rem; color: var(--brand-primary); }

        div[data-testid="stMetric"] {
            background: var(--brand-bg-card);
            border: 1px solid var(--brand-border);
            border-radius: 12px;
            padding: 14px 18px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(47, 191, 143, 0.18);
        }

        button[kind="primary"], button[kind="formSubmit"] {
            background-color: var(--brand-primary) !important;
            color: #0E1512 !important;
            border: none !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        button[kind="primary"]:hover, button[kind="formSubmit"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(47, 191, 143, 0.35);
        }

        /* narrow sidebar since it's navigation-only now */
        section[data-testid="stSidebar"] { min-width: 240px; max-width: 270px; }
        section[data-testid="stSidebar"] > div { background: #0B100E; }
        section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }

        section[data-testid="stSidebar"] .stRadio > label { display: none; }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            display: flex; flex-direction: column; gap: 6px;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
            position: relative;
            padding: 11px 14px 11px 18px;
            border-radius: 10px;
            margin-bottom: 0;
            background: transparent;
            border: 1px solid transparent;
            cursor: pointer;
            overflow: hidden;
            transition: background 0.18s ease, border-color 0.18s ease,
                        transform 0.18s ease, box-shadow 0.18s ease;
        }

        /* left accent bar, hidden by default, slides in on hover/active */
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label::before {
            content: "";
            position: absolute; left: 0; top: 50%;
            width: 3px; height: 0%;
            background: var(--brand-primary);
            border-radius: 0 3px 3px 0;
            transform: translateY(-50%);
            transition: height 0.2s ease;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
            background: var(--brand-primary-light);
            border-color: var(--brand-border);
            transform: translateX(3px);
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover::before {
            height: 55%;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
            background: var(--brand-primary-light);
            border-color: var(--brand-border);
            box-shadow: 0 2px 10px rgba(47, 191, 143, 0.15);
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked)::before {
            height: 70%;
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) p {
            color: var(--brand-primary);
            font-weight: 600;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
            font-size: 1.05rem;
            font-weight: 500;
            transition: color 0.18s ease;
            margin: 0;
        }

        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .result-block { animation: fadeSlideIn 0.45s ease-out; }

        .status-pill {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 8px 16px; border-radius: 999px;
            font-weight: 600; font-size: 1.05rem;
            animation: fadeSlideIn 0.45s ease-out;
        }
        .status-approved { background: rgba(63, 208, 138, 0.16); color: var(--brand-approve); }
        .status-rejected { background: rgba(242, 104, 92, 0.16); color: var(--brand-reject); }

        .info-card {
            border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
            background: var(--brand-bg-card); border: 1px solid var(--brand-border);
            color: var(--brand-text);
        }
        .section-label {
            font-weight: 600; color: var(--brand-primary);
            display: flex; align-items: center; gap: 6px; margin-bottom: 4px;
        }
        .nav-brand {
            display: flex; align-items: center; gap: 8px;
            font-weight: 700; color: var(--brand-primary);
            font-size: 1.15rem; padding: 10px 6px 16px 6px;
            margin-bottom: 8px;
            border-bottom: 1px solid var(--brand-border);
        }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 1. REBUILD THE EXACT PREPROCESSING + TRAINING PIPELINE FROM THE NOTEBOOK
# ----------------------------------------------------------------------
@st.cache_resource
def build_model():
    loan_df = pd.read_csv('Loan_Prediction.csv')

    loan_df['LoanAmount'] = loan_df['LoanAmount'].fillna(loan_df['LoanAmount'].median())
    loan_df['Loan_Amount_Term'] = loan_df['Loan_Amount_Term'].fillna(loan_df['Loan_Amount_Term'].median())
    loan_df['Credit_History'] = loan_df['Credit_History'].fillna(loan_df['Credit_History'].median())

    loan_df['Dependents'] = loan_df['Dependents'].replace('3+', '3')
    loan_df['Dependents'] = pd.to_numeric(loan_df['Dependents'], errors='coerce')
    loan_df['Dependents'] = loan_df['Dependents'].fillna(loan_df['Dependents'].median())

    loan_df = loan_df.dropna(subset=['Gender', 'Self_Employed', 'Married'])

    categorical_columns = loan_df.select_dtypes(include='object').columns.tolist()
    categorical_columns.remove('Loan_ID')
    categorical_columns.remove('Loan_Status')

    loan_df['Loan_Status'] = loan_df['Loan_Status'].map({'Y': 1, 'N': 0})

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder.fit(loan_df[categorical_columns])
    encoded_cols = encoder.get_feature_names_out(categorical_columns)
    loan_df[encoded_cols] = encoder.transform(loan_df[categorical_columns])

    loan_model_df = loan_df.drop(columns=categorical_columns)

    train_df, test_df = train_test_split(loan_model_df, test_size=0.2, random_state=42)

    inputs = loan_model_df.drop(columns=['Loan_ID', 'Loan_Status']).columns.tolist()
    target = 'Loan_Status'

    train_inputs = train_df[inputs].copy()
    train_target = train_df[target].copy()

    scaler = StandardScaler()
    train_inputs_scaled = scaler.fit_transform(train_inputs)

    model = LogisticRegression(solver='liblinear', class_weight='balanced')
    model.fit(train_inputs_scaled, train_target)

    weights_df = pd.DataFrame({
        'Feature': inputs,
        'Weight': model.coef_[0]
    }).sort_values('Weight', key=abs, ascending=False)

    # ---- test-set performance, for the Model Performance page ----
    test_inputs = test_df[inputs].copy()
    test_target = test_df[target].copy()
    test_inputs_scaled = scaler.transform(test_inputs)
    test_preds = model.predict(test_inputs_scaled)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )

    accuracy = accuracy_score(test_target, test_preds)
    precision = precision_score(test_target, test_preds, zero_division=0)
    recall = recall_score(test_target, test_preds, zero_division=0)
    f1 = f1_score(test_target, test_preds, zero_division=0)
    cm = confusion_matrix(test_target, test_preds)

    majority_class = test_target.mode()[0]
    majority_baseline_acc = (test_target == majority_class).mean()
    rng = np.random.RandomState(42)
    random_preds = rng.randint(0, 2, size=len(test_target))
    random_baseline_acc = accuracy_score(test_target, random_preds)

    performance = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'test_size': len(test_target),
        'majority_baseline_acc': majority_baseline_acc,
        'random_baseline_acc': random_baseline_acc,
    }

    return model, scaler, encoder, categorical_columns, encoded_cols, inputs, weights_df, performance


MODEL, SCALER, ENCODER, CAT_COLS, ENCODED_COLS, INPUTS, WEIGHTS_DF, PERFORMANCE = build_model()

CREDIT_HISTORY_NEUTRAL = float(SCALER.mean_[INPUTS.index('Credit_History')])


# ----------------------------------------------------------------------
# 2. PREDICTION + LOGGING
# ----------------------------------------------------------------------
def predict_customer(customer_dict):
    df = pd.DataFrame({k: [v] for k, v in customer_dict.items()})
    df[ENCODED_COLS] = ENCODER.transform(df[CAT_COLS])
    processed = df.drop(columns=CAT_COLS)[INPUTS]
    scaled = SCALER.transform(processed)

    prediction = MODEL.predict(scaled)[0]
    proba = MODEL.predict_proba(scaled)[0]
    return prediction, proba, scaled


def log_prediction(customer_dict, prediction, proba):
    file_exists = os.path.isfile(LOG_FILE)
    row = {
        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Gender': customer_dict['Gender'],
        'Married': customer_dict['Married'],
        'Dependents': customer_dict['Dependents'],
        'Education': customer_dict['Education'],
        'Self_Employed': customer_dict['Self_Employed'],
        'ApplicantIncome': customer_dict['ApplicantIncome'],
        'CoapplicantIncome': customer_dict['CoapplicantIncome'],
        'LoanAmount': customer_dict['LoanAmount'],
        'Loan_Amount_Term': customer_dict['Loan_Amount_Term'],
        'Credit_History': customer_dict['Credit_History'],
        'Property_Area': customer_dict['Property_Area'],
        'Prediction': 'Approved' if prediction == 1 else 'Rejected',
        'P_Rejected': round(float(proba[0]), 4),
        'P_Approved': round(float(proba[1]), 4),
    }
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    row_df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)


# ----------------------------------------------------------------------
# 3. LOAN STRUCTURING RULES — adapted from FEDHA-SYSTEM's LoanService.java
# ----------------------------------------------------------------------
LOAN_TYPES = {
    "Emergency Loan": {
        "monthly_rate_pct": 0.3,
        "repayment_months": 12,
        "share_multiplier": 1.0,
        "guarantors_required": 0,
        "description": "Fast, small-value loan for urgent needs. Equal to member's share value.",
    },
    "Short Loan": {
        "monthly_rate_pct": 0.6,
        "repayment_months": 24,
        "share_multiplier": 2.0,
        "guarantors_required": 2,
        "description": "Medium-term loan, up to 2x member's share value.",
    },
    "Normal Loan": {
        "monthly_rate_pct": 1.0,
        "repayment_months": 36,
        "share_multiplier": 3.0,
        "guarantors_required": 2,
        "description": "Standard loan product, up to 3x member's share value.",
    },
    "Development Loan": {
        "monthly_rate_pct": 1.4,
        "repayment_months": 48,
        "share_multiplier": 5.0,
        "guarantors_required": 2,
        "description": "Larger, longer-term loan for development purposes, up to 5x share value.",
    },
}


def calculate_loan_terms(principal: float, loan_type: str):
    terms = LOAN_TYPES[loan_type]
    monthly_rate = terms["monthly_rate_pct"] / 100.0
    months = terms["repayment_months"]

    total_interest = principal * monthly_rate * months
    total_repayable = principal + total_interest
    monthly_payment = total_repayable / months

    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_repayable": round(total_repayable, 2),
        "total_interest": round(total_interest, 2),
        "months": months,
        "monthly_rate_pct": terms["monthly_rate_pct"],
        "guarantors_required": terms["guarantors_required"],
    }


def build_amortization_schedule(principal: float, loan_type: str):
    """
    Month-by-month breakdown for a flat/simple-interest loan.
    Interest is charged once on the full original principal and spread
    evenly across the term, so each month's interest and principal
    portions are constant — only the running balance changes.
    """
    terms = LOAN_TYPES[loan_type]
    months = terms["repayment_months"]
    schedule = calculate_loan_terms(principal, loan_type)

    monthly_principal = principal / months
    monthly_interest = schedule["total_interest"] / months
    monthly_payment = schedule["monthly_payment"]

    rows = []
    balance = principal
    for month in range(1, months + 1):
        balance = max(balance - monthly_principal, 0.0)
        rows.append({
            "Month": month,
            "Payment (KES)": round(monthly_payment, 2),
            "Principal Portion (KES)": round(monthly_principal, 2),
            "Interest Portion (KES)": round(monthly_interest, 2),
            "Remaining Balance (KES)": round(balance, 2),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3b. RISK TIERS — bucket the approval probability into a readable label
# ----------------------------------------------------------------------
def risk_tier(p_approved: float):
    """
    Maps P(Approved) to a risk tier. Framed from the applicant's risk
    of rejection, so LOW risk = high approval probability.
    """
    if p_approved >= 0.70:
        return "Low Risk", "var(--brand-approve)", "shield"
    elif p_approved >= 0.30:
        return "Medium Risk", "var(--brand-accent)", "warning"
    else:
        return "High Risk", "var(--brand-reject)", "report"


# ----------------------------------------------------------------------
# 4. SIDEBAR — NAVIGATION ONLY
# ----------------------------------------------------------------------
PAGES = {
    "Prediction": "insights",
    "Loan Calculator": "calculate",
    "Batch Prediction": "upload_file",
    "Model Performance": "monitoring",
    "Model Insights": "explore",
    "History & Log": "history",
    "About": "info",
}

with st.sidebar:
    st.markdown(
        f"<div class='nav-brand'>{icon('account_balance', 24)} Loan Predictor</div>",
        unsafe_allow_html=True
    )

    # Build labels with inline icons via markdown-rendered radio is not
    # directly supported, so we render icon + text as the option label
    # using unicode-safe plain text (icons shown just above the group).
    page = st.radio(
        "Navigate",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )


# ----------------------------------------------------------------------
# 5. HEADER (shown on every page)
# ----------------------------------------------------------------------
st.markdown(
    f"<h1>{icon('account_balance', 30, 'var(--brand-primary)')} Loan Approval Predictor</h1>",
    unsafe_allow_html=True
)


# ----------------------------------------------------------------------
# 6. PAGE: PREDICTION — form now lives in the main content area
# ----------------------------------------------------------------------
if page == "Prediction":
    st.caption("Enter the applicant's details below, then click Predict.")

    with st.form("customer_form"):
        st.markdown(
            f"<div class='section-label'>{icon('person', 18)} Personal</div>",
            unsafe_allow_html=True
        )
        p1, p2, p3 = st.columns(3)
        with p1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with p2:
            married = st.selectbox("Married", ["Yes", "No"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        with p3:
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

        st.markdown(
            f"<div class='section-label' style='margin-top:14px;'>{icon('payments', 18)} Financial</div>",
            unsafe_allow_html=True
        )
        f1, f2, f3 = st.columns(3)
        with f1:
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
            loan_term = st.number_input("Loan Term (days)", min_value=0, value=360, step=30)
        with f2:
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=1800, step=100)
            credit_history = st.selectbox(
                "Credit History",
                ["1.0 (Good)", "0.0 (Bad)", "No prior credit history"]
            )
        with f3:
            loan_amount = st.number_input("Loan Amount (thousands)", min_value=0, value=128, step=1)

        st.write("")
        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if submitted:
        if "1.0" in credit_history:
            credit_history_value = 1.0
        elif "0.0" in credit_history:
            credit_history_value = 0.0
        else:
            credit_history_value = CREDIT_HISTORY_NEUTRAL

        customer = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents.replace('+', ''),
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': float(applicant_income),
            'CoapplicantIncome': float(coapplicant_income),
            'LoanAmount': float(loan_amount),
            'Loan_Amount_Term': float(loan_term),
            'Credit_History': credit_history_value,
            'Property_Area': property_area,
        }
        prediction, proba, scaled_row = predict_customer(customer)
        log_prediction(customer, prediction, proba)
        st.session_state['last_customer'] = customer
        st.session_state['last_prediction'] = prediction
        st.session_state['last_proba'] = proba
        st.session_state['last_scaled_row'] = scaled_row

    st.divider()

    if 'last_prediction' in st.session_state:
        prediction = st.session_state['last_prediction']
        proba = st.session_state['last_proba']
        customer = st.session_state['last_customer']
        scaled_row = st.session_state['last_scaled_row']

        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown('<div class="result-block">', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(
                    f'<span class="status-pill status-approved">'
                    f'{icon("check_circle", 22)} Predicted: Approved</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<span class="status-pill status-rejected">'
                    f'{icon("cancel", 22)} Predicted: Rejected</span>',
                    unsafe_allow_html=True
                )

            tier_label, tier_color, tier_icon = risk_tier(proba[1])
            st.markdown(
                f'&nbsp;<span class="status-pill" style="background:rgba(0,0,0,0.18); '
                f'color:{tier_color}; border:1px solid {tier_color};">'
                f'{icon(tier_icon, 18, tier_color)} {tier_label}</span>',
                unsafe_allow_html=True
            )

            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("P(Rejected)", f"{proba[0]:.1%}")
            m2.metric("P(Approved)", f"{proba[1]:.1%}")
            st.progress(float(proba[1]), text="Approval confidence")
            st.caption(
                "Risk tier: **Low** ≥ 70% approval confidence · "
                "**Medium** 30–70% · **High** < 30%."
            )

            if customer['Credit_History'] == 1.0:
                ch_label = 'good'
            elif customer['Credit_History'] == 0.0:
                ch_label = 'bad'
            else:
                ch_label = 'no prior history (neutral)'

            st.markdown(
                f"<div class='info-card' style='margin-top:10px;'>"
                f"{icon('summarize', 16)} &nbsp;"
                f"{customer['Gender']}, {customer['Married']} married, "
                f"{customer['Dependents']} dependents, {customer['Education']}, "
                f"income {int(customer['ApplicantIncome'])} "
                f"(+{int(customer['CoapplicantIncome'])} co-applicant), "
                f"loan {int(customer['LoanAmount'])}k over {int(customer['Loan_Amount_Term'])} days, "
                f"credit history {ch_label}, {customer['Property_Area']} property."
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(
                    f"<div class='info-card'>{icon('arrow_forward', 16, valign='text-bottom')} "
                    f"Head to <b>Loan Calculator</b> in the sidebar to structure repayment "
                    f"terms for this applicant.</div>",
                    unsafe_allow_html=True
                )

        with right:
            st.markdown(
                f"<div class='section-label'>{icon('insights', 18)} "
                f"What's driving THIS applicant's result?</div>",
                unsafe_allow_html=True
            )

            # Per-applicant contribution = weight_i * scaled_value_i for
            # this specific applicant, not just the model's global weights.
            # This can differ meaningfully from the global ranking, e.g. a
            # feature with a big global weight may contribute little here
            # if this applicant's scaled value is close to zero (near the
            # training-set mean), while a smaller-weight feature can matter
            # a lot if this applicant is a strong outlier on it.
            contributions = scaled_row[0] * MODEL.coef_[0]
            contrib_df = pd.DataFrame({
                'Feature': INPUTS,
                'Contribution': contributions,
            }).sort_values('Contribution', key=abs, ascending=False).head(6)
            contrib_df['Direction'] = contrib_df['Contribution'].apply(
                lambda v: '→ Approved' if v > 0 else '→ Rejected'
            )

            st.dataframe(
                contrib_df.style.format({'Contribution': '{:.3f}'}),
                hide_index=True,
                use_container_width=True
            )
            st.caption(
                "Contribution = this applicant's scaled value × the model's weight for "
                "that feature. Unlike the global weights, this reflects what actually "
                "moved THIS prediction, not just what tends to matter across all applicants."
            )

            with st.expander("Show global model weights instead"):
                st.dataframe(
                    WEIGHTS_DF.head(6).style.format({'Weight': '{:.3f}'}),
                    hide_index=True,
                    use_container_width=True
                )
                st.caption("Positive weights push toward Approved, negative toward Rejected, on average across all applicants.")

            st.markdown(
                f"<div class='info-card'>{icon('lightbulb', 16)} "
                f"<b>Why credit history matters most:</b> its model weight is roughly "
                f"4x larger than any other feature, so it dominates the decision "
                f"unless treated as neutral for applicants with no prior history."
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f"<div class='info-card'>{icon('arrow_upward', 16, valign='text-bottom')} "
            f"Fill in the applicant's details above and click "
            f"<b>Predict</b> to see a result here.</div>",
            unsafe_allow_html=True
        )


# ----------------------------------------------------------------------
# 7. PAGE: LOAN CALCULATOR — form lives in the main content area
# ----------------------------------------------------------------------
elif page == "Loan Calculator":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('calculate', 20)} Loan Structuring Calculator</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Rule-based calculator for structuring loan terms, adapted from SACCO-style "
        "loan rules (loan type, member shares, simple interest). This runs "
        "independently of the ML prediction — use it once an applicant is "
        "approved to work out what they'd actually repay."
    )

    calc_col1, calc_col2 = st.columns([1, 1.2], gap="large")

    with calc_col1:
        loan_type = st.selectbox("Loan Type", list(LOAN_TYPES.keys()))
        details = LOAN_TYPES[loan_type]

        st.markdown(
            f"<div class='info-card'>"
            f"{icon('info', 16)} {details['description']}<br><br>"
            f"{icon('percent', 14)} Interest: <b>{details['monthly_rate_pct']}% / month "
            f"({details['monthly_rate_pct']*12:.1f}% p.a. effective)</b><br>"
            f"{icon('event', 14)} Repayment period: <b>{details['repayment_months']} months</b><br>"
            f"{icon('groups', 14)} Guarantors required: <b>{details['guarantors_required']}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

        member_shares = st.number_input(
            "Member Shares (KES)", min_value=0, value=50000, step=1000,
            help="Determines the maximum loan amount for the selected loan type."
        )
        max_loan = member_shares * details["share_multiplier"]
        st.caption(
            f"Maximum eligible amount for {loan_type}: "
            f"**KES {max_loan:,.0f}** ({details['share_multiplier']}x shares)"
        )

        requested_amount = st.number_input(
            "Requested Loan Amount (KES)", min_value=0,
            value=min(100000, int(max_loan)) if max_loan > 0 else 0, step=1000
        )

        over_limit = requested_amount > max_loan

        if over_limit:
            st.markdown(
                f"<div class='info-card' style='border-color:var(--brand-reject);'>"
                f"{icon('warning', 18, 'var(--brand-reject)')} "
                f"Exceeds the maximum allowed for {loan_type} "
                f"(KES {max_loan:,.0f} based on shares of KES {member_shares:,.0f})."
                f"</div>",
                unsafe_allow_html=True
            )

    # ---- results update live as inputs change, no button needed ----
    with calc_col2:
        if requested_amount > 0 and not over_limit:
            schedule = calculate_loan_terms(requested_amount, loan_type)

            st.markdown(
                f"<div class='result-block'>"
                f"<span class='status-pill status-approved'>"
                f"{icon('check_circle', 20)} Within eligible limit</span></div>",
                unsafe_allow_html=True
            )
            st.write("")

            r1, r2 = st.columns(2)
            r1.metric("Monthly Repayment", f"KES {schedule['monthly_payment']:,.2f}")
            r2.metric("Total Repayable", f"KES {schedule['total_repayable']:,.2f}")

            r3, r4 = st.columns(2)
            r3.metric("Total Interest", f"KES {schedule['total_interest']:,.2f}")
            r4.metric("Repayment Period", f"{schedule['months']} months")

            cost_pct = (schedule['total_interest'] / requested_amount) * 100
            st.markdown(
                f"<div class='info-card' style='margin-top:8px;'>"
                f"{icon('functions', 16)} <b>Formula:</b> Total Interest = Principal × Monthly "
                f"Rate × Months &nbsp;→&nbsp; Total Repayable = Principal + Interest &nbsp;→&nbsp; "
                f"Monthly Payment = Total Repayable ÷ Months<br><br>"
                f"{icon('trending_up', 16)} <b>Cost of credit:</b> interest adds "
                f"<b>{cost_pct:.1f}%</b> on top of the principal over the full term."
                f"</div>",
                unsafe_allow_html=True
            )

            # ---- principal vs interest breakdown chart ----
            breakdown_df = pd.DataFrame({
                "Component": ["Principal", "Interest"],
                "Amount (KES)": [requested_amount, schedule['total_interest']]
            }).set_index("Component")
            st.caption("Principal vs. interest, over the full repayment term")
            st.bar_chart(breakdown_df, color="#2FBF8F", height=180)

        elif over_limit:
            st.markdown(
                f"<div class='info-card'>{icon('info', 16)} "
                f"Reduce the requested amount below KES {max_loan:,.0f}, or increase member "
                f"shares, to see a repayment breakdown here.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='info-card'>{icon('arrow_back', 16, valign='text-bottom')} "
                f"Enter a requested loan amount to see the repayment breakdown here.</div>",
                unsafe_allow_html=True
            )

    # ------------------------------------------------------------------
    # Amortization schedule — month-by-month breakdown for the currently
    # selected loan type and requested amount.
    # ------------------------------------------------------------------
    if requested_amount > 0 and not over_limit:
        st.divider()
        st.markdown(
            f"<div class='section-label' style='font-size:1.05rem;'>"
            f"{icon('event_repeat', 20)} Amortization Schedule</div>",
            unsafe_allow_html=True
        )
        st.caption(
            f"Month-by-month breakdown for {loan_type} at KES {requested_amount:,.0f}. "
            "Since interest is charged as a flat rate on the original principal, the "
            "payment amount and its principal/interest split stay constant each month — "
            "only the remaining balance declines."
        )

        schedule_df = build_amortization_schedule(requested_amount, loan_type)

        show_full = st.checkbox("Show all months (default: first 12 + last 3)", value=False)
        if show_full or len(schedule_df) <= 15:
            display_df = schedule_df
        else:
            display_df = pd.concat([schedule_df.head(12), schedule_df.tail(3)])

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        st.caption("Remaining balance over the repayment term")
        st.line_chart(
            schedule_df.set_index("Month")[["Remaining Balance (KES)"]],
            color="#2FBF8F",
            height=200
        )

        st.download_button(
            "Download amortization schedule as CSV",
            data=schedule_df.to_csv(index=False),
            file_name=f"amortization_{loan_type.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )

    st.divider()

    # ------------------------------------------------------------------
    # Compare all loan types side by side, at the currently requested
    # amount — fills the page with genuinely useful information instead
    # of leaving it blank, and helps applicants pick the right product.
    # ------------------------------------------------------------------
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('table_view', 20)} Compare All Loan Types at KES {requested_amount:,.0f}</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "How the same requested amount would be structured under each loan product. "
        "'—' means the amount exceeds that product's limit based on your shares."
    )

    compare_rows = []
    for lt_name, lt_details in LOAN_TYPES.items():
        lt_max = member_shares * lt_details["share_multiplier"]
        if requested_amount > 0 and requested_amount <= lt_max:
            sched = calculate_loan_terms(requested_amount, lt_name)
            compare_rows.append({
                "Loan Type": lt_name,
                "Rate / month": f"{lt_details['monthly_rate_pct']}%",
                "Term (months)": lt_details["repayment_months"],
                "Max eligible (KES)": f"{lt_max:,.0f}",
                "Monthly payment (KES)": f"{sched['monthly_payment']:,.2f}",
                "Total repayable (KES)": f"{sched['total_repayable']:,.2f}",
                "Guarantors": lt_details["guarantors_required"],
            })
        else:
            compare_rows.append({
                "Loan Type": lt_name,
                "Rate / month": f"{lt_details['monthly_rate_pct']}%",
                "Term (months)": lt_details["repayment_months"],
                "Max eligible (KES)": f"{lt_max:,.0f}",
                "Monthly payment (KES)": "—",
                "Total repayable (KES)": "—",
                "Guarantors": lt_details["guarantors_required"],
            })

    compare_df = pd.DataFrame(compare_rows)
    st.dataframe(compare_df, hide_index=True, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # FAQ / definitions — more information, low visual weight
    # ------------------------------------------------------------------
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('quiz', 20)} Loan Terms Explained</div>",
        unsafe_allow_html=True
    )

    with st.expander(f"What is 'simple interest' and how is it different from reducing balance?"):
        st.markdown(
            "Simple interest is calculated once on the **full original principal** for the "
            "entire loan term, so the interest amount is fixed from day one regardless of how "
            "much has already been repaid. This is what the calculator above uses.\n\n"
            "A *reducing balance* loan, by contrast, recalculates interest each period on the "
            "**remaining balance**, so interest gets smaller as the loan is paid down. "
            "Simple interest is easier to explain to members but usually results in paying "
            "more total interest than an equivalent reducing-balance loan."
        )

    with st.expander(f"Why do larger loan types require more guarantors?"):
        st.markdown(
            "Guarantors reduce the lender's risk on larger, longer-term loans by having other "
            "members vouch for repayment. The Emergency Loan needs none since it's capped at "
            "the member's own share value (self-secured); every other loan type here requires "
            "**2 guarantors**, each of whom can only guarantee up to a fixed limit "
            "(KES 100,000 in the FEDHA-SYSTEM rules this is based on) across all loans they "
            "co-sign."
        )

    with st.expander(f"How is the maximum loan amount determined?"):
        st.markdown(
            "Each loan type allows borrowing a multiple of the member's contributed shares:\n\n"
            "- Emergency Loan: 1× shares\n"
            "- Short Loan: 2× shares\n"
            "- Normal Loan: 3× shares\n"
            "- Development Loan: 5× shares\n\n"
            "This ties borrowing capacity to what a member has already saved, which is a "
            "standard risk-control mechanism in SACCO-style lending."
        )


# ----------------------------------------------------------------------
# 8. PAGE: BATCH PREDICTION
# ----------------------------------------------------------------------
elif page == "Batch Prediction":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('upload_file', 20)} Batch Prediction</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Upload a CSV of multiple applicants to get predictions for all of them at once. "
        "The file needs the same columns as the single-applicant form."
    )

    required_cols = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area',
    ]

    with st.expander("Expected CSV columns"):
        st.code(", ".join(required_cols))
        st.caption(
            "Credit_History should be 1.0, 0.0, or blank/NaN for 'no prior history' "
            "(treated the same as the neutral option in the single-applicant form)."
        )

    uploaded_file = st.file_uploader("Upload applicants CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.markdown(
                f"<div class='info-card' style='border-color:var(--brand-reject);'>"
                f"{icon('error', 18, 'var(--brand-reject)')} Couldn't read that file: {e}</div>",
                unsafe_allow_html=True
            )
            batch_df = None

        if batch_df is not None:
            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.markdown(
                    f"<div class='info-card' style='border-color:var(--brand-reject);'>"
                    f"{icon('warning', 18, 'var(--brand-reject)')} Missing required column(s): "
                    f"<b>{', '.join(missing)}</b></div>",
                    unsafe_allow_html=True
                )
            else:
                st.write(f"Loaded **{len(batch_df)}** applicants.")
                st.dataframe(batch_df.head(), hide_index=True, use_container_width=True)

                if st.button("Run batch prediction", type="primary"):
                    results = []
                    errors = 0
                    for _, row in batch_df.iterrows():
                        try:
                            credit_history_value = row['Credit_History']
                            if pd.isna(credit_history_value):
                                credit_history_value = CREDIT_HISTORY_NEUTRAL

                            customer = {
                                'Gender': row['Gender'],
                                'Married': row['Married'],
                                'Dependents': str(row['Dependents']).replace('+', ''),
                                'Education': row['Education'],
                                'Self_Employed': row['Self_Employed'],
                                'ApplicantIncome': float(row['ApplicantIncome']),
                                'CoapplicantIncome': float(row['CoapplicantIncome']),
                                'LoanAmount': float(row['LoanAmount']),
                                'Loan_Amount_Term': float(row['Loan_Amount_Term']),
                                'Credit_History': float(credit_history_value),
                                'Property_Area': row['Property_Area'],
                            }
                            prediction, proba, _ = predict_customer(customer)
                            results.append({
                                **row.to_dict(),
                                'Prediction': 'Approved' if prediction == 1 else 'Rejected',
                                'P_Rejected': round(float(proba[0]), 4),
                                'P_Approved': round(float(proba[1]), 4),
                            })
                        except Exception:
                            errors += 1
                            results.append({
                                **row.to_dict(),
                                'Prediction': 'Error',
                                'P_Rejected': None,
                                'P_Approved': None,
                            })

                    results_df = pd.DataFrame(results)
                    st.session_state['batch_results'] = results_df
                    if errors:
                        st.markdown(
                            f"<div class='info-card' style='border-color:var(--brand-reject);'>"
                            f"{icon('warning', 18, 'var(--brand-reject)')} {errors} row(s) "
                            f"couldn't be scored (bad or missing values) and are marked 'Error'.</div>",
                            unsafe_allow_html=True
                        )

    if 'batch_results' in st.session_state:
        st.divider()
        results_df = st.session_state['batch_results']
        scored = results_df[results_df['Prediction'] != 'Error']

        m1, m2, m3 = st.columns(3)
        m1.metric("Scored", len(scored))
        m2.metric("Approved", (scored['Prediction'] == 'Approved').sum())
        m3.metric("Rejected", (scored['Prediction'] == 'Rejected').sum())

        st.dataframe(results_df, hide_index=True, use_container_width=True)

        st.download_button(
            "Download results as CSV",
            data=results_df.to_csv(index=False),
            file_name="batch_prediction_results.csv",
            mime="text/csv",
            type="primary",
        )


# ----------------------------------------------------------------------
# 9. PAGE: MODEL PERFORMANCE
# ----------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('monitoring', 20)} Model Performance</div>",
        unsafe_allow_html=True
    )
    st.caption(
        f"Evaluated on a held-out test split ({PERFORMANCE['test_size']} applicants) "
        f"that the model never saw during training."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{PERFORMANCE['accuracy']:.1%}")
    m2.metric("Precision", f"{PERFORMANCE['precision']:.1%}")
    m3.metric("Recall", f"{PERFORMANCE['recall']:.1%}")
    m4.metric("F1 Score", f"{PERFORMANCE['f1']:.1%}")

    st.caption(
        "Precision/recall/F1 are computed for the **Approved (Y)** class. "
        "Precision: of applicants predicted Approved, how many actually were. "
        "Recall: of applicants who were actually Approved, how many the model caught."
    )

    st.write("")
    perf_col1, perf_col2 = st.columns([1, 1.2], gap="large")

    with perf_col1:
        st.markdown(
            f"<div class='section-label'>{icon('grid_view', 18)} Confusion Matrix</div>",
            unsafe_allow_html=True
        )
        cm = PERFORMANCE['confusion_matrix']
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Rejected", "Actual: Approved"],
            columns=["Predicted: Rejected", "Predicted: Approved"]
        )
        st.dataframe(cm_df, use_container_width=True)

    with perf_col2:
        st.markdown(
            f"<div class='section-label'>{icon('compare_arrows', 18)} Baseline Comparisons</div>",
            unsafe_allow_html=True
        )
        baseline_df = pd.DataFrame({
            "Approach": ["Random guess", "Always predict majority class", "This model"],
            "Accuracy": [
                f"{PERFORMANCE['random_baseline_acc']:.1%}",
                f"{PERFORMANCE['majority_baseline_acc']:.1%}",
                f"{PERFORMANCE['accuracy']:.1%}",
            ],
        })
        st.dataframe(baseline_df, hide_index=True, use_container_width=True)
        st.caption(
            "The model should clear both baselines by a meaningful margin — beating the "
            "majority-class baseline especially matters on imbalanced datasets like this "
            "one, where most historical loans were Approved."
        )

    st.divider()

    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('insights', 20)} Global Feature Weights</div>",
        unsafe_allow_html=True
    )
    st.caption("Full ranking of every feature's weight in the trained logistic regression model.")
    st.dataframe(
        WEIGHTS_DF.style.format({'Weight': '{:.3f}'}),
        hide_index=True,
        use_container_width=True
    )


# ----------------------------------------------------------------------
# 10. PAGE: MODEL INSIGHTS — interactive sigmoid curve explorer
# ----------------------------------------------------------------------
elif page == "Model Insights":
    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('insights', 20)} Sigmoid Curve Explorer</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "See how the model's fitted coefficients shape prediction probability. "
        "Sliders default to the actual trained model — drag them to explore "
        "what-if scenarios."
    )

    default_feature_idx = INPUTS.index('Credit_History') if 'Credit_History' in INPUTS else 0
    feature_choice = st.selectbox("Feature to explore", INPUTS, index=default_feature_idx)
    feature_idx = INPUTS.index(feature_choice)

    default_beta0 = float(MODEL.intercept_[0])
    default_beta1 = float(MODEL.coef_[0][feature_idx])

    # Reset button sets a session_state override BEFORE the sliders are
    # instantiated, which is the correct way to change a widget's value
    # programmatically in Streamlit (mutating st.session_state after the
    # widget exists raises an exception).
    reset_key_b0 = f"beta0_{feature_idx}"
    reset_key_b1 = f"beta1_{feature_idx}"

    top_l, top_r = st.columns([3, 1])
    with top_r:
        st.write("")
        if st.button("Reset to fitted values", use_container_width=True):
            st.session_state[reset_key_b0] = default_beta0
            st.session_state[reset_key_b1] = default_beta1
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        beta0 = st.slider(
            "Intercept (β₀)", -10.0, 10.0,
            st.session_state.get(reset_key_b0, default_beta0), 0.1,
            key=reset_key_b0
        )
    with col2:
        beta1 = st.slider(
            f"Coefficient (β₁) — {feature_choice}", -5.0, 5.0,
            st.session_state.get(reset_key_b1, default_beta1), 0.1,
            key=reset_key_b1
        )

    x = np.linspace(-4, 4, 300)  # standardized feature range (~±4 SD)
    z = beta0 + beta1 * x
    y = 1 / (1 + np.exp(-z))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name='Sigmoid',
        line=dict(width=3, color='#2FBF8F')
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="Decision threshold (0.5)")
    fig.update_layout(
        title=f"Predicted Approval Probability vs. {feature_choice} (standardized)",
        xaxis_title=f"{feature_choice} (standardized value)",
        yaxis_title="P(Approved)",
        yaxis_range=[0, 1],
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#EAEFEC'
    )
    st.plotly_chart(fig, use_container_width=True)

    is_default = (abs(beta0 - default_beta0) < 1e-6) and (abs(beta1 - default_beta1) < 1e-6)
    st.markdown(
        f"<div class='info-card'>"
        f"{icon('functions', 16)} Current: β₀ = <b>{beta0:.3f}</b>, β₁ = <b>{beta1:.3f}</b> "
        f"{'(actual fitted model values)' if is_default else '(what-if, not the fitted model)'}<br><br>"
        f"Steeper curve (higher |β₁|) → the model is more decisive as {feature_choice} changes. "
        f"Shifting β₀ moves where the 50% approval probability falls."
        f"</div>",
        unsafe_allow_html=True
    )

    with st.expander("Model's actual fitted values for this feature"):
        st.write(f"Fitted β₀ (intercept): `{default_beta0:.4f}`")
        st.write(f"Fitted β₁ for {feature_choice}: `{default_beta1:.4f}`")

    st.divider()

    st.markdown(
        f"<div class='section-label' style='font-size:1.05rem;'>"
        f"{icon('bar_chart', 20)} Credit History: Approval Probability by State</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Since Credit_History is really a discrete good/bad/unknown flag rather than a "
        "continuous value, this view is more directly interpretable than the curve above: "
        "it shows the model's actual predicted probability for each state, holding every "
        "other feature at its average (mean-scaled) value."
    )

    if 'Credit_History' in INPUTS:
        ch_idx = INPUTS.index('Credit_History')
        ch_mean = float(SCALER.mean_[ch_idx])
        ch_scale = float(SCALER.scale_[ch_idx])

        states = {
            "Bad (0.0)": 0.0,
            "No history (neutral)": ch_mean,
            "Good (1.0)": 1.0,
        }

        bar_rows = []
        for label, raw_value in states.items():
            z_all = 0.0
            for i, feat in enumerate(INPUTS):
                if i == ch_idx:
                    scaled_val = (raw_value - ch_mean) / ch_scale if ch_scale != 0 else 0.0
                else:
                    scaled_val = 0.0  # mean-centered => 0 in standardized space
                z_all += MODEL.coef_[0][i] * scaled_val
            z_all += float(MODEL.intercept_[0])
            p_approved = 1 / (1 + np.exp(-z_all))
            bar_rows.append({"State": label, "P(Approved)": p_approved})

        bar_df = pd.DataFrame(bar_rows).set_index("State")
        st.bar_chart(bar_df, color="#2FBF8F", height=280)
        st.caption(
            "All other features held at their training-set average, so this isolates the "
            "effect of Credit_History alone."
        )
    else:
        st.info("Credit_History is not present in this model's feature set.")
