"""
LOAN APPROVAL PREDICTOR — Streamlit Web App
---------------------------------------------
A browser-based app for entering new customer details and getting a live
Approved/Rejected prediction from your trained logistic regression model.

HOW TO RUN:
1. Install streamlit once (if you don't already have it):
       pip install streamlit
2. Place this file in the SAME FOLDER as 'Loan_Prediction.csv'
   (your LOGISTIC REGRESSION project folder).
3. From that folder, run:
       streamlit run loan_prediction_app.py
   This opens the app in your browser automatically (usually at
   http://localhost:8501).

NOTE ON THE LOG FILE:
This app writes to 'customer_loan_predictions_log.csv' using the SAME column
schema as loan_prediction_gui.py (the Tkinter version), so if you use
both apps in the same project folder, every prediction from either one
lands in one consistent, combined log.

WANT TO DEPLOY IT ONLINE (free, for your portfolio/CV)?
- Push this file + Loan_Prediction.csv to a GitHub repo.
- Go to https://share.streamlit.io, sign in with GitHub, and point it at
  the repo. It'll give you a public URL in a couple of minutes.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Always work relative to THIS file's own folder, regardless of where the
# app is launched from (terminal, VS Code Run button, double-click, etc.)
# so 'Loan_Prediction.csv' and the log CSV are always found/written correctly.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LOG_FILE = 'customer_loan_predictions_log.csv'

# Must match loan_prediction_gui.py's LOG_COLUMNS exactly, so both apps
# append to a single, consistent log file regardless of which one is used.
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
# LIGHT STYLING — softens the default "generic Streamlit" look
# ----------------------------------------------------------------------
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; max-width: 1100px; }
        div[data-testid="stMetric"] {
            background-color: rgba(120, 120, 120, 0.08);
            border-radius: 10px;
            padding: 12px 16px;
        }
        section[data-testid="stSidebar"] { min-width: 340px; }
        h1 { font-size: 1.9rem; }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 1. REBUILD THE EXACT PREPROCESSING + TRAINING PIPELINE FROM THE NOTEBOOK
#    (cached so it only runs once per session, not on every interaction)
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

    return model, scaler, encoder, categorical_columns, encoded_cols, inputs, weights_df


MODEL, SCALER, ENCODER, CAT_COLS, ENCODED_COLS, INPUTS, WEIGHTS_DF = build_model()

# The exact mean Credit_History value the scaler was fit on. Feeding this
# in (instead of 0 or 1) makes its scaled value ~0, so it contributes
# nothing to the decision — letting the applicant's other features
# determine the outcome instead of defaulting them into "bad credit".
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
    return prediction, proba


def log_prediction(customer_dict, prediction, proba):
    """Append this prediction using the SAME column schema as the Tkinter
    app's log_prediction(), so both apps share one consistent CSV."""
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
# 3. SIDEBAR — the input form lives here, freeing up the main area
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("New Customer")

    with st.form("customer_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

        st.divider()

        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=1800, step=100)
        loan_amount = st.number_input("Loan Amount (thousands)", min_value=0, value=128, step=1)
        loan_term = st.number_input("Loan Term (days)", min_value=0, value=360, step=30)
        credit_history = st.selectbox(
            "Credit History",
            ["1.0 (Good)", "0.0 (Bad)", "No prior credit history"]
        )

        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")


# ----------------------------------------------------------------------
# 4. MAIN AREA — tabs instead of one long scrolling page
# ----------------------------------------------------------------------
st.title("🏦 Loan Approval Predictor")
st.caption("Fill in the customer's details on the left, then click Predict.")

tab_predict, tab_history = st.tabs(["🔮 Prediction", "📋 History & Log"])

# ---- keep the latest result available across reruns within a session ----
if submitted:
    if "1.0" in credit_history:
        credit_history_value = 1.0
    elif "0.0" in credit_history:
        credit_history_value = 0.0
    else:
        credit_history_value = CREDIT_HISTORY_NEUTRAL  # neutral — no prior history

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
    prediction, proba = predict_customer(customer)
    log_prediction(customer, prediction, proba)
    st.session_state['last_customer'] = customer
    st.session_state['last_prediction'] = prediction
    st.session_state['last_proba'] = proba

with tab_predict:
    if 'last_prediction' in st.session_state:
        prediction = st.session_state['last_prediction']
        proba = st.session_state['last_proba']
        customer = st.session_state['last_customer']

        left, right = st.columns([1.1, 1], gap="large")

        with left:
            if prediction == 1:
                st.success("### ✅ Predicted: APPROVED")
            else:
                st.error("### ❌ Predicted: REJECTED")

            m1, m2 = st.columns(2)
            m1.metric("P(Rejected)", f"{proba[0]:.1%}")
            m2.metric("P(Approved)", f"{proba[1]:.1%}")
            st.progress(float(proba[1]), text="Approval confidence")

            if customer['Credit_History'] == 1.0:
                ch_label = 'good'
            elif customer['Credit_History'] == 0.0:
                ch_label = 'bad'
            else:
                ch_label = 'no prior history (neutral)'

            st.caption(
                f"{customer['Gender']}, {customer['Married']} married, "
                f"{customer['Dependents']} dependents, {customer['Education']}, "
                f"income {int(customer['ApplicantIncome'])} "
                f"(+{int(customer['CoapplicantIncome'])} co-applicant), "
                f"loan {int(customer['LoanAmount'])}k over {int(customer['Loan_Amount_Term'])} days, "
                f"credit history {ch_label}, "
                f"{customer['Property_Area']} property."
            )

        with right:
            st.markdown("**What's driving this prediction?**")
            st.dataframe(
                WEIGHTS_DF.head(6).style.format({'Weight': '{:.3f}'}),
                hide_index=True,
                use_container_width=True
            )
            st.caption("Positive weights push toward Approved, negative toward Rejected.")
    else:
        st.info("⬅️ Fill in the customer's details in the sidebar and click **Predict** to see a result here.")

with tab_history:
    if os.path.isfile(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        total = len(log_df)
        approved = (log_df['Prediction'] == 'Approved').sum()
        rejected = (log_df['Prediction'] == 'Rejected').sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total logged", total)
        m2.metric("Approved", approved)
        m3.metric("Rejected", rejected)

        st.dataframe(
            log_df.sort_values('Timestamp', ascending=False),
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download full log as CSV",
            data=log_df.to_csv(index=False),
            file_name="customer_loan_predictions_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions logged yet — submit the form in the sidebar to get started.")
