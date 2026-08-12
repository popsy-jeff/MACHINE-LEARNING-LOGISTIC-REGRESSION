"""
LOAN PREDICTION GUI
--------------------
A simple desktop form for entering a new customer's details and getting
a live Approved/Rejected prediction from your trained logistic regression
model — no manual pandas/DataFrame typing needed.

HOW TO USE:
1. Place this file in the SAME FOLDER as your notebook and
   'Loan_Prediction.csv' (i.e. LOGISTIC REGRESSION folder).
2. Run it with:  python loan_prediction_gui.py
3. Fill in the form and click "Predict".

This script rebuilds your exact preprocessing pipeline (cleaning, encoding,
scaling) and trains the same logistic regression model on startup, so the
prediction logic always matches your notebook. If you'd rather not retrain
every time, see the note at the bottom of this file about saving/loading
the model with joblib.
"""

import os
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# CSV file where every prediction made through the GUI gets appended
LOG_FILE = 'customer_predictions_log.csv'
LOG_COLUMNS = [
    'Timestamp', 'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area',
    'Prediction', 'P_Rejected', 'P_Approved'
]


# ----------------------------------------------------------------------
# 1. REBUILD THE EXACT PREPROCESSING + TRAINING PIPELINE FROM THE NOTEBOOK
# ----------------------------------------------------------------------
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

    return model, scaler, encoder, categorical_columns, encoded_cols, inputs


MODEL, SCALER, ENCODER, CAT_COLS, ENCODED_COLS, INPUTS = build_model()


# ----------------------------------------------------------------------
# 2. PREDICTION FUNCTION FOR A SINGLE NEW CUSTOMER
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
    """Append this prediction as a new row to the CSV log, creating the
    file with a header row the first time it's called."""
    file_exists = os.path.isfile(LOG_FILE)

    row = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        'P_Rejected': round(proba[0], 4),
        'P_Approved': round(proba[1], 4),
    }

    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------------------------------------------------
# 3. GUI
# ----------------------------------------------------------------------
class LoanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Loan Approval Predictor")
        self.geometry("420x560")
        self.resizable(False, False)

        pad = {'padx': 10, 'pady': 6}
        row = 0

        ttk.Label(self, text="New Customer Details", font=("Segoe UI", 13, "bold")).grid(
            row=row, column=0, columnspan=2, **pad)
        row += 1

        # ---- Gender ----
        ttk.Label(self, text="Gender").grid(row=row, column=0, sticky='w', **pad)
        self.gender = ttk.Combobox(self, values=["Male", "Female"], state="readonly")
        self.gender.set("Male")
        self.gender.grid(row=row, column=1, **pad)
        row += 1

        # ---- Married ----
        ttk.Label(self, text="Married").grid(row=row, column=0, sticky='w', **pad)
        self.married = ttk.Combobox(self, values=["Yes", "No"], state="readonly")
        self.married.set("Yes")
        self.married.grid(row=row, column=1, **pad)
        row += 1

        # ---- Dependents ----
        ttk.Label(self, text="Dependents").grid(row=row, column=0, sticky='w', **pad)
        self.dependents = ttk.Combobox(self, values=["0", "1", "2", "3+"], state="readonly")
        self.dependents.set("0")
        self.dependents.grid(row=row, column=1, **pad)
        row += 1

        # ---- Education ----
        ttk.Label(self, text="Education").grid(row=row, column=0, sticky='w', **pad)
        self.education = ttk.Combobox(self, values=["Graduate", "Not Graduate"], state="readonly")
        self.education.set("Graduate")
        self.education.grid(row=row, column=1, **pad)
        row += 1

        # ---- Self Employed ----
        ttk.Label(self, text="Self Employed").grid(row=row, column=0, sticky='w', **pad)
        self.self_employed = ttk.Combobox(self, values=["Yes", "No"], state="readonly")
        self.self_employed.set("No")
        self.self_employed.grid(row=row, column=1, **pad)
        row += 1

        # ---- Applicant Income ----
        ttk.Label(self, text="Applicant Income").grid(row=row, column=0, sticky='w', **pad)
        self.applicant_income = ttk.Entry(self)
        self.applicant_income.insert(0, "5000")
        self.applicant_income.grid(row=row, column=1, **pad)
        row += 1

        # ---- Coapplicant Income ----
        ttk.Label(self, text="Coapplicant Income").grid(row=row, column=0, sticky='w', **pad)
        self.coapplicant_income = ttk.Entry(self)
        self.coapplicant_income.insert(0, "1800")
        self.coapplicant_income.grid(row=row, column=1, **pad)
        row += 1

        # ---- Loan Amount ----
        ttk.Label(self, text="Loan Amount (thousands)").grid(row=row, column=0, sticky='w', **pad)
        self.loan_amount = ttk.Entry(self)
        self.loan_amount.insert(0, "128")
        self.loan_amount.grid(row=row, column=1, **pad)
        row += 1

        # ---- Loan Amount Term ----
        ttk.Label(self, text="Loan Term (days)").grid(row=row, column=0, sticky='w', **pad)
        self.loan_term = ttk.Entry(self)
        self.loan_term.insert(0, "360")
        self.loan_term.grid(row=row, column=1, **pad)
        row += 1

        # ---- Credit History ----
        ttk.Label(self, text="Credit History").grid(row=row, column=0, sticky='w', **pad)
        self.credit_history = ttk.Combobox(self, values=["1.0 (Good)", "0.0 (Bad)"], state="readonly")
        self.credit_history.set("1.0 (Good)")
        self.credit_history.grid(row=row, column=1, **pad)
        row += 1

        # ---- Property Area ----
        ttk.Label(self, text="Property Area").grid(row=row, column=0, sticky='w', **pad)
        self.property_area = ttk.Combobox(self, values=["Urban", "Semiurban", "Rural"], state="readonly")
        self.property_area.set("Semiurban")
        self.property_area.grid(row=row, column=1, **pad)
        row += 1

        # ---- Predict button ----
        ttk.Button(self, text="Predict", command=self.on_predict).grid(
            row=row, column=0, columnspan=2, pady=16)
        row += 1

        # ---- Result display ----
        self.result_label = ttk.Label(self, text="", font=("Segoe UI", 12, "bold"))
        self.result_label.grid(row=row, column=0, columnspan=2, **pad)
        row += 1

        self.proba_label = ttk.Label(self, text="", font=("Segoe UI", 10))
        self.proba_label.grid(row=row, column=0, columnspan=2, **pad)
        row += 1

        # ---- Log summary ----
        self.log_count_label = ttk.Label(self, text="", font=("Segoe UI", 9), foreground="gray")
        self.log_count_label.grid(row=row, column=0, columnspan=2, **pad)
        row += 1

        ttk.Button(self, text="View Log Summary", command=self.show_log_summary).grid(
            row=row, column=0, columnspan=2, pady=(0, 10))

        self.update_log_count()

    def on_predict(self):
        try:
            customer = {
                'Gender': self.gender.get(),
                'Married': self.married.get(),
                'Dependents': self.dependents.get().replace('+', ''),  # "3+" -> "3"
                'Education': self.education.get(),
                'Self_Employed': self.self_employed.get(),
                'ApplicantIncome': float(self.applicant_income.get()),
                'CoapplicantIncome': float(self.coapplicant_income.get()),
                'LoanAmount': float(self.loan_amount.get()),
                'Loan_Amount_Term': float(self.loan_term.get()),
                'Credit_History': 1.0 if "1.0" in self.credit_history.get() else 0.0,
                'Property_Area': self.property_area.get(),
            }

            prediction, proba = predict_customer(customer)

            if prediction == 1:
                self.result_label.config(text="Predicted: APPROVED", foreground="green")
            else:
                self.result_label.config(text="Predicted: REJECTED", foreground="red")

            self.proba_label.config(
                text=f"P(Rejected) = {proba[0]:.3f}   |   P(Approved) = {proba[1]:.3f}"
            )

            log_prediction(customer, prediction, proba)
            self.update_log_count()

        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numbers for income, loan amount, and term.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_log_count(self):
        if os.path.isfile(LOG_FILE):
            with open(LOG_FILE, newline='') as f:
                count = sum(1 for _ in csv.DictReader(f))
            self.log_count_label.config(
                text=f"{count} customer(s) logged in '{LOG_FILE}'"
            )
        else:
            self.log_count_label.config(text="No predictions logged yet.")

    def show_log_summary(self):
        if not os.path.isfile(LOG_FILE):
            messagebox.showinfo("Log Summary", "No predictions have been logged yet.")
            return

        log_df = pd.read_csv(LOG_FILE)
        total = len(log_df)
        approved = (log_df['Prediction'] == 'Approved').sum()
        rejected = (log_df['Prediction'] == 'Rejected').sum()

        summary = (
            f"Total customers logged: {total}\n\n"
            f"Approved: {approved} ({approved / total * 100:.1f}%)\n"
            f"Rejected: {rejected} ({rejected / total * 100:.1f}%)\n\n"
            f"Log file: {os.path.abspath(LOG_FILE)}"
        )
        messagebox.showinfo("Log Summary", summary)


if __name__ == "__main__":
    app = LoanApp()
    app.mainloop()


# ----------------------------------------------------------------------
# OPTIONAL: avoid retraining every time the GUI opens
# ----------------------------------------------------------------------
# Retraining on startup only takes a second or two with this dataset size,
# so it's fine as-is. But if you want to save time later, you can persist
# the fitted model/scaler/encoder once from your notebook:
#
#   import joblib
#   joblib.dump(logistics_model, 'model.joblib')
#   joblib.dump(scaler, 'scaler.joblib')
#   joblib.dump(encoder, 'encoder.joblib')
#
# ...then replace the build_model() call above with joblib.load(...) calls
# instead of re-running the full pipeline each time.
