"""
Loan Approval Prediction System
--------------------------------
A Streamlit web application that predicts whether a loan application
will be Approved or Rejected, using a pre-trained Machine Learning model.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD CUSTOM CSS
# ============================================================
def load_css(file_path):
    """Load custom CSS file to style the Streamlit app."""
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(os.path.join(BASE_DIR, "assets", "style.css"))

# ------------------------------------------------------------
# Sidebar contrast fix
# ------------------------------------------------------------
# The stylesheet gives the sidebar a light background, but Streamlit's
# dark theme keeps sidebar text/headings light-colored too, which makes
# them invisible (light text on a light background). Force dark, legible
# colors for every element inside the sidebar so it looks correct
# regardless of which Streamlit theme (light/dark) the viewer has active.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(0, 0, 0, 0.15) !important;
    }
    section[data-testid="stSidebar"] .stAlert p {
        color: #0f5132 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL & PREPROCESSING ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    """Load the trained model, scaler, encoders, and feature column order."""
    model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    best_model_name = joblib.load(os.path.join(BASE_DIR, "best_model_name.pkl"))

    # Fail early with a clear message if artifacts from different training runs
    # have accidentally been mixed together.
    if len(feature_columns) != scaler.n_features_in_:
        raise ValueError("Model artifacts do not match: feature count and scaler differ.")
    if getattr(model, "n_features_in_", len(feature_columns)) != len(feature_columns):
        raise ValueError("Model artifacts do not match: model feature count differs.")
    return model, scaler, encoders, feature_columns, best_model_name

try:
    model, scaler, encoders, feature_columns, best_model_name = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False


def predict_loan_status(input_dict):
    """
    Takes a dictionary of raw user inputs, encodes and scales them,
    and returns the prediction along with class probabilities.
    """
    # Build a single-row DataFrame in the exact column order used during training
    row = pd.DataFrame([input_dict]).reindex(columns=feature_columns)

    # Apply the exact encoders saved during model training.
    for col in ("education", "self_employed"):
        value = str(row.at[0, col]).strip()
        known_values = set(encoders[col].classes_)
        if value not in known_values:
            raise ValueError(f"Unsupported value for {col}: {value}")
        row[col] = encoders[col].transform([value])

    # Ensure the feature matrix is numeric and in the training column order.
    row = row.astype(float)
    row_scaled = scaler.transform(row)

    pred_encoded = int(model.predict(row_scaled)[0])
    pred_label = encoders["loan_status"].inverse_transform([pred_encoded])[0]

    # Map probabilities by encoded class ID, not assumed class ordering.
    raw_probabilities = model.predict_proba(row_scaled)[0]
    proba_dict = {}
    for encoded_class, probability in zip(model.classes_, raw_probabilities):
        label = encoders["loan_status"].inverse_transform([int(encoded_class)])[0]
        proba_dict[label] = float(probability)

    return pred_label, proba_dict


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🏦 Loan Approval Predictor")
    st.markdown("---")

    st.markdown("### ℹ️ About This Project")
    st.write(
        "This app uses a Machine Learning model trained on historical loan "
        "application data to predict whether a new loan application is "
        "likely to be **Approved ✅** or **Rejected ❌**."
    )

    st.markdown("### 🧠 Model Information")
    if artifacts_loaded:
        st.success(f"Model in use: **{best_model_name}**")
    st.write(
        "Three models were trained and compared during development: "
        "Logistic Regression, Decision Tree, and Random Forest. "
        "The best-performing model (by F1 Score & ROC AUC) was selected."
    )

    st.markdown("### 📊 Dataset Information")
    st.write(
        "- **4,269** historical loan applications\n"
        "- Features include income, loan amount, loan term, CIBIL score, "
        "and asset values\n"
        "- No missing values or duplicates"
    )

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & scikit-learn")


# ============================================================
# MAIN PAGE HEADER
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🏦 Loan Approval Prediction System</h1>
        <p>Fill in the applicant details below to check the likelihood of loan approval.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if not artifacts_loaded:
    st.error(
        "⚠️ Model files not found! Please make sure `best_model.pkl`, `scaler.pkl`, "
        "`encoder.pkl`, and `feature_columns.pkl` are in the same folder as this app "
        "(generate them by running `notebook.ipynb` first)."
    )
    st.stop()

st.markdown("---")

# ============================================================
# INPUT FORM
# ============================================================
st.markdown("## 📝 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Personal Details")
    no_of_dependents = st.slider(
        "Number of Dependents", min_value=0, max_value=5, value=0, step=1
    )
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
    cibil_score = st.slider(
        "CIBIL Score", min_value=300, max_value=900, value=650, step=1,
        help="Credit score: higher is better. 750+ is generally considered strong."
    )

with col2:
    st.markdown("#### 💰 Financial Details")
    income_annum = st.number_input(
        "Annual Income (₹)", min_value=0, max_value=20_000_000, value=5_000_000, step=100_000
    )
    loan_amount = st.number_input(
        "Loan Amount (₹)", min_value=0, max_value=50_000_000, value=10_000_000, step=100_000
    )
    loan_term = st.slider(
        "Loan Term (years)", min_value=2, max_value=20, value=10, step=1
    )

st.markdown("#### 🏠 Asset Details")
col3, col4 = st.columns(2)

with col3:
    residential_assets_value = st.number_input(
        "Residential Assets Value (₹)", min_value=0, max_value=50_000_000, value=5_000_000, step=100_000
    )
    commercial_assets_value = st.number_input(
        "Commercial Assets Value (₹)", min_value=0, max_value=50_000_000, value=3_000_000, step=100_000
    )

with col4:
    luxury_assets_value = st.number_input(
        "Luxury Assets Value (₹)", min_value=0, max_value=50_000_000, value=8_000_000, step=100_000
    )
    bank_asset_value = st.number_input(
        "Bank Assets Value (₹)", min_value=0, max_value=50_000_000, value=4_000_000, step=100_000
    )

st.markdown("---")

# ============================================================
# PREDICT / RESET BUTTONS
# ============================================================
col_predict, col_reset = st.columns([1, 1])

with col_predict:
    predict_clicked = st.button("🔍 Predict Loan Status", use_container_width=True, type="primary")

with col_reset:
    reset_clicked = st.button("🔄 Reset Form", use_container_width=True)

if reset_clicked:
    st.rerun()

# ============================================================
# PREDICTION RESULT
# ============================================================
if predict_clicked:
    input_dict = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }

    try:
        pred_label, proba_dict = predict_loan_status(input_dict)
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.stop()

    approved_prob = proba_dict.get("Approved", 0.0)
    rejected_prob = proba_dict.get("Rejected", 0.0)
    confidence = max(approved_prob, rejected_prob) * 100

    st.markdown("## 📢 Prediction Result")

    if pred_label == "Approved":
        st.markdown(
            f"""
            <div class="result-card result-approved">
                <h2>✅ Loan Approved!</h2>
                <p>The applicant is likely to be approved for this loan.</p>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-rejected">
                <h2>❌ Loan Rejected</h2>
                <p>The applicant is likely to be rejected for this loan.</p>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📊 Prediction Probability Breakdown")
    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.write("**✅ Approval Probability**")
        st.progress(float(approved_prob))
        st.write(f"{approved_prob * 100:.2f}%")

    with prob_col2:
        st.write("**❌ Rejection Probability**")
        st.progress(float(rejected_prob))
        st.write(f"{rejected_prob * 100:.2f}%")

    with st.expander("🔎 View Submitted Application Details"):
        st.json(input_dict)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p>Loan Approval Prediction System | Built with Streamlit & scikit-learn</p>
        <p>⚠️ This tool is for educational/demo purposes only and should not be used for real financial decisions.</p>
    </div>
    """,
    unsafe_allow_html=True
)
