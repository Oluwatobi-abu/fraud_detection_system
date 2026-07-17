"""
Credit Card Fraud Detection — Streamlit Frontend
Communicates with the FastAPI backend at http://localhost:8007

Run with:
    streamlit run app.py
"""

import requests
import time
import pandas as pd
import streamlit as st

API_URL = "https://fraud-detection-system-h6ct.onrender.com/predict"
HEALTH_URL = "https://fraud-detection-system-h6ct.onrender.com/health"

V_FIELDS = [f"V{i}" for i in range(1, 29)]
FEATURE_ORDER = ["Time"] + V_FIELDS + ["Amount"]

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.caption("Real-time and batch fraud scoring, backed by a FastAPI model-serving endpoint.")

# --- Backend health check ---
with st.sidebar:
    st.header("Backend Status")
    try:
        health = requests.get(HEALTH_URL, timeout=15).json()
        if health.get("model_loaded") and health.get("scaler_loaded"):
            st.success("Connected — model & scaler loaded")
        else:
            st.warning("Backend reachable but model/scaler not loaded")
    except requests.exceptions.RequestException:
        st.warning(
            "Backend is waking up (free-tier hosting spins down when idle). "
            "This can take 30–60 seconds — try Predict below, or refresh in a moment."
        )

    st.divider()
    st.markdown(
        "**Feature reference**\n\n"
        "- `Time`: seconds since first transaction\n"
        "- `V1`–`V28`: PCA-transformed features\n"
        "- `Amount`: transaction amount"
    )

mode = st.radio("Choose input mode:", ["Manual Entry", "CSV Upload"], horizontal=True)

st.divider()


def call_predict_api(payload: dict, retries: int = 12, backoff_seconds: int = 10) -> dict:
    # A refused connection fails instantly (it does NOT consume the request
    # timeout), so the real wait time here is retries * backoff_seconds =
    # 12 * 10 = 120 seconds, matching Render's free-tier cold-start window.
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < retries:
                time.sleep(backoff_seconds)
    raise last_error


# ------------------------------------------------------------------
# MANUAL ENTRY MODE
# ------------------------------------------------------------------
if mode == "Manual Entry":
    st.subheader("Enter Transaction Details")

    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
        with col2:
            amount_val = st.number_input("Amount", value=0.0, min_value=0.0, step=1.0)

        st.markdown("**PCA Components (V1–V28)**")
        v_values = {}
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v_{i}", format="%.4f")

        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if submitted:
        payload = {"Time": time_val, **v_values, "Amount": amount_val}
        try:
            with st.spinner("Scoring transaction... backend may be cold-starting (Render free tier), this can take up to 2 minutes"):
                result = call_predict_api(payload)

            if result["prediction"] == 1:
                st.error(f"🚨 {result['message']}  —  Fraud probability: {result['probability']:.2%}")
            else:
                st.success(f"✅ {result['message']}  —  Fraud probability: {result['probability']:.2%}")

            st.progress(min(max(result["probability"], 0.0), 1.0))

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            st.error("Backend didn't wake up within 2 minutes. This is unusual — please try Predict again, or check that the Render service is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Backend rejected the request: {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ------------------------------------------------------------------
# CSV UPLOAD MODE
# ------------------------------------------------------------------
else:
    st.subheader("Batch Prediction via CSV Upload")
    st.markdown(
        "Upload a CSV with columns: `Time, V1, V2, ..., V28, Amount` "
        "(the `Class` column, if present, will be ignored)."
    )

    uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        missing_cols = [c for c in FEATURE_ORDER if c not in batch_df.columns]
        if missing_cols:
            st.error(f"CSV is missing required columns: {missing_cols}")
            st.stop()

        st.write(f"Loaded **{len(batch_df)}** transactions.")

        if st.button("Run Batch Prediction", type="primary"):
            predictions, probabilities, errors = [], [], []
            progress_bar = st.progress(0.0)

            for idx, row in batch_df.iterrows():
                payload = {col: float(row[col]) for col in FEATURE_ORDER}
                try:
                    result = call_predict_api(payload)
                    predictions.append(result["prediction"])
                    probabilities.append(result["probability"])
                    errors.append(None)
                except Exception as e:
                    predictions.append(None)
                    probabilities.append(None)
                    errors.append(str(e))
                progress_bar.progress((idx + 1) / len(batch_df))

            batch_df["Prediction"] = predictions
            batch_df["Fraud Probability"] = probabilities
            batch_df["Error"] = errors

            n_fraud = sum(p == 1 for p in predictions)
            n_errors = sum(e is not None for e in errors)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transactions", len(batch_df))
            c2.metric("Flagged as Fraud", n_fraud)
            c3.metric("Failed Requests", n_errors)

            def highlight_fraud(row):
                if row["Prediction"] == 1:
                    return ["background-color: #fdd; "] * len(row)
                return [""] * len(row)

            st.dataframe(batch_df.style.apply(highlight_fraud, axis=1), use_container_width=True)

            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )
