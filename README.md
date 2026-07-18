# Credit Card Fraud Detection System

An end-to-end fraud detection pipeline: exploratory data analysis and model
training in a Jupyter notebook, a FastAPI backend serving real-time
predictions, and a Streamlit frontend for manual and batch (CSV) scoring.

Dataset: [Credit Card Fraud Detection (mlg-ulb) — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 European card transactions, 492 of which (0.17%) are fraudulent.

## 🔗 Live Demo

Try it here: **[frauddetectionsystem-h5kiuh3cq6lpcdhywjk2an.streamlit.app](https://frauddetectionsystem-h5kiuh3cq6lpcdhywjk2an.streamlit.app)**

Backend deployed on Render, frontend on Streamlit Community Cloud.

> **Note:** the backend runs on Render's free tier, which spins down after inactivity. The first prediction after idle time may take 30–60 seconds while it wakes up — this is expected, not a bug.

## Project Structure

```
fraud_detection_system/
├── notebook/
│   └── fraud_training.ipynb     # EDA, imbalance handling, model training & evaluation
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── convert_model.py         # one-time helper: pickle -> XGBoost native format
│   ├── fraud_model.json         # trained XGBoost model (native format, committed)
│   └── scaler.pkl               # fitted StandardScaler (committed)
├── frontend/
│   └── app.py                   # Streamlit UI (manual entry + CSV upload)
├── reports/
│   └── technical_report.md      # model comparison & justification (real results)
├── requirements.txt
└── README.md
```

## 1. Train the Model (Kaggle)

1. Open [the dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and create a new notebook against it (or upload `notebook/fraud_training.ipynb` to Kaggle directly).
2. Run every cell top to bottom. It will:
   - Load and explore `creditcard.csv`
   - Compare undersampling, SMOTE, and class-weighting for the imbalance problem
   - Train Logistic Regression, Random Forest, and XGBoost (plus an Isolation Forest baseline)
   - Evaluate with Precision, Recall, F1, ROC-AUC, and PR-AUC
   - Save the best model — `fraud_model.json` if XGBoost wins (native, version-stable format), or `fraud_model.pkl` otherwise — plus `scaler.pkl`
3. Download the model file and `scaler.pkl` from Kaggle's output panel and place them inside the local `backend/` folder.

On the real dataset, **XGBoost** was selected (PR-AUC 0.876, Recall 84.7%, Precision 87.4% — see `reports/technical_report.md` for the full comparison against Logistic Regression and Random Forest).

> If you'd rather run the notebook locally, drop `creditcard.csv` into the `notebook/` folder — the notebook falls back to a local path automatically if it can't find the Kaggle input path.

> Already have an older `fraud_model.pkl` from a previous XGBoost run? Run `python backend/convert_model.py` to convert it to the native `fraud_model.json` format without retraining.

## 2. Run the Backend

```bash
cd backend
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8007
```

Verify it's up:
- Swagger UI: http://localhost:8007/docs
- Health check: http://localhost:8007/health

**Endpoint:** `POST /predict`

Request:
```json
{
  "Time": 10000,
  "V1": 0.1, "V2": -0.2, "...": "...", "V28": -0.2,
  "Amount": 150.75
}
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.945,
  "message": "Fraudulent transaction detected"
}
```

## 3. Run the Frontend

In a **second terminal** (keep the backend running):

```bash
cd frontend
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). You'll get:

- **Manual Entry** — fill in `Time`, `V1`–`V28`, `Amount`, get an instant prediction with fraud probability.
- **CSV Upload** — upload a batch of transactions, get predictions for every row, fraud rows highlighted, with a summary count and a downloadable results CSV.

> **Note:** `frontend/app.py` points `API_URL`/`HEALTH_URL` at the deployed Render backend by default (so the live demo works out of the box). If you're running the backend locally too, change those two lines to `http://localhost:8007/...` first.

## Notes

- `fraud_model.json` and `scaler.pkl` **are committed** to this repo — deployment platforms (Render) need them directly available at build time, and they're small enough that this isn't a problem. `convert_model.py` exists for converting an already-trained XGBoost pickle into the native format without retraining.
- The backend and frontend were smoke-tested against a synthetic stand-in dataset (matching the real schema) before wiring up the real Kaggle data, and again against the real trained model before deployment.
- The frontend automatically retries the backend on connection failures for up to ~2 minutes, since Render's free tier spins down after inactivity and can take 30-90+ seconds to wake back up.
- See `reports/technical_report.md` for the full model comparison (Logistic Regression vs. Random Forest vs. XGBoost), final selection justification, limitations, and future improvements.

## Requirements

See `requirements.txt`. Core stack: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, joblib, fastapi, uvicorn, streamlit, requests, matplotlib, seaborn.
