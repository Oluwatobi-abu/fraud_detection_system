# Technical Report — Credit Card Fraud Detection System

**Author:** Abubakar Oluwatobi
**Dataset:** Credit Card Fraud Detection (mlg-ulb, Kaggle)

> Fill in the bracketed numbers after running `notebook/fraud_training.ipynb` against the real `creditcard.csv`.

## 1. Problem Summary

The dataset contains 284,807 European card transactions over two days, of which
492 (0.17%) are fraudulent — an extreme class imbalance. The goal is to flag
fraudulent transactions in real time while minimizing both missed fraud
(false negatives) and false alarms (false positives) on legitimate customers.

## 2. Handling Class Imbalance

Three approaches were implemented and compared:

- **Random undersampling** — fast, but discards the majority of legitimate-transaction data, increasing variance.
- **SMOTE** — generates synthetic minority-class samples by interpolation; preserves majority-class data but can introduce unrealistic synthetic points.
- **Class-weight adjustment** (`class_weight='balanced'` / `scale_pos_weight`) — no data manipulation, penalizes misclassifying fraud more heavily.

Random Forest was trained on SMOTE-resampled data; Logistic Regression and
XGBoost used class-weighting directly, leaving the test set untouched and
representative of real-world fraud rates.

## 3. Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.0609 | 0.9184 | 0.1141 | 0.9722 | 0.7189 | 90 | 1389 | 55475 | 8 |
| Random Forest | 0.8333 | 0.8163 | 0.8247 | 0.9619 | 0.8742 | 80 | 16 | 56848 | 18 |
| **XGBoost** | **0.8737** | **0.8469** | **0.8601** | **0.9760** | **0.8764** | **83** | **12** | **56852** | **15** |

Logistic Regression caught the most fraud cases by raw count (90 of 98,
91.8% recall) but at a steep cost: 1,389 legitimate transactions incorrectly
flagged, for a precision of just 6.1%. In practice that means roughly 15
false alarms for every real fraud caught — unworkable for a production
fraud team who'd be drowning in false positives. This illustrates exactly
why precision and recall have to be read together, not in isolation.

## 4. Final Model Selection

**Selected model: XGBoost**

On the real `creditcard.csv` test set (56,962 transactions, 98 actual fraud
cases), XGBoost caught 83 of 98 fraud transactions (84.7% recall) while
flagging only 12 legitimate transactions as false alarms, out of 56,864
legitimate transactions correctly left alone. Only 15 fraud cases were
missed. Precision of 87.4% means when the model flags a transaction as
fraud, it's right about 7 times out of 8.

**Justification:** Models were ranked primarily by **PR-AUC**, since
ROC-AUC can look artificially strong under extreme class imbalance and
PR-AUC more directly reflects how well the model finds the rare fraud
class. Ties were broken by **Recall**, since in fraud detection a missed
fraudulent transaction (false negative) is typically far more costly than a
false alarm on a legitimate transaction (false positive) — a false alarm
costs a verification step; a missed fraud costs the actual loss.

XGBoost won on both counts: highest PR-AUC (0.876, versus 0.874 for Random
Forest and 0.719 for Logistic Regression) and the best balance of recall
and precision among the two tree-based models. Random Forest came close on
PR-AUC but caught fewer fraud cases (80 vs. 83) and missed more (18 vs. 15).
Logistic Regression's high recall doesn't offset its unusable precision —
1,389 false positives makes it impractical despite catching the most raw
fraud cases, reinforcing why PR-AUC (which accounts for precision across
all thresholds) was the right primary metric rather than recall alone.

## 5. Evaluation Metrics — Detail

- **Confusion matrix** (TP / FP / TN / FN) for each model — see notebook output.
- **Precision** — of transactions flagged as fraud, what fraction actually were.
- **Recall** — of actual fraud transactions, what fraction were caught.
- **F1 Score** — harmonic mean of precision and recall.
- **ROC-AUC** — reported for completeness, but treated as secondary to PR-AUC given the imbalance.
- **PR-AUC** — primary selection metric.

## 6. Limitations

- The dataset's `V1`–`V28` features are PCA-transformed for privacy, so individual feature meanings (and therefore business-friendly explanations of *why* a transaction was flagged) aren't directly interpretable.
- The dataset spans only two days of transactions from European cardholders; fraud patterns and spending behavior can drift over time and across regions, so periodic retraining would be needed in production.
- SMOTE-generated synthetic samples may not perfectly reflect real fraud patterns not present in the training data.
- The current API scores one transaction at a time; the Streamlit batch mode loops over rows client-side rather than the backend supporting native batch requests.

## 7. Future Improvements

- Add a native batch-prediction endpoint (`POST /predict_batch`) to avoid one HTTP round trip per row.
- Add model monitoring/drift detection for production deployment.
- Try cost-sensitive learning with an explicit false-negative-to-false-positive cost ratio rather than the default class-weighting.
- Add SHAP-based explainability for individual predictions, to give the fraud team a reason code alongside the probability score.
- Containerize backend + frontend with Docker Compose for easier deployment.
