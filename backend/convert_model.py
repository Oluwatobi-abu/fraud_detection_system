"""
One-time conversion script.

Converts an already-trained fraud_model.pkl (joblib/pickle) into XGBoost's
native format (fraud_model.json), which is stable across XGBoost versions
and won't trigger the "loading a serialized model from an older version"
warning again.

Run this ONCE, locally, from inside the backend/ folder:
    python convert_model.py

It reads fraud_model.pkl and writes fraud_model.json next to it.
You can keep or delete fraud_model.pkl afterward — main.py will be updated
to load fraud_model.json instead.
"""

import os
import joblib
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
JSON_PATH = os.path.join(BASE_DIR, "fraud_model.json")

if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(f"Could not find {PKL_PATH}. Run this from inside backend/.")

model = joblib.load(PKL_PATH)

if not isinstance(model, XGBClassifier):
    raise TypeError(
        f"Loaded model is a {type(model).__name__}, not XGBClassifier. "
        "This conversion script is only for XGBoost models. "
        "If your best model was Logistic Regression or Random Forest instead, "
        "you don't need this conversion — joblib/pickle is fine for those."
    )

model.save_model(JSON_PATH)
print(f"Converted successfully: {JSON_PATH}")
print("You can now delete fraud_model.pkl if you like (main.py no longer needs it).")
