# Customer Churn Prediction - local website
# Run:  python app.py   then open  http://localhost:5000

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "customer_churn_elastic_net.joblib")
DATA_PATH = os.path.join(BASE, "data", "Telco-Customer-Churn.csv")

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------- load model (retrain automatically if the saved one can't be loaded) ----------
try:
    model = joblib.load(MODEL_PATH)
    model.predict_proba(pd.read_csv(DATA_PATH).drop(columns=["customerID", "Churn"]).head(1)
                        .assign(TotalCharges=lambda d: pd.to_numeric(d["TotalCharges"], errors="coerce")))
except Exception as e:
    print("Saved model could not be used (" + str(e) + "). Re-training from the CSV...")
    from train_model import train
    model = train()

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]
coefs = classifier.coef_[0]
feature_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]

# Average customer (in model space) - used to explain "why this customer scored high/low"
_df = pd.read_csv(DATA_PATH).drop(columns=["customerID", "Churn"])
_df["TotalCharges"] = pd.to_numeric(_df["TotalCharges"], errors="coerce")
_bg = preprocessor.transform(_df)
_bg = _bg.toarray() if hasattr(_bg, "toarray") else np.asarray(_bg)
background_mean = _bg.mean(axis=0)

LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Partner",
    "Dependents": "Dependents", "tenure": "Tenure", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}
FIELDS = list(LABELS.keys())


def original_feature(col_name):
    """'Contract_Month-to-month' -> 'Contract'"""
    for f in sorted(FIELDS, key=len, reverse=True):
        if col_name == f or col_name.startswith(f + "_"):
            return f
    return col_name


col_to_field = [original_feature(n) for n in feature_names]

METRICS = {  # test-set results from the notebook
    "accuracy": 0.7395, "precision": 0.5060, "recall": 0.7834,
    "f1": 0.6149, "roc_auc": 0.8406,
}


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/model-info")
def model_info():
    order = np.argsort(-np.abs(coefs))[:10]
    top = [{"name": feature_names[i].replace("_", ": ", 1), "coef": round(float(coefs[i]), 3)} for i in order]
    return jsonify({
        "metrics": METRICS,
        "C": float(classifier.C),
        "l1_ratio": float(classifier.l1_ratio),
        "kept": int((coefs != 0).sum()),
        "total": int(len(coefs)),
        "top_coefficients": top,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    missing = [f for f in FIELDS if f not in data]
    if missing:
        return jsonify({"error": "Missing fields: " + ", ".join(missing)}), 400

    row = dict(data)
    try:
        for f in ("SeniorCitizen", "tenure"):
            row[f] = int(row[f])
        for f in ("MonthlyCharges", "TotalCharges"):
            row[f] = float(row[f])
    except (TypeError, ValueError):
        return jsonify({"error": "Numeric fields must be numbers."}), 400

    customer = pd.DataFrame([row])[FIELDS]
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(model.predict(customer)[0])

    # per-feature contribution to the log-odds, relative to the average customer
    x = preprocessor.transform(customer)
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    contrib = coefs * (x - background_mean)
    per_field = {}
    for value, field in zip(contrib, col_to_field):
        per_field[field] = per_field.get(field, 0.0) + float(value)

    drivers = sorted(per_field.items(), key=lambda kv: -abs(kv[1]))[:6]
    drivers = [{"feature": LABELS[k], "value": str(row[k]), "impact": round(v, 3)} for k, v in drivers]

    return jsonify({"probability": probability, "prediction": prediction, "drivers": drivers})


if __name__ == "__main__":
    print("\n  Open http://localhost:5000 in your browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
