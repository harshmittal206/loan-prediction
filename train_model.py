"""
Train and export the Loan Approval Prediction model.

Run:
    python train_model.py

This script intentionally uses the same preprocessing, split, and model
settings as the supplied notebook so the saved artifacts remain compatible
with app.py.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    encoders = {}
    for col in ("education", "self_employed", "loan_status"):
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(
            random_state=42, n_estimators=200, max_depth=8
        ),
    }

    trained_models = {}
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        pred = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1 Score": float(f1_score(y_test, pred, zero_division=0)),
            "ROC AUC": float(roc_auc_score(y_test, probabilities)),
        })

    results.sort(key=lambda item: item["F1 Score"], reverse=True)
    best_model_name = results[0]["Model"]
    best_model = trained_models[best_model_name]

    # Refit preprocessing/model on all available rows after model selection.
    # This maximizes training data for the deployed model; reported metrics
    # above remain based on the held-out test set.
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X)
    best_model.fit(X_all_scaled, y)

    artifacts = {
        "best_model.pkl": best_model,
        "scaler.pkl": final_scaler,
        "encoder.pkl": encoders,
        "feature_columns.pkl": list(X.columns),
        "best_model_name.pkl": best_model_name,
    }
    for filename, artifact in artifacts.items():
        joblib.dump(artifact, os.path.join(BASE_DIR, filename))

    report = {
        "selected_model": best_model_name,
        "scikit_learn_version": sklearn.__version__,
        "test_size": 0.2,
        "random_state": 42,
        "test_metrics": results,
        "note": "Metrics are from the held-out test set. Final deployment model is refit on all rows."
    }
    with open(os.path.join(BASE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Selected model: {best_model_name}")
    print(pd.DataFrame(results).to_string(index=False))
    print("Saved model and preprocessing artifacts successfully.")

if __name__ == "__main__":
    main()
