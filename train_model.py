from pathlib import Path
import argparse
import json
import re
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import EmailFeatureExtractor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emails.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TEXT_CANDIDATES = ["email_text", "email text", "text", "body", "email", "message", "content"]
LABEL_CANDIDATES = ["label", "email type", "class", "email_type", "category", "target"]

def normalize_label(value):
    s = str(value).strip().lower()
    if s in {"1", "phishing", "phishing email", "spam", "malicious", "bad"}:
        return 1
    if s in {"0", "safe", "legitimate", "legitimate email", "ham", "benign", "good"}:
        return 0
    raise ValueError(f"Unknown label value: {value!r}. Use 1/0 or Phishing/Safe.")

def load_dataset(path):
    df = pd.read_csv(path)
    text_col = next((c for c in TEXT_CANDIDATES if c in df.columns.str.lower().tolist()), None)
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns.str.lower().tolist()), None)
    if text_col is None or label_col is None:
        raise ValueError(
            "CSV needs an email text column (email_text/text/body/email/message/content) "
            "and a label column (label/class/email_type/category/target). "
            f"Found: {list(df.columns)}"
        )
    # Preserve original column spelling after case-insensitive detection.
    text_col = next(c for c in df.columns if c.lower() == text_col)
    label_col = next(c for c in df.columns if c.lower() == label_col)
    clean = pd.DataFrame({
        "email_text": df[text_col].fillna("").astype(str),
        "label": df[label_col].map(normalize_label)
    })
    clean = clean[clean.email_text.str.strip().ne("")].drop_duplicates("email_text")
    if clean.label.nunique() != 2:
        raise ValueError("Dataset must contain both Safe (0) and Phishing (1) examples.")
    if clean.label.value_counts().min() < 2:
        raise ValueError("At least two examples per class are required.")
    return clean

def main():
    parser = argparse.ArgumentParser(description="Train a phishing email classifier.")
    parser.add_argument("--data", default=str(DATA_PATH), help="CSV path")
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Dataset not found: {data_path}\nSee README.md to add data/emails.csv.")

    df = load_dataset(data_path)
    X, y = df.email_text, df.label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("features", EmailFeatureExtractor()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "phishing_model.joblib")

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_names": ["Safe", "Phishing"],
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["Safe", "Phishing"], zero_division=0
        )
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nMODEL EVALUATION")
    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1-score : {metrics['f1_score']:.3f}")
    print("\n", metrics["classification_report"])

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Safe", "Phishing"], cmap="Blues", values_format="d"
    )
    disp.ax_.set_title("Phishing Email Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    print(f"Saved model: {MODEL_DIR / 'phishing_model.joblib'}")
    print(f"Saved report: {REPORT_DIR}")

if __name__ == "__main__":
    main()
