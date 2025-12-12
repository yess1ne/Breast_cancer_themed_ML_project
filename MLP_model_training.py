# ----------------------------------------------------
# MLP Training Script for WDBC Option A (10 mean features)
# Uses: wdbc_mean10_encoded.csv
# ----------------------------------------------------

import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# -------------------------
# Configuration
# -------------------------
DATA_FILE = "wdbc_mean10_encoded.csv"

OUTPUT_DIR = "models"
MODEL_FILENAME = os.path.join(OUTPUT_DIR, "mlp_mean10.pkl")
SCALER_FILENAME = os.path.join(OUTPUT_DIR, "scaler_mean10.pkl")
METRICS_FILENAME = os.path.join(OUTPUT_DIR, "mlp_mean10_metrics.json")
FEATURES_FILENAME = os.path.join(OUTPUT_DIR, "mean10_feature_order.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 10 doctor-interpretable mean morphology features
FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]

LABEL_COL = "diagnosis"  # already encoded: 0 = B, 1 = M

# -------------------------
# Metrics helper
# -------------------------
def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "Accuracy": round(acc, 4),
        "AUC (ROC)": round(auc, 4),
        "Sensitivity (TPR)": round(tpr, 4),
        "Specificity (TNR)": round(tnr, 4),
        "False Positive Rate": round(fpr, 4),
        "False Negative Rate": round(fnr, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

# -------------------------
# 1) Load dataset
# -------------------------
print("--- 1. Loading Option A dataset ---")
try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    raise SystemExit(1)

# Validate columns
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print("ERROR: Missing required features:")
    for f in missing:
        print(" -", f)
    raise SystemExit(1)

if LABEL_COL not in df.columns:
    print("ERROR: 'diagnosis' column missing.")
    raise SystemExit(1)

# Check NaNs
if df.isna().any().any():
    print("ERROR: NaN values detected.")
    print(df.isna().sum())
    raise SystemExit(1)

X = df[FEATURES].values.astype(np.float64)
y = df[LABEL_COL].values.astype(int)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# 2) Scaling
# -------------------------
print("--- 2. Scaling features ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(SCALER_FILENAME, "wb") as f:
    pickle.dump(scaler, f)
print(f"SUCCESS: Scaler saved to {SCALER_FILENAME}")

with open(FEATURES_FILENAME, "w") as f:
    json.dump({"feature_order": FEATURES}, f, indent=4)
print(f"SUCCESS: Feature order saved to {FEATURES_FILENAME}")

# -------------------------
# 3) Train MLP
# -------------------------
print("--- 3. Training MLP model ---")

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=42
)

try:
    model.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"ERROR during training: {e}")
    raise SystemExit(1)

print("SUCCESS: Training completed.")

# -------------------------
# 4) Evaluation & saving
# -------------------------
print("--- 4. Evaluation ---")

y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
metrics = calculate_metrics(y_test, y_test_proba)

with open(METRICS_FILENAME, "w") as f:
    json.dump(
        {
            "model": "MLPClassifier (sklearn)",
            "dataset": DATA_FILE,
            "features": FEATURES,
            "metrics": metrics
        },
        f,
        indent=4
    )
print(f"SUCCESS: Metrics saved to {METRICS_FILENAME}")

with open(MODEL_FILENAME, "wb") as f:
    pickle.dump(model, f)
print(f"SUCCESS: Model saved to {MODEL_FILENAME}")

print("\n--- Test Set Performance ---")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\n--- MLP training complete ---")
