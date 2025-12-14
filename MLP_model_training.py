import os
import json
import pickle
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

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

LABEL_COL = "diagnosis"

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
# Load Dataset
# -------------------------
print("--- 1. Loading Option A dataset ---")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found. Please check the file path.")
    exit()

X = df[FEATURES].values.astype(np.float64)
y = df[LABEL_COL].values.astype(int)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Scaling Features
# -------------------------
print("--- 2. Scaling features ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(SCALER_FILENAME, "wb") as f:
    pickle.dump(scaler, f)

with open(FEATURES_FILENAME, "w") as f:
    json.dump({"feature_order": FEATURES}, f, indent=4)

# -------------------------
# Train MLP Classifier
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

model.fit(X_train_scaled, y_train)
print("SUCCESS: Training completed.")

# -------------------------
# Evaluate Model and Save
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

with open(MODEL_FILENAME, "wb") as f:
    pickle.dump(model, f)

print("\n--- Test Set Performance ---")
for k, v in metrics.items():
    print(f"{k}: {v}")


# -------------------------
# SHAP Explanation (Client Friendly with Labels)
# -------------------------
print("\n--- 5. SHAP Explanations (Client Friendly with Labels) ---")

# Define a wrapper function that returns only the probability of the positive class (class 1)
def predict_proba_class_1(X):
    return model.predict_proba(X)[:, 1]

# Reduce background data size using kmeans for faster processing and stability
background_data = shap.kmeans(X_train_scaled, 10).data

# Create a SHAP explainer using KernelExplainer with the wrapped function
explainer = shap.KernelExplainer(predict_proba_class_1, background_data)

# Calculate SHAP values for a subset of the test data (first 5 samples)
X_test_subset = X_test_scaled[:5]
shap_values_class_1 = explainer.shap_values(X_test_subset)

# Diagnostic checks (checking shapes for consistency)
print(f"Shape of SHAP values for Class 1: {shap_values_class_1.shape}")
print(f"Shape of X_test_subset: {X_test_subset.shape}")
print("SUCCESS: Shapes match for plotting.")

# SHAP Summary Plot for Malignant Class (class index 1) with Client-friendly Labels
# Replace feature names with their respective labels
client_friendly_labels = [
    "Average size (radius)", 
    "Texture variation", 
    "Boundary length (perimeter)", 
    "Area (surface size)", 
    "Smoothness", 
    "Compactness", 
    "Concavity", 
    "Concave points", 
    "Symmetry", 
    "Boundary complexity (fractal dimension)"
]

# Creating a bar plot with custom feature names and clearer x-axis labels
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_class_1, 
    X_test_subset, 
    feature_names=client_friendly_labels,  # Use client-friendly labels
    plot_type="bar",  # Bar plot to highlight feature importance
    color_bar_label="Impact on Malignant Prediction",  # Adding a color bar label
    show=False
)

# Title for the plot
plt.title("Feature Importance for Malignant (Cancerous) Tumors", fontsize=14, weight='bold')

# Clean up x-axis label for better readability
plt.xlabel("Average impact on model output", fontsize=12)

# Save the plot to a file
plt.tight_layout()
plt.savefig('shap_summary_plot_client_friendly.png')

print("\n--- SHAP Explanation Complete ---")