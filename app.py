import os
import json
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# -------------------------
# Load assets
# -------------------------
MODEL_PATH = os.path.join("models", "mlp_mean10.pkl")
SCALER_PATH = os.path.join("models", "scaler_mean10.pkl")
FEATURE_ORDER_PATH = os.path.join("models", "mean10_feature_order.json")

for p in [MODEL_PATH, SCALER_PATH, FEATURE_ORDER_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURE_ORDER_PATH, "r") as f:
    FEATURE_ORDER = json.load(f).get("feature_order")

if not FEATURE_ORDER or not isinstance(FEATURE_ORDER, list):
    raise ValueError("Invalid mean10_feature_order.json. Expected { 'feature_order': [..] }")

# -------------------------
# UI helpers (placeholders + short descriptions)
# (These are just example placeholders. They don’t affect the model.)
# -------------------------
PLACEHOLDERS = {
    "radius_mean": "e.g., 14.13",
    "texture_mean": "e.g., 19.29",
    "perimeter_mean": "e.g., 91.97",
    "area_mean": "e.g., 654.89",
    "smoothness_mean": "e.g., 0.096",
    "compactness_mean": "e.g., 0.104",
    "concavity_mean": "e.g., 0.089",
    "concave points_mean": "e.g., 0.048",
    "symmetry_mean": "e.g., 0.181",
    "fractal_dimension_mean": "e.g., 0.062",
}

DESCRIPTIONS = {
    "radius_mean": "Average radius (overall size).",
    "texture_mean": "Intensity variation (texture).",
    "perimeter_mean": "Boundary length.",
    "area_mean": "Area size.",
    "smoothness_mean": "Local boundary smoothness.",
    "compactness_mean": "Compactness of shape.",
    "concavity_mean": "Inward curve severity.",
    "concave points_mean": "Count of concave points.",
    "symmetry_mean": "Symmetry score.",
    "fractal_dimension_mean": "Boundary complexity.",
}

# -------------------------
# Routes
# -------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/mb-form")
def mb_form():
    return render_template(
        "MB_form.html",
        features=FEATURE_ORDER,
        placeholders=PLACEHOLDERS,
        descriptions=DESCRIPTIONS,
        error=None
    )


@app.post("/mb-predict")
def mb_predict():
    inputs = {}
    values = []

    # Validate inputs
    try:
        for feat in FEATURE_ORDER:
            raw = request.form.get(feat, "").strip()
            if raw == "":
                return render_template(
                    "MB_form.html",
                    features=FEATURE_ORDER,
                    placeholders=PLACEHOLDERS,
                    descriptions=DESCRIPTIONS,
                    error=f"Missing value for: {feat}"
                )

            val = float(raw)
            if not np.isfinite(val):
                return render_template(
                    "MB_form.html",
                    features=FEATURE_ORDER,
                    placeholders=PLACEHOLDERS,
                    descriptions=DESCRIPTIONS,
                    error=f"Invalid numeric value for: {feat}"
                )

            inputs[feat] = val
            values.append(val)

    except ValueError:
        return render_template(
            "MB_form.html",
            features=FEATURE_ORDER,
            placeholders=PLACEHOLDERS,
            descriptions=DESCRIPTIONS,
            error="Please enter valid numeric values for all fields."
        )

    # Predict
    x = np.array([values], dtype=float)
    x_scaled = scaler.transform(x)
    proba_malignant = float(model.predict_proba(x_scaled)[0, 1])

    predicted_label = "Malignant" if proba_malignant >= 0.5 else "Benign"

    return render_template(
        "MB_result.html",
        predicted_label=predicted_label,
        proba_malignant=proba_malignant,
        inputs=inputs
    )


if __name__ == "__main__":
    app.run(debug=True)
