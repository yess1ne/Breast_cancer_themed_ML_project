import os
import json
import pickle
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from skimage.feature import local_binary_pattern
import joblib

app = Flask(__name__)

# -------------------------
# Load assets for objective 1 (feature-based prediction)
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
# Model paths for image classification (objective 2)
# -------------------------
IMAGE_MODEL_PATH = "risk_model.pkl"
IMAGE_SCALER_PATH = "scaler.pkl"

image_model = joblib.load(IMAGE_MODEL_PATH)
image_scaler = joblib.load(IMAGE_SCALER_PATH)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# -------------------------
# UI helpers (placeholders + short descriptions)
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

LABELS = {
    "radius_mean": "Average size (radius)",
    "texture_mean": "Texture variation",
    "perimeter_mean": "Boundary length (perimeter)",
    "area_mean": "Area (surface size)",
    "smoothness_mean": "Smoothness",
    "compactness_mean": "Compactness",
    "concavity_mean": "Concavity",
    "concave points_mean": "Concave points",
    "symmetry_mean": "Symmetry",
    "fractal_dimension_mean": "Boundary complexity (fractal dimension)",
}

# -------------------------
# Routes for Objective 1 (Tumor Classification)
# -------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/mb-form")
def mb_form():
    return render_template(
        "MB_form.html",
        features=FEATURE_ORDER,
        labels=LABELS,
        placeholders=PLACEHOLDERS,
        descriptions=DESCRIPTIONS,
        error=None
    )


@app.post("/mb-predict")
def mb_predict():
    inputs = {}
    values = []

    # Validate inputs for feature-based form
    try:
        for feat in FEATURE_ORDER:
            raw = request.form.get(feat, "").strip()
            if raw == "":
                return render_template(
                    "MB_form.html",
                    features=FEATURE_ORDER,
                    labels=LABELS,
                    placeholders=PLACEHOLDERS,
                    descriptions=DESCRIPTIONS,
                    error=f"Missing value for: {feat}"
                )

            val = float(raw)
            if not np.isfinite(val):
                return render_template(
                    "MB_form.html",
                    features=FEATURE_ORDER,
                    labels=LABELS,
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
            labels=LABELS,
            placeholders=PLACEHOLDERS,
            descriptions=DESCRIPTIONS,
            error="Please enter valid numeric values for all fields."
        )

    # Predict based on the input features
    x = np.array([values], dtype=float)
    x_scaled = scaler.transform(x)
    proba_malignant = float(model.predict_proba(x_scaled)[0, 1])

    predicted_label = "Malignant" if proba_malignant >= 0.5 else "Benign"

    # Generate highlights (optional visual aids)
    z = (x[0] - scaler.mean_) / scaler.scale_  # z-scores for the features
    abs_idx = np.argsort(np.abs(z))[::-1][:3]  # top 3 most unusual

    highlights = []
    for i in abs_idx:
        feat = FEATURE_ORDER[i]
        highlights.append({
            "label": LABELS.get(feat, feat),
            "value": inputs[feat],
            "z": float(np.round(z[i], 2))
        })

    return render_template(
        "MB_result.html",
        predicted_label=predicted_label,
        proba_malignant=proba_malignant,
        inputs=inputs,
        highlights=highlights
    )


@app.post("/mb-report.pdf")
def mb_report_pdf():
    # Read prediction data from the hidden fields posted by MB_result.html
    predicted_label = request.form.get("predicted_label", "Unknown")
    proba_raw = request.form.get("proba_malignant", "0")
    try:
        proba_malignant = float(proba_raw)
    except ValueError:
        proba_malignant = 0.0

    # Reconstruct inputs
    inputs = {}
    for feat in FEATURE_ORDER:
        raw = request.form.get(f"in_{feat}", "")
        try:
            inputs[feat] = float(raw)
        except ValueError:
            inputs[feat] = raw  # fallback (shouldn't happen)

    # Generate PDF in-memory
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Objective 1 — Tumor Classification Report")
    y -= 25

    c.setFont("Helvetica", 11)
    c.drawString(50, y, "This report is generated for educational purposes only.")
    y -= 16
    c.drawString(50, y, "It must not be interpreted as medical diagnosis.")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Predicted result: {predicted_label}")
    y -= 18
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Estimated probability (malignant): {proba_malignant*100:.2f}%")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Submitted measurements:")
    y -= 18

    c.setFont("Helvetica", 10)
    for feat in FEATURE_ORDER:
        label = LABELS.get(feat, feat)
        val = inputs.get(feat, "")
        line = f"- {label}: {val}"
        c.drawString(55, y, line)
        y -= 14
        if y < 70:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()

    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="MB_result_report.pdf",
        mimetype="application/pdf"
    )


# -------------------------
# Routes for Objective 2 (Image Classification)
# -------------------------
@app.post("/image-predict")
def image_predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file and allowed_file(file.filename):
        # Secure the filename and save it
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract features and scale the data
        feats = extract_features(filepath)
        feats = image_scaler.transform([feats])

        # Get prediction probability (ensure it's only calculated once)
        prob = image_model.predict_proba(feats)[0, 1]
        prediction = "Cancer suspecté" if prob > 0.5 else "Faible risque"

        # Debugging: print the score value here
        print(f"Prediction: {prediction}, Score: {prob}")

        # Correction: Pass the raw probability as a string ('prob') to the URL parameter 'score'.
        # This will be safely converted to a float and formatted in the template.
        return redirect(url_for('show_result', score=str(prob), result=prediction))

    return render_template('MB_form.html', error="No image found in the form.")


@app.get('/result')
def show_result():
    score = request.args.get('score')  # score is received as a string (e.g., '0.0650...')
    result = request.args.get('result')
    # The score is now passed to the template. The formatting fix is below in the HTML.
    cleanup_uploaded_files()
    return render_template('result.html', score=score, result=result)


# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # --- Colors ---
    mean_B, mean_G, mean_R = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    std_B, std_G, std_R = np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2])

    # --- LBP ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8 * 1, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, 8 * 1 + 3),
                           range=(0, 8 * 1 + 2))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6

    # --- Edge Density ---
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return [mean_B, mean_G, mean_R,
            std_B, std_G, std_R,
            *hist,
            edge_density]

def cleanup_uploaded_files():
    upload_folder = 'static/uploads'
    
    # Get all files in the upload folder
    files_in_directory = os.listdir(upload_folder)
    
    # You can implement conditions to delete files, e.g., delete after a certain time, or after they have been processed
    for file_name in files_in_directory:
        file_path = os.path.join(upload_folder, file_name)
        
        # Here we just delete everything; you can change this condition based on your needs
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_name}")



@app.post("/generate-pdf")
def generate_pdf():
    predicted_label = request.form.get('predicted_label', 'Unknown')
    proba_malignant = request.form.get('proba_malignant', '0')

    inputs = {}
    for feat in FEATURE_ORDER:
        raw = request.form.get(f'in_{feat}', '')
        try:
            inputs[feat] = float(raw)
        except ValueError:
            inputs[feat] = raw  # fallback (shouldn't happen)

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Objective 1 — Tumor Classification Report")
    y -= 25

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Predicted result: {predicted_label}")
    y -= 18
    c.drawString(50, y, f"Estimated probability (malignant): {proba_malignant}")
    y -= 30

    c.setFont("Helvetica", 10)
    for feat in FEATURE_ORDER:
        label = LABELS.get(feat, feat)
        val = inputs.get(feat, "")
        line = f"- {label}: {val}"
        c.drawString(55, y, line)
        y -= 14
        if y < 70:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()

    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="tumor_report.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    app.run(debug=True)
