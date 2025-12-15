import os
import json
import pickle
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask import jsonify 
from werkzeug.utils import secure_filename
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from skimage.feature import local_binary_pattern
import joblib
import sys

# --- SHAP Import ---
import shap 

# --- Gemini API Import ---
from google import genai
from google.genai.errors import APIError 

app = Flask(__name__)

# -------------------------
# GEMINI API CONFIGURATION
# -------------------------
# SECURITY RECOMMENDATION: Set your API key as an environment variable
 

# Prioritize the environment variable, but fall back to the hardcoded key
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

gemini_client = None
if not GEMINI_API_KEY:
    # This warning indicates the environment variable wasn't found by the script.
    print("WARNING: GEMINI_API_KEY environment variable is NOT set. Chatbot is disabled.", file=sys.stderr)
else:
    try:
        # Initialize client only if the key exists
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client initialized successfully using environment variable.")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}", file=sys.stderr)
        gemini_client = None

# Global dictionary to temporarily cache prediction results and SHAP data
global_shap_cache = {}

# -------------------------
# Load assets for objective 1 (feature-based prediction)
# -------------------------
MODEL_PATH = os.path.join("models", "mlp_mean10.pkl")
SCALER_PATH = os.path.join("models", "scaler_mean10.pkl")
FEATURE_ORDER_PATH = os.path.join("models", "mean10_feature_order.json")

# Check and load Objective 1 assets
for p in [MODEL_PATH, SCALER_PATH, FEATURE_ORDER_PATH]:
    if not os.path.exists(p):
        import sys
        print(f"Error: Missing required file: {p}", file=sys.stderr)
        sys.exit(1)

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

# Check and load Objective 2 assets
if not os.path.exists(IMAGE_MODEL_PATH) or not os.path.exists(IMAGE_SCALER_PATH):
    import sys
    print(f"Error: Missing required Objective 2 model files: {IMAGE_MODEL_PATH} or {IMAGE_SCALER_PATH}", file=sys.stderr)
    sys.exit(1)

image_model = joblib.load(IMAGE_MODEL_PATH)
image_scaler = joblib.load(IMAGE_SCALER_PATH)

try:
    image_explainer = shap.TreeExplainer(image_model)
    global LBP_POINTS
    LBP_POINTS = 8
except Exception as e:
    print(f"Warning: Failed to initialize TreeExplainer. SHAP functionality will be disabled. Error: {e}")
    image_explainer = None


UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# -------------------------
# Utility function for XAI feature names
# -------------------------
def get_shap_feature_names():
    names = [
        "Average Blue Intensity", "Average Green Intensity", "Average Red Intensity",
        "Standard Deviation of Blue Intensity", "Standard Deviation of Green Intensity", "Standard Deviation of Red Intensity"
    ]
    lbp_count = LBP_POINTS + 2 
    names += [f"LBP Texture Pattern (Bin {i})" for i in range(lbp_count)]
    names += ["Boundary Edge Density (Complexity)"]
    
    return names

# -------------------------
# UI helpers for Objective 1 (omitted for brevity, assume they are present)
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
# Objective 1 Routes (kept as-is for brevity)
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
    # ... (content for mb_predict)
    inputs = {}
    values = []

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

    x = np.array([values], dtype=float)
    x_scaled = scaler.transform(x)
    proba_malignant = float(model.predict_proba(x_scaled)[0, 1])

    predicted_label = "Malignant" if proba_malignant >= 0.5 else "Benign"

    z = (x[0] - scaler.mean_) / scaler.scale_
    abs_idx = np.argsort(np.abs(z))[::-1][:3]

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
    # ... (content for mb_report_pdf - omitted for brevity)
    predicted_label = request.form.get("predicted_label", "Unknown")
    proba_raw = request.form.get("proba_malignant", "0")
    try:
        proba_malignant = float(proba_raw)
    except ValueError:
        proba_malignant = 0.0

    inputs = {}
    for feat in FEATURE_ORDER:
        raw = request.form.get(f"in_{feat}", "")
        try:
            inputs[feat] = float(raw)
        except ValueError:
            inputs[feat] = raw

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
# Objective 2 Image Prediction (kept as-is)
# -------------------------
@app.post("/image-predict")
def image_predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file_contents = file.read()
        file.seek(0)
        file.save(filepath)

        feats = extract_features(filepath)
        X_input = [feats]
        feats_scaled = image_scaler.transform(X_input)

        prob = image_model.predict_proba(feats_scaled)[0, 1]
        prediction = "Cancer suspecté" if prob > 0.5 else "Faible risque"

        shap_values_list = []
        if image_explainer and feats_scaled is not None:
            try:
                shap_values = image_explainer.shap_values(feats_scaled)[1][0] # Fix index to get a flat array
            except Exception as e:
                print(f"SHAP calculation failed: {e}")
                shap_values = np.zeros_like(feats_scaled[0])
            
            feature_names = get_shap_feature_names()
            
            for i, (name, raw_val, shap_val) in enumerate(zip(feature_names, X_input[0], shap_values)):
                shap_values_list.append({
                    "name": name,
                    "value": float(np.round(raw_val, 4)),
                    "shap_score": float(np.round(shap_val, 4))
                })

        global_shap_cache[filename] = {
            'shap_data': shap_values_list,
            'prediction': prediction,
            'score': prob
        }
        
        return redirect(url_for('show_result', key=filename))

    return render_template('MB_form.html', error="No image found in the form or invalid file.")


@app.get('/result')
def show_result():
    key = request.args.get('key')
    
    cached_data = global_shap_cache.pop(key, None)
    
    cleanup_uploaded_files(key)

    if not cached_data:
        return redirect(url_for('index'))

    score = cached_data['score']
    result = cached_data['prediction']
    shap_data = cached_data['shap_data']

    return render_template(
        'result.html', 
        score=score, 
        result=result, 
        shap_data=shap_data, 
        image_filename=key
    )

# ... (Utility functions: allowed_file, extract_features, cleanup_uploaded_files - kept as-is)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    lbp_points_val = 8 
    lbp_radius_val = 1
    
    mean_B, mean_G, mean_R = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    std_B, std_G, std_R = np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, lbp_points_val, lbp_radius_val, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, lbp_points_val + 3),
                           range=(0, lbp_points_val + 2))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return [mean_B, mean_G, mean_R,
            std_B, std_G, std_R,
            *hist,
            edge_density]

def cleanup_uploaded_files(file_name_to_keep=None):
    upload_folder = 'static/uploads'
    
    if not os.path.exists(upload_folder):
        return

    files_in_directory = os.listdir(upload_folder)
    
    for file_name in files_in_directory:
        if file_name == file_name_to_keep:
            continue
            
        file_path = os.path.join(upload_folder, file_name)
        
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_name}: {e}")

# -------------------------
# XAI Chatbot Route with Gemini Integration
# -------------------------
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "GET":
        return jsonify({
            'reply': 'Please use the chat interface on the result page to initiate a conversation via POST request.'
        })

    if request.method == "POST":
        user_input = request.json.get('user_input', '').strip()
        result = request.json.get('result', 'Unknown')
        score = request.json.get('score', 0.0)
        shap_data = request.json.get('shap_data', [])
        
        # Safely convert score
        try:
            score_val = float(score)
        except ValueError:
            score_val = 0.0

        # Construct a detailed context for Gemini
        context = f"""
        You are an XAI (Explainable AI) Assistant designed for medical personnel and patients. 
        Your goal is to provide a clear, concise, and helpful explanation for an image-based breast cancer risk prediction. 
        The prediction result is: {result}, with a probability score of {score_val * 100:.2f}%.
        The SHAP values below explain the model's decision. Positive SHAP scores push the prediction toward '{result}' (Cancer suspecté), and negative scores push it away (Faible risque).

        SHAP Data (Feature Name, Feature Value, SHAP Score):
        {json.dumps(shap_data, indent=2)}

        User's question: "{user_input}"
        
        Guidelines:
        1. Keep the tone professional, educational, and empathetic.
        2. Always mention the prediction and score first.
        3. Prioritize explaining the most relevant features (highest absolute SHAP scores).
        4. Relate features like 'Boundary Edge Density' or 'LBP Texture Pattern' back to visual characteristics of the tumor.
        5. Keep the response under 100 words unless the user asks for exhaustive detail.
        """

        if gemini_client and shap_data:
            try:
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=context
                )
                # Clean up the response to remove any markdown headers or unnecessary quotes
                reply_text = response.text.strip().replace("###", "").replace("**", "")
                return jsonify({'reply': reply_text})
            
            except APIError as e:
                print(f"Gemini API Error: {e}")
                # Fallback to a simple rule-based explanation if the API fails
                return jsonify({'reply': f"I apologize, the advanced AI assistant is currently unavailable ({e}). The predicted result is {result} with a score of {score_val * 100:.2f}%. For more details, contact technical support."})
            except Exception as e:
                 return jsonify({'reply': f"An unexpected error occurred: {e}"})

        # Fallback if Gemini client is not initialized or SHAP data is missing
        else:
            # Fallback logic (similar to the previous rule-based approach)
            explanation = f"The prediction is **{result}** with a risk score of **{score_val * 100:.2f} %**. "
            sorted_shap = sorted(shap_data, key=lambda x: abs(x.get('shap_score', 0)), reverse=True)[:3]

            if sorted_shap:
                top_feature = sorted_shap[0]
                effect = "increased" if top_feature['shap_score'] > 0 else "decreased"
                
                explanation += (
                    f"The most important factor is **{top_feature['name']}** (value: {top_feature['value']:.4f}). "
                    f"This feature **{effect}** the model's confidence in the '{result}' prediction."
                )
            
            explanation += "\n\nAsk me about the features if you want more details."
            return jsonify({'reply': explanation})


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Note: Use app.secret_key if you implement Flask sessions later
    app.run(debug=True)