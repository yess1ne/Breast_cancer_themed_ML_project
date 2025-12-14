import os
import cv2
import numpy as np
import pandas as pd
import joblib

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#à changer selon ton environnement
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\Yessine\Documents\4DS3\4DS3_AML\proj\BreastCare_image_dataset_extraction\archive (3)\archive (3)\IDC_regular_ps50_idx5\9383"
OUTPUT_PATH = PROJECT_DIR

MAX_IMAGES_PER_CLASS = 200  # réduire le dataset
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # --- Couleurs ---
    mean_B, mean_G, mean_R = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    std_B, std_G, std_R = np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2])

    # --- LBP ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, LBP_POINTS + 3),
                           range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6

    # --- Densité de contours ---
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return [mean_B, mean_G, mean_R,
            std_B, std_G, std_R,
            *hist,
            edge_density]

data = []

for label in ["0", "1"]:
    folder = os.path.join(DATASET_PATH, label)
    images = os.listdir(folder)[:MAX_IMAGES_PER_CLASS]

    for img_name in images:
        img_path = os.path.join(folder, img_name)
        feats = extract_features(img_path)
        if feats is not None:
            data.append(feats + [int(label)])

columns = (
    ["mean_B","mean_G","mean_R",
     "std_B","std_G","std_R"] +
    [f"lbp_{i}" for i in range(LBP_POINTS + 2)] +
    ["edge_density","target"]
)

df = pd.DataFrame(data, columns=columns)
df.head()

df.to_csv(os.path.join(OUTPUT_PATH, "features.csv"), index=False)

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)
risk_scores = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, model.predict(X_test)))

df_results = pd.DataFrame(X_test, columns=X.columns)
df_results["risk_score"] = risk_scores
df_results["true_label"] = y_test.reset_index(drop=True)

df_results.head()
joblib.dump(model, os.path.join(OUTPUT_PATH, "risk_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_PATH, "scaler.pkl"))
df_results.to_csv(os.path.join(OUTPUT_PATH, "risk_results.csv"), index=False)
def predict_image(image_path):
    model = joblib.load(os.path.join(OUTPUT_PATH, "risk_model.pkl"))
    scaler = joblib.load(os.path.join(OUTPUT_PATH, "scaler.pkl"))

    feats = extract_features(image_path)
    feats = scaler.transform([feats])

    prob = model.predict_proba(feats)[0,1]
    prediction = "Cancer suspecté" if prob > 0.5 else "Faible risque"

    return prob, prediction

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
image_test = r"C:\Users\Yessine\Documents\4DS3\4DS3_AML\proj\BreastCare_image_dataset_extraction\archive (3)\archive (3)\IDC_regular_ps50_idx5\9383\0\9383_idx5_x51_y751_class0.png"
score, result = predict_image(image_test)

print("Risk score :", round(score, 3))
print("Interprétation :", result)
