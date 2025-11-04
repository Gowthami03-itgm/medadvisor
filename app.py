from flask import Flask, request, jsonify, render_template
import joblib, os, pandas as pd, numpy as np

app = Flask(__name__, static_folder="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")

# -------------------------------
# Load trained artifacts
# -------------------------------
vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
clf = joblib.load(os.path.join(models_dir, "disease_model.pkl"))
mlb = joblib.load(os.path.join(models_dir, "mlb.pkl"))
drug_map = joblib.load(os.path.join(models_dir, "drug_map.pkl"))

# -------------------------------
# Helper: preprocess symptoms
# -------------------------------
def preprocess_symptoms(text: str) -> str:
    """Match preprocessing from training: lowercase, underscores, no commas."""
    parts = [s.strip().lower().replace(" ", "_") for s in text.split(",") if s.strip()]
    return " ".join(parts)

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/symptoms")
def get_symptoms():
    """Provide symptom list for auto-suggestions in frontend"""
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "Symptom-severity.csv"))
    symptom_list = df["Symptom"].dropna().str.strip().tolist()
    return jsonify(symptom_list)

@app.route("/predict", methods=["POST"])
def predict():
    """Predict disease(s) and recommend medicines"""
    symptoms = request.form.get("symptoms", "")
    processed = preprocess_symptoms(symptoms)

    # Debug logs
    print("\n=== DEBUG ===")
    print("Raw input:", symptoms)
    print("Processed input:", processed)

    if not processed:
        return render_template("result.html", predictions=[{
            "disease": "⚠️ No symptoms provided",
            "probability": "0.00",
            "medicines": ["Please enter at least one symptom"]
        }])

    # Vectorize input
    X = vectorizer.transform([processed])
    print("Vectorized shape:", X.shape, "Nonzero features:", X.nnz)

    # Predict probabilities
    probs = clf.predict_proba(X)
    if isinstance(probs, list):  # OneVsRestClassifier returns list
        p_arr = np.array([
            arr[0, 1] if arr.shape[1] > 1 else arr[0, 0]
            for arr in probs
        ])
    else:
        p_arr = probs[0]

    # Sort by probability
    idxs = np.argsort(p_arr)[::-1]

    # Confidence threshold
    threshold = 0.10
    results = []

    for i in idxs:
        prob = p_arr[i]
        if prob < threshold:
            continue  # skip weak matches

        lbl = mlb.classes_[i]
        disease = lbl.lower()

        # Lookup medicines
        meds = drug_map.get(disease, [])
        if not meds:
            meds = ["⚠️ No medicine found in dataset", "Consult a physician"]

        results.append({
            "disease": lbl.title(),
            "probability": f"{prob:.2f}",
            "medicines": meds[:5]
        })

    # If no diseases cross threshold → return best guess
    if not results and len(idxs) > 0:
        i = idxs[0]
        lbl = mlb.classes_[i]
        disease = lbl.lower()
        meds = drug_map.get(disease, ["⚠️ No medicine found in dataset", "Consult a physician"])
        results.append({
            "disease": lbl.title(),
            "probability": f"{p_arr[i]:.2f}",
            "medicines": meds[:5]
        })

    print("Predictions returned:", results)
    print("=== END DEBUG ===\n")

    return render_template("result.html", predictions=results)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
