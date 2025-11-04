import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
data_dir = os.path.join(BASE_DIR, "data")
os.makedirs(models_dir, exist_ok=True)

# -------------------------------
# Load main symptom → disease dataset
# -------------------------------
df = pd.read_csv(os.path.join(data_dir, "disease prediction by symptom.csv"))

def clean_symptom(s):
    if isinstance(s, str) and s != "0":
        return s.strip().lower().replace(" ", "_")
    return None

def row_to_symptom_text(row):
    syms = [clean_symptom(row[c]) for c in row.index if "Symptom" in c]
    syms = [s for s in syms if s]
    return " ".join(syms)

df["symptom_text"] = df.drop("Disease", axis=1).apply(row_to_symptom_text, axis=1)
X_text = df["symptom_text"].astype(str)
y_labels = df["Disease"].astype(str)

# -------------------------------
# Vectorizer + Binarizer
# -------------------------------
tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2)
X = tfidf.fit_transform(X_text)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform([[d] for d in y_labels])

# -------------------------------
# Train classifier
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, solver="saga", C=2.0), n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print("Validation F1 (micro):", f1_score(y_val, y_pred, average="micro"))

# -------------------------------
# Build Disease → Drug mapping
# -------------------------------
drug_map = {}

# 1. From drug prescription CSV
drug_file = os.path.join(data_dir, "Drug prescription to disease.csv")
if os.path.exists(drug_file):
    drug_df = pd.read_csv(drug_file)
    for _, row in drug_df.iterrows():
        disease = str(row["disease"]).strip().lower()
        drug = str(row["drug"]).strip().lower()
        drug_map.setdefault(disease, []).append(drug)

# 2. From UCI dataset (include ratings)
uci_file = os.path.join(data_dir, "UCI_DtoDrug.csv")
if os.path.exists(uci_file):
    uci_df = pd.read_csv(uci_file)
    for _, row in uci_df.iterrows():
        disease = str(row["condition"]).strip().lower()
        drug = str(row["drugName"]).strip().lower()
        rating = row.get("rating", 5)
        try:
            rating = float(rating)
        except:
            rating = 5.0
        drug_map.setdefault(disease, []).append(f"{drug} (rating {rating})")

# -------------------------------
# Save artifacts into models/
# -------------------------------
joblib.dump(tfidf, os.path.join(models_dir, "vectorizer.pkl"))
joblib.dump(clf, os.path.join(models_dir, "disease_model.pkl"))
joblib.dump(mlb, os.path.join(models_dir, "mlb.pkl"))
joblib.dump(drug_map, os.path.join(models_dir, "drug_map.pkl"))

print("✅ Training complete. Model + drug map saved in models/")
