import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("data/emails/combined.csv")

X_text = df["text"].astype(str).tolist()
y = df["label"].tolist()   # 0 = legit, 1 = phishing

# -----------------------
# Load embedding model (CPU-optimized)
# -----------------------
print("[+] Loading MiniLM embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("[+] Generating embeddings...")
X_embeddings = embedder.encode(
    X_text,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

# -----------------------
# Train / Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# Train classifier
# -----------------------
print("[+] Training Logistic Regression classifier...")
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)
clf.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[+] Test Accuracy: {acc * 100:.2f}%")

# -----------------------
# Save model (IMPORTANT: order matters)
# -----------------------
joblib.dump((embedder, clf), "model/phishing_model.pkl")
print("[+] Model saved to model/phishing_model.pkl")
