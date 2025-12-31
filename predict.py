import sys
import joblib
import torch

# -----------------------------
# Load model
# -----------------------------
tokenizer, bert_model, clf = joblib.load("model/phishing_detector.pkl")

# -----------------------------
# Get email input
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python predict.py '<subject> | <body>'")
    sys.exit(1)

input_text = sys.argv[1]
if "|" in input_text:
    subject, body = map(str.strip, input_text.split("|",1))
else:
    subject, body = "", input_text

email_text = subject + " " + body

# -----------------------------
# Predict
# -----------------------------
inputs = tokenizer([email_text], return_tensors='pt', truncation=True, padding=True, max_length=128)
with torch.no_grad():
    outputs = bert_model(**inputs)
emb = outputs.last_hidden_state[:,0,:].numpy()  # CLS token

pred = clf.predict(emb)[0]
prob = clf.predict_proba(emb)[0]
conf = max(prob) * 100

if conf < 40:
    label = "SAFE ✅"
elif conf < 70:
    label = "SUSPICIOUS ⚠️ (Review Recommended)"
else:
    label = "PHISHING 🚨 (High Risk)"

print("\nResult:")
print("------")
print(f"Verdict   : {label}")
print(f"Confidence: {conf:.2f}%")
