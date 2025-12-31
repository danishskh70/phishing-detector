import joblib

# -----------------------------
# Load model artifacts
# -----------------------------
embedder, clf = joblib.load("model/phishing_model.pkl")

# -----------------------------
# Test emails
# -----------------------------
test_emails = [
    "Please verify your account immediately",
    "Team meeting at 5 PM today. See you all.",
    "Invoice attached for your recent payment. Please review.",
    "Your Amazon order has shipped. Track here.",
    "Update billing information to avoid service interruption.",
    "Family dinner on Sunday. RSVP please.",
    "Confirm your password reset now or lose access.",
    "Reminder: Meeting rescheduled to 3 PM tomorrow",
    "Some random newsletter text here"
]

# -----------------------------
# Prediction function
# -----------------------------
def predict_email(email_text):
    emb = embedder.encode(
        [email_text],
        normalize_embeddings=True
    )

    legit_prob, phishing_prob = clf.predict_proba(emb)[0]
    conf = phishing_prob * 100

    if phishing_prob >= 0.85:
        verdict = "PHISHING 🚨 (High Risk)"
    elif phishing_prob >= 0.65:
        verdict = "SUSPICIOUS ⚠️ (Review Recommended)"
    else:
        verdict = "LEGIT ✅"

    print(f"Email: {email_text}")
    print(f"Verdict: {verdict}")
    print(f"Phishing Probability: {conf:.2f}%")
    print("-" * 50)

# -----------------------------
# Run predictions
# -----------------------------
for email in test_emails:
    predict_email(email)
