<!-- # phishing-detector
This project is a simple command-line tool that analyzes email text and predicts whether the email is Phishing, Suspicious, or Legitimate. The goal of the project is to demonstrate how machine learning and modern NLP embeddings can be used to detect phishing emails in a practical, lightweight, and explainable way. -->
# AI Powered Phishing Email Detector

## Overview
This project is a simple command-line tool that analyzes email text and predicts whether the email is Phishing, Suspicious, or Legitimate.  
The goal of the project is to demonstrate how machine learning and modern NLP embeddings can be used to detect phishing emails in a practical, lightweight, and explainable way.

## What This Project Does
- Takes raw email text as input
- Converts the text into semantic embeddings using DistilBERT
- Uses a Logistic Regression classifier to predict phishing probability
- Outputs a human-friendly verdict with confidence

## Verdictsgit add .

- **LEGIT** (Safe email)
- **SUSPICIOUS** (Needs review)
- **PHISHING** (High risk)

## Tech Stack Used
- Python
- PyTorch
- HuggingFace Transformers (DistilBERT)
- Scikit-learn
- Pandas
- Joblib

## Why This Approach
Instead of using basic keyword matching or TF-IDF alone, this project uses DistilBERT embeddings to capture the intent and context of email text (urgency, threats, authority abuse).  
The classifier itself remains simple (Logistic Regression) to keep the system:
- Interpretable
- Lightweight
- Easy to debug
- Interview-friendly

## Project Structure
```
phishing-detector/
│
├── data/
│   └── emails/
│       └── combined.csv (dataset with text and labels)
│
├── model/
│   └── phishing_model.pkl (saved tokenizer + BERT + classifier)
│
├── train.py (training script)
├── test.py (CLI testing script)
└── README.md
```

## Dataset Format
The dataset CSV must contain at least two columns:  
- `text` → email content  
- `label` → 1 for phishing, 0 for legitimate

## How Training Works
1. Load phishing email dataset
2. Tokenize emails using DistilBERT tokenizer
3. Generate embeddings from DistilBERT CLS token
4. Train Logistic Regression on embeddings
5. Evaluate accuracy
6. Save trained model using Joblib

## How Testing Works
1. Load saved model
2. Accept email text
3. Generate embedding using the same BERT model
4. Predict phishing probability
5. Convert probability into verdict

## Example Output
**Email:** Please verify your account immediately  
**Verdict:** PHISHING (High Risk)  
**Phishing Probability:** 99.93%

**Email:** Team meeting at 5 PM today  
**Verdict:** LEGIT  
**Phishing Probability:** 0.37%

## How to Run
1. Train the model:  
   ```
   python train.py
   ```
2. Test emails:  
   ```
   python test.py
   ```

## Limitations
- Model is only as good as the dataset
- Some legitimate transactional emails may appear suspicious
- Does not analyze links, headers, or sender metadata

## Future Improvements
- Compare with TF-IDF baseline
- Add URL and domain analysis
- Add email header inspection
- Add simple web interface
- Add LLM-based explanation (optional)

