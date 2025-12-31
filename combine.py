import pandas as pd

# List of relative paths to datasets
files = [
    "data/emails/phishing_email.csv",
    "data/emails/SpamAssasin.csv",
    "data/emails/Enron.csv",
    "data/emails/Ling.csv",
    "data/emails/Nazario.csv",
    "data/emails/Nigerian_Fraud.csv"
]

dfs = []
for f in files:
    try:
        df = pd.read_csv(f, encoding='latin1')  # encoding fallback
        if not isinstance(df, pd.DataFrame):
            continue

        # Ensure subject/body columns exist
        if 'subject' not in df.columns:
            df['subject'] = ''
        if 'body' not in df.columns:
            df['body'] = ''
        
        df['subject'] = df['subject'].fillna('')
        df['body'] = df['body'].fillna('')
        df['text'] = df['subject'] + " " + df['body']

        # Only keep 'text' and 'label'
        if 'label' not in df.columns:
            continue  # skip CSVs without labels

        dfs.append(df[['text','label']])

    except Exception as e:
        print(f"[!] Failed to load {f}: {e}")

# Combine all
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv("data/emails/combined.csv", index=False)
    print("[✓] Combined dataset saved to data/emails/combined.csv")
else:
    print("[✗] No datasets were combined.")
