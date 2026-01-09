import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def load_dataset(csv_path: Path) -> pd.DataFrame:
    # Pruža kompatibilnost sa čestim verzijama dataset-a (v1/v2 ili label/text)
    df = pd.read_csv(csv_path, encoding="latin-1")
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    if {"v1", "v2"}.issubset(set(cols)):
        df = df.rename(columns={"v1": "label", "v2": "text"})
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV mora imati kolone 'label' i 'text' (ili 'v1' i 'v2').")

    # Očisti nepotrebne kolone
    df = df[["label", "text"]].dropna()
    return df

def train_and_eval(df: pd.DataFrame):
    X = df["text"].astype(str)
    y = df["label"].astype(str).str.lower().str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB())
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    return pipe

def main():
    parser = argparse.ArgumentParser(description="Simple Spam Classifier (TF-IDF + Naive Bayes)")
    parser.add_argument("--data", type=str, default="data/spam.csv",
                        help="Path do CSV fajla (default: data/spam.csv)")
    parser.add_argument("--save-model", type=str, default="model.joblib",
                        help="Putanja gde se čuva istrenirani model (default: model.joblib)")
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise SystemExit(f"Nema fajla: {csv_path}. Stavi dataset u data/ kao spam.csv.")

    df = load_dataset(csv_path)
    model = train_and_eval(df)

    joblib.dump(model, args.save_model)
    print(f"\nModel sačuvan u: {args.save_model}")

if __name__ == "__main__":
    main()
