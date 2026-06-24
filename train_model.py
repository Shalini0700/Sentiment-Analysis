"""
train_model.py
--------------
Downloads the IMDB movie reviews dataset from Hugging Face, trains a
TF-IDF + Logistic Regression classifier, evaluates it, and saves the
artifacts to the `model/` folder so the Streamlit app can load them.

Run once before starting the app:
    python train_model.py
"""

import os
import json

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
os.makedirs("model", exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load IMDB dataset (25 000 train, 25 000 test reviews)
# ---------------------------------------------------------------------------
print("Downloading IMDB dataset from Hugging Face (this may take a moment)...")
dataset = load_dataset("stanfordnlp/imdb")

train_texts  = dataset["train"]["text"]
train_labels = dataset["train"]["label"]   # 0 = Negative, 1 = Positive
test_texts   = dataset["test"]["text"]
test_labels  = dataset["test"]["label"]

print(f"  Training samples : {len(train_texts):,}")
print(f"  Test samples     : {len(test_texts):,}")

# ---------------------------------------------------------------------------
# 3. TF-IDF vectorisation
#    max_features=50 000 keeps vocab manageable; bigrams capture phrases like
#    "not good" or "highly recommend".
# ---------------------------------------------------------------------------
print("\nFitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),       # unigrams + bigrams
    stop_words="english",
    sublinear_tf=True,        # replace raw TF with 1 + log(TF) — helps on long reviews
)
X_train = vectorizer.fit_transform(train_texts)
X_test  = vectorizer.transform(test_texts)
print(f"  Vocabulary size  : {len(vectorizer.vocabulary_):,}")

# ---------------------------------------------------------------------------
# 4. Train Logistic Regression
#    lbfgs handles the dense weight matrix well; C=5.0 gives slight boost over
#    default C=1.0 on IMDB.
# ---------------------------------------------------------------------------
print("\nTraining Logistic Regression classifier...")
clf = LogisticRegression(
    C=5.0,
    max_iter=1000,
    solver="lbfgs",
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, train_labels)
print("  Training complete.")

# ---------------------------------------------------------------------------
# 5. Evaluate on the held-out test set
# ---------------------------------------------------------------------------
print("\nEvaluating on test set...")
y_pred   = clf.predict(X_test)
accuracy = accuracy_score(test_labels, y_pred)

print(f"  Test Accuracy : {accuracy:.4f} ({accuracy:.2%})")
print()
print(classification_report(test_labels, y_pred, target_names=["Negative", "Positive"]))

# ---------------------------------------------------------------------------
# 6. Save model artifacts
# ---------------------------------------------------------------------------
print("Saving model artifacts to model/ ...")
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")
joblib.dump(clf,        "model/logistic_regression.joblib")

# Persist metrics so the Streamlit app can display them without re-running tests
with open("model/metrics.json", "w") as f:
    json.dump({"sklearn_accuracy": round(accuracy, 4)}, f, indent=2)

print("  Saved: model/tfidf_vectorizer.joblib")
print("  Saved: model/logistic_regression.joblib")
print("  Saved: model/metrics.json")
print("\nAll done! You can now run:  streamlit run app.py")
