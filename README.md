# 🎬 Sentiment Analysis Web App

A beginner-friendly web application that analyses the sentiment of movie reviews (or any text) using **two different models** side-by-side and lets you compare their predictions in real time.

---

## 📸 Screenshots

> _Add screenshots after running the app._

| Home / Input | Results & Chart |
|:---:|:---:|
| ![Input screen](screenshots/input.png) | ![Results screen](screenshots/results.png) |

---

## ✨ Features

- **Dual-model comparison** — TF-IDF + Logistic Regression vs. DistilBERT, both shown at once
- **Confidence progress bars** — instantly see how certain each model is
- **Interactive grouped bar chart** — Plotly chart comparing both models across Positive / Negative classes
- **Model accuracy panel** — IMDB test-set accuracy for the sklearn model and published benchmark accuracy for DistilBERT
- **Beginner-friendly code** — every file is commented and structured for readability

---

## 🧠 How It Works

### Model 1 — Scikit-learn (TF-IDF + Logistic Regression)

1. The **IMDB movie reviews dataset** (50 000 reviews) is downloaded from Hugging Face.
2. Reviews are converted into numerical vectors using **TF-IDF** (Term Frequency – Inverse Document Frequency), which captures how important each word is across the corpus.
3. A **Logistic Regression** classifier is trained on 25 000 reviews and evaluated on the remaining 25 000.
4. The trained model is saved with **joblib** and loaded instantly by the app.

### Model 2 — DistilBERT (pre-trained, no local training needed)

- `distilbert-base-uncased-finetuned-sst-2-english` is a smaller, faster version of BERT fine-tuned by Hugging Face on the **SST-2** sentiment dataset.
- It understands word **context** (e.g. "not bad" vs. "bad"), making it better at nuanced language.
- Loaded via the Hugging Face `transformers` library; weights are downloaded automatically on first run (~250 MB).

---

## 🗂️ Project Structure

```
sentiment_app/
│
├── app.py              # Streamlit web application
├── train_model.py      # Download dataset, train & save sklearn model
├── requirements.txt    # Python dependencies
├── README.md           # This file
│
└── model/              # Created after running train_model.py
    ├── tfidf_vectorizer.joblib
    ├── logistic_regression.joblib
    └── metrics.json
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.9 or newer
- `pip` (comes with Python)

### Step 1 — Clone / download the project

```bash
git clone https://github.com/Shalini0700/retro-shooter.git
cd sentiment_app
```

Or just copy the `sentiment_app/` folder to your machine.

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is about 2 GB. If you only want CPU support and a smaller download, replace the `torch` line in `requirements.txt` with the wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

### Step 4 — Train the sklearn model (one-time setup, ~3–5 min)

```bash
python train_model.py
```

This will:
- Download the IMDB dataset from Hugging Face (~84 MB)
- Train and evaluate the TF-IDF + Logistic Regression model
- Save artifacts to `model/`

Sample output:
```
Downloading IMDB dataset from Hugging Face...
  Training samples : 25,000
  Test samples     : 25,000
Fitting TF-IDF vectorizer...
Training Logistic Regression classifier...
Test Accuracy : 0.8982 (89.82%)
               precision    recall  f1-score
    Negative       0.90      0.90      0.90
    Positive       0.90      0.90      0.90
Saving model artifacts to model/ ...
All done! You can now run:  streamlit run app.py
```

### Step 5 — Launch the Streamlit app

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

On first launch, DistilBERT weights are downloaded from Hugging Face (~250 MB, cached for future runs).

---

## 🛠️ Tech Stack

| Component | Library / Tool | Version |
|---|---|---|
| Web UI | [Streamlit](https://streamlit.io) | ≥ 1.34 |
| Classical ML | [scikit-learn](https://scikit-learn.org) | ≥ 1.4 |
| Dataset | [Hugging Face Datasets](https://huggingface.co/docs/datasets) | ≥ 2.18 |
| Deep learning model | [Hugging Face Transformers](https://huggingface.co/docs/transformers) | ≥ 4.39 |
| Neural network backend | [PyTorch](https://pytorch.org) | ≥ 2.2 |
| Model persistence | [joblib](https://joblib.readthedocs.io) | ≥ 1.3 |
| Charting | [Plotly](https://plotly.com/python/) | ≥ 5.20 |
| Numerics | [NumPy](https://numpy.org) | ≥ 1.26 |

---

## 📊 Model Performance

| Model | Test Accuracy | Dataset |
|---|---|---|
| TF-IDF + Logistic Regression | ~89–90 % | IMDB (25 000 reviews) |
| DistilBERT (SST-2 fine-tuned) | ~93 % | SST-2 benchmark |

DistilBERT wins on accuracy because its transformer architecture understands word order and context. The sklearn model is much faster to run (no GPU needed) and easier to interpret.

---

## 💡 Ideas for Extending the Project

- Add a third model (e.g. SVM or a fine-tuned RoBERTa)
- Allow the user to upload a `.csv` of reviews for batch analysis
- Display a word cloud of the most influential TF-IDF features
- Fine-tune DistilBERT on the IMDB dataset locally for even higher accuracy

---

## 📄 License

MIT — feel free to use, modify, and share.
