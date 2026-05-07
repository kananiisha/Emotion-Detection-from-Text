# 🧠 Emotion Detection from Text

A dark-themed Streamlit web app that detects the emotion hidden in any sentence — powered by a **LinearSVC** machine learning model with calibrated confidence scores and a live floating emoji rain background.

---

## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://emotion-detection-from-text-6obwvzmfujmjgskstwp6tg.streamlit.app/)

---

## 📸 Screenshots

| Home | Prediction Result |
|------|-------------------|
| ![App Home](screenshots/app_home.jpeg) | ![App Prediction](screenshots/app_prediction.jpeg) |

---

## 🎯 Features

- **Emotion Prediction** — Enter any sentence and instantly detect its underlying emotion
- **Confidence Score** — Animated progress bar showing model certainty for each prediction
- **Dark Glass UI** — Frosted-glass card design with animated floating emoji background
- **Smart Preprocessing** — Lowercase normalization, punctuation removal, and stopword filtering
- **Sidebar Dashboard** — Displays validation accuracy, model info, and all detectable emotions
- **Auto-Training** — Model trains automatically on first run if `.pkl` files are missing
- **Privacy First** — All processing happens locally; no data is stored or sent externally

---

## 📊 Model Performance

| Metric | Value |
|--------|:-----:|
| Algorithm | LinearSVC (CalibratedClassifierCV) |
| Vectorizer | TF-IDF · 1–2 word n-grams · 50k features |
| Training Samples | 16,000 |
| Validation Samples | 2,000 |
| Test Samples | 2,000 |
| **Validation Accuracy** | **~90.90%** ✅ |
| **Test Accuracy** | **~89.90%** ✅ |

### Per-Class Performance (on Test Set)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|:---------:|:------:|:--------:|:-------:|
| 😡 anger | ~0.90 | ~0.89 | ~0.90 | 275 |
| 😨 fear | ~0.88 | ~0.87 | ~0.87 | 224 |
| 😄 joy | ~0.92 | ~0.93 | ~0.92 | 695 |
| 😍 love | ~0.76 | ~0.81 | ~0.78 | 159 |
| 😢 sadness | ~0.94 | ~0.94 | ~0.94 | 581 |
| 😲 surprise | ~0.74 | ~0.65 | ~0.69 | 66 |
| **Overall** | **~0.90** | **~0.90** | **~0.90** | **2000** |

> **LinearSVC** wrapped in `CalibratedClassifierCV` was chosen over plain Logistic Regression for higher accuracy and native probability/confidence score support.

---

## 🏗️ Architecture

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS (glassmorphism dark theme) |
| Background Animation | Vanilla JS emoji rain via `streamlit.components` |
| ML Model | LinearSVC → CalibratedClassifierCV (scikit-learn) |
| Vectorizer | TF-IDF (1–2 word n-grams, max 50,000 features) |
| Text Preprocessing | NLTK stopwords + lowercase + punctuation removal |

---

## 📁 Project Structure

```
Emotion-Detection-from-Text/
│
├── app.py                              # Main Streamlit app — UI, prediction logic, auto-training
├── emotion_detection_pipeline.ipynb   # EDA + model training & experimentation notebook
│
├── train.txt                          # Labelled training dataset     (16,000 samples)
├── val.txt                            # Validation dataset            (2,000 samples)
├── test.txt                           # Test dataset                  (2,000 samples)
│
├── model.pkl                          # Serialized trained model      (auto-generated, gitignored)
├── vectorizer.pkl                     # Serialized TF-IDF vectorizer  (auto-generated, gitignored)
├── val_accuracy.txt                   # Cached validation accuracy    (auto-generated, gitignored)
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignores .pkl, .venv, __pycache__, val_accuracy.txt
│
└── screenshots/
    ├── app_home.jpeg                  # Screenshot — home screen
    └── app_prediction.jpeg            # Screenshot — prediction result
```

---

## 🔍 How It Works

```
User Input Text
       │
       ▼
  Preprocessing
  ┌─────────────────────────────────────┐
  │  Lowercase  →  Remove Punctuation   │
  │         →  Remove Stopwords         │
  └─────────────────────────────────────┘
       │
       ▼
  TF-IDF Vectorization
  (1–2 word n-grams, max 50,000 features)
       │
       ▼
  LinearSVC + CalibratedClassifierCV
       │
       ├──▶  Predicted Emotion Label  (e.g. "joy")
       └──▶  Confidence Score         (e.g. 87%)
```

---

## 💡 Detectable Emotions

| Emotion | Emoji | Training Samples |
|---------|:-----:|:----------------:|
| Joy | 😄 | 5,362 |
| Sadness | 😢 | 4,666 |
| Anger | 😡 | 2,159 |
| Fear | 😨 | 1,937 |
| Love | 😍 | 1,304 |
| Surprise | 😲 | 572 |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/kananiisha/Emotion-Detection-from-Text.git
cd Emotion-Detection-from-Text
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Stopwords *(one-time only)*
```python
import nltk
nltk.download('stopwords')
```

### 5. Run the App
```bash
streamlit run app.py
```

Open the local URL shown in your terminal. On first run, the model trains automatically using `train.txt` and evaluates on `val.txt` — accuracy is then displayed in the sidebar.

---

## 📦 Dependencies

```
streamlit
pandas
scikit-learn
nltk
```

---

## 📝 Notes

- `model.pkl` and `vectorizer.pkl` are **gitignored** — regenerated automatically on first run.
- Validation accuracy is cached in `val_accuracy.txt` after training and shown in the sidebar on every subsequent run.
- The `.gitignore` excludes `model.pkl`, `vectorizer.pkl`, `val_accuracy.txt`, `.venv/`, and `__pycache__/`.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m "Add your feature"`
4. Push to the branch — `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use and modify.

---

**Built with ❤️ using Streamlit + LinearSVC ML**