# Emotion Detection from Text

A dark-themed Streamlit app for detecting the emotion of user-entered text. The app uses TF-IDF vectorization with a **LinearSVC** model (calibrated for confidence scores), and includes an animated floating emoji background.

## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://emotion-detection-from-text-6obwvzmfujmjgskstwp6tg.streamlit.app/)

## 🖥️ App Preview

| Home | Prediction Result |
|---|---|
| ![App Home](screenshots/app_home.jpeg) | ![App Prediction](screenshots/app_prediction.jpeg) |

## What's included

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and emotion prediction logic |
| `emotion_detection_pipeline.ipynb` | Notebook for model training and experimentation |
| `train.txt` | Training dataset |
| `val.txt` | Validation dataset (used to measure accuracy after training) |
| `test.txt` | Test dataset |
| `requirements.txt` | Python dependencies |
| `model.pkl` | Serialized model (auto-generated, gitignored) |
| `vectorizer.pkl` | Serialized TF-IDF vectorizer (auto-generated, gitignored) |

## Features

- Clean, modern dark UI with glass-card design
- Floating emoji background animation
- Text preprocessing: lowercase, punctuation removal, stopword filtering
- **LinearSVC** model with calibrated probabilities (better accuracy than Logistic Regression)
- **Confidence score** displayed as an animated progress bar after each prediction
- **Sidebar** showing validation accuracy, model info, and all detectable emotions
- Auto-trains on first run if model files are missing

## Setup

1. Clone the repo and create a virtual environment.

```bash
git clone https://github.com/kananiisha/Emotion-Detection-from-Text.git
cd Emotion-Detection-from-Text
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Download NLTK stopwords (only needed once).

```python
import nltk
nltk.download('stopwords')
```

## Run the app

```bash
streamlit run app.py
```

Open the local URL shown in your terminal.

## Notes

- `app.py` auto-trains the model on first run using `train.txt`, then evaluates on `val.txt` and displays accuracy in the sidebar.
- Model files (`model.pkl`, `vectorizer.pkl`) are gitignored — they are regenerated automatically.
- The `.gitignore` excludes `model.pkl`, `vectorizer.pkl`, `.venv`, `__pycache__`, and `val_accuracy.txt`.

## Model Details

| Property | Value |
|---|---|
| Algorithm | LinearSVC (wrapped in CalibratedClassifierCV) |
| Vectorizer | TF-IDF with 1–2 word n-grams, max 50k features |
| Training data | `train.txt` |
| Evaluated on | `val.txt` |

## GitHub

```
https://github.com/kananiisha/Emotion-Detection-from-Text.git
```