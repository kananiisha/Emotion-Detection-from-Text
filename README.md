# 🎭 Emotion Detection from Text
### NLP Project — Text Classification with Streamlit

**Tools:** Python, Scikit-learn, NLTK, Streamlit, TF-IDF

**Objective:** Detect the emotion of user-entered text using a machine learning NLP pipeline with a clean dark-themed Streamlit web app.

---

## 📁 Files
- `app.py` — Streamlit interface and emotion prediction logic
- `main.ipynb` — Notebook for model training and experimentation
- `train.txt` — Training dataset
- `test.txt` — Test dataset
- `val.txt` — Validation dataset

## 🛠️ Libraries Used
pandas, scikit-learn, nltk, streamlit

## 🤖 Model Details
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifier:** Logistic Regression
- **Preprocessing:** Lowercase, punctuation removal, stopword filtering
- **Auto-training:** Model trains automatically if `model.pkl` not found

## ✨ Features
- Clean modern dark UI
- Floating emoji background animation
- Real-time emotion prediction on user input
- Text preprocessing pipeline built-in

## ▶️ How to Run

### Install dependencies
pip install streamlit pandas scikit-learn nltk

### Download NLTK stopwords
```python
import nltk
nltk.download('stopwords')
```

### Run the Streamlit app
streamlit run app.py

Open the local URL displayed by Streamlit in your browser.

## 📝 Notes
- `app.py` will train the model automatically if `model.pkl` and `vectorizer.pkl` are not present
- The app uses `train.txt` as the default training dataset
- The `.gitignore` excludes `model.pkl`, `vectorizer.pkl`, `.venv`, and `__pycache__`

## 👩‍💻 Author
Isha Kanani | Data Science & ML Enthusiast
