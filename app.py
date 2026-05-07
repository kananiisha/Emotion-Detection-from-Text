import os
import pickle
import pandas as pd
import nltk
import string
import streamlit as st
import streamlit.components.v1 as components
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

nltk.download('stopwords', quiet=True)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
TRAIN_PATH = "train.txt"
VAL_PATH = "val.txt"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path, sep=';', names=['text', 'emotion'])


@st.cache_resource(show_spinner=False)
def load_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        # Load val accuracy if cached
        val_acc = None
        if os.path.exists("val_accuracy.txt"):
            with open("val_accuracy.txt", "r") as f:
                val_acc = float(f.read().strip())
        return model, vectorizer, val_acc

    # Train
    train_df = load_data(TRAIN_PATH)
    train_df['clean_text'] = train_df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['emotion']

    # LinearSVC wrapped with CalibratedClassifierCV for probability support
    base_model = LinearSVC(max_iter=2000, C=1.0)
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train, y_train)

    # Evaluate on val.txt if available
    val_acc = None
    if os.path.exists(VAL_PATH):
        val_df = load_data(VAL_PATH)
        val_df['clean_text'] = val_df['text'].apply(clean_text)
        X_val = vectorizer.transform(val_df['clean_text'])
        y_val = val_df['emotion']
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        with open("val_accuracy.txt", "w") as f:
            f.write(str(val_acc))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer, val_acc


def predict_emotion(text, model, vectorizer):
    cleaned = clean_text(text)
    if not cleaned:
        return None, None
    vector = vectorizer.transform([cleaned])
    emotion = model.predict(vector)[0]
    proba = model.predict_proba(vector)
    confidence = max(proba[0])
    return emotion, confidence


st.set_page_config(page_title="Emotion Detection", layout="wide")

# ── Emoji rain iframe ────────────────────────────────────────────────────────
EMOJI_HTML = """<!DOCTYPE html>
<html><head>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;
background:linear-gradient(135deg,#0f0c29 0%,#1a0533 20%,#0d1b6e 45%,#0b4d6b 70%,#3b0764 100%)}
.em{position:absolute;line-height:1;user-select:none;will-change:transform;white-space:nowrap}
</style>
</head><body><script>
const ALL=['😊','😂','😢','😡','😍','😲','🥳','😨','🤩','😌','😤','🥰','😭','🤣','😱','🥶','🤗','😏','😬','🤔','😃','😠','😘','💕','🎉','😻','💖','😎','🫶','😆','😁','😄','🤯','😳','😞','😔','🥺','🙀','😿','💢'];
const SP=85,SMIN=45,SMAX=110,FMIN=1.5,FMAX=2.9,SWAP=3000;
const parts=[];let W=innerWidth,H=innerHeight;
const rand=(a,b)=>a+Math.random()*(b-a);
const pick=()=>ALL[0|Math.random()*ALL.length];
function make(x,y){
const el=document.createElement('span');
el.className='em';el.textContent=pick();
const fs=rand(FMIN,FMAX),sp=rand(SMIN,SMAX),wA=rand(5,20),wF=rand(.3,1.0),
wO=Math.random()*Math.PI*2,rs=rand(-30,30),op=rand(.45,.88),px=fs*18;
el.style.cssText=`font-size:${fs}rem;opacity:${op};left:${x}px;top:${y}px;`;
document.body.appendChild(el);
parts.push({el,x,y,sp,wA,wF,wO,rs,op,px,last:performance.now()});
}
function seed(){
const cols=Math.ceil(W/SP)+2,rows=Math.ceil(H/SP)+2;
for(let r=0;r<rows;r++)
for(let c=0;c<cols;c++){
const x=Math.min(Math.max(c*SP+rand(-12,12),10),W-55);
const y=r*SP-rows*SP+rand(-10,10);
make(x,y);
}
}
seed();
let last=performance.now();
function tick(now){
const dt=Math.min((now-last)/1000,.05);last=now;
for(const p of parts){
p.y+=p.sp*dt;
const wx=p.wA*Math.sin(p.wF*now/1000+p.wO),rot=p.rs*now/1000;
const cx=Math.min(Math.max(p.x+wx,10),W-p.px-10);
p.el.style.left=cx+'px';p.el.style.top=p.y+'px';
p.el.style.transform='rotate('+rot.toFixed(1)+'deg)';
if(now-p.last>SWAP+rand(0,2000)){
const el=p.el;el.style.opacity='.06';
setTimeout(()=>{el.textContent=pick();el.style.opacity=p.op;},220);
p.last=now;
}
if(p.y>H+70){p.y=rand(-90,-20);p.x=rand(10,W-65);p.el.textContent=pick();}
}
requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
window.addEventListener('resize',()=>{W=innerWidth;H=innerHeight;});
</script></body></html>"""

components.html(EMOJI_HTML, height=0, scrolling=False)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.main, .main > div,
section.main,
.block-container,
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlock"] > div {
    background: transparent !important;
    background-color: transparent !important;
}
iframe {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    z-index: -1 !important;
    pointer-events: none !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    padding: 4vh 1rem 4vh !important;
    max-width: 620px !important;
    margin: 0 auto !important;
}
.card {
    background: rgba(8, 4, 28, 0.72) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border: 1px solid rgba(167,139,250,0.22) !important;
    border-radius: 28px !important;
    padding: 2.6rem 2.4rem 2.4rem !important;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.03) inset,
        0 24px 80px rgba(0,0,0,0.7),
        0 0 80px rgba(109,40,217,0.10) !important;
}
.badge {
    text-align: center;
    font-size: 3rem;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 18px rgba(232,121,249,0.55));
    display: block;
}
.title {
    font-size: clamp(2rem,5vw,3rem);
    font-weight: 900;
    text-align: center;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg,#f0abfc 0%,#a78bfa 50%,#67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.subtitle {
    text-align: center;
    font-size: 0.93rem;
    color: rgba(196,181,253,0.60) !important;
    margin-bottom: 0.2rem;
    line-height: 1.5;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent,rgba(167,139,250,0.4),rgba(103,232,249,0.4),transparent);
    margin: 1.4rem 0 1.8rem;
}
label[data-testid="stWidgetLabel"] p,
.stTextInput label {
    color: rgba(196,181,253,0.80) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
}
.stTextInput > div > div > input {
    background: rgba(12, 16, 34, 0.94) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.1rem !important;
    caret-color: #a78bfa !important;
    transition: all .2s !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(167,139,250,0.55) !important;
    font-style: italic !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(167,139,250,0.85) !important;
    background: rgba(12, 16, 34, 0.98) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.20),
                0 0 20px rgba(124,58,237,0.10) !important;
    outline: none !important;
}
.stButton > button {
    width: 100% !important;
    margin-top: 1rem !important;
    border-radius: 999px !important;
    background: linear-gradient(135deg,#7c3aed,#2563eb,#0891b2) !important;
    color: #fff !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    padding: 0.82rem 1.5rem !important;
    box-shadow: 0 4px 26px rgba(124,58,237,0.52),
                inset 0 1px 0 rgba(255,255,255,0.08) !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 10px 36px rgba(124,58,237,0.68),
                inset 0 1px 0 rgba(255,255,255,0.10) !important;
}
.stButton > button:active { transform: scale(0.99) !important; }
.result-card {
    margin-top: 1.6rem;
    padding: 1.6rem;
    border-radius: 20px;
    background: linear-gradient(135deg,rgba(109,40,217,0.28),rgba(7,89,133,0.22));
    border: 1px solid rgba(167,139,250,0.32);
    box-shadow: 0 4px 30px rgba(109,40,217,0.18),
                inset 0 1px 0 rgba(255,255,255,0.04);
    text-align: center;
}
.result-label {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    color: rgba(196,181,253,0.55);
    margin-bottom: 0.7rem;
}
.result-icon {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: drop-shadow(0 0 18px rgba(232,121,249,0.65));
}
.result-emotion {
    font-size: 2.1rem;
    font-weight: 900;
    letter-spacing: 0.10em;
    background: linear-gradient(135deg,#f0abfc,#a78bfa,#67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.confidence-bar-wrap {
    margin-top: 1rem;
}
.confidence-label {
    font-size: 0.72rem;
    color: rgba(196,181,253,0.55);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.confidence-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg,#7c3aed,#67e8f9);
    transition: width 0.4s ease;
}
.confidence-pct {
    font-size: 1rem;
    font-weight: 800;
    color: #a78bfa;
    margin-top: 0.3rem;
}
.stAlert {
    background: rgba(234,179,8,0.09) !important;
    border: 1px solid rgba(234,179,8,0.28) !important;
    border-radius: 14px !important;
    margin-top: 1rem !important;
}
.stAlert p { color: #fde68a !important; }
.foot {
    text-align: center;
    font-size: 0.70rem;
    color: rgba(148,130,200,0.38);
    margin-top: 1.8rem;
    letter-spacing: 0.06em;
}

/* ── Sidebar dark styling ── */
[data-testid="stSidebar"] {
    background: rgba(8, 4, 28, 0.92) !important;
    border-right: 1px solid rgba(167,139,250,0.18) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}
[data-testid="stSidebar"] * {
    color: rgba(220, 210, 255, 0.85) !important;
}
[data-testid="stSidebar"] h3 {
    color: #f0abfc !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(167,139,250,0.25) !important;
}
/* Info/warning boxes inside sidebar */
[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] .stAlert {
    background: rgba(20, 10, 50, 0.80) !important;
    border: 1px solid rgba(167,139,250,0.30) !important;
    border-radius: 12px !important;
    color: rgba(196,181,253,0.80) !important;
}
[data-testid="stSidebar"] .stAlert p {
    color: rgba(196,181,253,0.80) !important;
}
/* Metric widget inside sidebar */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(20, 10, 50, 0.75) !important;
    border: 1px solid rgba(167,139,250,0.25) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
try:
    model, vectorizer, val_acc = load_model_and_vectorizer()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Sidebar: model stats ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Info")
    st.markdown("**Algorithm:** LinearSVC (Calibrated)")
    st.markdown("**Vectorizer:** TF-IDF (1–2 ngrams)")
    if val_acc is not None:
        st.metric("Validation Accuracy", f"{val_acc:.2%}")
    else:
        st.info("val.txt not found — accuracy unavailable")
    st.markdown("---")
    st.markdown("**Emotions detected:**")
    if os.path.exists(TRAIN_PATH):
        train_df = load_data(TRAIN_PATH)
        emotions = sorted(train_df['emotion'].unique())
        for e in emotions:
            st.markdown(f"- {e}")

# ── Card header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
  <span class="badge">🧠</span>
  <div class="title">Emotion Detection</div>
  <div class="subtitle">Powered by Machine Learning &nbsp;·&nbsp; Reveal the emotion hidden in any sentence.</div>
  <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Input & predict ───────────────────────────────────────────────────────────
text_input = st.text_input(
    "YOUR SENTENCE",
    placeholder="e.g. I can't believe how amazing today was!",
)

clicked = st.button("✨ Detect Emotion")

if clicked:
    if text_input.strip():
        emotion, confidence = predict_emotion(text_input, model, vectorizer)
        if emotion is None:
            st.warning("⚠️ Input is empty after cleaning. Try a different sentence.")
        else:
            emoji_map = {
                "joy": "😄", "happiness": "😊", "happy": "😊",
                "sadness": "😢", "sad": "😢", "anger": "😡", "angry": "😡",
                "fear": "😨", "surprise": "😲", "love": "😍",
                "disgust": "🤢", "neutral": "😐",
            }
            icon = emoji_map.get(emotion.lower(), "🎯")
            conf_pct = int(confidence * 100)
            st.markdown(f"""
<div class="result-card">
  <div class="result-label">✦ &nbsp; Detected Emotion &nbsp; ✦</div>
  <span class="result-icon">{icon}</span>
  <div class="result-emotion">{emotion.upper()}</div>
  <div class="confidence-bar-wrap">
    <div class="confidence-label">Confidence</div>
    <div class="confidence-bar-bg">
      <div class="confidence-bar-fill" style="width:{conf_pct}%"></div>
    </div>
    <div class="confidence-pct">{conf_pct}%</div>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please type a sentence first.")

st.markdown('<div class="foot">🔒 &nbsp; Processed locally &nbsp;·&nbsp; No data stored</div>',
            unsafe_allow_html=True)