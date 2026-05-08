import os
import pickle
import string
import datetime
import pandas as pd
import numpy as np
import nltk
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

nltk.download('stopwords', quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = "model.pkl"
VEC_PATH     = "vectorizer.pkl"
TRAIN_PATH   = "train.txt"
VAL_PATH     = "val.txt"
TEST_PATH    = "test.txt"
VAL_ACC_PATH = "val_accuracy.txt"
HISTORY_PATH = "prediction_history.csv"

EMOTION_EMOJI = {
    "joy":      "😄", "sadness":  "😢", "anger":    "😡",
    "fear":     "😨", "love":     "😍", "surprise": "😲",
}
EMOTION_COLOR = {
    "joy":      "#f59e0b", "sadness": "#3b82f6", "anger":    "#ef4444",
    "fear":     "#8b5cf6", "love":    "#ec4899", "surprise": "#10b981",
}

# ── Text helpers ───────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path, sep=';', names=['text', 'emotion'])


@st.cache_resource(show_spinner=False)
def load_model_and_vectorizer():
    """Load from disk if cached, otherwise train from scratch."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VEC_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        val_acc = None
        if os.path.exists(VAL_ACC_PATH):
            with open(VAL_ACC_PATH) as f:
                val_acc = float(f.read().strip())
        return model, vectorizer, val_acc

    # Auto-train ────────────────────────────────────────────────────────────────
    train_df = load_data(TRAIN_PATH)
    train_df['clean_text'] = train_df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)
    X_train    = vectorizer.fit_transform(train_df['clean_text'])
    y_train    = train_df['emotion']

    base  = LinearSVC(max_iter=2000, C=1.0)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X_train, y_train)

    val_acc = None
    if os.path.exists(VAL_PATH):
        val_df = load_data(VAL_PATH)
        val_df['clean_text'] = val_df['text'].apply(clean_text)
        X_val     = vectorizer.transform(val_df['clean_text'])
        val_preds = model.predict(X_val)
        val_acc   = accuracy_score(val_df['emotion'], val_preds)
        with open(VAL_ACC_PATH, 'w') as f:
            f.write(str(val_acc))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VEC_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer, val_acc


def predict_emotion(text, model, vectorizer):
    cleaned = clean_text(text)
    if not cleaned:
        return None, None, None
    vec     = vectorizer.transform([cleaned])
    emotion = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return emotion, max(proba), dict(zip(model.classes_, proba))


def save_history(text, emotion, confidence):
    record = pd.DataFrame([{
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text":       text,
        "emotion":    emotion,
        "confidence": round(confidence * 100, 1)
    }])
    if os.path.exists(HISTORY_PATH):
        existing = pd.read_csv(HISTORY_PATH)
        record   = pd.concat([existing, record], ignore_index=True)
    record.to_csv(HISTORY_PATH, index=False)


def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=["timestamp", "text", "emotion", "confidence"])


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Animated emoji rain background ────────────────────────────────────────────
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
const el=document.createElement('span');el.className='em';el.textContent=pick();
const fs=rand(FMIN,FMAX),sp=rand(SMIN,SMAX),wA=rand(5,20),wF=rand(.3,1.0),
wO=Math.random()*Math.PI*2,rs=rand(-30,30),op=rand(.45,.88),px=fs*18;
el.style.cssText=`font-size:${fs}rem;opacity:${op};left:${x}px;top:${y}px;`;
document.body.appendChild(el);
parts.push({el,x,y,sp,wA,wF,wO,rs,op,px,last:performance.now()});}
function seed(){
const cols=Math.ceil(W/SP)+2,rows=Math.ceil(H/SP)+2;
for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
const x=Math.min(Math.max(c*SP+rand(-12,12),10),W-55);
const y=r*SP-rows*SP+rand(-10,10);make(x,y);}}
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
setTimeout(()=>{el.textContent=pick();el.style.opacity=p.op;},220);p.last=now;}
if(p.y>H+70){p.y=rand(-90,-20);p.x=rand(10,W-65);p.el.textContent=pick();}
}requestAnimationFrame(tick);}
requestAnimationFrame(tick);
window.addEventListener('resize',()=>{W=innerWidth;H=innerHeight;});
</script></body></html>"""
components.html(EMOJI_HTML, height=0, scrolling=False)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap');

html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"],
[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],
.main,.main>div,section.main,.block-container,
[data-testid="stVerticalBlock"],[data-testid="stVerticalBlock"]>div{
    background:transparent!important;background-color:transparent!important;
    font-family:'Plus Jakarta Sans',sans-serif!important;}
iframe{position:fixed!important;inset:0!important;width:100vw!important;
    height:100vh!important;border:none!important;z-index:-1!important;pointer-events:none!important;}
#MainMenu,footer,header{visibility:hidden!important;}
.block-container{padding:3vh 1.5rem 4vh!important;max-width:900px!important;margin:0 auto!important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:rgba(8,4,28,.72)!important;border-radius:14px!important;
    padding:4px!important;border:1px solid rgba(167,139,250,.2)!important;gap:4px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:10px!important;
    color:rgba(196,181,253,.6)!important;font-weight:700!important;font-size:.85rem!important;
    letter-spacing:.04em!important;padding:.5rem 1.2rem!important;border:none!important;transition:all .2s!important;}
.stTabs [aria-selected="true"]{background:rgba(124,58,237,.35)!important;color:#f0abfc!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.5rem!important;}

/* Card */
.card{background:rgba(8,4,28,.72)!important;backdrop-filter:blur(30px)!important;
    -webkit-backdrop-filter:blur(30px)!important;border:1px solid rgba(167,139,250,.22)!important;
    border-radius:24px!important;padding:2.2rem 2.2rem 2rem!important;
    box-shadow:0 24px 80px rgba(0,0,0,.5),0 0 60px rgba(109,40,217,.08)!important;margin-bottom:1.2rem!important;}
.badge{text-align:center;font-size:2.8rem;margin-bottom:.3rem;
    filter:drop-shadow(0 0 18px rgba(232,121,249,.55));display:block}
.title{font-size:clamp(1.8rem,4vw,2.6rem);font-weight:900;text-align:center;
    letter-spacing:-.03em;line-height:1.1;margin-bottom:.4rem;
    background:linear-gradient(135deg,#f0abfc 0%,#a78bfa 50%,#67e8f9 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.subtitle{text-align:center;font-size:.88rem;color:rgba(196,181,253,.60)!important;
    margin-bottom:.2rem;line-height:1.5;}
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(167,139,250,.4),
    rgba(103,232,249,.4),transparent);margin:1.2rem 0 1.5rem;}
.section-label{font-size:.70rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;
    color:rgba(196,181,253,.50);margin-bottom:.7rem;}

/* Inputs */
label[data-testid="stWidgetLabel"] p,.stTextInput label{color:rgba(196,181,253,.80)!important;
    font-size:.72rem!important;font-weight:700!important;letter-spacing:.13em!important;text-transform:uppercase!important;}
.stTextInput>div>div>input,.stTextArea textarea{background:rgba(12,16,34,.94)!important;
    color:#f8fafc!important;border:1px solid rgba(167,139,250,.35)!important;border-radius:14px!important;
    font-size:1rem!important;padding:.75rem 1.1rem!important;caret-color:#a78bfa!important;transition:all .2s!important;}
.stTextInput>div>div>input::placeholder,.stTextArea textarea::placeholder{
    color:rgba(167,139,250,.55)!important;font-style:italic!important;}
.stTextInput>div>div>input:focus,.stTextArea textarea:focus{
    border-color:rgba(167,139,250,.85)!important;background:rgba(12,16,34,.98)!important;
    box-shadow:0 0 0 3px rgba(124,58,237,.20),0 0 20px rgba(124,58,237,.10)!important;outline:none!important;}
.stFileUploader{background:rgba(12,16,34,.94)!important;
    border:1px dashed rgba(167,139,250,.4)!important;border-radius:14px!important;}

/* Buttons */
.stButton>button{width:100%!important;margin-top:.8rem!important;border-radius:999px!important;
    background:linear-gradient(135deg,#7c3aed,#2563eb,#0891b2)!important;color:#fff!important;
    font-weight:800!important;font-size:1rem!important;letter-spacing:.05em!important;border:none!important;
    padding:.75rem 1.5rem!important;box-shadow:0 4px 26px rgba(124,58,237,.52),
    inset 0 1px 0 rgba(255,255,255,.08)!important;transition:transform .15s,box-shadow .15s!important;}
.stButton>button:hover{transform:translateY(-3px) scale(1.01)!important;
    box-shadow:0 10px 36px rgba(124,58,237,.68),inset 0 1px 0 rgba(255,255,255,.10)!important;}
.stButton>button:active{transform:scale(.99)!important;}
.stDownloadButton>button{background:rgba(12,16,34,.94)!important;
    border:1px solid rgba(167,139,250,.35)!important;border-radius:999px!important;
    color:#a78bfa!important;font-weight:700!important;width:auto!important;margin-top:.5rem!important;}

/* Result card */
.result-card{margin-top:1.4rem;padding:1.6rem;border-radius:20px;
    background:linear-gradient(135deg,rgba(109,40,217,.28),rgba(7,89,133,.22));
    border:1px solid rgba(167,139,250,.32);
    box-shadow:0 4px 30px rgba(109,40,217,.18),inset 0 1px 0 rgba(255,255,255,.04);text-align:center;}
.result-label{font-size:.68rem;font-weight:700;letter-spacing:.20em;text-transform:uppercase;
    color:rgba(196,181,253,.55);margin-bottom:.6rem}
.result-icon{font-size:3rem;display:block;margin-bottom:.3rem;
    filter:drop-shadow(0 0 18px rgba(232,121,249,.65))}
.result-emotion{font-size:2rem;font-weight:900;letter-spacing:.10em;
    background:linear-gradient(135deg,#f0abfc,#a78bfa,#67e8f9);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.confidence-bar-wrap{margin-top:.8rem}
.confidence-label{font-size:.68rem;color:rgba(196,181,253,.55);letter-spacing:.12em;
    text-transform:uppercase;margin-bottom:.4rem}
.confidence-bar-bg{background:rgba(255,255,255,.08);border-radius:999px;height:8px;overflow:hidden}
.confidence-bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#7c3aed,#67e8f9);
    transition:width 0.6s ease}
.confidence-pct{font-size:1rem;font-weight:800;color:#a78bfa;margin-top:.3rem}

/* Probability rows */
.prob-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.prob-label{width:80px;font-size:.78rem;color:#c4b5fd;text-align:right;font-weight:600;}
.prob-bar-bg{flex:1;background:rgba(255,255,255,.07);border-radius:999px;height:6px;overflow:hidden;}
.prob-bar-fill{height:100%;border-radius:999px;transition:width 0.5s ease;}
.prob-pct{width:40px;font-size:.75rem;color:#a78bfa;font-weight:700;text-align:right;}

/* Sidebar */
[data-testid="stSidebar"]{background:rgba(8,4,28,.88)!important;
    border-right:1px solid rgba(167,139,250,.15)!important;}
[data-testid="stSidebar"] *{color:#c4b5fd!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{color:#f0abfc!important;}
[data-testid="stMetricValue"]{color:#f0abfc!important;font-weight:800!important;}
[data-testid="stMetricLabel"]{color:rgba(196,181,253,.6)!important;font-size:.75rem!important;}

/* Misc */
.foot{text-align:center;font-size:.72rem;color:rgba(196,181,253,.30);
    padding:1.5rem 0 0;letter-spacing:.10em;}
[data-testid="stDataFrame"]{border-radius:12px!important;}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading model…"):
    model, vectorizer, val_acc = load_model_and_vectorizer()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Emotion Detection")
    st.markdown("---")

    st.markdown("### 📊 Model Info")
    st.markdown("""
<div style="font-size:.82rem;line-height:1.9;color:#c4b5fd">
<b>Algorithm</b>&nbsp;&nbsp;LinearSVC (Calibrated)<br>
<b>Vectorizer</b>&nbsp;&nbsp;TF-IDF · 1–2 n-grams<br>
<b>Features</b>&nbsp;&nbsp;50,000 max<br>
<b>Train samples</b>&nbsp;&nbsp;16,000<br>
<b>Val samples</b>&nbsp;&nbsp;2,000<br>
<b>Test samples</b>&nbsp;&nbsp;2,000
</div>""", unsafe_allow_html=True)

    if val_acc is not None:
        st.metric("Validation Accuracy", f"{val_acc:.2%}")
    st.metric("Test Accuracy", "~89.90 %")

    st.markdown("---")
    st.markdown("### 🏷️ Detectable Emotions")
    for em, emoji in EMOTION_EMOJI.items():
        color = EMOTION_COLOR.get(em, "#a78bfa")
        st.markdown(
            f"<span style='color:{color};font-weight:700'>{emoji} {em.capitalize()}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🔬 Test-Set Metrics")
    if os.path.exists(TEST_PATH):
        try:
            @st.cache_data(show_spinner=False)
            def get_test_metrics():
                test_df = load_data(TEST_PATH)
                test_df['clean_text'] = test_df['text'].apply(clean_text)
                X_test  = vectorizer.transform(test_df['clean_text'])
                preds   = model.predict(X_test)
                acc     = accuracy_score(test_df['emotion'], preds)
                f1      = f1_score(test_df['emotion'], preds, average='weighted')
                report  = classification_report(test_df['emotion'], preds, output_dict=True)
                cm      = confusion_matrix(test_df['emotion'], preds, labels=sorted(test_df['emotion'].unique()))
                return acc, f1, report, cm, sorted(test_df['emotion'].unique())

            t_acc, t_f1, t_report, t_cm, t_labels = get_test_metrics()
            col1, col2 = st.columns(2)
            col1.metric("Test Accuracy", f"{t_acc:.2%}")
            col2.metric("Weighted F1",   f"{t_f1:.3f}")

            st.markdown("**Per-class F1:**")
            for em in t_labels:
                f1_val = t_report[em]['f1-score']
                emoji  = EMOTION_EMOJI.get(em, "🎯")
                color  = EMOTION_COLOR.get(em, "#a78bfa")
                bar_w  = int(f1_val * 100)
                st.markdown(f"""
<div class="prob-row">
  <div class="prob-label">{emoji} {em}</div>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{bar_w}%;background:{color}"></div>
  </div>
  <div class="prob-pct">{f1_val:.2f}</div>
</div>""", unsafe_allow_html=True)

            with st.expander("🔲 Confusion Matrix"):
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                fig_cm.patch.set_facecolor('#1a1040')
                ax_cm.set_facecolor('#1a1040')
                sns.heatmap(
                    t_cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=t_labels, yticklabels=t_labels,
                    ax=ax_cm, linewidths=0.3,
                    annot_kws={"size": 8, "color": "#f0abfc"}
                )
                ax_cm.set_title('Confusion Matrix', color='#f0abfc', fontsize=9, pad=8)
                ax_cm.tick_params(colors='#c4b5fd', labelsize=7)
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close()
        except Exception:
            st.info("Run app once so test metrics generate.")

    st.markdown("---")
    st.markdown(
        '<div class="foot">🔒 Processed locally · No data stored externally<br>'
        '<a href="https://github.com/kananiisha/Emotion-Detection-from-Text" '
        'style="color:rgba(167,139,250,.5);">GitHub</a></div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="card">
  <span class="badge">🧠</span>
  <div class="title">Emotion Detection</div>
  <div class="subtitle">Powered by Machine Learning &nbsp;·&nbsp; Reveal the emotion hidden in any sentence.</div>
  <div class="divider"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict", "📦 Batch Mode", "📈 History", "🔬 Model Comparison"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single prediction
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    text_input = st.text_input(
        "YOUR SENTENCE",
        placeholder="e.g. I can't believe how amazing today was!",
        key="single_input"
    )
    clicked = st.button("✨ Detect Emotion", key="detect_btn")

    if clicked:
        if text_input.strip():
            emotion, confidence, prob_dict = predict_emotion(text_input, model, vectorizer)
            if emotion is None:
                st.warning("⚠️ Input is empty after cleaning. Try a different sentence.")
            else:
                icon     = EMOTION_EMOJI.get(emotion.lower(), "🎯")
                conf_pct = int(confidence * 100)
                em_color = EMOTION_COLOR.get(emotion.lower(), "#a78bfa")

                st.markdown(f"""
<div class="result-card">
  <div class="result-label">✦ &nbsp; Detected Emotion &nbsp; ✦</div>
  <span class="result-icon">{icon}</span>
  <div class="result-emotion">{emotion.upper()}</div>
  <div class="confidence-bar-wrap">
    <div class="confidence-label">Confidence</div>
    <div class="confidence-bar-bg">
      <div class="confidence-bar-fill"
           style="width:{conf_pct}%;background:linear-gradient(90deg,{em_color},{em_color}88)">
      </div>
    </div>
    <div class="confidence-pct">{conf_pct}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

                st.markdown(
                    '<div class="section-label" style="margin-top:1.2rem">'
                    'Probability breakdown — all emotions</div>',
                    unsafe_allow_html=True
                )
                for em, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                    emoji  = EMOTION_EMOJI.get(em, "🎯")
                    color  = EMOTION_COLOR.get(em, "#a78bfa")
                    bar_w  = int(prob * 100)
                    bold   = "font-weight:900" if em == emotion else ""
                    st.markdown(f"""
<div class="prob-row">
  <div class="prob-label" style="{bold}">{emoji} {em}</div>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill"
         style="width:{bar_w}%;background:{color}{'cc' if em != emotion else ''}">
    </div>
  </div>
  <div class="prob-pct">{prob*100:.1f}%</div>
</div>""", unsafe_allow_html=True)

                save_history(text_input, emotion, confidence)
        else:
            st.warning("⚠️ Please type a sentence first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch mode
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="section-label">Upload a CSV with a column named "text" — get predictions back</div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            if 'text' not in batch_df.columns:
                st.error("❌ CSV must contain a column named **text**.")
            else:
                st.markdown(f"**{len(batch_df):,} rows loaded.** Preview:")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("🚀 Run Batch Predictions", key="batch_btn"):
                    with st.spinner("Analysing emotions…"):
                        results = []
                        for txt in batch_df['text']:
                            em, conf, pd_ = predict_emotion(str(txt), model, vectorizer)
                            results.append({
                                "text":       txt,
                                "emotion":    em if em else "unknown",
                                "confidence": round(conf * 100, 1) if conf else 0.0,
                                **(
                                    {f"prob_{k}": round(v * 100, 1) for k, v in pd_.items()}
                                    if pd_ else {}
                                )
                            })
                        result_df = pd.DataFrame(results)

                    st.success(f"✅ Done! Predicted {len(result_df):,} rows.")
                    st.dataframe(result_df, use_container_width=True)

                    dist = result_df['emotion'].value_counts()
                    st.markdown("**Emotion distribution in your batch:**")
                    fig_b, ax_b = plt.subplots(figsize=(8, 3))
                    fig_b.patch.set_facecolor('#0f0c29')
                    ax_b.set_facecolor('#1a1040')
                    colors_b = [EMOTION_COLOR.get(e, "#a78bfa") for e in dist.index]
                    ax_b.barh(dist.index, dist.values, color=colors_b, edgecolor='#0f0c29', linewidth=0.6)
                    ax_b.set_xlabel('Count', color='#c4b5fd')
                    ax_b.tick_params(colors='#c4b5fd')
                    ax_b.set_title('Batch Emotion Distribution', color='#f0abfc', fontsize=10, pad=8)
                    for spine in ax_b.spines.values():
                        spine.set_edgecolor('#2d1b69')
                    plt.tight_layout()
                    st.pyplot(fig_b, use_container_width=True)
                    plt.close()

                    csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv_bytes,
                        file_name="emotion_predictions.csv",
                        mime="text/csv",
                        key="download_results"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown("""
<div style="background:rgba(12,16,34,.85);border:1px dashed rgba(167,139,250,.3);
border-radius:14px;padding:1.2rem 1.5rem;color:#c4b5fd;font-size:.85rem;">
<b>Expected CSV format:</b><br><br>
<code style="color:#f0abfc">text</code><br>
<code>I feel so happy today!</code><br>
<code>This is incredibly frustrating.</code><br>
<code>I'm scared of what might happen next.</code>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Prediction history
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    history_df = load_history()

    if history_df.empty:
        st.info("No predictions yet. Run some from the **Predict** tab!")
    else:
        st.markdown(
            f'<div class="section-label">{len(history_df)} predictions recorded this session</div>',
            unsafe_allow_html=True
        )
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Predictions",  len(history_df))
        col_b.metric("Most Common Emotion", history_df['emotion'].mode()[0].capitalize())
        col_c.metric("Avg Confidence",      f"{history_df['confidence'].mean():.1f}%")

        dist_h   = history_df['emotion'].value_counts()
        fig_h, axes_h = plt.subplots(1, 2, figsize=(10, 3))
        fig_h.patch.set_facecolor('#0f0c29')
        for ax in axes_h:
            ax.set_facecolor('#1a1040')

        colors_h = [EMOTION_COLOR.get(e, "#a78bfa") for e in dist_h.index]
        axes_h[0].bar(dist_h.index, dist_h.values, color=colors_h, edgecolor='#0f0c29', linewidth=0.6)
        axes_h[0].set_title('Emotion Distribution', color='#f0abfc', fontsize=10, pad=8)
        axes_h[0].tick_params(colors='#c4b5fd', labelsize=8)
        axes_h[0].set_xlabel('Emotion', color='#c4b5fd', fontsize=8)

        axes_h[1].plot(range(len(history_df)), history_df['confidence'],
                       color='#a78bfa', linewidth=1.5, marker='o', markersize=3)
        axes_h[1].set_title('Confidence Over Time', color='#f0abfc', fontsize=10, pad=8)
        axes_h[1].tick_params(colors='#c4b5fd', labelsize=8)
        axes_h[1].set_xlabel('Prediction #', color='#c4b5fd', fontsize=8)
        axes_h[1].set_ylabel('Confidence %', color='#c4b5fd', fontsize=8)
        axes_h[1].set_ylim(0, 105)
        for spine in axes_h[1].spines.values():
            spine.set_edgecolor('#2d1b69')
        for spine in axes_h[0].spines.values():
            spine.set_edgecolor('#2d1b69')

        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True)
        plt.close()

        st.dataframe(
            history_df[::-1].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        hist_csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download History CSV",
            data=hist_csv,
            file_name="emotion_history.csv",
            mime="text/csv",
            key="download_history"
        )

        if st.button("🗑️ Clear History", key="clear_history"):
            if os.path.exists(HISTORY_PATH):
                os.remove(HISTORY_PATH)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Model comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        '<div class="section-label">Live comparison: LinearSVC vs Logistic Regression vs Multinomial NB</div>',
        unsafe_allow_html=True
    )

    @st.cache_data(show_spinner=True)
    def run_model_comparison():
        train_df = load_data(TRAIN_PATH)
        val_df   = load_data(VAL_PATH)
        train_df['clean'] = train_df['text'].apply(clean_text)
        val_df['clean']   = val_df['text'].apply(clean_text)

        vec  = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)
        X_tr = vec.fit_transform(train_df['clean'])
        X_v  = vec.transform(val_df['clean'])
        y_tr, y_v = train_df['emotion'], val_df['emotion']

        candidates = {
            "LinearSVC (Calibrated)": CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3),
            "Logistic Regression":    LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
            "Multinomial NB":         MultinomialNB(alpha=0.1),
        }
        rows = []
        for name, clf in candidates.items():
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_v)
            rows.append({
                "Model":        name,
                "Val Accuracy": round(accuracy_score(y_v, preds) * 100, 2),
                "Weighted F1":  round(f1_score(y_v, preds, average='weighted'), 4),
                "Macro F1":     round(f1_score(y_v, preds, average='macro'), 4),
            })
        return pd.DataFrame(rows).sort_values("Val Accuracy", ascending=False)

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        st.warning("train.txt and val.txt are required for model comparison.")
    else:
        if st.button("▶️ Run Comparison (~30 sec)", key="compare_btn"):
            with st.spinner("Training 3 models…"):
                comp_df = run_model_comparison()

            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            fig_c, ax_c = plt.subplots(figsize=(9, 4))
            fig_c.patch.set_facecolor('#0f0c29')
            ax_c.set_facecolor('#1a1040')
            x = np.arange(len(comp_df))
            w = 0.35
            ax_c.bar(x - w/2, comp_df['Val Accuracy'],      w, label='Val Accuracy (%)',
                     color='#7c3aed', alpha=0.9, edgecolor='#0f0c29')
            ax_c.bar(x + w/2, comp_df['Weighted F1'] * 100, w, label='Weighted F1 × 100',
                     color='#0891b2', alpha=0.9, edgecolor='#0f0c29')
            ax_c.set_xticks(x)
            ax_c.set_xticklabels(comp_df['Model'], rotation=10, color='#c4b5fd', fontsize=9)
            ax_c.set_ylim(70, 100)
            ax_c.tick_params(colors='#c4b5fd')
            ax_c.set_title('Model Comparison — Validation Set', color='#f0abfc', fontsize=12, pad=10)
            ax_c.legend(facecolor='#1a1040', edgecolor='#7c3aed', labelcolor='#c4b5fd', fontsize=8)
            for spine in ax_c.spines.values():
                spine.set_edgecolor('#2d1b69')
            for bar in ax_c.patches:
                ax_c.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{bar.get_height():.1f}', ha='center', va='bottom',
                    fontsize=8, color='#c4b5fd'
                )
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close()

            best = comp_df.iloc[0]['Model']
            st.success(
                f"🏆 Best model: **{best}** with {comp_df.iloc[0]['Val Accuracy']}% validation accuracy"
            )
        else:
            st.markdown("""
<div style="background:rgba(12,16,34,.85);border:1px solid rgba(167,139,250,.25);
border-radius:14px;padding:1.4rem;color:#c4b5fd;font-size:.88rem;">
<b style="color:#f0abfc">Models compared:</b><br><br>
• <b>LinearSVC (Calibrated)</b> — hinge-loss margin maximisation, wrapped for probability output<br>
• <b>Logistic Regression</b> — multinomial softmax, native probabilities, interpretable<br>
• <b>Multinomial Naive Bayes</b> — fast baseline designed for TF-IDF / count features
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="foot">🔒 &nbsp; Processed locally &nbsp;·&nbsp; No data stored externally &nbsp;·&nbsp; '
    '<a href="https://github.com/kananiisha/Emotion-Detection-from-Text" '
    'style="color:rgba(167,139,250,.5);">GitHub</a></div>',
    unsafe_allow_html=True
)