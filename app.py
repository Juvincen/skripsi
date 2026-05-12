"""
Dark Triad Personality Predictor
─────────────────────────────────
Flow (same as notebook):
  1. Load & preprocess SD3_data.csv  (dropna country)
  2. Feature Engineering (score_M/N/P, label via median split)
  3. Cross-trait Feature Selection   (no data leakage)
  4. StandardScaler inside Pipeline
  5. Hybrid Ensemble  RF + XGB  weighted soft-voting
  6. Evaluate (Acc, Precision, Recall, F1, ROC-AUC)
  7. SHAP  - global explainability (summary beeswarm + bar)
  8. User fills 27 SD3 items -> same pipeline predicts + SHAP
"""

import warnings, io
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

M_ITEMS = [f"M{i}" for i in range(1, 10)]
N_ITEMS = [f"N{i}" for i in range(1, 10)]
P_ITEMS = [f"P{i}" for i in range(1, 10)]

QUESTIONS = {
    "M1": "It's not wise to tell your secrets.",
    "M2": "I like to use clever manipulation to get my way.",
    "M3": "Whatever it takes, you must get the important people on your side.",
    "M4": "Avoid direct conflict with others because they may be useful in the future.",
    "M5": "It's wise to keep track of information that you can use against people later.",
    "M6": "You should wait for the right time to get back at people.",
    "M7": "There are things you should hide from other people because they don't need to know.",
    "M8": "Make sure your plans benefit you, not others.",
    "M9": "Most people can be manipulated.",
    "N1": "People see me as a natural leader.",
    "N2": "I hate being the center of attention.",
    "N3": "Many group activities tend to be dull without me.",
    "N4": "I know that I am special because everyone keeps telling me so.",
    "N5": "I like to get acquainted with important people.",
    "N6": "I feel embarrassed if someone compliments me.",
    "N7": "I have been compared to famous people.",
    "N8": "I am an average person.",
    "N9": "I insist on getting the respect I deserve.",
    "P1": "I like to get revenge on authorities.",
    "P2": "I avoid dangerous situations.",
    "P3": "Payback needs to be quick and nasty.",
    "P4": "People often say I'm out of control.",
    "P5": "It's true that I can be mean to others.",
    "P6": "People who mess with me always regret it.",
    "P7": "I have never gotten into trouble with the law.",
    "P8": "I enjoy having sex with people I hardly know.",
    "P9": "I'll say anything to get what I want.",
}

TRAIT_META = {
    "Machiavellianism": {
        "color": "#60A5FA", "bg": "#0e1f35", "icon": "♟️",
        "desc": "Manipulatif, strategis, dan pragmatis demi kepentingan diri.",
        "features": N_ITEMS + P_ITEMS,
        "score_col": "score_M",
    },
    "Narcissism": {
        "color": "#FBBF24", "bg": "#2a1a00", "icon": "👑",
        "desc": "Membutuhkan pengakuan, merasa superior, dan kurang berempati.",
        "features": M_ITEMS + P_ITEMS,
        "score_col": "score_N",
    },
    "Psychopathy": {
        "color": "#34D399", "bg": "#051f12", "icon": "🎭",
        "desc": "Impulsif, kurang penyesalan, dan berani mengambil risiko.",
        "features": M_ITEMS + N_ITEMS,
        "score_col": "score_P",
    },
}

XGB_PARAMS = dict(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, eval_metric="logloss",
    use_label_encoder=False,
)
RF_PARAMS = dict(
    n_estimators=100, max_depth=10, min_samples_split=10,
    min_samples_leaf=4, random_state=RANDOM_STATE, n_jobs=-1,
)

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dark Triad Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.dt-hero {
    background: linear-gradient(135deg, #07090f 0%, #0c1220 60%, #07090f 100%);
    border: 1px solid #1a2540;
    border-radius: 18px;
    padding: 2.2rem 2.8rem 2rem;
    margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.dt-hero::before {
    content: ''; position: absolute; inset: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(96,165,250,.06) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 50%, rgba(251,191,36,.04) 0%, transparent 55%);
    pointer-events: none;
}
.dt-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800; letter-spacing: -.03em; line-height: 1.1;
    background: linear-gradient(90deg, #60A5FA 0%, #818CF8 45%, #FBBF24 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 .4rem;
}
.dt-sub { color: #5a6a8a; font-size: .9rem; font-weight: 300; letter-spacing: .03em; margin: 0; }

.section-hd {
    font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700;
    color: #dde4f0; margin: 1.6rem 0 .8rem;
    border-bottom: 1px solid #1a2540; padding-bottom: .4rem;
    letter-spacing: -.01em;
}
.chip {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: .68rem; font-weight: 600; letter-spacing: .09em;
    background: rgba(96,165,250,.12); color: #60A5FA;
    border: 1px solid rgba(96,165,250,.25); margin-right: 7px; vertical-align: middle;
}

/* Question blocks */
.q-wrap { background: #0b1525; border: 1px solid #182035;
          border-radius: 9px; padding: .85rem 1rem; margin-bottom: .5rem; }
.q-id   { font-size: .68rem; font-weight: 700; color: #60A5FA;
           font-family: monospace; margin-bottom: .2rem; }
.q-txt  { color: #b0bcd4; font-size: .85rem; line-height: 1.45; }

/* Trait result cards */
.tr-card {
    border-radius: 14px; padding: 1.3rem 1.5rem;
    border: 1px solid transparent; margin-bottom: .4rem;
}
.tr-icon  { font-size: 1.7rem; margin-bottom: .3rem; }
.tr-name  { font-family: 'Syne', sans-serif; font-size: .95rem; font-weight: 700; }
.tr-pct   { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin: .25rem 0; }
.badge-H  { display:inline-block; padding:2px 11px; border-radius:20px;
            font-size:.72rem; font-weight:600; letter-spacing:.08em;
            background:rgba(239,68,68,.14); color:#f87171; border:1px solid rgba(239,68,68,.28); }
.badge-L  { display:inline-block; padding:2px 11px; border-radius:20px;
            font-size:.72rem; font-weight:600; letter-spacing:.08em;
            background:rgba(52,211,153,.12); color:#34d399; border:1px solid rgba(52,211,153,.25); }
.prog-bg  { background:#182035; border-radius:5px; height:5px; margin-top:.7rem; overflow:hidden; }
.prog-fill{ height:100%; border-radius:5px; }

/* Dominant banner */
.dom-banner {
    border-radius: 16px; padding: 1.6rem 2rem; text-align: center;
    margin: 1.2rem 0; border: 1px solid rgba(255,255,255,.07);
}
.dom-lbl  { font-size:.7rem; color:#7a8aaa; letter-spacing:.1em;
            text-transform:uppercase; font-weight:600; }
.dom-name { font-family:'Syne',sans-serif; font-size:2rem;
            font-weight:800; margin:.3rem 0; }
.dom-desc { color:#7a8aaa; font-size:.88rem; }

/* Metric boxes */
.m-row    { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:.8rem; }
.m-box    { flex:1; min-width:80px; background:#0b1525;
            border:1px solid #182035; border-radius:9px; padding:.75rem; text-align:center; }
.m-lbl    { font-size:.6rem; color:#5a6a8a; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }
.m-val    { font-family:'Syne',sans-serif; font-size:1.2rem;
            font-weight:700; color:#dde4f0; margin-top:2px; }

div[data-testid="stSidebar"] > div:first-child { background: #07090f; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PIPELINE / SHAP HELPERS  (identical to notebook logic)
# ═══════════════════════════════════════════════════════════

def make_ensemble(cv_scores=None):
    """Hybrid Ensemble: StandardScaler + Weighted Soft-Voting RF+XGB."""
    weights = [1, 1]
    if cv_scores:
        tot = sum(cv_scores.values())
        weights = [cv_scores["xgb"] / tot, cv_scores["rf"] / tot]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(**XGB_PARAMS)),
                ("rf",  RandomForestClassifier(**RF_PARAMS)),
            ],
            voting="soft", weights=weights, n_jobs=-1,
        )),
    ])


def fit_and_eval(pipe, X_tr, X_te, y_tr, y_te):
    """Fit pipeline + return metrics dict (Bug-1 fix: use params, not outer vars)."""
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    tr_pred = pipe.predict(X_tr)
    tr_acc  = accuracy_score(y_tr, tr_pred)
    te_acc  = accuracy_score(y_te, y_pred)
    return {
        "Train Accuracy": tr_acc,
        "Test Accuracy":  te_acc,
        "Accuracy Gap":   abs(tr_acc - te_acc),
        "Precision":      precision_score(y_te, y_pred, zero_division=0),
        "Recall":         recall_score(y_te, y_pred, zero_division=0),
        "F1-Score":       f1_score(y_te, y_pred, zero_division=0),
        "ROC-AUC":        roc_auc_score(y_te, y_proba),
        "Predictions":    y_pred,
        "Probabilities":  y_proba,
        "Pipeline":       pipe,   # Bug-1: store from param
        "X_train":        X_tr,
        "X_test":         X_te,
        "y_test":         y_te,
    }


# SHAP Bug-3 / 4 / 5 fixes
def scale_for_shap(pipe, X):
    """Bug-3: use fitted scaler from pipeline, never refit."""
    return pd.DataFrame(pipe.named_steps["scaler"].transform(X), columns=X.columns)

def get_rf(pipe):
    """Bug-4: access RF through pipeline.named_steps['model']."""
    return pipe.named_steps["model"].named_estimators_["rf"]

def shap_class1(sv):
    """Bug-5: normalize RF list / XGB ndarray to class-1 2-D array."""
    if isinstance(sv, list):   return sv[1]
    if sv.ndim == 3:           return sv[:, :, 1]
    return sv


# ═══════════════════════════════════════════════════════════
# TRAIN (cached per uploaded CSV)
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def run_pipeline(csv_bytes: bytes):
    # ── 1. Load + preprocess ──────────────────────────────
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if "country" in df.columns:
        df = df.dropna(subset=["country"])

    # ── 2. Feature engineering: composite scores ──────────
    df["score_M"] = df[M_ITEMS].mean(axis=1)
    df["score_N"] = df[N_ITEMS].mean(axis=1)
    df["score_P"] = df[P_ITEMS].mean(axis=1)

    # ── 3. Binary labels via median split ─────────────────
    med = {k: df[f"score_{k}"].median() for k in ("M", "N", "P")}
    for k in ("M", "N", "P"):
        df[f"label_{k}"] = (df[f"score_{k}"] >= med[k]).astype(int)

    # ── 4. Cross-trait feature selection (no leakage) ─────
    #   Predict M → features: N + P
    #   Predict N → features: M + P
    #   Predict P → features: M + N
    feature_map = {
        "Machiavellianism": (N_ITEMS + P_ITEMS, "label_M"),
        "Narcissism":       (M_ITEMS + P_ITEMS, "label_N"),
        "Psychopathy":      (M_ITEMS + N_ITEMS, "label_P"),
    }

    # ── 5. 80/20 stratified split ─────────────────────────
    results = {}
    for trait, (feats, lbl) in feature_map.items():
        X = df[feats]; y = df[lbl]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )
        # Bug-2: clone per trait — fresh, unfitted pipeline each iteration
        pipe = clone(make_ensemble())
        results[trait] = fit_and_eval(pipe, X_tr, X_te, y_tr, y_te)

    return df, results, med


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Dark Triad")
    st.markdown("---")
    uploaded = st.file_uploader(
        "📂 Upload **SD3_data.csv**", type=["csv"],
        help="Dataset harus mengandung kolom M1–M9, N1–N9, P1–P9, country"
    )
    st.markdown("---")
    nav = st.radio("Navigasi", [
        "📋 Kuesioner SD3",
        "📊 Performa Model",
        "🔍 SHAP Global",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Hybrid Ensemble · RF + XGBoost\nWeighted Soft Voting · SHAP Global")

# ═══════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="dt-hero">
  <div class="dt-title">Dark Triad Predictor</div>
  <p class="dt-sub">Hybrid Ensemble &nbsp;·&nbsp; Random Forest + XGBoost &nbsp;·&nbsp;
     Weighted Soft Voting &nbsp;·&nbsp; Global SHAP Explainability</p>
</div>
""", unsafe_allow_html=True)

# ── Gate ──────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload **SD3_data.csv** di sidebar untuk melatih model.")
    c1, c2, c3 = st.columns(3)
    for col, (trait, m) in zip([c1, c2, c3], TRAIT_META.items()):
        with col:
            st.markdown(f"""
            <div class="tr-card" style="background:{m['bg']};border-color:{m['color']}33;">
              <div class="tr-icon">{m['icon']}</div>
              <div class="tr-name" style="color:{m['color']};">{trait}</div>
              <p style="color:#7a8aaa;font-size:.82rem;margin-top:.4rem;">{m['desc']}</p>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── Run pipeline ──────────────────────────────────────────
with st.spinner("⚙️ Melatih Hybrid Ensemble (RF + XGBoost)…"):
    df_data, model_res, medians = run_pipeline(uploaded.read())


# ═══════════════════════════════════════════════════════════
# PAGE ①  KUESIONER SD3
# ═══════════════════════════════════════════════════════════
if nav == "📋 Kuesioner SD3":

    st.markdown(
        '<div class="section-hd"><span class="chip">STEP 1</span>Isi Kuesioner SD3</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Jawab **27 pernyataan** berikut secara jujur. "
        "Skala: **1** = Sangat Tidak Setuju &nbsp;·&nbsp; **5** = Sangat Setuju"
    )

    with st.form("sd3_form"):
        resp = {}
        for sec, items in [
            ("♟️ Bagian M — Machiavellianism", M_ITEMS),
            ("👑 Bagian N — Narcissism",        N_ITEMS),
            ("🎭 Bagian P — Psychopathy",       P_ITEMS),
        ]:
            st.markdown(f"#### {sec}")
            cols = st.columns(3)
            for i, key in enumerate(items):
                with cols[i % 3]:
                    st.markdown(
                        f'<div class="q-wrap">'
                        f'<div class="q-id">{key}</div>'
                        f'<div class="q-txt">{QUESTIONS[key]}</div>'
                        f'</div>', unsafe_allow_html=True
                    )
                    resp[key] = st.slider(key, 1, 5, 3, label_visibility="collapsed")
            st.write("")

        go = st.form_submit_button(
            "🔮  Prediksi Kepribadian Saya",
            use_container_width=True, type="primary"
        )

    # ── Results ───────────────────────────────────────────
    if go:
        m_arr = np.array([resp[k] for k in M_ITEMS], dtype=float)
        n_arr = np.array([resp[k] for k in N_ITEMS], dtype=float)
        p_arr = np.array([resp[k] for k in P_ITEMS], dtype=float)

        # Feature engineering (same as training)
        score_M, score_N, score_P = m_arr.mean(), n_arr.mean(), p_arr.mean()

        # Cross-trait inputs (no leakage — identical to notebook)
        X_inputs = {
            "Machiavellianism": pd.DataFrame(
                [np.concatenate([n_arr, p_arr])], columns=N_ITEMS + P_ITEMS),
            "Narcissism": pd.DataFrame(
                [np.concatenate([m_arr, p_arr])], columns=M_ITEMS + P_ITEMS),
            "Psychopathy": pd.DataFrame(
                [np.concatenate([m_arr, n_arr])], columns=M_ITEMS + N_ITEMS),
        }

        # Predict
        preds = {}
        for trait, X_in in X_inputs.items():
            pipe  = model_res[trait]["Pipeline"]
            pred  = int(pipe.predict(X_in)[0])
            proba = float(pipe.predict_proba(X_in)[0][1])
            preds[trait] = {"pred": pred, "proba": proba,
                            "label": "HIGH" if pred == 1 else "LOW"}

        # Dominant trait (highest probability)
        dom = max(preds, key=lambda t: preds[t]["proba"])
        dm  = TRAIT_META[dom]
        st.markdown(f"""
        <div class="dom-banner" style="background:{dm['bg']};border-color:{dm['color']}44;">
          <div class="dom-lbl">Dominant Dark Triad Trait</div>
          <div class="dom-name" style="color:{dm['color']};">{dm['icon']} {dom}</div>
          <div class="dom-desc">{dm['desc']}</div>
        </div>""", unsafe_allow_html=True)

        # Per-trait cards
        st.markdown(
            '<div class="section-hd"><span class="chip">RESULT</span>Prediksi per Trait</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        for col, (trait, m) in zip([c1, c2, c3], TRAIT_META.items()):
            p  = preds[trait]
            bk = "badge-H" if p["pred"] == 1 else "badge-L"
            with col:
                st.markdown(f"""
                <div class="tr-card" style="background:{m['bg']};border-color:{m['color']}44;">
                  <div class="tr-icon">{m['icon']}</div>
                  <div class="tr-name" style="color:{m['color']};">{trait}</div>
                  <div class="tr-pct" style="color:{m['color']};">{p['proba']*100:.1f}%</div>
                  <span class="{bk}">{p['label']}</span>
                  <div class="prog-bg">
                    <div class="prog-fill" style="width:{p['proba']*100:.1f}%;background:{m['color']};"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Score vs median chart
        st.markdown(
            '<div class="section-hd"><span class="chip">STEP 2</span>Skor Anda vs Median Dataset</div>',
            unsafe_allow_html=True,
        )
        user_scores = [score_M, score_N, score_P]
        med_vals    = [medians["M"], medians["N"], medians["P"]]
        sc_cols     = ["score_M", "score_N", "score_P"]

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), facecolor="#07090f")
        for ax, (trait, m), us, mv, sc in zip(axes, TRAIT_META.items(),
                                               user_scores, med_vals, sc_cols):
            ax.set_facecolor("#0b1525")
            ax.hist(df_data[sc], bins=28, color=m["color"], alpha=0.22, edgecolor="none")
            ax.axvline(mv, color=m["color"], lw=1.6, linestyle="--", alpha=0.75,
                       label=f"Median dataset: {mv:.2f}")
            ax.axvline(us, color="#ffffff",  lw=2.2,
                       label=f"Skor Anda: {us:.2f}")
            ax.set_title(trait, color=m["color"], fontsize=9, fontweight="bold")
            ax.tick_params(colors="#4a5a7a", labelsize=7)
            ax.set_xlabel("Score", color="#4a5a7a", fontsize=7)
            ax.legend(fontsize=7, labelcolor="white",
                      facecolor="#0b1525", edgecolor="#182035")
            for sp in ax.spines.values(): sp.set_color("#182035")
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # SHAP for user's input
        st.markdown(
            '<div class="section-hd"><span class="chip">STEP 3</span>'
            'SHAP — Kontribusi Fitur terhadap Prediksi Anda</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Menggunakan RF dari Hybrid Ensemble. "
            "Bar biru/hijau/kuning → mendorong HIGH · Bar merah → mendorong LOW"
        )

        for trait, X_in in X_inputs.items():
            m     = TRAIT_META[trait]
            pipe  = model_res[trait]["Pipeline"]
            rf    = get_rf(pipe)                          # Bug-4
            X_sc  = scale_for_shap(pipe, X_in)           # Bug-3

            explainer = shap.TreeExplainer(rf)
            sv_raw    = explainer.shap_values(X_sc)
            sv        = shap_class1(sv_raw)[0]            # Bug-5, single sample

            feat_names = X_in.columns.tolist()
            top_idx    = np.argsort(np.abs(sv))[::-1][:10]

            fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#07090f")
            ax.set_facecolor("#0b1525")
            bar_colors = [m["color"] if sv[i] > 0 else "#f87171" for i in top_idx]
            ax.barh(
                [feat_names[i] for i in top_idx][::-1],
                sv[top_idx][::-1],
                color=bar_colors[::-1], height=0.55
            )
            ax.axvline(0, color="#3a4a6a", lw=1)
            ax.set_title(f"{m['icon']} {trait} — SHAP (input Anda)",
                         color=m["color"], fontsize=10, fontweight="bold")
            ax.tick_params(colors="#7a8aaa", labelsize=8)
            ax.set_xlabel("SHAP value", color="#4a5a7a", fontsize=8)
            for sp in ax.spines.values(): sp.set_color("#182035")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE ②  MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
elif nav == "📊 Performa Model":

    st.markdown(
        '<div class="section-hd">Hybrid Ensemble — Evaluasi</div>',
        unsafe_allow_html=True,
    )
    st.caption("RF + XGBoost · Weighted Soft Voting · 80/20 Stratified Split · StandardScaler in Pipeline")

    # Summary table
    rows = []
    for trait, res in model_res.items():
        rows.append({
            "Trait":     trait,
            "Train Acc": res["Train Accuracy"],
            "Test Acc":  res["Test Accuracy"],
            "Acc Gap":   res["Accuracy Gap"],
            "Precision": res["Precision"],
            "Recall":    res["Recall"],
            "F1-Score":  res["F1-Score"],
            "ROC-AUC":   res["ROC-AUC"],
        })
    df_r = pd.DataFrame(rows)
    num_cols = [c for c in df_r.columns if c != "Trait"]
    st.dataframe(
        df_r.style
            .format({c: "{:.4f}" for c in num_cols})
            .highlight_max(subset=["Test Acc", "F1-Score", "ROC-AUC"], color="#0f2a0f")
            .highlight_min(subset=["Acc Gap"], color="#0f2a0f"),
        use_container_width=True, hide_index=True,
    )

    # Per-trait detail
    st.markdown('<div class="section-hd">Detail per Trait</div>', unsafe_allow_html=True)
    for trait, res in model_res.items():
        m = TRAIT_META[trait]
        st.markdown(f"##### {m['icon']} {trait}")
        st.markdown(f"""
        <div class="m-row">
          <div class="m-box"><div class="m-lbl">Train Acc</div>
            <div class="m-val" style="color:{m['color']};">{res['Train Accuracy']:.4f}</div></div>
          <div class="m-box"><div class="m-lbl">Test Acc</div>
            <div class="m-val" style="color:{m['color']};">{res['Test Accuracy']:.4f}</div></div>
          <div class="m-box"><div class="m-lbl">Acc Gap</div>
            <div class="m-val">{res['Accuracy Gap']:.4f}</div></div>
          <div class="m-box"><div class="m-lbl">Precision</div>
            <div class="m-val">{res['Precision']:.4f}</div></div>
          <div class="m-box"><div class="m-lbl">Recall</div>
            <div class="m-val">{res['Recall']:.4f}</div></div>
          <div class="m-box"><div class="m-lbl">F1-Score</div>
            <div class="m-val" style="color:{m['color']};">{res['F1-Score']:.4f}</div></div>
          <div class="m-box"><div class="m-lbl">ROC-AUC</div>
            <div class="m-val" style="color:{m['color']};">{res['ROC-AUC']:.4f}</div></div>
        </div>""", unsafe_allow_html=True)

        cm = confusion_matrix(res["y_test"], res["Predictions"])
        fig, ax = plt.subplots(figsize=(4, 3.2), facecolor="#07090f")
        ax.set_facecolor("#0b1525")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Low", "High"], yticklabels=["Low", "High"],
                    ax=ax, linewidths=.5, linecolor="#182035",
                    annot_kws={"size": 14, "color": "white"})
        ax.set_title("Confusion Matrix", color=m["color"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted", color="#4a5a7a", fontsize=8)
        ax.set_ylabel("Actual",    color="#4a5a7a", fontsize=8)
        ax.tick_params(colors="#7a8aaa", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#182035")
        plt.tight_layout()
        cc, _ = st.columns([1, 2])
        with cc: st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("---")


# ═══════════════════════════════════════════════════════════
# PAGE ③  SHAP GLOBAL
# ═══════════════════════════════════════════════════════════
elif nav == "🔍 SHAP Global":

    st.markdown('<div class="section-hd">Global SHAP Explainability</div>',
                unsafe_allow_html=True)
    st.caption(
        "RF dari Hybrid Ensemble · "
        "Beeswarm: distribusi SHAP seluruh test set · "
        "Bar: mean |SHAP| per fitur (feature importance global)"
    )

    n_samp = st.slider(
        "Jumlah sampel SHAP (lebih banyak = lebih akurat, lebih lambat):",
        100, 1000, 400, step=50
    )

    for trait, res in model_res.items():
        m     = TRAIT_META[trait]
        pipe  = res["Pipeline"]
        X_te  = res["X_test"]
        rf    = get_rf(pipe)                              # Bug-4
        X_sc  = scale_for_shap(pipe, X_te)               # Bug-3
        n     = min(n_samp, len(X_sc))

        st.markdown(f"### {m['icon']} {trait}")
        with st.spinner(f"Menghitung SHAP — {trait}…"):
            explainer = shap.TreeExplainer(rf)
            sv_raw    = explainer.shap_values(X_sc.iloc[:n])
            sv        = shap_class1(sv_raw)               # Bug-5

        feat_names = X_te.columns.tolist()

        # ① Beeswarm Summary Plot
        st.markdown("**① SHAP Summary Plot (Beeswarm)**")
        fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor="#07090f")
        plt.sca(ax1)
        shap.summary_plot(
            sv, X_sc.iloc[:n],
            feature_names=feat_names,
            show=False, plot_size=None,
        )
        ax1.set_facecolor("#0b1525")
        ax1.set_title(f"SHAP Beeswarm — {trait}",
                      color=m["color"], fontsize=11, fontweight="bold")
        ax1.set_xlabel("SHAP value (impact on model output)",
                       color="#4a5a7a", fontsize=9)
        ax1.tick_params(colors="#7a8aaa", labelsize=8)
        for sp in ax1.spines.values(): sp.set_color("#182035")
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close()

        # ② Bar Feature Importance (Mean |SHAP|)
        st.markdown("**② SHAP Feature Importance — Mean |SHAP|**")
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:15]
        bar_clrs = [m["color"] if i == top_idx[0] else m["color"] + "80" for i in top_idx]

        fig2, ax2 = plt.subplots(figsize=(9, 4.5), facecolor="#07090f")
        ax2.set_facecolor("#0b1525")
        ax2.barh(
            [feat_names[i] for i in top_idx][::-1],
            mean_abs[top_idx][::-1],
            color=bar_clrs[::-1], height=0.6,
        )
        ax2.set_title(f"Feature Importance — {trait}",
                      color=m["color"], fontsize=11, fontweight="bold")
        ax2.set_xlabel("Mean |SHAP value|", color="#4a5a7a", fontsize=9)
        ax2.tick_params(colors="#7a8aaa", labelsize=8)
        for sp in ax2.spines.values(): sp.set_color("#182035")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Top-5 table with question text
        top5 = pd.DataFrame({
            "Rank":        range(1, 6),
            "Feature":     [feat_names[i] for i in top_idx[:5]],
            "Pertanyaan":  [QUESTIONS.get(feat_names[i], "—") for i in top_idx[:5]],
            "Mean |SHAP|": [round(float(mean_abs[i]), 4) for i in top_idx[:5]],
        })
        st.markdown("**Top 5 Fitur Paling Berpengaruh**")
        st.dataframe(top5, use_container_width=True, hide_index=True)
        st.markdown("---")
