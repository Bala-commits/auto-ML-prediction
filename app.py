# ════════════════════════════════════════════════════════════════════════
#  Intelligent Data Analysis & ML System  —  v7
#  v6 FIXES + NEW: Module 7 — Unsupervised Learning (Manual Algorithms)
#  Includes: K-Means · Hierarchical · DBSCAN · PCA · Apriori
# ════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent ML System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:      #060d18;
    --surface: #0b1628;
    --border:  #0f2236;
    --text:    #7ea8c4;
    --bright:  #a8ccde;
    --a1:      #2d8fcb;
    --a2:      #22a878;
    --a3:      #c05555;
    --a4:      #e0a844;
    --module:  #0d1e30;
}
html, body, [class*="css"] { font-family:'Inter',sans-serif; background-color:var(--bg); color:var(--text); }
h1,h2,h3 { font-family:'JetBrains Mono',monospace; color:var(--bright); letter-spacing:-0.02em; }

.hero-title { font-family:'JetBrains Mono',monospace; font-size:1.9rem; font-weight:700;
    color:#c4dff0; letter-spacing:-0.03em; margin-bottom:2px; }
.hero-sub { font-size:0.78rem; font-family:'JetBrains Mono',monospace; color:var(--a1);
    letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1.4rem; }

.module-banner { background:linear-gradient(90deg,var(--module) 0%,#0a1c30 100%);
    border-left:3px solid var(--a1); border-radius:6px; padding:14px 20px; margin:28px 0 18px 0; }
.module-banner .mod-label { font-family:'JetBrains Mono',monospace; font-size:0.72rem;
    color:var(--a1); letter-spacing:0.16em; text-transform:uppercase; margin-bottom:2px; }
.module-banner .mod-title { font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:700; color:var(--bright); }
.module-banner .mod-sub { font-size:0.73rem; color:var(--text); margin-top:3px; }

.mode-reg { background:linear-gradient(90deg,#081c30 0%,#061420 100%);
    border:1px solid #1a4a6a; border-left:5px solid #2d8fcb; border-radius:8px;
    padding:16px 22px; margin:14px 0; }
.mode-reg .mode-icon { font-size:1.6rem; float:left; margin-right:14px; margin-top:2px; }
.mode-reg .mode-tag { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#2d8fcb;
    letter-spacing:0.18em; text-transform:uppercase; margin-bottom:3px; }
.mode-reg .mode-name { font-family:'JetBrains Mono',monospace; font-size:1.15rem; font-weight:700; color:#6dc8f0; }
.mode-reg .mode-desc { font-size:0.74rem; color:var(--text); margin-top:4px; }

.mode-cls { background:linear-gradient(90deg,#1a0d28 0%,#110820 100%);
    border:1px solid #4a2a6a; border-left:5px solid #a060c8; border-radius:8px;
    padding:16px 22px; margin:14px 0; }
.mode-cls .mode-icon { font-size:1.6rem; float:left; margin-right:14px; margin-top:2px; }
.mode-cls .mode-tag { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#a060c8;
    letter-spacing:0.18em; text-transform:uppercase; margin-bottom:3px; }
.mode-cls .mode-name { font-family:'JetBrains Mono',monospace; font-size:1.15rem; font-weight:700; color:#c890f0; }
.mode-cls .mode-desc { font-size:0.74rem; color:var(--text); margin-top:4px; }

.card-grid { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }
.stat-card { flex:1; min-width:110px; background:var(--surface); border:1px solid var(--border);
    border-radius:8px; padding:14px 16px 12px; text-align:center; }
.stat-card .val { font-family:'JetBrains Mono',monospace; font-size:1.45rem; font-weight:700; color:var(--a1); line-height:1; }
.stat-card .lbl { font-size:0.69rem; color:var(--text); letter-spacing:0.08em; text-transform:uppercase; margin-top:5px; }
.metric-card { flex:1; min-width:110px; background:linear-gradient(135deg,#0b1c30 0%,#091625 100%);
    border:1px solid #0f2e4a; border-radius:8px; padding:14px 16px 12px; text-align:center; }
.metric-card .val { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:var(--a2); line-height:1; }
.metric-card .lbl { font-size:0.68rem; color:var(--text); letter-spacing:0.08em; text-transform:uppercase; margin-top:5px; }

.cls-metric-card { flex:1; min-width:110px; background:linear-gradient(135deg,#180d2a 0%,#0f0820 100%);
    border:1px solid #3a1a5a; border-radius:8px; padding:14px 16px 12px; text-align:center; }
.cls-metric-card .val { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#c890f0; line-height:1; }
.cls-metric-card .lbl { font-size:0.68rem; color:var(--text); letter-spacing:0.08em; text-transform:uppercase; margin-top:5px; }

.unsup-metric-card { flex:1; min-width:110px; background:linear-gradient(135deg,#0a1e1a 0%,#061510 100%);
    border:1px solid #1a4a38; border-radius:8px; padding:14px 16px 12px; text-align:center; }
.unsup-metric-card .val { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#22e8a0; line-height:1; }
.unsup-metric-card .lbl { font-size:0.68rem; color:var(--text); letter-spacing:0.08em; text-transform:uppercase; margin-top:5px; }

.assumption-row { display:flex; align-items:flex-start; gap:14px;
    background:var(--surface); border:1px solid var(--border); border-radius:7px; padding:12px 16px; margin-bottom:9px; }
.badge { font-family:'JetBrains Mono',monospace; font-size:0.65rem; font-weight:700;
    padding:3px 9px; border-radius:4px; letter-spacing:0.12em; white-space:nowrap; margin-top:2px; }
.badge-pass { background:#0e2e20; color:#22c97a; border:1px solid #1a5c38; }
.badge-fail { background:#2e0e0e; color:#d96060; border:1px solid #5c2020; }
.assumption-title { font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:var(--bright); margin-bottom:2px; }
.assumption-desc { font-size:0.75rem; color:var(--text); }
.assumption-extra { font-size:0.73rem; color:var(--a4); margin-top:3px; }

.insight-pos { background:#0a2318; border:1px solid #1a4a30; color:#34d490;
    border-radius:5px; padding:6px 12px; font-size:0.76rem; margin:4px 0; display:inline-block; }
.insight-neg { background:#230a0a; border:1px solid #4a1a1a; color:#e07070;
    border-radius:5px; padding:6px 12px; font-size:0.76rem; margin:4px 0; display:inline-block; }
.insight-neu { background:#111e2a; border:1px solid #1d3347; color:var(--text);
    border-radius:5px; padding:6px 12px; font-size:0.76rem; margin:4px 0; display:inline-block; }
.insight-cls { background:#1a0d28; border:1px solid #4a2a6a; color:#c890f0;
    border-radius:5px; padding:6px 12px; font-size:0.76rem; margin:4px 0; display:inline-block; }
.insight-unsup { background:#0a201a; border:1px solid #1a5038; color:#22e8a0;
    border-radius:5px; padding:6px 12px; font-size:0.76rem; margin:4px 0; display:inline-block; }

.overfit-warn { background:#2a1a08; border:1px solid #7a4a10; color:#f0b060;
    border-radius:7px; padding:12px 18px; margin:10px 0;
    font-family:'JetBrains Mono',monospace; font-size:0.82rem; }

.pred-box { background:linear-gradient(135deg,#0c1e32 0%,#091526 100%);
    border:1px solid #1a4060; border-radius:9px; padding:18px 22px; text-align:center; margin-top:12px; }
.pred-box .pred-val { font-family:'JetBrains Mono',monospace; font-size:2.1rem; font-weight:700; color:var(--a2); }
.pred-box .pred-label { font-size:0.72rem; color:var(--text); letter-spacing:0.1em; text-transform:uppercase; }
.pred-box .pred-ci { font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:var(--a4); margin-top:5px; }
.pred-box .pred-model { font-size:0.7rem; color:var(--a1); margin-top:4px; font-family:'JetBrains Mono',monospace; }

.pred-box-cls { background:linear-gradient(135deg,#150a24 0%,#0e0618 100%);
    border:1px solid #3a1a5a; border-radius:9px; padding:18px 22px; text-align:center; margin-top:12px; }
.pred-box-cls .pred-val { font-family:'JetBrains Mono',monospace; font-size:2.1rem; font-weight:700; color:#c890f0; }
.pred-box-cls .pred-label { font-size:0.72rem; color:var(--text); letter-spacing:0.1em; text-transform:uppercase; }
.pred-box-cls .pred-prob { font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:var(--a4); margin-top:5px; }
.pred-box-cls .pred-model { font-size:0.7rem; color:#a060c8; margin-top:4px; font-family:'JetBrains Mono',monospace; }

.eq-block { background:#060f1c; border:1px solid #0f2236; border-radius:8px;
    padding:16px 20px; margin:10px 0; font-family:'JetBrains Mono',monospace;
    font-size:0.85rem; color:#6ac8a8; white-space:pre-wrap; line-height:1.8; }

.outlier-badge { font-family:'JetBrains Mono',monospace; font-size:0.74rem;
    padding:4px 12px; border-radius:4px; margin-top:5px; display:inline-block; }
.outlier-found { background:#2a0f0f; border:1px solid #6a2020; color:#f08080; }
.outlier-clean { background:#0a2010; border:1px solid #1a5030; color:#50c090; }

.summary-box { background:linear-gradient(135deg,#0a1a2a 0%,#080f1c 100%);
    border:1px solid #1a3a54; border-radius:9px; padding:20px 24px; margin:10px 0; }
.summary-title { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--a1);
    letter-spacing:0.14em; text-transform:uppercase; margin-bottom:14px;
    padding-bottom:8px; border-bottom:1px solid var(--border); }
.summary-item { display:flex; align-items:flex-start; gap:10px; font-size:0.8rem;
    color:var(--text); padding:7px 0; border-bottom:1px solid #0b1e30; line-height:1.5; }
.summary-item:last-child { border-bottom:none; }
.summary-icon { font-size:0.9rem; min-width:22px; }
.summary-key { color:var(--bright); font-weight:500; min-width:220px; }
.summary-val { color:var(--a4); font-family:'JetBrains Mono',monospace; font-size:0.76rem; }

.best-model-banner { background:linear-gradient(90deg,#0a2a1a 0%,#081c10 100%);
    border:1px solid #1a6a3a; border-left:4px solid var(--a2);
    border-radius:7px; padding:14px 20px; margin:10px 0; }
.best-model-banner .bm-label { font-family:'JetBrains Mono',monospace; font-size:0.68rem;
    color:var(--a2); letter-spacing:0.16em; text-transform:uppercase; margin-bottom:3px; }
.best-model-banner .bm-name { font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#4de8a0; }
.best-model-banner .bm-stats { font-size:0.76rem; color:var(--text); margin-top:4px; }

.best-model-banner-cls { background:linear-gradient(90deg,#1a0a2a 0%,#0f0618 100%);
    border:1px solid #5a2a7a; border-left:4px solid #a060c8;
    border-radius:7px; padding:14px 20px; margin:10px 0; }
.best-model-banner-cls .bm-label { font-family:'JetBrains Mono',monospace; font-size:0.68rem;
    color:#a060c8; letter-spacing:0.16em; text-transform:uppercase; margin-bottom:3px; }
.best-model-banner-cls .bm-name { font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#c890f0; }
.best-model-banner-cls .bm-stats { font-size:0.76rem; color:var(--text); margin-top:4px; }

.unsup-result-banner { background:linear-gradient(90deg,#081e18 0%,#051210 100%);
    border:1px solid #1a5a40; border-left:4px solid #22e8a0;
    border-radius:7px; padding:14px 20px; margin:10px 0; }
.unsup-result-banner .bm-label { font-family:'JetBrains Mono',monospace; font-size:0.68rem;
    color:#22e8a0; letter-spacing:0.16em; text-transform:uppercase; margin-bottom:3px; }
.unsup-result-banner .bm-name { font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#4dffc0; }
.unsup-result-banner .bm-stats { font-size:0.76rem; color:var(--text); margin-top:4px; }

.apriori-rule-row { background:#0a1820; border:1px solid #1a3a50; border-radius:6px;
    padding:10px 14px; margin:5px 0; font-family:'JetBrains Mono',monospace; font-size:0.78rem; }
.apriori-rule-row .rule-text { color:#6dc8f0; font-weight:700; margin-bottom:4px; }
.apriori-rule-row .rule-stats { color:var(--text); font-size:0.72rem; }
.apriori-rule-row .rule-lift { color:#e0a844; font-weight:700; }

.model-config-box { background:var(--surface); border:1px solid var(--border);
    border-radius:8px; padding:16px 18px; margin-bottom:12px; }
.model-config-title { font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    color:var(--a1); letter-spacing:0.12em; text-transform:uppercase; margin-bottom:10px; }
.section-divider { height:1px; background:var(--border); margin:24px 0; }
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:6px; }
.stButton > button { background:var(--a1); color:#fff; border:none; border-radius:6px;
    font-family:'JetBrains Mono',monospace; font-size:0.82rem; letter-spacing:0.06em;
    padding:8px 20px; transition:background 0.2s; }
.stButton > button:hover { background:#1a6da8; }
.stSelectbox label, .stMultiSelect label, .stSlider label,
.stNumberInput label, .stCheckbox label { color:var(--text) !important; font-size:0.8rem !important; }
.cat-tag { display:inline-block; background:#1a0d28; border:1px solid #5a2a7a;
    color:#c890f0; font-size:0.65rem; padding:1px 6px; border-radius:3px;
    font-family:JetBrains Mono,monospace; margin-left:6px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ────────────────────────────────────────────────────────────────────────
PLOT_BG  = "#060d18"
AXES_BG  = "#0b1628"
GRID_CLR = "#0f2236"
TEXT_CLR = "#7ea8c4"
ACCENT1  = "#2d8fcb"
ACCENT2  = "#22a878"
ACCENT3  = "#c05555"
ACCENT4  = "#e0a844"
ACCENT5  = "#a060c8"
ACCENT6  = "#d05090"
ACCENT7  = "#22e8a0"

CLUSTER_PALETTE = [
    "#2d8fcb", "#22e8a0", "#e0a844", "#c05555", "#a060c8",
    "#d05090", "#44c8e0", "#e07030", "#70c040", "#c0a020",
]

REG_MODEL_COLORS = {
    "Linear":        ACCENT1,
    "Ridge":         ACCENT2,
    "Lasso":         ACCENT4,
    "KNN":           ACCENT5,
    "Decision Tree": ACCENT3,
    "Random Forest": ACCENT6,
}
CLS_MODEL_COLORS = {
    "Logistic":      ACCENT1,
    "KNN":           ACCENT5,
    "Naive Bayes":   ACCENT4,
    "Decision Tree": ACCENT3,
    "Random Forest": ACCENT6,
}

def apply_theme(fig, axes):
    fig.patch.set_facecolor(PLOT_BG)
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor(AXES_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color("#7eb8d4")
        ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)

# ════════════════════════════════════════════════════════════════════════
# MODE DETECTION
# ════════════════════════════════════════════════════════════════════════
CLASSIFICATION_THRESHOLD = 15

def detect_mode(series):
    if series.dtype == object or str(series.dtype) in ("string", "category"):
        return "classification"
    n_unique = series.nunique()
    n_total  = len(series.dropna())
    if pd.api.types.is_integer_dtype(series) and n_unique <= CLASSIFICATION_THRESHOLD:
        return "classification"
    if n_unique <= 5 and n_unique / max(n_total, 1) < 0.05:
        return "classification"
    return "regression"

# ════════════════════════════════════════════════════════════════════════
# REGRESSION — MANUAL MODEL IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════════
class ManualLinearRegression:
    name = "Linear"
    def __init__(self): self.beta = None
    def fit(self, X, y):
        Xb = np.hstack([np.ones((len(X), 1)), X])
        self.beta = np.linalg.pinv(Xb) @ y
    def predict(self, X):
        return np.hstack([np.ones((len(X), 1)), X]) @ self.beta
    def get_coefficients(self): return self.beta

class ManualRidgeRegression:
    name = "Ridge"
    def __init__(self, lam=1.0): self.lam = lam; self.beta = None
    def fit(self, X, y):
        Xb  = np.hstack([np.ones((len(X), 1)), X])
        reg = self.lam * np.eye(Xb.shape[1]); reg[0, 0] = 0
        self.beta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    def predict(self, X):
        return np.hstack([np.ones((len(X), 1)), X]) @ self.beta
    def get_coefficients(self): return self.beta

class ManualLassoRegression:
    name = "Lasso"
    def __init__(self, lam=0.1, max_iter=1000, tol=1e-4):
        self.lam = lam; self.max_iter = max_iter; self.tol = tol; self.beta = None
    def _soft(self, rho, lam): return np.sign(rho) * max(abs(rho) - lam, 0.0)
    def fit(self, X, y):
        Xb   = np.hstack([np.ones((len(X), 1)), X])
        beta = np.zeros(Xb.shape[1])
        for _ in range(self.max_iter):
            old = beta.copy()
            for j in range(Xb.shape[1]):
                r   = y - Xb @ beta + Xb[:, j] * beta[j]
                rho = Xb[:, j] @ r / len(X)
                d   = Xb[:, j] @ Xb[:, j] / len(X)
                beta[j] = rho / d if j == 0 else self._soft(rho, self.lam) / d
            if np.max(np.abs(beta - old)) < self.tol: break
        self.beta = beta
    def predict(self, X):
        return np.hstack([np.ones((len(X), 1)), X]) @ self.beta
    def get_coefficients(self): return self.beta

class ManualKNNRegression:
    name = "KNN"
    def __init__(self, k=5): self.k = k
    def fit(self, X, y): self.X_tr = X; self.y_tr = y
    def predict(self, X):
        return np.array([self.y_tr[np.argsort(np.linalg.norm(self.X_tr - xi, axis=1))[:self.k]].mean()
                         for xi in X])

# ════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — MANUAL MODEL IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════════
class ManualLogisticRegression:
    name = "Logistic"
    def __init__(self, lr=0.1, max_iter=500):
        self.lr = lr; self.max_iter = max_iter
        self.W = None; self.b = None; self.classes_ = None
    def _softmax(self, z):
        ez = np.exp(z - z.max(axis=1, keepdims=True))
        return ez / ez.sum(axis=1, keepdims=True)
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_); n, p = X.shape
        self.W = np.zeros((p, K)); self.b = np.zeros(K)
        y_oh = (y[:, None] == self.classes_[None, :]).astype(float)
        for _ in range(self.max_iter):
            probs = self._softmax(X @ self.W + self.b)
            diff  = probs - y_oh
            self.W -= self.lr * X.T @ diff / n
            self.b -= self.lr * diff.mean(axis=0)
    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

class ManualKNNClassifier:
    name = "KNN"
    def __init__(self, k=5): self.k = k
    def fit(self, X, y): self.X_tr = X; self.y_tr = y
    def predict(self, X):
        out = []
        for xi in X:
            idx = np.argsort(np.linalg.norm(self.X_tr - xi, axis=1))[:self.k]
            vals, cnts = np.unique(self.y_tr[idx], return_counts=True)
            out.append(vals[cnts.argmax()])
        return np.array(out)
    def predict_proba(self, X):
        classes = np.unique(self.y_tr)
        out = []
        for xi in X:
            idx = np.argsort(np.linalg.norm(self.X_tr - xi, axis=1))[:self.k]
            row = np.array([(self.y_tr[idx] == c).mean() for c in classes])
            out.append(row)
        return np.array(out)

class ManualNaiveBayes:
    name = "Naive Bayes"
    def __init__(self): self.classes_ = None; self.params = {}
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors[c] = len(Xc) / len(X)
            self.params[c] = (Xc.mean(axis=0), Xc.var(axis=0) + 1e-9)
    def _log_likelihood(self, X, mu, var):
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mu) ** 2 / var, axis=1)
    def predict_proba(self, X):
        log_posts = np.array([
            np.log(self.priors[c]) + self._log_likelihood(X, *self.params[c])
            for c in self.classes_
        ]).T
        log_posts -= log_posts.max(axis=1, keepdims=True)
        probs = np.exp(log_posts); probs /= probs.sum(axis=1, keepdims=True)
        return probs
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

# ════════════════════════════════════════════════════════════════════════
# UNSUPERVISED LEARNING — MANUAL IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════════

class ManualKMeans:
    """K-Means Clustering from scratch using Euclidean distance."""
    def __init__(self, k=3, max_iter=300, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(X), self.k, replace=False)
        return X[idx].copy()

    def _assign_labels(self, X):
        # Euclidean distance from each point to each centroid
        dists = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return dists.argmin(axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids_)
        for k in range(self.k):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = self.centroids_[k]
        return new_centroids

    def fit(self, X):
        self.centroids_ = self._init_centroids(X)
        for i in range(self.max_iter):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)
            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            self.n_iter_ = i + 1
            if shift < self.tol:
                break
        self.labels_ = self._assign_labels(X)
        self.inertia_ = float(sum(
            np.linalg.norm(X[self.labels_ == k] - self.centroids_[k]) ** 2
            for k in range(self.k) if (self.labels_ == k).sum() > 0
        ))
        return self

    def predict(self, X):
        return self._assign_labels(X)


class ManualHierarchicalClustering:
    """Agglomerative Hierarchical Clustering (single linkage) from scratch."""
    def __init__(self, n_clusters=3, linkage="average"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.merge_history_ = []   # list of (cluster_a, cluster_b, distance)

    def _cluster_dist(self, X, c_a, c_b):
        """Compute linkage distance between two clusters."""
        pts_a = X[list(c_a)]
        pts_b = X[list(c_b)]
        dists = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=2)
        if self.linkage == "single":
            return dists.min()
        elif self.linkage == "complete":
            return dists.max()
        else:  # average
            return dists.mean()

    def fit(self, X):
        n = len(X)
        # Each point starts as its own cluster (cluster id → set of indices)
        clusters = {i: {i} for i in range(n)}

        while len(clusters) > self.n_clusters:
            ids = list(clusters.keys())
            best_dist = np.inf
            best_pair = (ids[0], ids[1])

            # Find the two closest clusters
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    d = self._cluster_dist(X, clusters[ids[i]], clusters[ids[j]])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ids[i], ids[j])

            a, b = best_pair
            self.merge_history_.append((a, b, best_dist))
            # Merge b into a
            clusters[a] = clusters[a] | clusters[b]
            del clusters[b]

        # Assign integer labels 0..n_clusters-1
        self.labels_ = np.zeros(n, dtype=int)
        for label_idx, (_, members) in enumerate(clusters.items()):
            for pt in members:
                self.labels_[pt] = label_idx
        return self


class ManualDBSCAN:
    """DBSCAN from scratch using region-growing approach."""
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = 0
        self.n_noise_ = 0

    def _get_neighbors(self, X, idx):
        dists = np.linalg.norm(X - X[idx], axis=1)
        return np.where(dists <= self.eps)[0]

    def fit(self, X):
        n = len(X)
        labels = np.full(n, -2, dtype=int)   # -2 = unvisited, -1 = noise
        cluster_id = 0

        for i in range(n):
            if labels[i] != -2:
                continue
            neighbors = self._get_neighbors(X, i)

            # Not a core point → mark noise (may be revisited as border)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            # Start a new cluster
            labels[i] = cluster_id
            seed_set = list(neighbors)
            seed_set = [s for s in seed_set if s != i]

            ptr = 0
            while ptr < len(seed_set):
                q = seed_set[ptr]
                ptr += 1
                if labels[q] == -1:
                    labels[q] = cluster_id   # border point
                if labels[q] != -2:
                    continue
                labels[q] = cluster_id
                q_neighbors = self._get_neighbors(X, q)
                if len(q_neighbors) >= self.min_samples:
                    seed_set.extend([s for s in q_neighbors if labels[s] == -2 or labels[s] == -1])

            cluster_id += 1

        # Any remaining unvisited (shouldn't happen) → noise
        labels[labels == -2] = -1
        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.n_noise_ = int((labels == -1).sum())
        return self


class ManualPCA:
    """Principal Component Analysis from scratch via eigendecomposition."""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        cov = np.cov(X_c, rowvar=False)
        if cov.ndim == 0:  # single feature edge case
            cov = np.array([[float(cov)]])
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var if total_var > 0 else self.explained_variance_ * 0
        )
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ManualApriori:
    """Apriori algorithm for association rule mining from scratch."""
    def __init__(self, min_support=0.2, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets_ = {}   # frozenset → support
        self.rules_ = []               # list of dicts

    def _get_support(self, transactions, itemset):
        count = sum(1 for t in transactions if itemset.issubset(t))
        return count / len(transactions)

    def _generate_candidates(self, prev_itemsets, k):
        """Generate k-itemset candidates by joining (k-1)-itemsets."""
        items_list = [sorted(list(s)) for s in prev_itemsets]
        candidates = set()
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                a, b = items_list[i], items_list[j]
                # Join if first k-2 elements match
                if a[:k-2] == b[:k-2]:
                    candidate = frozenset(a) | frozenset(b)
                    if len(candidate) == k:
                        candidates.add(candidate)
        return candidates

    def fit(self, transactions):
        """
        transactions: list of sets/frozensets of items.
        """
        transactions = [frozenset(t) for t in transactions]
        n = len(transactions)

        # 1-itemsets
        all_items = set(item for t in transactions for item in t)
        current_frequent = set()
        for item in all_items:
            fs = frozenset([item])
            sup = self._get_support(transactions, fs)
            if sup >= self.min_support:
                self.frequent_itemsets_[fs] = sup
                current_frequent.add(fs)

        k = 2
        while current_frequent:
            candidates = self._generate_candidates(current_frequent, k)
            current_frequent = set()
            for cand in candidates:
                sup = self._get_support(transactions, cand)
                if sup >= self.min_support:
                    self.frequent_itemsets_[cand] = sup
                    current_frequent.add(cand)
            k += 1

        # Generate association rules
        for itemset, sup in self.frequent_itemsets_.items():
            if len(itemset) < 2:
                continue
            items = list(itemset)
            # Generate all non-empty proper subsets as antecedents
            for size in range(1, len(items)):
                for i in range(len(items)):
                    # Use combinations-style iteration
                    pass
            # Proper subset generation
            for mask in range(1, 2 ** len(items) - 1):
                antecedent = frozenset(items[i] for i in range(len(items)) if mask & (1 << i))
                consequent = itemset - antecedent
                if not consequent:
                    continue
                ant_sup = self.frequent_itemsets_.get(antecedent)
                if ant_sup is None or ant_sup == 0:
                    continue
                confidence = sup / ant_sup
                if confidence >= self.min_confidence:
                    cons_sup = self.frequent_itemsets_.get(consequent, self._get_support(transactions, consequent))
                    lift = confidence / cons_sup if cons_sup > 0 else 0.0
                    self.rules_.append({
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support":    round(sup, 4),
                        "confidence": round(confidence, 4),
                        "lift":       round(lift, 4),
                    })

        # Deduplicate rules (same antecedent+consequent may appear multiple times)
        seen = set()
        unique_rules = []
        for r in self.rules_:
            key = (r["antecedent"], r["consequent"])
            if key not in seen:
                seen.add(key)
                unique_rules.append(r)
        self.rules_ = sorted(unique_rules, key=lambda x: x["lift"], reverse=True)
        return self


# ════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ════════════════════════════════════════════════════════════════════════
def calc_r2(yt, yp):
    ss_res = np.sum((yt - yp)**2); ss_tot = np.sum((yt - yt.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot else 0.0
def calc_rmse(yt, yp): return float(np.sqrt(np.mean((yt - yp)**2)))
def calc_mae(yt, yp):  return float(np.mean(np.abs(yt - yp)))
def calc_adj_r2(r2, n, k):
    return float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if n > k + 1 else float("nan")

def reg_evaluate(model, X_te, y_te):
    yp = model.predict(X_te)
    return {"R²": calc_r2(y_te, yp), "RMSE": calc_rmse(y_te, yp),
            "MAE": calc_mae(y_te, yp), "yp": yp}

def silhouette_score_manual(X, labels):
    """Compute silhouette score from scratch (mean over all samples)."""
    unique_labels = [l for l in np.unique(labels) if l != -1]
    if len(unique_labels) < 2:
        return float("nan")
    n = len(X)
    s_vals = []
    for i in range(n):
        if labels[i] == -1:
            continue
        same = X[labels == labels[i]]
        if len(same) <= 1:
            a = 0.0
        else:
            a = float(np.linalg.norm(same - X[i], axis=1).sum() / (len(same) - 1))
        b_vals = []
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other = X[labels == lbl]
            if len(other) == 0:
                continue
            b_vals.append(float(np.linalg.norm(other - X[i], axis=1).mean()))
        if not b_vals:
            continue
        b = min(b_vals)
        denom = max(a, b)
        s_vals.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(s_vals)) if s_vals else float("nan")

# ════════════════════════════════════════════════════════════════════════
# IQR OUTLIER HELPER
# ════════════════════════════════════════════════════════════════════════
def detect_outliers_iqr(series):
    q1 = series.quantile(0.25); q3 = series.quantile(0.75)
    iqr = q3 - q1; lo = q1 - 1.5*iqr; hi = q3 + 1.5*iqr
    return (series < lo) | (series > hi), q1, q3, iqr, lo, hi

# ════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ════════════════════════════════════════════════════════════════════════
def stat_card(v, l):
    return f'<div class="stat-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
def metric_card(v, l, cls="metric-card"):
    return f'<div class="{cls}"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
def module_banner(num, title, sub=""):
    s = f'<div class="mod-sub">{sub}</div>' if sub else ""
    return (f'<div class="module-banner"><div class="mod-label">⬡ Module {num}</div>'
            f'<div class="mod-title">{title}</div>{s}</div>')
def assumption_row(title, desc, ok, extra=""):
    badge = f'<span class="badge badge-{"pass" if ok else "fail"}">{"PASS" if ok else "FAIL"}</span>'
    ex    = f'<div class="assumption-extra">{extra}</div>' if extra else ""
    return (f'<div class="assumption-row">{badge}<div>'
            f'<div class="assumption-title">{title}</div>'
            f'<div class="assumption-desc">{desc}</div>{ex}</div></div>')
def summary_row(icon, key, val):
    return (f'<div class="summary-item"><span class="summary-icon">{icon}</span>'
            f'<span class="summary-key">{key}</span>'
            f'<span class="summary-val">{val}</span></div>')

# ════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">Intelligent Data Analysis & ML System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Auto-detects Regression vs Classification · '
    'Supports Numeric & Categorical Features/Targets · Manual & Library Models · '
    'Unsupervised Learning</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"], label_visibility="collapsed")
if uploaded_file is None:
    st.info("Upload a CSV file above to begin.")
    st.stop()

data         = pd.read_csv(uploaded_file)
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
cat_cols     = data.select_dtypes(exclude=np.number).columns.tolist()
all_cols     = data.columns.tolist()
total_miss   = int(data.isnull().sum().sum())

# ════════════════════════════════════════════════════════════════════════
# MODULE 1 — SMART EDA
# ════════════════════════════════════════════════════════════════════════
st.markdown(module_banner("1", "Smart Exploratory Data Analysis",
                          "Auto-generated insights before model training"), unsafe_allow_html=True)

st.markdown("## Dataset Overview")
cards = '<div class="card-grid">'
for v, l in [(f"{data.shape[0]:,}", "Rows"), (str(data.shape[1]), "Columns"),
             (str(len(numeric_cols)), "Numeric"), (str(len(cat_cols)), "Categorical"),
             (f"{total_miss:,}", "Missing")]:
    cards += stat_card(v, l)
cards += '</div>'
st.markdown(cards, unsafe_allow_html=True)

cl, cr = st.columns(2)
with cl:
    st.markdown("#### Dataset Preview (First 5 Rows)")
    st.dataframe(data.head(5), use_container_width=True)
with cr:
    st.markdown("#### Missing Value Summary")
    mv = pd.DataFrame({"Column": data.columns,
                        "Missing": data.isnull().sum().values,
                        "% Missing": (data.isnull().sum()/len(data)*100).round(2).values})
    mv = mv[mv["Missing"] > 0]
    if mv.empty: st.success("No missing values found.")
    else:        st.dataframe(mv, use_container_width=True, hide_index=True)

stats_rows = []
if numeric_cols:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## Statistical Summary")
    for c in numeric_cols:
        cd = data[c].dropna()
        stats_rows.append({"Column": c, "Mean": round(cd.mean(), 4), "Median": round(cd.median(), 4),
                            "Std Dev": round(cd.std(), 4), "Min": round(cd.min(), 4),
                            "Max": round(cd.max(), 4), "Skewness": round(cd.skew(), 4)})
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

corr_matrix = corr_with_target = None
if len(numeric_cols) >= 2:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## Correlation Heatmap")
    corr_matrix = data[numeric_cols].corr()
    fig_c, ax_c = plt.subplots(figsize=(max(6, len(numeric_cols)*0.9), max(5, len(numeric_cols)*0.8)))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.4, linecolor=GRID_CLR,
                annot_kws={"size": 8, "color": "#c9d8e8"},
                cbar_kws={"shrink": 0.8}, ax=ax_c)
    ax_c.set_title("Feature Correlation Matrix", fontsize=11, color="#7eb8d4")
    apply_theme(fig_c, ax_c); ax_c.tick_params(colors=TEXT_CLR, labelsize=8)
    fig_c.patch.set_facecolor(PLOT_BG); fig_c.tight_layout()
    st.pyplot(fig_c); plt.close()

    st.markdown("### Correlation Insights")
    found = False; seen = set()
    for i, a in enumerate(numeric_cols):
        for j, b in enumerate(numeric_cols):
            if i >= j: continue
            k = tuple(sorted([a, b]))
            if k in seen: continue
            seen.add(k); r = corr_matrix.loc[a, b]
            if r > 0.7:
                st.markdown(f'<div class="insight-pos">📈 {a} ↔ {b} (r = {r:.2f})</div>', unsafe_allow_html=True)
                found = True
            elif r < -0.7:
                st.markdown(f'<div class="insight-neg">📉 {a} ↔ {b} (r = {r:.2f})</div>', unsafe_allow_html=True)
                found = True
    if not found:
        st.markdown('<div class="insight-neu">ℹ No strong correlations (|r| > 0.7) detected.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Target Correlation Analysis")

target_col_eda = st.selectbox(
    "Select Target Column for EDA",
    all_cols,
    key="eda_target"
)

eda_target_is_numeric = target_col_eda in numeric_cols

if eda_target_is_numeric:
    other_cols = [c for c in numeric_cols if c != target_col_eda]
    if other_cols and corr_matrix is not None and target_col_eda in corr_matrix.columns:
        corr_with_target = corr_matrix[target_col_eda].drop(target_col_eda).sort_values()
        cp, cn = st.columns(2)
        with cp:
            st.markdown("#### Strongest Positive Predictors")
            for feat, r in corr_with_target.tail(3)[::-1].items():
                st.markdown(f'<div class="insight-pos">📈 {feat} — r = {r:.4f}</div>', unsafe_allow_html=True)
        with cn:
            st.markdown("#### Strongest Negative Predictors")
            for feat, r in corr_with_target.head(3).items():
                st.markdown(f'<div class="insight-neg">📉 {feat} — r = {r:.4f}</div>', unsafe_allow_html=True)
        fig_t, ax_t = plt.subplots(figsize=(8, max(3, len(other_cols)*0.45)))
        ax_t.barh(corr_with_target.index, corr_with_target.values,
                  color=[ACCENT2 if v >= 0 else ACCENT3 for v in corr_with_target.values],
                  edgecolor=AXES_BG, linewidth=0.5)
        ax_t.axvline(0, color=TEXT_CLR, linewidth=0.8)
        ax_t.set_xlabel("Correlation with Target", fontsize=9)
        ax_t.set_title(f"Feature Correlations with '{target_col_eda}'", fontsize=11)
        apply_theme(fig_t, ax_t); fig_t.tight_layout()
        st.pyplot(fig_t); plt.close()
    else:
        st.info("Select a different target or add more numeric columns.")
else:
    st.markdown(f'<div class="insight-cls">🏷️ <strong>{target_col_eda}</strong> is a categorical target — showing class distribution and numeric feature breakdown by class.</div>', unsafe_allow_html=True)
    classes_eda = data[target_col_eda].dropna().unique()
    n_classes   = len(classes_eda)
    palette     = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6]
    value_counts = data[target_col_eda].value_counts()
    fig_vd, ax_vd = plt.subplots(figsize=(max(5, n_classes * 1.2), 3.5))
    bar_clrs = [palette[i % len(palette)] for i in range(len(value_counts))]
    ax_vd.bar(value_counts.index.astype(str), value_counts.values, color=bar_clrs, edgecolor=AXES_BG, linewidth=0.5)
    for i, (lbl, cnt) in enumerate(value_counts.items()):
        ax_vd.text(i, cnt + max(value_counts)*0.01, f"{cnt}\n({cnt/len(data)*100:.1f}%)",
                   ha="center", va="bottom", fontsize=8, color=TEXT_CLR, fontfamily="monospace")
    ax_vd.set_title(f"Class Distribution — '{target_col_eda}'", fontsize=11)
    ax_vd.set_ylabel("Count", fontsize=9)
    apply_theme(fig_vd, ax_vd); fig_vd.tight_layout()
    st.pyplot(fig_vd); plt.close()
    if numeric_cols:
        st.markdown(f"#### Numeric Feature Breakdown by '{target_col_eda}' Class")
        for feat in numeric_cols:
            fig_bp2, ax_bp2 = plt.subplots(figsize=(max(6, n_classes * 1.3), 3.5))
            group_data = [data[data[target_col_eda] == cls][feat].dropna().values for cls in value_counts.index]
            bp = ax_bp2.boxplot(group_data, patch_artist=True, labels=[str(c) for c in value_counts.index],
                                widths=0.55, medianprops=dict(color=ACCENT4, linewidth=2))
            for patch, clr in zip(bp["boxes"], bar_clrs):
                patch.set_facecolor(clr); patch.set_alpha(0.5)
            for w in bp["whiskers"]: w.set(color=TEXT_CLR, linewidth=1, linestyle="--")
            for c in bp["caps"]:     c.set(color=TEXT_CLR, linewidth=1)
            for f in bp["fliers"]:   f.set(marker="o", color=ACCENT3, alpha=0.5, markersize=3)
            ax_bp2.set_title(f"{feat} by {target_col_eda}", fontsize=10)
            ax_bp2.set_ylabel(feat, fontsize=9)
            apply_theme(fig_bp2, ax_bp2); fig_bp2.tight_layout()
            st.pyplot(fig_bp2); plt.close()
        st.markdown(f"#### Feature–Class Correlation Table (Mean per Class)")
        means = data.groupby(target_col_eda)[numeric_cols].mean().round(3)
        st.dataframe(means.T, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Outlier Detection (IQR Method)")
st.markdown('<p style="font-size:0.78rem;color:#7ea8c4;margin-bottom:14px;">'
            'Outlier rule: value &lt; Q1 − 1.5×IQR &nbsp;|&nbsp; value &gt; Q3 + 1.5×IQR</p>',
            unsafe_allow_html=True)

outlier_counts = {}
if numeric_cols:
    for row_i in range(int(np.ceil(len(numeric_cols)/2))):
        st_cols = st.columns(2)
        for ci in range(2):
            fi = row_i*2 + ci
            if fi >= len(numeric_cols): break
            cn_name = numeric_cols[fi]
            cd = data[cn_name].dropna()
            mask, q1, q3, iqr, lo_b, hi_b = detect_outliers_iqr(cd)
            n_out = int(mask.sum()); outlier_counts[cn_name] = n_out
            with st_cols[ci]:
                fig_bp3, ax_bp3 = plt.subplots(figsize=(5.5, 2.6))
                ax_bp3.boxplot(cd, vert=False, patch_artist=True, widths=0.55,
                               boxprops=dict(facecolor="#0e2840", color=ACCENT1, linewidth=1.2),
                               medianprops=dict(color=ACCENT2, linewidth=2),
                               whiskerprops=dict(color=TEXT_CLR, linewidth=1, linestyle="--"),
                               capprops=dict(color=ACCENT1, linewidth=1.5),
                               flierprops=dict(marker="o", color=ACCENT3, alpha=0.7, markersize=4, markeredgewidth=0))
                ax_bp3.set_title(cn_name, fontsize=10); ax_bp3.set_xlabel("Value", fontsize=8); ax_bp3.set_yticks([])
                apply_theme(fig_bp3, ax_bp3); fig_bp3.tight_layout(pad=1.2)
                st.pyplot(fig_bp3); plt.close()
                is_bad = n_out > 0
                badge_cls = "outlier-found" if is_bad else "outlier-clean"
                badge_txt = f"{n_out} outlier{'s' if n_out!=1 else ''} detected" if is_bad else "No outliers"
                st.markdown(f'<div class="outlier-badge {badge_cls}">{"⚠" if is_bad else "✓"} {badge_txt}'
                            f'&nbsp;&nbsp;|&nbsp;&nbsp;IQR={iqr:.3f} [{lo_b:.3f},{hi_b:.3f}]</div>',
                            unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Automated Data Insights Summary")

def build_summary(data, numeric_cols, cat_cols, total_miss,
                  corr_with_target, outlier_counts, stats_rows):
    items = []
    items.append(("📐", "Dataset Dimensions",
        f"{data.shape[0]:,} rows × {data.shape[1]} cols ({len(numeric_cols)} numeric, {len(cat_cols)} categorical)"))
    if total_miss == 0:
        items.append(("✅", "Missing Values", "Complete — no missing cells detected"))
    else:
        top_m = data.isnull().sum().idxmax()
        items.append(("⚠", "Missing Values",
            f"{total_miss:,} missing ({round(total_miss/data.size*100,2)}%) — worst: '{top_m}'"))
    if corr_with_target is not None and len(corr_with_target):
        best = corr_with_target.abs().idxmax(); r = corr_with_target[best]
        items.append(("🔗", "Strongest Predictor",
            f"'{best}' — {'positive' if r>0 else 'negative'} (r = {r:.4f})"))
    if outlier_counts:
        total_o = sum(outlier_counts.values()); cols_hit = [c for c, n in outlier_counts.items() if n > 0]
        if total_o == 0:
            items.append(("🧹", "Outlier Summary", "Clean — no outliers detected"))
        else:
            worst = max(outlier_counts, key=outlier_counts.get)
            items.append(("🔴", "Outlier Summary",
                f"{total_o} outlier(s) in {len(cols_hit)} column(s) — worst: '{worst}' ({outlier_counts[worst]})"))
    if stats_rows:
        hs = [(r["Column"], r["Skewness"]) for r in stats_rows if abs(r["Skewness"]) > 1.0]
        if not hs: items.append(("📊", "Distribution Shape", "All features approximately symmetric (|skew|≤1)"))
        else:
            desc = ", ".join(f"'{c}' ({s:+.2f})" for c, s in hs[:5])
            items.append(("📊", "Highly Skewed Features", f"{len(hs)} feature(s): {desc}"))
    if cat_cols:
        items.append(("🏷️", "Categorical Columns", f"{len(cat_cols)}: {', '.join(cat_cols)}"))
    low_v = [c for c in numeric_cols if data[c].std() < 1e-6]
    items.append(("⚡", "Near-Constant Columns",
        f"{len(low_v)}: {', '.join(low_v)}" if low_v else "None — all columns have meaningful variance"))
    n_dup = int(data.duplicated().sum())
    items.append(("🔁", "Duplicate Rows",
        f"{n_dup} duplicate row(s) detected" if n_dup else "None detected"))
    return items

summary_items = build_summary(data, numeric_cols, cat_cols, total_miss,
                               corr_with_target, outlier_counts, stats_rows)
rows_html = "".join(summary_row(ic, k, v) for ic, k, v in summary_items)
st.markdown(f'<div class="summary-box"><div class="summary-title">⬡ Auto-Generated EDA Summary</div>'
            f'{rows_html}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# MODULE 2 — ML DASHBOARD
# ════════════════════════════════════════════════════════════════════════
st.markdown(module_banner(
    "2", "Machine Learning Dashboard",
    "Auto-detects Regression or Classification · Supports Numeric & Categorical Features/Targets"
), unsafe_allow_html=True)

st.markdown("## Data Configuration")
if cat_cols:
    st.info(f"🏷️ Categorical columns detected: {', '.join(cat_cols)}. "
            f"These are available as both target and input features (auto label-encoded).")

dc1, dc2 = st.columns(2)
with dc1:
    ml_target = st.selectbox(
        "Target Column (Y) — numeric or categorical",
        all_cols,
        key="ml_target"
    )
with dc2:
    avail_feats = [c for c in all_cols if c != ml_target]
    ml_features = st.multiselect(
        "Input Features (X) — numeric & categorical supported",
        avail_feats,
        default=avail_feats,
        key="ml_features"
    )

if ml_features:
    tags = ""
    for f in ml_features:
        if f in cat_cols:
            tags += f'<span style="background:#1a0d28;border:1px solid #5a2a7a;color:#c890f0;font-size:0.65rem;padding:2px 7px;border-radius:3px;font-family:JetBrains Mono,monospace;margin:2px;">{f} 🏷️ cat</span> '
        else:
            tags += f'<span style="background:#0a1a2a;border:1px solid #1a4060;color:#6dc8f0;font-size:0.65rem;padding:2px 7px;border-radius:3px;font-family:JetBrains Mono,monospace;margin:2px;">{f} # num</span> '
    st.markdown(f'<div style="margin:6px 0 12px;">{tags}</div>', unsafe_allow_html=True)

oc1, oc2 = st.columns(2)
with oc1:
    split_pct = st.slider("Training Data %", 60, 90, 80, 5)
with oc2:
    apply_std = st.checkbox("Apply Z-score Standardization", value=False)

ml_mode      = detect_mode(data[ml_target])
n_unique_tgt = data[ml_target].nunique()

if ml_mode == "regression":
    st.markdown(
        f'<div class="mode-reg"><div class="mode-icon">📈</div>'
        f'<div class="mode-tag">Auto-Detected Mode</div>'
        f'<div class="mode-name">Regression Mode</div>'
        f'<div class="mode-desc">Target <strong>{ml_target}</strong> is continuous numeric '
        f'({n_unique_tgt} unique values) — training 6 regression models.</div></div>',
        unsafe_allow_html=True,
    )
else:
    classes_preview = data[ml_target].dropna().unique()
    classes_str = ", ".join([str(c) for c in classes_preview[:8]])
    if len(classes_preview) > 8: classes_str += "…"
    st.markdown(
        f'<div class="mode-cls"><div class="mode-icon">🏷️</div>'
        f'<div class="mode-tag">Auto-Detected Mode</div>'
        f'<div class="mode-name">Classification Mode</div>'
        f'<div class="mode-desc">Target <strong>{ml_target}</strong> has {n_unique_tgt} unique classes '
        f'[{classes_str}] — training 5 classification models.</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## Model Hyperparameters")
hp1, hp2, hp3 = st.columns(3)
with hp2:
    st.markdown('<div class="model-config-box"><div class="model-config-title">KNN</div>', unsafe_allow_html=True)
    knn_k = st.slider("K neighbours", 1, 30, 5, 1, key="knn_k")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
train_btn = st.button("🚀  Train All Models", use_container_width=True)

if train_btn:
    if not ml_features:
        st.warning("Select at least one input feature."); st.stop()

    with st.spinner(f"Training all models in {ml_mode} mode…"):
        np.random.seed(42)

        feature_encoders = {}
        X_parts = []
        for feat in ml_features:
            col = data[feat]
            if col.dtype == object or str(col.dtype) in ("string", "category"):
                fe       = LabelEncoder()
                mode_val = col.mode()[0] if not col.mode().empty else "missing"
                filled   = col.fillna(mode_val).astype(str)
                encoded  = fe.fit_transform(filled).astype(float)
                feature_encoders[feat] = fe
                X_parts.append(encoded.reshape(-1, 1))
            else:
                filled = col.fillna(col.median()).values.astype(float)
                X_parts.append(filled.reshape(-1, 1))
        X_raw = np.hstack(X_parts)

        xmu = xsg = None
        if apply_std:
            xmu = X_raw.mean(axis=0); xsg = X_raw.std(axis=0); xsg[xsg == 0] = 1.0
            X   = (X_raw - xmu) / xsg
        else:
            X = X_raw.copy()

        le = None
        if ml_mode == "regression":
            y = data[ml_target].fillna(data[ml_target].median()).values.astype(float)
            classes = None
        else:
            le = LabelEncoder()
            raw_y = data[ml_target].fillna(data[ml_target].mode()[0]).astype(str)
            y = le.fit_transform(raw_y)
            classes = le.classes_

        n = len(X); split_n = int(n * split_pct / 100)
        idx = np.random.permutation(n)
        X_tr, X_te = X[idx[:split_n]], X[idx[split_n:]]
        y_tr, y_te = y[idx[:split_n]], y[idx[split_n:]]

        results = {}; fitted = {}

        if ml_mode == "regression":
            models = {
                "Linear":        ManualLinearRegression(),
                "Ridge":         ManualRidgeRegression(lam=1.0),
                "Lasso":         ManualLassoRegression(lam=0.1),
                "KNN":           ManualKNNRegression(k=knn_k),
                "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            }
            for mname, m in models.items():
                m.fit(X_tr, y_tr)
                ev = reg_evaluate(m, X_te, y_te)
                ev["r2_train"] = calc_r2(y_tr, m.predict(X_tr))
                results[mname] = ev; fitted[mname] = m
            best_name = max(results, key=lambda k: results[k]["R²"])
        else:
            cls_models = {
                "Logistic":      ManualLogisticRegression(lr=0.1, max_iter=500),
                "KNN":           ManualKNNClassifier(k=knn_k),
                "Naive Bayes":   ManualNaiveBayes(),
                "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            }
            avg = "binary" if len(classes) == 2 else "weighted"
            for mname, m in cls_models.items():
                m.fit(X_tr, y_tr)
                yp_te = m.predict(X_te); yp_tr = m.predict(X_tr)
                cm  = confusion_matrix(y_te, yp_te)
                cr  = classification_report(y_te, yp_te, output_dict=True, zero_division=0)
                results[mname] = {
                    "Accuracy":   float(accuracy_score(y_te, yp_te)),
                    "Precision":  float(precision_score(y_te, yp_te, average=avg, zero_division=0)),
                    "Recall":     float(recall_score(y_te, yp_te, average=avg, zero_division=0)),
                    "F1":         float(f1_score(y_te, yp_te, average=avg, zero_division=0)),
                    "Acc_train":  float(accuracy_score(y_tr, yp_tr)),
                    "CM": cm, "CR": cr, "yp": yp_te,
                }
                fitted[mname] = m
            best_name = max(results, key=lambda k: results[k]["F1"])

        st.session_state.update({
            "_trained":          True,
            "_mode":             ml_mode,
            "_feat":             ml_features,
            "_target":           ml_target,
            "_results":          results,
            "_fitted":           fitted,
            "_best":             best_name,
            "_X_te":             X_te,
            "_y_te":             y_te,
            "_X_tr":             X_tr,
            "_y_tr":             y_tr,
            "_data_snap":        data,
            "_apply_std":        apply_std,
            "_xmu":              xmu,
            "_xsg":              xsg,
            "_le":               le,
            "_classes":          classes,
            "_feature_encoders": feature_encoders,
            "_last_pred":        None,
            "_last_pred_cls":    None,
        })

    mode_label = "Regression" if ml_mode == "regression" else "Classification"
    best_metric = f"R² = {results[best_name]['R²']:.4f}" if ml_mode == "regression" \
                  else f"F1 = {results[best_name]['F1']:.4f}"
    st.success(f"✓ All models trained [{mode_label}]. Best: **{best_name}** ({best_metric})")


# ════════════════════════════════════════════════════════════════════════
# POST-TRAINING DASHBOARD
# ════════════════════════════════════════════════════════════════════════
if st.session_state.get("_trained"):
    results   = st.session_state["_results"]
    fitted    = st.session_state["_fitted"]
    best_name = st.session_state["_best"]
    t_feat    = st.session_state["_feat"]
    t_tgt     = st.session_state["_target"]
    X_te      = st.session_state["_X_te"]
    y_te      = st.session_state["_y_te"]
    X_tr      = st.session_state["_X_tr"]
    y_tr      = st.session_state["_y_tr"]
    dsnap     = st.session_state["_data_snap"]
    a_std     = st.session_state["_apply_std"]
    xmu       = st.session_state["_xmu"]
    xsg       = st.session_state["_xsg"]
    le        = st.session_state["_le"]
    classes   = st.session_state["_classes"]
    ml_mode   = st.session_state["_mode"]
    feat_enc  = st.session_state.get("_feature_encoders", {})
    best_res  = results[best_name]
    MODEL_COLORS = REG_MODEL_COLORS if ml_mode == "regression" else CLS_MODEL_COLORS

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # REGRESSION POST-TRAINING
    # ════════════════════════════════════════════════════════════════════
    if ml_mode == "regression":
        st.markdown(
            f'<div class="best-model-banner">'
            f'<div class="bm-label">🏆 Best Regression Model</div>'
            f'<div class="bm-name">{best_name}</div>'
            f'<div class="bm-stats">R² = {best_res["R²"]:.4f} &nbsp;|&nbsp; '
            f'RMSE = {best_res["RMSE"]:.4f} &nbsp;|&nbsp; MAE = {best_res["MAE"]:.4f}</div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown(module_banner("3", "Model Comparison", "All 6 regression models ranked by Test R²"), unsafe_allow_html=True)
        comp_rows = []
        for mname, ev in results.items():
            gap = ev["r2_train"] - ev["R²"]
            comp_rows.append({"Model": mname, "R² (Test)": round(ev["R²"], 4),
                               "R² (Train)": round(ev["r2_train"], 4),
                               "RMSE": round(ev["RMSE"], 4), "MAE": round(ev["MAE"], 4),
                               "Overfit Gap": round(gap, 4)})
        comp_df = pd.DataFrame(comp_rows).sort_values("R² (Test)", ascending=False).reset_index(drop=True)
        comp_df.insert(0, "Rank", range(1, len(comp_df)+1))
        comp_df.insert(1, "🏆", comp_df["Model"].apply(lambda m: "★" if m == best_name else ""))
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        sorted_names = comp_df["Model"].tolist()
        bar_colors   = [MODEL_COLORS.get(m, ACCENT1) for m in sorted_names]
        edge_colors  = ["#ffd700" if m == best_name else AXES_BG for m in sorted_names]
        edge_widths  = [2.0 if m == best_name else 0.5 for m in sorted_names]

        r2_vals = [results[m]["R²"] for m in sorted_names]
        fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
        bars = ax_cmp.bar(sorted_names, r2_vals, color=bar_colors, edgecolor=edge_colors, linewidth=edge_widths)
        ax_cmp.axhline(0, color=TEXT_CLR, linewidth=0.6)
        for bar, val in zip(bars, r2_vals):
            ax_cmp.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_CLR)
        ax_cmp.set_ylabel("R² Score"); ax_cmp.set_title("Regression Model Comparison — Test R²")
        apply_theme(fig_cmp, ax_cmp); fig_cmp.tight_layout(); st.pyplot(fig_cmp); plt.close()

        st.markdown(module_banner("4", "Per-Model Diagnostics", "Actual vs Predicted · Residuals"), unsafe_allow_html=True)
        for mname in sorted_names:
            ev       = results[mname]
            yp_te    = ev["yp"]; resid = y_te - yp_te
            r2_tr    = ev["r2_train"]; r2_te = ev["R²"]; gap = r2_tr - r2_te
            lc       = MODEL_COLORS.get(mname, ACCENT1)
            is_best  = mname == best_name
            badge_h  = ('&nbsp;<span style="background:#1a4a28;color:#4de8a0;font-size:0.65rem;'
                        'padding:2px 7px;border-radius:3px;">BEST</span>' if is_best else "")
            st.markdown(f'<div style="border-left:3px solid {lc};padding:4px 12px;margin:18px 0 8px;'
                        f'font-family:JetBrains Mono,monospace;font-size:0.95rem;color:{lc};font-weight:700;">'
                        f'{mname}{badge_h}</div>', unsafe_allow_html=True)
            adj_r2 = calc_adj_r2(r2_te, len(y_te), len(t_feat))
            mc = '<div class="card-grid">'
            for v, l in [(f"{r2_te:.4f}", "R² Test"), (f"{r2_tr:.4f}", "R² Train"),
                         (f"{ev['RMSE']:.4f}", "RMSE"), (f"{ev['MAE']:.4f}", "MAE"),
                         (f"{adj_r2:.4f}", "Adj R²"), (f"{gap:.4f}", "Overfit Gap")]:
                mc += metric_card(v, l)
            mc += '</div>'
            st.markdown(mc, unsafe_allow_html=True)
            if gap > 0.15:
                st.markdown(f'<div class="overfit-warn">⚠ Possible Overfitting — gap={gap:.4f}</div>', unsafe_allow_html=True)

            pg1, pg2 = st.columns(2)
            with pg1:
                fig1, ax1 = plt.subplots(figsize=(5.5, 4))
                ax1.scatter(y_te, yp_te, s=20, color=lc, alpha=0.6, edgecolors="none")
                lo, hi = min(y_te.min(), yp_te.min()), max(y_te.max(), yp_te.max())
                ax1.plot([lo,hi],[lo,hi], color=ACCENT3, linewidth=1.5, linestyle="--", label="Ideal")
                ax1.set_xlabel("Actual"); ax1.set_ylabel("Predicted")
                ax1.set_title(f"{mname} — Actual vs Predicted", fontsize=9)
                ax1.legend(fontsize=7, framealpha=0.2, labelcolor=TEXT_CLR)
                apply_theme(fig1, ax1); fig1.tight_layout(); st.pyplot(fig1); plt.close()
            with pg2:
                fig2, ax2 = plt.subplots(figsize=(5.5, 4))
                ax2.scatter(yp_te, resid, s=20, color=ACCENT4, alpha=0.6, edgecolors="none")
                ax2.axhline(0, color=ACCENT3, linewidth=1.4, linestyle="--")
                ax2.set_xlabel("Predicted"); ax2.set_ylabel("Residuals")
                ax2.set_title(f"{mname} — Residual Plot", fontsize=9)
                apply_theme(fig2, ax2); fig2.tight_layout(); st.pyplot(fig2); plt.close()

            m_obj = fitted[mname]
            if hasattr(m_obj, "get_coefficients") and m_obj.get_coefficients() is not None:
                coeffs = m_obj.get_coefficients()
                fi_df  = pd.DataFrame({"Feature": t_feat, "Importance": np.abs(coeffs[1:])}).sort_values("Importance", ascending=True)
                fig_fi, ax_fi = plt.subplots(figsize=(7, max(2.5, len(t_feat)*0.45)))
                bc = [lc]*len(fi_df); bc[-1] = ACCENT2
                ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color=bc, edgecolor=AXES_BG)
                ax_fi.set_title(f"{mname} — Feature Importance"); apply_theme(fig_fi, ax_fi)
                fig_fi.tight_layout(); st.pyplot(fig_fi); plt.close()
            elif hasattr(m_obj, "feature_importances_"):
                fi_df = pd.DataFrame({"Feature": t_feat, "Importance": m_obj.feature_importances_}).sort_values("Importance", ascending=True)
                fig_fi, ax_fi = plt.subplots(figsize=(7, max(2.5, len(t_feat)*0.45)))
                ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color=lc, edgecolor=AXES_BG)
                ax_fi.set_title(f"{mname} — Feature Importance (Gini)"); apply_theme(fig_fi, ax_fi)
                fig_fi.tight_layout(); st.pyplot(fig_fi); plt.close()
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown(module_banner("5", "Regression Assumptions", f"Evaluated on best model: {best_name}"), unsafe_allow_html=True)
        yp_best = best_res["yp"]; resid_best = y_te - yp_best
        lin_ok  = best_res["R²"] > 0.3
        dw      = float(np.sum(np.diff(resid_best)**2) / np.sum(resid_best**2))
        ind_ok  = 1.5 < dw < 2.5
        _, hom_p = stats.pearsonr(np.abs(resid_best), yp_best); hom_ok = hom_p > 0.05
        samp = resid_best[:200] if len(resid_best) > 200 else resid_best
        _, nor_p = stats.shapiro(samp); nor_ok = nor_p > 0.05
        st.markdown(assumption_row("1. Linearity", "R² > 0.3", lin_ok, f"R² = {best_res['R²']:.4f}"), unsafe_allow_html=True)
        st.markdown(assumption_row("2. Independence", "Durbin-Watson test", ind_ok, f"DW = {dw:.4f}"), unsafe_allow_html=True)
        st.markdown(assumption_row("3. Homoscedasticity", "Constant residual variance", hom_ok, f"p = {hom_p:.4f}"), unsafe_allow_html=True)
        st.markdown(assumption_row("4. Normality of Residuals", "Shapiro-Wilk test", nor_ok, f"p = {nor_p:.4f}"), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # CLASSIFICATION POST-TRAINING
    # ════════════════════════════════════════════════════════════════════
    else:
        st.markdown(
            f'<div class="best-model-banner-cls">'
            f'<div class="bm-label">🏆 Best Classification Model</div>'
            f'<div class="bm-name">{best_name}</div>'
            f'<div class="bm-stats">'
            f'Accuracy = {best_res["Accuracy"]:.4f} &nbsp;|&nbsp; '
            f'Precision = {best_res["Precision"]:.4f} &nbsp;|&nbsp; '
            f'Recall = {best_res["Recall"]:.4f} &nbsp;|&nbsp; '
            f'F1 = {best_res["F1"]:.4f}</div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown(module_banner("3", "Model Comparison", "All 5 classification models ranked by F1 Score"), unsafe_allow_html=True)
        comp_rows = []
        for mname, ev in results.items():
            gap = ev["Acc_train"] - ev["Accuracy"]
            comp_rows.append({"Model": mname,
                               "Accuracy": round(ev["Accuracy"], 4),
                               "Precision": round(ev["Precision"], 4),
                               "Recall": round(ev["Recall"], 4),
                               "F1 Score": round(ev["F1"], 4),
                               "Train Acc": round(ev["Acc_train"], 4),
                               "Overfit Gap": round(gap, 4)})
        comp_df = pd.DataFrame(comp_rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
        comp_df.insert(0, "Rank", range(1, len(comp_df)+1))
        comp_df.insert(1, "🏆", comp_df["Model"].apply(lambda m: "★" if m == best_name else ""))
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        sorted_names = comp_df["Model"].tolist()
        bar_colors   = [MODEL_COLORS.get(m, ACCENT5) for m in sorted_names]
        edge_colors  = ["#ffd700" if m == best_name else AXES_BG for m in sorted_names]
        edge_widths  = [2.0 if m == best_name else 0.5 for m in sorted_names]

        metrics_to_plot = [("F1 Score", [results[m]["F1"] for m in sorted_names]),
                           ("Accuracy", [results[m]["Accuracy"] for m in sorted_names]),
                           ("Precision", [results[m]["Precision"] for m in sorted_names]),
                           ("Recall", [results[m]["Recall"] for m in sorted_names])]
        fig_4, axes_4 = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (metric_name, vals) in zip(axes_4, metrics_to_plot):
            bars = ax.bar(sorted_names, vals, color=bar_colors, edgecolor=edge_colors, linewidth=edge_widths)
            ax.set_ylim(0, 1.1)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, color=TEXT_CLR)
            ax.set_title(metric_name, fontsize=10); ax.set_ylabel("Score", fontsize=8)
            ax.tick_params(axis="x", rotation=15, labelsize=7)
            apply_theme(fig_4, ax)
        fig_4.suptitle("Classification Model Comparison", fontsize=11, color="#7eb8d4")
        fig_4.tight_layout(); st.pyplot(fig_4); plt.close()

        st.markdown(module_banner("4", "Per-Model Diagnostics", "Metrics · Confusion Matrix · Classification Report"), unsafe_allow_html=True)
        for mname in sorted_names:
            ev      = results[mname]
            lc      = MODEL_COLORS.get(mname, ACCENT5)
            is_best = mname == best_name
            badge_h = ('&nbsp;<span style="background:#1a0d28;color:#c890f0;font-size:0.65rem;'
                       'padding:2px 7px;border-radius:3px;">BEST</span>' if is_best else "")
            st.markdown(f'<div style="border-left:3px solid {lc};padding:4px 12px;margin:18px 0 8px;'
                        f'font-family:JetBrains Mono,monospace;font-size:0.95rem;color:{lc};font-weight:700;">'
                        f'{mname}{badge_h}</div>', unsafe_allow_html=True)
            mc = '<div class="card-grid">'
            gap = ev["Acc_train"] - ev["Accuracy"]
            for v, l in [(f"{ev['Accuracy']:.4f}", "Accuracy"), (f"{ev['Precision']:.4f}", "Precision"),
                         (f"{ev['Recall']:.4f}", "Recall"), (f"{ev['F1']:.4f}", "F1 Score"),
                         (f"{ev['Acc_train']:.4f}", "Train Acc"), (f"{gap:.4f}", "Overfit Gap")]:
                mc += metric_card(v, l, "cls-metric-card")
            mc += '</div>'
            st.markdown(mc, unsafe_allow_html=True)

            if gap > 0.15:
                st.markdown(f'<div class="overfit-warn">⚠ Possible Overfitting — gap={gap:.4f}</div>', unsafe_allow_html=True)

            cm_data    = ev["CM"]
            disp_labels = [str(le.inverse_transform([c])[0]) for c in range(len(classes))] if le else [str(c) for c in range(len(cm_data))]
            fig_cm, ax_cm = plt.subplots(figsize=(max(4, len(disp_labels)*1.1), max(3.5, len(disp_labels)*0.9)))
            sns.heatmap(cm_data, annot=True, fmt="d",
                        cmap=sns.light_palette("#a060c8", as_cmap=True),
                        xticklabels=disp_labels, yticklabels=disp_labels,
                        linewidths=0.5, linecolor=GRID_CLR,
                        annot_kws={"size": 10, "color": "#e8d8f8"},
                        cbar_kws={"shrink": 0.8}, ax=ax_cm)
            ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
            ax_cm.set_title(f"{mname} — Confusion Matrix", fontsize=10, color="#7eb8d4")
            ax_cm.tick_params(colors=TEXT_CLR, labelsize=8)
            fig_cm.patch.set_facecolor(PLOT_BG); ax_cm.set_facecolor(AXES_BG)
            for spine in ax_cm.spines.values(): spine.set_edgecolor(GRID_CLR)
            fig_cm.tight_layout(); st.pyplot(fig_cm); plt.close()

            cr = ev["CR"]
            cr_rows = []
            for label_key, metrics_val in cr.items():
                if isinstance(metrics_val, dict):
                    display_key = label_key
                    try:
                        int_key = int(label_key)
                        if le is not None:
                            display_key = str(le.inverse_transform([int_key])[0])
                    except (ValueError, TypeError):
                        pass
                    cr_rows.append({"Class": display_key,
                                    "Precision": round(metrics_val.get("precision", 0), 4),
                                    "Recall":    round(metrics_val.get("recall", 0), 4),
                                    "F1-Score":  round(metrics_val.get("f1-score", 0), 4),
                                    "Support":   int(metrics_val.get("support", 0))})
            if cr_rows:
                st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

            m_obj = fitted[mname]
            if hasattr(m_obj, "feature_importances_"):
                fi_df = pd.DataFrame({"Feature": t_feat, "Importance": m_obj.feature_importances_}).sort_values("Importance", ascending=True)
                fig_fi, ax_fi = plt.subplots(figsize=(7, max(2.5, len(t_feat)*0.45)))
                bc = [lc]*len(fi_df); bc[-1] = ACCENT2
                ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color=bc, edgecolor=AXES_BG)
                ax_fi.set_title(f"{mname} — Feature Importance (Gini)")
                apply_theme(fig_fi, ax_fi); fig_fi.tight_layout(); st.pyplot(fig_fi); plt.close()

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # MODULE 6 — PREDICTION & ADVANCED ANALYTICS
    # ════════════════════════════════════════════════════════════════════
    sub6 = "Numeric value + CI" if ml_mode == "regression" else "Class label + probability"
    st.markdown(module_banner("6", "Prediction & Advanced Analytics",
        f"{sub6} · Explanation · What-If · Similar Points · Insights · Batch"), unsafe_allow_html=True)

    pred_model_choice = st.selectbox(
        "Predict with model", ["Best Model: " + best_name] + list(fitted.keys()),
        key="pred_model_choice")
    chosen_model_name = best_name if pred_model_choice.startswith("Best Model") else pred_model_choice
    chosen_model      = fitted[chosen_model_name]
    lc_chosen = MODEL_COLORS.get(chosen_model_name, ACCENT1)
    st.markdown(f'<p style="font-size:0.76rem;color:{lc_chosen};font-family:JetBrains Mono,monospace;margin-bottom:10px;">Using: {chosen_model_name}</p>', unsafe_allow_html=True)

    user_vals     = []
    user_vals_raw = []
    n_inp    = min(3, len(t_feat))
    inp_cols = st.columns(n_inp)
    for i, feat in enumerate(t_feat):
        with inp_cols[i % n_inp]:
            if feat in feat_enc:
                fe       = feat_enc[feat]
                cat_opts = list(fe.classes_)
                sel      = st.selectbox(f"{feat} 🏷️", options=cat_opts, key=f"pred_{feat}")
                user_vals.append(float(fe.transform([sel])[0]))
                user_vals_raw.append(sel)
            else:
                lo_v   = float(dsnap[feat].min())
                hi_v   = float(dsnap[feat].max())
                mean_v = float(dsnap[feat].mean())
                v = st.number_input(feat, min_value=lo_v, max_value=hi_v,
                                    value=mean_v, step=max((hi_v - lo_v) / 200, 1e-6),
                                    format="%.4f", key=f"pred_{feat}")
                user_vals.append(v)
                user_vals_raw.append(v)

    pc, _ = st.columns([1, 3])
    with pc:
        do_pred = st.button("Predict", use_container_width=True)

    def _std(arr):
        return (arr - xmu) / xsg if (a_std and xmu is not None) else arr

    if do_pred:
        arr     = np.array(user_vals).reshape(1, -1)
        arr_std = _std(arr)
        if ml_mode == "regression":
            pred_val = chosen_model.predict(arr_std)[0]
            st.session_state["_last_pred"]     = pred_val
            st.session_state["_last_pred_cls"] = None
        else:
            pred_cls = chosen_model.predict(arr_std)[0]
            pred_prob = None
            if hasattr(chosen_model, "predict_proba"):
                proba_arr = chosen_model.predict_proba(arr_std)[0]
                pred_prob = float(proba_arr.max())
            st.session_state["_last_pred"]     = None
            st.session_state["_last_pred_cls"] = (pred_cls, pred_prob)
        st.session_state["_last_user_vals"]      = user_vals
        st.session_state["_last_user_vals_raw"]  = user_vals_raw
        st.session_state["_last_arr_std"]        = arr_std
        st.session_state["_last_chosen"]         = chosen_model_name

    last_pred     = st.session_state.get("_last_pred")
    last_pred_cls = st.session_state.get("_last_pred_cls")
    last_shown    = (last_pred is not None) or (last_pred_cls is not None)

    if last_shown:
        user_vals_snap     = st.session_state["_last_user_vals"]
        user_vals_raw_snap = st.session_state.get("_last_user_vals_raw", user_vals_snap)
        arr_std_snap       = st.session_state["_last_arr_std"]
        snap_model_nm      = st.session_state["_last_chosen"]
        snap_model         = fitted[snap_model_nm]

        if ml_mode == "regression":
            pred      = last_pred
            snap_rmse = results[snap_model_nm]["RMSE"]
            ci_lo = pred - 1.96 * snap_rmse
            ci_hi = pred + 1.96 * snap_rmse
            st.markdown(f'<div class="pred-box">'
                        f'<div class="pred-label">Predicted {t_tgt}</div>'
                        f'<div class="pred-val">{pred:.4f}</div>'
                        f'<div class="pred-ci">95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]</div>'
                        f'<div class="pred-model">Model: {snap_model_nm}</div>'
                        f'</div>', unsafe_allow_html=True)
        else:
            pred_cls_enc, pred_prob = last_pred_cls
            pred_label = str(le.inverse_transform([pred_cls_enc])[0]) if le else str(pred_cls_enc)
            prob_str   = f"Confidence: {pred_prob*100:.1f}%" if pred_prob is not None else ""
            st.markdown(f'<div class="pred-box-cls">'
                        f'<div class="pred-label">Predicted Class — {t_tgt}</div>'
                        f'<div class="pred-val">{pred_label}</div>'
                        f'<div class="pred-prob">{prob_str}</div>'
                        f'<div class="pred-model">Model: {snap_model_nm}</div>'
                        f'</div>', unsafe_allow_html=True)

            if hasattr(snap_model, "predict_proba"):
                proba_arr    = snap_model.predict_proba(arr_std_snap)[0]
                proba_labels = [str(le.inverse_transform([c])[0]) for c in range(len(proba_arr))] if le else [str(c) for c in range(len(proba_arr))]
                fig_pb, ax_pb = plt.subplots(figsize=(max(5, len(proba_labels)*0.9), 3))
                ax_pb.bar(proba_labels, proba_arr,
                          color=[ACCENT5 if i == proba_arr.argmax() else "#1a2a3a" for i in range(len(proba_arr))],
                          edgecolor=ACCENT5, linewidth=0.8)
                for i, val in enumerate(proba_arr):
                    ax_pb.text(i, val+0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_CLR)
                ax_pb.set_ylabel("Probability"); ax_pb.set_ylim(0, 1.15)
                ax_pb.set_title("Class Probability Distribution")
                apply_theme(fig_pb, ax_pb); fig_pb.tight_layout(); st.pyplot(fig_pb); plt.close()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="module-banner" style="border-left-color:#a060c8;">'
                    '<div class="mod-label" style="color:#a060c8;">⬡ Analytics · Feature 1</div>'
                    '<div class="mod-title">Prediction Explanation</div></div>', unsafe_allow_html=True)

        contribs = {}
        if hasattr(snap_model, "get_coefficients") and snap_model.get_coefficients() is not None:
            coeffs = snap_model.get_coefficients()
            for i, feat in enumerate(t_feat):
                contribs[feat] = float(coeffs[i+1]) * float(user_vals_snap[i])
        elif hasattr(snap_model, "W") and snap_model.W is not None:
            W = snap_model.W
            if ml_mode == "classification" and last_pred_cls is not None:
                pred_cls_enc2 = last_pred_cls[0]
                for i, feat in enumerate(t_feat):
                    contribs[feat] = float(W[i, pred_cls_enc2]) * float(user_vals_snap[i])
            else:
                for i, feat in enumerate(t_feat):
                    contribs[feat] = float(np.abs(W[i]).max()) * float(user_vals_snap[i])
        elif hasattr(snap_model, "feature_importances_"):
            fi   = snap_model.feature_importances_
            base = 0.0
            diff = (last_pred - base) if ml_mode == "regression" and last_pred is not None else 1.0
            for i, feat in enumerate(t_feat):
                contribs[feat] = fi[i] * diff
        else:
            for i, feat in enumerate(t_feat):
                contribs[feat] = float(user_vals_snap[i])

        contrib_rows = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        tbl_html = '<div class="summary-box"><div class="summary-title">⬡ Feature Contributions</div>'
        for feat, contrib in contrib_rows:
            sc = "#34d490" if contrib >= 0 else "#e07070"
            si = "📈" if contrib >= 0 else "📉"
            ss = f"+{contrib:.4f}" if contrib >= 0 else f"{contrib:.4f}"
            tbl_html += (f'<div class="summary-item"><span class="summary-icon">{si}</span>'
                         f'<span class="summary-key">{feat}</span>'
                         f'<span class="summary-val" style="color:{sc};">{ss}</span></div>')
        tbl_html += '</div>'
        st.markdown(tbl_html, unsafe_allow_html=True)

        feat_names  = [f for f, _ in contrib_rows]
        feat_values = [v for _, v in contrib_rows]
        fig_ex, ax_ex = plt.subplots(figsize=(8, max(2.8, len(feat_names)*0.5)))
        ax_ex.barh(feat_names[::-1], feat_values[::-1],
                   color=[ACCENT2 if v >= 0 else ACCENT3 for v in feat_values[::-1]], edgecolor=AXES_BG)
        ax_ex.axvline(0, color=TEXT_CLR, linewidth=0.8)
        ax_ex.set_title("Feature Contributions")
        apply_theme(fig_ex, ax_ex); fig_ex.tight_layout(); st.pyplot(fig_ex); plt.close()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="module-banner" style="border-left-color:#e0a844;">'
                    '<div class="mod-label" style="color:#e0a844;">⬡ Analytics · Feature 2</div>'
                    '<div class="mod-title">What-If Analysis</div></div>', unsafe_allow_html=True)

        wi_vals = []
        wi_cols = st.columns(min(3, len(t_feat)))
        for i, feat in enumerate(t_feat):
            with wi_cols[i % min(3, len(t_feat))]:
                if feat in feat_enc:
                    fe       = feat_enc[feat]
                    cat_opts = list(fe.classes_)
                    cur_raw  = user_vals_raw_snap[i] if i < len(user_vals_raw_snap) else cat_opts[0]
                    cur_sel  = cur_raw if cur_raw in cat_opts else cat_opts[0]
                    wi_sel   = st.selectbox(f"{feat} 🏷️", options=cat_opts,
                                            index=cat_opts.index(cur_sel), key=f"wi_{feat}")
                    wi_vals.append(float(fe.transform([wi_sel])[0]))
                else:
                    lo_v = float(dsnap[feat].min()); hi_v = float(dsnap[feat].max())
                    wv = st.slider(feat, min_value=lo_v, max_value=hi_v,
                                   value=float(user_vals_snap[i]),
                                   step=max((hi_v - lo_v) / 100, 1e-6), key=f"wi_{feat}")
                    wi_vals.append(wv)

        wi_arr     = np.array(wi_vals).reshape(1, -1)
        wi_arr_std = _std(wi_arr)

        if ml_mode == "regression":
            wi_pred   = snap_model.predict(wi_arr_std)[0]
            delta     = wi_pred - last_pred
            delta_pct = (delta / abs(last_pred) * 100) if last_pred != 0 else 0.0
            dc = "#34d490" if delta >= 0 else "#e07070"
            di = "▲" if delta >= 0 else "▼"
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-card"><div class="val" style="color:#2d8fcb;">{last_pred:.4f}</div><div class="lbl">Original</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="val" style="color:#22a878;">{wi_pred:.4f}</div><div class="lbl">Modified</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="val" style="color:{dc};">{di} {abs(delta):.4f}</div><div class="lbl">Change ({delta_pct:+.2f}%)</div></div>', unsafe_allow_html=True)
        else:
            wi_cls   = snap_model.predict(wi_arr_std)[0]
            wi_label   = str(le.inverse_transform([wi_cls])[0]) if le else str(wi_cls)
            orig_label = str(le.inverse_transform([last_pred_cls[0]])[0]) if le else str(last_pred_cls[0])
            changed    = wi_label != orig_label
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="cls-metric-card"><div class="val">{orig_label}</div><div class="lbl">Original Class</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="cls-metric-card"><div class="val">{wi_label}</div><div class="lbl">Modified Class</div></div>', unsafe_allow_html=True)
            with c3:
                ch_color = "#e07070" if changed else "#34d490"
                st.markdown(f'<div class="cls-metric-card"><div class="val" style="color:{ch_color};font-size:1rem;">{"CHANGED" if changed else "SAME"}</div><div class="lbl">Change Status</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="module-banner" style="border-left-color:#22a878;">'
                    '<div class="mod-label" style="color:#22a878;">⬡ Analytics · Feature 5</div>'
                    '<div class="mod-title">Similar Historical Data Points</div></div>', unsafe_allow_html=True)

        X_sim_parts = []
        for feat in t_feat:
            col = dsnap[feat]
            if feat in feat_enc:
                fe       = feat_enc[feat]
                mode_val = fe.classes_[0]
                filled   = col.fillna(mode_val).astype(str)
                safe     = filled.apply(lambda v: v if v in fe.classes_ else mode_val)
                X_sim_parts.append(fe.transform(safe).astype(float).reshape(-1, 1))
            else:
                X_sim_parts.append(col.fillna(col.median()).values.astype(float).reshape(-1, 1))
        X_full_enc = np.hstack(X_sim_parts)
        query_enc  = np.array(user_vals_snap, dtype=float)
        feat_std_v = X_full_enc.std(axis=0); feat_std_v[feat_std_v == 0] = 1.0
        dists      = np.linalg.norm(X_full_enc / feat_std_v - query_enc / feat_std_v, axis=1)
        top5_idx   = np.argsort(dists)[:5]
        sim_df     = dsnap[t_feat + [t_tgt]].iloc[top5_idx].copy().reset_index(drop=True)
        sim_df.insert(0, "Rank", range(1, 6))
        sim_df.insert(1, "Distance", np.round(dists[top5_idx], 4))
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="module-banner" style="border-left-color:#2d8fcb;">'
                    '<div class="mod-label" style="color:#2d8fcb;">⬡ Analytics · Feature 9</div>'
                    '<div class="mod-title">Automatic Prediction Insights</div></div>', unsafe_allow_html=True)

        if contribs:
            max_abs = max(abs(v) for v in contribs.values()) or 1.0
            for feat, contrib in sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True):
                ratio     = abs(contrib) / max_abs
                strength  = "significantly" if ratio >= 0.6 else ("moderately" if ratio >= 0.25 else "slightly")
                direction = "increases" if contrib >= 0 else "decreases"
                badge_cls = "insight-pos" if contrib >= 0 else "insight-neg"
                icon      = "📈" if contrib >= 0 else "📉"
                st.markdown(f'<div class="{badge_cls}">{icon} <strong>{feat}</strong> {strength} '
                            f'the predicted {t_tgt} ({direction} by ~{abs(contrib):.4f})</div>',
                            unsafe_allow_html=True)
            if ml_mode == "regression":
                r2_val = results[snap_model_nm]["R²"]
                conf   = "high confidence" if r2_val >= 0.85 else ("moderate confidence" if r2_val >= 0.55 else "low confidence")
                cc     = "insight-pos" if r2_val >= 0.85 else ("insight-neu" if r2_val >= 0.55 else "insight-neg")
                st.markdown(f'<div class="{cc}">🎯 Model <strong>{snap_model_nm}</strong> predicts with {conf} (R² = {r2_val:.4f})</div>', unsafe_allow_html=True)
            else:
                f1_val = results[snap_model_nm]["F1"]
                conf   = "high confidence" if f1_val >= 0.85 else ("moderate confidence" if f1_val >= 0.60 else "low confidence")
                cc     = "insight-pos" if f1_val >= 0.85 else ("insight-neu" if f1_val >= 0.60 else "insight-neg")
                st.markdown(f'<div class="{cc}">🎯 Model <strong>{snap_model_nm}</strong> classifies with {conf} (F1 = {f1_val:.4f})</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# MODULE 7 — UNSUPERVISED LEARNING & PATTERN DISCOVERY
# ════════════════════════════════════════════════════════════════════════
st.markdown(module_banner(
    "7", "Unsupervised Learning & Pattern Discovery",
    "Manual Algorithms · K-Means Clustering · Apriori Association Rules"
), unsafe_allow_html=True)

if len(numeric_cols) < 2:
    st.warning("⚠ Module 7 requires at least 2 numeric columns for unsupervised learning.")
else:
    # ── Algorithm selector ────────────────────────────────────────────────
    UNSUP_ALGOS = [
        "K-Means Clustering",
        "Apriori — Association Rules",
    ]
    unsup_algo = st.selectbox(
        "Select Unsupervised Algorithm",
        UNSUP_ALGOS,
        key="unsup_algo"
    )

    # ── Feature selection ─────────────────────────────────────────────────
    st.markdown("#### Feature Selection for Unsupervised Learning")
    default_unsup_feats = numeric_cols[:min(5, len(numeric_cols))]
    unsup_feats = st.multiselect(
        "Select numeric features (min 2)",
        numeric_cols,
        default=default_unsup_feats,
        key="unsup_feats"
    )

    if len(unsup_feats) < 2:
        st.warning("Please select at least 2 numeric features.")
    else:
        # Build the unsupervised feature matrix (impute + standardize)
        X_unsup_raw = np.hstack([
            data[f].fillna(data[f].median()).values.astype(float).reshape(-1, 1)
            for f in unsup_feats
        ])
        # Always standardize for unsupervised learning
        unsup_mu = X_unsup_raw.mean(axis=0)
        unsup_sg = X_unsup_raw.std(axis=0); unsup_sg[unsup_sg == 0] = 1.0
        X_unsup  = (X_unsup_raw - unsup_mu) / unsup_sg

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        # K-MEANS
        # ══════════════════════════════════════════════════════════════════
        if unsup_algo == "K-Means Clustering":
            st.markdown('<div class="module-banner" style="border-left-color:#22e8a0;">'
                        '<div class="mod-label" style="color:#22e8a0;">⬡ Algorithm · K-Means</div>'
                        '<div class="mod-title">Manual K-Means Clustering</div>'
                        '<div class="mod-sub">Euclidean distance · centroid update · convergence detection</div>'
                        '</div>', unsafe_allow_html=True)

            kp1, kp2, kp3 = st.columns(3)
            with kp1:
                km_k       = st.slider("Number of Clusters (K)", 2, 10, 3, 1, key="km_k")
            with kp2:
                km_maxiter = st.slider("Max Iterations", 50, 500, 300, 50, key="km_maxiter")
            with kp3:
                km_seed    = st.number_input("Random Seed", min_value=0, max_value=999, value=42, step=1, key="km_seed")

            # Elbow method preview
            show_elbow = st.checkbox("Show Elbow Plot (K=2..8)", value=True, key="km_elbow")

            run_km = st.button("▶  Run K-Means", use_container_width=True, key="run_km")

            if run_km:
                with st.spinner("Running Manual K-Means…"):
                    km_model = ManualKMeans(k=km_k, max_iter=km_maxiter, random_state=int(km_seed))
                    km_model.fit(X_unsup)
                    km_labels    = km_model.labels_
                    km_centroids = km_model.centroids_
                    km_sil       = silhouette_score_manual(X_unsup, km_labels)

                    st.session_state["_km_labels"]    = km_labels
                    st.session_state["_km_centroids"] = km_centroids
                    st.session_state["_km_model"]     = km_model
                    st.session_state["_km_sil"]       = km_sil
                    st.session_state["_km_k"]         = km_k

                # ── Metrics banner ────────────────────────────────────────
                mc = '<div class="card-grid">'
                for v, l in [(str(km_k), "Clusters"), (str(km_model.n_iter_), "Iterations"),
                             (f"{km_model.inertia_:.2f}", "Inertia (WCSS)"),
                             (f"{km_sil:.4f}" if not np.isnan(km_sil) else "N/A", "Silhouette Score")]:
                    mc += metric_card(v, l, "unsup-metric-card")
                mc += '</div>'
                st.markdown(mc, unsafe_allow_html=True)

                # ── Cluster size summary ──────────────────────────────────
                uniq, cnts = np.unique(km_labels, return_counts=True)
                size_html = '<div class="summary-box"><div class="summary-title">⬡ Cluster Size Summary</div>'
                for uid, cnt in zip(uniq, cnts):
                    pct = cnt / len(km_labels) * 100
                    clr = CLUSTER_PALETTE[uid % len(CLUSTER_PALETTE)]
                    size_html += (f'<div class="summary-item">'
                                  f'<span class="summary-icon" style="color:{clr};">●</span>'
                                  f'<span class="summary-key">Cluster {uid}</span>'
                                  f'<span class="summary-val">{cnt} points ({pct:.1f}%)</span></div>')
                size_html += '</div>'
                st.markdown(size_html, unsafe_allow_html=True)

                # ── Elbow plot ────────────────────────────────────────────
                if show_elbow:
                    elbow_ks = range(2, min(9, len(X_unsup)))
                    inertias = []
                    for ek in elbow_ks:
                        em = ManualKMeans(k=ek, max_iter=200, random_state=42)
                        em.fit(X_unsup)
                        inertias.append(em.inertia_)
                    fig_elb, ax_elb = plt.subplots(figsize=(7, 3.5))
                    ax_elb.plot(list(elbow_ks), inertias, color=ACCENT7, linewidth=2, marker="o",
                                markersize=7, markerfacecolor=ACCENT4, markeredgecolor=AXES_BG)
                    ax_elb.axvline(km_k, color=ACCENT3, linewidth=1.5, linestyle="--", label=f"K={km_k}")
                    ax_elb.set_xlabel("Number of Clusters (K)"); ax_elb.set_ylabel("Inertia (WCSS)")
                    ax_elb.set_title("Elbow Method — Optimal K Selection")
                    ax_elb.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT_CLR)
                    apply_theme(fig_elb, ax_elb); fig_elb.tight_layout()
                    st.pyplot(fig_elb); plt.close()

                # ── Scatter plot (PC1 vs PC2 if >2 features) ─────────────
                if X_unsup.shape[1] == 2:
                    plot_X = X_unsup
                    ax_labels = [unsup_feats[0], unsup_feats[1]]
                else:
                    pca_viz = ManualPCA(n_components=2)
                    plot_X = pca_viz.fit_transform(X_unsup)
                    var_r  = pca_viz.explained_variance_ratio_
                    ax_labels = [f"PC1 ({var_r[0]*100:.1f}% var)", f"PC2 ({var_r[1]*100:.1f}% var)"]

                fig_km, ax_km = plt.subplots(figsize=(8, 5))
                for cid in range(km_k):
                    mask = km_labels == cid
                    clr  = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
                    ax_km.scatter(plot_X[mask, 0], plot_X[mask, 1],
                                  s=30, color=clr, alpha=0.7, edgecolors="none",
                                  label=f"Cluster {cid} (n={mask.sum()})")
                # Plot centroids — use "P" (filled plus) instead of unicode star
                if X_unsup.shape[1] > 2:
                    c_proj = pca_viz.transform(km_centroids)
                else:
                    c_proj = km_centroids
                ax_km.scatter(c_proj[:, 0], c_proj[:, 1], s=220, marker="P",
                              color=ACCENT4, edgecolors="#fff", linewidth=0.8, zorder=5, label="Centroids")
                ax_km.set_xlabel(ax_labels[0]); ax_km.set_ylabel(ax_labels[1])
                ax_km.set_title(f"K-Means Clustering (K={km_k}) — Scatter Plot")
                ax_km.legend(fontsize=7, framealpha=0.15, labelcolor=TEXT_CLR, loc="best")
                apply_theme(fig_km, ax_km); fig_km.tight_layout()
                st.pyplot(fig_km); plt.close()

                # ── Cluster profile table (mean values per cluster) ───────
                st.markdown("#### Cluster Profiles — Mean Feature Values per Cluster")
                profile_data = data[unsup_feats].copy()
                profile_data["Cluster"] = km_labels
                cluster_profile = profile_data.groupby("Cluster")[unsup_feats].mean().round(3)
                st.dataframe(cluster_profile, use_container_width=True)

                # ── Insights ──────────────────────────────────────────────
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### K-Means Insights")
                if km_sil > 0.5:
                    st.markdown(f'<div class="insight-unsup">✅ Strong cluster structure — Silhouette Score = {km_sil:.4f} (> 0.5)</div>', unsafe_allow_html=True)
                elif km_sil > 0.25:
                    st.markdown(f'<div class="insight-neu">⚠ Moderate cluster structure — Silhouette Score = {km_sil:.4f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="insight-neg">⚠ Weak cluster structure — Silhouette Score = {km_sil:.4f}. Consider adjusting K.</div>', unsafe_allow_html=True)

                uniq_c, cnt_c = np.unique(km_labels, return_counts=True)
                dominant = uniq_c[cnt_c.argmax()]
                st.markdown(f'<div class="insight-unsup">🔵 Largest cluster: <strong>Cluster {dominant}</strong> with {cnt_c.max()} points ({cnt_c.max()/len(km_labels)*100:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-neu">⚙ Converged in {km_model.n_iter_} iterations out of max {km_maxiter}</div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        # APRIORI — ASSOCIATION RULES
        # ══════════════════════════════════════════════════════════════════
        if unsup_algo == "Apriori — Association Rules":
            st.markdown('<div class="module-banner" style="border-left-color:#d05090;">'
                        '<div class="mod-label" style="color:#d05090;">⬡ Algorithm · Apriori</div>'
                        '<div class="mod-title">Manual Apriori — Association Rule Mining</div>'
                        '<div class="mod-sub">Frequent itemsets · confidence · lift · market basket analysis</div>'
                        '</div>', unsafe_allow_html=True)

            st.info("💡 Apriori mines co-occurrence rules from data. Features are discretized into bins (Low/Mid/High) before mining.")

            ap1, ap2, ap3 = st.columns(3)
            with ap1:
                ap_min_sup  = st.slider("Min Support", 0.05, 0.8, 0.2, 0.05, key="ap_sup")
            with ap2:
                ap_min_conf = st.slider("Min Confidence", 0.3, 1.0, 0.5, 0.05, key="ap_conf")
            with ap3:
                ap_bins     = st.slider("Discretization Bins", 2, 5, 3, 1, key="ap_bins")

            ap_max_rows = st.slider("Max Rows to Mine", 50, min(1000, len(data)), min(500, len(data)), 50, key="ap_maxrows")
            ap_feats_sel = st.multiselect(
                "Features to include in Association Mining",
                unsup_feats,
                default=unsup_feats[:min(4, len(unsup_feats))],
                key="ap_feats"
            )

            bin_labels_map = {2: ["Low", "High"],
                              3: ["Low", "Mid", "High"],
                              4: ["Low", "Med-Low", "Med-High", "High"],
                              5: ["Very Low", "Low", "Mid", "High", "Very High"]}
            blabels = bin_labels_map.get(ap_bins, [f"B{i}" for i in range(ap_bins)])

            run_ap = st.button("▶  Run Apriori", use_container_width=True, key="run_ap")

            if run_ap:
                if len(ap_feats_sel) < 2:
                    st.warning("Select at least 2 features for Apriori.")
                else:
                    with st.spinner("Building transactions and running Manual Apriori…"):
                        # ── Discretize numeric features ───────────────────
                        sub_data = data[ap_feats_sel].head(ap_max_rows).copy()
                        disc_data = pd.DataFrame()
                        for feat in ap_feats_sel:
                            col = sub_data[feat].fillna(sub_data[feat].median())
                            try:
                                disc_data[feat] = pd.cut(col, bins=ap_bins, labels=blabels, duplicates="drop")
                            except Exception:
                                disc_data[feat] = col.astype(str)

                        # ── Build transaction list ─────────────────────────
                        transactions = []
                        for _, row in disc_data.iterrows():
                            t = frozenset(f"{feat}={val}" for feat, val in row.items() if pd.notna(val))
                            if t:
                                transactions.append(t)

                        if not transactions:
                            st.error("No valid transactions generated.")
                        else:
                            apriori_model = ManualApriori(min_support=ap_min_sup, min_confidence=ap_min_conf)
                            apriori_model.fit(transactions)

                    # ── Metrics ───────────────────────────────────────────
                    mc = '<div class="card-grid">'
                    for v, l in [(str(len(transactions)), "Transactions"),
                                 (str(len(apriori_model.frequent_itemsets_)), "Frequent Itemsets"),
                                 (str(len(apriori_model.rules_)), "Association Rules"),
                                 (str(ap_bins), "Bins")]:
                        mc += metric_card(v, l, "unsup-metric-card")
                    mc += '</div>'
                    st.markdown(mc, unsafe_allow_html=True)

                    # ── Frequent itemsets ─────────────────────────────────
                    if apriori_model.frequent_itemsets_:
                        st.markdown("#### Frequent Itemsets (sorted by support)")
                        fi_rows = sorted(
                            [{"Itemset": " & ".join(sorted(fs)), "Size": len(fs), "Support": round(sup, 4)}
                             for fs, sup in apriori_model.frequent_itemsets_.items()],
                            key=lambda x: -x["Support"]
                        )
                        st.dataframe(pd.DataFrame(fi_rows).head(20), use_container_width=True, hide_index=True)
                    else:
                        st.warning("No frequent itemsets found. Try lowering min_support.")

                    # ── Association rules ─────────────────────────────────
                    if apriori_model.rules_:
                        st.markdown(f'#### Association Rules ({len(apriori_model.rules_)} found, sorted by lift)')

                        # Rules table
                        rules_rows = []
                        for r in apriori_model.rules_[:30]:
                            ant_str = " & ".join(sorted(r["antecedent"]))
                            con_str = " & ".join(sorted(r["consequent"]))
                            rules_rows.append({
                                "Antecedent → Consequent": f"{ant_str}  →  {con_str}",
                                "Support": r["support"],
                                "Confidence": r["confidence"],
                                "Lift": r["lift"],
                            })
                        rules_df = pd.DataFrame(rules_rows)
                        st.dataframe(rules_df, use_container_width=True, hide_index=True)

                        # ── Top 5 rules as styled cards ───────────────────
                        st.markdown("#### Top 5 Rules by Lift")
                        for r in apriori_model.rules_[:5]:
                            ant_str = " & ".join(sorted(r["antecedent"]))
                            con_str = " & ".join(sorted(r["consequent"]))
                            lift_clr = "#22e8a0" if r["lift"] > 1.5 else ("#e0a844" if r["lift"] > 1.0 else "#e07070")
                            st.markdown(
                                f'<div class="apriori-rule-row">'
                                f'<div class="rule-text">{ant_str}  →  {con_str}</div>'
                                f'<div class="rule-stats">'
                                f'Support: <strong>{r["support"]:.4f}</strong> &nbsp;|&nbsp; '
                                f'Confidence: <strong>{r["confidence"]:.4f}</strong> &nbsp;|&nbsp; '
                                f'Lift: <span class="rule-lift" style="color:{lift_clr};">{r["lift"]:.4f}</span>'
                                f'</div></div>',
                                unsafe_allow_html=True
                            )

                        # ── Support vs Confidence scatter plot ────────────
                        if len(rules_rows) > 1:
                            sups  = [r["support"] for r in apriori_model.rules_[:50]]
                            confs = [r["confidence"] for r in apriori_model.rules_[:50]]
                            lifts = [r["lift"] for r in apriori_model.rules_[:50]]
                            fig_ar, ax_ar = plt.subplots(figsize=(8, 5))
                            sc = ax_ar.scatter(sups, confs, c=lifts,
                                               cmap="YlGn", s=60,
                                               edgecolors=AXES_BG, linewidth=0.5, alpha=0.85)
                            plt.colorbar(sc, ax=ax_ar, label="Lift").ax.yaxis.label.set_color(TEXT_CLR)
                            ax_ar.set_xlabel("Support"); ax_ar.set_ylabel("Confidence")
                            ax_ar.set_title("Association Rules — Support vs. Confidence (colored by Lift)")
                            apply_theme(fig_ar, ax_ar); fig_ar.tight_layout()
                            st.pyplot(fig_ar); plt.close()

                        # ── Insights ──────────────────────────────────────
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        st.markdown("#### Apriori Insights")
                        best_rule = apriori_model.rules_[0]
                        ant_b = " & ".join(sorted(best_rule["antecedent"]))
                        con_b = " & ".join(sorted(best_rule["consequent"]))
                        st.markdown(f'<div class="insight-unsup">🏆 Best rule by lift: <strong>{ant_b} → {con_b}</strong> (lift = {best_rule["lift"]:.4f})</div>', unsafe_allow_html=True)
                        high_lift = [r for r in apriori_model.rules_ if r["lift"] > 1.5]
                        st.markdown(f'<div class="insight-unsup">📊 {len(high_lift)} rules have lift > 1.5 (strong associations)</div>', unsafe_allow_html=True)
                        avg_conf = np.mean([r["confidence"] for r in apriori_model.rules_])
                        st.markdown(f'<div class="insight-neu">📈 Average rule confidence: {avg_conf:.4f}</div>', unsafe_allow_html=True)
                        if len(apriori_model.rules_) == 0:
                            st.markdown('<div class="insight-neg">⚠ No rules found — try lowering min_support or min_confidence</div>', unsafe_allow_html=True)
                        elif len(apriori_model.rules_) > 20:
                            st.markdown('<div class="insight-neu">💡 Many rules found — increase min_support or min_confidence to focus on stronger patterns</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No association rules found. Try lowering min_confidence or min_support.")

# ════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;padding:14px 0;font-family:JetBrains Mono,monospace;'
            'font-size:0.68rem;color:#1d3a54;letter-spacing:0.1em;">'
            'INTELLIGENT DATA ANALYSIS &amp; ML SYSTEM v7 · AUTO REGRESSION / CLASSIFICATION · '
            'UNSUPERVISED: K-MEANS · APRIORI ASSOCIATION RULES · '
            'SUPPORTS NUMERIC &amp; CATEGORICAL FEATURES/TARGETS'
            '</div>', unsafe_allow_html=True)