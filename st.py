# ════════════════════════════════════════════════════════════════════════
#  Intelligent Data Analysis & ML System  —  v10 + Landing UI
#  All original functionality preserved.
#  NEW: Interactive landing page shown before dataset upload.
# ════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import warnings
import io
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

np.random.seed(42)

st.set_page_config(
    page_title="Intelligent ML System v10",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════ GLOBAL CSS (Original + Landing) ═══════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');
:root{--bg:#060d18;--surface:#0b1628;--border:#0f2236;--text:#7ea8c4;--bright:#a8ccde;--a1:#2d8fcb;--a2:#22a878;--a3:#c05555;--a4:#e0a844;--module:#0d1e30;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background-color:var(--bg);color:var(--text);}
h1,h2,h3{font-family:'JetBrains Mono',monospace;color:var(--bright);letter-spacing:-0.02em;}
.hero-title{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:700;color:#c4dff0;letter-spacing:-0.03em;margin-bottom:2px;}
.hero-sub{font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:var(--a1);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1.4rem;}
.module-banner{background:linear-gradient(90deg,var(--module) 0%,#0a1c30 100%);border-left:3px solid var(--a1);border-radius:6px;padding:14px 20px;margin:28px 0 18px 0;}
.module-banner .mod-label{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--a1);letter-spacing:0.16em;text-transform:uppercase;margin-bottom:2px;}
.module-banner .mod-title{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;color:var(--bright);}
.module-banner .mod-sub{font-size:0.73rem;color:var(--text);margin-top:3px;}
.mode-reg{background:linear-gradient(90deg,#081c30 0%,#061420 100%);border:1px solid #1a4a6a;border-left:5px solid #2d8fcb;border-radius:8px;padding:16px 22px;margin:14px 0;}
.mode-reg .mode-icon{font-size:1.6rem;float:left;margin-right:14px;margin-top:2px;}
.mode-reg .mode-tag{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#2d8fcb;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:3px;}
.mode-reg .mode-name{font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:700;color:#6dc8f0;}
.mode-reg .mode-desc{font-size:0.74rem;color:var(--text);margin-top:4px;}
.mode-cls{background:linear-gradient(90deg,#1a0d28 0%,#110820 100%);border:1px solid #4a2a6a;border-left:5px solid #a060c8;border-radius:8px;padding:16px 22px;margin:14px 0;}
.mode-cls .mode-icon{font-size:1.6rem;float:left;margin-right:14px;margin-top:2px;}
.mode-cls .mode-tag{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#a060c8;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:3px;}
.mode-cls .mode-name{font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:700;color:#c890f0;}
.mode-cls .mode-desc{font-size:0.74rem;color:var(--text);margin-top:4px;}
.card-grid{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px;}
.stat-card{flex:1;min-width:110px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px 12px;text-align:center;}
.stat-card .val{font-family:'JetBrains Mono',monospace;font-size:1.45rem;font-weight:700;color:var(--a1);line-height:1;}
.stat-card .lbl{font-size:0.69rem;color:var(--text);letter-spacing:0.08em;text-transform:uppercase;margin-top:5px;}
.metric-card{flex:1;min-width:110px;background:linear-gradient(135deg,#0b1c30 0%,#091625 100%);border:1px solid #0f2e4a;border-radius:8px;padding:14px 16px 12px;text-align:center;}
.metric-card .val{font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:var(--a2);line-height:1;}
.metric-card .lbl{font-size:0.68rem;color:var(--text);letter-spacing:0.08em;text-transform:uppercase;margin-top:5px;}
.cls-metric-card{flex:1;min-width:110px;background:linear-gradient(135deg,#180d2a 0%,#0f0820 100%);border:1px solid #3a1a5a;border-radius:8px;padding:14px 16px 12px;text-align:center;}
.cls-metric-card .val{font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:#c890f0;line-height:1;}
.cls-metric-card .lbl{font-size:0.68rem;color:var(--text);letter-spacing:0.08em;text-transform:uppercase;margin-top:5px;}
.unsup-metric-card{flex:1;min-width:110px;background:linear-gradient(135deg,#0a1e1a 0%,#061510 100%);border:1px solid #1a4a38;border-radius:8px;padding:14px 16px 12px;text-align:center;}
.unsup-metric-card .val{font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:#22e8a0;line-height:1;}
.unsup-metric-card .lbl{font-size:0.68rem;color:var(--text);letter-spacing:0.08em;text-transform:uppercase;margin-top:5px;}
.assumption-row{display:flex;align-items:flex-start;gap:14px;background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:12px 16px;margin-bottom:9px;}
.badge{font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700;padding:3px 9px;border-radius:4px;letter-spacing:0.12em;white-space:nowrap;margin-top:2px;}
.badge-pass{background:#0e2e20;color:#22c97a;border:1px solid #1a5c38;}
.badge-fail{background:#2e0e0e;color:#d96060;border:1px solid #5c2020;}
.assumption-title{font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:var(--bright);margin-bottom:2px;}
.assumption-desc{font-size:0.75rem;color:var(--text);}
.assumption-extra{font-size:0.73rem;color:var(--a4);margin-top:3px;}
.insight-pos{background:#0a2318;border:1px solid #1a4a30;color:#34d490;border-radius:5px;padding:6px 12px;font-size:0.76rem;margin:4px 0;display:inline-block;}
.insight-neg{background:#230a0a;border:1px solid #4a1a1a;color:#e07070;border-radius:5px;padding:6px 12px;font-size:0.76rem;margin:4px 0;display:inline-block;}
.insight-neu{background:#111e2a;border:1px solid #1d3347;color:var(--text);border-radius:5px;padding:6px 12px;font-size:0.76rem;margin:4px 0;display:inline-block;}
.insight-cls{background:#1a0d28;border:1px solid #4a2a6a;color:#c890f0;border-radius:5px;padding:6px 12px;font-size:0.76rem;margin:4px 0;display:inline-block;}
.insight-unsup{background:#0a201a;border:1px solid #1a5038;color:#22e8a0;border-radius:5px;padding:6px 12px;font-size:0.76rem;margin:4px 0;display:inline-block;}
.pred-box{background:linear-gradient(135deg,#0c1e32 0%,#091526 100%);border:1px solid #1a4060;border-radius:9px;padding:18px 22px;text-align:center;margin-top:12px;}
.pred-box .pred-val{font-family:'JetBrains Mono',monospace;font-size:2.1rem;font-weight:700;color:var(--a2);}
.pred-box .pred-label{font-size:0.72rem;color:var(--text);letter-spacing:0.1em;text-transform:uppercase;}
.pred-box .pred-ci{font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:var(--a4);margin-top:5px;}
.pred-box .pred-model{font-size:0.7rem;color:var(--a1);margin-top:4px;font-family:'JetBrains Mono',monospace;}
.pred-box-cls{background:linear-gradient(135deg,#150a24 0%,#0e0618 100%);border:1px solid #3a1a5a;border-radius:9px;padding:18px 22px;text-align:center;margin-top:12px;}
.pred-box-cls .pred-val{font-family:'JetBrains Mono',monospace;font-size:2.1rem;font-weight:700;color:#c890f0;}
.pred-box-cls .pred-label{font-size:0.72rem;color:var(--text);letter-spacing:0.1em;text-transform:uppercase;}
.pred-box-cls .pred-prob{font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:var(--a4);margin-top:5px;}
.pred-box-cls .pred-model{font-size:0.7rem;color:#a060c8;margin-top:4px;font-family:'JetBrains Mono',monospace;}
.outlier-badge{font-family:'JetBrains Mono',monospace;font-size:0.74rem;padding:4px 12px;border-radius:4px;margin-top:5px;display:inline-block;}
.outlier-found{background:#2a0f0f;border:1px solid #6a2020;color:#f08080;}
.outlier-clean{background:#0a2010;border:1px solid #1a5030;color:#50c090;}
.summary-box{background:linear-gradient(135deg,#0a1a2a 0%,#080f1c 100%);border:1px solid #1a3a54;border-radius:9px;padding:20px 24px;margin:10px 0;}
.summary-title{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--a1);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid var(--border);}
.summary-item{display:flex;align-items:flex-start;gap:10px;font-size:0.8rem;color:var(--text);padding:7px 0;border-bottom:1px solid #0b1e30;line-height:1.5;}
.summary-item:last-child{border-bottom:none;}
.summary-icon{font-size:0.9rem;min-width:22px;}
.summary-key{color:var(--bright);font-weight:500;min-width:220px;}
.summary-val{color:var(--a4);font-family:'JetBrains Mono',monospace;font-size:0.76rem;}
.best-model-banner{background:linear-gradient(90deg,#0a2a1a 0%,#081c10 100%);border:1px solid #1a6a3a;border-left:4px solid var(--a2);border-radius:7px;padding:14px 20px;margin:10px 0;}
.best-model-banner .bm-label{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--a2);letter-spacing:0.16em;text-transform:uppercase;margin-bottom:3px;}
.best-model-banner .bm-name{font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;color:#4de8a0;}
.best-model-banner .bm-stats{font-size:0.76rem;color:var(--text);margin-top:4px;}
.best-model-banner-cls{background:linear-gradient(90deg,#1a0a2a 0%,#0f0618 100%);border:1px solid #5a2a7a;border-left:4px solid #a060c8;border-radius:7px;padding:14px 20px;margin:10px 0;}
.best-model-banner-cls .bm-label{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#a060c8;letter-spacing:0.16em;text-transform:uppercase;margin-bottom:3px;}
.best-model-banner-cls .bm-name{font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;color:#c890f0;}
.best-model-banner-cls .bm-stats{font-size:0.76rem;color:var(--text);margin-top:4px;}
.apriori-rule-row{background:#0a1820;border:1px solid #1a3a50;border-radius:6px;padding:10px 14px;margin:5px 0;font-family:'JetBrains Mono',monospace;font-size:0.78rem;}
.apriori-rule-row .rule-text{color:#6dc8f0;font-weight:700;margin-bottom:4px;}
.apriori-rule-row .rule-stats{color:var(--text);font-size:0.72rem;}
.apriori-rule-row .rule-lift{color:#e0a844;font-weight:700;}
.model-config-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px 18px;margin-bottom:12px;}
.model-config-title{font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--a1);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;}
.section-divider{height:1px;background:var(--border);margin:24px 0;}
[data-testid="stDataFrame"]{border:1px solid var(--border) !important;border-radius:6px;}
.stButton>button{background:var(--a1);color:#fff;border:none;border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.82rem;letter-spacing:0.06em;padding:8px 20px;transition:background 0.2s;}
.stButton>button:hover{background:#1a6da8;}
.stSelectbox label,.stMultiSelect label,.stSlider label,.stNumberInput label,.stCheckbox label{color:var(--text) !important;font-size:0.8rem !important;}
.health-score-box{border-radius:10px;padding:20px 26px;margin:14px 0;display:flex;align-items:center;gap:24px;}
.health-score-green{background:#0a2318;border:2px solid #22a878;}
.health-score-yellow{background:#1e1800;border:2px solid #e0a844;}
.health-score-red{background:#230a0a;border:2px solid #c05555;}
.health-score-number{font-family:'JetBrains Mono',monospace;font-size:3rem;font-weight:700;line-height:1;}
.health-score-green .health-score-number{color:#22e8a0;}
.health-score-yellow .health-score-number{color:#e0a844;}
.health-score-red .health-score-number{color:#e07070;}
.health-score-label{font-size:0.72rem;color:var(--text);letter-spacing:0.1em;text-transform:uppercase;margin-top:4px;}
.health-score-grade{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;margin-top:2px;}
.health-score-green .health-score-grade{color:#22e8a0;}
.health-score-yellow .health-score-grade{color:#e0a844;}
.health-score-red .health-score-grade{color:#e07070;}
.health-breakdown{font-size:0.78rem;color:var(--text);line-height:2.0;}
.english-summary-box{background:linear-gradient(135deg,#081828 0%,#060f1c 100%);border:1px solid #1a3a54;border-radius:10px;padding:18px 22px;margin:10px 0;}
.english-summary-box .es-title{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--a1);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border);}
.english-summary-box .es-item{font-size:0.82rem;color:#a8ccde;padding:5px 0;display:flex;align-items:flex-start;gap:10px;line-height:1.6;}
.english-summary-box .es-bullet{color:var(--a2);font-weight:700;min-width:16px;}
.problem-explain-box{background:linear-gradient(90deg,#071428 0%,#050e1e 100%);border:1px solid #1a3050;border-left:4px solid #6dc8f0;border-radius:8px;padding:14px 20px;margin:12px 0;font-size:0.82rem;color:#a0c8e0;line-height:1.7;}
.problem-explain-box strong{color:#6dc8f0;}
.grade-banner{display:inline-flex;align-items:center;gap:14px;border-radius:8px;padding:14px 22px;margin:10px 0;}
.grade-A{background:#0a2318;border:1px solid #22a878;}
.grade-B{background:#0d1e30;border:1px solid #2d8fcb;}
.grade-C{background:#1e1800;border:1px solid #e0a844;}
.grade-D{background:#230a0a;border:1px solid #c05555;}
.grade-letter{font-family:'JetBrains Mono',monospace;font-size:2.2rem;font-weight:700;line-height:1;}
.grade-A .grade-letter{color:#22e8a0;}
.grade-B .grade-letter{color:#6dc8f0;}
.grade-C .grade-letter{color:#e0a844;}
.grade-D .grade-letter{color:#e07070;}
.grade-text{font-size:0.8rem;color:var(--text);line-height:1.6;}
.grade-text strong{color:var(--bright);}
.top3-box{background:#081420;border:1px solid #0f2a3a;border-radius:9px;padding:16px 20px;margin:10px 0;}
.top3-title{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--a1);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border);}
.top3-item{display:flex;align-items:center;gap:14px;padding:8px 0;border-bottom:1px solid #0b1e2a;font-size:0.84rem;}
.top3-item:last-child{border-bottom:none;}
.top3-medal{font-size:1.3rem;min-width:30px;}
.top3-feat{color:var(--bright);font-weight:500;flex:1;}
.top3-score{font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--a4);}
.warn-box{border-radius:8px;padding:12px 18px;margin:8px 0;font-size:0.8rem;line-height:1.7;display:flex;align-items:flex-start;gap:12px;}
.warn-box-icon{font-size:1.1rem;min-width:24px;margin-top:1px;}
.warn-box-body{flex:1;}
.warn-box-title{font-family:'JetBrains Mono',monospace;font-size:0.78rem;font-weight:700;margin-bottom:2px;}
.warn-critical{background:#2a0a0a;border:1px solid #7a2020;color:#f09090;}
.warn-critical .warn-box-title{color:#f09090;}
.warn-moderate{background:#2a1a08;border:1px solid #7a4a10;color:#f0b060;}
.warn-moderate .warn-box-title{color:#f0b060;}
.warn-info{background:#0a1828;border:1px solid #1a4060;color:#6dc8f0;}
.warn-info .warn-box-title{color:#6dc8f0;}
.compare-table{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.82rem;margin:12px 0;border-radius:8px;overflow:hidden;}
.compare-table th{background:#0a1c2e;color:var(--a1);font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;padding:10px 16px;text-align:left;border-bottom:1px solid var(--border);}
.compare-table td{padding:10px 16px;border-bottom:1px solid #0b1e30;color:var(--bright);}
.compare-table tr:last-child td{border-bottom:none;}
.compare-table tr:hover td{background:#0b1c2e;}
.compare-orig td{color:#6dc8f0;}
.compare-mod td{color:#22e8a0;}
.km-step-header{display:flex;align-items:center;gap:14px;background:linear-gradient(90deg,#071828 0%,#050f18 100%);border:1px solid #0f2e46;border-left:4px solid #22e8a0;border-radius:8px;padding:14px 20px;margin:24px 0 12px 0;}
.km-step-num{font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:#22e8a0;min-width:36px;line-height:1;}
.km-step-title{font-family:'JetBrains Mono',monospace;font-size:0.95rem;font-weight:700;color:#a8ccde;line-height:1.2;}
.km-step-sub{font-size:0.73rem;color:#7ea8c4;margin-top:3px;}
.km-explainer{background:linear-gradient(135deg,#081828 0%,#060f1c 100%);border:1px solid #0f2e46;border-radius:10px;padding:20px 26px;margin:10px 0 18px 0;}
.km-explainer p{font-size:0.85rem;color:#a0bfd4;line-height:1.7;margin:0 0 10px 0;}
.km-explainer p:last-child{margin-bottom:0;}
.km-explainer strong{color:#22e8a0;}
.km-summary-strip{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0;}
.km-summary-tile{flex:1;min-width:130px;background:linear-gradient(135deg,#0a1e1a 0%,#070f14 100%);border:1px solid #1a4a38;border-radius:10px;padding:16px 18px;text-align:center;}
.km-summary-tile .kmt-val{font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:#22e8a0;line-height:1;}
.km-summary-tile .kmt-lbl{font-size:0.69rem;color:#7ea8c4;letter-spacing:0.1em;text-transform:uppercase;margin-top:6px;}
.km-quality-badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;padding:4px 12px;border-radius:5px;margin-left:10px;vertical-align:middle;}
.km-quality-strong{background:#0a2e1e;color:#22e8a0;border:1px solid #1a6040;}
.km-quality-moderate{background:#2a2008;color:#e0a844;border:1px solid #5a4010;}
.km-quality-weak{background:#2a0808;color:#e07070;border:1px solid #5a2020;}
@keyframes cardFadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.algo-card{background:linear-gradient(135deg,#081828 0%,#060f1c 100%);border:1px solid #1a3a54;border-radius:12px;padding:0;margin:14px 0 18px 0;overflow:hidden;animation:cardFadeIn 0.4s ease both;}
.algo-card-inner{padding:20px 24px 22px;}
.algo-card-header{display:flex;align-items:center;gap:14px;margin-bottom:14px;}
.algo-card-icon{width:42px;height:42px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;flex-shrink:0;}
.algo-card-name{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;color:#a8ccde;line-height:1.2;}
.algo-card-tag{font-family:'JetBrains Mono',monospace;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;margin-top:3px;}
.algo-card-summary{font-size:0.83rem;color:#a0bfd4;line-height:1.7;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #0f2236;}
.algo-pill{font-family:'JetBrains Mono',monospace;font-size:0.62rem;font-weight:600;letter-spacing:0.1em;padding:3px 10px;border-radius:4px;}
.algo-pill-blue{background:#0c2034;color:#6dc8f0;border:1px solid #1a4060;}
.algo-pill-green{background:#0a2318;color:#22e8a0;border:1px solid #1a5038;}
.algo-pill-amber{background:#1e1400;color:#e0a844;border:1px solid #4a3800;}
.algo-pill-red{background:#200a0a;color:#e07070;border:1px solid #4a1818;}
.algo-pill-purple{background:#150a24;color:#c890f0;border:1px solid #3a1a5a;}
.noise-badge{display:inline-block;background:#2a0a1a;border:1px solid #6a2050;color:#f080c0;font-family:'JetBrains Mono',monospace;font-size:0.72rem;padding:4px 12px;border-radius:4px;margin:4px 0;}
.cluster-info-card{background:linear-gradient(135deg,#081828 0%,#060f1c 100%);border:1px solid #1a3a54;border-radius:8px;padding:14px 18px;margin:8px 0;}
.cluster-info-card .ci-header{font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--a2);font-weight:700;margin-bottom:6px;}
.cluster-info-card .ci-body{font-size:0.78rem;color:var(--text);line-height:1.6;}
/* ═══════════ LANDING PAGE CSS ═══════════ */
.landing-hero{text-align:center;padding:52px 20px 40px;position:relative;}
.landing-hero::before{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:700px;height:2px;background:linear-gradient(90deg,transparent,#2d8fcb 30%,#22a878 70%,transparent);}
.landing-badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:0.62rem;letter-spacing:0.22em;text-transform:uppercase;color:#22a878;background:#071f14;border:1px solid #1a5038;padding:4px 16px;border-radius:20px;margin-bottom:20px;}
.landing-title{font-family:'JetBrains Mono',monospace;font-size:2.6rem;font-weight:700;color:#c4dff0;letter-spacing:-0.04em;line-height:1.18;margin-bottom:16px;}
.landing-title .grad{background:linear-gradient(135deg,#2d8fcb 0%,#22e8a0 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.landing-sub{font-size:0.94rem;color:#7ea8c4;max-width:540px;margin:0 auto 10px;line-height:1.75;}
.landing-stat-strip{display:flex;justify-content:center;gap:36px;margin:28px 0 0;flex-wrap:wrap;}
.landing-stat{text-align:center;}
.landing-stat-num{font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:#2d8fcb;line-height:1;}
.landing-stat-lbl{font-size:0.62rem;color:#3a6a88;letter-spacing:0.12em;text-transform:uppercase;margin-top:5px;}
@keyframes cardReveal{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.feat-card{background:linear-gradient(150deg,#0c1e32 0%,#08131f 100%);border:1px solid #0f2840;border-radius:14px;padding:26px 22px 22px;position:relative;overflow:hidden;transition:transform 0.28s cubic-bezier(.22,.68,0,1.2),border-color 0.25s ease,box-shadow 0.28s ease;cursor:default;animation:cardReveal 0.5s ease both;height:100%;box-sizing:border-box;}
.feat-card:hover{transform:translateY(-6px) scale(1.015);border-color:var(--card-accent,#2d8fcb);box-shadow:0 20px 48px rgba(0,0,0,0.6),0 0 0 1px var(--card-accent,#2d8fcb),inset 0 1px 0 rgba(255,255,255,0.04);}
.feat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--card-accent,#2d8fcb);opacity:0;transition:opacity 0.25s ease;border-radius:14px 14px 0 0;}
.feat-card:hover::before{opacity:1;}
.feat-card::after{content:'';position:absolute;top:-50px;right:-50px;width:140px;height:140px;border-radius:50%;background:var(--card-accent,#2d8fcb);opacity:0.03;transition:opacity 0.3s ease,transform 0.3s ease;}
.feat-card:hover::after{opacity:0.10;transform:scale(1.3);}
.feat-card-icon{font-size:2.1rem;margin-bottom:14px;display:block;line-height:1;filter:drop-shadow(0 0 14px var(--card-accent,#2d8fcb));transition:transform 0.25s ease;}
.feat-card:hover .feat-card-icon{transform:scale(1.1);}
.feat-card-tag{font-family:'JetBrains Mono',monospace;font-size:0.60rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--card-accent,#2d8fcb);margin-bottom:6px;opacity:0.8;}
.feat-card-title{font-family:'JetBrains Mono',monospace;font-size:1.02rem;font-weight:700;color:#b8d8ea;margin-bottom:10px;line-height:1.2;}
.feat-card-desc{font-size:0.78rem;color:#5a8aaa;line-height:1.7;margin-bottom:16px;}
.feat-card-pills{display:flex;flex-wrap:wrap;gap:5px;}
.feat-pill{font-family:'JetBrains Mono',monospace;font-size:0.58rem;letter-spacing:0.07em;padding:2px 9px;border-radius:3px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);color:#4a7a96;transition:background 0.2s,color 0.2s;}
.feat-card:hover .feat-pill{background:rgba(255,255,255,0.06);color:#6a9ab6;}
</style>
""", unsafe_allow_html=True)


PLOT_BG="#060d18";AXES_BG="#0b1628";GRID_CLR="#0f2236";TEXT_CLR="#7ea8c4"
ACCENT1="#2d8fcb";ACCENT2="#22a878";ACCENT3="#c05555";ACCENT4="#e0a844"
ACCENT5="#a060c8";ACCENT6="#d05090";ACCENT7="#22e8a0"
CLUSTER_PALETTE=["#22e8a0","#2d8fcb","#e0a844","#d05090","#a060c8","#44c8e0","#e07030","#c05555","#70c040","#c0a020"]
NOISE_COLOR="#444466"
REG_MODEL_COLORS={"Linear":ACCENT1,"Ridge":ACCENT2,"Lasso":ACCENT4,"KNN":ACCENT5,"Decision Tree":ACCENT3,"Random Forest":ACCENT6,"SVR":"#44c8e0","Polynomial":"#e07030","Gradient Boosting":"#70c040"}
CLS_MODEL_COLORS={"Logistic":ACCENT1,"KNN":ACCENT5,"Naive Bayes":ACCENT4,"Decision Tree":ACCENT3,"Random Forest":ACCENT6,"SVM":"#44c8e0","Gradient Boosting":"#70c040","XGBoost":"#e07030"}

def apply_theme(fig,axes):
    fig.patch.set_facecolor(PLOT_BG)
    for ax in (axes if isinstance(axes,list) else [axes]):
        ax.set_facecolor(AXES_BG);ax.tick_params(colors=TEXT_CLR,labelsize=9)
        ax.xaxis.label.set_color(TEXT_CLR);ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color("#7eb8d4");ax.grid(True,color=GRID_CLR,linewidth=0.5,linestyle="--",alpha=0.7)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID_CLR)

CLASSIFICATION_THRESHOLD=15
def detect_mode(series):
    if series.dtype==object or str(series.dtype) in ("string","category"): return "classification"
    n_unique=series.nunique();n_total=len(series.dropna())
    if pd.api.types.is_integer_dtype(series) and n_unique<=CLASSIFICATION_THRESHOLD: return "classification"
    if n_unique<=5 and n_unique/max(n_total,1)<0.05: return "classification"
    return "regression"

class ManualLinearRegression:
    name="Linear"
    def __init__(self): self.beta=None
    def fit(self,X,y):
        Xb=np.hstack([np.ones((len(X),1)),X]); self.beta=np.linalg.pinv(Xb)@y
    def predict(self,X): return np.hstack([np.ones((len(X),1)),X])@self.beta
    def get_coefficients(self): return self.beta

class ManualRidgeRegression:
    name="Ridge"
    def __init__(self,lam=1.0): self.lam=lam;self.beta=None
    def fit(self,X,y):
        Xb=np.hstack([np.ones((len(X),1)),X]);reg=self.lam*np.eye(Xb.shape[1]);reg[0,0]=0
        self.beta=np.linalg.solve(Xb.T@Xb+reg,Xb.T@y)
    def predict(self,X): return np.hstack([np.ones((len(X),1)),X])@self.beta
    def get_coefficients(self): return self.beta

class ManualLassoRegression:
    name="Lasso"
    def __init__(self,lam=0.1,max_iter=1000,tol=1e-4): self.lam=lam;self.max_iter=max_iter;self.tol=tol;self.beta=None
    def _soft(self,rho,lam): return np.sign(rho)*max(abs(rho)-lam,0.0)
    def fit(self,X,y):
        Xb=np.hstack([np.ones((len(X),1)),X]);beta=np.zeros(Xb.shape[1])
        for _ in range(self.max_iter):
            old=beta.copy()
            for j in range(Xb.shape[1]):
                r=y-Xb@beta+Xb[:,j]*beta[j];rho=Xb[:,j]@r/len(X);d=Xb[:,j]@Xb[:,j]/len(X)
                beta[j]=rho/d if j==0 else self._soft(rho,self.lam)/d
            if np.max(np.abs(beta-old))<self.tol: break
        self.beta=beta
    def predict(self,X): return np.hstack([np.ones((len(X),1)),X])@self.beta
    def get_coefficients(self): return self.beta

class ManualKNNRegression:
    name="KNN"
    def __init__(self,k=5): self.k=k
    def fit(self,X,y): self.X_tr=X;self.y_tr=y
    def predict(self,X): return np.array([self.y_tr[np.argsort(np.linalg.norm(self.X_tr-xi,axis=1))[:self.k]].mean() for xi in X])

class ManualPolynomialRegression:
    name="Polynomial"
    def __init__(self,degree=2):
        self.degree=degree;self._linear=ManualLinearRegression();self.beta=None;self.feature_names_out_=[]
    def _expand(self,X):
        from itertools import combinations_with_replacement
        n,p=X.shape;features=[np.ones((n,1))];feat_names=["1"]
        for j in range(p): features.append(X[:,j:j+1]);feat_names.append(f"x{j+1}")
        for d in range(2,self.degree+1):
            for combo in combinations_with_replacement(range(p),d):
                term=np.ones((n,1));name_parts=[]
                for idx in combo: term=term*X[:,idx:idx+1];name_parts.append(f"x{idx+1}")
                features.append(term);feat_names.append("·".join(name_parts))
        self.feature_names_out_=feat_names;return np.hstack(features)
    def fit(self,X,y):
        Xp=self._expand(X)[:,1:];self._linear.fit(Xp,y);self.beta=self._linear.get_coefficients()
    def predict(self,X): return self._linear.predict(self._expand(X)[:,1:])
    def get_coefficients(self): return self.beta

class ManualLogisticRegression:
    name="Logistic"
    def __init__(self,lr=0.1,max_iter=500): self.lr=lr;self.max_iter=max_iter;self.W=None;self.b=None;self.classes_=None
    def _softmax(self,z): ez=np.exp(z-z.max(axis=1,keepdims=True));return ez/ez.sum(axis=1,keepdims=True)
    def fit(self,X,y):
        self.classes_=np.unique(y);K=len(self.classes_);n,p=X.shape
        self.W=np.zeros((p,K));self.b=np.zeros(K);y_oh=(y[:,None]==self.classes_[None,:]).astype(float)
        for _ in range(self.max_iter):
            probs=self._softmax(X@self.W+self.b);diff=probs-y_oh
            self.W-=self.lr*X.T@diff/n;self.b-=self.lr*diff.mean(axis=0)
    def predict_proba(self,X): return self._softmax(X@self.W+self.b)
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(axis=1)]

class ManualKNNClassifier:
    name="KNN"
    def __init__(self,k=5): self.k=k
    def fit(self,X,y): self.X_tr=X;self.y_tr=y
    def predict(self,X):
        out=[]
        for xi in X:
            idx=np.argsort(np.linalg.norm(self.X_tr-xi,axis=1))[:self.k];vals,cnts=np.unique(self.y_tr[idx],return_counts=True);out.append(vals[cnts.argmax()])
        return np.array(out)
    def predict_proba(self,X):
        classes=np.unique(self.y_tr);out=[]
        for xi in X:
            idx=np.argsort(np.linalg.norm(self.X_tr-xi,axis=1))[:self.k];row=np.array([(self.y_tr[idx]==c).mean() for c in classes]);out.append(row)
        return np.array(out)

class ManualNaiveBayes:
    name="Naive Bayes"
    def __init__(self): self.classes_=None;self.params={}
    def fit(self,X,y):
        self.classes_=np.unique(y);self.priors={}
        for c in self.classes_:
            Xc=X[y==c];self.priors[c]=len(Xc)/len(X);self.params[c]=(Xc.mean(axis=0),Xc.var(axis=0)+1e-9)
    def _log_likelihood(self,X,mu,var): return -0.5*np.sum(np.log(2*np.pi*var)+(X-mu)**2/var,axis=1)
    def predict_proba(self,X):
        log_posts=np.array([np.log(self.priors[c])+self._log_likelihood(X,*self.params[c]) for c in self.classes_]).T
        log_posts-=log_posts.max(axis=1,keepdims=True);probs=np.exp(log_posts);probs/=probs.sum(axis=1,keepdims=True);return probs
    def predict(self,X): return self.classes_[self.predict_proba(X).argmax(axis=1)]

class ManualKMeans:
    def __init__(self,k=3,max_iter=300,tol=1e-4,random_state=42):
        self.k=k;self.max_iter=max_iter;self.tol=tol;self.random_state=random_state
        self.centroids_=None;self.labels_=None;self.inertia_=None;self.n_iter_=0
    def _init_centroids(self,X): rng=np.random.RandomState(self.random_state);idx=rng.choice(len(X),self.k,replace=False);return X[idx].copy()
    def _assign_labels(self,X): dists=np.linalg.norm(X[:,None,:]-self.centroids_[None,:,:],axis=2);return dists.argmin(axis=1)
    def _update_centroids(self,X,labels):
        new_c=np.zeros_like(self.centroids_)
        for k in range(self.k):
            mask=labels==k;new_c[k]=X[mask].mean(axis=0) if mask.sum()>0 else self.centroids_[k]
        return new_c
    def fit(self,X):
        self.centroids_=self._init_centroids(X)
        for i in range(self.max_iter):
            labels=self._assign_labels(X);new_c=self._update_centroids(X,labels);shift=np.linalg.norm(new_c-self.centroids_)
            self.centroids_=new_c;self.n_iter_=i+1
            if shift<self.tol: break
        self.labels_=self._assign_labels(X)
        self.inertia_=float(sum(np.linalg.norm(X[self.labels_==k]-self.centroids_[k])**2 for k in range(self.k) if (self.labels_==k).sum()>0))
        return self
    def predict(self,X): return self._assign_labels(X)

class ManualPCA:
    def __init__(self,n_components=2):
        self.n_components=n_components;self.components_=None;self.explained_variance_=None;self.explained_variance_ratio_=None;self.mean_=None
    def fit(self,X):
        self.mean_=X.mean(axis=0);X_c=X-self.mean_;cov=np.cov(X_c,rowvar=False)
        if cov.ndim==0: cov=np.array([[float(cov)]])
        eigenvalues,eigenvectors=np.linalg.eigh(cov);order=np.argsort(eigenvalues)[::-1]
        eigenvalues=eigenvalues[order];eigenvectors=eigenvectors[:,order]
        self.components_=eigenvectors[:,:self.n_components].T;self.explained_variance_=eigenvalues[:self.n_components]
        total_var=eigenvalues.sum();self.explained_variance_ratio_=self.explained_variance_/total_var if total_var>0 else self.explained_variance_*0
        return self
    def transform(self,X): return (X-self.mean_)@self.components_.T
    def fit_transform(self,X): return self.fit(X).transform(X)

class ManualHierarchicalClustering:
    def __init__(self,n_clusters=3,linkage_type="complete"): self.n_clusters=n_clusters;self.linkage_type=linkage_type;self.labels_=None
    def _cluster_distance(self,X,a,b):
        dists=[np.linalg.norm(X[i]-X[j]) for i in a for j in b]
        if not dists: return np.inf
        return max(dists) if self.linkage_type=="complete" else min(dists) if self.linkage_type=="single" else np.mean(dists)
    def fit(self,X):
        n=len(X);clusters={i:[i] for i in range(n)}
        while len(clusters)>self.n_clusters:
            keys=list(clusters.keys());best_dist=np.inf;merge_a=merge_b=None
            for i in range(len(keys)):
                for j in range(i+1,len(keys)):
                    d=self._cluster_distance(X,clusters[keys[i]],clusters[keys[j]])
                    if d<best_dist: best_dist=d;merge_a,merge_b=keys[i],keys[j]
            if merge_a is None: break
            clusters[merge_a]=clusters[merge_a]+clusters[merge_b];del clusters[merge_b]
        labels=np.full(n,-1,dtype=int)
        for li,members in enumerate(clusters.values()):
            for m in members: labels[m]=li
        self.labels_=labels;return self
    def fit_predict(self,X): self.fit(X);return self.labels_

class ManualApriori:
    def __init__(self,min_support=0.2,min_confidence=0.5): self.min_support=min_support;self.min_confidence=min_confidence;self.frequent_itemsets_={};self.rules_=[]
    def _get_support(self,transactions,itemset): return sum(1 for t in transactions if itemset.issubset(t))/len(transactions)
    def _generate_candidates(self,prev_itemsets,k):
        items_list=[sorted(list(s)) for s in prev_itemsets];candidates=set()
        for i in range(len(items_list)):
            for j in range(i+1,len(items_list)):
                a,b=items_list[i],items_list[j]
                if a[:k-2]==b[:k-2]:
                    c=frozenset(a)|frozenset(b)
                    if len(c)==k: candidates.add(c)
        return candidates
    def fit(self,transactions):
        transactions=[frozenset(t) for t in transactions];all_items=set(item for t in transactions for item in t);current_frequent=set()
        for item in all_items:
            fs=frozenset([item]);sup=self._get_support(transactions,fs)
            if sup>=self.min_support: self.frequent_itemsets_[fs]=sup;current_frequent.add(fs)
        k=2
        while current_frequent:
            candidates=self._generate_candidates(current_frequent,k);current_frequent=set()
            for cand in candidates:
                sup=self._get_support(transactions,cand)
                if sup>=self.min_support: self.frequent_itemsets_[cand]=sup;current_frequent.add(cand)
            k+=1
        for itemset,sup in self.frequent_itemsets_.items():
            if len(itemset)<2: continue
            items=list(itemset)
            for mask in range(1,2**len(items)-1):
                antecedent=frozenset(items[i] for i in range(len(items)) if mask&(1<<i));consequent=itemset-antecedent
                if not consequent: continue
                ant_sup=self.frequent_itemsets_.get(antecedent)
                if ant_sup is None or ant_sup==0: continue
                confidence=sup/ant_sup
                if confidence>=self.min_confidence:
                    cons_sup=self.frequent_itemsets_.get(consequent,self._get_support(transactions,consequent))
                    lift=confidence/cons_sup if cons_sup>0 else 0.0
                    self.rules_.append({"antecedent":antecedent,"consequent":consequent,"support":round(sup,4),"confidence":round(confidence,4),"lift":round(lift,4)})
        seen=set();unique_rules=[]
        for r in self.rules_:
            key=(r["antecedent"],r["consequent"])
            if key not in seen: seen.add(key);unique_rules.append(r)
        self.rules_=sorted(unique_rules,key=lambda x:x["lift"],reverse=True);return self

def calc_r2(yt,yp): ss_res=np.sum((yt-yp)**2);ss_tot=np.sum((yt-yt.mean())**2);return float(1-ss_res/ss_tot) if ss_tot else 0.0
def calc_rmse(yt,yp): return float(np.sqrt(np.mean((yt-yp)**2)))
def calc_mae(yt,yp): return float(np.mean(np.abs(yt-yp)))
def calc_adj_r2(r2,n,k): return float(1-(1-r2)*(n-1)/(n-k-1)) if n>k+1 else float("nan")
def reg_evaluate(model,X_te,y_te):
    yp=model.predict(X_te);return {"R²":calc_r2(y_te,yp),"RMSE":calc_rmse(y_te,yp),"MAE":calc_mae(y_te,yp),"yp":yp}
def silhouette_score_manual(X,labels):
    unique_labels=[l for l in np.unique(labels) if l!=-1]
    if len(unique_labels)<2: return float("nan")
    s_vals=[]
    for i in range(len(X)):
        if labels[i]==-1: continue
        same=X[labels==labels[i]];a=float(np.linalg.norm(same-X[i],axis=1).sum()/(len(same)-1)) if len(same)>1 else 0.0
        b_vals=[float(np.linalg.norm(X[labels==lbl]-X[i],axis=1).mean()) for lbl in unique_labels if lbl!=labels[i] and (labels==lbl).sum()>0]
        if not b_vals: continue
        b=min(b_vals);denom=max(a,b);s_vals.append((b-a)/denom if denom>0 else 0.0)
    return float(np.mean(s_vals)) if s_vals else float("nan")

EXPLAINER_DATA={
    "Linear":{"color":"#2d8fcb","icon":"📏","tag":"Regression · Manual","summary":"Finds the best-fit straight line minimising squared errors.","analogy":"A rubber band stretched to sit as close as possible to all data points.","strengths":["Fast","Interpretable","Works well for linear relationships"],"weaknesses":["Assumes linearity","Sensitive to outliers","No interactions"],"when_to_use":"When you expect a straight-line relationship and need explainability.","pills":[("Interpretable","algo-pill-green"),("Fast","algo-pill-blue"),("Low complexity","algo-pill-blue")]},
    "Ridge":{"color":"#22a878","icon":"🏔️","tag":"Regression · Regularised · Manual","summary":"Linear Regression + L2 penalty to shrink large coefficients.","analogy":"A spring pulling every coefficient back toward zero.","strengths":["Handles correlated features","Rarely overfits","Interpretable"],"weaknesses":["Keeps all features","Needs λ tuning","Assumes linearity"],"when_to_use":"When you have many correlated features and Linear is unstable.","pills":[("Regularised","algo-pill-green"),("Handles collinearity","algo-pill-blue")]},
    "Lasso":{"color":"#e0a844","icon":"🔦","tag":"Regression · Sparse · Manual","summary":"Like Ridge but L1 penalty drives unimportant weights to exactly zero.","analogy":"A hard budget cap — once a coefficient hits zero it stays there.","strengths":["Built-in feature selection","Sparse output","Works with few important features"],"weaknesses":["Drops correlated features arbitrarily","Slower","λ needs tuning"],"when_to_use":"When you suspect only a few features really matter.","pills":[("Feature selector","algo-pill-green"),("Sparse output","algo-pill-blue")]},
    "KNN":{"color":"#a060c8","icon":"🗺️","tag":"Regression & Classification · Non-parametric · Manual","summary":"Predicts by averaging the K nearest training examples.","analogy":"Asking your K nearest neighbours for advice and averaging their answers.","strengths":["Zero training time","Captures non-linearity","No distribution assumptions"],"weaknesses":["Slow prediction","Bad in high dimensions","Sensitive to K"],"when_to_use":"Small-to-medium datasets with local, non-linear patterns.","pills":[("Non-parametric","algo-pill-purple"),("No training","algo-pill-blue"),("Slow predict","algo-pill-amber")]},
    "Decision Tree":{"color":"#c05555","icon":"🌳","tag":"Regression & Classification · Tree · sklearn","summary":"Learns a series of yes/no questions to split data into pure groups.","analogy":"A flowchart of questions splitting the data step by step.","strengths":["Fully interpretable","Handles mixed types","No preprocessing"],"weaknesses":["Overfits easily","Unstable","Piecewise-constant predictions"],"when_to_use":"When interpretability is the top priority.","pills":[("Interpretable","algo-pill-green"),("No preprocessing","algo-pill-blue"),("Overfits","algo-pill-red")]},
    "Random Forest":{"color":"#d05090","icon":"🌲","tag":"Regression & Classification · Ensemble · sklearn","summary":"Hundreds of Decision Trees trained on random subsets — aggregate their votes.","analogy":"100 experts each seeing partial evidence then voting collectively.","strengths":["High accuracy","Resistant to overfitting","Feature importance"],"weaknesses":["Slow to predict","Black box","Memory-heavy"],"when_to_use":"First serious model on any tabular dataset.","pills":[("Ensemble","algo-pill-green"),("High accuracy","algo-pill-green"),("Feature importance","algo-pill-blue")]},
    "SVR":{"color":"#44c8e0","icon":"📐","tag":"Regression · Kernel · sklearn","summary":"Support Vector Regression finds a tube of width ε around the data. Only points outside the tube influence the model. RBF kernel allows non-linear curves.","analogy":"Drawing the widest possible road through your data — only the cars that fall off the road edge matter.","strengths":["Works well in high dimensions","Robust to outliers inside ε-tube","Non-linear via RBF"],"weaknesses":["Slow on large datasets","Needs feature scaling","C, ε, γ need tuning"],"when_to_use":"Medium-sized datasets where non-linear relationship exists and you need outlier robustness.","pills":[("Kernel trick","algo-pill-blue"),("Non-linear","algo-pill-green"),("Needs scaling","algo-pill-amber"),("Slow on big data","algo-pill-red")]},
    "Polynomial":{"color":"#e07030","icon":"〽️","tag":"Regression · Manual · Polynomial Features","summary":"Extends Linear Regression by adding squared/cubed/interaction terms, then fits OLS through them.","analogy":"Instead of fitting a straight ruler, you bend it into a curve by adding x² as an extra column.","strengths":["Captures non-linear trends","Still uses interpretable OLS formula","Manual — full transparency"],"weaknesses":["Feature explosion with high degree","Prone to overfitting at d≥4","Blows up outside training range"],"when_to_use":"When the relationship is clearly curved (U-shaped) but you want to keep it simple. Use degree=2 first.","pills":[("Manual OLS","algo-pill-green"),("Curved patterns","algo-pill-blue"),("Overfits at d≥4","algo-pill-red"),("Feature explosion","algo-pill-amber")]},
    "Gradient Boosting":{"color":"#70c040","icon":"🚀","tag":"Regression & Classification · Ensemble · sklearn","summary":"Builds trees one at a time, where each new tree corrects errors of all previous trees.","analogy":"A student who studies only the questions they got wrong — improving with each round.","strengths":["Most accurate on structured data","Handles mixed feature types","Robust with tuning"],"weaknesses":["Slow (sequential)","Many hyperparameters","Can overfit"],"when_to_use":"When accuracy is the primary goal and you have compute time.","pills":[("Highest accuracy","algo-pill-green"),("Boosting ensemble","algo-pill-blue"),("Sequential train","algo-pill-amber"),("Many HPs","algo-pill-amber")]},
    "Logistic":{"color":"#2d8fcb","icon":"📊","tag":"Classification · Manual","summary":"Estimates class probabilities via softmax; predicts the highest.","analogy":"Weighted voting system converting raw scores to percentage chances.","strengths":["Fast","Calibrated probabilities","Interpretable"],"weaknesses":["Linear boundary","Struggles with non-linear separation","Needs scaling"],"when_to_use":"When you need probabilities and classes are roughly linearly separable.","pills":[("Interpretable","algo-pill-green"),("Outputs probs","algo-pill-blue"),("Linear boundary","algo-pill-amber")]},
    "Naive Bayes":{"color":"#e0a844","icon":"🧮","tag":"Classification · Probabilistic · Manual","summary":"Bayes' theorem assuming all features are independent.","analogy":"A spam filter multiplying per-word probabilities assuming independence.","strengths":["Extremely fast","Low data needs","Robust"],"weaknesses":["Independence rarely true","Poor with correlated features","No interactions"],"when_to_use":"Scarce data, fast baselines, or text classification.","pills":[("Very fast","algo-pill-green"),("Probabilistic","algo-pill-blue"),("Strong independence","algo-pill-red")]},
    "SVM":{"color":"#44c8e0","icon":"🏹","tag":"Classification · Kernel · sklearn","summary":"Finds the widest margin hyperplane separating classes. RBF kernel separates non-linearly separable data.","analogy":"Drawing the fattest possible line between two groups — positioned by only the dots at the edge.","strengths":["Effective in high dimensions","Robust to outliers","Flexible via kernels"],"weaknesses":["Slow on large datasets","Requires feature scaling","No direct probability output"],"when_to_use":"Small-to-medium datasets with complex non-linear decision boundary.","pills":[("Max margin","algo-pill-green"),("Kernel trick","algo-pill-blue"),("Needs scaling","algo-pill-amber"),("Slow on large","algo-pill-red")]},
    "XGBoost":{"color":"#e07030","icon":"⚡","tag":"Classification · XGBoost · sklearn-compatible","summary":"Optimised parallelised gradient boosting with L1/L2 regularisation and native missing value handling.","analogy":"Like Gradient Boosting with a turbocharger — faster, regularised, handles missing values.","strengths":["Extremely fast (parallelised)","Regularised","Handles missing values"],"weaknesses":["Many hyperparameters","Hard to interpret","Requires separate install"],"when_to_use":"Competition-style tabular classification where maximum accuracy is required.","pills":[("Fastest boosting","algo-pill-green"),("Regularised","algo-pill-blue"),("Handles NaN","algo-pill-blue"),("Many HPs","algo-pill-amber")]},
}
_PILL_STYLES={"algo-pill-blue":("#6dc8f0","#0c2034","#1a4060"),"algo-pill-green":("#22e8a0","#0a2318","#1a5038"),"algo-pill-amber":("#e0a844","#1e1400","#4a3800"),"algo-pill-red":("#e07070","#200a0a","#4a1818"),"algo-pill-purple":("#c890f0","#150a24","#3a1a5a")}

def _pill_inline(label,ct,cb,cbr):
    return f'<span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;font-weight:600;letter-spacing:0.1em;padding:3px 10px;border-radius:4px;background:{cb};color:{ct};border:1px solid {cbr};margin-right:6px;display:inline-block;margin-bottom:4px;">{label}</span>'

def render_explainer_card(model_name,mode="regression"):
    import streamlit.components.v1 as components
    data=EXPLAINER_DATA.get(model_name)
    if data is None: st.markdown(f'<div class="problem-explain-box">No explainer for <strong>{model_name}</strong>.</div>',unsafe_allow_html=True);return
    color=data["color"];pills_html=""
    for lbl,css_cls in data["pills"]:
        ct,cb,cbr=_PILL_STYLES.get(css_cls,("#a8ccde","#0b1628","#0f2236"));pills_html+=_pill_inline(lbl,ct,cb,cbr)
    li_style='style="padding:3px 0;color:#7ea8c4;font-size:0.78rem;"'
    def make_li(items): return "".join(f'<li {li_style}>{x}</li>' for x in items)
    def detail_box(lbl,lbl_color,content):
        return f'<div style="background:#060d18;border:1px solid #0f2236;border-radius:8px;padding:12px 14px;"><div style="font-family:JetBrains Mono,monospace;font-size:0.64rem;letter-spacing:0.14em;text-transform:uppercase;color:{lbl_color};margin-bottom:8px;">{lbl}</div><div style="font-size:0.78rem;color:#7ea8c4;line-height:1.65;">{content}</div></div>'
    grid_html='<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;">'+detail_box("Real-world analogy","#e0a844",data["analogy"])+detail_box("Strengths","#22a878",f'<ul style="list-style:none;padding:0;margin:0;">{make_li(data["strengths"])}</ul>')+detail_box("Limitations","#c05555",f'<ul style="list-style:none;padding:0;margin:0;">{make_li(data["weaknesses"])}</ul>')+'</div>'
    card=f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>@import url(\'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500&display=swap\');*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:\'Inter\',sans-serif;}}@keyframes cardIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}.card{{animation:cardIn 0.35s ease both;}}li::before{{content:"› ";color:#2d8fcb;}}</style></head><body><div class="card" style="background:linear-gradient(135deg,#081828 0%,#060f1c 100%);border:1px solid #1a3a54;border-radius:12px;overflow:hidden;margin:4px 0 12px;"><div style="height:4px;background:{color};width:100%;"></div><div style="padding:20px 24px 22px;"><div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;"><div style="width:42px;height:42px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;flex-shrink:0;background:{color}22;border:1px solid {color}44;">{data["icon"]}</div><div><div style="font-family:JetBrains Mono,monospace;font-size:1.05rem;font-weight:700;color:#a8ccde;">{model_name}</div><div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;color:{color};margin-top:3px;">{data["tag"]}</div></div></div><div style="margin-bottom:14px;">{pills_html}</div><div style="font-size:0.83rem;color:#a0bfd4;line-height:1.7;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #0f2236;">{data["summary"]}</div>{grid_html}<div style="display:flex;align-items:flex-start;gap:10px;background:#060f1c;border:1px solid #0f2236;border-radius:8px;padding:11px 14px;font-size:0.78rem;color:#7ea8c4;line-height:1.6;"><span style="font-size:1rem;min-width:20px;">💡</span><div><span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#2d8fcb;letter-spacing:0.12em;text-transform:uppercase;">When to use</span><br>{data["when_to_use"]}</div></div></div></div></body></html>'
    components.html(card,height=400+max(len(data["strengths"]),len(data["weaknesses"]))*28,scrolling=False)

def render_all_explainers(mode="regression"):
    reg_models=["Linear","Ridge","Lasso","KNN","Decision Tree","Random Forest","SVR","Polynomial","Gradient Boosting"]
    cls_models=["Logistic","KNN","Naive Bayes","Decision Tree","Random Forest","SVM","Gradient Boosting"]
    if XGBOOST_AVAILABLE: cls_models.append("XGBoost")
    with st.expander("📚  All Models Explained — click to expand",expanded=False):
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#2d8fcb;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #0f2236;">⬡ Algorithm Reference — All Models</div>',unsafe_allow_html=True)
        for mname in (reg_models if mode=="regression" else cls_models): render_explainer_card(mname,mode)

def detect_outliers_iqr(series):
    q1=series.quantile(0.25);q3=series.quantile(0.75);iqr=q3-q1;lo=q1-1.5*iqr;hi=q3+1.5*iqr
    return (series<lo)|(series>hi),q1,q3,iqr,lo,hi

def stat_card(v,l): return f'<div class="stat-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
def metric_card(v,l,cls="metric-card"): return f'<div class="{cls}"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
def module_banner(num,title,sub=""):
    s=f'<div class="mod-sub">{sub}</div>' if sub else ""
    return f'<div class="module-banner"><div class="mod-label">⬡ Module {num}</div><div class="mod-title">{title}</div>{s}</div>'
def assumption_row(title,desc,ok,extra=""):
    badge=f'<span class="badge badge-{"pass" if ok else "fail"}">{"PASS" if ok else "FAIL"}</span>'
    ex=f'<div class="assumption-extra">{extra}</div>' if extra else ""
    return f'<div class="assumption-row">{badge}<div><div class="assumption-title">{title}</div><div class="assumption-desc">{desc}</div>{ex}</div></div>'
def summary_row(icon,key,val): return f'<div class="summary-item"><span class="summary-icon">{icon}</span><span class="summary-key">{key}</span><span class="summary-val">{val}</span></div>'
def km_step(num,title,sub=""):
    sh=f'<div class="km-step-sub">{sub}</div>' if sub else ""
    return f'<div class="km-step-header"><div class="km-step-num">{num}</div><div><div class="km-step-title">{title}</div>{sh}</div></div>'

def compute_health_score(data,numeric_cols,outlier_counts):
    score=100;breakdown={}
    miss_pct=data.isnull().sum().sum()/max(data.size,1)*100;miss_penalty=min(30,int(miss_pct*1.5));score-=miss_penalty
    breakdown["Missing values"]=f"-{miss_penalty} pts  ({miss_pct:.1f}% cells missing)"
    total_outliers=sum(outlier_counts.values()) if outlier_counts else 0
    total_vals=sum(data[c].count() for c in numeric_cols) if numeric_cols else 1
    out_penalty=min(25,int(total_outliers/max(total_vals,1)*100*2));score-=out_penalty
    breakdown["Outliers (IQR)"]=f"-{out_penalty} pts  ({total_outliers} outlier values)"
    dup_count=int(data.duplicated().sum());dup_penalty=min(20,int(dup_count/max(len(data),1)*100*2));score-=dup_penalty
    breakdown["Duplicate rows"]=f"-{dup_penalty} pts  ({dup_count} duplicates)"
    skewed_cols=[c for c in numeric_cols if abs(data[c].skew())>1.0];skew_penalty=min(25,len(skewed_cols)*5);score-=skew_penalty
    breakdown["High skewness"]=f"-{skew_penalty} pts  ({len(skewed_cols)} skewed columns)"
    score=max(0,min(100,score))
    if score>=80: label,css="Excellent","health-score-green"
    elif score>=60: label,css="Fair","health-score-yellow"
    else: label,css="Poor","health-score-red"
    return score,breakdown,label,css

def render_health_score(score,breakdown,label,css_class):
    bd_html="".join(f'<div>{"✓" if "-0" in v or "-0 " in v else "⚠"} <b>{k}:</b> {v}</div>' for k,v in breakdown.items())
    return f'<div class="health-score-box {css_class}"><div style="text-align:center;min-width:90px;"><div class="health-score-number">{score}</div><div class="health-score-label">/ 100</div><div class="health-score-grade">{label}</div></div><div class="health-breakdown">{bd_html}</div></div>'

def build_english_summary(data,numeric_cols,cat_cols,total_miss,outlier_counts,corr_with_target,target_col):
    bullets=[]
    bullets.append(f"This dataset has <b>{data.shape[0]:,} rows</b> and <b>{data.shape[1]} columns</b> ({len(numeric_cols)} numeric, {len(cat_cols)} categorical).")
    if target_col in numeric_cols: bullets.append(f"The target column <b>'{target_col}'</b> is <b>numeric</b> — likely a <b>regression</b> problem.")
    else: bullets.append(f"The target column <b>'{target_col}'</b> has <b>{data[target_col].nunique() if target_col in data.columns else '?'} categories</b> — a <b>classification</b> problem.")
    if corr_with_target is not None and len(corr_with_target):
        best_feat=corr_with_target.abs().idxmax();r_val=corr_with_target[best_feat];direction="positively" if r_val>0 else "negatively"
        bullets.append(f"The strongest predictor is <b>'{best_feat}'</b> (r = {r_val:.2f} — {direction} related to the target).")
    if total_miss==0: bullets.append("The dataset is <b>complete</b> — no missing values detected.")
    else: bullets.append(f"There are <b>{total_miss:,} missing values</b>. Worst column: <b>'{data.isnull().sum().idxmax()}'</b>.")
    if outlier_counts:
        bad_cols=[c for c,n in outlier_counts.items() if n>0]
        if bad_cols: bullets.append(f"Outliers in <b>{len(bad_cols)} column(s)</b>: {', '.join(f'<b>{c}</b>' for c in bad_cols[:4])}.")
    return bullets

def render_english_summary(bullets):
    items_html="".join(f'<div class="es-item"><span class="es-bullet">›</span><span>{b}</span></div>' for b in bullets)
    return f'<div class="english-summary-box"><div class="es-title">⬡ Plain English Summary</div>{items_html}</div>'

def get_model_grade(score_val):
    if score_val>=0.90: return "A","Excellent performance","A"
    elif score_val>=0.75: return "B","Good performance","B"
    elif score_val>=0.60: return "C","Fair performance","C"
    else: return "D","Needs improvement","D"

def render_grade_banner(model_name,score_val,metric_name="R²"):
    letter,label,css=get_model_grade(score_val)
    return f'<div class="grade-banner grade-{css}"><div class="grade-letter">{letter}</div><div class="grade-text"><strong>Model Grade: {letter} — {label}</strong><br>{model_name} scored {metric_name} = {score_val:.4f}</div></div>'

MODEL_EXPLANATIONS={"Random Forest":"Random Forest combines many decision trees to reduce errors and handles complex non-linear patterns.","Decision Tree":"Decision Tree makes decisions by splitting data on the most informative features — easy to understand.","Linear":"Linear Regression achieved the best score — ideal when the relationship is roughly linear.","Ridge":"Ridge Regression adds a penalty on large coefficients, preventing overfitting when features are correlated.","Lasso":"Lasso automatically sets unimportant coefficients to zero, acting as a built-in feature selector.","KNN":"K-Nearest Neighbours topped the models — intuitive and effective for datasets with clear local structure.","Logistic":"Logistic Regression was the best classifier — estimates class probabilities, very fast and interpretable.","Naive Bayes":"Naive Bayes performed best — applies Bayes' theorem assuming feature independence, very fast.","SVR":"Support Vector Regression found the optimal ε-tube, making it robust to outliers and non-linear patterns.","SVM":"Support Vector Machine found the widest-margin boundary, handling complex non-linear separations via RBF kernel.","Polynomial":"Polynomial Regression captured curved trends by expanding features to higher degrees before fitting.","Gradient Boosting":"Gradient Boosting iteratively corrects its own errors — one of the most accurate algorithms on tabular data.","XGBoost":"XGBoost combines speed, regularisation, and handling of missing values for top accuracy."}

def render_model_explanation(model_name):
    text=MODEL_EXPLANATIONS.get(model_name,f"{model_name} achieved the highest score on the test set.")
    return f'<div class="problem-explain-box" style="border-left-color:var(--a2);margin-top:8px;"><strong>Why {model_name}?</strong><br>{text}</div>'

def render_problem_explanation(mode,target_col,n_unique):
    if mode=="regression": body=f"<strong>This is a Regression problem</strong> because the target column <code>{target_col}</code> contains continuous numeric values ({n_unique} unique values)."
    else: body=f"<strong>This is a Classification problem</strong> because the target column <code>{target_col}</code> contains discrete categories ({n_unique} unique classes)."
    return f'<div class="problem-explain-box">{body}</div>'

MEDALS=["🥇","🥈","🥉"]
def get_top3_features(model,feature_names):
    importances=None
    if hasattr(model,"feature_importances_"): importances=model.feature_importances_
    elif hasattr(model,"get_coefficients") and model.get_coefficients() is not None:
        coeffs=model.get_coefficients()
        if len(coeffs)>1: importances=np.abs(coeffs[1:len(feature_names)+1])
    elif hasattr(model,"W") and model.W is not None: importances=np.abs(model.W).max(axis=1)
    elif hasattr(model,"coef_"):
        c=model.coef_;importances=np.abs(c).mean(axis=0) if c.ndim>1 else np.abs(c)
    if importances is None or len(importances)!=len(feature_names): return []
    pairs=sorted(zip(feature_names,importances),key=lambda x:x[1],reverse=True);top3=pairs[:3];total=sum(abs(v) for _,v in pairs) or 1.0
    return [(MEDALS[i],name,f"{val/total*100:.1f}%") for i,(name,val) in enumerate(top3)]

def render_top3_features(top3):
    if not top3: return ""
    rows="".join(f'<div class="top3-item"><span class="top3-medal">{medal}</span><span class="top3-feat">{feat}</span><span class="top3-score">{score} importance</span></div>' for medal,feat,score in top3)
    return f'<div class="top3-box"><div class="top3-title">⬡ Top 3 Most Important Features</div>{rows}</div>'

def render_warning(title,body,level="moderate"):
    icons={"critical":"🔴","moderate":"⚠️","info":"ℹ️"};icon=icons.get(level,"⚠️")
    return f'<div class="warn-box warn-{level}"><div class="warn-box-icon">{icon}</div><div class="warn-box-body"><div class="warn-box-title">{title}</div>{body}</div></div>'

def data_warnings(data,numeric_cols):
    w=[]
    if len(data)<100: w.append(render_warning("Small Dataset",f"Only {len(data)} rows. Models may not generalise well.",level="critical"))
    miss_pct=data.isnull().sum().sum()/max(data.size,1)*100
    if miss_pct>20: w.append(render_warning("High Missing Values",f"{miss_pct:.1f}% of cells are missing. Consider imputation.",level="critical"))
    elif miss_pct>5: w.append(render_warning("Some Missing Values",f"{miss_pct:.1f}% of cells missing. Auto-filled with median/mode.",level="moderate"))
    return w

def overfitting_warning(gap):
    if gap>0.15: return render_warning("Possible Overfitting",f"Train–test gap = {gap:.4f}. Try reducing max_depth or adding more data.",level="moderate")
    return ""

def build_report_text(mode,best_name,best_res,feature_names,top3,health_score,health_label,last_pred=None,last_pred_cls=None,target_col="target"):
    lines=["="*60,"  INTELLIGENT ML SYSTEM v10 — ANALYSIS REPORT","="*60,f"\nAnalysis Mode   : {mode.capitalize()}",f"Target Column   : {target_col}",f"Data Health     : {health_score}/100 ({health_label})","\n--- BEST MODEL ---",f"Model           : {best_name}"]
    if mode=="regression": lines+=[f"R²  (Test)      : {best_res.get('R²','N/A'):.4f}",f"RMSE            : {best_res.get('RMSE','N/A'):.4f}",f"MAE             : {best_res.get('MAE','N/A'):.4f}"];grade,glabel,_=get_model_grade(best_res.get("R²",0))
    else: lines+=[f"Accuracy        : {best_res.get('Accuracy','N/A'):.4f}",f"F1 Score        : {best_res.get('F1','N/A'):.4f}"];grade,glabel,_=get_model_grade(best_res.get("F1",0))
    lines.append(f"Performance Grade: {grade} — {glabel}");lines.append("\n--- TOP FEATURES ---");lines+=[f"  {m}  {f}  ({s})" for m,f,s in top3] if top3 else ["  N/A"]
    lines.append("\n--- FEATURES USED ---");lines+=[f"  • {f}" for f in feature_names]
    if last_pred is not None: lines+=["\n--- LAST PREDICTION ---",f"  Predicted value : {last_pred:.4f}"]
    elif last_pred_cls is not None:
        cls_enc,prob=last_pred_cls;lines+=["\n--- LAST PREDICTION ---",f"  Predicted class : {cls_enc}"]
        if prob: lines.append(f"  Confidence      : {prob*100:.1f}%")
    lines.append("\n"+"="*60);return "\n".join(lines)

# ════════════════════════════════════════════════════════════════════════
# PRE-UPLOAD LANDING UI
# ════════════════════════════════════════════════════════════════════════

# Session state for uploader visibility
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# Landing CSS
st.markdown("""
<style>
/* ── Landing Wrapper ─────────────────────────────────────────────── */
.landing-hero {
    text-align: center;
    padding: 48px 20px 36px;
    position: relative;
}
.landing-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 2px;
    background: linear-gradient(90deg, transparent, #2d8fcb, #22a878, transparent);
}
.landing-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #22a878;
    background: #0a2318;
    border: 1px solid #1a5038;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 18px;
}
.landing-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #c4dff0;
    letter-spacing: -0.04em;
    line-height: 1.15;
    margin-bottom: 14px;
}
.landing-title span {
    background: linear-gradient(135deg, #2d8fcb, #22e8a0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.landing-sub {
    font-size: 0.95rem;
    color: #7ea8c4;
    max-width: 520px;
    margin: 0 auto 10px;
    line-height: 1.7;
}
.landing-stat-strip {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin: 24px 0 0;
    flex-wrap: wrap;
}
.landing-stat {
    text-align: center;
}
.landing-stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: #2d8fcb;
    line-height: 1;
}
.landing-stat-lbl {
    font-size: 0.65rem;
    color: #4a7a96;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Feature Cards ───────────────────────────────────────────────── */
@keyframes cardReveal {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.feat-card {
    background: linear-gradient(145deg, #0c1e32 0%, #091525 100%);
    border: 1px solid #0f2a40;
    border-radius: 14px;
    padding: 26px 22px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    cursor: default;
    animation: cardReveal 0.5s ease both;
    height: 100%;
    box-sizing: border-box;
}
.feat-card:hover {
    transform: translateY(-5px) scale(1.01);
    border-color: var(--card-accent, #2d8fcb);
    box-shadow: 0 16px 40px rgba(0,0,0,0.5), 0 0 0 1px var(--card-accent, #2d8fcb);
}
.feat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-accent, #2d8fcb);
    opacity: 0;
    transition: opacity 0.25s ease;
}
.feat-card:hover::before {
    opacity: 1;
}
.feat-card::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: var(--card-accent, #2d8fcb);
    opacity: 0.04;
    transition: opacity 0.25s ease;
}
.feat-card:hover::after {
    opacity: 0.10;
}
.feat-card-icon {
    font-size: 2.0rem;
    margin-bottom: 12px;
    display: block;
    line-height: 1;
    filter: drop-shadow(0 0 12px var(--card-accent, #2d8fcb));
}
.feat-card-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.60rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--card-accent, #2d8fcb);
    margin-bottom: 6px;
    opacity: 0.85;
}
.feat-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.0rem;
    font-weight: 700;
    color: #b8d8ea;
    margin-bottom: 8px;
    line-height: 1.2;
}
.feat-card-desc {
    font-size: 0.78rem;
    color: #6a94b0;
    line-height: 1.65;
    margin-bottom: 14px;
}
.feat-card-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}
.feat-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #5a8aaa;
}

/* ── Upload CTA ──────────────────────────────────────────────────── */
.upload-cta-wrapper {
    text-align: center;
    padding: 38px 20px 28px;
    position: relative;
}
.upload-cta-wrapper::before {
    content: '';
    position: absolute;
    bottom: 0; left: 50%; transform: translateX(-50%);
    width: 400px; height: 1px;
    background: linear-gradient(90deg, transparent, #0f2236, transparent);
}
.upload-hint {
    font-size: 0.74rem;
    color: #3a6a88;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
}

/* ── Upload Button Override ──────────────────────────────────────── */
.upload-btn-container .stButton > button {
    background: linear-gradient(135deg, #1a5a8a 0%, #0e3a5e 100%) !important;
    border: 1px solid #2d8fcb !important;
    color: #c4dff0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.10em !important;
    padding: 12px 40px !important;
    border-radius: 8px !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
}
.upload-btn-container .stButton > button:hover {
    background: linear-gradient(135deg, #2d8fcb 0%, #1a6aaa 100%) !important;
    box-shadow: 0 8px 28px rgba(45,143,203,0.35) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────
st.markdown("""
<div class="landing-hero">
  <div class="landing-badge">⬡ v10 · Production AI System</div>
  <div class="landing-title">Intelligent Data Analysis<br>&amp; <span>ML System</span></div>
  <div class="landing-sub">Upload your dataset to unlock powerful AI insights — automatic model selection, deep EDA, clustering, and prediction in one dashboard.</div>
  <div class="landing-stat-strip">
    <div class="landing-stat"><div class="landing-stat-num">9+</div><div class="landing-stat-lbl">Regression Models</div></div>
    <div class="landing-stat"><div class="landing-stat-num">7+</div><div class="landing-stat-lbl">Classifiers</div></div>
    <div class="landing-stat"><div class="landing-stat-num">4</div><div class="landing-stat-lbl">Clustering Algos</div></div>
    <div class="landing-stat"><div class="landing-stat-num">Auto</div><div class="landing-stat-lbl">Mode Detection</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Feature Cards Grid ────────────────────────────────────────────────
CARDS = [
    {
        "icon": "📊",
        "tag": "Module 1",
        "title": "Data Analysis",
        "desc": "Automatic EDA with statistical summaries, correlation matrices, outlier detection, and a data health score — all before writing a single line of code.",
        "pills": ["Correlations", "Outliers", "Health Score", "Missing Values"],
        "accent": "#2d8fcb",
    },
    {
        "icon": "🤖",
        "tag": "Module 2",
        "title": "ML Models",
        "desc": "Auto-detects regression vs classification. Trains 9+ regression and 7+ classification models simultaneously — Linear, Ridge, Lasso, SVM, Gradient Boosting, XGBoost and more.",
        "pills": ["Auto-detect", "9 Regressors", "7+ Classifiers", "One Click"],
        "accent": "#a060c8",
    },
    {
        "icon": "📈",
        "tag": "Module 4",
        "title": "Visualization",
        "desc": "Rich diagnostic charts for every model — actual vs predicted plots, residual analysis, confusion matrices, feature importances, and class probability distributions.",
        "pills": ["Residuals", "Confusion Matrix", "Feature Importance", "Charts"],
        "accent": "#22a878",
    },
    {
        "icon": "🧠",
        "tag": "Module 1–2",
        "title": "Smart Insights",
        "desc": "Auto-generated insights explain your data in plain English. Model grades (A–D), overfitting warnings, and actionable recommendations surfaced automatically.",
        "pills": ["Plain English", "Model Grade", "Overfit Alert", "Recommendations"],
        "accent": "#e0a844",
    },
    {
        "icon": "⚡",
        "tag": "Module 3",
        "title": "Performance",
        "desc": "Side-by-side model comparison tables with R², RMSE, F1, Accuracy, and overfitting gap. Best model auto-selected and highlighted with explainer cards.",
        "pills": ["R² & RMSE", "F1 Score", "Ranking Table", "Best Model"],
        "accent": "#44c8e0",
    },
    {
        "icon": "🔍",
        "tag": "Module 7",
        "title": "Clustering & Discovery",
        "desc": "Unsupervised learning suite: K-Means, DBSCAN (with noise detection), Hierarchical clustering with dendrogram, t-SNE 2D projections, and Apriori association rules.",
        "pills": ["K-Means", "DBSCAN", "t-SNE", "Apriori"],
        "accent": "#22e8a0",
    },
]

col_sets = [st.columns(3), st.columns(3)]
for idx, card in enumerate(CARDS):
    row = idx // 3
    col = idx % 3
    with col_sets[row][col]:
        pills_html = "".join(f'<span class="feat-pill">{p}</span>' for p in card["pills"])
        st.markdown(f"""
<div class="feat-card" style="--card-accent:{card['accent']};" 
     data-delay="{idx * 0.08}s">
  <span class="feat-card-icon">{card['icon']}</span>
  <div class="feat-card-tag">{card['tag']}</div>
  <div class="feat-card-title">{card['title']}</div>
  <div class="feat-card-desc">{card['desc']}</div>
  <div class="feat-card-pills">{pills_html}</div>
</div>
""", unsafe_allow_html=True)

# ── Upload CTA ────────────────────────────────────────────────────────
st.markdown('<div class="upload-cta-wrapper">', unsafe_allow_html=True)
st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#3a6a88;letter-spacing:0.18em;text-transform:uppercase;text-align:center;margin-bottom:18px;">Ready to begin?</p>', unsafe_allow_html=True)

cta_l, cta_c, cta_r = st.columns([2, 1.4, 2])
with cta_c:
    st.markdown('<div class="upload-btn-container">', unsafe_allow_html=True)
    if st.button("⬆  Upload Dataset", use_container_width=True, key="cta_upload_btn"):
        st.session_state.show_uploader = True
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="upload-hint" style="text-align:center;">Supports CSV files · All processing happens locally</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ════ GATE: stop here until user clicks "Upload Dataset" ════
if not st.session_state.show_uploader:
    st.stop()

# ════ POST-LANDING HEADER ════
st.markdown('<div class="hero-title">Intelligent Data Analysis & ML System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">v10 · Auto-Detects Regression vs Classification · 9 Regression Models · 7+ Classification Models · DBSCAN · Hierarchical · t-SNE · Manual & Library Implementations</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"], label_visibility="collapsed")
if uploaded_file is None:
    st.info("Upload a CSV file above to begin.")
    st.stop()

data         = pd.read_csv(uploaded_file)
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
cat_cols     = data.select_dtypes(exclude=np.number).columns.tolist()
all_cols     = data.columns.tolist()
total_miss   = int(data.isnull().sum().sum())


# MODULE 1 — SMART EDA
st.markdown(module_banner("1","Smart Exploratory Data Analysis","Auto-generated insights before model training"),unsafe_allow_html=True)
for w in data_warnings(data,numeric_cols): st.markdown(w,unsafe_allow_html=True)
st.markdown("## Dataset Overview")
cards='<div class="card-grid">'
for v,l in [(f"{data.shape[0]:,}","Rows"),(str(data.shape[1]),"Columns"),(str(len(numeric_cols)),"Numeric"),(str(len(cat_cols)),"Categorical"),(f"{total_miss:,}","Missing")]:
    cards+=stat_card(v,l)
cards+='</div>';st.markdown(cards,unsafe_allow_html=True)
cl,cr=st.columns(2)
with cl:
    st.markdown("#### Dataset Preview (First 5 Rows)");st.dataframe(data.head(5),use_container_width=True)
with cr:
    st.markdown("#### Missing Value Summary")
    mv=pd.DataFrame({"Column":data.columns,"Missing":data.isnull().sum().values,"% Missing":(data.isnull().sum()/len(data)*100).round(2).values});mv=mv[mv["Missing"]>0]
    if mv.empty: st.success("No missing values found.")
    else: st.dataframe(mv,use_container_width=True,hide_index=True)
stats_rows=[]
if numeric_cols:
    st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True);st.markdown("## Statistical Summary")
    for c in numeric_cols:
        cd=data[c].dropna();stats_rows.append({"Column":c,"Mean":round(cd.mean(),4),"Median":round(cd.median(),4),"Std Dev":round(cd.std(),4),"Min":round(cd.min(),4),"Max":round(cd.max(),4),"Skewness":round(cd.skew(),4)})
    st.dataframe(pd.DataFrame(stats_rows),use_container_width=True,hide_index=True)
corr_matrix=corr_with_target=None
if len(numeric_cols)>=2:
    st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True);st.markdown("## Correlation Heatmap")
    corr_matrix=data[numeric_cols].corr()
    fig_c,ax_c=plt.subplots(figsize=(max(6,len(numeric_cols)*0.9),max(5,len(numeric_cols)*0.8)))
    sns.heatmap(corr_matrix,annot=True,fmt=".2f",cmap=sns.diverging_palette(220,10,as_cmap=True),center=0,vmin=-1,vmax=1,square=True,linewidths=0.4,linecolor=GRID_CLR,annot_kws={"size":8,"color":"#c9d8e8"},cbar_kws={"shrink":0.8},ax=ax_c)
    ax_c.set_title("Feature Correlation Matrix",fontsize=11,color="#7eb8d4");apply_theme(fig_c,ax_c);ax_c.tick_params(colors=TEXT_CLR,labelsize=8)
    fig_c.patch.set_facecolor(PLOT_BG);fig_c.tight_layout();st.pyplot(fig_c);plt.close()
    st.markdown("### Correlation Insights");found=False;seen=set()
    for i,a in enumerate(numeric_cols):
        for j,b in enumerate(numeric_cols):
            if i>=j: continue
            k=tuple(sorted([a,b]))
            if k in seen: continue
            seen.add(k);r=corr_matrix.loc[a,b]
            if r>0.7: st.markdown(f'<div class="insight-pos">📈 {a} ↔ {b} (r = {r:.2f})</div>',unsafe_allow_html=True);found=True
            elif r<-0.7: st.markdown(f'<div class="insight-neg">📉 {a} ↔ {b} (r = {r:.2f})</div>',unsafe_allow_html=True);found=True
    if not found: st.markdown('<div class="insight-neu">ℹ No strong correlations (|r| > 0.7) detected.</div>',unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True);st.markdown("## Target Correlation Analysis")
target_col_eda=st.selectbox("Select Target Column for EDA",all_cols,key="eda_target")
eda_target_is_numeric=target_col_eda in numeric_cols
if eda_target_is_numeric:
    other_cols=[c for c in numeric_cols if c!=target_col_eda]
    if other_cols and corr_matrix is not None and target_col_eda in corr_matrix.columns:
        corr_with_target=corr_matrix[target_col_eda].drop(target_col_eda).sort_values()
        cp,cn=st.columns(2)
        with cp:
            st.markdown("#### Strongest Positive Predictors")
            for feat,r in corr_with_target.tail(3)[::-1].items(): st.markdown(f'<div class="insight-pos">📈 {feat} — r = {r:.4f}</div>',unsafe_allow_html=True)
        with cn:
            st.markdown("#### Strongest Negative Predictors")
            for feat,r in corr_with_target.head(3).items(): st.markdown(f'<div class="insight-neg">📉 {feat} — r = {r:.4f}</div>',unsafe_allow_html=True)
        fig_t,ax_t=plt.subplots(figsize=(8,max(3,len(other_cols)*0.45)))
        ax_t.barh(corr_with_target.index,corr_with_target.values,color=[ACCENT2 if v>=0 else ACCENT3 for v in corr_with_target.values],edgecolor=AXES_BG,linewidth=0.5)
        ax_t.axvline(0,color=TEXT_CLR,linewidth=0.8);ax_t.set_xlabel("Correlation with Target");ax_t.set_title(f"Feature Correlations with '{target_col_eda}'")
        apply_theme(fig_t,ax_t);fig_t.tight_layout();st.pyplot(fig_t);plt.close()
else:
    st.markdown(f'<div class="insight-cls">🏷️ <strong>{target_col_eda}</strong> is a categorical target.</div>',unsafe_allow_html=True)
    value_counts=data[target_col_eda].value_counts();n_classes=len(value_counts);palette=[ACCENT1,ACCENT2,ACCENT3,ACCENT4,ACCENT5,ACCENT6]
    fig_vd,ax_vd=plt.subplots(figsize=(max(5,n_classes*1.2),3.5))
    ax_vd.bar(value_counts.index.astype(str),value_counts.values,color=[palette[i%len(palette)] for i in range(len(value_counts))],edgecolor=AXES_BG)
    for i,(lbl,cnt) in enumerate(value_counts.items()):
        ax_vd.text(i,cnt+max(value_counts)*0.01,f"{cnt}\n({cnt/len(data)*100:.1f}%)",ha="center",va="bottom",fontsize=8,color=TEXT_CLR,fontfamily="monospace")
    ax_vd.set_title(f"Class Distribution — '{target_col_eda}'");ax_vd.set_ylabel("Count");apply_theme(fig_vd,ax_vd);fig_vd.tight_layout();st.pyplot(fig_vd);plt.close()
st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True);st.markdown("## Outlier Detection (IQR Method)")
outlier_counts={}
if numeric_cols:
    for row_i in range(int(np.ceil(len(numeric_cols)/2))):
        st_cols=st.columns(2)
        for ci in range(2):
            fi=row_i*2+ci
            if fi>=len(numeric_cols): break
            cn_name=numeric_cols[fi];cd=data[cn_name].dropna();mask,q1,q3,iqr,lo_b,hi_b=detect_outliers_iqr(cd);n_out=int(mask.sum());outlier_counts[cn_name]=n_out
            with st_cols[ci]:
                fig_bp3,ax_bp3=plt.subplots(figsize=(5.5,2.6))
                ax_bp3.boxplot(cd,vert=False,patch_artist=True,widths=0.55,boxprops=dict(facecolor="#0e2840",color=ACCENT1,linewidth=1.2),medianprops=dict(color=ACCENT2,linewidth=2),whiskerprops=dict(color=TEXT_CLR,linewidth=1,linestyle="--"),capprops=dict(color=ACCENT1,linewidth=1.5),flierprops=dict(marker="o",color=ACCENT3,alpha=0.7,markersize=4,markeredgewidth=0))
                ax_bp3.set_title(cn_name,fontsize=10);ax_bp3.set_xlabel("Value",fontsize=8);ax_bp3.set_yticks([]);apply_theme(fig_bp3,ax_bp3);fig_bp3.tight_layout(pad=1.2);st.pyplot(fig_bp3);plt.close()
                is_bad=n_out>0;badge_cls="outlier-found" if is_bad else "outlier-clean";badge_txt=f"{n_out} outlier{'s' if n_out!=1 else ''} detected" if is_bad else "No outliers"
                st.markdown(f'<div class="outlier-badge {badge_cls}">{"⚠" if is_bad else "✓"} {badge_txt}&nbsp;&nbsp;|&nbsp;&nbsp;IQR={iqr:.3f} [{lo_b:.3f},{hi_b:.3f}]</div>',unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True);st.markdown("## Automated Data Insights Summary")
def build_summary(data,numeric_cols,cat_cols,total_miss,corr_with_target,outlier_counts,stats_rows):
    items=[]
    items.append(("📐","Dataset Dimensions",f"{data.shape[0]:,} rows × {data.shape[1]} cols ({len(numeric_cols)} numeric, {len(cat_cols)} categorical)"))
    if total_miss==0: items.append(("✅","Missing Values","Complete — no missing cells detected"))
    else: items.append(("⚠","Missing Values",f"{total_miss:,} missing ({round(total_miss/data.size*100,2)}%) — worst: '{data.isnull().sum().idxmax()}'"))
    if corr_with_target is not None and len(corr_with_target):
        best=corr_with_target.abs().idxmax();r=corr_with_target[best];items.append(("🔗","Strongest Predictor",f"'{best}' — {'positive' if r>0 else 'negative'} (r = {r:.4f})"))
    if outlier_counts:
        total_o=sum(outlier_counts.values());cols_hit=[c for c,n in outlier_counts.items() if n>0]
        if total_o==0: items.append(("🧹","Outlier Summary","Clean — no outliers detected"))
        else: items.append(("🔴","Outlier Summary",f"{total_o} outlier(s) in {len(cols_hit)} column(s) — worst: '{max(outlier_counts,key=outlier_counts.get)}' ({outlier_counts[max(outlier_counts,key=outlier_counts.get)]})"))
    if stats_rows:
        hs=[(r["Column"],r["Skewness"]) for r in stats_rows if abs(r["Skewness"])>1.0]
        if not hs: items.append(("📊","Distribution Shape","All features approximately symmetric"))
        else: items.append(("📊","Highly Skewed Features",f"{len(hs)} feature(s): {', '.join(f'{c} ({s:+.2f})' for c,s in hs[:5])}"))
    if cat_cols: items.append(("🏷️","Categorical Columns",f"{len(cat_cols)}: {', '.join(cat_cols)}"))
    low_v=[c for c in numeric_cols if data[c].std()<1e-6]
    items.append(("⚡","Near-Constant Columns",f"{len(low_v)}: {', '.join(low_v)}" if low_v else "None — all columns have meaningful variance"))
    n_dup=int(data.duplicated().sum());items.append(("🔁","Duplicate Rows",f"{n_dup} duplicate row(s) detected" if n_dup else "None detected"))
    return items
summary_items=build_summary(data,numeric_cols,cat_cols,total_miss,corr_with_target,outlier_counts,stats_rows)
rows_html="".join(summary_row(ic,k,v) for ic,k,v in summary_items)
st.markdown(f'<div class="summary-box"><div class="summary-title">⬡ Auto-Generated EDA Summary</div>{rows_html}</div>',unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True);st.markdown("## Data Health Score")
hs,hbd,hlabel,hcss=compute_health_score(data,numeric_cols,outlier_counts);st.markdown(render_health_score(hs,hbd,hlabel,hcss),unsafe_allow_html=True)
st.session_state["_health_score"]=hs;st.session_state["_health_label"]=hlabel
bullets=build_english_summary(data,numeric_cols,cat_cols,total_miss,outlier_counts,corr_with_target,target_col_eda);st.markdown(render_english_summary(bullets),unsafe_allow_html=True)

# MODULE 2 — ML DASHBOARD
st.markdown(module_banner("2","Machine Learning Dashboard","Auto-detects Regression or Classification · 9 Regression + 7 Classification Models"),unsafe_allow_html=True)
st.markdown("## Data Configuration")
if cat_cols: st.info(f"🏷️ Categorical columns: {', '.join(cat_cols)}. Auto label-encoded.")
dc1,dc2=st.columns(2)
with dc1: ml_target=st.selectbox("Target Column (Y)",all_cols,key="ml_target")
with dc2: avail_feats=[c for c in all_cols if c!=ml_target];ml_features=st.multiselect("Input Features (X)",avail_feats,default=avail_feats,key="ml_features")
if ml_features:
    tags="".join(f'<span style="background:#1a0d28;border:1px solid #5a2a7a;color:#c890f0;font-size:0.65rem;padding:2px 7px;border-radius:3px;font-family:JetBrains Mono,monospace;margin:2px;">{f} 🏷️ cat</span> ' if f in cat_cols else f'<span style="background:#0a1a2a;border:1px solid #1a4060;color:#6dc8f0;font-size:0.65rem;padding:2px 7px;border-radius:3px;font-family:JetBrains Mono,monospace;margin:2px;">{f} # num</span> ' for f in ml_features)
    st.markdown(f'<div style="margin:6px 0 12px;">{tags}</div>',unsafe_allow_html=True)
oc1,oc2=st.columns(2)
with oc1: split_pct=st.slider("Training Data %",60,90,80,5)
with oc2: apply_std=st.checkbox("Apply Z-score Standardization",value=True,help="Recommended — especially for SVR, SVM, KNN, Logistic")
ml_mode=detect_mode(data[ml_target]);n_unique_tgt=data[ml_target].nunique()
if ml_mode=="regression":
    st.markdown(f'<div class="mode-reg"><div class="mode-icon">📈</div><div class="mode-tag">Auto-Detected Mode</div><div class="mode-name">Regression Mode</div><div class="mode-desc">Target <strong>{ml_target}</strong> is continuous numeric ({n_unique_tgt} unique values) — training 9 regression models.</div></div>',unsafe_allow_html=True)
else:
    classes_preview=data[ml_target].dropna().unique();classes_str=", ".join([str(c) for c in classes_preview[:8]])
    if len(classes_preview)>8: classes_str+="…"
    st.markdown(f'<div class="mode-cls"><div class="mode-icon">🏷️</div><div class="mode-tag">Auto-Detected Mode</div><div class="mode-name">Classification Mode</div><div class="mode-desc">Target <strong>{ml_target}</strong> has {n_unique_tgt} classes [{classes_str}] — training {"8" if XGBOOST_AVAILABLE else "7"} classification models.</div></div>',unsafe_allow_html=True)
st.markdown(render_problem_explanation(ml_mode,ml_target,n_unique_tgt),unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True);st.markdown("## Model Hyperparameters")
hp_cols=st.columns(4)
with hp_cols[0]:
    st.markdown('<div class="model-config-box"><div class="model-config-title">KNN</div>',unsafe_allow_html=True);knn_k=st.slider("K neighbours",1,30,5,1,key="knn_k");st.markdown('</div>',unsafe_allow_html=True)
with hp_cols[1]:
    st.markdown('<div class="model-config-box"><div class="model-config-title">SVR / SVM</div>',unsafe_allow_html=True);svm_c=st.slider("Regularisation C",0.01,10.0,1.0,0.01,key="svm_c");st.markdown('</div>',unsafe_allow_html=True)
with hp_cols[2]:
    st.markdown('<div class="model-config-box"><div class="model-config-title">Polynomial Regression</div>',unsafe_allow_html=True);poly_degree=st.slider("Degree",2,3,2,1,key="poly_degree");st.markdown('</div>',unsafe_allow_html=True)
with hp_cols[3]:
    st.markdown('<div class="model-config-box"><div class="model-config-title">Gradient Boosting</div>',unsafe_allow_html=True);gb_n=st.slider("N estimators",50,300,100,50,key="gb_n");gb_lr=st.slider("Learning rate",0.01,0.3,0.1,0.01,key="gb_lr");st.markdown('</div>',unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)
train_btn=st.button("🚀  Train All Models",use_container_width=True)
if train_btn:
    if not ml_features: st.warning("Select at least one input feature.");st.stop()
    with st.spinner(f"Training all models in {ml_mode} mode…"):
        np.random.seed(42);feature_encoders={};X_parts=[]
        for feat in ml_features:
            col=data[feat]
            if col.dtype==object or str(col.dtype) in ("string","category"):
                fe=LabelEncoder();mode_val=col.mode()[0] if not col.mode().empty else "missing";filled=col.fillna(mode_val).astype(str);encoded=fe.fit_transform(filled).astype(float);feature_encoders[feat]=fe;X_parts.append(encoded.reshape(-1,1))
            else: X_parts.append(col.fillna(col.median()).values.astype(float).reshape(-1,1))
        X_raw=np.hstack(X_parts);xmu=xsg=None
        if apply_std: xmu=X_raw.mean(axis=0);xsg=X_raw.std(axis=0);xsg[xsg==0]=1.0;X=(X_raw-xmu)/xsg
        else: X=X_raw.copy()
        le=None
        if ml_mode=="regression": y=data[ml_target].fillna(data[ml_target].median()).values.astype(float);classes=None
        else:
            le=LabelEncoder();raw_y=data[ml_target].fillna(data[ml_target].mode()[0]).astype(str);y=le.fit_transform(raw_y);classes=le.classes_
        n=len(X);split_n=int(n*split_pct/100);idx=np.random.permutation(n)
        X_tr,X_te=X[idx[:split_n]],X[idx[split_n:]];y_tr,y_te=y[idx[:split_n]],y[idx[split_n:]]
        results={};fitted={}
        if ml_mode=="regression":
            models={"Linear":ManualLinearRegression(),"Ridge":ManualRidgeRegression(lam=1.0),"Lasso":ManualLassoRegression(lam=0.1),"KNN":ManualKNNRegression(k=knn_k),"Decision Tree":DecisionTreeRegressor(max_depth=5,random_state=42),"Random Forest":RandomForestRegressor(n_estimators=100,max_depth=5,random_state=42),"SVR":SVR(C=svm_c,kernel="rbf",epsilon=0.1),"Polynomial":ManualPolynomialRegression(degree=poly_degree),"Gradient Boosting":GradientBoostingRegressor(n_estimators=gb_n,learning_rate=gb_lr,max_depth=4,random_state=42)}
            for mname,m in models.items():
                try:
                    m.fit(X_tr,y_tr);ev=reg_evaluate(m,X_te,y_te);ev["r2_train"]=calc_r2(y_tr,m.predict(X_tr));results[mname]=ev;fitted[mname]=m
                except Exception as ex: st.warning(f"⚠ {mname} failed: {ex}")
            best_name=max(results,key=lambda k:results[k]["R²"])
        else:
            avg="binary" if len(classes)==2 else "weighted"
            cls_models_dict={"Logistic":ManualLogisticRegression(lr=0.1,max_iter=500),"KNN":ManualKNNClassifier(k=knn_k),"Naive Bayes":ManualNaiveBayes(),"Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),"Random Forest":RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42),"SVM":SVC(C=svm_c,kernel="rbf",probability=True,random_state=42),"Gradient Boosting":GradientBoostingClassifier(n_estimators=gb_n,learning_rate=gb_lr,max_depth=4,random_state=42)}
            if XGBOOST_AVAILABLE: cls_models_dict["XGBoost"]=XGBClassifier(n_estimators=gb_n,learning_rate=gb_lr,max_depth=4,random_state=42,eval_metric="logloss",verbosity=0)
            for mname,m in cls_models_dict.items():
                try:
                    m.fit(X_tr,y_tr);yp_te=m.predict(X_te);yp_tr=m.predict(X_tr)
                    results[mname]={"Accuracy":float(accuracy_score(y_te,yp_te)),"Precision":float(precision_score(y_te,yp_te,average=avg,zero_division=0)),"Recall":float(recall_score(y_te,yp_te,average=avg,zero_division=0)),"F1":float(f1_score(y_te,yp_te,average=avg,zero_division=0)),"Acc_train":float(accuracy_score(y_tr,yp_tr)),"CM":confusion_matrix(y_te,yp_te),"CR":classification_report(y_te,yp_te,output_dict=True,zero_division=0),"yp":yp_te}
                    fitted[mname]=m
                except Exception as ex: st.warning(f"⚠ {mname} failed: {ex}")
            best_name=max(results,key=lambda k:results[k]["F1"])
        st.session_state.update({"_trained":True,"_mode":ml_mode,"_feat":ml_features,"_target":ml_target,"_results":results,"_fitted":fitted,"_best":best_name,"_X_te":X_te,"_y_te":y_te,"_X_tr":X_tr,"_y_tr":y_tr,"_data_snap":data,"_apply_std":apply_std,"_xmu":xmu,"_xsg":xsg,"_le":le,"_classes":classes,"_feature_encoders":feature_encoders,"_last_pred":None,"_last_pred_cls":None})
    mode_label="Regression" if ml_mode=="regression" else "Classification"
    best_metric=f"R² = {results[best_name]['R²']:.4f}" if ml_mode=="regression" else f"F1 = {results[best_name]['F1']:.4f}"
    st.success(f"✓ All models trained [{mode_label}]. Best: **{best_name}** ({best_metric})")


# POST-TRAINING DASHBOARD
if st.session_state.get("_trained"):
    results=st.session_state["_results"];fitted=st.session_state["_fitted"];best_name=st.session_state["_best"]
    t_feat=st.session_state["_feat"];t_tgt=st.session_state["_target"];X_te=st.session_state["_X_te"]
    y_te=st.session_state["_y_te"];X_tr=st.session_state["_X_tr"];y_tr=st.session_state["_y_tr"]
    dsnap=st.session_state["_data_snap"];a_std=st.session_state["_apply_std"];xmu=st.session_state["_xmu"]
    xsg=st.session_state["_xsg"];le=st.session_state["_le"];classes=st.session_state["_classes"]
    ml_mode=st.session_state["_mode"];feat_enc=st.session_state.get("_feature_encoders",{})
    best_res=results[best_name];MODEL_COLORS=REG_MODEL_COLORS if ml_mode=="regression" else CLS_MODEL_COLORS
    st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)

    if ml_mode=="regression":
        st.markdown(f'<div class="best-model-banner"><div class="bm-label">🏆 Best Regression Model</div><div class="bm-name">{best_name}</div><div class="bm-stats">R² = {best_res["R²"]:.4f} &nbsp;|&nbsp; RMSE = {best_res["RMSE"]:.4f} &nbsp;|&nbsp; MAE = {best_res["MAE"]:.4f}</div></div>',unsafe_allow_html=True)
        render_explainer_card(best_name,"regression");st.markdown(render_grade_banner(best_name,best_res["R²"],"R²"),unsafe_allow_html=True)
        st.markdown(render_model_explanation(best_name),unsafe_allow_html=True);top3=get_top3_features(fitted[best_name],t_feat);st.markdown(render_top3_features(top3),unsafe_allow_html=True)
        st.markdown(module_banner("3","Model Comparison","All regression models ranked by Test R²"),unsafe_allow_html=True)
        comp_rows=[]
        for mname,ev in results.items():
            gap=ev["r2_train"]-ev["R²"];comp_rows.append({"Model":mname,"R² (Test)":round(ev["R²"],4),"R² (Train)":round(ev["r2_train"],4),"RMSE":round(ev["RMSE"],4),"MAE":round(ev["MAE"],4),"Overfit Gap":round(gap,4),"Grade":get_model_grade(ev["R²"])[0]})
        comp_df=pd.DataFrame(comp_rows).sort_values("R² (Test)",ascending=False).reset_index(drop=True)
        comp_df.insert(0,"Rank",range(1,len(comp_df)+1));comp_df.insert(1,"🏆",comp_df["Model"].apply(lambda m:"★" if m==best_name else ""))
        st.dataframe(comp_df,use_container_width=True,hide_index=True);render_all_explainers("regression")
        sorted_names=comp_df["Model"].tolist();bar_colors=[MODEL_COLORS.get(m,ACCENT1) for m in sorted_names]
        edge_colors=["#ffd700" if m==best_name else AXES_BG for m in sorted_names];edge_widths=[2.0 if m==best_name else 0.5 for m in sorted_names]
        fig_cmp,ax_cmp=plt.subplots(figsize=(max(10,len(sorted_names)*1.3),4))
        r2_vals=[results[m]["R²"] for m in sorted_names];bars=ax_cmp.bar(sorted_names,r2_vals,color=bar_colors,edgecolor=edge_colors,linewidth=edge_widths)
        ax_cmp.axhline(0,color=TEXT_CLR,linewidth=0.6)
        for bar,val in zip(bars,r2_vals): ax_cmp.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f"{val:.3f}",ha="center",va="bottom",fontsize=8,color=TEXT_CLR)
        ax_cmp.set_ylabel("R² Score");ax_cmp.set_title("Regression Model Comparison — Test R²");ax_cmp.tick_params(axis="x",rotation=20,labelsize=8)
        apply_theme(fig_cmp,ax_cmp);fig_cmp.tight_layout();st.pyplot(fig_cmp);plt.close()
        st.markdown(module_banner("4","Per-Model Diagnostics","Actual vs Predicted · Residuals · Feature Importance"),unsafe_allow_html=True)
        for mname in sorted_names:
            ev=results[mname];yp_te=ev["yp"];resid=y_te-yp_te;r2_tr=ev["r2_train"];r2_te=ev["R²"];gap=r2_tr-r2_te
            lc=MODEL_COLORS.get(mname,ACCENT1);is_best=mname==best_name
            badge_h='&nbsp;<span style="background:#1a4a28;color:#4de8a0;font-size:0.65rem;padding:2px 7px;border-radius:3px;">BEST</span>' if is_best else ""
            st.markdown(f'<div style="border-left:3px solid {lc};padding:4px 12px;margin:18px 0 8px;font-family:JetBrains Mono,monospace;font-size:0.95rem;color:{lc};font-weight:700;">{mname}{badge_h}</div>',unsafe_allow_html=True)
            adj_r2=calc_adj_r2(r2_te,len(y_te),len(t_feat));mc='<div class="card-grid">'
            for v,l in [(f"{r2_te:.4f}","R² Test"),(f"{r2_tr:.4f}","R² Train"),(f"{ev['RMSE']:.4f}","RMSE"),(f"{ev['MAE']:.4f}","MAE"),(f"{adj_r2:.4f}","Adj R²"),(f"{gap:.4f}","Overfit Gap")]: mc+=metric_card(v,l)
            mc+='</div>';st.markdown(mc,unsafe_allow_html=True);st.markdown(overfitting_warning(gap),unsafe_allow_html=True)
            pg1,pg2=st.columns(2)
            with pg1:
                fig1,ax1=plt.subplots(figsize=(5.5,4));ax1.scatter(y_te,yp_te,s=20,color=lc,alpha=0.6,edgecolors="none")
                lo,hi=min(y_te.min(),yp_te.min()),max(y_te.max(),yp_te.max());ax1.plot([lo,hi],[lo,hi],color=ACCENT3,linewidth=1.5,linestyle="--",label="Ideal")
                ax1.set_xlabel("Actual");ax1.set_ylabel("Predicted");ax1.set_title(f"{mname} — Actual vs Predicted",fontsize=9);ax1.legend(fontsize=7,framealpha=0.2,labelcolor=TEXT_CLR)
                apply_theme(fig1,ax1);fig1.tight_layout();st.pyplot(fig1);plt.close()
            with pg2:
                fig2,ax2=plt.subplots(figsize=(5.5,4));ax2.scatter(yp_te,resid,s=20,color=ACCENT4,alpha=0.6,edgecolors="none");ax2.axhline(0,color=ACCENT3,linewidth=1.4,linestyle="--")
                ax2.set_xlabel("Predicted");ax2.set_ylabel("Residuals");ax2.set_title(f"{mname} — Residual Plot",fontsize=9)
                apply_theme(fig2,ax2);fig2.tight_layout();st.pyplot(fig2);plt.close()
            m_obj=fitted[mname];fi_data=None
            if hasattr(m_obj,"feature_importances_"): fi_data=m_obj.feature_importances_;fi_title=f"{mname} — Feature Importance (Gini)"
            elif hasattr(m_obj,"get_coefficients") and m_obj.get_coefficients() is not None:
                coeffs=m_obj.get_coefficients();fi_data=np.abs(coeffs[1:len(t_feat)+1]);fi_title=f"{mname} — Feature Importance (|coefficient|)"
            elif hasattr(m_obj,"coef_"): c=m_obj.coef_;fi_data=np.abs(c).flatten()[:len(t_feat)];fi_title=f"{mname} — Feature Importance (|coefficient|)"
            if fi_data is not None and len(fi_data)==len(t_feat):
                fi_df=pd.DataFrame({"Feature":t_feat,"Importance":fi_data}).sort_values("Importance",ascending=True)
                fig_fi,ax_fi=plt.subplots(figsize=(7,max(2.5,len(t_feat)*0.45)));ax_fi.barh(fi_df["Feature"],fi_df["Importance"],color=lc,edgecolor=AXES_BG)
                ax_fi.set_title(fi_title);apply_theme(fig_fi,ax_fi);fig_fi.tight_layout();st.pyplot(fig_fi);plt.close()
            st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)
        st.markdown(module_banner("5","Regression Assumptions",f"Evaluated on best model: {best_name}"),unsafe_allow_html=True)
        yp_best=best_res["yp"];resid_best=y_te-yp_best;lin_ok=best_res["R²"]>0.3
        dw=float(np.sum(np.diff(resid_best)**2)/np.sum(resid_best**2));ind_ok=1.5<dw<2.5
        _,hom_p=stats.pearsonr(np.abs(resid_best),yp_best);hom_ok=hom_p>0.05
        samp=resid_best[:200] if len(resid_best)>200 else resid_best;_,nor_p=stats.shapiro(samp);nor_ok=nor_p>0.05
        st.markdown(assumption_row("1. Linearity","R² > 0.3",lin_ok,f"R² = {best_res['R²']:.4f}"),unsafe_allow_html=True)
        st.markdown(assumption_row("2. Independence","Durbin-Watson test",ind_ok,f"DW = {dw:.4f}"),unsafe_allow_html=True)
        st.markdown(assumption_row("3. Homoscedasticity","Constant residual variance",hom_ok,f"p = {hom_p:.4f}"),unsafe_allow_html=True)
        st.markdown(assumption_row("4. Normality of Residuals","Shapiro-Wilk test",nor_ok,f"p = {nor_p:.4f}"),unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="best-model-banner-cls"><div class="bm-label">🏆 Best Classification Model</div><div class="bm-name">{best_name}</div><div class="bm-stats">Accuracy = {best_res["Accuracy"]:.4f} &nbsp;|&nbsp; F1 = {best_res["F1"]:.4f}</div></div>',unsafe_allow_html=True)
        render_explainer_card(best_name,"classification");st.markdown(render_grade_banner(best_name,best_res["F1"],"F1"),unsafe_allow_html=True)
        st.markdown(render_model_explanation(best_name),unsafe_allow_html=True);top3=get_top3_features(fitted[best_name],t_feat);st.markdown(render_top3_features(top3),unsafe_allow_html=True)
        st.markdown(module_banner("3","Model Comparison","All classification models ranked by F1 Score"),unsafe_allow_html=True)
        comp_rows=[]
        for mname,ev in results.items():
            gap=ev["Acc_train"]-ev["Accuracy"];comp_rows.append({"Model":mname,"Accuracy":round(ev["Accuracy"],4),"Precision":round(ev["Precision"],4),"Recall":round(ev["Recall"],4),"F1 Score":round(ev["F1"],4),"Train Acc":round(ev["Acc_train"],4),"Overfit Gap":round(gap,4),"Grade":get_model_grade(ev["F1"])[0]})
        comp_df=pd.DataFrame(comp_rows).sort_values("F1 Score",ascending=False).reset_index(drop=True)
        comp_df.insert(0,"Rank",range(1,len(comp_df)+1));comp_df.insert(1,"🏆",comp_df["Model"].apply(lambda m:"★" if m==best_name else ""))
        st.dataframe(comp_df,use_container_width=True,hide_index=True);render_all_explainers("classification")
        sorted_names=comp_df["Model"].tolist();bar_colors=[CLS_MODEL_COLORS.get(m,ACCENT5) for m in sorted_names]
        edge_colors=["#ffd700" if m==best_name else AXES_BG for m in sorted_names];edge_widths=[2.0 if m==best_name else 0.5 for m in sorted_names]
        metrics_to_plot=[("F1 Score",[results[m]["F1"] for m in sorted_names]),("Accuracy",[results[m]["Accuracy"] for m in sorted_names]),("Precision",[results[m]["Precision"] for m in sorted_names]),("Recall",[results[m]["Recall"] for m in sorted_names])]
        fig_4,axes_4=plt.subplots(1,4,figsize=(max(16,len(sorted_names)*2.2),4))
        for ax,(metric_name,vals) in zip(axes_4,metrics_to_plot):
            bars=ax.bar(sorted_names,vals,color=bar_colors,edgecolor=edge_colors,linewidth=edge_widths);ax.set_ylim(0,1.1)
            for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,f"{val:.3f}",ha="center",va="bottom",fontsize=7,color=TEXT_CLR)
            ax.set_title(metric_name,fontsize=10);ax.set_ylabel("Score",fontsize=8);ax.tick_params(axis="x",rotation=20,labelsize=7);apply_theme(fig_4,ax)
        fig_4.suptitle("Classification Model Comparison",fontsize=11,color="#7eb8d4");fig_4.tight_layout();st.pyplot(fig_4);plt.close()
        st.markdown(module_banner("4","Per-Model Diagnostics","Metrics · Confusion Matrix · Classification Report"),unsafe_allow_html=True)
        for mname in sorted_names:
            ev=results[mname];lc=CLS_MODEL_COLORS.get(mname,ACCENT5);is_best=mname==best_name
            badge_h='&nbsp;<span style="background:#1a0d28;color:#c890f0;font-size:0.65rem;padding:2px 7px;border-radius:3px;">BEST</span>' if is_best else ""
            st.markdown(f'<div style="border-left:3px solid {lc};padding:4px 12px;margin:18px 0 8px;font-family:JetBrains Mono,monospace;font-size:0.95rem;color:{lc};font-weight:700;">{mname}{badge_h}</div>',unsafe_allow_html=True)
            mc='<div class="card-grid">';gap=ev["Acc_train"]-ev["Accuracy"]
            for v,l in [(f"{ev['Accuracy']:.4f}","Accuracy"),(f"{ev['Precision']:.4f}","Precision"),(f"{ev['Recall']:.4f}","Recall"),(f"{ev['F1']:.4f}","F1 Score"),(f"{ev['Acc_train']:.4f}","Train Acc"),(f"{gap:.4f}","Overfit Gap")]: mc+=metric_card(v,l,"cls-metric-card")
            mc+='</div>';st.markdown(mc,unsafe_allow_html=True);st.markdown(overfitting_warning(gap),unsafe_allow_html=True)
            cm_data=ev["CM"];disp_labels=([str(le.inverse_transform([c])[0]) for c in range(len(classes))] if le else [str(c) for c in range(len(cm_data))])
            fig_cm,ax_cm=plt.subplots(figsize=(max(4,len(disp_labels)*1.1),max(3.5,len(disp_labels)*0.9)))
            sns.heatmap(cm_data,annot=True,fmt="d",cmap=sns.light_palette("#a060c8",as_cmap=True),xticklabels=disp_labels,yticklabels=disp_labels,linewidths=0.5,linecolor=GRID_CLR,annot_kws={"size":10,"color":"#e8d8f8"},cbar_kws={"shrink":0.8},ax=ax_cm)
            ax_cm.set_xlabel("Predicted");ax_cm.set_ylabel("Actual");ax_cm.set_title(f"{mname} — Confusion Matrix",fontsize=10,color="#7eb8d4");ax_cm.tick_params(colors=TEXT_CLR,labelsize=8)
            fig_cm.patch.set_facecolor(PLOT_BG);ax_cm.set_facecolor(AXES_BG)
            for spine in ax_cm.spines.values(): spine.set_edgecolor(GRID_CLR)
            fig_cm.tight_layout();st.pyplot(fig_cm);plt.close()
            cr=ev["CR"];cr_rows=[]
            for label_key,metrics_val in cr.items():
                if isinstance(metrics_val,dict):
                    display_key=label_key
                    try:
                        int_key=int(label_key)
                        if le: display_key=str(le.inverse_transform([int_key])[0])
                    except (ValueError,TypeError): pass
                    cr_rows.append({"Class":display_key,"Precision":round(metrics_val.get("precision",0),4),"Recall":round(metrics_val.get("recall",0),4),"F1-Score":round(metrics_val.get("f1-score",0),4),"Support":int(metrics_val.get("support",0))})
            if cr_rows: st.dataframe(pd.DataFrame(cr_rows),use_container_width=True,hide_index=True)
            m_obj=fitted[mname]
            if hasattr(m_obj,"feature_importances_"):
                fi_df=pd.DataFrame({"Feature":t_feat,"Importance":m_obj.feature_importances_}).sort_values("Importance",ascending=True)
                fig_fi,ax_fi=plt.subplots(figsize=(7,max(2.5,len(t_feat)*0.45)));ax_fi.barh(fi_df["Feature"],fi_df["Importance"],color=lc,edgecolor=AXES_BG)
                ax_fi.set_title(f"{mname} — Feature Importance");apply_theme(fig_fi,ax_fi);fig_fi.tight_layout();st.pyplot(fig_fi);plt.close()
            st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)

    # MODULE 6 — PREDICTION & ADVANCED ANALYTICS
    sub6="Numeric value + CI" if ml_mode=="regression" else "Class label + probability"
    st.markdown(module_banner("6","Prediction & Advanced Analytics",f"{sub6} · What-If · Insights"),unsafe_allow_html=True)
    pred_model_choice=st.selectbox("Predict with model",["Best Model: "+best_name]+list(fitted.keys()),key="pred_model_choice")
    chosen_model_name=best_name if pred_model_choice.startswith("Best Model") else pred_model_choice
    chosen_model=fitted[chosen_model_name];lc_chosen=MODEL_COLORS.get(chosen_model_name,ACCENT1)
    st.markdown(f'<p style="font-size:0.76rem;color:{lc_chosen};font-family:JetBrains Mono,monospace;margin-bottom:10px;">Using: {chosen_model_name}</p>',unsafe_allow_html=True)
    user_vals=[];user_vals_raw=[];n_inp=min(3,len(t_feat));inp_cols=st.columns(n_inp)
    for i,feat in enumerate(t_feat):
        with inp_cols[i%n_inp]:
            if feat in feat_enc:
                fe=feat_enc[feat];cat_opts=list(fe.classes_);sel=st.selectbox(f"{feat} 🏷️",options=cat_opts,key=f"pred_{feat}");user_vals.append(float(fe.transform([sel])[0]));user_vals_raw.append(sel)
            else:
                lo_v=float(dsnap[feat].min());hi_v=float(dsnap[feat].max());mean_v=float(dsnap[feat].mean())
                v=st.number_input(feat,min_value=lo_v,max_value=hi_v,value=mean_v,step=max((hi_v-lo_v)/200,1e-6),format="%.4f",key=f"pred_{feat}");user_vals.append(v);user_vals_raw.append(v)
    pc,_=st.columns([1,3])
    with pc: do_pred=st.button("Predict",use_container_width=True)
    def _std(arr): return (arr-xmu)/xsg if (a_std and xmu is not None) else arr
    if do_pred:
        arr=np.array(user_vals).reshape(1,-1);arr_std=_std(arr)
        if ml_mode=="regression": pred_val=chosen_model.predict(arr_std)[0];st.session_state["_last_pred"]=pred_val;st.session_state["_last_pred_cls"]=None
        else:
            pred_cls=chosen_model.predict(arr_std)[0];pred_prob=None
            if hasattr(chosen_model,"predict_proba"): pred_prob=float(chosen_model.predict_proba(arr_std)[0].max())
            st.session_state["_last_pred"]=None;st.session_state["_last_pred_cls"]=(pred_cls,pred_prob)
        st.session_state["_last_user_vals"]=user_vals;st.session_state["_last_user_vals_raw"]=user_vals_raw;st.session_state["_last_arr_std"]=arr_std;st.session_state["_last_chosen"]=chosen_model_name
    last_pred=st.session_state.get("_last_pred");last_pred_cls=st.session_state.get("_last_pred_cls");last_shown=(last_pred is not None) or (last_pred_cls is not None)
    if last_shown:
        user_vals_snap=st.session_state["_last_user_vals"];user_vals_raw_snap=st.session_state.get("_last_user_vals_raw",user_vals_snap)
        arr_std_snap=st.session_state["_last_arr_std"];snap_model_nm=st.session_state["_last_chosen"];snap_model=fitted[snap_model_nm]
        if ml_mode=="regression":
            pred=last_pred;snap_rmse=results[snap_model_nm]["RMSE"];ci_lo=pred-1.96*snap_rmse;ci_hi=pred+1.96*snap_rmse
            st.markdown(f'<div class="pred-box"><div class="pred-label">Predicted {t_tgt}</div><div class="pred-val">{pred:.4f}</div><div class="pred-ci">95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]</div><div class="pred-model">Model: {snap_model_nm}</div></div>',unsafe_allow_html=True)
        else:
            pred_cls_enc,pred_prob=last_pred_cls;pred_label=str(le.inverse_transform([pred_cls_enc])[0]) if le else str(pred_cls_enc);prob_str=f"Confidence: {pred_prob*100:.1f}%" if pred_prob is not None else ""
            st.markdown(f'<div class="pred-box-cls"><div class="pred-label">Predicted Class — {t_tgt}</div><div class="pred-val">{pred_label}</div><div class="pred-prob">{prob_str}</div><div class="pred-model">Model: {snap_model_nm}</div></div>',unsafe_allow_html=True)
            if hasattr(snap_model,"predict_proba"):
                proba_arr=snap_model.predict_proba(arr_std_snap)[0];proba_labels=([str(le.inverse_transform([c])[0]) for c in range(len(proba_arr))] if le else [str(c) for c in range(len(proba_arr))])
                fig_pb,ax_pb=plt.subplots(figsize=(max(5,len(proba_labels)*0.9),3))
                ax_pb.bar(proba_labels,proba_arr,color=[ACCENT5 if i==proba_arr.argmax() else "#1a2a3a" for i in range(len(proba_arr))],edgecolor=ACCENT5,linewidth=0.8)
                for i,val in enumerate(proba_arr): ax_pb.text(i,val+0.01,f"{val:.3f}",ha="center",va="bottom",fontsize=8,color=TEXT_CLR)
                ax_pb.set_ylabel("Probability");ax_pb.set_ylim(0,1.15);ax_pb.set_title("Class Probability Distribution")
                apply_theme(fig_pb,ax_pb);fig_pb.tight_layout();st.pyplot(fig_pb);plt.close()
        st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)
        st.markdown('<div class="module-banner" style="border-left-color:#a060c8;"><div class="mod-label" style="color:#a060c8;">⬡ Analytics · Feature 1</div><div class="mod-title">Prediction Explanation</div></div>',unsafe_allow_html=True)
        contribs={}
        if hasattr(snap_model,"get_coefficients") and snap_model.get_coefficients() is not None:
            coeffs=snap_model.get_coefficients()
            for i,feat in enumerate(t_feat): contribs[feat]=float(coeffs[i+1])*float(user_vals_snap[i])
        elif hasattr(snap_model,"W") and snap_model.W is not None:
            for i,feat in enumerate(t_feat): contribs[feat]=float(np.abs(snap_model.W[i]).max())*float(user_vals_snap[i])
        elif hasattr(snap_model,"feature_importances_"):
            fi=snap_model.feature_importances_;diff=(last_pred if ml_mode=="regression" and last_pred is not None else 1.0)
            for i,feat in enumerate(t_feat): contribs[feat]=fi[i]*diff
        elif hasattr(snap_model,"coef_"):
            c=snap_model.coef_;c=np.abs(c).mean(axis=0) if c.ndim>1 else np.abs(c)
            for i,feat in enumerate(t_feat): contribs[feat]=float(c[i])*float(user_vals_snap[i]) if i<len(c) else 0.0
        else:
            for i,feat in enumerate(t_feat): contribs[feat]=float(user_vals_snap[i])
        contrib_rows=sorted(contribs.items(),key=lambda x:abs(x[1]),reverse=True)
        tbl_html='<div class="summary-box"><div class="summary-title">⬡ Feature Contributions</div>'
        for feat,contrib in contrib_rows:
            sc="#34d490" if contrib>=0 else "#e07070";si="📈" if contrib>=0 else "📉";ss=f"+{contrib:.4f}" if contrib>=0 else f"{contrib:.4f}"
            tbl_html+=f'<div class="summary-item"><span class="summary-icon">{si}</span><span class="summary-key">{feat}</span><span class="summary-val" style="color:{sc};">{ss}</span></div>'
        tbl_html+='</div>';st.markdown(tbl_html,unsafe_allow_html=True)
        feat_names=[f for f,_ in contrib_rows];feat_values=[v for _,v in contrib_rows]
        fig_ex,ax_ex=plt.subplots(figsize=(8,max(2.8,len(feat_names)*0.5)))
        ax_ex.barh(feat_names[::-1],feat_values[::-1],color=[ACCENT2 if v>=0 else ACCENT3 for v in feat_values[::-1]],edgecolor=AXES_BG)
        ax_ex.axvline(0,color=TEXT_CLR,linewidth=0.8);ax_ex.set_title("Feature Contributions");apply_theme(fig_ex,ax_ex);fig_ex.tight_layout();st.pyplot(fig_ex);plt.close()
        st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)
        st.markdown('<div class="module-banner" style="border-left-color:#e0a844;"><div class="mod-label" style="color:#e0a844;">⬡ Analytics · Feature 2</div><div class="mod-title">What-If Analysis</div></div>',unsafe_allow_html=True)
        wi_vals=[];wi_cols=st.columns(min(3,len(t_feat)))
        for i,feat in enumerate(t_feat):
            with wi_cols[i%min(3,len(t_feat))]:
                if feat in feat_enc:
                    fe=feat_enc[feat];cat_opts=list(fe.classes_);cur_raw=user_vals_raw_snap[i] if i<len(user_vals_raw_snap) else cat_opts[0];cur_sel=cur_raw if cur_raw in cat_opts else cat_opts[0]
                    wi_sel=st.selectbox(f"{feat} 🏷️",options=cat_opts,index=cat_opts.index(cur_sel),key=f"wi_{feat}");wi_vals.append(float(fe.transform([wi_sel])[0]))
                else:
                    lo_v=float(dsnap[feat].min());hi_v=float(dsnap[feat].max())
                    wv=st.slider(feat,min_value=lo_v,max_value=hi_v,value=float(user_vals_snap[i]),step=max((hi_v-lo_v)/100,1e-6),key=f"wi_{feat}");wi_vals.append(wv)
        wi_arr=np.array(wi_vals).reshape(1,-1);wi_arr_std=_std(wi_arr)
        if ml_mode=="regression":
            wi_pred=snap_model.predict(wi_arr_std)[0];delta=wi_pred-last_pred;delta_pct=(delta/abs(last_pred)*100) if last_pred!=0 else 0.0
            dc="#34d490" if delta>=0 else "#e07070";di="▲" if delta>=0 else "▼"
            c1,c2,c3=st.columns(3)
            with c1: st.markdown(f'<div class="metric-card"><div class="val" style="color:#2d8fcb;">{last_pred:.4f}</div><div class="lbl">Original</div></div>',unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="val" style="color:#22a878;">{wi_pred:.4f}</div><div class="lbl">Modified</div></div>',unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="val" style="color:{dc};">{di} {abs(delta):.4f}</div><div class="lbl">Change ({delta_pct:+.2f}%)</div></div>',unsafe_allow_html=True)
            st.markdown("#### Scenario Comparison")
            st.markdown(f'<table class="compare-table"><thead><tr><th>Scenario</th><th>Prediction</th><th>Difference</th></tr></thead><tbody><tr class="compare-orig"><td>🔵 Original inputs</td><td>{last_pred:.4f}</td><td>—</td></tr><tr class="compare-mod"><td>🟢 Modified inputs</td><td>{wi_pred:.4f}</td><td style="color:{dc};">{di} {abs(delta):.4f} ({delta_pct:+.1f}%)</td></tr></tbody></table>',unsafe_allow_html=True)
        else:
            wi_cls=snap_model.predict(wi_arr_std)[0];wi_label=str(le.inverse_transform([wi_cls])[0]) if le else str(wi_cls)
            orig_label=str(le.inverse_transform([last_pred_cls[0]])[0]) if le else str(last_pred_cls[0]);changed=wi_label!=orig_label
            c1,c2,c3=st.columns(3)
            with c1: st.markdown(f'<div class="cls-metric-card"><div class="val">{orig_label}</div><div class="lbl">Original Class</div></div>',unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="cls-metric-card"><div class="val">{wi_label}</div><div class="lbl">Modified Class</div></div>',unsafe_allow_html=True)
            with c3:
                ch_color="#e07070" if changed else "#34d490"
                st.markdown(f'<div class="cls-metric-card"><div class="val" style="color:{ch_color};font-size:1rem;">{"CHANGED" if changed else "SAME"}</div><div class="lbl">Change Status</div></div>',unsafe_allow_html=True)
            ch_color="#e07070" if changed else "#34d490"
            st.markdown(f'<table class="compare-table"><thead><tr><th>Scenario</th><th>Predicted Class</th><th>Status</th></tr></thead><tbody><tr class="compare-orig"><td>🔵 Original</td><td>{orig_label}</td><td>—</td></tr><tr class="compare-mod"><td>🟢 Modified</td><td>{wi_label}</td><td style="color:{ch_color};">{"⚠ Changed!" if changed else "✓ Same"}</td></tr></tbody></table>',unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)

    # Download Report
    st.markdown('<div class="module-banner" style="border-left-color:#e0a844;"><div class="mod-label" style="color:#e0a844;">⬡ Export</div><div class="mod-title">Download Analysis Report</div></div>',unsafe_allow_html=True)
    top3_best=get_top3_features(fitted[best_name],t_feat)
    report_text=build_report_text(mode=ml_mode,best_name=best_name,best_res=best_res,feature_names=t_feat,top3=top3_best,health_score=st.session_state.get("_health_score",0),health_label=st.session_state.get("_health_label","Unknown"),last_pred=st.session_state.get("_last_pred"),last_pred_cls=st.session_state.get("_last_pred_cls"),target_col=t_tgt)
    if ml_mode=="regression": metrics_df=pd.DataFrame([{"Model":m,"R2_Test":round(r["R²"],4),"R2_Train":round(r["r2_train"],4),"RMSE":round(r["RMSE"],4),"MAE":round(r["MAE"],4),"Grade":get_model_grade(r["R²"])[0]} for m,r in results.items()])
    else: metrics_df=pd.DataFrame([{"Model":m,"Accuracy":round(r["Accuracy"],4),"Precision":round(r["Precision"],4),"Recall":round(r["Recall"],4),"F1":round(r["F1"],4),"Grade":get_model_grade(r["F1"])[0]} for m,r in results.items()])
    col_dl1,col_dl2=st.columns(2)
    with col_dl1: st.download_button("📄  Download Text Report (.txt)",report_text.encode("utf-8"),"ml_analysis_report.txt","text/plain",use_container_width=True)
    with col_dl2:
        csv_buffer=io.StringIO();metrics_df.to_csv(csv_buffer,index=False);st.download_button("📊  Download Metrics CSV (.csv)",csv_buffer.getvalue().encode("utf-8"),"ml_metrics.csv","text/csv",use_container_width=True)


# MODULE 7 — UNSUPERVISED LEARNING
st.markdown(module_banner("7","Unsupervised Learning & Pattern Discovery","K-Means · DBSCAN · Hierarchical Clustering · t-SNE Visualization · Apriori Rules"),unsafe_allow_html=True)
if len(numeric_cols)<2:
    st.warning("⚠ Module 7 requires at least 2 numeric columns.")
else:
    UNSUP_ALGOS=["K-Means Clustering","DBSCAN — Density-Based Clustering","Hierarchical Clustering (Manual)","t-SNE 2D Visualization","Apriori — Association Rules"]
    unsup_algo=st.selectbox("Select Unsupervised Algorithm",UNSUP_ALGOS,key="unsup_algo")
    st.markdown("#### Feature Selection for Unsupervised Learning")
    default_unsup_feats=numeric_cols[:min(5,len(numeric_cols))];unsup_feats=st.multiselect("Select numeric features (min 2)",numeric_cols,default=default_unsup_feats,key="unsup_feats")
    if len(unsup_feats)<2: st.warning("Please select at least 2 numeric features.")
    else:
        X_unsup_raw=np.hstack([data[f].fillna(data[f].median()).values.astype(float).reshape(-1,1) for f in unsup_feats])
        unsup_mu=X_unsup_raw.mean(axis=0);unsup_sg=X_unsup_raw.std(axis=0);unsup_sg[unsup_sg==0]=1.0;X_unsup=(X_unsup_raw-unsup_mu)/unsup_sg
        st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)

        if unsup_algo=="K-Means Clustering":
            st.markdown('<div class="module-banner" style="border-left-color:#22e8a0;"><div class="mod-label" style="color:#22e8a0;">⬡ Algorithm · K-Means</div><div class="mod-title">K-Means Clustering</div><div class="mod-sub">Manual implementation — groups data into K clusters by minimising inertia</div></div>',unsafe_allow_html=True)
            kp1,kp2,kp3=st.columns(3)
            with kp1: km_k=st.slider("Number of Groups (K)",2,10,3,1,key="km_k")
            with kp2: km_maxiter=st.slider("Max Iterations",50,500,300,50,key="km_maxiter")
            with kp3: km_seed=st.number_input("Random Seed",0,999,42,1,key="km_seed")
            show_elbow=st.checkbox("Show Elbow Chart",value=True,key="km_elbow");run_km=st.button("▶  Run K-Means",use_container_width=True,key="run_km")
            if run_km:
                with st.spinner("Running K-Means…"):
                    km_model=ManualKMeans(k=km_k,max_iter=km_maxiter,random_state=int(km_seed));km_model.fit(X_unsup);km_labels=km_model.labels_;km_sil=silhouette_score_manual(X_unsup,km_labels)
                    st.session_state.update({"_km_labels":km_labels,"_km_centroids":km_model.centroids_,"_km_model":km_model,"_km_sil":km_sil,"_km_k":km_k})
            if st.session_state.get("_km_labels") is not None:
                km_labels=st.session_state["_km_labels"];km_centroids=st.session_state["_km_centroids"];km_sil=st.session_state["_km_sil"];km_k_used=st.session_state["_km_k"]
                uniq_c,cnt_c=np.unique(km_labels,return_counts=True);n_points=len(km_labels)
                if not np.isnan(km_sil):
                    if km_sil>0.5: q_label,q_cls="Strong","km-quality-strong"
                    elif km_sil>0.25: q_label,q_cls="Moderate","km-quality-moderate"
                    else: q_label,q_cls="Weak","km-quality-weak"
                else: q_label,q_cls="N/A","km-quality-moderate"
                st.markdown(km_step("1","What is K-Means?"),unsafe_allow_html=True)
                st.markdown(f'<div class="km-explainer"><p>K-Means divided your <strong>{n_points:,} records</strong> into <strong>{km_k_used} groups</strong> based on: {", ".join(f"<em>{f}</em>" for f in unsup_feats)}. Quality: <span class="{q_cls} km-quality-badge">{q_label} (sil={km_sil:.2f})</span></p></div>',unsafe_allow_html=True)
                if X_unsup.shape[1]==2: plot_X=X_unsup;ax_lx=unsup_feats[0];ax_ly=unsup_feats[1];cent_plot=km_centroids
                else:
                    pca_viz=ManualPCA(2);plot_X=pca_viz.fit_transform(X_unsup);vr=pca_viz.explained_variance_ratio_
                    ax_lx=f"PC1 ({vr[0]*100:.1f}%)";ax_ly=f"PC2 ({vr[1]*100:.1f}%)";cent_plot=pca_viz.transform(km_centroids)
                fig_km,ax_km=plt.subplots(figsize=(10,6))
                for cid in range(km_k_used):
                    mask=km_labels==cid;clr=CLUSTER_PALETTE[cid%len(CLUSTER_PALETTE)]
                    ax_km.scatter(plot_X[mask,0],plot_X[mask,1],s=55,color=clr,alpha=0.75,edgecolors="none",label=f"G{cid+1} (n={mask.sum():,})")
                    ax_km.scatter(cent_plot[cid,0],cent_plot[cid,1],s=320,marker="D",color=clr,edgecolors="#fff",linewidth=1.8,zorder=6)
                    ax_km.annotate(f"G{cid+1}",(cent_plot[cid,0],cent_plot[cid,1]),textcoords="offset points",xytext=(0,10),ha="center",fontsize=9,color="#fff",fontweight="bold")
                ax_km.set_xlabel(ax_lx);ax_km.set_ylabel(ax_ly);ax_km.set_title(f"K-Means — {km_k_used} Groups",fontsize=13)
                ax_km.legend(fontsize=8.5,framealpha=0.18,labelcolor=TEXT_CLR,facecolor=AXES_BG,edgecolor=GRID_CLR);apply_theme(fig_km,ax_km);fig_km.tight_layout();st.pyplot(fig_km);plt.close()
                if show_elbow:
                    st.markdown(km_step("2","Elbow Chart — Finding Best K"),unsafe_allow_html=True)
                    elbow_ks=range(2,min(9,len(X_unsup)));inertias=[ManualKMeans(k=ek,max_iter=200,random_state=42).fit(X_unsup).inertia_ for ek in elbow_ks]
                    fig_elb,ax_elb=plt.subplots(figsize=(7,3.8));ax_elb.plot(list(elbow_ks),inertias,color=ACCENT7,linewidth=2.5,marker="o",markersize=9,markerfacecolor=ACCENT4,markeredgecolor=AXES_BG)
                    ax_elb.axvline(km_k_used,color=ACCENT3,linewidth=2,linestyle="--",label=f"K={km_k_used}");ax_elb.set_xlabel("K");ax_elb.set_ylabel("Inertia");ax_elb.set_title("Elbow Chart");ax_elb.legend(fontsize=8.5,framealpha=0.2,labelcolor=TEXT_CLR)
                    apply_theme(fig_elb,ax_elb);fig_elb.tight_layout();st.pyplot(fig_elb);plt.close()
                strip='<div class="km-summary-strip">'
                for v_s,l_s in [(str(km_k_used),"Groups"),(f"{n_points:,}","Records"),(f"{km_sil:.2f}" if not np.isnan(km_sil) else "N/A","Silhouette")]:
                    strip+=f'<div class="km-summary-tile"><div class="kmt-val">{v_s}</div><div class="kmt-lbl">{l_s}</div></div>'
                strip+='</div>';st.markdown(strip,unsafe_allow_html=True)

        elif unsup_algo=="DBSCAN — Density-Based Clustering":
            st.markdown('<div class="module-banner" style="border-left-color:#d05090;"><div class="mod-label" style="color:#d05090;">⬡ Algorithm · DBSCAN · sklearn</div><div class="mod-title">DBSCAN — Density-Based Spatial Clustering</div><div class="mod-sub">Finds clusters of arbitrary shape · Automatically labels noise points as outliers</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="problem-explain-box" style="border-left-color:#d05090;"><strong>How DBSCAN works:</strong><br>DBSCAN groups points that are <em>close together</em> (within distance <code>eps</code>) and have enough neighbours (<code>min_samples</code>). Unlike K-Means, you don\'t need to specify the number of clusters — DBSCAN discovers it automatically. Points that don\'t belong to any dense region are labelled as <em>noise</em> (shown in grey). Great for detecting outliers!</div>',unsafe_allow_html=True)
            db1,db2,db3=st.columns(3)
            with db1: db_eps=st.slider("ε (eps) — neighbourhood radius",0.1,5.0,0.5,0.05,key="db_eps")
            with db2: db_min=st.slider("min_samples — core point threshold",2,30,5,1,key="db_min")
            with db3: db_metric=st.selectbox("Distance metric",["euclidean","manhattan","cosine"],key="db_metric")
            run_db=st.button("▶  Run DBSCAN",use_container_width=True,key="run_db")
            if run_db:
                with st.spinner("Running DBSCAN…"):
                    db_model=DBSCAN(eps=db_eps,min_samples=db_min,metric=db_metric);db_labels=db_model.fit_predict(X_unsup)
                    st.session_state["_db_labels"]=db_labels;st.session_state["_db_eps"]=db_eps;st.session_state["_db_min"]=db_min
            if st.session_state.get("_db_labels") is not None:
                db_labels=st.session_state["_db_labels"];unique_clusters=sorted(set(db_labels));n_clusters=len([c for c in unique_clusters if c!=-1])
                n_noise=int((db_labels==-1).sum());n_points=len(db_labels);db_sil=silhouette_score_manual(X_unsup,db_labels) if n_clusters>=2 else float("nan")
                mc='<div class="card-grid">'
                for v,l in [(str(n_clusters),"Clusters Found"),(str(n_noise),"Noise Points"),(str(n_points),"Total Records"),(f"{db_sil:.2f}" if not np.isnan(db_sil) else "N/A","Silhouette Score")]: mc+=metric_card(v,l,"unsup-metric-card")
                mc+='</div>';st.markdown(mc,unsafe_allow_html=True)
                noise_pct=n_noise/n_points*100
                if noise_pct>30: st.markdown(render_warning("High Noise",f"{noise_pct:.1f}% of points labelled as noise. Try increasing eps or reducing min_samples.",level="moderate"),unsafe_allow_html=True)
                elif n_clusters==0: st.markdown(render_warning("No Clusters Found","All points are noise! Increase eps or reduce min_samples significantly.",level="critical"),unsafe_allow_html=True)
                if X_unsup.shape[1]==2: plot_X=X_unsup;ax_lx=unsup_feats[0];ax_ly=unsup_feats[1]
                else: pca_db=ManualPCA(2);plot_X=pca_db.fit_transform(X_unsup);vr=pca_db.explained_variance_ratio_;ax_lx=f"PC1 ({vr[0]*100:.1f}%)";ax_ly=f"PC2 ({vr[1]*100:.1f}%)"
                fig_db,ax_db=plt.subplots(figsize=(10,6));noise_mask=db_labels==-1
                if noise_mask.sum()>0: ax_db.scatter(plot_X[noise_mask,0],plot_X[noise_mask,1],s=25,color=NOISE_COLOR,alpha=0.5,marker="x",linewidths=1.2,label=f"Noise / Outliers (n={noise_mask.sum()})",zorder=2)
                for cid in [c for c in unique_clusters if c!=-1]:
                    mask=db_labels==cid;clr=CLUSTER_PALETTE[cid%len(CLUSTER_PALETTE)];ax_db.scatter(plot_X[mask,0],plot_X[mask,1],s=55,color=clr,alpha=0.8,edgecolors="none",label=f"Cluster {cid+1} (n={mask.sum():,})",zorder=3)
                ax_db.set_xlabel(ax_lx);ax_db.set_ylabel(ax_ly);ax_db.set_title(f"DBSCAN — {n_clusters} Clusters, {n_noise} Noise Points  (ε={db_eps}, min={db_min})",fontsize=12)
                ax_db.legend(fontsize=8.5,framealpha=0.2,labelcolor=TEXT_CLR,facecolor=AXES_BG,edgecolor=GRID_CLR);apply_theme(fig_db,ax_db);fig_db.tight_layout();st.pyplot(fig_db);plt.close()
                if n_clusters>0:
                    st.markdown("#### Cluster Profiles");profile_df=data[unsup_feats].copy();profile_df["__cluster__"]=db_labels
                    cluster_means=profile_df[profile_df["__cluster__"]!=-1].groupby("__cluster__")[unsup_feats].mean().round(3);cluster_means.index=[f"Cluster {i+1}" for i in cluster_means.index];st.dataframe(cluster_means.T,use_container_width=True)
                if n_noise>0:
                    with st.expander(f"🔴 View {min(n_noise,20)} Noise / Outlier Records"):
                        noise_df=data[unsup_feats].iloc[noise_mask].head(20).copy().reset_index(drop=True);st.dataframe(noise_df,use_container_width=True)
                st.markdown(f'<div class="cluster-info-card"><div class="ci-header">📊 What the results mean</div><div class="ci-body">DBSCAN found <strong>{n_clusters} dense group(s)</strong> in your data. {"No noise points were detected." if n_noise==0 else f"<strong>{n_noise} point(s) ({noise_pct:.1f}%)</strong> are labelled as <em>noise (outliers)</em>."} {"Silhouette score " + f"{db_sil:.2f}" + " indicates " + ("strong" if db_sil>0.5 else "moderate" if db_sil>0.25 else "weak") + " cluster separation." if not np.isnan(db_sil) else ""}</div></div>',unsafe_allow_html=True)

        elif unsup_algo=="Hierarchical Clustering (Manual)":
            st.markdown('<div class="module-banner" style="border-left-color:#44c8e0;"><div class="mod-label" style="color:#44c8e0;">⬡ Algorithm · Hierarchical · Manual + sklearn</div><div class="mod-title">Hierarchical (Agglomerative) Clustering</div><div class="mod-sub">Manual implementation for small datasets · sklearn dendrogram for visualisation</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="problem-explain-box" style="border-left-color:#44c8e0;"><strong>How Hierarchical Clustering works:</strong><br>Start with every point as its own group. Then repeatedly <em>merge</em> the two closest groups until only the desired number remain. The result is a tree (dendrogram) showing how groups split apart.</div>',unsafe_allow_html=True)
            hc1,hc2,hc3=st.columns(3)
            with hc1: hc_n_clusters=st.slider("Number of Clusters",2,10,3,1,key="hc_n")
            with hc2: hc_linkage=st.selectbox("Linkage type",["complete","single","average"],key="hc_link")
            with hc3: hc_max_rows=st.slider("Max rows for manual algo",50,min(500,len(data)),min(200,len(data)),50,key="hc_rows")
            run_hc=st.button("▶  Run Hierarchical Clustering",use_container_width=True,key="run_hc")
            if run_hc:
                with st.spinner("Running hierarchical clustering…"):
                    X_hc=X_unsup[:hc_max_rows];hc_manual=ManualHierarchicalClustering(n_clusters=hc_n_clusters,linkage_type=hc_linkage);hc_labels_manual=hc_manual.fit_predict(X_hc)
                    X_dend=X_unsup[:min(len(X_unsup),300)];link_matrix=linkage(X_dend,method=hc_linkage)
                    st.session_state["_hc_labels"]=hc_labels_manual;st.session_state["_hc_link_matrix"]=link_matrix;st.session_state["_hc_n"]=hc_n_clusters;st.session_state["_hc_X"]=X_hc
            if st.session_state.get("_hc_labels") is not None:
                hc_labels=st.session_state["_hc_labels"];link_matrix=st.session_state["_hc_link_matrix"];hc_k_used=st.session_state["_hc_n"];X_hc_plot=st.session_state["_hc_X"]
                uniq_hc,cnt_hc=np.unique(hc_labels,return_counts=True);n_hc=len(hc_labels);hc_sil=silhouette_score_manual(X_hc_plot,hc_labels) if hc_k_used>=2 else float("nan")
                mc='<div class="card-grid">'
                for v,l in [(str(hc_k_used),"Clusters"),(str(n_hc),"Records Used"),(f"{hc_sil:.2f}" if not np.isnan(hc_sil) else "N/A","Silhouette"),(hc_linkage.capitalize(),"Linkage")]: mc+=metric_card(v,l,"unsup-metric-card")
                mc+='</div>';st.markdown(mc,unsafe_allow_html=True)
                vis1,vis2=st.columns(2)
                with vis1:
                    if X_hc_plot.shape[1]==2: plot_X_hc=X_hc_plot;lx=unsup_feats[0];ly=unsup_feats[1]
                    else: pca_hc=ManualPCA(2);plot_X_hc=pca_hc.fit_transform(X_hc_plot);vr=pca_hc.explained_variance_ratio_;lx=f"PC1 ({vr[0]*100:.1f}%)";ly=f"PC2 ({vr[1]*100:.1f}%)"
                    fig_hcs,ax_hcs=plt.subplots(figsize=(6,5))
                    for cid,cnt in zip(uniq_hc,cnt_hc):
                        mask=hc_labels==cid;clr=CLUSTER_PALETTE[cid%len(CLUSTER_PALETTE)];ax_hcs.scatter(plot_X_hc[mask,0],plot_X_hc[mask,1],s=55,color=clr,alpha=0.8,edgecolors="none",label=f"Cluster {cid+1} (n={cnt})")
                    ax_hcs.set_xlabel(lx);ax_hcs.set_ylabel(ly);ax_hcs.set_title(f"Hierarchical — {hc_k_used} Clusters",fontsize=11)
                    ax_hcs.legend(fontsize=8,framealpha=0.2,labelcolor=TEXT_CLR,facecolor=AXES_BG,edgecolor=GRID_CLR);apply_theme(fig_hcs,ax_hcs);fig_hcs.tight_layout();st.pyplot(fig_hcs);plt.close()
                with vis2:
                    fig_dnd,ax_dnd=plt.subplots(figsize=(6,5));dendrogram(link_matrix,ax=ax_dnd,no_labels=True,color_threshold=link_matrix[-hc_k_used+1,2],above_threshold_color=TEXT_CLR,link_color_func=lambda k:CLUSTER_PALETTE[k%len(CLUSTER_PALETTE)])
                    ax_dnd.set_title("Dendrogram — Merge History",fontsize=11);ax_dnd.set_xlabel("Data Points");ax_dnd.set_ylabel("Distance")
                    ax_dnd.axhline(link_matrix[-hc_k_used+1,2],color=ACCENT3,linestyle="--",linewidth=1.5,label=f"Cut at k={hc_k_used}");ax_dnd.legend(fontsize=8,framealpha=0.2,labelcolor=TEXT_CLR)
                    apply_theme(fig_dnd,ax_dnd);fig_dnd.tight_layout();st.pyplot(fig_dnd);plt.close()
                st.markdown("#### Cluster Profiles (Feature Means)")
                prof_df=pd.DataFrame(X_hc_plot,columns=unsup_feats);prof_df["__cluster__"]=hc_labels;prof_means=prof_df.groupby("__cluster__")[unsup_feats].mean().round(3);prof_means.index=[f"Cluster {i+1}" for i in prof_means.index];st.dataframe(prof_means.T,use_container_width=True)
                st.markdown(f'<div class="cluster-info-card"><div class="ci-header">📊 What the results mean</div><div class="ci-body">Hierarchical clustering merged your data bottom-up into <strong>{hc_k_used} clusters</strong> using <strong>{hc_linkage} linkage</strong>. The dendrogram shows how clusters were formed — taller bars = bigger merge jumps (natural boundaries). {"Silhouette " + f"{hc_sil:.2f}" + " → " + ("strong" if hc_sil>0.5 else "moderate" if hc_sil>0.25 else "weak") + " separation." if not np.isnan(hc_sil) else ""}</div></div>',unsafe_allow_html=True)

        elif unsup_algo=="t-SNE 2D Visualization":
            st.markdown('<div class="module-banner" style="border-left-color:#e0a844;"><div class="mod-label" style="color:#e0a844;">⬡ Algorithm · t-SNE · sklearn</div><div class="mod-title">t-SNE — 2D Dimensionality Reduction</div><div class="mod-sub">Reveals hidden clusters and structure by projecting high-dimensional data into 2D</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="problem-explain-box" style="border-left-color:#e0a844;"><strong>How t-SNE works:</strong><br>t-SNE compresses your data from many dimensions down to just 2, preserving <em>neighbourhoods</em> — points similar in the original space appear close together in the 2D plot. ⚠ t-SNE is for <em>visualisation only</em> — distances in the 2D plot are not directly interpretable as real distances.</div>',unsafe_allow_html=True)
            ts1,ts2,ts3,ts4=st.columns(4)
            with ts1: ts_perp=st.slider("Perplexity",5,80,30,5,key="ts_perp")
            with ts2: ts_iter=st.slider("Iterations",250,1000,500,250,key="ts_iter")
            with ts3: ts_lr=st.slider("Learning rate",10,1000,200,10,key="ts_lr")
            with ts4: ts_max_rows=st.slider("Max rows",100,min(2000,len(data)),min(500,len(data)),100,key="ts_rows")
            colour_by_opts=["None (single colour)"]+all_cols;ts_colour=st.selectbox("Colour points by column (optional)",colour_by_opts,key="ts_colour")
            run_tsne=st.button("▶  Run t-SNE",use_container_width=True,key="run_tsne")
            if run_tsne:
                X_ts=X_unsup[:ts_max_rows]
                if len(X_ts)<ts_perp: st.error(f"Perplexity ({ts_perp}) must be < number of data points ({len(X_ts)}).")
                else:
                    with st.spinner("Running t-SNE (this may take a moment)…"):
                        tsne_model=TSNE(n_components=2,perplexity=min(ts_perp,len(X_ts)-1),max_iter=ts_iter,learning_rate=ts_lr,random_state=42,init="pca");X_ts_2d=tsne_model.fit_transform(X_ts)
                        st.session_state["_tsne_2d"]=X_ts_2d;st.session_state["_tsne_rows"]=ts_max_rows;st.session_state["_tsne_colour"]=ts_colour;st.session_state["_tsne_perp"]=ts_perp
            if st.session_state.get("_tsne_2d") is not None:
                X_ts_2d=st.session_state["_tsne_2d"];n_ts=len(X_ts_2d);ts_colour=st.session_state["_tsne_colour"];ts_perp_val=st.session_state["_tsne_perp"]
                fig_ts,ax_ts=plt.subplots(figsize=(10,7))
                if ts_colour!="None (single colour)" and ts_colour in data.columns:
                    colour_col=data[ts_colour].iloc[:n_ts].reset_index(drop=True)
                    if colour_col.dtype==object or colour_col.nunique()<=15:
                        unique_vals=colour_col.dropna().unique()
                        for i,val in enumerate(unique_vals):
                            mask=colour_col==val;clr=CLUSTER_PALETTE[i%len(CLUSTER_PALETTE)];ax_ts.scatter(X_ts_2d[mask,0],X_ts_2d[mask,1],s=40,color=clr,alpha=0.75,edgecolors="none",label=str(val))
                        ax_ts.legend(title=ts_colour,fontsize=8,framealpha=0.2,labelcolor=TEXT_CLR,facecolor=AXES_BG,edgecolor=GRID_CLR)
                    else:
                        norm_vals=colour_col.fillna(colour_col.median()).values;sc=ax_ts.scatter(X_ts_2d[:,0],X_ts_2d[:,1],s=40,c=norm_vals,cmap="plasma",alpha=0.75,edgecolors="none");plt.colorbar(sc,ax=ax_ts,shrink=0.8,label=ts_colour)
                else: ax_ts.scatter(X_ts_2d[:,0],X_ts_2d[:,1],s=40,color=ACCENT1,alpha=0.65,edgecolors="none")
                ax_ts.set_xlabel("t-SNE Dimension 1");ax_ts.set_ylabel("t-SNE Dimension 2");ax_ts.set_title(f"t-SNE 2D Projection — {n_ts} points  (perplexity={ts_perp_val})",fontsize=12)
                apply_theme(fig_ts,ax_ts);fig_ts.tight_layout();st.pyplot(fig_ts);plt.close()
                st.markdown('<div class="cluster-info-card"><div class="ci-header">💡 How to read this chart</div><div class="ci-body">Each dot represents one row in your dataset. <strong>Dots close together have similar features</strong>. Visible clusters (blobs) suggest natural groupings. Try running <strong>K-Means or DBSCAN</strong> on the same features to formally label these groups.</div></div>',unsafe_allow_html=True)
                if st.session_state.get("_km_labels") is not None and len(st.session_state["_km_labels"])==n_ts:
                    st.markdown("#### t-SNE coloured by K-Means cluster labels");km_lbl_ts=st.session_state["_km_labels"]
                    fig_ts2,ax_ts2=plt.subplots(figsize=(9,6))
                    for cid in np.unique(km_lbl_ts):
                        mask=km_lbl_ts==cid;clr=CLUSTER_PALETTE[cid%len(CLUSTER_PALETTE)];ax_ts2.scatter(X_ts_2d[mask,0],X_ts_2d[mask,1],s=40,color=clr,alpha=0.8,edgecolors="none",label=f"K-Means G{cid+1}")
                    ax_ts2.set_xlabel("t-SNE 1");ax_ts2.set_ylabel("t-SNE 2");ax_ts2.set_title("t-SNE + K-Means Labels Overlay",fontsize=11)
                    ax_ts2.legend(fontsize=8,framealpha=0.2,labelcolor=TEXT_CLR,facecolor=AXES_BG,edgecolor=GRID_CLR);apply_theme(fig_ts2,ax_ts2);fig_ts2.tight_layout();st.pyplot(fig_ts2);plt.close()

        elif unsup_algo=="Apriori — Association Rules":
            st.markdown('<div class="module-banner" style="border-left-color:#d05090;"><div class="mod-label" style="color:#d05090;">⬡ Algorithm · Apriori · Manual</div><div class="mod-title">Manual Apriori — Association Rule Mining</div></div>',unsafe_allow_html=True)
            st.info("💡 Apriori mines co-occurrence rules. Features are discretized into bins (Low/Mid/High).")
            ap1,ap2,ap3=st.columns(3)
            with ap1: ap_min_sup=st.slider("Min Support",0.05,0.8,0.2,0.05,key="ap_sup")
            with ap2: ap_min_conf=st.slider("Min Confidence",0.3,1.0,0.5,0.05,key="ap_conf")
            with ap3: ap_bins=st.slider("Discretisation Bins",2,5,3,1,key="ap_bins")
            ap_max_rows=st.slider("Max Rows",50,min(1000,len(data)),min(500,len(data)),50,key="ap_maxrows")
            ap_feats_sel=st.multiselect("Features for Association Mining",unsup_feats,default=unsup_feats[:min(4,len(unsup_feats))],key="ap_feats")
            bin_labels_map={2:["Low","High"],3:["Low","Mid","High"],4:["Low","Med-Low","Med-High","High"],5:["Very Low","Low","Mid","High","Very High"]}
            blabels=bin_labels_map.get(ap_bins,[f"B{i}" for i in range(ap_bins)])
            if st.button("▶  Run Apriori",use_container_width=True,key="run_ap"):
                if len(ap_feats_sel)<2: st.warning("Select at least 2 features.")
                else:
                    with st.spinner("Running Apriori…"):
                        sub_data=data[ap_feats_sel].head(ap_max_rows).copy();disc_data=pd.DataFrame()
                        for feat in ap_feats_sel:
                            col=sub_data[feat].fillna(sub_data[feat].median())
                            try: disc_data[feat]=pd.cut(col,bins=ap_bins,labels=blabels,duplicates="drop")
                            except: disc_data[feat]=col.astype(str)
                        transactions=[frozenset(f"{feat}={val}" for feat,val in row.items() if pd.notna(val)) for _,row in disc_data.iterrows()]
                        if not transactions: st.error("No valid transactions.")
                        else:
                            ap_model=ManualApriori(min_support=ap_min_sup,min_confidence=ap_min_conf);ap_model.fit(transactions)
                            mc='<div class="card-grid">'
                            for v,l in [(str(len(transactions)),"Transactions"),(str(len(ap_model.frequent_itemsets_)),"Itemsets"),(str(len(ap_model.rules_)),"Rules")]: mc+=metric_card(v,l,"unsup-metric-card")
                            mc+='</div>';st.markdown(mc,unsafe_allow_html=True)
                            if ap_model.frequent_itemsets_:
                                fi_rows=sorted([{"Itemset":" & ".join(sorted(fs)),"Support":round(s,4)} for fs,s in ap_model.frequent_itemsets_.items()],key=lambda x:-x["Support"]);st.dataframe(pd.DataFrame(fi_rows).head(20),use_container_width=True,hide_index=True)
                            if ap_model.rules_:
                                for r in ap_model.rules_[:5]:
                                    ant_s=" & ".join(sorted(r["antecedent"]));con_s=" & ".join(sorted(r["consequent"]))
                                    lc_="#22e8a0" if r["lift"]>1.5 else ("#e0a844" if r["lift"]>1.0 else "#e07070")
                                    st.markdown(f'<div class="apriori-rule-row"><div class="rule-text">{ant_s} → {con_s}</div><div class="rule-stats">Sup: {r["support"]:.4f} | Conf: {r["confidence"]:.4f} | Lift: <span class="rule-lift" style="color:{lc_};">{r["lift"]:.4f}</span></div></div>',unsafe_allow_html=True)

# FOOTER
st.markdown('<div class="section-divider"></div>',unsafe_allow_html=True)
st.markdown('<div style="text-align:center;padding:14px 0;font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#1d3a54;letter-spacing:0.1em;">INTELLIGENT DATA ANALYSIS &amp; ML SYSTEM v10 · 9 REGRESSION MODELS · 7+ CLASSIFICATION MODELS · DBSCAN · HIERARCHICAL · t-SNE · K-MEANS · APRIORI · MANUAL &amp; SKLEARN IMPLEMENTATIONS</div>',unsafe_allow_html=True)