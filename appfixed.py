"""
Thai Black Swan Risk Dashboard
Streamlit app for presenting the fragility model.
Run with: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from datetime import date
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Thai Black Swan Risk Model",
    page_icon="⚠️",
    layout="wide",
)

# ============================================================
# DATA LOADING (cached so it only runs once per session)
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    tickers = {
        'SET':    '^SET.BK',
        'THB':    'THB=X',
        'US10Y':  '^TNX',
        'OIL':    'CL=F',
        'GOLD':   'GC=F',
        'HSI':    '^HSI',
        'SSEC':   '000001.SS',
    }
    raw = {}
    for name, tk in tickers.items():
        d = yf.download(tk, start='2000-01-01', progress=False, auto_adjust=True)
        raw[name] = d['Close'].squeeze()
    data = pd.DataFrame(raw).ffill().dropna()
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data

@st.cache_data(show_spinner=False)
def build_features(data):
    df = pd.DataFrame(index=data.index)
    set_ret = data['SET'].pct_change()
    hsi_ret = data['HSI'].pct_change()

    df['vol_20d']          = set_ret.rolling(20).std() * np.sqrt(252)
    df['vol_of_vol']       = df['vol_20d'].rolling(20).std()

    rolling_max            = data['SET'].rolling(252, min_periods=20).max()
    df['drawdown']         = (data['SET'] / rolling_max) - 1

    df['corr_set_hsi']     = set_ret.rolling(60).corr(hsi_ret)
    df['autocorr_20d']     = set_ret.rolling(60).apply(lambda x: x.autocorr(lag=1), raw=False)

    def variance_slope(x):
        if np.isnan(x).any() or len(x) < 10:
            return np.nan
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    df['variance_trend']   = df['vol_20d'].rolling(60).apply(variance_slope, raw=True)

    df['thb_depreciation'] = data['THB'].pct_change(60)
    df['thb_vol']          = data['THB'].pct_change().rolling(20).std() * np.sqrt(252)
    df['oil_shock']        = data['OIL'].pct_change(60)
    df['us10y_delta']      = data['US10Y'].diff(60)
    df['regional_decouple']= (data['SET'].pct_change(60) - data['HSI'].pct_change(60)).abs()
    df['gold_rally']       = data['GOLD'].pct_change(60)

    df = df.dropna()
    df['SET_close'] = data['SET']
    return df

def compute_target(df, drawdown_threshold, forward_window):
    """
    For each day t, find the LOWEST SET value in the next `forward_window` days.
    Label = 1 if (lowest_future / today_price) - 1 < drawdown_threshold.
    The last `forward_window` days have NaN target (no future to look at).
    """
    set_close = df['SET_close'].values
    n = len(set_close)
    target = np.full(n, np.nan)

    for i in range(n - forward_window):
        future_window = set_close[i+1 : i+1+forward_window]
        future_min = future_window.min()
        future_drawdown = (future_min / set_close[i]) - 1
        target[i] = 1.0 if future_drawdown < drawdown_threshold else 0.0

    return pd.Series(target, index=df.index)

def train_model(df_in, split_date, drawdown_threshold, forward_window):
    df = df_in.copy()
    df['target'] = compute_target(df, drawdown_threshold, forward_window)

    # Separate the recent days that have no target (we'll predict on them live)
    df_predict = df[df['target'].isna()].copy()
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    feature_cols = [c for c in df.columns if c not in ['target', 'SET_close']]

    train = df[df.index < split_date]
    test  = df[df.index >= split_date]

    X_train, y_train = train[feature_cols], train['target']
    X_test,  y_test  = test[feature_cols],  test['target']

    if y_train.sum() == 0 or y_test.sum() == 0:
        st.error(
            f"Not enough positive examples to train. "
            f"Train positives: {int(y_train.sum())}, Test positives: {int(y_test.sum())}. "
            f"Try a less strict drawdown threshold or move the train/test split date."
        )
        st.stop()

    scale_pos_weight = (y_train==0).sum() / max((y_train==1).sum(), 1)

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc', random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred  = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred) if y_train.nunique() > 1 else float('nan')
    test_auc  = roc_auc_score(y_test,  test_pred)  if y_test.nunique()  > 1 else float('nan')

    df['risk_score'] = np.concatenate([train_pred, test_pred]) * 100

    # Predict for the last forward_window days (no label, but valid features)
    if len(df_predict) > 0:
        live_pred = model.predict_proba(df_predict[feature_cols])[:, 1] * 100
        df_predict['risk_score'] = live_pred
        df_full = pd.concat([df, df_predict[df.columns]])
    else:
        df_full = df

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return df_full, train_auc, test_auc, importances, feature_cols

# ============================================================
# HEADER
# ============================================================
st.title("⚠️ Thai Black Swan Risk Model")
st.markdown(
    "**An AI fragility dashboard for the Thai stock market (SET Index)** — "
    "trained on historical Thai market crises to detect periods of elevated systemic risk."
)
st.markdown("---")

# ============================================================
# SIDEBAR — MODEL CONTROLS
# ============================================================
st.sidebar.header("Model Settings")
split_date_input = st.sidebar.date_input(
    "Train/Test split date",
    value=date(2015, 1, 1),
    min_value=date(2005, 1, 1),
    max_value=date(2023, 1, 1),
)
drawdown_threshold = st.sidebar.slider(
    "Drawdown threshold (%)",
    min_value=-30, max_value=-5, value=-15, step=1,
    help="A 'pre-crash' day is one where the SET will fall at least this much within the forward window."
) / 100
forward_window = st.sidebar.slider(
    "Forward window (trading days)",
    min_value=10, max_value=60, value=30, step=5,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "Built for an academic project on black swan event prediction. "
    "The model does not predict *when* a crash will happen — it estimates "
    "*how fragile* the market currently is, based on 12 theoretically-grounded indicators."
)

# ============================================================
# RUN MODEL
# ============================================================
with st.spinner("Loading market data and training model..."):
    data = load_data()
    df_features = build_features(data)
    df, train_auc, test_auc, importances, feature_cols = train_model(
        df_features, pd.Timestamp(split_date_input), drawdown_threshold, forward_window
    )

# ============================================================
# CURRENT RISK SCORE — TOP METRICS
# ============================================================
latest = df.iloc[-1]
current_score = latest['risk_score']

if current_score < 25:
    risk_label, risk_color = "🟢 LOW", "green"
elif current_score < 50:
    risk_label, risk_color = "🟡 MODERATE", "orange"
elif current_score < 75:
    risk_label, risk_color = "🟠 ELEVATED", "orange"
else:
    risk_label, risk_color = "🔴 HIGH ALERT", "red"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Risk Score", f"{current_score:.1f} / 100", risk_label)
col2.metric("SET Index", f"{latest['SET_close']:.2f}")
col3.metric("Train AUC", f"{train_auc:.3f}")
col4.metric("Test AUC", f"{test_auc:.3f}")

st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Backtest", "🔍 Feature Importance", "📈 Current Reading", "📚 How It Works", "⚠️ Limitations"
])

# --- TAB 1: BACKTEST ---
with tab1:
    st.subheader("SET Index vs. Black Swan Risk Score")
    st.markdown(
        "The top panel shows the SET Index with known Thai market crises marked. "
        "The bottom panel shows the model's daily risk score. Look for spikes in the "
        "risk score that align with crisis events."
    )

    crises = {
        '2006-09-19': '2006 Coup',
        '2008-09-15': 'Lehman / GFC',
        '2011-10-01': '2011 Floods',
        '2013-12-01': 'Political Crisis',
        '2020-03-12': 'COVID-19',
        '2022-09-26': 'Fed Hawkish',
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})

    split_ts = pd.Timestamp(split_date_input)

    axes[0].plot(df.index, df['SET_close'], color='steelblue', linewidth=1)
    axes[0].axvline(split_ts, color='black', linestyle='--', alpha=0.5, label='Train/Test split')
    for date_str, label in crises.items():
        d = pd.Timestamp(date_str)
        if d >= df.index.min() and d <= df.index.max():
            axes[0].axvline(d, color='red', alpha=0.4, linestyle=':')
            axes[0].text(d, df['SET_close'].max()*0.98, label,
                         rotation=90, fontsize=8, va='top', ha='right', color='darkred')
    axes[0].set_title('SET Index with Historical Black Swan Events', fontsize=12)
    axes[0].set_ylabel('SET Index')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(df.index, 0, df['risk_score'], color='crimson', alpha=0.6)
    axes[1].axhline(50, color='orange', linestyle='--', alpha=0.6, label='Elevated (50)')
    axes[1].axhline(75, color='red', linestyle='--', alpha=0.6, label='High alert (75)')
    axes[1].axvline(split_ts, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('Black Swan Risk Index (model output, 0–100)', fontsize=12)
    axes[1].set_ylabel('Risk Score')
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        f"**Train AUC = {train_auc:.3f}** | **Test AUC = {test_auc:.3f}**  \n"
        "AUC of 0.5 = random guessing. Higher = better. Train AUC will be higher than Test AUC; "
        "what matters is whether Test AUC is comfortably above 0.5."
    )

# --- TAB 2: FEATURE IMPORTANCE ---
with tab2:
    st.subheader("Which Fragility Signals Matter Most?")
    st.markdown(
        "XGBoost tells us which features it relied on most when making predictions. "
        "These are the strongest fragility signals the model found in Thai market history."
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    importances.sort_values().plot(kind='barh', color='steelblue', ax=ax2)
    ax2.set_title('Feature Importance — What Predicts Thai Market Fragility', fontsize=12)
    ax2.set_xlabel('XGBoost Importance')
    ax2.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("**Top 5 fragility signals:**")
    top5 = importances.head(5)
    for feat, imp in top5.items():
        st.markdown(f"- `{feat}` — importance: **{imp:.3f}**")

# --- TAB 3: CURRENT READING ---
with tab3:
    st.subheader("Today's Market Reading")

    st.markdown(f"### Risk Score: :{risk_color}[{current_score:.1f} / 100] — {risk_label}")
    st.markdown(f"**Date:** {latest.name.date()}  |  **SET Index:** {latest['SET_close']:.2f}")

    st.markdown("---")
    st.markdown("**Top 5 contributing fragility features (current values):**")

    feature_explanations = {
        'vol_20d':           'Recent realized volatility (annualized)',
        'vol_of_vol':        'Volatility of volatility — instability signal',
        'drawdown':          'Distance below 252-day high',
        'corr_set_hsi':      'Correlation between SET and Hang Seng',
        'autocorr_20d':      'Return autocorrelation (Scheffer critical slowing down)',
        'variance_trend':    'Slope of variance — Scheffer early warning signal',
        'thb_depreciation':  '60-day USD/THB change (positive = baht weakening)',
        'thb_vol':           'Thai baht volatility',
        'oil_shock':         '60-day oil price change',
        'us10y_delta':       '60-day change in US 10Y Treasury yield',
        'regional_decouple': 'How much SET is moving differently from Hang Seng',
        'gold_rally':        '60-day gold price change (flight-to-safety signal)',
    }

    rows = []
    for feat in importances.head(5).index:
        rows.append({
            'Feature': feat,
            'Current value': f"{latest[feat]:+.4f}",
            'Meaning': feature_explanations.get(feat, ''),
        })
    st.table(pd.DataFrame(rows))

# --- TAB 4: HOW IT WORKS ---
with tab4:
    st.subheader("Methodology")
    st.markdown("""
**The paradox of black swan prediction**

By Nassim Taleb's definition, a black swan is an event that lies outside regular expectations,
carries massive impact, and is only explainable in hindsight. If you could reliably predict it,
it wouldn't be a black swan.

**So instead of predicting events, we model fragility.**

We engineer 12 indicators that capture different *types* of market fragility, train a machine
learning model on historical Thai market crises, and output a daily Black Swan Risk Score.

**The 12 fragility features:**

| Feature | Theory |
|---|---|
| Realized volatility | Standard risk measure |
| Vol-of-vol | Instability signal — regime change |
| Drawdown from 252d high | Momentum failure |
| Cross-asset correlation | "Everything falls together" — diversification breakdown |
| Return autocorrelation | Scheffer's critical slowing down (complex systems theory) |
| Variance trend | Critical slowing down — variance rises before tipping points |
| THB depreciation | Baht weakness preceded the 1997 Asian Crisis |
| THB volatility | Foreign capital flight signal |
| Oil shock | Thailand is a net oil importer |
| US10Y rate of change | Sudden Fed hawkishness triggers EM outflows |
| Regional decoupling | Idiosyncratic Thai stress vs contagion |
| Gold rally | Global risk-off / flight-to-safety |

**The model:** XGBoost classifier, trained chronologically (no shuffling) to avoid lookahead bias.
The training set deliberately excludes recent crises so they can serve as out-of-sample tests.

**The label:** A day is "pre-crash" if the SET will fall more than the threshold (default 15%)
within the next forward window (default 30 trading days).
    """)

# --- TAB 5: LIMITATIONS ---
with tab5:
    st.subheader("Honest Limitations")
    st.markdown("""
1. **True black swans are unpredictable.** This model identifies fragility, not events.
   A market can be fragile for months without crashing, and crashes can come from low-fragility
   states (e.g. sudden geopolitical shocks).

2. **Survivorship bias in feature design.** Our features were chosen knowing which past crises
   occurred. A real out-of-sample test would require features chosen *before* the events.

3. **Class imbalance.** Black swan periods are rare, so high accuracy is trivial — recall and AUC
   are the meaningful metrics.

4. **Regime changes.** Thailand's market structure has evolved since 2000 (foreign ownership rules,
   derivatives market, capital flow regulations). Old patterns may not repeat.

5. **The Taleb critique still applies.** Any model trained on past crises is, by construction,
   fitting yesterday's black swans. The next one will likely come from a direction we didn't
   engineer features for.

**Bottom line:** Use this as a fragility dashboard, not a crystal ball. The Talebian prescription —
robustness, optionality, barbell strategy — is still the only reliable defense against true black swans.
    """)

st.markdown("---")
st.caption("Built for academic purposes. Not investment advice.")
