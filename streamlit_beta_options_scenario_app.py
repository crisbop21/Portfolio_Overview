# streamlit_app.py
# ------------------
# Webapp to parse IBKR portfolio screenshots, estimate beta exposure, compute option deltas,
# and run portfolio P&L scenario analysis vs SPY.
#
# Features
# 1) Upload multiple screenshots or a CSV/XLSX fallback
# 2) OCR the screenshots (EasyOCR preferred, pytesseract fallback) to pre-fill a positions table
# 3) Robust option-name parser [e.g., "AAPL 17JAN2026 150 C" or "AAPL JAN 17 2026 150 C"]
# 4) Estimate betas with yfinance or allow manual override
# 5) Compute option deltas with Black–Scholes, contract multiplier default 100
# 6) Scenario analysis vs SPY with adjustable range and step, including per-position breakdown
# 7) Edit table in-app, then download results as CSV
#
# Notes
# - OCR is best-effort. Please review and correct the parsed table using the editor before running analysis.
# - Black–Scholes for American options is an approximation. Use with care.
# - yfinance calls can be slow. Use caching and consider manual betas if needed.

import io
import re
import math
import base64
from datetime import datetime, date
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional dependencies
try:
    import easyocr  # Pure-Python OCR, easier deploy than system tesseract
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

try:
    import pytesseract  # Requires system tesseract installed
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

import matplotlib.pyplot as plt

st.set_page_config(page_title="IBKR Beta & Options Scenario", layout="wide")
st.title("IBKR Portfolio Beta and Options Scenario Analyzer")

with st.sidebar:
    st.header("Settings")
    ref_index = st.text_input("Reference index ticker", value="SPY", help="Used for betas and scenario driver")
    lookback_years = st.slider("Beta lookback [years]", 1, 5, 2, help="History window to estimate betas")
    interval = st.selectbox("Beta return interval", ["1d", "1wk"], index=0)
    r_rate = st.number_input("Risk-free rate r", min_value=0.0, max_value=0.2, value=0.04, step=0.005)
    default_iv = st.number_input("Default implied vol (IV)", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
    default_div_yield = st.number_input("Default dividend yield q", min_value=0.0, max_value=0.1, value=0.0, step=0.005)
    contract_multiplier = st.number_input("Contract multiplier", min_value=1, max_value=1000, value=100, step=1)
    st.caption("If your options use a different multiplier, adjust here.")

    st.subheader("Scenario")
    scen_min = st.number_input("Min SPY move [%]", -100.0, 100.0, -10.0, step=1.0)
    scen_max = st.number_input("Max SPY move [%]", -100.0, 100.0, 10.0, step=1.0)
    scen_step = st.number_input("Step [%]", 0.1, 50.0, 1.0, step=0.1)
    include_gamma = st.checkbox("Include simple gamma approx", value=False, help="Requires IV and time, rough approximation")

st.markdown("""
**Workflow**
1. Upload IBKR screenshots, or a CSV/XLSX, or paste raw text. OCR will pre-fill a positions table.
2. Review and correct the table below, especially `asset_type` [equity, option], `quantity`, and `symbol`.
3. Click **Run analysis** to estimate betas, deltas, and scenario P&L vs SPY.
""")

# ------------------
# Utilities
# ------------------

_MONTHS = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

@st.cache_data(show_spinner=False)
def _download_prices(tickers: List[str], period: str, interval: str) -> Optional[pd.DataFrame]:
    if not _HAS_YF:
        return None
    try:
        df = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)
        # yfinance returns multi-index cols when multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close'] if 'Close' in df.columns.levels[0] else df.iloc[:, 0]
        else:
            # Single ticker, may have columns like ['Close','Open',...]
            if 'Close' in df.columns:
                df = df['Close']
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def estimate_beta(ticker: str, ref: str, lookback_years: int, interval: str) -> Optional[float]:
    if not _HAS_YF:
        return None
    period = f"{lookback_years}y"
    df = _download_prices([ticker, ref], period=period, interval=interval)
    if df is None:
        return None
    try:
        # df may be a DataFrame with columns per ticker
        if isinstance(df, pd.Series):
            return 1.0
        if ticker not in df.columns or ref not in df.columns:
            return None
        r_t = df[ticker].pct_change().dropna()
        r_m = df[ref].pct_change().dropna()
        x, y = r_m.align(r_t, join='inner')
        if len(x) < 10:
            return None
        cov = np.cov(x, y)[0, 1]
        var = np.var(x)
        if var == 0:
            return None
        beta = cov / var
        return float(beta)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def fetch_last_price(ticker: str) -> Optional[float]:
    if not _HAS_YF:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.history(period='5d')
        if info is None or info.empty:
            return None
        return float(info['Close'].iloc[-1])
    except Exception:
        return None

# Black–Scholes delta (call or put), continuous dividend yield q

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, call: bool = True, q: float = 0.0) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if call:
            return math.exp(-q * T) * _norm_cdf(d1)
        else:
            return -math.exp(-q * T) * _norm_cdf(-d1)
    except Exception:
        return 0.0

# Very rough gamma for Black–Scholes, used only if user opts in

def bs_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        return math.exp(-q * T) * pdf / (S * sigma * math.sqrt(T))
    except Exception:
        return 0.0

# Helper to clean price strings like C19.50 or $23.88 coming from IBKR exports

def clean_price(x):
    try:
        if isinstance(x, str):
            s = x.strip().replace(',', '')
            # extract first numeric substring
            num = []
            started = False
            for ch in s:
                if ch.isdigit() or ch in ['.', '-', '+']:
                    num.append(ch)
                    started = True
                elif started:
                    break
            if num:
                return float(''.join(num))
            return None
        return float(x)
    except Exception:
        return None

# ------------------
# OCR helpers
# ------------------

def ocr_images(files: List[io.BytesIO]) -> str:
    texts = []
    if _HAS_EASYOCR:
        reader = easyocr.Reader(['en'], gpu=False)
        for f in files:
            image = Image.open(f).convert('RGB')
            result = reader.readtext(np.array(image), detail=0, paragraph=True)
            texts.append("\n".join(result))
    elif _HAS_TESSERACT:
        for f in files:
            image = Image.open(f).convert('L')
            txt = pytesseract.image_to_string(image)
            texts.append(txt)
    else:
        st.warning("No OCR backend available. Install easyocr or pytesseract, or use CSV upload.")
    return "\n".join(texts)

# ------------------
# Parsing IBKR lines
# ------------------

def parse_option_name(name: str) -> Tuple[Optional[str], Optional[date], Optional[float], Optional[str]]:
    """Parse common IBKR-like option strings into [symbol, expiry_date, strike, cp].
    Supported examples:
    - "AAPL 17JAN2026 150 C"
    - "AAPL JAN 17 2026 150 C"
    - "MSFT 20 DEC 24 300 P"
    - "TSLA 2026-01-17 250 C"
    Returns (symbol, expiry_date, strike, 'C' or 'P').
    """
    s = re.sub(r"\s+", " ", name.strip().upper())

    # Pattern 1: TICKER DDMMMYYYY STRIKE C/P
    m = re.search(r"^([A-Z\.-]{1,10})\s+(\d{1,2})([A-Z]{3})(\d{2,4})\s+(\d+(?:\.\d+)?)\s+([CP])$", s)
    if m:
        sym = m.group(1)
        dd = int(m.group(2))
        mm = _MONTHS.get(m.group(3), None)
        yy = int(m.group(4))
        yy = yy + 2000 if yy < 100 else yy
        K = float(m.group(5))
        cp = m.group(6)
        if mm:
            return sym, date(yy, mm, dd), K, cp

    # Pattern 2: TICKER MON DD YYYY STRIKE C/P
    m = re.search(r"^([A-Z\.-]{1,10})\s+([A-Z]{3})\s+(\d{1,2})\s+(\d{4})\s+(\d+(?:\.\d+)?)\s+([CP])$", s)
    if m:
        sym = m.group(1)
        mm = _MONTHS.get(m.group(2), None)
        dd = int(m.group(3))
        yy = int(m.group(4))
        K = float(m.group(5))
        cp = m.group(6)
        if mm:
            return sym, date(yy, mm, dd), K, cp

    # Pattern 3: TICKER YYYY-MM-DD STRIKE C/P
    m = re.search(r"^([A-Z\.-]{1,10})\s+(\d{4}-\d{2}-\d{2})\s+(\d+(?:\.\d+)?)\s+([CP])$", s)
    if m:
        sym = m.group(1)
        dt = datetime.strptime(m.group(2), "%Y-%m-%d").date()
        K = float(m.group(3))
        cp = m.group(4)
        return sym, dt, K, cp

    # IBKR sometimes has extra tokens; attempt a looser search
    m = re.search(r"([A-Z\.-]{1,10}).*?(\d{1,2})\s*([A-Z]{3})\s*(\d{2,4}).*?(\d+(?:\.\d+)?).*?([CP])", s)
    if m:
        sym = m.group(1)
        dd = int(m.group(2))
        mm = _MONTHS.get(m.group(3), None)
        yy = int(m.group(4))
        yy = yy + 2000 if yy < 100 else yy
        K = float(m.group(5))
        cp = m.group(6)
        if mm:
            return sym, date(yy, mm, dd), K, cp

    return None, None, None, None


def parse_option_name_v2(name: str) -> Tuple[Optional[str], Optional[date], Optional[float], Optional[str]]:
    """Lightweight parser for IBKR mobile style like "SEP 18 '26 400 Call".
    Returns (symbol, expiry_date, strike, 'C'/'P') or Nones.
    """
    try:
        s = " ".join(name.strip().upper().split())
        # Tokenize, remove obvious separators
        for ch in ["/", "-", ","]:
            s = s.replace(ch, " ")
        tokens = s.split()
        if not tokens:
            return None, None, None, None
        # CP at the end
        cp = None
        if tokens[-1] in ("CALL", "PUT", "C", "P"):
            cp = 'C' if tokens[-1] in ("CALL", "C") else 'P'
            tokens = tokens[:-1]
        # Find month token
        mon_idx = None
        for i, tok in enumerate(tokens):
            if tok in _MONTHS:
                mon_idx = i
                break
        if mon_idx is None or mon_idx + 2 >= len(tokens):
            return None, None, None, None
        sym = tokens[0]
        # Day
        try:
            dd = int(tokens[mon_idx + 1])
        except Exception:
            return None, None, None, None
        # Year, allow '26 or 2026
        yy_tok = tokens[mon_idx + 2].lstrip("'")
        if not yy_tok.isdigit():
            return None, None, None, None
        yy = int(yy_tok)
        yy = yy + 2000 if yy < 100 else yy
        # Strike, first numeric token after year
        strike = None
        for j in range(mon_idx + 3, len(tokens)):
            t = tokens[j].replace("$", "")
            try:
                strike = float(t)
                break
            except Exception:
                continue
        if strike is None or cp is None:
            return None, None, None, None
        mm = _MONTHS.get(tokens[mon_idx])
        if not mm:
            return None, None, None, None
        return sym, date(yy, mm, dd), float(strike), cp
    except Exception:
        return None, None, None, None


def seed_positions_from_text(text: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        # Try option parse first
        sym, dt, K, cp = parse_option_name(ln)
        if not (sym and dt and K and cp):
            sym, dt, K, cp = parse_option_name_v2(ln)
        if sym and dt and K and cp:
            rows.append({
                'instrument': ln,
                'asset_type': 'option',
                'symbol': sym,
                'expiry': dt.isoformat(),
                'strike': float(K),
                'cp': cp,
                'quantity': None,
                'price': None,
                'beta_override': None,
                'iv': None,
                'div_yield': None
            })
            continue
        # Try equity-like: capture uppercase ticker tokens and a quantity if present
        m = re.search(r"\b([A-Z][A-Z0-9\.-]{0,9})\b", ln)
        if m:
            sym = m.group(1)
            mqty = re.search(r"([-+]?\d{1,3}(?:,?\d{3})*|[-+]?\d+)(?:\s*SH|\b)", ln)
            qty = None
            if mqty:
                try:
                    qty = int(mqty.group(1).replace(',', ''))
                except Exception:
                    qty = None
            rows.append({
                'instrument': ln,
                'asset_type': 'equity',
                'symbol': sym,
                'expiry': None,
                'strike': None,
                'cp': None,
                'quantity': qty,
                'price': None,
                'beta_override': None,
                'iv': None,
                'div_yield': None
            })
    if not rows:
        # Provide a blank template
        rows = [{
            'instrument': '', 'asset_type': 'equity', 'symbol': '', 'expiry': None, 'strike': None, 'cp': None,
            'quantity': None, 'price': None, 'beta_override': None, 'iv': None, 'div_yield': None
        }]
    df = pd.DataFrame(rows)
    return df

# ------------------
# Inputs
# ------------------

col_u1, col_u2 = st.columns([2, 1])
with col_u1:
    up_imgs = st.file_uploader("Upload IBKR screenshots (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
with col_u2:
    up_csv = st.file_uploader("Or upload CSV/XLSX positions", type=["csv", "xlsx"], accept_multiple_files=False)

raw_text_input = st.text_area("Or paste raw IBKR table text here", height=120, placeholder="Paste table-like text or option lines, one per row")

positions_df = None

if up_csv is not None:
    try:
        if up_csv.name.lower().endswith('.csv'):
            positions_df = pd.read_csv(up_csv)
        else:
            positions_df = pd.read_excel(up_csv)
        # Normalize headers and map common IBKR columns
        positions_df.rename(columns={c: c.strip().lower() for c in positions_df.columns}, inplace=True)

        # Map IBKR 'last' to 'price' and clean values like C19.50
        if 'last' in positions_df.columns and 'price' not in positions_df.columns:
            positions_df['price'] = positions_df['last']
        if 'price' in positions_df.columns:
            positions_df['price'] = positions_df['price'].apply(clean_price)

        # Map 'position' to 'quantity' when present
        if 'position' in positions_df.columns:
            if 'quantity' not in positions_df.columns:
                positions_df['quantity'] = positions_df['position']
            else:
                positions_df['quantity'] = positions_df['quantity'].where(positions_df['quantity'].notna(), positions_df['position'])
            positions_df['quantity'] = pd.to_numeric(positions_df['quantity'], errors='coerce')

        # Normalize cp values like 'Call'/'Put'
        if 'cp' in positions_df.columns:
            positions_df['cp'] = positions_df['cp'].astype(str).str.upper().str.strip().replace({'CALL':'C','PUT':'P'})
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

elif up_imgs:
    with st.spinner("Running OCR on screenshots..."):
        txt = ocr_images(up_imgs)
    positions_df = seed_positions_from_text(txt)

elif raw_text_input.strip():
    positions_df = seed_positions_from_text(raw_text_input)

else:
    positions_df = seed_positions_from_text("")

st.markdown("### Review and edit positions")
st.caption("Required columns per asset type. Equity: symbol, quantity, price [optional], beta_override [optional]. Option: symbol, quantity, expiry, strike, cp [C or P], iv [optional].")

# Ensure canonical columns
REQ_COLS = ['instrument','asset_type','symbol','quantity','price','beta_override','expiry','strike','cp','iv','div_yield']
for c in REQ_COLS:
    if c not in positions_df.columns:
        positions_df[c] = None

# Coerce dtypes for editor friendliness
positions_df['asset_type'] = positions_df['asset_type'].fillna('equity')
positions_df['cp'] = positions_df['cp'].apply(lambda x: x if x in ['C','P'] else None)

edited_df = st.data_editor(
    positions_df[REQ_COLS],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        'asset_type': st.column_config.SelectboxColumn("asset_type", options=['equity','option']),
        'cp': st.column_config.SelectboxColumn("cp", options=['C','P', None]),
    },
    key="editor_positions"
)

st.divider()

run = st.button("Run analysis", type="primary")

# ------------------
# Analysis
# ------------------

def safe_float(x, default=None):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def parse_date_safe(x) -> Optional[date]:
    if x in [None, '', np.nan]:
        return None
    if isinstance(x, date):
        return x
    if isinstance(x, datetime):
        return x.date()
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%b %d %Y", "%d %b %Y"]:
        try:
            return datetime.strptime(str(x), fmt).date()
        except Exception:
            continue
    return None

if run:
    df = edited_df.copy()

    # Drop empty rows
    df = df[~(df['symbol'].astype(str).str.strip() == '')].copy()
    if df.empty:
        st.warning("No positions to analyze.")
        st.stop()

    # Prepare containers for results
    # Fetch last prices if missing
    if _HAS_YF:
        st.info("Fetching last prices and estimating betas. This can take a moment.")
    else:
        st.warning("yfinance not installed. Please fill 'price' and 'beta_override' manually.")

    # Normalize fields
    df['quantity'] = df['quantity'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
    df['price'] = df['price'].apply(lambda x: safe_float(x))
    df['beta_override'] = df['beta_override'].apply(lambda x: safe_float(x))
    df['iv'] = df['iv'].apply(lambda x: safe_float(x))
    df['div_yield'] = df['div_yield'].apply(lambda x: safe_float(x))
    df['expiry_dt'] = df['expiry'].apply(parse_date_safe)

    # Determine unique tickers to price
    tickers = sorted(df['symbol'].dropna().astype(str).unique().tolist())

    last_prices = {}
    betas = {}

    for t in tickers:
        # Last price
        p = fetch_last_price(t) if _HAS_YF else None
        last_prices[t] = p
        # Beta
        b = estimate_beta(t, ref_index, lookback_years, interval) if _HAS_YF else None
        betas[t] = b

    # Fill missing prices and betas from overrides/defaults
    filled_prices = []
    filled_betas = []

    for i, row in df.iterrows():
        sym = str(row['symbol'])
        price = row['price'] if pd.notna(row['price']) else last_prices.get(sym)
        beta = row['beta_override'] if pd.notna(row['beta_override']) else betas.get(sym, None)
        if beta is None:
            # If still None, default to 1 for equities, 1 for option underlying beta mapping
            beta = 1.0
        filled_prices.append(price)
        filled_betas.append(beta)

    df['S0'] = filled_prices
    df['beta'] = filled_betas

    if df['S0'].isna().any():
        st.warning("Some prices could not be fetched. Please fill missing 'price' values in the table and rerun.")

    # Compute deltas and gammas where applicable
    today = date.today()

    deltas = []
    gammas = []

    for i, row in df.iterrows():
        if row['asset_type'] == 'option':
            S = safe_float(row['S0'], 0.0) or 0.0
            K = safe_float(row['strike'], 0.0) or 0.0
            cp = str(row['cp']).upper() if pd.notna(row['cp']) else None
            iv = row['iv'] if pd.notna(row['iv']) else default_iv
            q = row['div_yield'] if pd.notna(row['div_yield']) else default_div_yield
            T = 0.0
            if row['expiry_dt']:
                T = max((row['expiry_dt'] - today).days / 365.25, 1/365)  # at least 1 day
            d = bs_delta(S, K, T, r_rate, iv, call=(cp == 'C'), q=q) if cp in ['C','P'] else 0.0
            g = bs_gamma(S, K, T, r_rate, iv, q=q) if include_gamma and cp in ['C','P'] else 0.0
            deltas.append(d)
            gammas.append(g)
        else:
            deltas.append(np.nan)
            gammas.append(np.nan)

    df['delta'] = deltas
    df['gamma'] = gammas

    # Scenario grid
    moves = np.arange(scen_min, scen_max + 1e-9, scen_step)
    moves = np.round(moves, 4)

    scen_rows = []
    pos_rows = []

    for mv in moves:
        m = mv / 100.0
        total_pnl = 0.0
        for i, row in df.iterrows():
            sym = str(row['symbol'])
            qty = int(row['quantity'])
            beta = float(row['beta']) if pd.notna(row['beta']) else 1.0
            S0 = safe_float(row['S0'], 0.0) or 0.0

            if row['asset_type'] == 'equity':
                # Linear beta mapping to SPY move
                dS = S0 * beta * m
                pnl = dS * qty
                pos_rows.append({
                    'scenario_%': mv,
                    'symbol': sym,
                    'asset_type': 'equity',
                    'pnl': pnl
                })
                total_pnl += pnl
            else:
                # Option: map SPY move to underlying via beta, then use delta, optional gamma
                dS = S0 * beta * m
                pnl_delta = row['delta'] * dS * contract_multiplier * qty
                pnl_gamma = 0.0
                if include_gamma and pd.notna(row['gamma']):
                    pnl_gamma = 0.5 * row['gamma'] * (dS ** 2) * contract_multiplier * qty
                pnl = pnl_delta + pnl_gamma
                pos_rows.append({
                    'scenario_%': mv,
                    'symbol': sym,
                    'asset_type': 'option',
                    'pnl': pnl
                })
                total_pnl += pnl
        scen_rows.append({'scenario_%': mv, 'portfolio_pnl': total_pnl})

    scen_df = pd.DataFrame(scen_rows)
    breakdown_df = pd.DataFrame(pos_rows)

    # Display
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Portfolio P&L by scenario")
        st.dataframe(scen_df, use_container_width=True)
        # Chart
        fig, ax = plt.subplots()
        ax.plot(scen_df['scenario_%'], scen_df['portfolio_pnl'])
        ax.set_xlabel("SPY move [%]")
        ax.set_ylabel("Portfolio P&L")
        ax.set_title("Scenario P&L vs SPY move")
        st.pyplot(fig)

    with c2:
        st.subheader("Per-position breakdown")
        st.dataframe(breakdown_df.pivot_table(index=['symbol','asset_type'], columns='scenario_%', values='pnl', aggfunc='sum').fillna(0.0), use_container_width=True)

    # Downloads
    st.download_button("Download scenario table (CSV)", data=scen_df.to_csv(index=False), file_name="scenario_portfolio_pnl.csv", mime="text/csv")
    st.download_button("Download per-position breakdown (CSV)", data=breakdown_df.to_csv(index=False), file_name="scenario_breakdown.csv", mime="text/csv")

    st.divider()
    st.markdown("""
    **Assumptions and caveats**
    - Equity P&L is linear in SPY move multiplied by each asset beta. No idiosyncratic alpha or basis is modeled.
    - Option P&L uses delta, optionally a simple gamma term. No vega, theta, or cross-greeks. IV is assumed constant.
    - Betas are estimated from historical returns if yfinance is available, otherwise use `beta_override`.
    - Prices default to latest close from yfinance, override if needed.
    - Contract multiplier applies to all options globally.
    """)

else:
    st.info("Edit your positions above, then click Run analysis.")

st.divider()
st.markdown("""
### Input format reference
You can upload CSV/XLSX with columns: `asset_type, symbol, quantity, price, beta_override, expiry, strike, cp, iv, div_yield`.
The app also maps common IBKR headers. In particular, a column named `Last` or `last` will be used as `price` automatically, values like `C19.50` are cleaned to numeric. If a `Position` column is present, it will be used as `quantity`.
For options, the `instrument` name can also be parsed if it contains something like `AAPL 17JAN2026 150 C` or IBKR mobile style like `SEP 18 '26 400 Call`.


st.caption("© 2025 Prototype. Educational use only, not investment advice.")
