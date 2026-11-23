
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import html
import re

from model_inference import predict_next_close

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Reliance | Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# frontend/app.py is one level below the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

STOCK_FILE = DATA_DIR / "RELIANCE.csv"
FUSION_FILE = DATA_DIR / "fusion_dataset_with_historical_news.csv"
ORIGINAL_FUSION_FILE = DATA_DIR / "fusion_dataset.csv"

PREDICTIONS_FILE = DATA_DIR / "training" / "model_comparison_predictions.csv"
RESULTS_FILE = DATA_DIR / "training" / "model_comparison_results.csv"
TRAINING_FILE = DATA_DIR / "training" / "training_data.npz"

MODEL_FILE = BASE_DIR / "models" / "best_lstm_model.pth"

SCALER_FILES = [
    BASE_DIR / "models" / "feature_scaler.pkl",
    BASE_DIR / "models" / "target_scaler.pkl"
]

NEWS_FILES = {
    "Raw Historical News": DATA_DIR / "historical_news" / "2024" / "reliance_news_2024_11.csv",
    "Cleaned Historical News": DATA_DIR / "processed_news" / "reliance_news_2024_11_cleaned.csv",
    "FinBERT Sentiment News": DATA_DIR / "sentiment_news" / "reliance_news_2024_11_sentiment.csv",
    "Daily Sentiment": DATA_DIR / "daily_sentiment" / "reliance_daily_sentiment_2024_11.csv",
    "Aligned News": DATA_DIR / "aligned_news" / "reliance_all_news_aligned.csv",
    "Combined Daily Sentiment": DATA_DIR / "aligned_news" / "reliance_combined_daily_sentiment.csv",
    "Additional Filtered News": DATA_DIR / "reliance_sentiment_filtered.csv",
}

# ============================================================
# 2. DARK / LIGHT MODE
# ============================================================

st.sidebar.markdown("## RELIANCE INTELLIGENCE")
st.sidebar.caption("Premium Stock Research Platform")

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

if dark_mode:
    BG = "#0B1220"
    SIDEBAR = "#101B2D"
    CARD = "#17263B"
    CARD_ALT = "#1E3048"
    BORDER = "#354963"
    TEXT = "#F8FAFC"
    MUTED = "#C1CDDD"
    SUBTLE = "#AAB9CC"
    GRID = "#3A4D65"
    BLUE = "#60A5FA"
    CYAN = "#67E8F9"
    GREEN = "#4ADE80"
    RED = "#FB7185"
    ORANGE = "#FBBF24"
    PURPLE = "#C4B5FD"
    ACTUAL_COLOR = "#F8FAFC"

    BLUE_FILL = "rgba(96, 165, 250, 0.25)"
    GREEN_FILL = "rgba(74, 222, 128, 0.25)"
    RED_FILL = "rgba(251, 113, 133, 0.25)"
    ORANGE_FILL = "rgba(251, 191, 36, 0.25)"
    PURPLE_FILL = "rgba(196, 181, 253, 0.25)"

else:
    BG = "#F1F5F9"
    SIDEBAR = "#FFFFFF"
    CARD = "#FFFFFF"
    CARD_ALT = "#F8FAFC"
    BORDER = "#D7E0EB"
    TEXT = "#142033"
    MUTED = "#42536B"
    SUBTLE = "#5E6F85"
    GRID = "#DCE4EE"
    BLUE = "#1D4ED8"
    CYAN = "#0E7490"
    GREEN = "#15803D"
    RED = "#DC2626"
    ORANGE = "#B45309"
    PURPLE = "#6D28D9"
    ACTUAL_COLOR = "#142033"

    BLUE_FILL = "rgba(29, 78, 216, 0.18)"
    GREEN_FILL = "rgba(21, 128, 61, 0.18)"
    RED_FILL = "rgba(220, 38, 38, 0.18)"
    ORANGE_FILL = "rgba(180, 83, 9, 0.18)"
    PURPLE_FILL = "rgba(109, 40, 217, 0.18)"

# ============================================================
# 3. PROFESSIONAL CSS
# ============================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR};
        border-right: 1px solid {BORDER};
    }}

    /* Hide only the Print menu item, not the entire menu. */
    [data-testid="stMainMenu"] {{
        visibility: visible !important;
    }}

    [data-testid="stMainMenu"] button {{
        color: {TEXT} !important;
    }}

    [data-testid="stAppDeployButton"] {{
        display: none !important;
    }}

    [data-testid="stHeader"] {{
        background-color: {BG};
    }}

    /* Hide Print controls where Streamlit exposes them. */
    button[aria-label="Print"],
    button[title="Print"],
    [data-testid="stToolbar"] button[aria-label="Print"] {{
        display: none !important;
    }}

    h1, h2, h3, h4, h5, h6,
    p, label, li,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
    }}

    [data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
    }}

    [data-testid="stMetricDelta"] {{
        color: {GREEN} !important;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    [data-testid="stTable"] {{
        color: {TEXT};
    }}

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        background-color: {CARD} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}

    [data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [role="listbox"] *,
    [role="option"] {{
        color: {TEXT} !important;
    }}

    [data-testid="stTextInput"] input {{
        color: {TEXT} !important;
        background-color: {CARD} !important;
    }}

    .stButton button,
    .stDownloadButton button,
    [data-testid="stDownloadButton"] button {{
        background-color: {CARD_ALT} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}

    .stButton button:hover,
    .stDownloadButton button:hover {{
        border-color: {BLUE} !important;
        color: {BLUE} !important;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {CARD} !important;
    }}

    [role="dialog"],
    [data-testid="stDialog"],
    [data-baseweb="modal"] > div {{
        background-color: {CARD} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
    }}

    [role="dialog"] *,
    [data-testid="stDialog"] * {{
        color: {TEXT} !important;
    }}

    [role="dialog"] input,
    [role="dialog"] textarea,
    [role="dialog"] button,
    [data-testid="stDialog"] button {{
        background-color: {CARD_ALT} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}

    [data-baseweb="popover"],
    [data-baseweb="popover"] > div {{
        background-color: {CARD} !important;
        color: {TEXT} !important;
    }}

    /* Improved readability for screen recordings and presentations. */
    .main-title {{
        font-size: 32px;
        font-weight: 900;
        color: {TEXT};
        letter-spacing: 0.5px;
        margin-bottom: 3px;
    }}

    .subtitle {{
        font-size: 15px;
        color: {MUTED};
        margin-bottom: 20px;
        line-height: 1.6;
    }}

    .section-title {{
        font-size: 22px;
        font-weight: 850;
        color: {TEXT};
        margin-top: 16px;
        margin-bottom: 14px;
    }}

    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 19px;
        min-height: 130px;
        margin-bottom: 13px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 13px;
        font-weight: 750;
        letter-spacing: 0.7px;
        margin-bottom: 11px;
    }}

    .metric-value {{
        color: {TEXT};
        font-size: 27px;
        font-weight: 900;
        overflow-wrap: anywhere;
    }}

    .metric-caption {{
        color: {MUTED};
        font-size: 13px;
        margin-top: 7px;
    }}

    .info-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 13px;
    }}

    .info-card strong {{
        color: {TEXT};
        font-size: 16px;
    }}

    .info-card p {{
        color: {MUTED} !important;
        font-size: 14px;
        margin-top: 7px;
        margin-bottom: 0;
        line-height: 1.7;
    }}

    .notice {{
        background: {CARD_ALT};
        border: 1px solid {BORDER};
        border-left: 5px solid {BLUE};
        border-radius: 11px;
        padding: 15px 17px;
        margin: 12px 0;
        color: {TEXT};
        line-height: 1.7;
    }}

    .notice.success {{ border-left-color: {GREEN}; }}
    .notice.warning {{ border-left-color: {ORANGE}; }}
    .notice.danger {{ border-left-color: {RED}; }}

    .notice-title {{
        font-weight: 800;
        margin-bottom: 5px;
        color: {TEXT};
        font-size: 15px;
    }}

    .notice-body {{
        color: {MUTED};
        font-size: 14px;
    }}

    .feature-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 19px;
        margin-bottom: 12px;
        min-height: 190px;
    }}

    .feature-heading {{
        color: {TEXT};
        font-weight: 850;
        font-size: 17px;
        margin-bottom: 12px;
    }}

    .feature-item {{
        color: {MUTED};
        padding: 8px 0;
        border-bottom: 1px solid {BORDER};
        font-size: 14px;
    }}

    .feature-item:last-child {{ border-bottom: none; }}

    .timeline-item {{
        display: flex;
        gap: 14px;
        padding: 13px 14px;
        margin-bottom: 9px;
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        align-items: center;
    }}

    .timeline-number {{
        min-width: 34px;
        height: 34px;
        border-radius: 50%;
        background: {CARD_ALT};
        border: 1px solid {BLUE};
        color: {BLUE};
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
    }}

    .timeline-name {{
        color: {TEXT};
        font-weight: 800;
        font-size: 15px;
    }}

    .timeline-description {{
        color: {MUTED};
        font-size: 13px;
        margin-top: 3px;
        line-height: 1.6;
    }}

    .company-hero {{
        background: linear-gradient(135deg, {CARD}, {CARD_ALT});
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 18px;
    }}

    .company-hero h2 {{
        color: {TEXT} !important;
        font-size: 28px;
        margin-bottom: 8px;
    }}

    .company-hero p {{
        color: {MUTED} !important;
        line-height: 1.7;
        font-size: 15px;
    }}

    .company-tag {{
        display: inline-block;
        background: {CARD_ALT};
        border: 1px solid {BORDER};
        border-radius: 20px;
        color: {TEXT};
        padding: 6px 12px;
        margin: 4px 5px 4px 0;
        font-size: 13px;
        font-weight: 700;
    }}

    .footer {{
        text-align: center;
        color: {MUTED};
        font-size: 13px;
        padding: 20px;
        line-height: 1.8;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 4. REUSABLE FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame()

def find_column(df, candidates):
    if df.empty:
        return None

    def normalize(value):
        return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

    normalized = {normalize(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize(candidate)

        if key in normalized:
            return normalized[key]

    return None

def safe_html(value):
    return html.escape(str(value))

def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_html(label)}</div>
            <div class="metric-value">{safe_html(value)}</div>
            <div class="metric-caption">{safe_html(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def section_title(title):
    st.markdown(
        f'<div class="section-title">{safe_html(title)}</div>',
        unsafe_allow_html=True
    )

def info_card(title, description):
    st.markdown(
        f"""
        <div class="info-card">
            <strong>{safe_html(title)}</strong>
            <p>{safe_html(description)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def notice(title, message, kind="info"):
    st.markdown(
        f"""
        <div class="notice {kind}">
            <div class="notice-title">{safe_html(title)}</div>
            <div class="notice-body">{safe_html(message)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def feature_card(title, items):
    item_html = "".join(
        f'<div class="feature-item">{safe_html(item)}</div>'
        for item in items
    )

    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-heading">{safe_html(title)}</div>
            {item_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def money(value):
    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"

def chart_theme(fig, height=420):
    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family="Arial", size=14, color=TEXT),
        title_font=dict(family="Arial", size=19, color=TEXT),
        height=height,
        margin=dict(l=25, r=25, t=65, b=35),
        hoverlabel=dict(
            bgcolor=CARD_ALT,
            bordercolor=BLUE,
            font=dict(color=TEXT, size=14)
        ),
        legend=dict(
            font=dict(color=TEXT, size=13),
            bgcolor=CARD,
            bordercolor=BORDER,
            borderwidth=1
        ),
        xaxis=dict(
            color=TEXT,
            tickfont=dict(color=TEXT, size=12),
            title_font=dict(color=TEXT),
            gridcolor=GRID,
            zerolinecolor=GRID,
            showgrid=True
        ),
        yaxis=dict(
            color=TEXT,
            tickfont=dict(color=TEXT, size=12),
            title_font=dict(color=TEXT),
            gridcolor=GRID,
            zerolinecolor=GRID,
            showgrid=True
        )
    )

    return fig

def file_status(path):
    if path.exists():
        return {
            "Status": "Available",
            "Size (KB)": round(path.stat().st_size / 1024, 2)
        }

    return {
        "Status": "Not found",
        "Size (KB)": None
    }

def normalize_dates(series):
    """
    Convert date values into consistent UTC-aware timestamps.
    Invalid dates become NaT.
    """
    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True
    )

def get_date_range(df, candidates=None):
    if df.empty:
        return None, None

    candidates = candidates or [
        "published_at",
        "date",
        "datetime",
        "trading_date",
        "aligned_date"
    ]

    date_col = find_column(df, candidates)

    if date_col is None:
        return None, None

    dates = normalize_dates(df[date_col]).dropna()

    if dates.empty:
        return None, None

    # Both values are UTC-aware, so comparisons are consistent.
    return dates.min(), dates.max()

def timeline_item(number, title, description):
    st.markdown(
        f"""
        <div class="timeline-item">
            <div class="timeline-number">{number}</div>
            <div>
                <div class="timeline-name">{safe_html(title)}</div>
                <div class="timeline-description">{safe_html(description)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 5. LOAD PROJECT DATA
# ============================================================

stock = load_csv(str(STOCK_FILE))

fusion = load_csv(str(FUSION_FILE))

if fusion.empty:
    fusion = load_csv(str(ORIGINAL_FUSION_FILE))

predictions = load_csv(str(PREDICTIONS_FILE))
results = load_csv(str(RESULTS_FILE))

news_data = {}

for name, path in NEWS_FILES.items():
    df = load_csv(str(path))

    if not df.empty:
        news_data[name] = df

if stock.empty:
    st.error(
        "RELIANCE stock data could not be loaded. "
        "Check data/RELIANCE.csv."
    )
    st.stop()

stock.columns = stock.columns.str.strip()

if "Date" not in stock.columns or "Close" not in stock.columns:
    st.error("Stock dataset must contain Date and Close columns.")
    st.stop()


stock["Date"] = pd.to_datetime(
    stock["Date"],
    errors="coerce",
    utc=True
).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)


for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits"
]:
    if col in stock.columns:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")

stock = (
    stock.dropna(subset=["Date", "Close"])
    .sort_values("Date")
    .reset_index(drop=True)
)

if stock.empty:
    st.error("No valid stock records were found.")
    st.stop()

# ============================================================
# 6. TECHNICAL INDICATORS
# ============================================================

stock["Daily_Return"] = stock["Close"].pct_change() * 100
stock["MA20"] = stock["Close"].rolling(20).mean()
stock["MA50"] = stock["Close"].rolling(50).mean()
stock["MA200"] = stock["Close"].rolling(200).mean()
stock["Volatility"] = stock["Daily_Return"].rolling(20).std()

latest = stock.iloc[-1]
previous = stock.iloc[-2] if len(stock) > 1 else latest

latest_price = float(latest["Close"])
previous_price = float(previous["Close"])

daily_change = latest_price - previous_price

daily_change_pct = (
    daily_change / previous_price * 100
    if previous_price != 0 else 0
)

first_price = float(stock.iloc[0]["Close"])

period_change = latest_price - first_price

period_return = (
    period_change / first_price * 100
    if first_price != 0 else 0
)

highest_price = stock["Close"].max()
lowest_price = stock["Close"].min()

latest_volatility = stock["Volatility"].iloc[-1]

if pd.isna(latest_volatility):
    latest_volatility = 0

# ============================================================
# 7. SIDEBAR NAVIGATION AND PROJECT STATUS
# ============================================================

st.sidebar.divider()

page = st.sidebar.radio(
    "DASHBOARD NAVIGATION",
    [
        "Overview",
        "About the Company",
        "Technical Analysis",
        "Data Collected",
        "News Analysis",
        "Models & Methodology",
        "Model Performance",
        "Historical Predictions",
        "Insights & Recommendations"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### PROJECT STATUS")
st.sidebar.caption("Company: Reliance Industries Limited")
st.sidebar.caption(f"Stock records: {len(stock):,}")
st.sidebar.caption(f"News datasets: {len(news_data)}")

st.sidebar.caption(
    f"Fusion records: {len(fusion):,}"
    if not fusion.empty else "Fusion records: unavailable"
)

st.sidebar.caption(
    f"Stock period: {stock['Date'].min():%d %b %Y} – "
    f"{stock['Date'].max():%d %b %Y}"
)

st.sidebar.caption("Ticker: NSE: RELIANCE")

# ============================================================
# 8. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📈 RELIANCE STOCK INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Premium Market Research | Historical News | AI Predictions | '
    'Sentiment Analysis | Investment Research'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "Latest available stock record: "
    + latest["Date"].strftime("%d %B %Y")
)

# ============================================================
# 9. OVERVIEW
# ============================================================

if page == "Overview":

    section_title("Market Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "LATEST CLOSING PRICE",
            money(latest_price),
            latest["Date"].strftime("%d %b %Y")
        )

    with c2:
        metric_card(
            "DAILY CHANGE",
            f"{daily_change:+,.2f}",
            f"{daily_change_pct:+.2f}%"
        )

    with c3:
        metric_card(
            "PRICE CHANGE OVER DATA PERIOD",
            f"{period_return:+.2f}%",
            f"{period_change:+,.2f} per share"
        )

    with c4:
        metric_card(
            "HIGHEST CLOSING PRICE",
            money(highest_price),
            "Available dataset"
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        metric_card(
            "LOWEST CLOSING PRICE",
            money(lowest_price),
            "Available dataset"
        )

    with c6:
        metric_card(
            "MA20",
            money(stock["MA20"].iloc[-1]),
            "20-session moving average"
        )

    with c7:
        metric_card(
            "MA50",
            money(stock["MA50"].iloc[-1]),
            "50-session moving average"
        )

    with c8:
        metric_card(
            "20-DAY VOLATILITY",
            f"{latest_volatility:.2f}%",
            "Historical daily return volatility"
        )

    st.divider()

    section_title("RELIANCE Historical Stock Price")

    price_chart = go.Figure()

    if all(col in stock.columns for col in ["Open", "High", "Low", "Close"]):
        price_chart.add_trace(
            go.Candlestick(
                x=stock["Date"],
                open=stock["Open"],
                high=stock["High"],
                low=stock["Low"],
                close=stock["Close"],
                name="RELIANCE",
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
                increasing_fillcolor=GREEN,
                decreasing_fillcolor=RED
            )
        )

    else:
        price_chart.add_trace(
            go.Scatter(
                x=stock["Date"],
                y=stock["Close"],
                name="Closing Price",
                line=dict(color=BLUE, width=2.5),
                fill="tozeroy",
                fillcolor=BLUE_FILL
            )
        )

    price_chart.update_layout(
        title="Historical Price Movement",
        xaxis_rangeslider_visible=False,
        yaxis_title="Price (₹)"
    )

    st.plotly_chart(
        chart_theme(price_chart, 500),
        use_container_width=True
    )

    section_title("Historical Trading Volume")

    if "Volume" in stock.columns:

        rising = (
            stock["Close"] >= stock["Open"]
            if "Open" in stock.columns
            else stock["Daily_Return"] >= 0
        )

        volume_chart = go.Figure()

        volume_chart.add_trace(
            go.Bar(
                x=stock["Date"],
                y=stock["Volume"].where(rising),
                name="Positive Session",
                marker=dict(
                    color=GREEN,
                    opacity=0.95,
                    line=dict(color=GREEN, width=0.5)
                )
            )
        )

        volume_chart.add_trace(
            go.Bar(
                x=stock["Date"],
                y=stock["Volume"].where(~rising),
                name="Negative Session",
                marker=dict(
                    color=RED,
                    opacity=0.95,
                    line=dict(color=RED, width=0.5)
                )
            )
        )

        volume_chart.update_layout(
            title="Daily Trading Volume — Green: Positive | Red: Negative",
            barmode="overlay",
            yaxis_title="Shares Traded"
        )

        st.plotly_chart(
            chart_theme(volume_chart, 420),
            use_container_width=True
        )

        st.caption(
            "Green represents sessions where Close >= Open. "
            "Red represents sessions where Close < Open."
        )

    else:
        notice(
            "Volume unavailable",
            "The stock dataset does not contain a Volume column.",
            "warning"
        )

    st.divider()

    section_title("Recent Trading Records")

    st.dataframe(
        stock.tail(15).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 10. ABOUT THE COMPANY
# ============================================================

elif page == "About the Company":

    st.markdown(
        """
        <div class="company-hero">
            <h2>🏢 Reliance Industries Limited</h2>
            <p>
                Reliance Industries Limited (RIL) is a major Indian
                diversified company with businesses spanning energy,
                materials, retail, digital services, media and
                entertainment, and new energy.
            </p>
            <span class="company-tag">🇮🇳 Indian Company</span>
            <span class="company-tag">Publicly Listed</span>
            <span class="company-tag">Fortune Global 500</span>
            <span class="company-tag">Founded by Dhirubhai Ambani</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_title("Company at a Glance")

    company_facts = [
        ("Full Name", "Reliance Industries Limited"),
        ("Country", "India"),
        ("Headquarters", "Navi Mumbai, Maharashtra, India"),
        ("Founder", "Dhirubhai H. Ambani"),
        ("Chairman & Managing Director", "Mukesh D. Ambani"),
        ("Company Type", "Publicly listed Indian company"),
        ("Stock Exchange", "National Stock Exchange (NSE) and BSE"),
        ("NSE Symbol", "RELIANCE"),
        ("BSE Code", "500325"),
        ("Official Website", "https://www.ril.com"),
        ("Corporate Motto", "Growth is Life")
    ]

    facts_df = pd.DataFrame(
        company_facts,
        columns=["Company Information", "Details"]
    )

    st.dataframe(
        facts_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section_title("What Does Reliance Industries Do?")

    st.write(
        "Reliance is not limited to one industry. It operates across "
        "several major sectors of the Indian and global economy."
    )

    col1, col2 = st.columns(2)

    with col1:
        info_card(
            "1. Energy and Oil & Gas",
            "Exploration and production of hydrocarbons, oil refining, "
            "petroleum products and energy-related operations."
        )

        info_card(
            "2. Petrochemicals and Materials",
            "Manufacturing of petrochemicals, polymers, polyester, "
            "advanced materials and related industrial products."
        )

        info_card(
            "3. Retail",
            "Retail stores and digital commerce serving customers "
            "through grocery, fashion, electronics and other categories."
        )

        info_card(
            "4. Digital Services",
            "Telecommunications, broadband, connectivity and digital "
            "services through the Jio business."
        )

    with col2:
        info_card(
            "5. New Energy",
            "Developing clean-energy capabilities including solar, "
            "batteries, hydrogen and related technologies."
        )

        info_card(
            "6. Media and Entertainment",
            "Television, digital content, entertainment and media "
            "distribution businesses."
        )

        info_card(
            "7. Financial Services",
            "Financial services activities through associated "
            "businesses, including Jio Financial Services."
        )

        info_card(
            "8. Consumer Products",
            "Consumer brands and everyday products across beverages, "
            "food, home care and personal care categories."
        )

    st.divider()

    section_title("Popular Businesses and Brands")

    brand_col1, brand_col2, brand_col3 = st.columns(3)

    with brand_col1:
        feature_card(
            "DIGITAL",
            [
                "Jio",
                "JioFiber",
                "Jio Platforms",
                "Digital connectivity services"
            ]
        )

    with brand_col2:
        feature_card(
            "RETAIL",
            [
                "Reliance Retail",
                "Trends",
                "Smart Bazaar",
                "Reliance Digital"
            ]
        )

    with brand_col3:
        feature_card(
            "CONSUMER PRODUCTS",
            [
                "Campa",
                "Independence",
                "Reliance Consumer Products",
                "Everyday consumer goods"
            ]
        )

    st.divider()

    section_title("History of Reliance")

    timeline_item(
        1,
        "1957 — Early Beginnings",
        "Dhirubhai Ambani returned to India and began a yarn trading business in Mumbai."
    )

    timeline_item(
        2,
        "1977 — Public Issue",
        "Reliance Textile Industries' public issue helped broaden Indian retail participation in equity markets."
    )

    timeline_item(
        3,
        "1980s — Petrochemicals",
        "Reliance expanded into polyester and petrochemicals manufacturing."
    )

    timeline_item(
        4,
        "2000s — Energy and Retail",
        "The company expanded its energy operations and entered organised retail."
    )

    timeline_item(
        5,
        "2010s — Digital Services",
        "Jio's commercial launch in 2016 helped transform India's telecom and digital services market."
    )

    timeline_item(
        6,
        "Today — Diversified Businesses",
        "Reliance operates across energy, materials, retail, digital services, media and new energy."
    )

    st.divider()

    section_title("Leadership")

    info_card(
        "Dhirubhai H. Ambani — Founder-Chairman",
        "Dhirubhai Ambani founded the business that grew into Reliance "
        "Industries. His entrepreneurial journey began with a yarn "
        "trading business and expanded into textiles and industry."
    )

    info_card(
        "Mukesh D. Ambani — Chairman and Managing Director",
        "Mukesh Ambani leads Reliance Industries Limited as its "
        "Chairman and Managing Director. He has been associated with "
        "the company's Board since 1977."
    )

    st.divider()

    section_title("Why Is Reliance Important to India?")

    info_card(
        "Employment and Economic Activity",
        "Its businesses operate across major sectors and support "
        "large supply chains, customers, employees and business partners."
    )

    info_card(
        "Digital Connectivity",
        "Jio has played a major role in expanding mobile data and "
        "digital connectivity in India."
    )

    info_card(
        "Retail and Consumer Markets",
        "Reliance Retail operates across multiple consumer categories "
        "and serves customers throughout India."
    )

    info_card(
        "Energy and Industrial Development",
        "Its refining, petrochemical and energy operations form an "
        "important part of India's industrial ecosystem."
    )

    st.divider()

    section_title("Official Company Resources")

    st.markdown(
        "- [Reliance Industries — Official Website](https://www.ril.com)\n"
        "- [About Reliance](https://www.ril.com/about)\n"
        "- [Company History](https://www.ril.com/about/our-history)\n"
        "- [Board of Directors](https://www.ril.com/about/board-of-directors)\n"
        "- [Chairman and Managing Director](https://www.ril.com/about/chairman-managing-director)"
    )

    notice(
        "Company information note",
        "Company structure, business activities and leadership can change. "
        "For the latest official information, refer to Reliance's website "
        "and its published annual reports.",
        "info"
    )

# ============================================================
# 11. TECHNICAL ANALYSIS
# ============================================================

elif page == "Technical Analysis":

    section_title("Technical Market Analysis")

    a, b, c = st.columns(3)

    with a:
        metric_card(
            "20-DAY MOVING AVERAGE",
            money(stock["MA20"].iloc[-1]),
            "Short-term trend"
        )

    with b:
        metric_card(
            "50-DAY MOVING AVERAGE",
            money(stock["MA50"].iloc[-1]),
            "Medium-term trend"
        )

    with c:
        metric_card(
            "200-DAY MOVING AVERAGE",
            money(stock["MA200"].iloc[-1]),
            "Long-term trend"
        )

    st.divider()

    section_title("Moving Average Analysis")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stock["Date"],
            y=stock["Close"],
            name="Closing Price",
            line=dict(color=BLUE, width=2.5),
            fill="tozeroy",
            fillcolor=BLUE_FILL
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stock["Date"],
            y=stock["MA20"],
            name="MA20",
            line=dict(color=GREEN, width=2),
            fill="tozeroy",
            fillcolor=GREEN_FILL
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stock["Date"],
            y=stock["MA50"],
            name="MA50",
            line=dict(color=ORANGE, width=2),
            fill="tozeroy",
            fillcolor=ORANGE_FILL
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stock["Date"],
            y=stock["MA200"],
            name="MA200",
            line=dict(color=PURPLE, width=2),
            fill="tozeroy",
            fillcolor=PURPLE_FILL
        )
    )

    fig.update_layout(
        title="RELIANCE Closing Price and Moving Averages",
        yaxis_title="Price (₹)"
    )

    st.plotly_chart(
        chart_theme(fig, 500),
        use_container_width=True
    )

    st.divider()

    section_title("Daily Percentage Returns")

    return_data = stock.dropna(subset=["Daily_Return"]).copy()

    return_chart = go.Figure()

    return_chart.add_trace(
        go.Bar(
            x=return_data["Date"],
            y=return_data["Daily_Return"].where(return_data["Daily_Return"] >= 0),
            name="Positive Return",
            marker=dict(
                color=GREEN,
                opacity=0.95,
                line=dict(color=GREEN, width=0.5)
            )
        )
    )

    return_chart.add_trace(
        go.Bar(
            x=return_data["Date"],
            y=return_data["Daily_Return"].where(return_data["Daily_Return"] < 0),
            name="Negative Return",
            marker=dict(
                color=RED,
                opacity=0.95,
                line=dict(color=RED, width=0.5)
            )
        )
    )

    return_chart.update_layout(
        title="Daily Share Price Returns",
        yaxis_title="Daily Return (%)",
        barmode="relative",
        bargap=0.15
    )

    return_chart.add_hline(
        y=0,
        line_width=1.5,
        line_color=TEXT
    )

    st.plotly_chart(
        chart_theme(return_chart, 450),
        use_container_width=True
    )

    notice(
        "How to read this chart",
        "Green bars show positive daily percentage returns. Red bars show "
        "negative daily percentage returns. The horizontal zero line "
        "separates gains from losses.",
        "info"
    )

    st.divider()

    section_title("Technical Indicator Records")

    st.dataframe(
        stock.tail(50).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 12. DATA COLLECTED
# ============================================================

elif page == "Data Collected":

    section_title("Project Data Collection Centre")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "STOCK RECORDS",
            f"{len(stock):,}",
            "Historical trading sessions"
        )

    with c2:
        metric_card(
            "STOCK COLUMNS",
            str(len(stock.columns)),
            "Available fields"
        )

    with c3:
        metric_card(
            "NEWS FILES",
            str(len(news_data)),
            "Available CSV files"
        )

    with c4:
        metric_card(
            "FUSION RECORDS",
            f"{len(fusion):,}" if not fusion.empty else "N/A",
            "Market and news dataset"
        )

    st.divider()

    section_title("Project Collection Timeline")

    stock_start = stock["Date"].min().strftime("%d %b %Y")
    stock_end = stock["Date"].max().strftime("%d %b %Y")

    # FIX: normalize all date values to UTC-aware timestamps
    # before calculating minimum and maximum dates.
    news_ranges = []

    for name, df in news_data.items():
        start, end = get_date_range(df)

        if start is not None and end is not None:
            news_ranges.append((start, end))

    if news_ranges:

        # All timestamps are UTC-aware, so comparisons are safe.
        news_start_date = min(start for start, end in news_ranges)
        news_end_date = max(end for start, end in news_ranges)

        news_start = news_start_date.strftime("%d %b %Y")
        news_end = news_end_date.strftime("%d %b %Y")

        news_period = f"{news_start} – {news_end}"

    else:
        news_period = "No valid news dates detected"

    timeline_item(
        1,
        "Company",
        "Reliance Industries Limited — NSE ticker: RELIANCE"
    )

    timeline_item(
        2,
        "Stock Records",
        f"{len(stock):,} records | {stock_start} to {stock_end}"
    )

    timeline_item(
        3,
        "News Collection",
        f"{len(news_data)} available news datasets | {news_period}"
    )

    timeline_item(
        4,
        "Fusion Records",
        f"{len(fusion):,} combined records"
        if not fusion.empty
        else "Fusion dataset not available"
    )

    timeline_item(
        5,
        "Data Cleaning and Preparation",
        "Historical stock and news data prepared for analysis"
    )

    timeline_item(
        6,
        "Sentiment and Model Preparation",
        "FinBERT sentiment, aligned news and prepared training sequences"
    )

    timeline_item(
        7,
        "New Datasets and Saved Artifacts",
        "Training data, model predictions, comparison results and model files"
    )

    st.divider()

    section_title("Stock Dataset")

    st.write("**File:** `data/RELIANCE.csv`")
    st.write(f"**Start date:** {stock_start}")
    st.write(f"**End date:** {stock_end}")
    st.write("**Features:** " + ", ".join(stock.columns.astype(str)))

    quality = pd.DataFrame({
        "Column": stock.columns,
        "Data Type": [str(stock[c].dtype) for c in stock.columns],
        "Missing Values": [int(stock[c].isna().sum()) for c in stock.columns]
    })

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section_title("Project Files and Availability")

    project_files = {
        "Stock dataset": STOCK_FILE,
        "Fusion dataset": FUSION_FILE,
        "Original fusion dataset": ORIGINAL_FUSION_FILE,
        "Model comparison predictions": PREDICTIONS_FILE,
        "Model comparison results": RESULTS_FILE,
        "Prepared training data": TRAINING_FILE,
        "Best LSTM model": MODEL_FILE
    }

    project_files.update(NEWS_FILES)

    file_rows = []

    for name, path in project_files.items():

        status = file_status(path)

        try:
            relative_path = str(path.relative_to(BASE_DIR))
        except ValueError:
            relative_path = str(path)

        file_rows.append({
            "Dataset / File": name,
            "Path": relative_path,
            "Status": status["Status"],
            "Size (KB)": status["Size (KB)"]
        })

    st.dataframe(
        pd.DataFrame(file_rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section_title("News Dataset Inventory")

    if news_data:

        inventory = []

        for name, df in news_data.items():

            start, end = get_date_range(df)

            inventory.append({
                "Dataset": name,
                "Article / Record Count": len(df),
                "Column Count": len(df.columns),
                "Date Range": (
                    f"{start:%d %b %Y} – {end:%d %b %Y}"
                    if start is not None
                    else "Not identified"
                )
            })

        st.dataframe(
            pd.DataFrame(inventory),
            use_container_width=True,
            hide_index=True
        )

        selected = st.selectbox(
            "Explore a collected dataset",
            list(news_data.keys()),
            key="news_dataset_selector"
        )

        selected_df = news_data[selected]

        st.write(f"**Selected dataset:** {selected}")

        st.dataframe(
            selected_df.head(50),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Selected Dataset",
            data=selected_df.to_csv(index=False).encode("utf-8"),
            file_name=selected.lower().replace(" ", "_") + ".csv",
            mime="text/csv"
        )

    else:

        notice(
            "News data unavailable",
            "No historical news CSV files were found in the expected folders.",
            "warning"
        )

    st.divider()

    section_title("Fusion Dataset")

    if not fusion.empty:

        metric_card(
            "FUSION DATASET",
            f"{len(fusion):,} records",
            f"{len(fusion.columns)} columns"
        )

        st.write(
            "The fusion dataset combines stock-market information "
            "with available historical news and sentiment information."
        )

        st.dataframe(
            fusion.head(30),
            use_container_width=True,
            hide_index=True
        )

    else:

        notice(
            "Fusion dataset unavailable",
            "Check the fusion dataset path and confirm that the CSV exists.",
            "warning"
        )

    notice(
        "Important data interpretation",
        "A zero-news trading session is not automatically neutral. "
        "It can simply mean that no relevant article was available.",
        "info"
    )

# ============================================================
# 13. NEWS ANALYSIS
# ============================================================

elif page == "News Analysis":

    section_title("Historical News Intelligence")

    if not news_data:

        notice(
            "No news data found",
            "No news files were found in the expected folders.",
            "warning"
        )

    else:

        if "FinBERT Sentiment News" in news_data:
            article_df = news_data["FinBERT Sentiment News"]
            source_name = "FinBERT Sentiment News"

        elif "Aligned News" in news_data:
            article_df = news_data["Aligned News"]
            source_name = "Aligned News"

        elif "Cleaned Historical News" in news_data:
            article_df = news_data["Cleaned Historical News"]
            source_name = "Cleaned Historical News"

        else:
            source_name = list(news_data.keys())[0]
            article_df = news_data[source_name]

        article_df = article_df.copy()

        sentiment_col = find_column(
            article_df,
            ["sentiment", "predicted_sentiment", "sentiment_label", "label"]
        )

        title_col = find_column(
            article_df,
            ["title", "headline"]
        )

        date_col = find_column(
            article_df,
            ["published_at", "date", "datetime", "trading_date", "aligned_date"]
        )

        domain_col = find_column(
            article_df,
            ["domain", "source", "publisher"]
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "ARTICLES",
                f"{len(article_df):,}",
                source_name
            )

        with c2:
            metric_card(
                "NEWS SOURCES",
                str(article_df[domain_col].nunique())
                if domain_col else "N/A",
                "Unique publishers/domains"
            )

        with c3:
            metric_card(
                "SENTIMENT ANALYSIS",
                "Available" if sentiment_col else "Not available",
                "FinBERT output"
            )

        st.divider()

        if sentiment_col:

            section_title("Sentiment Distribution")

            sentiment_counts = (
                article_df[sentiment_col]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts()
            )

            sentiment_color_map = {
                "positive": GREEN,
                "negative": RED,
                "neutral": BLUE,
                "unknown": MUTED
            }

            pie_colors = [
                sentiment_color_map.get(
                    str(label).lower(),
                    PURPLE
                )
                for label in sentiment_counts.index
            ]

            pie = go.Figure(
                data=[
                    go.Pie(
                        labels=sentiment_counts.index.tolist(),
                        values=sentiment_counts.values.tolist(),
                        hole=0.5,
                        marker=dict(
                            colors=pie_colors,
                            line=dict(color=CARD, width=3)
                        ),
                        textinfo="label+percent",
                        textfont=dict(color=TEXT, size=14),
                        sort=False
                    )
                ]
            )

            pie.update_layout(
                title="Positive / Negative / Neutral Articles"
            )

            st.plotly_chart(
                chart_theme(pie, 400),
                use_container_width=True
            )

            sentiment_table = (
                sentiment_counts.rename_axis("Sentiment")
                .reset_index(name="Article Count")
            )

            st.dataframe(
                sentiment_table,
                use_container_width=True,
                hide_index=True
            )

        else:

            notice(
                "Sentiment column unavailable",
                "A recognized sentiment column was not found in the selected file.",
                "warning"
            )

        st.divider()

        section_title("Daily News Coverage")

        if date_col:

            article_df[date_col] = normalize_dates(article_df[date_col])

            dated = article_df.dropna(subset=[date_col]).copy()

            if not dated.empty:

                dated["_day"] = dated[date_col].dt.date

                daily_counts = (
                    dated.groupby("_day")
                    .size()
                    .reset_index(name="Articles")
                )

                coverage = go.Figure()

                coverage.add_trace(
                    go.Bar(
                        x=daily_counts["_day"],
                        y=daily_counts["Articles"],
                        name="Articles",
                        marker=dict(
                            color=BLUE,
                            opacity=0.95,
                            line=dict(color=BLUE, width=0.5)
                        )
                    )
                )

                coverage.update_layout(
                    title="Historical Articles Collected Per Day",
                    yaxis_title="Number of Articles"
                )

                st.plotly_chart(
                    chart_theme(coverage, 380),
                    use_container_width=True
                )

            else:

                notice(
                    "No valid dates",
                    "No valid publication dates were found in this dataset.",
                    "warning"
                )

        else:

            notice(
                "Date column unavailable",
                "No recognized publication date column was identified.",
                "warning"
            )

        st.divider()

        section_title("Daily Sentiment Summary")

        daily_df = news_data.get(
            "Combined Daily Sentiment",
            news_data.get("Daily Sentiment", pd.DataFrame())
        )

        if not daily_df.empty:

            st.dataframe(
                daily_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            notice(
                "Daily sentiment unavailable",
                "The daily sentiment file is not available.",
                "info"
            )

        st.divider()

        section_title("Collected Articles")

        display_columns = [
            col
            for col in [date_col, title_col, domain_col, sentiment_col]
            if col and col in article_df.columns
        ]

        if display_columns:

            st.dataframe(
                article_df[display_columns].head(100),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.dataframe(
                article_df.head(30),
                use_container_width=True,
                hide_index=True
            )

        notice(
            "Sentiment interpretation",
            "FinBERT sentiment is an automated text classification. "
            "It does not establish that news caused a particular stock-price movement.",
            "info"
        )

# ============================================================
# 14. MODELS AND METHODOLOGY
# ============================================================

elif page == "Models & Methodology":

    section_title("Machine Learning Models")

    info_card(
        "01 — Market-Only LSTM",
        "A Long Short-Term Memory neural network trained on historical "
        "market information. It learns patterns from price, trading "
        "volume and corporate action fields."
    )

    info_card(
        "02 — Market + News LSTM",
        "Uses the same market inputs plus aggregated news information "
        "and FinBERT sentiment probabilities."
    )

    info_card(
        "03 — Previous-Close Baseline",
        "A benchmark that predicts the next closing price as the "
        "previous trading session's closing price. It is not a trained neural network."
    )

    st.divider()

    section_title("LSTM Model Architecture")

    architecture = pd.DataFrame({
        "Layer": [
            "Input",
            "LSTM Layer 1",
            "Dropout",
            "LSTM Layer 2",
            "Output"
        ],
        "Configuration": [
            "12 input features",
            "64 hidden units",
            "0.2 dropout",
            "32 hidden units",
            "1 predicted value"
        ],
        "Purpose": [
            "Market and news information",
            "Learn sequential patterns",
            "Reduce overfitting",
            "Extract higher-level patterns",
            "Predict closing price"
        ]
    })

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section_title("Training Configuration")

    training = pd.DataFrame({
        "Parameter": [
            "Sequence Length",
            "Training Data",
            "Validation Data",
            "Testing Data",
            "Batch Size",
            "Optimizer",
            "Learning Rate",
            "Loss Function",
            "Maximum Epochs"
        ],
        "Value": [
            "60 trading sessions",
            "80%",
            "10%",
            "10%",
            "32",
            "Adam",
            "0.001",
            "Mean Squared Error",
            "100"
        ]
    })

    st.dataframe(
        training,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section_title("Input Feature Design")

    col1, col2 = st.columns(2)

    with col1:

        feature_card(
            "MARKET DATA INPUTS",
            [
                "Open — Opening share price",
                "High — Highest price in the session",
                "Low — Lowest price in the session",
                "Close — Closing share price",
                "Volume — Shares traded",
                "Dividends — Dividend information",
                "Stock Splits — Split information"
            ]
        )

    with col2:

        feature_card(
            "NEWS & SENTIMENT INPUTS",
            [
                "Article Count — Number of relevant articles",
                "Average Sentiment — Aggregated sentiment score",
                "Positive Probability — Positive news probability",
                "Negative Probability — Negative news probability",
                "Neutral Probability — Neutral news probability"
            ]
        )

    st.divider()

    section_title("Project Workflow")

    workflow = [
        "Collect historical Reliance stock prices.",
        "Collect historical news articles.",
        "Clean and prepare news data.",
        "Run FinBERT sentiment analysis.",
        "Align news with trading sessions.",
        "Combine market and news information.",
        "Prepare chronological training sequences.",
        "Train market-only and market-plus-news LSTMs.",
        "Compare predictions with actual closing prices.",
        "Evaluate results against the previous-close baseline."
    ]

    for i, item in enumerate(workflow, 1):
        timeline_item(i, f"Stage {i}", item)

    notice(
        "Model validation approach",
        "The project uses chronological data splits and training-only "
        "scaler fitting to reduce future-information leakage.",
        "info"
    )

    st.divider()

    section_title("Saved Training Artifacts")

    artifact_rows = []

    for path in [TRAINING_FILE, MODEL_FILE] + SCALER_FILES:

        status = file_status(path)

        artifact_rows.append({
            "Artifact": path.name,
            "Status": status["Status"],
            "Size (KB)": status["Size (KB)"]
        })

    st.dataframe(
        pd.DataFrame(artifact_rows),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 15. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    section_title("AI Model Performance Comparison")

    if results.empty:

        notice(
            "Model results unavailable",
            "Check data/training/model_comparison_results.csv.",
            "danger"
        )

    else:

        model_col = find_column(
            results,
            ["model", "model_name"]
        )

        mae_col = find_column(
            results,
            ["mae", "mean_absolute_error"]
        )

        rmse_col = find_column(
            results,
            ["rmse", "root_mean_squared_error"]
        )

        mape_col = find_column(
            results,
            ["mape", "mean_absolute_percentage_error"]
        )

        direction_col = find_column(
            results,
            [
                "directional_accuracy",
                "directional_accuracy_%",
                "directional_accuracy_percent"
            ]
        )

        if model_col is None:

            notice(
                "Model column missing",
                "The model name column could not be identified.",
                "danger"
            )

        else:

            result_df = results.copy()

            for col in [
                mae_col,
                rmse_col,
                mape_col,
                direction_col
            ]:

                if col:
                    result_df[col] = pd.to_numeric(
                        result_df[col],
                        errors="coerce"
                    )

            best_name = "N/A"
            best_mae = np.nan

            if mae_col:

                valid = result_df.dropna(subset=[mae_col])

                if not valid.empty:

                    best = valid.loc[valid[mae_col].idxmin()]

                    best_name = str(best[model_col])
                    best_mae = best[mae_col]

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card(
                    "BEST MODEL BY MAE",
                    best_name,
                    "Lowest test error"
                )

            with c2:
                metric_card(
                    "LOWEST MAE",
                    money(best_mae),
                    "Lower is better"
                )

            with c3:
                metric_card(
                    "MODELS COMPARED",
                    str(len(result_df)),
                    "Including baseline"
                )

            st.divider()

            section_title("Detailed Model Results")

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            if mae_col:

                section_title("MAE Comparison")

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=result_df[model_col],
                        y=result_df[mae_col],
                        name="MAE",
                        marker=dict(
                            color=BLUE,
                            opacity=0.95,
                            line=dict(color=BLUE, width=0.8)
                        ),
                        text=result_df[mae_col].round(2),
                        textposition="outside",
                        textfont=dict(color=TEXT, size=14)
                    )
                )

                fig.update_layout(
                    title="Mean Absolute Error",
                    yaxis_title="Error (₹)"
                )

                st.plotly_chart(
                    chart_theme(fig, 420),
                    use_container_width=True
                )

            if rmse_col:

                section_title("RMSE Comparison")

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=result_df[model_col],
                        y=result_df[rmse_col],
                        name="RMSE",
                        marker=dict(
                            color=PURPLE,
                            opacity=0.95,
                            line=dict(color=PURPLE, width=0.8)
                        ),
                        text=result_df[rmse_col].round(2),
                        textposition="outside",
                        textfont=dict(color=TEXT)
                    )
                )

                fig.update_layout(
                    title="Root Mean Squared Error",
                    yaxis_title="Error (₹)"
                )

                st.plotly_chart(
                    chart_theme(fig, 420),
                    use_container_width=True
                )

            if mape_col:

                section_title("MAPE Comparison")

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=result_df[model_col],
                        y=result_df[mape_col],
                        name="MAPE",
                        marker=dict(
                            color=ORANGE,
                            opacity=0.95,
                            line=dict(color=ORANGE, width=0.8)
                        ),
                        text=result_df[mape_col].round(2),
                        textposition="outside",
                        textfont=dict(color=TEXT)
                    )
                )

                fig.update_layout(
                    title="Mean Absolute Percentage Error",
                    yaxis_title="Error (%)"
                )

                st.plotly_chart(
                    chart_theme(fig, 420),
                    use_container_width=True
                )

            if direction_col:

                section_title("Directional Accuracy")

                direction_values = result_df[direction_col].dropna()

                upper = (
                    max(10, float(direction_values.max()) * 1.2)
                    if not direction_values.empty
                    else 100
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=result_df[model_col],
                        y=result_df[direction_col],
                        name="Directional Accuracy",
                        marker=dict(
                            color=GREEN,
                            opacity=0.95,
                            line=dict(color=GREEN, width=0.8)
                        ),
                        text=[
                            f"{v:.2f}%" if pd.notna(v) else ""
                            for v in result_df[direction_col]
                        ],
                        textposition="outside",
                        textfont=dict(color=TEXT, size=14),
                        cliponaxis=False
                    )
                )

                fig.update_layout(
                    title="Directional Accuracy by Model",
                    yaxis_title="Directional Accuracy (%)",
                    yaxis=dict(range=[0, upper])
                )

                st.plotly_chart(
                    chart_theme(fig, 450),
                    use_container_width=True
                )

            notice(
                "Baseline interpretation",
                "The previous-close baseline predicts zero price change. "
                "Its directional accuracy is not directly comparable "
                "to the LSTM models because a zero-change prediction "
                "does not provide a meaningful up/down forecast.",
                "warning"
            )

# ============================================================
# 16. HISTORICAL PREDICTIONS
# ============================================================

elif page == "Historical Predictions":

    section_title("Historical Prediction Analysis")

    # --------------------------------------------------------
    # Model inference using the saved Market + News LSTM
    # --------------------------------------------------------

    section_title("Run Market + News LSTM Prediction")

    st.write(
        "Run the saved PyTorch LSTM model using the latest available "
        "market and news-sentiment records in the fusion dataset."
    )

    if st.button("Run Model Prediction", type="primary", key="run_lstm_prediction"):

        try:
            with st.spinner("Loading model and preparing the latest 60-session sequence..."):
                live_result = predict_next_close(BASE_DIR)

            predicted_close = float(live_result["predicted_close"])
            last_actual_close = float(live_result["last_actual_close"])
            latest_data_date = pd.to_datetime(
                live_result["latest_data_date"],
                errors="coerce"
            )
            sequence_length = int(live_result["sequence_length"])
            model_name = str(live_result["model_name"])

            if pd.isna(latest_data_date):
                raise ValueError("The model returned an invalid latest data date.")

            st.session_state["reliance_lstm_prediction"] = {
                "predicted_close": predicted_close,
                "last_actual_close": last_actual_close,
                "latest_data_date": latest_data_date,
                "sequence_length": sequence_length,
                "model_name": model_name
            }

        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

    saved_prediction = st.session_state.get("reliance_lstm_prediction")

    if saved_prediction is not None:

        predicted_close = saved_prediction["predicted_close"]
        last_actual_close = saved_prediction["last_actual_close"]
        latest_data_date = saved_prediction["latest_data_date"]

        change_amount = predicted_close - last_actual_close
        change_percent = (
            change_amount / last_actual_close * 100
            if last_actual_close != 0 else 0.0
        )

        p1, p2, p3 = st.columns(3)

        with p1:
            metric_card(
                "MODEL-PREDICTED CLOSING PRICE",
                money(predicted_close),
                "Market + News LSTM output"
            )

        with p2:
            metric_card(
                "LAST ACTUAL CLOSING PRICE",
                money(last_actual_close),
                latest_data_date.strftime("%d %b %Y")
            )

        with p3:
            metric_card(
                "PREDICTED CHANGE VS LAST CLOSE",
                f"{change_amount:+,.2f}",
                f"{change_percent:+.2f}% (model estimate)"
            )

        st.caption(
            f"Model: {saved_prediction['model_name']} | "
            f"Input sequence: {saved_prediction['sequence_length']} trading sessions | "
            f"Latest input-data date: {latest_data_date:%d %B %Y}"
        )

        notice(
            "Historical-data limitation",
            "This output is calculated from the latest records included in the "
            "project's saved dataset. It is not a live market quote. Update the "
            "stock and news datasets before treating it as a forecast for a newer session.",
            "warning"
        )

    st.divider()

    required = [
        "Date",
        "Actual_Close",
        "Market_Only_Prediction",
        "Market_News_Prediction",
        "Previous_Close_Prediction"
    ]

    if predictions.empty:

        notice(
            "Prediction file unavailable",
            "Historical prediction file was not found.",
            "danger"
        )

    elif not all(col in predictions.columns for col in required):

        notice(
            "Prediction columns missing",
            "Expected prediction columns were not found in the CSV.",
            "danger"
        )

        st.write("Available columns:", list(predictions.columns))

    else:

        pred = predictions.copy()

        pred["Date"] = pd.to_datetime(
            pred["Date"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        for col in required[1:]:

            pred[col] = pd.to_numeric(
                pred[col],
                errors="coerce"
            )

        pred = (
            pred.dropna(subset=["Date", "Actual_Close"])
            .sort_values("Date")
        )

        if pred.empty:

            notice(
                "No valid prediction records",
                "No valid historical prediction rows were found.",
                "warning"
            )

        else:

            latest_pred = pred.iloc[-1]

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card(
                    "ACTUAL CLOSING PRICE",
                    money(latest_pred["Actual_Close"]),
                    latest_pred["Date"].strftime("%d %b %Y")
                )

            with c2:
                metric_card(
                    "MARKET-ONLY PREDICTION",
                    money(latest_pred["Market_Only_Prediction"]),
                    "Historical test prediction"
                )

            with c3:
                metric_card(
                    "MARKET + NEWS PREDICTION",
                    money(latest_pred["Market_News_Prediction"]),
                    "Historical test prediction"
                )

            st.divider()

            section_title("Actual vs Predicted Closing Prices")

            model_options = {
                "Market-Only LSTM": "Market_Only_Prediction",
                "Market + News LSTM": "Market_News_Prediction",
                "Previous-Close Baseline": "Previous_Close_Prediction"
            }

            selected = st.multiselect(
                "Select prediction lines",
                list(model_options.keys()),
                default=list(model_options.keys())
            )

            prediction_chart = go.Figure()

            prediction_chart.add_trace(
                go.Scatter(
                    x=pred["Date"],
                    y=pred["Actual_Close"],
                    name="Actual Closing Price",
                    line=dict(color=ACTUAL_COLOR, width=3.5)
                )
            )

            model_colors = {
                "Market-Only LSTM": BLUE,
                "Market + News LSTM": GREEN,
                "Previous-Close Baseline": ORANGE
            }

            for model_name in selected:

                column = model_options[model_name]

                prediction_chart.add_trace(
                    go.Scatter(
                        x=pred["Date"],
                        y=pred[column],
                        name=model_name,
                        line=dict(
                            color=model_colors[model_name],
                            width=2
                        )
                    )
                )

            prediction_chart.update_layout(
                title="Historical Actual vs Model Predictions",
                yaxis_title="Closing Price (₹)"
            )

            st.plotly_chart(
                chart_theme(prediction_chart, 500),
                use_container_width=True
            )

            st.divider()

            section_title("Prediction Error Summary")

            error_rows = []

            for model_name, column in model_options.items():

                valid = pred.dropna(
                    subset=[column, "Actual_Close"]
                )

                if not valid.empty:

                    errors = (
                        valid[column] - valid["Actual_Close"]
                    )

                    error_rows.append({
                        "Model": model_name,
                        "MAE (₹)": errors.abs().mean(),
                        "RMSE (₹)": np.sqrt((errors ** 2).mean()),
                        "Mean Error (₹)": errors.mean()
                    })

            if error_rows:

                st.dataframe(
                    pd.DataFrame(error_rows).round(3),
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            section_title("Historical Prediction Records")

            st.dataframe(
                pred.sort_values("Date", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Prediction Data",
                data=pred.to_csv(index=False).encode("utf-8"),
                file_name="reliance_model_predictions.csv",
                mime="text/csv"
            )

            st.caption(
                f"Total test predictions: {len(pred):,}"
            )

# ============================================================
# 17. INSIGHTS AND RECOMMENDATIONS
# ============================================================

elif page == "Insights & Recommendations":

    section_title("Reliance Research and Business Insights")

    st.write(
        "This section interprets historical share-price movement, "
        "news sentiment, model performance and technical indicators."
    )

    section_title("1. Historical Share Price Gain / Loss")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "FIRST RECORDED CLOSE",
            money(first_price),
            stock.iloc[0]["Date"].strftime("%d %b %Y")
        )

    with c2:
        metric_card(
            "LATEST RECORDED CLOSE",
            money(latest_price),
            latest["Date"].strftime("%d %b %Y")
        )

    with c3:
        metric_card(
            "TOTAL PRICE RETURN",
            f"{period_return:+.2f}%",
            f"{period_change:+,.2f} per share"
        )

    if period_return > 0:

        notice(
            "Historical share-price gain",
            f"The share price increased by {period_return:.2f}% "
            "between the first and last available records.",
            "success"
        )

    elif period_return < 0:

        notice(
            "Historical share-price decline",
            f"The share price declined by {abs(period_return):.2f}% "
            "between the first and last available records.",
            "danger"
        )

    else:

        notice(
            "No overall price change",
            "The first and last recorded closing prices are equal.",
            "info"
        )

    st.caption(
        "This calculation measures share-price movement only. It is not "
        "Reliance's corporate net profit or loss. It excludes dividends, "
        "fees, taxes and inflation."
    )

    st.divider()

    section_title("2. What Did Our Models Discover?")

    if not results.empty:

        model_col = find_column(
            results,
            ["model", "model_name"]
        )

        mae_col = find_column(
            results,
            ["mae", "mean_absolute_error"]
        )

        if model_col and mae_col:

            result_df = results.copy()

            result_df[mae_col] = pd.to_numeric(
                result_df[mae_col],
                errors="coerce"
            )

            valid = result_df.dropna(subset=[mae_col])

            if not valid.empty:

                best = valid.loc[valid[mae_col].idxmin()]

                c1, c2 = st.columns(2)

                with c1:
                    metric_card(
                        "LOWEST MAE MODEL",
                        str(best[model_col]),
                        "Best price-error result"
                    )

                with c2:
                    metric_card(
                        "LOWEST MAE",
                        money(best[mae_col]),
                        "Lower is better"
                    )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            notice(
                "Model result columns unavailable",
                "The model name or MAE column could not be identified.",
                "warning"
            )

    else:

        notice(
            "Model comparison unavailable",
            "The model comparison results file could not be loaded.",
            "warning"
        )

    st.divider()

    section_title("3. Findings From Our Completed Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Market-only LSTM",
            "Market + News LSTM",
            "Previous-close baseline"
        ],
        "MAE (₹)": [
            13.9496,
            13.9599,
            13.2310
        ],
        "RMSE (₹)": [
            20.4488,
            21.0131,
            19.6821
        ],
        "MAPE (%)": [
            1.0937,
            1.0923,
            1.0394
        ]
    })

    st.dataframe(
        comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    feature_a, feature_b = st.columns(2)

    with feature_a:
        metric_card(
            "MARKET + NEWS DIRECTIONAL ACCURACY",
            "61.29%",
            "Completed test comparison"
        )

    with feature_b:
        metric_card(
            "MARKET-ONLY DIRECTIONAL ACCURACY",
            "54.84%",
            "Completed test comparison"
        )

    notice(
        "Interpretation",
        "The previous-close baseline achieved the lowest MAE and RMSE. "
        "Adding news features did not improve price-error metrics. "
        "However, the market-plus-news LSTM achieved higher directional "
        "accuracy than the market-only LSTM in this test.",
        "info"
    )

    notice(
        "Important baseline limitation",
        "The baseline's directional accuracy was approximately 0.81% "
        "under the project's existing metric. Because it predicts no "
        "price change, this is not a normal up/down forecasting score.",
        "warning"
    )

    st.divider()

    section_title("4. News Sentiment Findings")

    sentiment_df = news_data.get(
        "FinBERT Sentiment News",
        pd.DataFrame()
    )

    if not sentiment_df.empty:

        sentiment_col = find_column(
            sentiment_df,
            [
                "sentiment",
                "predicted_sentiment",
                "sentiment_label",
                "label"
            ]
        )

        if sentiment_col:

            counts = (
                sentiment_df[sentiment_col]
                .fillna("Unknown")
                .astype(str)
                .str.lower()
                .value_counts()
            )

            sentiment_rows = []

            for label, count in counts.items():

                sentiment_rows.append({
                    "Sentiment": label.title(),
                    "Article Count": int(count),
                    "Share (%)": round(
                        count / len(sentiment_df) * 100,
                        2
                    )
                })

            st.dataframe(
                pd.DataFrame(sentiment_rows),
                use_container_width=True,
                hide_index=True
            )

        else:

            notice(
                "Sentiment field unavailable",
                "No recognized sentiment column was found.",
                "warning"
            )

    else:

        notice(
            "FinBERT data unavailable",
            "Article-level FinBERT sentiment data was not found.",
            "info"
        )

    st.caption(
        "Article sentiment is not a direct measure of company profitability "
        "or proof of price causation."
    )

    st.divider()

    section_title("5. Technical Market Observations")

    ma20 = stock["MA20"].iloc[-1]
    ma50 = stock["MA50"].iloc[-1]
    ma200 = stock["MA200"].iloc[-1]

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "LATEST CLOSE",
            money(latest_price),
            "Most recent available price"
        )

    with c2:
        metric_card(
            "MA20",
            money(ma20),
            "Short-term moving average"
        )

    with c3:
        metric_card(
            "MA50",
            money(ma50),
            "Medium-term moving average"
        )

    if pd.notna(ma20) and pd.notna(ma50):

        if ma20 > ma50:

            notice(
                "Short-term market observation",
                "MA20 is above MA50. Recent price action is stronger than "
                "the medium-term average. This may indicate positive short-term momentum.",
                "success"
            )

        else:

            notice(
                "Short-term market observation",
                "MA20 is below MA50. Recent price action is weaker than "
                "the medium-term average. This may indicate weaker short-term momentum.",
                "warning"
            )

    if pd.notna(ma200):

        if latest_price > ma200:

            notice(
                "Long-term market observation",
                "The latest closing price is above MA200. The share is "
                "currently trading above its 200-session average.",
                "success"
            )

        else:

            notice(
                "Long-term market observation",
                "The latest closing price is below MA200. The share is "
                "currently trading below its 200-session average.",
                "warning"
            )

    else:

        notice(
            "Long-term market observation",
            "There are not enough valid records to calculate MA200.",
            "info"
        )

    st.divider()

    section_title("6. Share Price, Prediction Error and Company Profit")

    info_card(
        "Share Price Gain / Loss",
        "Calculated from the first and last stock closing prices in the "
        "collected dataset. It describes the market performance of one share over the selected period."
    )

    info_card(
        "Prediction Error",
        "MAE, RMSE and MAPE measure how far predicted closing prices differ "
        "from actual closing prices. They measure model accuracy, not financial losses incurred by Reliance."
    )

    info_card(
        "Company Accounting Profit / Loss",
        "To evaluate actual business profitability, we need Reliance's "
        "revenue, operating profit, net profit, expenses, debt, cash flow and quarterly financial statements."
    )

    st.divider()

    section_title("7. Research Recommendations")

    info_card(
        "Recommendation 1 — Improve prediction accuracy",
        "The previous-close baseline currently outperforms both LSTM "
        "models on price-error metrics. Test different sequence lengths, "
        "features and hyperparameters before selecting a forecasting model."
    )

    info_card(
        "Recommendation 2 — Increase historical news coverage",
        "News is currently available for relatively few trading sessions "
        "compared with the complete stock dataset. Collect broader and more consistent historical news."
    )

    info_card(
        "Recommendation 3 — Validate news sentiment",
        "Test market-only and market-plus-news models on several "
        "chronological test periods to check whether sentiment adds consistent predictive value."
    )

    info_card(
        "Recommendation 4 — Add financial fundamentals",
        "Collect revenue, operating profit, net profit, debt, cash flow, "
        "earnings and valuation information to evaluate the company's actual financial performance."
    )

    info_card(
        "Recommendation 5 — Perform realistic strategy backtesting",
        "Test any trading strategy using out-of-sample prices, transaction "
        "costs, slippage, risk controls and drawdown analysis."
    )

    notice(
        "Final research conclusion",
        "Current evidence does not establish a buy or sell recommendation. "
        "The news-based LSTM showed a directional-accuracy improvement, "
        "but the previous-close baseline performed better on price-error "
        "metrics. Further data collection and validation are required.",
        "warning"
    )

# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>RELIANCE STOCK INTELLIGENCE</b>
        <br>
        Historical Market Data | News Sentiment | Machine Learning
        <br><br>
        Built for research and educational purposes.
        <br>
        Not financial advice.
    </div>
    """,
    unsafe_allow_html=True
)