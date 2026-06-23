from __future__ import annotations

import argparse
import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only in stripped envs
    yf = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "docs" / "macro_cycle"
USER_AGENT = "faber-dashboard-macro-cycle/0.1"
TE_ISM_PMI_URL = "https://tradingeconomics.com/united-states/business-confidence"


FRED_SERIES = {
    "leading": [
        {
            "id": "NAPM",
            "name": "ISM Manufacturing PMI",
            "unit": "index",
            "reading": "50 above = expansion; below 50 but rising can signal recovery.",
        },
        {
            "id": "UMCSENT",
            "name": "University of Michigan Consumer Sentiment",
            "unit": "index",
            "reading": "Consumer expectations and sentiment, treated as leading pressure.",
        },
        {
            "id": "T10Y2Y",
            "name": "10Y minus 2Y Treasury spread",
            "unit": "percentage points",
            "reading": "Curve steepening after inversion can mark late slowdown/recession transition.",
        },
        {
            "id": "AMTMNO",
            "name": "Manufacturers' New Orders: Total Manufacturing",
            "unit": "millions USD",
            "reading": "Live FRED proxy for the demand side of the ISM PMI. Use 6M/12M direction; falling new orders lead a manufacturing slowdown.",
        },
        {
            "id": "GACDISA066MSFRBNY",
            "name": "Empire State Mfg General Business Conditions",
            "unit": "diffusion index",
            "reading": "Live regional Fed diffusion index (oscillates around 0). A trend-able PMI proxy: below 0 and falling supports slowdown.",
        },
        {
            "id": "GACDFSA066MSFRBPHI",
            "name": "Philadelphia Fed Mfg General Activity",
            "unit": "diffusion index",
            "reading": "Live regional Fed diffusion index (oscillates around 0). Cross-check with Empire State and PMI direction.",
        },
    ],
    "coincident": [
        {
            "id": "INDPRO",
            "name": "Industrial Production Index",
            "unit": "index",
            "reading": "Use 12-month moving average and YoY direction, not noisy MoM alone.",
        },
        {
            "id": "RSAFS",
            "name": "Advance Retail Sales",
            "unit": "millions USD",
            "reading": "Use YoY/12-month trend as annual retail-sales proxy.",
        },
        {
            "id": "TCU",
            "name": "Capacity Utilization",
            "unit": "percent",
            "reading": "Coincident production-cycle pressure.",
        },
    ],
    "lagging": [
        {
            "id": "GDPC1",
            "name": "Real GDP",
            "unit": "billions chained 2017 USD",
            "reading": "Quarterly GDP, included because annual GDP is too slow for current-cycle reads.",
        },
        {
            "id": "A191RL1A225NBEA",
            "name": "Real GDP annual growth",
            "unit": "percent",
            "reading": "Annual GDP growth, slow but useful as a final confirmation.",
        },
        {
            "id": "UNRATE",
            "name": "Unemployment Rate",
            "unit": "percent",
            "reading": "Low and flat can be late-cycle; rising from lows is slowdown/recession evidence.",
        },
        {
            "id": "CES0500000003",
            "name": "Average Hourly Earnings",
            "unit": "USD/hour",
            "reading": "Lagging wage pressure.",
        },
        {
            "id": "AWHMAN",
            "name": "Average Weekly Hours, Manufacturing",
            "unit": "hours",
            "reading": "Labor-cycle deterioration often appears before layoffs fully show.",
        },
    ],
}


PRICE_ASSETS = [
    {"group": "leading", "ticker": "^GSPC", "name": "S&P 500"},
    {"group": "leading", "ticker": "^IXIC", "name": "NASDAQ Composite"},
    {"group": "leading", "ticker": "^DJI", "name": "Dow Jones Industrial Average"},
    {"group": "leading", "ticker": "^RUT", "name": "Russell 2000"},
]


ROTATION_PAIRS = [
    ("DM vs EM", "VEA", "EEM", "DM stronger supports recovery/slowdown/recession; EM stronger supports growth."),
    ("US vs Non-US", "VTI", "VEA", "US stronger supports recovery/slowdown/recession; Non-US stronger supports growth."),
    ("NASDAQ vs S&P 500", "QQQ", "SPY", "NASDAQ stronger supports recovery/growth; weakness supports slowdown."),
    ("Dow vs S&P 500", "DIA", "SPY", "Dow stronger supports slowdown defensiveness."),
    ("Growth vs Value", "IWF", "IWD", "Growth stronger supports recovery; value/defensive strength supports slowdown."),
    ("Mega-cap vs Small-cap", "OEF", "IWM", "Mega-cap strength supports recession/defensive concentration."),
    ("Dollar index", "DX-Y.NYB", None, "USD strength supports slowdown/recession; weakness supports growth."),
]


SECTOR_ETFS = [
    ("Technology", "XLK"),
    ("Communication Services", "XLC"),
    ("Consumer Discretionary", "XLY"),
    ("Financials", "XLF"),
    ("Industrials", "XLI"),
    ("Health Care", "XLV"),
    ("Real Estate", "XLRE"),
    ("Energy", "XLE"),
    ("Materials", "XLB"),
    ("Consumer Staples", "XLP"),
    ("Utilities", "XLU"),
]


# Local accumulation file so the ISM PMI headline (proprietary, only latest/previous
# available via the public snapshot) builds a real 3M/6M/12M trend over time.
PMI_HISTORY_FILENAME = "pmi_history.csv"


# Data-driven contrarian sentiment layer ("fear/greed"). This is the quantitative
# shadow of the video's psychology cycle: panic/capitulation vs euphoria.
SENTIMENT_PRICE_ASSETS = [
    {"key": "vix", "ticker": "^VIX", "name": "VIX (equity fear gauge)",
     "reading": "High and spiking = fear/capitulation (contrarian buy zone); low and flat = complacency/euphoria near tops."},
]
SENTIMENT_FRED = [
    {
        "id": "BAMLH0A0HYM2",
        "name": "High-Yield OAS credit spread",
        "unit": "percentage points",
        "reading": "Widening = credit stress/capitulation; tight and falling = risk appetite/late-cycle complacency.",
    },
]


# Asset peak-order tracker. The video's idea: money rotates so assets top in
# sequence (equities first, consumer/luxury next, real estate last). We measure
# each asset's drawdown from its own trailing high and months since that high,
# letting the agent infer where each sits relative to its peak.
PEAK_ASSETS = [
    {"kind": "price", "name": "US Equity (S&P 500)", "ticker": "^GSPC",
     "note": "Leads the cycle; usually tops first."},
    {"kind": "price", "name": "Consumer Discretionary (XLY)", "ticker": "XLY",
     "note": "Consumer/luxury spending proxy; tends to top after equities broadly."},
    {"kind": "price", "name": "Real Estate (XLRE)", "ticker": "XLRE",
     "note": "Rate-sensitive; listed real estate proxy."},
    {"kind": "fred", "name": "US Home Prices (Case-Shiller)", "ticker": "CSUSHPINSA",
     "note": "Physical real estate; slowest to turn, usually tops last. Reported with a ~2 month lag."},
]


# Qualitative top-signals that have no reliable data feed (video's bookstore and
# human indicators). These are NOT auto-fetched. They are blank manual-input
# prompts; the agent must treat any answer as soft confirmation only, never as a
# primary cycle judge.
QUALITATIVE_TOP_SIGNALS = [
    ("서점 지표", "베스트셀러가 가치투자/원칙서(침체·바닥 분위기) vs 차트/파동/일목균형표 등 기술적 매매서(과열·고점 분위기) 중 어느 쪽인가?"),
    ("인간 지표", "평소 주식과 무관하던 대중·지인이 갑자기 추격매수에 뛰어드는 강세장 끝물 신호가 보이는가?"),
    ("거래 열기", "주변에서 레버리지·빚투·신규계좌 개설 분위기가 과열인가, 아니면 관심을 끊고 무기력(항복/우울)한가?"),
]


def _as_ts(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2%}"


def _nearest_at_or_before(series: pd.Series, date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if series.empty:
        return None
    eligible = series[series.index <= date]
    if eligible.empty:
        return None
    return eligible.index[-1], float(eligible.iloc[-1])


def _change_since(series: pd.Series, months: int, pct: bool = False) -> float | None:
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(series) < 2:
        return None
    latest_date = series.index[-1]
    latest = float(series.iloc[-1])
    ref = _nearest_at_or_before(series, latest_date - pd.DateOffset(months=months))
    if ref is None:
        return None
    _, previous = ref
    if pct:
        if previous == 0:
            return None
        return latest / previous - 1.0
    return latest - previous


def _slope_last(series: pd.Series, periods: int) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna().tail(periods)
    if len(values) < max(3, min(periods, 6)):
        return None
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values.values.astype(float), 1)[0]
    return float(slope)


def _direction_label(change: float | None, flat_band: float = 0.0) -> str:
    if change is None or not np.isfinite(change):
        return "unknown"
    if change > flat_band:
        return "rising"
    if change < -flat_band:
        return "falling"
    return "flat"


def fetch_fred_series(series_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series | None:
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start_date.date()}&coed={end_date.date()}"
    )
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        response.raise_for_status()
        raw = pd.read_csv(io.StringIO(response.text))
        if raw.shape[1] < 2:
            return None
        dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
        values = pd.to_numeric(raw.iloc[:, 1].replace(".", np.nan), errors="coerce")
        series = pd.Series(values.values, index=dates).dropna()
        series = series[(series.index >= start_date) & (series.index <= end_date)]
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series.astype(float) if not series.empty else None
    except Exception:
        return None


def parse_tradingeconomics_pmi_snapshot(html: str) -> dict | None:
    match = re.search(
        r"Business Confidence in the United States\s+"
        r"(?:increased|decreased|remained unchanged)\s+to\s+"
        r"(?P<latest>[0-9]+(?:\.[0-9]+)?)\s+points in\s+"
        r"(?P<latest_month>[A-Za-z]+)\s+from\s+"
        r"(?P<previous>[0-9]+(?:\.[0-9]+)?)\s+points in\s+"
        r"(?P<previous_month>[A-Za-z]+)\s+of\s+"
        r"(?P<year>\d{4})",
        html,
        flags=re.I,
    )
    if not match:
        match = re.search(
            r"Actual\s+Previous\s+Highest\s+Lowest\s+Dates\s+Unit\s+Frequency\s+"
            r"(?P<latest>[0-9]+(?:\.[0-9]+)?)\s+"
            r"(?P<previous>[0-9]+(?:\.[0-9]+)?)",
            re.sub(r"\s+", " ", html),
            flags=re.I,
        )
    if not match:
        return None

    latest = float(match.group("latest"))
    previous = float(match.group("previous"))
    latest_month = match.groupdict().get("latest_month")
    previous_month = match.groupdict().get("previous_month")
    year = match.groupdict().get("year")
    latest_date = f"{latest_month} {year}" if latest_month and year else None
    previous_date = f"{previous_month} {year}" if previous_month and year else None
    return {
        "latest": latest,
        "previous": previous,
        "latest_date": latest_date,
        "previous_date": previous_date,
        "delta_latest": latest - previous,
    }


def fetch_tradingeconomics_pmi_snapshot() -> dict:
    try:
        response = requests.get(TE_ISM_PMI_URL, headers={"User-Agent": USER_AGENT}, timeout=15)
        response.raise_for_status()
        parsed = parse_tradingeconomics_pmi_snapshot(response.text)
        if not parsed:
            raise ValueError("PMI snapshot not found")
        return {
            "name": "ISM Manufacturing PMI",
            "series_id": "TradingEconomics:business-confidence",
            "status": "ok",
            "latest_date": parsed["latest_date"],
            "latest": parsed["latest"],
            "previous": parsed["previous"],
            "previous_date": parsed.get("previous_date"),
            "delta_3m": None,
            "delta_6m": None,
            "delta_12m": None,
            "slope_6_obs": None,
            "slope_12_obs": None,
            "direction_6m": _direction_label(parsed["delta_latest"]),
            "note": (
                "Trading Economics latest/previous snapshot fallback because the public FRED PMI "
                f"series was unavailable. Source: {TE_ISM_PMI_URL}"
            ),
        }
    except Exception:
        return {
            "name": "ISM Manufacturing PMI",
            "series_id": "TradingEconomics:business-confidence",
            "status": "missing",
            "latest_date": None,
            "latest": None,
            "delta_3m": None,
            "delta_6m": None,
            "delta_12m": None,
            "slope_6_obs": None,
            "slope_12_obs": None,
            "direction_6m": "unknown",
            "note": (
                "PMI needs manual/paid-source review if both FRED and Trading Economics fallback fail. "
                f"Candidate source: {TE_ISM_PMI_URL}"
            ),
        }


def fetch_yahoo_close(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series | None:
    if yf is None:
        return None
    try:
        raw = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            timeout=20,
        )
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        close = pd.to_numeric(raw[col], errors="coerce").dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close[~close.index.duplicated(keep="last")].sort_index()
        return close.astype(float) if not close.empty else None
    except Exception:
        return None


def summarize_level_series(meta: dict, series: pd.Series | None) -> dict:
    if series is None or series.empty:
        return {
            "name": meta["name"],
            "series_id": meta["id"],
            "status": "missing",
            "latest_date": None,
            "latest": None,
            "delta_3m": None,
            "delta_6m": None,
            "delta_12m": None,
            "slope_6_obs": None,
            "slope_12_obs": None,
            "direction_6m": "unknown",
            "note": meta["reading"],
        }
    series = series.sort_index()
    delta_6m = _change_since(series, 6, pct=False)
    return {
        "name": meta["name"],
        "series_id": meta["id"],
        "status": "ok",
        "latest_date": series.index[-1].strftime("%Y-%m-%d"),
        "latest": float(series.iloc[-1]),
        "delta_3m": _change_since(series, 3, pct=False),
        "delta_6m": delta_6m,
        "delta_12m": _change_since(series, 12, pct=False),
        "slope_6_obs": _slope_last(series, 6),
        "slope_12_obs": _slope_last(series, 12),
        "direction_6m": _direction_label(delta_6m),
        "note": meta["reading"],
    }


def summarize_price_asset(asset: dict, series: pd.Series | None) -> dict:
    if series is None or series.empty:
        return {
            "name": asset["name"],
            "ticker": asset["ticker"],
            "status": "missing",
            "latest_date": None,
            "latest": None,
            "return_3m": None,
            "return_6m": None,
            "return_12m": None,
            "return_5y": None,
            "above_200d_ma": None,
            "direction_12m": "unknown",
        }
    series = series.sort_index()
    ma200 = series.rolling(200).mean().iloc[-1] if len(series) >= 200 else np.nan
    ret_12m = _change_since(series, 12, pct=True)
    return {
        "name": asset["name"],
        "ticker": asset["ticker"],
        "status": "ok",
        "latest_date": series.index[-1].strftime("%Y-%m-%d"),
        "latest": float(series.iloc[-1]),
        "return_3m": _change_since(series, 3, pct=True),
        "return_6m": _change_since(series, 6, pct=True),
        "return_12m": ret_12m,
        "return_5y": _change_since(series, 60, pct=True),
        "above_200d_ma": bool(series.iloc[-1] > ma200) if np.isfinite(ma200) else None,
        "direction_12m": _direction_label(ret_12m),
    }


def summarize_relative_pair(name: str, a: str, b: str | None, note: str, prices: dict[str, pd.Series | None]) -> dict:
    if b is None:
        series = prices.get(a)
        summary = summarize_price_asset({"name": name, "ticker": a}, series)
        summary["pair"] = name
        summary["interpretation"] = note
        return summary

    a_series = prices.get(a)
    b_series = prices.get(b)
    if a_series is None or b_series is None or a_series.empty or b_series.empty:
        return {
            "pair": name,
            "ticker": f"{a}/{b}",
            "status": "missing",
            "latest_date": None,
            "return_3m": None,
            "return_6m": None,
            "return_12m": None,
            "direction_6m": "unknown",
            "interpretation": note,
        }
    aligned = pd.concat([a_series.rename("a"), b_series.rename("b")], axis=1).dropna()
    ratio = (aligned["a"] / aligned["b"]).dropna()
    ret_6m = _change_since(ratio, 6, pct=True)
    return {
        "pair": name,
        "ticker": f"{a}/{b}",
        "status": "ok",
        "latest_date": ratio.index[-1].strftime("%Y-%m-%d"),
        "return_3m": _change_since(ratio, 3, pct=True),
        "return_6m": ret_6m,
        "return_12m": _change_since(ratio, 12, pct=True),
        "direction_6m": _direction_label(ret_6m),
        "interpretation": note,
    }


def _month_str_to_ts(month_year: str | None) -> pd.Timestamp | None:
    if not month_year:
        return None
    try:
        return pd.Timestamp(pd.to_datetime(month_year)).normalize().replace(day=1)
    except Exception:
        return None


def _percentile_rank(series: pd.Series, value: float | None) -> float | None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty or value is None or not np.isfinite(value):
        return None
    return float((series <= value).mean())


def _drawdown_stats(series: pd.Series | None) -> dict | None:
    """How far the latest value sits below its own trailing high, and how long ago that high was."""
    if series is None:
        return None
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if series.empty:
        return None
    latest = float(series.iloc[-1])
    peak_value = float(series.max())
    peak_date = series.idxmax()
    drawdown = latest / peak_value - 1.0 if peak_value else None
    months_since_peak = None
    if isinstance(peak_date, pd.Timestamp):
        delta_days = (series.index[-1] - peak_date).days
        months_since_peak = round(delta_days / 30.44, 1)
    return {
        "latest": latest,
        "peak_value": peak_value,
        "peak_date": peak_date.strftime("%Y-%m-%d") if isinstance(peak_date, pd.Timestamp) else None,
        "drawdown": drawdown,
        "months_since_peak": months_since_peak,
        "at_high": bool(months_since_peak is not None and months_since_peak <= 1.0),
    }


def merge_pmi_history(existing: pd.Series | None, updates: dict[pd.Timestamp, float]) -> pd.Series:
    """Upsert monthly PMI observations into the accumulated history (pure)."""
    series = existing.copy() if existing is not None else pd.Series(dtype=float)
    for date, value in updates.items():
        if date is None or value is None or not np.isfinite(value):
            continue
        series.loc[pd.Timestamp(date).normalize()] = float(value)
    if series.empty:
        return series
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.astype(float)


def load_pmi_history(out_dir: Path) -> pd.Series | None:
    path = out_dir / PMI_HISTORY_FILENAME
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path)
        dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
        values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
        series = pd.Series(values.values, index=dates).dropna()
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series.astype(float) if not series.empty else None
    except Exception:
        return None


def save_pmi_history(out_dir: Path, series: pd.Series) -> None:
    if series is None or series.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"date": series.index.strftime("%Y-%m-%d"), "pmi": series.values})
    frame.to_csv(out_dir / PMI_HISTORY_FILENAME, index=False)


def apply_pmi_history(summary: dict, history: pd.Series) -> dict:
    """Overlay 3M/6M/12M deltas and slopes computed from accumulated PMI history."""
    if history is None or history.empty:
        return summary
    summary = dict(summary)
    summary["delta_3m"] = _change_since(history, 3, pct=False)
    summary["delta_6m"] = _change_since(history, 6, pct=False)
    summary["delta_12m"] = _change_since(history, 12, pct=False)
    summary["slope_6_obs"] = _slope_last(history, 6)
    summary["slope_12_obs"] = _slope_last(history, 12)
    six = summary["delta_6m"]
    if six is not None and np.isfinite(six):
        summary["direction_6m"] = _direction_label(six)
    summary["history_points"] = int(len(history))
    note = summary.get("note", "")
    if "Accumulated locally" not in note:
        summary["note"] = (note + f" Accumulated locally ({len(history)} months) in "
                           f"{PMI_HISTORY_FILENAME}; trend grows as more runs are stored.").strip()
    return summary


def summarize_sentiment_price(asset: dict, series: pd.Series | None) -> dict:
    if series is None or series.empty:
        return {
            "name": asset["name"], "ticker": asset["ticker"], "status": "missing",
            "latest_date": None, "latest": None, "change_3m": None,
            "percentile_1y": None, "direction_3m": "unknown", "note": asset["reading"],
        }
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    latest = float(series.iloc[-1])
    last_1y = series[series.index >= series.index[-1] - pd.DateOffset(years=1)]
    change_3m = _change_since(series, 3, pct=True)
    return {
        "name": asset["name"], "ticker": asset["ticker"], "status": "ok",
        "latest_date": series.index[-1].strftime("%Y-%m-%d"), "latest": latest,
        "change_3m": change_3m,
        "percentile_1y": _percentile_rank(last_1y, latest),
        "direction_3m": _direction_label(change_3m),
        "note": asset["reading"],
    }


def summarize_peak_asset(asset: dict, series: pd.Series | None) -> dict:
    stats = _drawdown_stats(series)
    if stats is None:
        return {
            "name": asset["name"], "ticker": asset["ticker"], "status": "missing",
            "latest": None, "peak_date": None, "drawdown": None,
            "months_since_peak": None, "at_high": None, "note": asset["note"],
        }
    return {
        "name": asset["name"], "ticker": asset["ticker"], "status": "ok",
        "latest": stats["latest"], "peak_date": stats["peak_date"],
        "drawdown": stats["drawdown"], "months_since_peak": stats["months_since_peak"],
        "at_high": stats["at_high"], "note": asset["note"],
    }


def collect_evidence(as_of: pd.Timestamp, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    start_10y = as_of - pd.DateOffset(years=10)
    start_5y = as_of - pd.DateOffset(years=5)
    start_3y = as_of - pd.DateOffset(years=3)

    fred = {}
    for group, metas in FRED_SERIES.items():
        fred[group] = []
        for meta in metas:
            start = start_3y if meta["id"] in {"NAPM", "UMCSENT", "UNRATE"} else start_5y
            if meta["id"] in {"GDPC1", "A191RL1A225NBEA"}:
                start = start_10y
            summary = summarize_level_series(meta, fetch_fred_series(meta["id"], start, as_of))
            if meta["id"] == "NAPM" and summary["status"] == "missing":
                summary = fetch_tradingeconomics_pmi_snapshot()
            fred[group].append(summary)

    # Persist the headline PMI so 3M/6M/12M trend rebuilds over repeated runs,
    # then overlay the accumulated-history deltas back onto the summary.
    for i, summary in enumerate(fred["leading"]):
        if summary.get("name") != "ISM Manufacturing PMI" or summary.get("status") != "ok":
            continue
        updates: dict[pd.Timestamp, float] = {}
        latest_ts = _month_str_to_ts(summary.get("latest_date"))
        if latest_ts is not None and summary.get("latest") is not None:
            updates[latest_ts] = float(summary["latest"])
        prev_ts = _month_str_to_ts(summary.get("previous_date"))
        if prev_ts is not None and summary.get("previous") is not None:
            updates[prev_ts] = float(summary["previous"])
        history = merge_pmi_history(load_pmi_history(out_dir), updates)
        save_pmi_history(out_dir, history)
        fred["leading"][i] = apply_pmi_history(summary, history)
        break

    price_summaries = []
    prices: dict[str, pd.Series | None] = {}
    yahoo_tickers = {asset["ticker"] for asset in PRICE_ASSETS}
    for _, a, b, _ in ROTATION_PAIRS:
        yahoo_tickers.add(a)
        if b:
            yahoo_tickers.add(b)
    for _, ticker in SECTOR_ETFS:
        yahoo_tickers.add(ticker)
    for asset in SENTIMENT_PRICE_ASSETS:
        yahoo_tickers.add(asset["ticker"])
    for asset in PEAK_ASSETS:
        if asset["kind"] == "price":
            yahoo_tickers.add(asset["ticker"])

    for ticker in sorted(yahoo_tickers):
        prices[ticker] = fetch_yahoo_close(ticker, start_10y, as_of)

    for asset in PRICE_ASSETS:
        price_summaries.append(summarize_price_asset(asset, prices.get(asset["ticker"])))

    rotation = [
        summarize_relative_pair(name, a, b, note, prices)
        for name, a, b, note in ROTATION_PAIRS
    ]

    sector = []
    spy = prices.get("SPY")
    for name, ticker in SECTOR_ETFS:
        sector.append(summarize_relative_pair(f"{name} vs S&P 500", ticker, "SPY", "Sector relative strength versus broad US equities.", prices))
        if spy is None:
            sector[-1]["status"] = "missing"

    # Data-driven sentiment / fear-greed layer.
    sentiment = [
        summarize_sentiment_price(asset, prices.get(asset["ticker"]))
        for asset in SENTIMENT_PRICE_ASSETS
    ]
    for meta in SENTIMENT_FRED:
        sentiment.append(summarize_level_series(meta, fetch_fred_series(meta["id"], start_5y, as_of)))
    spx = prices.get("^GSPC")
    spx_dd = _drawdown_stats(spx)
    sentiment.append({
        "name": "S&P 500 drawdown from trailing high", "ticker": "^GSPC",
        "status": "ok" if spx_dd else "missing",
        "latest_date": spx.index[-1].strftime("%Y-%m-%d") if spx is not None and not spx.empty else None,
        "latest": spx_dd["drawdown"] if spx_dd else None,
        "change_3m": None, "percentile_1y": None,
        "direction_3m": "unknown",
        "note": "Deep drawdown = capitulation/contrarian-buy territory; near 0% = at highs, late-cycle complacency.",
    })

    # Asset peak-order tracker.
    asset_peaks = []
    for asset in PEAK_ASSETS:
        if asset["kind"] == "price":
            series = prices.get(asset["ticker"])
        else:
            series = fetch_fred_series(asset["ticker"], start_10y, as_of)
        asset_peaks.append(summarize_peak_asset(asset, series))

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of_requested": as_of.strftime("%Y-%m-%d"),
        "fred": fred,
        "price_assets": price_summaries,
        "rotation": rotation,
        "sector_rotation": sector,
        "sentiment": sentiment,
        "asset_peaks": asset_peaks,
        "qualitative_top_signals": [
            {"name": name, "prompt": prompt, "reading": "(수동 입력 — 비워두면 미사용)"}
            for name, prompt in QUALITATIVE_TOP_SIGNALS
        ],
        "method": {
            "purpose": "Evidence pack only. GPT-5.5/Codex should make the final cycle judgment.",
            "no_score_policy": "Do not collapse the evidence into a deterministic score before reasoning.",
            "portfolio_policy": "Portfolio guidance is a macro overlay/satellite guide, not a replacement for Haenam P rules.",
        },
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(evidence: dict) -> str:
    lines = [
        "# Macro Cycle Evidence Pack",
        "",
        f"- Generated: {evidence['generated_at']}",
        f"- Requested as-of date: {evidence['as_of_requested']}",
        "- Role: evidence package for GPT-5.5/Codex judgment, not a deterministic scorecard.",
        "",
        "## Judgment Instruction",
        "",
        "Use the evidence below to judge the US macro cycle directly. Do not score mechanically. "
        "Classify the economy as Recovery, Growth, Slowdown, or Recession. If evidence is transitional, "
        "name the transition path and explain why one side dominates. Mention conflicting evidence.",
        "",
        "Portfolio output is allowed, but it must be framed as a macro satellite/overlay guide. "
        "It must not override Haenam P's trend-following/rebalancing rules unless the user explicitly asks for a separate strategy.",
        "",
        "## Leading Indicators",
        "",
    ]

    price_rows = []
    for item in evidence["price_assets"]:
        price_rows.append(
            [
                item["name"],
                item["ticker"],
                item["status"],
                item.get("latest_date") or "n/a",
                _format_float(item.get("latest")),
                _format_pct(item.get("return_3m")),
                _format_pct(item.get("return_6m")),
                _format_pct(item.get("return_12m")),
                _format_pct(item.get("return_5y")),
                str(item.get("above_200d_ma")),
            ]
        )
    lines.append(
        _markdown_table(
            ["Asset", "Ticker", "Status", "Latest date", "Latest", "3M", "6M", "12M", "5Y", ">200D MA"],
            price_rows,
        )
    )

    for group_label, key in [("Leading Macro", "leading"), ("Coincident Indicators", "coincident"), ("Lagging Indicators", "lagging")]:
        lines.extend(["", f"## {group_label}", ""])
        rows = []
        for item in evidence["fred"][key]:
            rows.append(
                [
                    item["name"],
                    item["series_id"],
                    item["status"],
                    item.get("latest_date") or "n/a",
                    _format_float(item.get("latest")),
                    _format_float(item.get("delta_3m")),
                    _format_float(item.get("delta_6m")),
                    _format_float(item.get("delta_12m")),
                    item.get("direction_6m", "unknown"),
                    item["note"],
                ]
            )
        lines.append(
            _markdown_table(
                ["Indicator", "FRED", "Status", "Latest date", "Latest", "3M delta", "6M delta", "12M delta", "6M dir", "Reading note"],
                rows,
            )
        )

    lines.extend(["", "## Market Rotation", ""])
    rotation_rows = []
    for item in evidence["rotation"]:
        rotation_rows.append(
            [
                item["pair"],
                item["ticker"],
                item["status"],
                item.get("latest_date") or "n/a",
                _format_pct(item.get("return_3m")),
                _format_pct(item.get("return_6m")),
                _format_pct(item.get("return_12m")),
                item.get("direction_6m", "unknown"),
                item["interpretation"],
            ]
        )
    lines.append(
        _markdown_table(
            ["Rotation", "Ticker", "Status", "Latest date", "3M", "6M", "12M", "6M dir", "Interpretation"],
            rotation_rows,
        )
    )

    lines.extend(["", "## Sector Rotation", ""])
    sector_rows = []
    for item in evidence["sector_rotation"]:
        sector_rows.append(
            [
                item["pair"],
                item["ticker"],
                item["status"],
                item.get("latest_date") or "n/a",
                _format_pct(item.get("return_3m")),
                _format_pct(item.get("return_6m")),
                _format_pct(item.get("return_12m")),
                item.get("direction_6m", "unknown"),
            ]
        )
    lines.append(
        _markdown_table(
            ["Sector", "Ticker", "Status", "Latest date", "3M", "6M", "12M", "6M dir"],
            sector_rows,
        )
    )

    sentiment_items = evidence.get("sentiment") or []
    if sentiment_items:
        lines.extend([
            "",
            "## Investor Sentiment (Fear / Greed)",
            "",
            "Contrarian layer: the quantitative shadow of the psychology cycle. Fear/capitulation "
            "extremes are contrarian-bullish; complacency/greed extremes are late-cycle warnings. "
            "Cross-validation only, not the primary cycle judge.",
            "",
        ])
        sent_rows = []
        for item in sentiment_items:
            sent_rows.append([
                item["name"],
                item.get("ticker") or item.get("series_id") or "n/a",
                item["status"],
                item.get("latest_date") or "n/a",
                _format_pct(item["latest"]) if "drawdown" in item.get("name", "") else _format_float(item.get("latest")),
                _format_pct(item.get("change_3m")) if item.get("change_3m") is not None else _format_float(item.get("delta_3m")),
                _format_pct(item["percentile_1y"]) if item.get("percentile_1y") is not None else "n/a",
                item.get("direction_3m") or item.get("direction_6m", "unknown"),
                item.get("note", ""),
            ])
        lines.append(_markdown_table(
            ["Signal", "Ticker", "Status", "Latest date", "Latest", "3M", "1Y %ile", "Dir", "Reading note"],
            sent_rows,
        ))

    peak_items = evidence.get("asset_peaks") or []
    if peak_items:
        lines.extend([
            "",
            "## Asset Peak Order",
            "",
            "Money rotates, so assets tend to top in sequence (equities first, consumer/luxury next, "
            "real estate last). Each row shows how far the asset sits below its own trailing high and "
            "how long ago that high was. Use the sequence to gauge how late the cycle is.",
            "",
        ])
        peak_rows = []
        for item in peak_items:
            peak_rows.append([
                item["name"],
                item.get("ticker", "n/a"),
                item["status"],
                item.get("peak_date") or "n/a",
                _format_pct(item.get("drawdown")),
                "n/a" if item.get("months_since_peak") is None else f"{item['months_since_peak']:.1f}",
                str(item.get("at_high")),
                item.get("note", ""),
            ])
        lines.append(_markdown_table(
            ["Asset", "Ticker", "Status", "Own peak date", "From peak", "Months since peak", "At high", "Note"],
            peak_rows,
        ))

    qual_items = evidence.get("qualitative_top_signals") or []
    if qual_items:
        lines.extend([
            "",
            "## Qualitative Top-Signals (manual input)",
            "",
            "No reliable data feed exists for these (the video's bookstore and human indicators). "
            "Leave blank if unknown. The agent must treat any answer as **soft confirmation only**, "
            "never as a primary cycle judge.",
            "",
        ])
        qual_rows = [[item["name"], item["prompt"], item.get("reading", "(수동 입력)")] for item in qual_items]
        lines.append(_markdown_table(["Signal", "Prompt", "Current reading"], qual_rows))

    lines.extend(
        [
            "",
            "## Required Final Output Shape",
            "",
            "**[경기 국면 최종 진단]**",
            "* 현재 국면: [회복 / 성장 / 둔화 / 침체]",
            "* 진단 신뢰도: [높음 / 보통 / 낮음] (이유 요약)",
            "* 과도기 여부: [아님 / 회복->성장 / 성장->둔화 / 둔화->침체 / 침체->회복]",
            "* 국면 위치: [초입 / 초중반 / 중반 / 후반 / 말기] (대략 개월 범위와 근거)",
            "",
            "**[경제 지표 분석 근거]**",
            "* 선행 지표 추세: ...",
            "* 동행 지표 추세: ...",
            "* 후행 지표 추세: ...",
            "",
            "**[시장 로테이션 교차 검증]**",
            "* 현재 시장 강세 자산/지수/업종과 경기국면 진단의 일치 여부",
            "",
            "**[투자자 심리 · 고점 신호]**",
            "* 공포/탐욕(VIX·신용스프레드·낙폭): [항복/공포 / 중립 / 탐욕/과열] -> 역발상 해석",
            "* 자산 고점 순서(주식->소비재->부동산): 각 자산의 고점대비 위치로 본 사이클 성숙도",
            "* 정성 신호(서점/인간 지표): 수동 입력값이 있으면 보조 확인용으로만 언급, 없으면 '입력 없음'",
            "",
            "**[포트폴리오 대응 지침]**",
            "* 해남P와의 관계: [독립/위성/오버레이/적용 안 함]",
            "* 위험자산 포지션 행동: [선호 / 수익실현 / 분할 축소 / 방어 완료]",
            "* 100% 기준 매크로 포트폴리오: [주식/채권/현금/금/대체 또는 기타 합산 100%]",
            "* 주식 내부 100% 배분: [지역/스타일/섹터 합산 100%]",
            "* 추천 탑픽 섹터/스타일: ...",
            "* 금지: 개인 맞춤 확정 매수·매도 지시처럼 쓰지 말 것. 국면 기반 자산배분 가이드로 제한.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(evidence: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latest_evidence.json"
    md_path = out_dir / "latest_evidence.md"
    json_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(evidence), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evidence pack for GPT-5.5 macro-cycle judgment.")
    parser.add_argument("--as-of", default=datetime.now().strftime("%Y-%m-%d"), help="Requested as-of date, YYYY-MM-DD.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    args = parser.parse_args()

    as_of = _as_ts(args.as_of)
    out_dir = Path(args.out_dir)
    evidence = collect_evidence(as_of, out_dir)
    json_path, md_path = write_outputs(evidence, out_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
