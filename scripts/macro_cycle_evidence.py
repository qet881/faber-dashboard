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
    year = match.groupdict().get("year")
    latest_date = f"{latest_month} {year}" if latest_month and year else None
    return {
        "latest": latest,
        "previous": previous,
        "latest_date": latest_date,
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


def collect_evidence(as_of: pd.Timestamp) -> dict:
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

    price_summaries = []
    prices: dict[str, pd.Series | None] = {}
    yahoo_tickers = {asset["ticker"] for asset in PRICE_ASSETS}
    for _, a, b, _ in ROTATION_PAIRS:
        yahoo_tickers.add(a)
        if b:
            yahoo_tickers.add(b)
    for _, ticker in SECTOR_ETFS:
        yahoo_tickers.add(ticker)

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

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of_requested": as_of.strftime("%Y-%m-%d"),
        "fred": fred,
        "price_assets": price_summaries,
        "rotation": rotation,
        "sector_rotation": sector,
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
    evidence = collect_evidence(as_of)
    json_path, md_path = write_outputs(evidence, Path(args.out_dir))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
