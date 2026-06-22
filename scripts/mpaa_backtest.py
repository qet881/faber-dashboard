from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_INPUT = Path.home() / "Downloads" / "MPAA 데이터.xlsx"
DEFAULT_OUT_DIR = Path("docs/mpaa_backtest")
SOURCE_NOTEBOOK_URL = "https://github.com/hermian/startetf/blob/main/3.33_MPAA_bt.ipynb"

ASSET_GROUPS = ("국가", "섹터", "팩터", "채권")
GROUP_RANKS = {"국가": 4, "섹터": 8, "팩터": 10, "채권": 1}
GROUP_WEIGHTS = {"국가": 1 / 6, "섹터": 1 / 6, "팩터": 1 / 6, "채권": 3 / 6}
CASH_GROUP = "현금"
CASH_ASSET = "현금"
INITIAL_NAV = 100.0

TICKERS = {
    "국가": [
        "S&P 500",
        "러셀 3000",
        "니케이 225",
        "항셍",
        "홍콩 H",
        "대만 가권",
        "상해종합",
        "영국 FTSE 100",
        "프랑스 CAC 40",
        "독일 DAX",
        "KOSPI",
        "인도 SENSEX",
    ],
    "섹터": [
        "에너지",
        "화학",
        "금속및광물",
        "기타 소재",
        "건설",
        "기타자본재",
        "상업서비스",
        "운송",
        "자동차및부품",
        "내구소비재및의류",
        "소비자서비스",
        "미디어",
        "유통",
        "음식료및담배",
        "생활용품",
        "의료",
        "은행",
        "보험",
        "증권",
        "기타금융",
        "소프트웨어",
        "하드웨어",
        "반도체",
        "디스플레이",
        "통신서비스",
        "유틸리티",
    ],
    "팩터": [
        "블루칩30",
        "모멘텀",
        "경기방어주",
        "고배당주",
        "베타플러스",
        "Low Vol",
        "배당성장",
        "Big Vol 지수",
        "FnGuide 컨트래리안",
        "FnGuide 퀄리티밸류 지수",
        "대형가치",
        "대형성장",
        "대형순수가치",
        "대형순수성장",
        "중형가치",
        "중형성장",
        "중형순수가치",
        "중형순수성장",
        "소형가치",
        "소형성장",
        "소형순수가치",
        "소형순수성장",
        "중대형가치",
        "중대형성장",
        "중대형순수가치",
        "중대형순수성장",
        "중소형가치",
        "중소형순수가치",
        "중소형성장",
        "중소형순수성장",
    ],
    "채권": ["채권", "20년채권", "채권인버스"],
    "현금": ["현금"],
}

# Public proxies are intentionally explicit. Several original FnGuide/WiseIndex
# sector/factor series do not have free one-to-one public tickers.
PUBLIC_PROXY_TICKERS = {
    "S&P 500": "^GSPC",
    "러셀 3000": "^RUA",
    "니케이 225": "^N225",
    "항셍": "^HSI",
    "홍콩 H": "^HSCE",
    "대만 가권": "^TWII",
    "상해종합": "000001.SS",
    "영국 FTSE 100": "^FTSE",
    "프랑스 CAC 40": "^FCHI",
    "독일 DAX": "^GDAXI",
    "KOSPI": "^KS11",
    "인도 SENSEX": "^BSESN",
    "에너지": "XLE",
    "화학": "XLB",
    "금속및광물": "XME",
    "기타 소재": "XLB",
    "건설": "ITB",
    "기타자본재": "XLI",
    "상업서비스": "XLI",
    "운송": "IYT",
    "자동차및부품": "CARZ",
    "내구소비재및의류": "XLY",
    "소비자서비스": "XLY",
    "미디어": "VOX",
    "유통": "XRT",
    "음식료및담배": "XLP",
    "생활용품": "XLP",
    "의료": "XLV",
    "은행": "KBE",
    "보험": "KIE",
    "증권": "IAI",
    "기타금융": "XLF",
    "소프트웨어": "IGV",
    "하드웨어": "VGT",
    "반도체": "SOXX",
    "디스플레이": "VGT",
    "통신서비스": "IYZ",
    "유틸리티": "XLU",
    "블루칩30": "DIA",
    "모멘텀": "MTUM",
    "경기방어주": "USMV",
    "고배당주": "VYM",
    "베타플러스": "SPHB",
    "Low Vol": "SPLV",
    "배당성장": "VIG",
    "Big Vol 지수": "IWM",
    "FnGuide 컨트래리안": "VTV",
    "FnGuide 퀄리티밸류 지수": "QUAL",
    "대형가치": "VTV",
    "대형성장": "VUG",
    "대형순수가치": "RPV",
    "대형순수성장": "RPG",
    "중형가치": "IWS",
    "중형성장": "IWP",
    "중형순수가치": "IJJ",
    "중형순수성장": "IJK",
    "소형가치": "IWN",
    "소형성장": "IWO",
    "소형순수가치": "IJS",
    "소형순수성장": "IJT",
    "중대형가치": "IUSV",
    "중대형성장": "IUSG",
    "중대형순수가치": "RPV",
    "중대형순수성장": "RPG",
    "중소형가치": "IWN",
    "중소형순수가치": "IJS",
    "중소형성장": "IWO",
    "중소형순수성장": "IJT",
    "채권": "IEF",
    "20년채권": "TLT",
    "채권인버스": "TBF",
    "현금": "BIL",
}


@dataclass(frozen=True)
class BacktestResult:
    prices: dict[str, pd.DataFrame]
    appended_prices: pd.DataFrame
    source_status: pd.DataFrame
    base_nav: pd.DataFrame
    final_nav: pd.DataFrame
    weights: pd.DataFrame
    summary: pd.DataFrame


def month_last(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    if clean.empty:
        return clean
    monthly = clean.groupby(clean.index.to_period("M")).tail(1)
    monthly.index = monthly.index.to_period("M").to_timestamp()
    return monthly[~monthly.index.duplicated(keep="last")]


def normalize_monthly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp()
    return out.groupby(out.index).last().sort_index()


def read_mpaa_workbook(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    out: dict[str, pd.DataFrame] = {}
    for sheet, columns in TICKERS.items():
        df = pd.read_excel(path, sheet_name=sheet)
        if "날짜" not in df.columns:
            raise ValueError(f"Sheet {sheet!r} must contain a 날짜 column.")
        df["날짜"] = pd.to_datetime(df["날짜"])
        df = df.set_index("날짜").sort_index()
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Sheet {sheet!r} is missing columns: {missing}")
        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
        out[sheet] = numeric[~numeric.index.duplicated(keep="last")]
    return out


def _extract_downloaded_series(raw: pd.DataFrame, symbol: str) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float, name=symbol)
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol in raw.columns.get_level_values(0):
            frame = raw[symbol]
        else:
            field_level = raw.columns.get_level_values(0)
            field = "Adj Close" if "Adj Close" in field_level else "Close"
            selected = raw[field]
            series = selected.iloc[:, 0] if isinstance(selected, pd.DataFrame) else selected
            return pd.to_numeric(series, errors="coerce").rename(symbol).dropna()
    else:
        frame = raw
    col = "Adj Close" if "Adj Close" in frame.columns else "Close"
    if col not in frame.columns:
        return pd.Series(dtype=float, name=symbol)
    return pd.to_numeric(frame[col], errors="coerce").rename(symbol).dropna()


def download_proxy_prices(symbols: list[str], start: str, end: str | None = None) -> dict[str, pd.Series]:
    unique_symbols = sorted(set(symbols))
    raw = yf.download(
        unique_symbols,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return {symbol: _extract_downloaded_series(raw, symbol) for symbol in unique_symbols}


def append_public_proxy_data(
    original: dict[str, pd.DataFrame],
    end: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    all_symbols = [PUBLIC_PROXY_TICKERS[column] for columns in TICKERS.values() for column in columns]
    first_last_date = min(frame.index.max() for frame in original.values())
    start = (first_last_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    downloaded = download_proxy_prices(all_symbols, start=start, end=end)
    status_rows: list[dict[str, Any]] = []
    appended: dict[str, pd.DataFrame] = {}

    for sheet, frame in original.items():
        combined_columns: dict[str, pd.Series] = {}
        last_date = frame.index.max()
        for column in TICKERS[sheet]:
            symbol = PUBLIC_PROXY_TICKERS[column]
            proxy = downloaded.get(symbol, pd.Series(dtype=float))
            original_series = frame[column].dropna()
            monthly_proxy = month_last(proxy)
            base_proxy = proxy.loc[:last_date].dropna()
            status = "ok"
            notes = ""
            if original_series.empty:
                status = "missing_original"
                combined_columns[column] = frame[column]
            elif monthly_proxy.empty:
                status = "missing_proxy"
                combined_columns[column] = frame[column]
            elif base_proxy.empty:
                base_value = monthly_proxy.iloc[0]
                extension = monthly_proxy[monthly_proxy.index > monthly_proxy.index[0]]
                status = "proxy_starts_after_original"
                notes = "Extension starts after proxy inception; gap remains before first proxy month."
                scaled = extension / base_value * original_series.iloc[-1]
                combined_columns[column] = pd.concat([frame[column], scaled]).sort_index()
            else:
                base_value = base_proxy.iloc[-1]
                extension = monthly_proxy[monthly_proxy.index > last_date]
                scaled = extension / base_value * original_series.iloc[-1]
                combined_columns[column] = pd.concat([frame[column], scaled]).sort_index()
            added_count = max(0, len(combined_columns[column].dropna()) - len(original_series))
            proxy_start = proxy.index.min().strftime("%Y-%m-%d") if not proxy.empty else ""
            proxy_end = proxy.index.max().strftime("%Y-%m-%d") if not proxy.empty else ""
            status_rows.append(
                {
                    "sheet": sheet,
                    "asset": column,
                    "proxy_symbol": symbol,
                    "status": status,
                    "original_start": original_series.index.min().strftime("%Y-%m-%d") if not original_series.empty else "",
                    "original_end": original_series.index.max().strftime("%Y-%m-%d") if not original_series.empty else "",
                    "proxy_start": proxy_start,
                    "proxy_end": proxy_end,
                    "rows_added": int(added_count),
                    "notes": notes,
                }
            )
        appended[sheet] = normalize_monthly_frame(pd.DataFrame(combined_columns).sort_index())
    return appended, pd.DataFrame(status_rows)


def average_momentum(prices: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    total = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for month in range(1, lookback + 1):
        total = total + prices / prices.shift(month)
    return total / lookback


def average_momentum_score(prices: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    total = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for month in range(1, lookback + 1):
        total = total + (prices / prices.shift(month) > 1.0).astype(float)
    return total / lookback


def calculate_mpaa_weights(
    prices: dict[str, pd.DataFrame],
    cash_weight: float = 1.0,
    lookback: int = 12,
) -> pd.DataFrame:
    common_index = prices[CASH_GROUP].index
    for group in ASSET_GROUPS:
        common_index = common_index.intersection(prices[group].index)
    common_index = common_index.sort_values()

    momentum = {group: average_momentum(prices[group].reindex(common_index), lookback) for group in ASSET_GROUPS}
    scores = {group: average_momentum_score(prices[group].reindex(common_index), lookback) for group in ASSET_GROUPS}
    cash_scores = average_momentum_score(prices[CASH_GROUP].reindex(common_index), lookback)[CASH_ASSET] * cash_weight
    all_assets = [asset for group in ASSET_GROUPS for asset in TICKERS[group]] + [CASH_ASSET]
    weight_rows: list[pd.Series] = []

    for row_number in range(1, len(common_index)):
        date = common_index[row_number]
        signal_date = common_index[row_number - 1]
        weights = pd.Series(0.0, index=all_assets, dtype=float)
        cash_score = cash_scores.loc[signal_date]
        if not np.isfinite(cash_score):
            cash_score = 0.0

        for group in ASSET_GROUPS:
            group_weight = GROUP_WEIGHTS[group]
            rank_count = GROUP_RANKS[group]
            group_momentum = momentum[group].loc[signal_date].dropna()
            if group_momentum.empty:
                continue
            selected = list(group_momentum.nlargest(rank_count).index)
            selected_count = len(selected)
            if selected_count == 0:
                continue
            group_scores = scores[group].loc[signal_date]
            for asset in selected:
                asset_score = group_scores.get(asset, np.nan)
                asset_score = float(asset_score) if np.isfinite(asset_score) else 0.0
                denominator = asset_score + cash_score
                if denominator <= 0:
                    risky_share = 0.0
                    cash_share = 1.0
                else:
                    risky_share = asset_score / denominator
                    cash_share = cash_score / denominator
                sleeve_weight = group_weight / selected_count
                weights[asset] += risky_share * sleeve_weight
                weights[CASH_ASSET] += cash_share * sleeve_weight
        weights.name = date
        weight_rows.append(weights)
    return pd.DataFrame(weight_rows).fillna(0.0)


def combine_price_panel(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [prices[group][TICKERS[group]] for group in ASSET_GROUPS]
    frames.append(prices[CASH_GROUP][[CASH_ASSET]])
    return pd.concat(frames, axis=1).sort_index()


def simulate_weighted_portfolio(price_panel: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    aligned_prices = price_panel.reindex(weights.index)
    returns = aligned_prices.pct_change(fill_method=None).fillna(0.0)
    aligned_weights = weights.reindex(returns.index).fillna(0.0)
    portfolio_return = (returns[aligned_weights.columns] * aligned_weights).sum(axis=1)
    nav = INITIAL_NAV * (1.0 + portfolio_return).cumprod()
    out = pd.DataFrame(
        {
            "nav": nav,
            "return": portfolio_return,
            "risky_weight": aligned_weights.drop(columns=[CASH_ASSET], errors="ignore").sum(axis=1),
            "cash_weight": aligned_weights.get(CASH_ASSET, pd.Series(0.0, index=aligned_weights.index)),
        },
        index=aligned_weights.index,
    )
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1.0
    return out


def apply_equity_curve_momentum_overlay(
    base_nav: pd.DataFrame,
    cash_prices: pd.Series,
    lookback: int = 6,
) -> pd.DataFrame:
    joined = pd.concat([base_nav["nav"].rename("mpaa"), cash_prices.rename(CASH_ASSET)], axis=1).dropna()
    score = average_momentum_score(joined[["mpaa"]], lookback=lookback)["mpaa"]
    returns = joined.pct_change(fill_method=None).fillna(0.0)
    final_returns: list[float] = []
    final_scores: list[float] = []
    for row_number, date in enumerate(joined.index):
        if row_number == 0:
            final_scores.append(0.0)
            final_returns.append(0.0)
            continue
        signal_date = joined.index[row_number - 1]
        risky_weight = score.loc[signal_date]
        risky_weight = float(risky_weight) if np.isfinite(risky_weight) else 0.0
        final_scores.append(risky_weight)
        final_returns.append(
            risky_weight * returns.loc[date, "mpaa"] + (1.0 - risky_weight) * returns.loc[date, CASH_ASSET]
        )
    nav = INITIAL_NAV * (1.0 + pd.Series(final_returns, index=joined.index)).cumprod()
    out = pd.DataFrame(
        {
            "nav": nav,
            "return": final_returns,
            "mpaa_weight": final_scores,
            "cash_weight": 1.0 - pd.Series(final_scores, index=joined.index),
        },
        index=joined.index,
    )
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1.0
    return out.iloc[lookback + 1 :]


def performance_metrics(nav: pd.DataFrame, label: str) -> dict[str, Any]:
    if nav.empty or len(nav) < 2:
        return {"strategy": label}
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    total_return = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0
    cagr = (nav["nav"].iloc[-1] / nav["nav"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = nav["return"].std(ddof=0) * np.sqrt(12)
    sharpe = (nav["return"].mean() * 12) / vol if vol > 0 else np.nan
    mdd = nav["drawdown"].min()
    return {
        "strategy": label,
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "years": years,
        "final_nav": nav["nav"].iloc[-1],
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "vol": vol,
        "sharpe": sharpe,
        "avg_risky_weight": nav.get("risky_weight", nav.get("mpaa_weight", pd.Series(dtype=float))).mean(),
        "avg_cash_weight": nav.get("cash_weight", pd.Series(dtype=float)).mean(),
    }


def build_summary(base_nav: pd.DataFrame, final_nav: pd.DataFrame, price_panel: pd.DataFrame) -> pd.DataFrame:
    rows = [performance_metrics(base_nav, "MPAA base"), performance_metrics(final_nav, "MPAA 6M equity overlay")]
    if "KOSPI" in price_panel:
        kospi = price_panel["KOSPI"].reindex(final_nav.index).dropna()
        if not kospi.empty:
            kospi_nav = pd.DataFrame(index=kospi.index)
            kospi_nav["return"] = kospi.pct_change(fill_method=None).fillna(0.0)
            kospi_nav["nav"] = INITIAL_NAV * (1.0 + kospi_nav["return"]).cumprod()
            kospi_nav["drawdown"] = kospi_nav["nav"] / kospi_nav["nav"].cummax() - 1.0
            rows.append(performance_metrics(kospi_nav, "KOSPI buy-and-hold"))
    return pd.DataFrame(rows)


def run_backtest(input_path: Path, end: str | None = None) -> BacktestResult:
    original = read_mpaa_workbook(input_path)
    appended, source_status = append_public_proxy_data(original, end=end)
    price_panel = combine_price_panel(appended)
    weights = calculate_mpaa_weights(appended)
    base_nav = simulate_weighted_portfolio(price_panel, weights)
    final_nav = apply_equity_curve_momentum_overlay(base_nav, appended[CASH_GROUP][CASH_ASSET])
    summary = build_summary(base_nav, final_nav, price_panel)
    return BacktestResult(appended, price_panel, source_status, base_nav, final_nav, weights, summary)


def write_outputs(result: BacktestResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.appended_prices.to_csv(out_dir / "mpaa_appended_prices.csv", encoding="utf-8-sig")
    result.source_status.to_csv(out_dir / "source_status.csv", index=False, encoding="utf-8-sig")
    result.base_nav.to_csv(out_dir / "base_nav.csv", encoding="utf-8-sig")
    result.final_nav.to_csv(out_dir / "final_nav.csv", encoding="utf-8-sig")
    result.weights.to_csv(out_dir / "weights.csv", encoding="utf-8-sig")
    result.summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    latest_weights = result.weights.tail(1).T.reset_index()
    latest_weights.columns = ["asset", "weight"]
    latest_weights = latest_weights[latest_weights["weight"] > 0].sort_values("weight", ascending=False)
    latest_weights.to_csv(out_dir / "latest_weights.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest systrader79-style MPAA with public proxy extension data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--end", default=None, help="Optional yfinance end date, exclusive, YYYY-MM-DD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_backtest(args.input, end=args.end)
    write_outputs(result, args.out_dir)
    print(f"Wrote MPAA backtest outputs to {args.out_dir}")
    print(result.summary.to_string(index=False))
    print(f"Source notebook: {SOURCE_NOTEBOOK_URL}")


if __name__ == "__main__":
    main()
