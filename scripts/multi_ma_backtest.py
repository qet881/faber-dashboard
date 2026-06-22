from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


DOWNLOAD_START = "1999-01-01"
DEFAULT_START = "2005-01-01"
INITIAL_NAV = 100.0
MIN_OBSERVATIONS = 260

CORE_ASSETS = OrderedDict(
    [
        ("US Large Cap", "SPY"),
        ("Nasdaq 100", "QQQ"),
        ("US Small Cap", "IWM"),
        ("Developed ex-US", "EFA"),
        ("Emerging Markets", "EEM"),
        ("Long Treasury", "TLT"),
        ("Intermediate Treasury", "IEF"),
        ("Short Treasury", "SHY"),
        ("Gold", "GLD"),
        ("Commodities", "DBC"),
        ("US REIT", "VNQ"),
        ("Bitcoin", "BTC-USD"),
    ]
)

KOREA_ASSETS = OrderedDict(
    [
        ("KOSPI200 ETF", "069500.KS"),
        ("KOSPI200 Index", "^KS200"),
        ("KOSDAQ150 ETF", "229200.KS"),
        ("Korea Nasdaq100 ETF", "133690.KS"),
        ("Samsung Electronics", "005930.KS"),
        ("SK Hynix", "000660.KS"),
        ("Korea 10Y Bond ETF", "148070.KS"),
        ("Korea 30Y Bond ETF", "476760.KS"),
        ("Korea Gold ETF", "411060.KS"),
        ("USD/KRW", "USDKRW=X"),
    ]
)

EXTENDED_ASSETS = OrderedDict(
    [
        ("Technology", "XLK"),
        ("Financials", "XLF"),
        ("Energy", "XLE"),
        ("Health Care", "XLV"),
        ("Industrials", "XLI"),
        ("Consumer Staples", "XLP"),
        ("Consumer Discretionary", "XLY"),
        ("Utilities", "XLU"),
        ("US Value", "VTV"),
        ("US Growth", "VUG"),
        ("Momentum", "MTUM"),
        ("Quality", "QUAL"),
        ("Min Volatility", "USMV"),
        ("Japan", "EWJ"),
        ("Germany", "EWG"),
        ("United Kingdom", "EWU"),
        ("India", "INDA"),
        ("China", "MCHI"),
        ("Silver", "SLV"),
        ("Oil", "USO"),
        ("Agriculture", "DBA"),
    ]
)


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    windows: tuple[int, ...]
    stack_mode: str = "none"
    require_52w_high: bool = False
    high_52w_tolerance: float = 0.0


STRATEGIES = [
    StrategyConfig("ma5", (10, 20, 60, 120, 200), "none"),
    StrategyConfig("ma4", (20, 60, 120, 200), "none"),
    StrategyConfig("ma5_52w_high", (10, 20, 60, 120, 200), "none", True),
    StrategyConfig("ma4_52w_high", (20, 60, 120, 200), "none", True),
    StrategyConfig("ma5_52w_near5", (10, 20, 60, 120, 200), "none", True, 0.05),
    StrategyConfig("ma4_52w_near5", (20, 60, 120, 200), "none", True, 0.05),
    StrategyConfig("ma5_stack_hard", (10, 20, 60, 120, 200), "hard"),
    StrategyConfig("ma4_stack_hard", (20, 60, 120, 200), "hard"),
    StrategyConfig("ma5_stack_half", (10, 20, 60, 120, 200), "half"),
    StrategyConfig("ma4_stack_half", (20, 60, 120, 200), "half"),
]


def all_assets() -> OrderedDict[str, str]:
    out: OrderedDict[str, str] = OrderedDict()
    for group in (CORE_ASSETS, KOREA_ASSETS, EXTENDED_ASSETS):
        out.update(group)
    return out


def asset_group(asset: str) -> str:
    if asset in CORE_ASSETS:
        return "core"
    if asset in KOREA_ASSETS:
        return "korea"
    return "extended"


def download_price(symbol: str, start: str, end: str | None = None) -> pd.Series:
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        return pd.Series(dtype=float, name=symbol)
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        if field not in raw.columns.get_level_values(0):
            return pd.Series(dtype=float, name=symbol)
        selected = raw[field]
        out = selected.iloc[:, 0] if isinstance(selected, pd.DataFrame) else selected
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if col not in raw.columns:
            return pd.Series(dtype=float, name=symbol)
        out = raw[col]
    if out is None:
        return pd.Series(dtype=float, name=symbol)
    out = pd.to_numeric(out, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    out.name = symbol
    return out[out > 0]


def download_prices(tickers: OrderedDict[str, str], start: str, end: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices: dict[str, pd.Series] = {}
    status: list[dict[str, Any]] = []
    for asset, symbol in tickers.items():
        try:
            series = download_price(symbol, start, end)
            ok = len(series) >= MIN_OBSERVATIONS
            if ok:
                prices[asset] = series.rename(asset)
            status.append(
                {
                    "asset": asset,
                    "symbol": symbol,
                    "group": asset_group(asset),
                    "status": "ok" if ok else "too_short_or_empty",
                    "rows": int(len(series)),
                    "start": series.index.min().strftime("%Y-%m-%d") if not series.empty else "",
                    "end": series.index.max().strftime("%Y-%m-%d") if not series.empty else "",
                }
            )
        except Exception as exc:
            status.append(
                {
                    "asset": asset,
                    "symbol": symbol,
                    "group": asset_group(asset),
                    "status": f"error: {exc}",
                    "rows": 0,
                    "start": "",
                    "end": "",
                }
            )
    if not prices:
        raise RuntimeError("No usable price series downloaded.")
    return pd.concat(prices.values(), axis=1).sort_index(), pd.DataFrame(status)


def moving_average_signal(
    price: pd.Series,
    windows: tuple[int, ...],
    stack_mode: str = "none",
    require_52w_high: bool = False,
    high_52w_tolerance: float = 0.0,
) -> pd.DataFrame:
    clean = pd.to_numeric(price, errors="coerce").dropna().sort_index()
    ma = pd.DataFrame({f"ma{window}": clean.rolling(window, min_periods=window).mean() for window in windows})
    above = ma.lt(clean, axis=0)
    ready = ma.notna().all(axis=1)
    exposure = above.sum(axis=1).astype(float) / float(len(windows))
    exposure = exposure.where(ready)

    ma_cols = [f"ma{window}" for window in windows]
    stack = pd.Series(True, index=clean.index)
    for left, right in zip(ma_cols, ma_cols[1:]):
        stack &= ma[left] > ma[right]
    stack = stack.where(ready, False)

    if stack_mode == "hard":
        exposure = exposure.where(stack, 0.0).where(ready)
    elif stack_mode == "half":
        exposure = exposure.where(stack, exposure * 0.5).where(ready)
    elif stack_mode != "none":
        raise ValueError(f"Unknown stack_mode: {stack_mode}")

    rolling_high_52w = clean.rolling(252, min_periods=252).max()
    high_52w = clean >= rolling_high_52w
    near_52w_high = clean >= rolling_high_52w * (1.0 - high_52w_tolerance)
    if require_52w_high:
        exposure = exposure.where(near_52w_high, 0.0).where(ready)

    out = pd.DataFrame(index=clean.index)
    out["price"] = clean
    out["exposure"] = exposure
    out["above_count"] = above.sum(axis=1).where(ready)
    out["stacked"] = stack.astype(bool)
    out["high_52w"] = high_52w.where(ready, False).astype(bool)
    out["near_52w_high"] = near_52w_high.where(ready, False).astype(bool)
    for col in ma_cols:
        out[col] = ma[col]
    return out


def rebalance_position(signal: pd.Series, rebalance: str) -> pd.Series:
    clean = signal.sort_index()
    if rebalance == "daily":
        raw_position = clean
    else:
        rule = {"weekly": "W-FRI", "monthly": "ME"}[rebalance]
        rebal_dates = clean.dropna().resample(rule).last().index
        selected = clean.reindex(clean.index.union(rebal_dates)).sort_index().ffill().reindex(rebal_dates)
        selected.index = clean.index[clean.index.searchsorted(selected.index, side="right") - 1]
        selected = selected[~selected.index.duplicated(keep="last")]
        raw_position = selected.reindex(clean.index).ffill()
    return raw_position.shift(1).fillna(0.0).clip(0.0, 1.0)


def simulate_strategy(price: pd.Series, exposure: pd.Series, rebalance: str = "monthly") -> tuple[pd.DataFrame, pd.Series]:
    aligned = pd.concat([price.rename("price"), exposure.rename("signal")], axis=1).dropna(subset=["price"])
    valid_signals = aligned["signal"].dropna()
    if aligned.empty or valid_signals.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    returns = aligned["price"].pct_change().fillna(0.0)
    position = rebalance_position(aligned["signal"], rebalance).reindex(aligned.index).fillna(0.0)
    strategy_return = position * returns
    nav = INITIAL_NAV * (1.0 + strategy_return).cumprod()
    out = pd.DataFrame({"nav": nav, "return": strategy_return, "exposure": position}, index=aligned.index)
    start_idx = aligned.index.searchsorted(valid_signals.index[0], side="right")
    out = out.iloc[start_idx:]
    if out.empty:
        return out, position
    out["running_max"] = out["nav"].cummax()
    out["drawdown"] = out["nav"] / out["running_max"] - 1.0
    return out, position


def buy_and_hold(price: pd.Series, index: pd.Index | None = None) -> pd.DataFrame:
    clean = price.dropna().sort_index()
    if index is not None:
        clean = clean.reindex(index).dropna()
    if clean.empty:
        return pd.DataFrame()
    returns = clean.pct_change().fillna(0.0)
    nav = INITIAL_NAV * (1.0 + returns).cumprod()
    out = pd.DataFrame({"nav": nav, "return": returns, "exposure": 1.0}, index=clean.index)
    out["running_max"] = out["nav"].cummax()
    out["drawdown"] = out["nav"] / out["running_max"] - 1.0
    return out


def performance_metrics(nav: pd.DataFrame) -> dict[str, float | str]:
    if nav is None or nav.empty or len(nav) < 2:
        return {}
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    total = float(nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0)
    cagr = float((nav["nav"].iloc[-1] / nav["nav"].iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(nav["return"].std(ddof=0) * np.sqrt(252))
    sharpe = float((nav["return"].mean() * 252) / vol) if vol > 0 and np.isfinite(vol) else np.nan
    mdd = float(nav["drawdown"].min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    turnover = float(nav["exposure"].diff().abs().sum() / years) if years > 0 and "exposure" in nav else np.nan
    return {
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "years": float(years),
        "final_nav": float(nav["nav"].iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "mdd": mdd,
        "vol": vol,
        "sharpe": sharpe,
        "calmar": calmar,
        "avg_exposure": float(nav["exposure"].mean()) if "exposure" in nav else np.nan,
        "annual_turnover": turnover,
    }


def run_asset_backtests(prices: pd.DataFrame, start: str, rebalance: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    nav_cols: list[pd.Series] = []
    weight_cols: list[pd.Series] = []
    for asset in prices.columns:
        price = prices[asset].dropna().loc[pd.Timestamp(start) :]
        if len(price) < MIN_OBSERVATIONS:
            continue
        for strategy in STRATEGIES:
            signal = moving_average_signal(
                price,
                strategy.windows,
                strategy.stack_mode,
                strategy.require_52w_high,
                strategy.high_52w_tolerance,
            )
            nav, position = simulate_strategy(price, signal["exposure"], rebalance=rebalance)
            if nav.empty:
                continue
            bh = buy_and_hold(price, nav.index)
            if bh.empty:
                continue
            metrics = performance_metrics(nav)
            bh_metrics = performance_metrics(bh)
            row = {
                "asset": asset,
                "symbol": all_assets().get(asset, ""),
                "group": asset_group(asset),
                "strategy": strategy.name,
                **metrics,
                "bh_cagr": bh_metrics.get("cagr", np.nan),
                "bh_mdd": bh_metrics.get("mdd", np.nan),
                "bh_sharpe": bh_metrics.get("sharpe", np.nan),
                "delta_cagr": metrics.get("cagr", np.nan) - bh_metrics.get("cagr", np.nan),
                "delta_mdd": metrics.get("mdd", np.nan) - bh_metrics.get("mdd", np.nan),
                "delta_sharpe": metrics.get("sharpe", np.nan) - bh_metrics.get("sharpe", np.nan),
            }
            rows.append(row)
            nav_cols.append(nav["nav"].rename(f"{asset}__{strategy.name}"))
            weight_cols.append(position.rename(f"{asset}__{strategy.name}"))
    asset_summary = pd.DataFrame(rows)
    nav_df = pd.concat(nav_cols, axis=1) if nav_cols else pd.DataFrame()
    weights_df = pd.concat(weight_cols, axis=1) if weight_cols else pd.DataFrame()
    return asset_summary, nav_df, weights_df


def aggregate_strategy_summary(asset_summary: pd.DataFrame) -> pd.DataFrame:
    if asset_summary.empty:
        return pd.DataFrame()
    rows = []
    for strategy, group in asset_summary.groupby("strategy"):
        rows.append(
            {
                "strategy": strategy,
                "asset_count": int(len(group)),
                "mean_cagr": group["cagr"].mean(),
                "median_cagr": group["cagr"].median(),
                "mean_mdd": group["mdd"].mean(),
                "median_mdd": group["mdd"].median(),
                "mean_sharpe": group["sharpe"].mean(),
                "median_sharpe": group["sharpe"].median(),
                "cagr_win_rate": float((group["delta_cagr"] > 0).mean()),
                "mdd_improve_rate": float((group["delta_mdd"] > 0).mean()),
                "sharpe_win_rate": float((group["delta_sharpe"] > 0).mean()),
                "avg_exposure": group["avg_exposure"].mean(),
                "annual_turnover": group["annual_turnover"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe_win_rate", "mdd_improve_rate", "median_sharpe"], ascending=False)


def simulate_portfolio(prices: pd.DataFrame, strategy: StrategyConfig, start: str, rebalance: str) -> pd.DataFrame:
    usable = prices.loc[pd.Timestamp(start) :].dropna(axis=1, thresh=MIN_OBSERVATIONS)
    signals = []
    first_signal_dates = []
    for asset in usable.columns:
        sig = moving_average_signal(
            usable[asset].dropna(),
            strategy.windows,
            strategy.stack_mode,
            strategy.require_52w_high,
            strategy.high_52w_tolerance,
        )["exposure"]
        valid_sig = sig.dropna()
        if valid_sig.empty:
            continue
        first_signal_dates.append(valid_sig.index[0])
        pos = rebalance_position(sig, rebalance).rename(asset)
        signals.append(pos)
    if not signals:
        return pd.DataFrame()
    positions = pd.concat(signals, axis=1).reindex(usable.index).ffill().fillna(0.0)
    data = usable.reindex(positions.index).dropna(how="any")
    positions = positions.reindex(data.index).fillna(0.0)
    if len(data) < MIN_OBSERVATIONS or data.shape[1] == 0:
        return pd.DataFrame()
    base_weight = 1.0 / data.shape[1]
    returns = data.pct_change().fillna(0.0)
    risky_weights = positions * base_weight
    portfolio_return = (risky_weights * returns).sum(axis=1)
    nav = INITIAL_NAV * (1.0 + portfolio_return).cumprod()
    out = pd.DataFrame(
        {
            "nav": nav,
            "return": portfolio_return,
            "exposure": risky_weights.sum(axis=1),
        },
        index=data.index,
    )
    start_idx = out.index.searchsorted(max(first_signal_dates), side="right")
    out = out.iloc[start_idx:]
    if out.empty:
        return out
    out["running_max"] = out["nav"].cummax()
    out["drawdown"] = out["nav"] / out["running_max"] - 1.0
    out["annual_turnover_component"] = risky_weights.diff().abs().sum(axis=1)
    return out


def run_portfolio_backtests(prices: pd.DataFrame, start: str, rebalance: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = {
        "core": [asset for asset in CORE_ASSETS if asset in prices.columns],
        "korea": [asset for asset in KOREA_ASSETS if asset in prices.columns],
        "extended": [asset for asset in EXTENDED_ASSETS if asset in prices.columns],
        "all": list(prices.columns),
    }
    rows = []
    nav_cols = []
    for set_name, assets in sets.items():
        subset = prices[assets].dropna(axis=1, thresh=MIN_OBSERVATIONS)
        if subset.shape[1] < 2:
            continue
        for strategy in STRATEGIES:
            nav = simulate_portfolio(subset, strategy, start, rebalance)
            if nav.empty:
                continue
            metrics = performance_metrics(nav)
            rows.append({"portfolio": set_name, "asset_count": int(subset.shape[1]), "strategy": strategy.name, **metrics})
            nav_cols.append(nav["nav"].rename(f"{set_name}__{strategy.name}"))
    return pd.DataFrame(rows), pd.concat(nav_cols, axis=1) if nav_cols else pd.DataFrame()


def latest_signals(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset in prices.columns:
        price = prices[asset].dropna()
        if len(price) < MIN_OBSERVATIONS:
            continue
        for strategy in STRATEGIES:
            signal = moving_average_signal(
                price,
                strategy.windows,
                strategy.stack_mode,
                strategy.require_52w_high,
                strategy.high_52w_tolerance,
            ).dropna(subset=["exposure"])
            if signal.empty:
                continue
            last = signal.iloc[-1]
            rows.append(
                {
                    "asset": asset,
                    "symbol": all_assets().get(asset, ""),
                    "group": asset_group(asset),
                    "strategy": strategy.name,
                    "date": signal.index[-1].strftime("%Y-%m-%d"),
                    "price": float(last["price"]),
                    "exposure": float(last["exposure"]),
                    "above_count": int(last["above_count"]),
                    "stacked": bool(last["stacked"]),
                    "high_52w": bool(last["high_52w"]),
                    "near_52w_high": bool(last["near_52w_high"]),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi moving-average allocation backtest.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--download-start", default=DOWNLOAD_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="monthly")
    parser.add_argument("--out-dir", default="docs/multi_ma_backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, status = download_prices(all_assets(), args.download_start, args.end)
    asset_summary, asset_nav, weights = run_asset_backtests(prices, args.start, args.rebalance)
    summary = aggregate_strategy_summary(asset_summary)
    portfolio_summary, portfolio_nav = run_portfolio_backtests(prices, args.start, args.rebalance)
    signals = latest_signals(prices)

    status.to_csv(out_dir / "data_status.csv", index=False, encoding="utf-8-sig")
    prices.to_csv(out_dir / "prices.csv", encoding="utf-8-sig")
    asset_summary.to_csv(out_dir / "asset_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    portfolio_summary.to_csv(out_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    if not asset_nav.empty:
        asset_nav.to_csv(out_dir / "asset_nav.csv", encoding="utf-8-sig")
    if not portfolio_nav.empty:
        portfolio_nav.to_csv(out_dir / "nav.csv", encoding="utf-8-sig")
    if not weights.empty:
        weights.to_csv(out_dir / "weights.csv", encoding="utf-8-sig")

    print(f"Wrote results to {out_dir}")
    if not summary.empty:
        print(summary[["strategy", "asset_count", "median_cagr", "median_mdd", "median_sharpe", "mdd_improve_rate", "sharpe_win_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
