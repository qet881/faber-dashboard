from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


INITIAL_CAPITAL = 100.0


def _month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    dates = []
    for _, group in pd.Series(index=index, data=index).groupby(index.to_period("M")):
        if len(group) > 0:
            dates.append(pd.Timestamp(group.iloc[-1]))
    return dates


def _normalize_nav(nav: pd.DataFrame) -> pd.DataFrame:
    out = nav[["nav"]].copy().sort_index()
    out["nav"] = out["nav"] / float(out["nav"].iloc[0]) * INITIAL_CAPITAL
    out["return"] = out["nav"].pct_change().fillna(0.0)
    out["running_max"] = out["nav"].cummax()
    out["drawdown"] = out["nav"] / out["running_max"] - 1.0
    return out


def _period_return(nav: pd.DataFrame, months: int) -> float | None:
    cutoff = nav.index[0] + relativedelta(months=months)
    future = nav[nav.index >= cutoff]
    if future.empty:
        return None
    end_value = float(future["nav"].iloc[0])
    return end_value / float(nav["nav"].iloc[0]) - 1.0


def _period_mdd(nav: pd.DataFrame, months: int) -> float | None:
    cutoff = nav.index[0] + relativedelta(months=months)
    window = nav[nav.index <= cutoff].copy()
    if len(window) < 2:
        return None
    window["running_max"] = window["nav"].cummax()
    window["drawdown"] = window["nav"] / window["running_max"] - 1.0
    return float(window["drawdown"].min())


def _days_to_initial_recovery(nav: pd.DataFrame) -> int | None:
    start_value = float(nav["nav"].iloc[0])
    recovered = nav.iloc[1:][nav["nav"].iloc[1:] >= start_value]
    if recovered.empty:
        return None
    return int((recovered.index[0] - nav.index[0]).days)


def _cagr(nav: pd.DataFrame) -> float:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (float(nav["nav"].iloc[-1]) / float(nav["nav"].iloc[0])) ** (1.0 / years) - 1.0


def _entry_metrics(label: str, start_date: pd.Timestamp, nav: pd.DataFrame) -> dict:
    normalized = _normalize_nav(nav)
    return {
        "group": label,
        "start": normalized.index[0].strftime("%Y-%m-%d"),
        "end": normalized.index[-1].strftime("%Y-%m-%d"),
        "years": (normalized.index[-1] - normalized.index[0]).days / 365.25,
        "cagr": _cagr(normalized),
        "total_return": float(normalized["nav"].iloc[-1] / normalized["nav"].iloc[0] - 1.0),
        "mdd_full": float(normalized["drawdown"].min()),
        "mdd_1y": _period_mdd(normalized, 12),
        "mdd_3y": _period_mdd(normalized, 36),
        "mdd_5y": _period_mdd(normalized, 60),
        "return_1y": _period_return(normalized, 12),
        "return_3y": _period_return(normalized, 36),
        "return_5y": _period_return(normalized, 60),
        "days_to_recover_initial": _days_to_initial_recovery(normalized),
    }


def _summarize_entries(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, df in entries.groupby("group"):
        recover = df["days_to_recover_initial"].dropna()
        rows.append(
            {
                "group": group,
                "count": int(len(df)),
                "median_recovery_days": float(recover.median()) if not recover.empty else np.nan,
                "p90_recovery_days": float(recover.quantile(0.90)) if not recover.empty else np.nan,
                "max_recovery_days": float(recover.max()) if not recover.empty else np.nan,
                "unrecovered_count": int(df["days_to_recover_initial"].isna().sum()),
                "worst_1y_return": float(df["return_1y"].min(skipna=True)),
                "median_1y_return": float(df["return_1y"].median(skipna=True)),
                "worst_3y_return": float(df["return_3y"].min(skipna=True)),
                "median_3y_return": float(df["return_3y"].median(skipna=True)),
                "worst_5y_return": float(df["return_5y"].min(skipna=True)),
                "median_5y_return": float(df["return_5y"].median(skipna=True)),
                "worst_full_mdd": float(df["mdd_full"].min(skipna=True)),
                "median_full_mdd": float(df["mdd_full"].median(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def _underwater_periods(nav: pd.DataFrame) -> pd.DataFrame:
    series = nav["nav"].astype(float).sort_index()
    running_max = series.cummax()
    periods = []
    in_dd = False
    peak_date = None
    trough_date = None
    trough_dd = 0.0

    for date, value in series.items():
        high = float(running_max.loc[date])
        dd = float(value / high - 1.0)
        if dd < -1e-12:
            if not in_dd:
                in_dd = True
                prior_peak = series.loc[:date][series.loc[:date] >= high - abs(high) * 1e-10]
                peak_date = prior_peak.index[-1] if not prior_peak.empty else date
                trough_date = date
                trough_dd = dd
            elif dd < trough_dd:
                trough_date = date
                trough_dd = dd
        elif in_dd:
            periods.append(
                {
                    "peak": peak_date,
                    "trough": trough_date,
                    "recovery": date,
                    "days": int((date - peak_date).days),
                    "max_drawdown": trough_dd,
                    "recovered": True,
                }
            )
            in_dd = False

    if in_dd:
        periods.append(
            {
                "peak": peak_date,
                "trough": trough_date,
                "recovery": series.index[-1],
                "days": int((series.index[-1] - peak_date).days),
                "max_drawdown": trough_dd,
                "recovered": False,
            }
        )
    return pd.DataFrame(periods)


def _underwater_summary(nav: pd.DataFrame) -> dict:
    normalized = _normalize_nav(nav)
    periods = _underwater_periods(normalized)
    dd = normalized["drawdown"]
    recovered = periods[periods["recovered"]] if not periods.empty else periods
    longest = periods.sort_values("days", ascending=False).head(1)
    deepest = periods.sort_values("max_drawdown").head(1)
    return {
        "underwater_ratio": float((dd < -1e-12).mean()),
        "current_drawdown": float(dd.iloc[-1]),
        "ulcer_index": app.calculate_ulcer_index(normalized),
        "period_count": int(len(periods)),
        "recovered_period_count": int(len(recovered)),
        "avg_recovered_days": float(recovered["days"].mean()) if not recovered.empty else np.nan,
        "median_recovered_days": float(recovered["days"].median()) if not recovered.empty else np.nan,
        "p90_recovered_days": float(recovered["days"].quantile(0.90)) if not recovered.empty else np.nan,
        "longest_peak": longest["peak"].iloc[0].strftime("%Y-%m-%d") if not longest.empty else "",
        "longest_trough": longest["trough"].iloc[0].strftime("%Y-%m-%d") if not longest.empty else "",
        "longest_recovery": longest["recovery"].iloc[0].strftime("%Y-%m-%d") if not longest.empty else "",
        "longest_days": int(longest["days"].iloc[0]) if not longest.empty else 0,
        "longest_recovered": bool(longest["recovered"].iloc[0]) if not longest.empty else True,
        "deepest_peak": deepest["peak"].iloc[0].strftime("%Y-%m-%d") if not deepest.empty else "",
        "deepest_trough": deepest["trough"].iloc[0].strftime("%Y-%m-%d") if not deepest.empty else "",
        "deepest_recovery": deepest["recovery"].iloc[0].strftime("%Y-%m-%d") if not deepest.empty else "",
        "deepest_days": int(deepest["days"].iloc[0]) if not deepest.empty else 0,
        "deepest_drawdown": float(deepest["max_drawdown"].iloc[0]) if not deepest.empty else 0.0,
        "deepest_recovered": bool(deepest["recovered"].iloc[0]) if not deepest.empty else True,
    }


def _stock_basket(strategy_data: dict, dates: pd.DatetimeIndex, price_col: str) -> pd.Series:
    values = {}
    for asset in (app.KR_STOCK_MIX_ASSET, app.NASDAQ100_ASSET_NAME):
        prices = [
            app.get_price_at_date(strategy_data.get(asset), date, price_col=price_col)
            for date in dates
        ]
        series = pd.Series(prices, index=dates, dtype=float).ffill()
        first_valid = series.dropna()
        if first_valid.empty or float(first_valid.iloc[0]) <= 0:
            values[asset] = pd.Series(index=dates, dtype=float)
        else:
            values[asset] = series / float(first_valid.iloc[0])
    return 0.5 * values[app.KR_STOCK_MIX_ASSET] + 0.5 * values[app.NASDAQ100_ASSET_NAME]


def _is_peak_dates(series: pd.Series, dates: list[pd.Timestamp]) -> set[pd.Timestamp]:
    monthly = series.reindex(dates).dropna()
    peak = monthly.cummax()
    return set(monthly[monthly >= peak - peak.abs() * 1e-10].index)


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2%}"


def _fmt_days(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = int(round(value))
    years = value // 365
    months = (value % 365) // 30
    if years and months:
        return f"{years}y {months}m"
    if years:
        return f"{years}y"
    if months:
        return f"{months}m"
    return f"{value}d"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _write_readme(out_dir: Path, summary: pd.DataFrame, entries: pd.DataFrame, underwater: dict) -> None:
    def row(group: str) -> pd.Series:
        return summary[summary["group"] == group].iloc[0]

    all_row = row("all_month_ends")
    strat_row = row("strategy_ath_month_ends")
    stock_row = row("stock_basket_ath_month_ends")

    worst_rows = entries.sort_values("mdd_1y").head(10).copy()
    worst_rows["1Y return"] = worst_rows["return_1y"].map(_fmt_pct)
    worst_rows["1Y MDD"] = worst_rows["mdd_1y"].map(_fmt_pct)
    worst_rows["recover"] = worst_rows["days_to_recover_initial"].map(_fmt_days)
    worst_table = _markdown_table(
        worst_rows[["group", "start", "1Y return", "1Y MDD", "recover"]],
        ["group", "start", "1Y return", "1Y MDD", "recover"],
    )

    text = f"""# HaenamP Entry Timing and Underwater Test

## Setup

- Run date: 2026-06-17
- Engine: exact HaenamP strategy data and simulator from `app.py`
- Entry dates: monthly rebalance dates from 2000-01-03 to 2026-06-17
- Entry method: normalize the already-simulated HaenamP NAV to 100 at each entry date.
- High-entry definitions:
  - `strategy_ath_month_ends`: the already-running HaenamP NAV was at an all-time high on that month-end.
  - `stock_basket_ath_month_ends`: the 50/50 KOSPI200/Nasdaq100 basket was at an all-time high on that month-end.

## Entry Timing Summary

| Group | Count | Median recovery | P90 recovery | Max recovery | Worst 1Y return | Median 1Y return | Worst 3Y return | Worst full MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All month-ends | {int(all_row['count'])} | {_fmt_days(all_row['median_recovery_days'])} | {_fmt_days(all_row['p90_recovery_days'])} | {_fmt_days(all_row['max_recovery_days'])} | {_fmt_pct(all_row['worst_1y_return'])} | {_fmt_pct(all_row['median_1y_return'])} | {_fmt_pct(all_row['worst_3y_return'])} | {_fmt_pct(all_row['worst_full_mdd'])} |
| HaenamP ATH entries | {int(strat_row['count'])} | {_fmt_days(strat_row['median_recovery_days'])} | {_fmt_days(strat_row['p90_recovery_days'])} | {_fmt_days(strat_row['max_recovery_days'])} | {_fmt_pct(strat_row['worst_1y_return'])} | {_fmt_pct(strat_row['median_1y_return'])} | {_fmt_pct(strat_row['worst_3y_return'])} | {_fmt_pct(strat_row['worst_full_mdd'])} |
| Stock-basket ATH entries | {int(stock_row['count'])} | {_fmt_days(stock_row['median_recovery_days'])} | {_fmt_days(stock_row['p90_recovery_days'])} | {_fmt_days(stock_row['max_recovery_days'])} | {_fmt_pct(stock_row['worst_1y_return'])} | {_fmt_pct(stock_row['median_1y_return'])} | {_fmt_pct(stock_row['worst_3y_return'])} | {_fmt_pct(stock_row['worst_full_mdd'])} |

## Underwater Summary

- Underwater ratio: {underwater['underwater_ratio']:.1%}
- Current drawdown: {underwater['current_drawdown']:.2%}
- Ulcer Index: {underwater['ulcer_index']:.2f}
- Recovered underwater periods: {underwater['recovered_period_count']}
- Median recovered period: {_fmt_days(underwater['median_recovered_days'])}
- Average recovered period: {_fmt_days(underwater['avg_recovered_days'])}
- P90 recovered period: {_fmt_days(underwater['p90_recovered_days'])}
- Longest period: {underwater['longest_peak']} -> {underwater['longest_recovery']} ({_fmt_days(underwater['longest_days'])}, recovered={underwater['longest_recovered']})
- Deepest period: {underwater['deepest_peak']} -> {underwater['deepest_trough']} -> {underwater['deepest_recovery']} ({underwater['deepest_drawdown']:.2%}, recovered={underwater['deepest_recovered']})

## Worst 1Y Entry Starts

{worst_table}

## Interpretation

- Entering at a HaenamP all-time high was not catastrophic in this sample. The worst full-period drawdown after those entries still stayed near the historical HaenamP MDD range.
- Stock-market all-time-high entries were also manageable because HaenamP may still hold bonds, gold, or cash depending on momentum.
- The uncomfortable part is not the depth alone; it is time underwater. The strategy spends a meaningful share of days below its own prior high, so investors should expect long flat or recovery stretches even when MDD is modest.
- This supports using HaenamP as a main candidate, but it does not support performance chasing after a strong month. The rule is strongest when followed mechanically at scheduled rebalance dates.
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=app.DEFAULT_BACKTEST_START_DATE.strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default="docs/haenam_p_entry_underwater")
    parser.add_argument("--price-col", default="Adj Close")
    args = parser.parse_args()

    start = pd.Timestamp(args.start).to_pydatetime()
    end = pd.Timestamp(args.end).to_pydatetime()
    data_start = start - relativedelta(months=18)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_data = app.load_market_data(data_start, end, hybrid=True)
    base_data = app.clamp_market_data_to_date(base_data, end)
    strategy_data = app.build_haenam_p_strategy_data(base_data, data_start, end, price_col=args.price_col)
    if strategy_data is None:
        raise RuntimeError("Failed to build HaenamP strategy data.")

    full_nav = app.simulate_haenam_p_strategy(start, end, INITIAL_CAPITAL, strategy_data, price_col=args.price_col)
    if full_nav is None or full_nav.empty:
        raise RuntimeError("Failed to simulate HaenamP.")
    full_nav = _normalize_nav(full_nav)

    month_ends = _month_end_dates(full_nav.index)
    stock_basket = _stock_basket(strategy_data, full_nav.index, args.price_col)
    strategy_peak_dates = _is_peak_dates(full_nav["nav"], month_ends)
    stock_peak_dates = _is_peak_dates(stock_basket, month_ends)

    rows = []
    min_years = 1.0
    for entry_date in month_ends:
        if (pd.Timestamp(end) - entry_date).days / 365.25 < min_years:
            continue
        nav = full_nav.loc[full_nav.index >= entry_date].copy()
        if nav.empty:
            continue
        rows.append(_entry_metrics("all_month_ends", entry_date, nav))
        if entry_date in strategy_peak_dates:
            rows.append(_entry_metrics("strategy_ath_month_ends", entry_date, nav))
        if entry_date in stock_peak_dates:
            rows.append(_entry_metrics("stock_basket_ath_month_ends", entry_date, nav))

    entries = pd.DataFrame(rows)
    summary = _summarize_entries(entries)
    underwater = _underwater_summary(full_nav)
    periods = _underwater_periods(full_nav)

    entries.to_csv(out_dir / "entry_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "entry_summary.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(out_dir / "underwater_periods.csv", index=False, encoding="utf-8-sig")
    full_nav.to_csv(out_dir / "haenam_p_nav.csv", encoding="utf-8-sig")
    pd.DataFrame([underwater]).to_csv(out_dir / "underwater_summary.csv", index=False, encoding="utf-8-sig")
    _write_readme(out_dir, summary, entries, underwater)

    print("Entry summary")
    print(summary.to_string(index=False))
    print("\nUnderwater summary")
    print(pd.DataFrame([underwater]).to_string(index=False))
    print("\nWorst 1Y entries")
    print(entries.sort_values("mdd_1y").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
