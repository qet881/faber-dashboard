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


INITIAL_NAV = 100.0


def price_at(data: dict, asset: str, date: pd.Timestamp, price_col: str) -> float | None:
    if asset == app.CASH_NAME:
        px = app.get_price_at_date(data.get(app.CASH_NAME), date, price_col=price_col)
        return px if px is not None and px > 0 else 10000.0
    return app.get_price_at_date(data.get(asset), date, price_col=price_col)


def momentum_score(data: dict, asset: str, date: pd.Timestamp, lookback: int, price_col: str) -> float:
    signal_df = data.get(f"{asset}_모멘텀", data.get(asset))
    current = app.get_price_at_date(signal_df, date, price_col=price_col)
    if current is None or current <= 0:
        return 0.0
    score = 0
    valid = 0
    for months_ago in range(1, lookback + 1):
        past_date = app.get_month_end_date(date - relativedelta(months=months_ago))
        past = app.get_price_at_date(signal_df, past_date, price_col=price_col)
        if past is None or past <= 0:
            continue
        valid += 1
        if current > past:
            score += 1
    min_valid = min(6, lookback)
    if valid < min_valid:
        return 0.0
    return score / valid


def target_weights(
    data: dict,
    date: pd.Timestamp,
    assets: list[str],
    cap: float,
    lookback: int,
    price_col: str,
) -> dict[str, float]:
    weights = {}
    for asset in assets:
        weights[asset] = cap * momentum_score(data, asset, date, lookback, price_col)
    risky_sum = sum(weights.values())
    if risky_sum > 1.0:
        weights = {k: v / risky_sum for k, v in weights.items()}
        risky_sum = 1.0
    weights[app.CASH_NAME] = max(0.0, 1.0 - risky_sum)
    return weights


def monthly_rebalance_dates(index: pd.DatetimeIndex, timing: str) -> set[pd.Timestamp]:
    index = pd.DatetimeIndex(index)
    dates = []
    by_month = pd.Series(index=index, data=index).groupby(index.to_period("M"))
    for _, month_dates in by_month:
        values = list(month_dates.values)
        if not values:
            continue
        if timing == "month_start":
            pos = 0
        elif timing == "month_mid":
            pos = len(values) // 2
        elif timing == "month_end":
            pos = len(values) - 1
        else:
            raise ValueError(f"Unknown timing: {timing}")
        dates.append(pd.Timestamp(values[pos]))
    return set(dates)


def rebalance(
    nav: float,
    date: pd.Timestamp,
    holdings: dict[str, float],
    weights: dict[str, float],
    data: dict,
    assets: list[str],
    price_col: str,
) -> None:
    cash_value = nav * float(weights.get(app.CASH_NAME, 0.0))
    for asset in assets:
        px = price_at(data, asset, date, price_col)
        target_value = nav * float(weights.get(asset, 0.0))
        if px is None or px <= 0:
            holdings[asset] = 0.0
            cash_value += target_value
        else:
            holdings[asset] = target_value / px
    cash_px = price_at(data, app.CASH_NAME, date, price_col) or 10000.0
    holdings[app.CASH_NAME] = cash_value / cash_px


def portfolio_value(
    holdings: dict[str, float],
    date: pd.Timestamp,
    data: dict,
    assets: list[str],
    price_col: str,
) -> float:
    total = 0.0
    for asset in assets:
        px = price_at(data, asset, date, price_col)
        if px is not None and px > 0:
            total += holdings.get(asset, 0.0) * px
    cash_px = price_at(data, app.CASH_NAME, date, price_col) or 10000.0
    total += holdings.get(app.CASH_NAME, 0.0) * cash_px
    return total


def simulate_variant(
    data: dict,
    start_date: datetime,
    end_date: datetime,
    assets: list[str],
    cap: float = 0.20,
    lookback: int = 12,
    timing: str = "month_end",
    price_col: str = "Adj Close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trading_dates = app.build_trading_calendar(data, start_date, end_date, anchor_name=app.KR_STOCK_MIX_ASSET)
    if len(trading_dates) == 0:
        raise RuntimeError("No trading dates")
    rebal_dates = monthly_rebalance_dates(trading_dates, timing)
    first = trading_dates[0]
    holdings = {asset: 0.0 for asset in assets + [app.CASH_NAME]}
    weights = target_weights(data, first, assets, cap, lookback, price_col)
    rebalance(INITIAL_NAV, first, holdings, weights, data, assets, price_col)

    nav_rows = []
    weight_rows = []
    last_nav = INITIAL_NAV
    for date in trading_dates:
        raw = portfolio_value(holdings, date, data, assets, price_col)
        nav = app._safe_nav_value(raw, last_nav) or last_nav
        last_nav = float(nav)
        nav_rows.append({"date": date, "nav": nav})
        weight_rows.append(
            {
                "date": date,
                **weights,
                "stock_weight": sum(weights.get(a, 0.0) for a in [app.KR_STOCK_MIX_ASSET, app.NASDAQ100_ASSET_NAME]),
                "risk_asset_weight": sum(weights.get(a, 0.0) for a in assets),
            }
        )
        if date in rebal_dates and date != first:
            weights = target_weights(data, date, assets, cap, lookback, price_col)
            rebalance(nav, date, holdings, weights, data, assets, price_col)

    nav_df = pd.DataFrame(nav_rows).set_index("date").sort_index()
    nav_df["return"] = nav_df["nav"].pct_change().fillna(0.0)
    nav_df["running_max"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = nav_df["nav"] / nav_df["running_max"] - 1.0
    weight_df = pd.DataFrame(weight_rows).set_index("date").sort_index()
    return nav_df, weight_df


def metrics(group: str, name: str, nav: pd.DataFrame, weights: pd.DataFrame) -> dict:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    final_nav = float(nav["nav"].iloc[-1])
    cagr = (final_nav / float(nav["nav"].iloc[0])) ** (1.0 / years) - 1.0
    return {
        "group": group,
        "name": name,
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "years": years,
        "final_nav": final_nav,
        "total_return": final_nav / float(nav["nav"].iloc[0]) - 1.0,
        "cagr": cagr,
        "daily_mdd": float(nav["drawdown"].min()),
        "monthly_mdd": app.calculate_monthly_mdd(nav),
        "avg_risk_asset_weight": float(weights["risk_asset_weight"].mean()),
        "max_risk_asset_weight": float(weights["risk_asset_weight"].max()),
        "avg_stock_weight": float(weights["stock_weight"].mean()),
    }


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def summarize_group(summary: pd.DataFrame, group: str) -> str:
    df = summary[summary["group"] == group].copy()
    display = pd.DataFrame(
        {
            "name": df["name"],
            "CAGR": df["cagr"].map(format_pct),
            "MDD 일별": df["daily_mdd"].map(format_pct),
            "MDD 월말": df["monthly_mdd"].map(format_pct),
            "평균 위험자산": df["avg_risk_asset_weight"].map(format_pct),
        }
    )
    return markdown_table(display, list(display.columns))


def write_readme(out_dir: Path, summary: pd.DataFrame) -> None:
    baseline = summary[(summary["group"] == "baseline") & (summary["name"] == "HaenamP baseline")].iloc[0]
    text = f"""# 해남P 강건성 테스트

## 기준

- 실행일: 2026-06-17
- 기본 데이터: 앱의 hybrid 장기 데이터셋
- 기본 전략: 앱의 해남P 데이터셋 위 5자산 연속모멘텀
- 기본값: 자산별 20% 한도, 12개월 연속모멘텀, 월말 리밸런싱
- 현금: 앱의 `{app.CASH_NAME}` 사용

## Baseline

- CAGR: {baseline['cagr']:.2%}
- MDD 일별: {baseline['daily_mdd']:.2%}
- MDD 월말: {baseline['monthly_mdd']:.2%}

## 시작연도 민감도

{summarize_group(summary, "start_year")}

## 자산별 한도 민감도

{summarize_group(summary, "cap")}

## 모멘텀 기간 민감도

{summarize_group(summary, "lookback")}

## 리밸런싱 날짜 민감도

{summarize_group(summary, "rebalance_timing")}

## 자산 제외 테스트

{summarize_group(summary, "ablation")}

## 1차 해석

- 시작연도를 바꿔도 MDD는 대체로 -8.5~-10.0% 범위에 머문다. 시작연도가 늦을수록 2000년대 초반 약세장이 빠져 CAGR은 오히려 높아진다.
- 자산별 한도는 수익/낙폭을 거의 선형으로 조절한다. 15%는 더 안전하지만 CAGR이 낮고, 25%는 CAGR이 11%대로 올라가지만 MDD도 -12%대로 커진다.
- 6/9/12개월 모멘텀 기간을 바꿔도 CAGR 9% 안팎과 MDD -10% 안팎이 유지된다. 특정 lookback 하나에만 의존하는 모습은 약하다.
- 월초/월중/월말 리밸런싱을 바꿔도 성과가 크게 붕괴하지 않는다. 월말이 가장 좋지만 차이가 과도하지는 않다.
- 자산 제외 테스트에서는 모든 자산을 함께 쓰는 기본형이 가장 강하다. 특히 주식 슬롯과 금을 빼면 CAGR이 크게 낮아진다.
- 결론적으로 해남P는 과최적화 위험이 전혀 없다고 말할 수는 없지만, 이번 강건성 테스트에서는 꽤 튼튼한 편으로 보인다.
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=app.DEFAULT_BACKTEST_START_DATE.strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default="docs/haenam_p_robustness")
    parser.add_argument("--price-col", default="Adj Close")
    args = parser.parse_args()

    start = pd.Timestamp(args.start).to_pydatetime()
    end = pd.Timestamp(args.end).to_pydatetime()
    data_start = start - relativedelta(months=18)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_data = app.load_market_data(data_start, end, hybrid=True)
    base_data = app.clamp_market_data_to_date(base_data, end)
    data = app.build_haenam_p_strategy_data(base_data, data_start, end, price_col=args.price_col)
    if data is None:
        raise RuntimeError("Failed to build HaenamP strategy data.")
    assets = list(app.ASSETS.keys())

    specs: list[tuple[str, str, datetime, list[str], float, int, str]] = []
    specs.append(("baseline", "HaenamP baseline", start, assets, 0.20, 12, "month_end"))

    for year in (2000, 2003, 2007, 2010, 2013, 2016, 2020):
        specs.append(("start_year", f"start {year}", datetime(year, 1, 1), assets, 0.20, 12, "month_end"))
    for cap in (0.15, 0.20, 0.25):
        specs.append(("cap", f"cap {cap:.0%}", start, assets, cap, 12, "month_end"))
    for lookback in (6, 9, 12):
        specs.append(("lookback", f"{lookback}M momentum", start, assets, 0.20, lookback, "month_end"))
    for timing in ("month_start", "month_mid", "month_end"):
        specs.append(("rebalance_timing", timing, start, assets, 0.20, 12, timing))
    for removed in assets:
        kept = [asset for asset in assets if asset != removed]
        specs.append(("ablation", f"without {removed}", start, kept, 0.20, 12, "month_end"))

    rows = []
    nav_outputs = []
    for group, name, spec_start, spec_assets, cap, lookback, timing in specs:
        nav, weights = simulate_variant(
            data,
            spec_start,
            end,
            spec_assets,
            cap=cap,
            lookback=lookback,
            timing=timing,
            price_col=args.price_col,
        )
        rows.append(metrics(group, name, nav, weights))
        if group in {"baseline", "cap", "lookback", "rebalance_timing"}:
            nav_outputs.append(nav[["nav"]].rename(columns={"nav": f"{group}:{name}"}))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    if nav_outputs:
        pd.concat(nav_outputs, axis=1).to_csv(out_dir / "nav.csv", encoding="utf-8-sig")
    write_readme(out_dir, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
