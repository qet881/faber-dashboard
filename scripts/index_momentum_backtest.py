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
ASSET_LABELS = {
    app.KR_STOCK_MIX_ASSET: "KOSPI200",
    app.NASDAQ100_ASSET_NAME: "Nasdaq100",
}


def _price(df: pd.DataFrame | None, date: pd.Timestamp, price_col: str) -> float | None:
    return app.get_price_at_date(df, date, price_col=price_col)


def continuous_momentum_exposure(asset: str, date: pd.Timestamp, data: dict, price_col: str) -> float:
    signal_df = data.get(f"{asset}_모멘텀", data.get(asset))
    _, score = app.calculate_momentum_score_at_date(asset, date, signal_df, price_col=price_col)
    if score is None or not np.isfinite(score):
        return 0.0
    return float(np.clip(score, 0.0, 1.0))


def four_point_momentum_exposure(asset: str, date: pd.Timestamp, data: dict, price_col: str) -> float:
    signal_df = data.get(f"{asset}_모멘텀", data.get(asset))
    current = _price(signal_df, date, price_col)
    if current is None or current <= 0:
        return 0.0

    score = 0
    valid = 0
    for months in (3, 6, 9, 12):
        ref_date = app.get_month_end_date(date - relativedelta(months=months))
        past = _price(signal_df, ref_date, price_col)
        if past is None or past <= 0:
            continue
        valid += 1
        if current > past:
            score += 1
    if valid < 4:
        return 0.0
    return score / 4.0


def _calc_portfolio_value(holdings: dict[str, float], date: pd.Timestamp, data: dict, assets: list[str], price_col: str) -> float:
    total = 0.0
    for asset in assets:
        px = _price(data.get(asset), date, price_col)
        if px is not None and px > 0:
            total += holdings.get(asset, 0.0) * px
    cash_px = _price(data.get(app.CASH_NAME), date, price_col)
    if cash_px is None or cash_px <= 0:
        cash_px = 10000.0
    total += holdings.get(app.CASH_NAME, 0.0) * cash_px
    return total


def _rebalance(
    nav: float,
    date: pd.Timestamp,
    target_weights: dict[str, float],
    holdings: dict[str, float],
    data: dict,
    assets: list[str],
    price_col: str,
) -> None:
    cash_value = nav * float(target_weights.get(app.CASH_NAME, 0.0))
    for asset in assets:
        px = _price(data.get(asset), date, price_col)
        target_value = nav * float(target_weights.get(asset, 0.0))
        if px is None or px <= 0:
            holdings[asset] = 0.0
            cash_value += target_value
        else:
            holdings[asset] = target_value / px
    cash_px = _price(data.get(app.CASH_NAME), date, price_col)
    if cash_px is None or cash_px <= 0:
        cash_px = 10000.0
    holdings[app.CASH_NAME] = cash_value / cash_px


def simulate(
    name: str,
    sleeves: dict[str, float],
    exposure_func,
    data: dict,
    start_date: datetime,
    end_date: datetime,
    price_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = list(sleeves.keys())
    trading_dates = app.build_trading_calendar(data, start_date, end_date, anchor_name=app.KR_STOCK_MIX_ASSET)
    if len(trading_dates) == 0:
        raise RuntimeError(f"No trading dates for {name}")

    holdings = {asset: 0.0 for asset in assets + [app.CASH_NAME]}

    def target_weights(date: pd.Timestamp) -> dict[str, float]:
        weights = {}
        stock_total = 0.0
        for asset, sleeve_weight in sleeves.items():
            exposure = exposure_func(asset, date, data, price_col)
            w = float(sleeve_weight) * exposure
            weights[asset] = w
            stock_total += w
        weights[app.CASH_NAME] = max(0.0, 1.0 - stock_total)
        return weights

    actual_start = trading_dates[0]
    weights = target_weights(actual_start)
    _rebalance(INITIAL_NAV, actual_start, weights, holdings, data, assets, price_col)

    nav_rows = []
    weight_rows = []
    last_nav = INITIAL_NAV
    for i, date in enumerate(trading_dates):
        nav_raw = _calc_portfolio_value(holdings, date, data, assets, price_col)
        nav = app._safe_nav_value(nav_raw, last_nav) or last_nav
        last_nav = float(nav)
        nav_rows.append({"date": date, "nav": nav})
        row = {"date": date, **weights, "stock_weight": sum(weights.get(asset, 0.0) for asset in assets)}
        weight_rows.append(row)

        if app._is_month_end_rebalance_day(trading_dates, i) and date != actual_start:
            weights = target_weights(date)
            _rebalance(nav, date, weights, holdings, data, assets, price_col)

    nav_df = pd.DataFrame(nav_rows).set_index("date").sort_index()
    nav_df["return"] = nav_df["nav"].pct_change().fillna(0.0)
    nav_df["running_max"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = nav_df["nav"] / nav_df["running_max"] - 1.0
    weight_df = pd.DataFrame(weight_rows).set_index("date").sort_index()
    return nav_df, weight_df


def metrics(name: str, nav: pd.DataFrame, weights: pd.DataFrame) -> dict:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    final_nav = float(nav["nav"].iloc[-1])
    cagr = (final_nav / float(nav["nav"].iloc[0])) ** (1.0 / years) - 1.0
    monthly_mdd = app.calculate_monthly_mdd(nav)
    return {
        "strategy": name,
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "years": years,
        "final_nav": final_nav,
        "total_return": final_nav / float(nav["nav"].iloc[0]) - 1.0,
        "cagr": cagr,
        "daily_mdd": float(nav["drawdown"].min()),
        "monthly_mdd": monthly_mdd,
        "avg_stock_weight": float(weights["stock_weight"].mean()),
        "max_stock_weight": float(weights["stock_weight"].max()),
    }


def write_readme(out_dir: Path, summary: pd.DataFrame) -> None:
    display = summary.copy()
    display["CAGR"] = display["cagr"].map(lambda x: f"{x:.2%}")
    display["MDD 일별"] = display["daily_mdd"].map(lambda x: f"{x:.2%}")
    display["MDD 월말"] = display["monthly_mdd"].map(lambda x: f"{x:.2%}")
    display["평균 주식비중"] = display["avg_stock_weight"].map(lambda x: f"{x:.2%}")
    display["최대 주식비중"] = display["max_stock_weight"].map(lambda x: f"{x:.2%}")
    table = display[["strategy", "CAGR", "MDD 일별", "MDD 월말", "평균 주식비중", "최대 주식비중"]]
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_table = "\n".join(lines)
    text = f"""# 지수 모멘텀 백테스트

## 기준

- 기본 데이터: 앱의 hybrid 장기 데이터셋
- 대상: 코스피200, 나스닥100
- 리밸런싱: 월말 거래일
- 현금: 앱의 `{app.CASH_NAME}` 데이터 사용
- 연속모멘텀: 최근 12개월 중 현재가가 과거 월말보다 높은 비율만큼 투자
- 4분할 모멘텀: 3/6/9/12개월 전 월말보다 높을 때마다 1점, 점수/4만큼 투자

## 결과

{md_table}

## 해석

- 단일 지수에서는 이번 기간 기준 코스피200 모멘텀이 나스닥100보다 CAGR도 높고 MDD도 낮았다.
- 4분할 모멘텀은 신호가 더 거칠어서 연속모멘텀보다 주식비중이 높게 유지되는 구간이 생긴다.
- 50:50 바구니는 단일 지수보다 균형이 좋아지지만, 해남P처럼 채권/금/현금이 같이 들어간 포트폴리오의 안정성에는 미치지 못할 수 있다.
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=app.DEFAULT_BACKTEST_START_DATE.strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default="docs/index_momentum_backtest")
    parser.add_argument("--price-col", default="Adj Close")
    args = parser.parse_args()

    start = pd.Timestamp(args.start).to_pydatetime()
    end = pd.Timestamp(args.end).to_pydatetime()
    data_start = start - relativedelta(months=18)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = app.load_market_data(data_start, end, hybrid=True)
    data = app.clamp_market_data_to_date(data, end)

    specs = [
        ("KOSPI200 continuous", {app.KR_STOCK_MIX_ASSET: 1.0}, continuous_momentum_exposure),
        ("Nasdaq100 continuous", {app.NASDAQ100_ASSET_NAME: 1.0}, continuous_momentum_exposure),
        ("KOSPI200/Nasdaq100 50-50 continuous", {app.KR_STOCK_MIX_ASSET: 0.5, app.NASDAQ100_ASSET_NAME: 0.5}, continuous_momentum_exposure),
        ("KOSPI200 4-point", {app.KR_STOCK_MIX_ASSET: 1.0}, four_point_momentum_exposure),
        ("Nasdaq100 4-point", {app.NASDAQ100_ASSET_NAME: 1.0}, four_point_momentum_exposure),
        ("KOSPI200/Nasdaq100 50-50 4-point", {app.KR_STOCK_MIX_ASSET: 0.5, app.NASDAQ100_ASSET_NAME: 0.5}, four_point_momentum_exposure),
    ]

    summary_rows = []
    nav_outputs = []
    for name, sleeves, fn in specs:
        nav, weights = simulate(name, sleeves, fn, data, start, end, args.price_col)
        summary_rows.append(metrics(name, nav, weights))
        safe = name.lower().replace("/", "_").replace(" ", "_")
        nav.to_csv(out_dir / f"nav_{safe}.csv", encoding="utf-8-sig")
        weights.to_csv(out_dir / f"weights_{safe}.csv", encoding="utf-8-sig")
        nav_outputs.append(nav[["nav"]].rename(columns={"nav": name}))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(nav_outputs, axis=1).to_csv(out_dir / "nav.csv", encoding="utf-8-sig")
    write_readme(out_dir, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
