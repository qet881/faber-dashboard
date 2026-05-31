from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf


START = "2000-01-01"
DEFAULT_START = "2011-01-31"
INITIAL_NAV = 100.0


TICKERS = {
    "kospi200": "^KS200",
    "samsung": "005930.KS",
    "hynix": "000660.KS",
    "tiger_nasdaq100": "133690.KS",
    "qqq": "QQQ",
    "usdk_rw": "USDKRW=X",
    "gld": "GLD",
    "tlt": "TLT",
}


@dataclass(frozen=True)
class BacktestConfig:
    name: str
    kr_exec: str
    kr_signal: str
    us_signal: str
    us_exec: str
    fear_rule: str
    defense: str
    kr_mode: str = "combined"


def download_adj_close(tickers: dict[str, str], start: str) -> pd.DataFrame:
    raw = yf.download(
        list(tickers.values()),
        start=start,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No yfinance data downloaded.")

    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Adj Close"].copy()
    else:
        px = raw[["Adj Close"]].copy()
        px.columns = list(tickers.values())

    inverse = {v: k for k, v in tickers.items()}
    px = px.rename(columns=inverse)
    return px.sort_index().dropna(how="all")


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("ME").last().dropna(how="all")


def as_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype=float)
        return obj.iloc[:, 0]
    return obj


def build_series(monthly: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=monthly.index)
    out["KOSPI200"] = as_series(monthly["kospi200"])
    out["Samsung"] = as_series(monthly["samsung"])
    out["Hynix"] = as_series(monthly["hynix"])
    out["KR_Semi_50_50"] = normalized_mix(monthly[["samsung", "hynix"]], [0.5, 0.5])
    out["KR_Attack_25_37_37"] = normalized_mix(
        monthly[["kospi200", "samsung", "hynix"]],
        [0.25, 0.375, 0.375],
    )
    out["TIGER_Nasdaq100"] = as_series(monthly["tiger_nasdaq100"])
    usdk_rw = as_series(monthly["usdk_rw"])
    out["QQQ_KRW"] = as_series(monthly["qqq"]) * usdk_rw
    out["GLD_KRW"] = as_series(monthly["gld"]) * usdk_rw
    out["TLT_KRW"] = as_series(monthly["tlt"]) * usdk_rw
    out["Cash"] = 1.0
    out["Defense_Basket"] = normalized_mix(out[["GLD_KRW", "TLT_KRW", "Cash"]], [1 / 3, 1 / 3, 1 / 3])
    return out


def normalized_mix(df: pd.DataFrame, weights: list[float]) -> pd.Series:
    rets = df.pct_change()
    w = pd.Series(weights, index=df.columns, dtype=float)
    out = pd.Series(index=df.index, dtype=float)
    out.iloc[0] = 1.0
    nav = 1.0
    for i in range(1, len(df)):
        row = rets.iloc[i]
        valid = row.dropna()
        if valid.empty:
            out.iloc[i] = np.nan
            continue
        ww = w.loc[valid.index]
        ww = ww / ww.sum()
        nav *= float((valid * ww).sum() + 1.0)
        out.iloc[i] = nav
    return out.ffill()


def near_12m_high(series: pd.Series, threshold: float = 0.05) -> pd.Series:
    high = series.rolling(12, min_periods=12).max()
    return series >= high * (1.0 - threshold)


def ma_stage(series: pd.Series) -> pd.Series:
    ma60 = series.rolling(60, min_periods=60).mean()
    ma120 = series.rolling(120, min_periods=120).mean()
    stage = pd.Series(0, index=series.index, dtype=int)
    stage = stage.mask(series <= ma60, 1)
    stage = stage.mask(series <= ma60 * 0.85, 2)
    stage = stage.mask((series <= ma120) | (series <= ma60 * 0.75), 3)
    return stage.fillna(0).astype(int)


def drawdown_stage(series: pd.Series, thresholds: tuple[float, float, float]) -> pd.Series:
    peak = series.cummax()
    dd = series / peak - 1.0
    t1, t2, t3 = thresholds
    stage = pd.Series(0, index=series.index, dtype=int)
    stage = stage.mask(dd <= -abs(t1), 1)
    stage = stage.mask(dd <= -abs(t2), 2)
    stage = stage.mask(dd <= -abs(t3), 3)
    return stage.astype(int)


def annual5_live_stage(series: pd.Series) -> pd.Series:
    yearly_close = series.resample("YE").last()
    stage = pd.Series(0, index=series.index, dtype=int)
    for dt, px in series.items():
        completed = yearly_close[yearly_close.index.year < dt.year].tail(4)
        if len(completed) < 4 or not np.isfinite(px):
            continue
        live_ma = pd.concat([completed, pd.Series([px], index=[dt])]).mean()
        if px <= live_ma * 1.05:
            stage.loc[dt] = 1
        if px <= live_ma:
            stage.loc[dt] = 2
        if px <= live_ma * 0.9:
            stage.loc[dt] = 3
    return stage


def ma_reclaim_stage(series: pd.Series) -> pd.Series:
    ma60 = series.rolling(60, min_periods=60).mean()
    down_stage = ma_stage(series)
    out = pd.Series(0, index=series.index, dtype=int)
    seen = 0
    active = 0
    prev_below = False
    for dt, px in series.items():
        current_ma = ma60.loc[dt]
        if not np.isfinite(px) or not np.isfinite(current_ma):
            continue
        below = bool(px <= current_ma)
        if below:
            seen = max(seen, int(down_stage.loc[dt]))
            active = 0
        elif prev_below and seen > 0:
            active = seen
            seen = 0
        out.loc[dt] = active
        prev_below = below
    return out


def fear_stages(series: pd.Series, asset_group: str, rule: str) -> pd.Series:
    if asset_group == "kr":
        dd = drawdown_stage(series, (0.35, 0.45, 0.55))
    else:
        dd = drawdown_stage(series, (0.25, 0.35, 0.45))
    ma = ma_stage(series)

    if rule == "ma_down":
        return ma
    if rule == "drawdown":
        return dd
    if rule == "or":
        return pd.concat([ma, dd], axis=1).max(axis=1).astype(int)
    if rule == "and":
        return pd.concat([ma, dd], axis=1).min(axis=1).astype(int)
    if rule == "annual5_live":
        return annual5_live_stage(series) if asset_group == "us" else ma
    if rule == "ma_reclaim":
        return ma_reclaim_stage(series)
    raise ValueError(f"Unknown fear rule: {rule}")


def calc_on_and_stage(data: pd.DataFrame, signal_col: str, asset_group: str, rule: str) -> tuple[pd.Series, pd.Series]:
    on = near_12m_high(data[signal_col]).reindex(data.index).fillna(False)
    stage = fear_stages(data[signal_col], asset_group, rule).reindex(data.index).fillna(0)
    return on, stage


def simulate(series: pd.DataFrame, cfg: BacktestConfig, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    kr_signal_cols = ["Samsung", "Hynix"] if cfg.kr_mode == "split" else [cfg.kr_signal]
    cols = list(dict.fromkeys([cfg.kr_exec, *kr_signal_cols, cfg.us_exec, cfg.us_signal]))
    defense_col = "Defense_Basket" if cfg.defense == "basket" else "Cash"
    cols.append(defense_col)
    data = series[cols].dropna().loc[pd.Timestamp(start) :]
    if len(data) < 36:
        raise RuntimeError(f"Not enough data for {cfg.name}.")

    returns = data.pct_change().shift(-1)
    us_on, us_raw_stage = calc_on_and_stage(data, cfg.us_signal, "us", cfg.fear_rule)
    if cfg.kr_mode == "split":
        samsung_on, samsung_raw_stage = calc_on_and_stage(data, "Samsung", "kr", cfg.fear_rule)
        hynix_on, hynix_raw_stage = calc_on_and_stage(data, "Hynix", "kr", cfg.fear_rule)
        kr_on = samsung_on & hynix_on
        kr_raw_stage = pd.concat([samsung_raw_stage, hynix_raw_stage], axis=1).max(axis=1)
    else:
        kr_on, kr_raw_stage = calc_on_and_stage(data, cfg.kr_signal, "kr", cfg.fear_rule)

    nav = INITIAL_NAV
    rows = []
    weights = []
    kr_state = 0
    samsung_state = 0
    hynix_state = 0
    us_state = 0

    for dt in data.index[:-1]:
        if cfg.kr_mode == "split":
            if cfg.fear_rule == "ma_reclaim":
                samsung_state = int(samsung_raw_stage.loc[dt])
                hynix_state = int(hynix_raw_stage.loc[dt])
            else:
                if bool(samsung_on.loc[dt]):
                    samsung_state = 0
                elif int(samsung_raw_stage.loc[dt]) > samsung_state:
                    samsung_state = int(samsung_raw_stage.loc[dt])
                if bool(hynix_on.loc[dt]):
                    hynix_state = 0
                elif int(hynix_raw_stage.loc[dt]) > hynix_state:
                    hynix_state = int(hynix_raw_stage.loc[dt])
            kr_state = max(samsung_state, hynix_state)
        elif cfg.fear_rule == "ma_reclaim":
            kr_state = int(kr_raw_stage.loc[dt])
        elif bool(kr_on.loc[dt]):
            kr_state = 0
        elif int(kr_raw_stage.loc[dt]) > kr_state:
            kr_state = int(kr_raw_stage.loc[dt])

        if cfg.fear_rule == "ma_reclaim":
            us_state = int(us_raw_stage.loc[dt])
        elif bool(us_on.loc[dt]):
            us_state = 0
        elif int(us_raw_stage.loc[dt]) > us_state:
            us_state = int(us_raw_stage.loc[dt])

        if cfg.kr_mode == "split":
            samsung_w = 0.10 if bool(samsung_on.loc[dt]) or samsung_state >= 1 else 0.0
            hynix_w = 0.10 if bool(hynix_on.loc[dt]) or hynix_state >= 1 else 0.0
            if samsung_state >= 2:
                samsung_w += 0.0625
            if samsung_state >= 3:
                samsung_w += 0.0625
            if hynix_state >= 2:
                hynix_w += 0.0625
            if hynix_state >= 3:
                hynix_w += 0.0625
            kr_w = samsung_w + hynix_w
        else:
            kr_w = 0.20 if bool(kr_on.loc[dt]) or kr_state >= 1 else 0.0
            if kr_state >= 2:
                kr_w += 0.125
            if kr_state >= 3:
                kr_w += 0.125

        us_w = 0.20 if bool(us_on.loc[dt]) or us_state >= 1 else 0.0
        if us_state >= 2:
            us_w += 0.125
        if us_state >= 3:
            us_w += 0.125

        defense_w = max(0.0, 1.0 - kr_w - us_w)
        r = (
            us_w * float(returns.loc[dt, cfg.us_exec])
            + defense_w * float(returns.loc[dt, defense_col])
        )
        if cfg.kr_mode == "split":
            r += samsung_w * float(returns.loc[dt, "Samsung"])
            r += hynix_w * float(returns.loc[dt, "Hynix"])
        else:
            r += kr_w * float(returns.loc[dt, cfg.kr_exec])
        if not np.isfinite(r):
            r = 0.0
        nav *= 1.0 + r
        next_dt = data.index[data.index.get_loc(dt) + 1]
        rows.append({"date": next_dt, "nav": nav, "return": r})
        weights.append(
            {
                "date": dt,
                "kr_weight": kr_w,
                "samsung_weight": samsung_w if cfg.kr_mode == "split" else np.nan,
                "hynix_weight": hynix_w if cfg.kr_mode == "split" else np.nan,
                "us_weight": us_w,
                "defense_weight": defense_w,
                "kr_on": bool(kr_on.loc[dt]),
                "us_on": bool(us_on.loc[dt]),
                "kr_state": kr_state,
                "samsung_state": samsung_state if cfg.kr_mode == "split" else np.nan,
                "hynix_state": hynix_state if cfg.kr_mode == "split" else np.nan,
                "us_state": us_state,
                "kr_raw_stage": int(kr_raw_stage.loc[dt]),
                "us_raw_stage": int(us_raw_stage.loc[dt]),
            }
        )

    nav_df = pd.DataFrame(rows).set_index("date")
    nav_df["running_max"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = nav_df["nav"] / nav_df["running_max"] - 1.0
    weight_df = pd.DataFrame(weights).set_index("date")
    return nav_df, weight_df


def buy_and_hold(series: pd.DataFrame, weights: dict[str, float], start: str) -> pd.DataFrame:
    data = series[list(weights)].dropna().loc[pd.Timestamp(start) :]
    returns = data.pct_change().shift(-1)
    nav = INITIAL_NAV
    rows = []
    for dt in data.index[:-1]:
        r = sum(float(returns.loc[dt, col]) * w for col, w in weights.items())
        nav *= 1.0 + r
        rows.append({"date": data.index[data.index.get_loc(dt) + 1], "nav": nav, "return": r})
    out = pd.DataFrame(rows).set_index("date")
    out["running_max"] = out["nav"].cummax()
    out["drawdown"] = out["nav"] / out["running_max"] - 1.0
    return out


def metrics(nav: pd.DataFrame) -> dict[str, float]:
    if nav.empty:
        return {}
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    total = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0
    cagr = (nav["nav"].iloc[-1] / nav["nav"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    mdd = nav["drawdown"].min()
    monthly = nav["return"].dropna()
    vol = monthly.std(ddof=0) * np.sqrt(12)
    sharpe = (monthly.mean() * 12) / vol if vol and np.isfinite(vol) else np.nan
    return {
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "years": years,
        "final_nav": nav["nav"].iloc[-1],
        "total_return": total,
        "cagr": cagr,
        "mdd": mdd,
        "vol": vol,
        "sharpe": sharpe,
    }


def max_after_first_buy_drop(nav: pd.DataFrame, weights: pd.DataFrame, side: str) -> float:
    state_col = f"{side}_state"
    first = weights.index[weights[state_col] >= 1]
    if len(first) == 0:
        return np.nan
    start = first[0]
    later = nav.loc[nav.index > start]
    if later.empty:
        return np.nan
    base = later["nav"].iloc[0]
    return later["nav"].min() / base - 1.0


def make_configs() -> list[BacktestConfig]:
    configs = []
    kr_execs = {
        "KOSPI200": "KOSPI200",
        "Semi50": "KR_Semi_50_50",
        "Attack": "KR_Attack_25_37_37",
    }
    rules = ["ma_down", "ma_reclaim", "drawdown", "or", "and", "annual5_live"]
    defenses = ["cash", "basket"]
    for kr_name, kr_exec in kr_execs.items():
        for rule in rules:
            for defense in defenses:
                configs.append(
                    BacktestConfig(
                        name=f"{kr_name}/{rule}/{defense}",
                        kr_exec=kr_exec,
                        kr_signal="KOSPI200",
                        us_signal="QQQ_KRW" if rule == "annual5_live" else "TIGER_Nasdaq100",
                        us_exec="TIGER_Nasdaq100",
                        fear_rule=rule,
                        defense=defense,
                    )
                )
    for rule in rules:
        for defense in defenses:
            configs.append(
                BacktestConfig(
                    name=f"SemiSplit/{rule}/{defense}",
                    kr_exec="KR_Semi_50_50",
                    kr_signal="KR_Semi_50_50",
                    us_signal="QQQ_KRW" if rule == "annual5_live" else "TIGER_Nasdaq100",
                    us_exec="TIGER_Nasdaq100",
                    fear_rule=rule,
                    defense=defense,
                    kr_mode="split",
                )
            )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--out-dir", default="docs/fear_backtest")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = download_adj_close(TICKERS, START)
    monthly = to_monthly(daily)
    series = build_series(monthly)

    rows = []
    nav_outputs = {}
    weight_outputs = {}
    for cfg in make_configs():
        try:
            nav, weights = simulate(series, cfg, args.start)
        except Exception as exc:
            print(f"skip {cfg.name}: {exc}")
            continue
        m = metrics(nav)
        m.update(
            {
                "name": cfg.name,
                "kr_exec": cfg.kr_exec,
                "rule": cfg.fear_rule,
                "defense": cfg.defense,
                "kr_mode": cfg.kr_mode,
                "kr_first_buy_drop": max_after_first_buy_drop(nav, weights, "kr"),
                "us_first_buy_drop": max_after_first_buy_drop(nav, weights, "us"),
                "kr_stage1_count": int((weights["kr_state"] >= 1).sum()),
                "kr_stage2_count": int((weights["kr_state"] >= 2).sum()),
                "kr_stage3_count": int((weights["kr_state"] >= 3).sum()),
                "us_stage1_count": int((weights["us_state"] >= 1).sum()),
                "us_stage2_count": int((weights["us_state"] >= 2).sum()),
                "us_stage3_count": int((weights["us_state"] >= 3).sum()),
            }
        )
        rows.append(m)
        nav_outputs[cfg.name] = nav["nav"].rename(cfg.name)
        weight_outputs[cfg.name] = weights

    summary = pd.DataFrame(rows).sort_values("cagr", ascending=False)

    bh_60_40 = buy_and_hold(
        series,
        {"KR_Attack_25_37_37": 0.20, "TIGER_Nasdaq100": 0.20, "Defense_Basket": 0.60},
        args.start,
    )
    bh_stock = buy_and_hold(
        series,
        {"KR_Attack_25_37_37": 0.50, "TIGER_Nasdaq100": 0.50},
        args.start,
    )
    baselines = pd.DataFrame(
        [
            {"name": "Baseline 20KR/20US/60DefenseBasket", **metrics(bh_60_40)},
            {"name": "Baseline 50KR/50US StockOnly", **metrics(bh_stock)},
        ]
    )

    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(out_dir / "baselines.csv", index=False, encoding="utf-8-sig")
    if nav_outputs:
        pd.concat(nav_outputs.values(), axis=1).to_csv(out_dir / "nav.csv", encoding="utf-8-sig")
    focus_name = "Semi50/or/cash"
    if focus_name in weight_outputs:
        focus = weight_outputs[focus_name].copy()
        focus["stage_change"] = focus["kr_state"].diff().fillna(focus["kr_state"])
        focus_events = focus[
            (focus["kr_state"] >= 1)
            & (
                (focus["stage_change"] > 0)
                | (focus["kr_state"].ne(focus["kr_state"].shift(1)))
            )
        ].copy()
        focus_events.to_csv(out_dir / "semi50_or_cash_events.csv", encoding="utf-8-sig")
    print(f"Wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
