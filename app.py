import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import pytz
import requests
import hashlib
import os
import re
import json
from pathlib import Path
from typing import Any

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import io

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


@st.cache_data(ttl=60)
def get_realtime_gold_krw():
    """GC=F(금 선물) × USDKRW=X 실시간(15분 지연) 금 원화 환산가 반환.
    GLD 스케일(1/10 oz)에 맞게 GC=F를 GLD 직전 종가 비율로 보정.
    실패 시 (None, None, None) 반환."""
    if not _YF_AVAILABLE:
        return None, None, None
    try:
        gc = yf.Ticker("GC=F")
        gld = yf.Ticker("GLD")
        fx = yf.Ticker("USDKRW=X")

        # 2일치 일봉: GC=F와 GLD 직전 종가 비율 계산용
        gc_daily = gc.history(period="5d", interval="1d")
        gld_daily = gld.history(period="5d", interval="1d")
        fx_hist = fx.history(period="1d", interval="1m")

        if gc_daily.empty or gld_daily.empty or fx_hist.empty:
            return None, None, None

        # GLD/GC=F 비율 (전일 종가 기준 보정계수)
        # 데이터가 2행 이상이면 직전 종가(-2), 1행뿐이면 마지막(-1) 사용
        gc_idx = -2 if len(gc_daily) >= 2 else -1
        gld_idx = -2 if len(gld_daily) >= 2 else -1
        gc_close = float(gc_daily["Close"].iloc[gc_idx])
        gld_close = float(gld_daily["Close"].iloc[gld_idx])
        gld_gc_ratio = gld_close / gc_close  # 보통 ~0.092

        # GC=F 실시간 (1분봉) - 별도 호출 없이 gc_daily 1분봉 재사용
        gc_rt_hist = gc.history(period="1d", interval="1m")
        gc_rt = float(gc_rt_hist["Close"].iloc[-1]) if not gc_rt_hist.empty else gc_close

        fx_price = float(fx_hist["Close"].iloc[-1])

        # GLD 스케일 환산: GC=F_실시간 × (GLD/GC 비율) × 환율
        gld_equiv = gc_rt * gld_gc_ratio
        gold_krw = gld_equiv * fx_price

        return gld_equiv, fx_price, gold_krw
    except Exception:
        return None, None, None


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).replace(",", "").strip()
    if s == "" or s == "-":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _read_fdr_with_fallback(ticker, start_date, end_date):
    """FDR 티커 조회(알파벳+숫자 6자리 코드는 .KS 자동 fallback)."""
    candidates = [ticker]
    if (
        isinstance(ticker, str)
        and "." not in ticker
        and len(ticker) == 6
        and any(ch.isalpha() for ch in ticker)
        and any(ch.isdigit() for ch in ticker)
    ):
        candidates.append(f"{ticker}.KS")

    for symbol in candidates:
        try:
            q_end = end_date
            if isinstance(symbol, str) and symbol.endswith(".KS"):
                # fdr→Yahoo는 end를 배타적으로 처리해 마지막 요청일을 빠뜨린다.
                # 알파벳 포함 KRX 티커(0193G0·0064K0·0015B0 등)는 .KS(야후)로만 받으므로
                # +1일 해서 최신 거래일을 포함시킨다. (_read_yfinance_history와 동일한 보정)
                q_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            df = fdr.DataReader(symbol, start_date, q_end)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            continue
    return None


def _standardize_price_df(raw):
    """Convert an external OHLCV frame into the app's Close/Adj Close shape."""
    if raw is None or raw.empty:
        return None

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.names) >= 2 and "Price" in df.columns.names:
            df = df.droplevel(1, axis=1)
        else:
            df.columns = [
                next((part for part in col if part in ("Adj Close", "Close", "Open", "High", "Low", "Volume")), col[-1])
                if isinstance(col, tuple) else col
                for col in df.columns
            ]

    df = df[~df.index.duplicated(keep="last")].sort_index()
    if "Close" not in df.columns:
        return None

    out = pd.DataFrame(index=pd.to_datetime(df.index).tz_localize(None))
    out["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if "Adj Close" in df.columns:
        adj_close = pd.to_numeric(df["Adj Close"], errors="coerce")
        # Some Korean yfinance histories contain impossible non-positive adjusted
        # prices in old corporate-action periods. Keep those dates tradable by
        # falling back to the raw close only where the adjusted value is invalid.
        out["Adj Close"] = adj_close.where(adj_close > 0, out["Close"])
    else:
        out["Adj Close"] = out["Close"]
    out = out.dropna(subset=["Close"])
    return out if not out.empty else None


def _read_yfinance_history(ticker, start_date, end_date):
    """Read long US-market histories when FDR is truncated or unavailable."""
    if not _YF_AVAILABLE:
        return None
    try:
        start_ts = pd.Timestamp(start_date)
        # yfinance treats end as exclusive; include the requested final day.
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        raw = yf.download(
            ticker,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            timeout=20,
        )
        return _standardize_price_df(raw)
    except Exception:
        return None


def _read_us_market_data(ticker, start_date, end_date, prefer_long_history=True):
    """US ETF/futures reader that keeps FDR's data when complete, else uses yfinance."""
    fdr_df = None
    try:
        fdr_df = _standardize_price_df(fdr.DataReader(ticker, start_date, end_date))
    except Exception:
        fdr_df = None

    yf_df = None
    if prefer_long_history:
        requested_start = pd.Timestamp(start_date)
        fdr_start = fdr_df.index.min() if fdr_df is not None and not fdr_df.empty else None
        if fdr_start is None or fdr_start > requested_start + pd.Timedelta(days=45):
            yf_df = _read_yfinance_history(ticker, start_date, end_date)

    candidates = [df for df in (fdr_df, yf_df) if df is not None and not df.empty]
    if not candidates:
        return None
    return min(candidates, key=lambda df: df.index.min())


def _read_kr_market_data(ticker, start_date, end_date, prefer_long_history=True):
    """Korean security reader with Yahoo .KS fallback for older histories."""
    fdr_df = _standardize_price_df(_read_fdr_with_fallback(ticker, start_date, end_date))
    yf_df = None
    if (
        prefer_long_history
        and isinstance(ticker, str)
        and "." not in ticker
        and len(ticker) == 6
        and ticker.isdigit()
    ):
        requested_start = pd.Timestamp(start_date)
        fdr_start = fdr_df.index.min() if fdr_df is not None and not fdr_df.empty else None
        if fdr_start is None or fdr_start > requested_start + pd.Timedelta(days=45):
            yf_df = _read_yfinance_history(f"{ticker}.KS", start_date, end_date)

    candidates = [df for df in (fdr_df, yf_df) if df is not None and not df.empty]
    if not candidates:
        return None
    return min(candidates, key=lambda df: df.index.min())


def _fred_values_to_series(raw, start_ts, end_ts):
    if raw is None or raw.empty or len(raw.columns) < 2:
        return None
    date_col = raw.columns[0]
    value_col = raw.columns[1]
    dt = pd.to_datetime(raw[date_col], errors="coerce")
    vals = pd.to_numeric(raw[value_col], errors="coerce")
    s = pd.Series(vals.values, index=dt).dropna()
    s = s[(s.index >= start_ts) & (s.index <= end_ts)]
    if s.empty:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)


@st.cache_data(ttl=86400)
def _fetch_fred_series_csv(series_id, start_date, end_date):
    """FRED CSV 공개 엔드포인트로 시계열을 조회한다 (API 키 불필요)."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    headers = {"User-Agent": "Mozilla/5.0 faber-dashboard/1.0"}
    urls = [
        (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&cosd={start_ts.date()}&coed={end_ts.date()}"
        ),
        (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&nd=1900-01-01&cosd={start_ts.date()}&coed={end_ts.date()}"
            f"&revision_date={end_ts.date()}&vintage_date={end_ts.date()}"
        ),
    ]

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=6)
            if response.status_code != 200 or not response.text.strip():
                continue
            raw = pd.read_csv(io.StringIO(response.text))
            s = _fred_values_to_series(raw, start_ts, end_ts)
            if s is not None and not s.empty:
                return s
        except Exception:
            continue

    try:
        response = requests.get(
            f"https://fred.stlouisfed.org/data/{series_id}",
            headers=headers,
            timeout=6,
        )
        if response.status_code != 200:
            return None
        rows = []
        for line in response.text.splitlines():
            match = re.match(r"^\s*(\d{4}-\d{2}-\d{2})\s+([-+]?\d+(?:\.\d+)?)\s*$", line)
            if match:
                rows.append((match.group(1), match.group(2)))
        if not rows:
            return None
        raw = pd.DataFrame(rows, columns=["DATE", series_id])
        return _fred_values_to_series(raw, start_ts, end_ts)
    except Exception:
        return None


@st.cache_data(ttl=86400)
def _fetch_fred_series(series_id, start_date, end_date):
    """FRED 시계열 조회: API 키가 있으면 fredapi, 없거나 실패하면 CSV fallback."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # 1) fredapi (키가 있을 때)
    try:
        from fredapi import Fred

        fred_key = None
        try:
            fred_key = st.secrets["FRED_API_KEY"]
        except Exception:
            fred_key = None

        if fred_key:
            fred = Fred(api_key=fred_key)
            s = fred.get_series(series_id, observation_start=start_ts, observation_end=end_ts)
            if s is not None and len(s) > 0:
                s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
                s = s[~s.index.duplicated(keep="last")].sort_index()
                s = s[(s.index >= start_ts) & (s.index <= end_ts)]
                if len(s) > 0:
                    return s.astype(float)
    except Exception:
        pass

    # 2) 공개 CSV fallback (키 불필요)
    return _fetch_fred_series_csv(series_id, start_ts, end_ts)


@st.cache_data(ttl=86400)
def _fetch_oecd_finmarket_series(ref_area, measure, start_date, end_date, freq="M"):
    """Fetch OECD Data Explorer financial-market CSV series without an API key."""
    try:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        start_period = start_ts.strftime("%Y-%m") if freq == "M" else start_ts.strftime("%Y")
        end_period = end_ts.strftime("%Y-%m") if freq == "M" else end_ts.strftime("%Y")
        key = f"{ref_area}.{freq}.{measure}.PA....."
        url = (
            "https://sdmx.oecd.org/public/rest/data/"
            f"OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/{key}"
            f"?startPeriod={start_period}&endPeriod={end_period}&format=csvfile"
        )
        response = requests.get(url, headers={"User-Agent": "faber-dashboard/1.0"}, timeout=20)
        if response.status_code != 200 or not response.text.strip():
            return None
        raw = pd.read_csv(io.StringIO(response.text))
        if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
            return None
        dates = pd.to_datetime(raw["TIME_PERIOD"].astype(str) + "-01", errors="coerce")
        values = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
        s = pd.Series(values.values, index=dates).dropna()
        s = s[(s.index >= start_ts) & (s.index <= end_ts)]
        if s.empty:
            return None
        return s[~s.index.duplicated(keep="last")].sort_index().astype(float)
    except Exception:
        return None


def _get_config_secret(*names):
    for name in names:
        try:
            value = st.secrets[name]
            if value:
                return str(value)
        except Exception:
            pass
        value = os.environ.get(name)
        if value:
            return str(value)
    return None


@st.cache_data(ttl=86400)
def _fetch_ecos_daily_series(stat_code, item_code, start_date, end_date):
    """ECOS 일별 시계열 조회. st.secrets 또는 환경변수의 ECOS_API_KEY/BOK_API_KEY를 사용한다."""
    api_key = _get_config_secret("ECOS_API_KEY", "BOK_API_KEY")
    if not api_key:
        return None

    try:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        start_s = start_ts.strftime("%Y%m%d")
        end_s = end_ts.strftime("%Y%m%d")
        url = (
            f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/100000/"
            f"{stat_code}/D/{start_s}/{end_s}/{item_code}"
        )
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        data = res.json()
        payload = data.get("StatisticSearch", {})
        rows = payload.get("row", [])
        if not rows:
            return None

        dates = pd.to_datetime([r.get("TIME") for r in rows], format="%Y%m%d", errors="coerce")
        values = pd.to_numeric([r.get("DATA_VALUE") for r in rows], errors="coerce")
        s = pd.Series(values, index=dates).dropna()
        s = s[(s.index >= start_ts) & (s.index <= end_ts)]
        if s.empty:
            return None
        return s[~s.index.duplicated(keep="last")].sort_index().astype(float)
    except Exception:
        return None


@st.cache_data(ttl=86400)
def fetch_cd91_rate_series(start_date, end_date):
    """ECOS CD(91일) 일별 금리(연 %)를 조회한다."""
    return _fetch_ecos_daily_series("817Y002", "010502000", start_date, end_date)


@st.cache_data(ttl=86400)
def fetch_kr_long_bond_yield_series(start_date, end_date):
    """Korean long-term yield series, preferring ECOS and falling back to FRED."""
    pieces = []

    # ECOS 10Y is the best match when a BOK/ECOS key is configured.
    ecos_10y = _fetch_ecos_daily_series("817Y002", "010210000", start_date, end_date)
    if ecos_10y is not None and len(ecos_10y) > 0:
        pieces.append(ecos_10y.rename("yield"))

    # Before 10Y history is available, 3Y is a rough but useful early proxy.
    ecos_3y = _fetch_ecos_daily_series("817Y002", "010200000", start_date, end_date)
    if ecos_3y is not None and len(ecos_3y) > 0:
        if pieces:
            first_10y = pieces[0].index.min()
            ecos_3y = ecos_3y[ecos_3y.index < first_10y]
        pieces.insert(0, ecos_3y.rename("yield"))

    if not pieces:
        oecd_yield = _fetch_oecd_finmarket_series("KOR", "IRLT", start_date, end_date)
        if oecd_yield is not None and len(oecd_yield) > 0:
            pieces.append(oecd_yield.rename("yield"))

    if not pieces:
        fred_yield = _fetch_fred_series("IRLTLT01KRM156N", start_date, end_date)
        if fred_yield is not None and len(fred_yield) > 0:
            pieces.append(fred_yield.rename("yield"))

    if not pieces:
        return None

    merged = pd.concat(pieces).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged[(merged.index >= pd.Timestamp(start_date)) & (merged.index <= pd.Timestamp(end_date))]
    return merged.astype(float) if len(merged) > 0 else None


def fetch_kr_30y_bond_yield_series(start_date, end_date):
    """실제 국고채 30년 일별 금리(ECOS 817Y002 / 010230000). 2012-09-11~ 가용.
    ECOS 키가 없으면 None → 호출부에서 10년물 기반 합성으로 폴백한다.
    ※ 30년 국고채는 2012년 최초 발행되어 그 이전 구간은 존재하지 않는다."""
    s = _fetch_ecos_daily_series("817Y002", "010230000", start_date, end_date)
    if s is None or len(s) == 0:
        return None
    return s


def build_cash_price_index_from_annual_rates(rates, start_date, end_date, spread=0.0, fallback_annual_rate=0.025):
    """연율 금리 시계열을 영업일 기준 현금 가격지수로 변환한다."""
    dates = pd.bdate_range(start=start_date, end=end_date)
    if len(dates) == 0:
        return None

    if rates is None or len(rates) == 0:
        annual_rates = pd.Series(float(fallback_annual_rate), index=dates)
    else:
        rate_decimal = pd.Series(rates).sort_index().astype(float) / 100.0
        annual_rates = rate_decimal.reindex(dates).ffill().bfill()
        if annual_rates.isna().all():
            annual_rates = pd.Series(float(fallback_annual_rate), index=dates)
        else:
            annual_rates = annual_rates.fillna(float(fallback_annual_rate))
        annual_rates = (annual_rates - float(spread)).clip(lower=0.0)

    prices = [10000.0]
    for i in range(1, len(dates)):
        elapsed_days = max(1, (dates[i] - dates[i - 1]).days)
        annual_rate = float(annual_rates.iloc[i - 1])
        period_ret = (1 + annual_rate) ** (elapsed_days / 365.0) - 1
        prices.append(prices[-1] * (1 + period_ret))

    return pd.DataFrame({"Close": prices, "Adj Close": prices}, index=dates)


@st.cache_data(ttl=86400)
def get_usdkrw_series(start_date, end_date):
    """USD/KRW 환율 시계열.
    - 우선: FRED DEXKOUS (긴 히스토리)
    - 보강: FDR USD/KRW (최근 구간 품질 보강)
    """
    pieces = []
    try:
        fred_fx = _fetch_fred_series("DEXKOUS", start_date, end_date)
        if fred_fx is not None and len(fred_fx) > 0:
            pieces.append(fred_fx.rename("Close"))
    except Exception:
        pass

    try:
        fdr_fx = fdr.DataReader("USD/KRW", start_date, end_date)
        if fdr_fx is not None and not fdr_fx.empty and "Close" in fdr_fx.columns:
            fdr_close = (
                fdr_fx["Close"]
                .astype(float)
                .rename("Close")
                .sort_index()
            )
            fdr_close = fdr_close[~fdr_close.index.duplicated(keep="last")]
            pieces.append(fdr_close)
    except Exception:
        pass

    if not pieces:
        return None

    # 같은 날짜는 뒤에 붙은 소스(FDR)를 우선 사용.
    merged = pd.concat(pieces).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged[(merged.index >= pd.Timestamp(start_date)) & (merged.index <= pd.Timestamp(end_date))]
    merged = merged.dropna()
    if merged.empty:
        return None

    out = pd.DataFrame(index=merged.index)
    out["Close"] = merged.values.astype(float)
    out["Adj Close"] = out["Close"]
    return out


def _to_price_df_from_close_series(close_series):
    if close_series is None:
        return None
    s = pd.to_numeric(pd.Series(close_series), errors="coerce").dropna()
    if s.empty:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame(index=s.index)
    out["Close"] = s.astype(float)
    out["Adj Close"] = out["Close"]
    return out


@st.cache_data(ttl=86400)
def get_fx_series_to_krw(base_ccy, start_date, end_date):
    """base_ccy/KRW 환율 시계열 생성. direct 우선, 실패 시 USD cross 사용."""
    ccy = str(base_ccy).upper().strip()
    if ccy == "USD":
        return get_usdkrw_series(start_date, end_date)

    # 1) direct quote (예: CNY/KRW)
    try:
        direct = fdr.DataReader(f"{ccy}/KRW", start_date, end_date)
        if direct is not None and not direct.empty and "Close" in direct.columns:
            s = direct["Close"].astype(float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            df = _to_price_df_from_close_series(s)
            if df is not None and not df.empty:
                return df
    except Exception:
        pass

    # 2) USD cross: (USD/KRW) / (USD/base_ccy)
    try:
        usdkrw = get_usdkrw_series(start_date, end_date)
        usdbase = fdr.DataReader(f"USD/{ccy}", start_date, end_date)
        if (
            usdkrw is None or usdkrw.empty or
            usdbase is None or usdbase.empty or "Close" not in usdbase.columns
        ):
            return None
        lhs = usdkrw["Close"].astype(float)
        rhs = usdbase["Close"].astype(float)
        merged = pd.concat([lhs.rename("USDKRW"), rhs.rename("USDBASE")], axis=1).ffill().dropna()
        if merged.empty:
            return None
        cross = (merged["USDKRW"] / merged["USDBASE"]).replace([np.inf, -np.inf], np.nan).dropna()
        return _to_price_df_from_close_series(cross)
    except Exception:
        return None


@st.cache_data(ttl=20)
def get_realtime_kodex_gold_active():
    """KODEX 금액티브(0064K0) 실시간 시세 조회. 실패 시 None."""
    url = "https://polling.finance.naver.com/api/realtime/domestic/stock/0064K0"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.naver.com",
    }
    for _ in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            datas = payload.get("datas") or []
            if not datas:
                continue
            d0 = datas[0]
            px = _to_float(d0.get("closePrice"))
            if px is None:
                continue
            return {
                "price": px,
                "open": _to_float(d0.get("openPrice")),
                "high": _to_float(d0.get("highPrice")),
                "low": _to_float(d0.get("lowPrice")),
                "market_status": d0.get("marketStatus"),
                "traded_at": d0.get("localTradedAt"),
            }
        except Exception:
            continue
    return None


def resolve_gold_signal_runtime(current_dt, stable_mode=True, sticky_minutes=120):
    """금 신호용 실시간 가격 소스 결정.
    우선순위:
    1) 0064K0 실시간
    2) (stable_mode일 때만) 최근 0064K0 성공값(sticky_minutes 이내)
    3) (stable_mode일 때만) GC=F × USD/KRW 환산 실시간
    4) 종가 기준
    """
    sticky_key = "_gold_kodex_last_success"
    kodex = get_realtime_kodex_gold_active()
    if kodex and _to_float(kodex.get("price")) and _to_float(kodex.get("price")) > 0:
        px = float(kodex["price"])
        st.session_state[sticky_key] = {
            "price": px,
            "saved_at": current_dt,
            "traded_at": kodex.get("traded_at"),
        }
        return {
            "source": "KODEX_REALTIME",
            "price": px,
            "kodex": kodex,
            "gc": None,
            "fx": None,
            "sticky_age_min": None,
        }

    if stable_mode:
        last = st.session_state.get(sticky_key)
        if isinstance(last, dict):
            saved_at = last.get("saved_at")
            last_px = _to_float(last.get("price"))
            if isinstance(saved_at, datetime) and last_px and last_px > 0:
                age_min = (current_dt - saved_at).total_seconds() / 60.0
                if age_min <= sticky_minutes:
                    return {
                        "source": "KODEX_STICKY",
                        "price": float(last_px),
                        "kodex": {
                            "price": float(last_px),
                            "traded_at": last.get("traded_at"),
                        },
                        "gc": None,
                        "fx": None,
                        "sticky_age_min": age_min,
                    }

        rt_gc, rt_fx, rt_gold_krw = get_realtime_gold_krw()
        if rt_gold_krw and rt_gold_krw > 0:
            return {
                "source": "GC_FX_REALTIME",
                "price": float(rt_gold_krw),
                "kodex": None,
                "gc": rt_gc,
                "fx": rt_fx,
                "sticky_age_min": None,
            }

    return {
        "source": "NONE",
        "price": None,
        "kodex": None,
        "gc": None,
        "fx": None,
        "sticky_age_min": None,
    }
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="MAIN", page_icon="💎", layout="wide")

# ==============================
# 1) 기본 설정값
# ==============================
DEFAULT_INVESTMENT_START_DATE = datetime(2026, 3, 31)
DEFAULT_INITIAL_CAPITAL = 249_008_318  # 3/31 종가 확정 총자산 — 수익률 계산 기준점, 수정 금지
DEFAULT_HISTORICAL_REALIZED_PROFIT = 67_571_303  # (249,008,318 - 226,356,552) + 44,919,537
DEFAULT_BACKTEST_START_DATE = datetime(2000, 1, 1)
BACKTEST_DEFAULT_END_LAG_MONTHS = 1

DEFAULT_GEN_KOSPI_BAL = 150_664_120  # 사이드바 기본값 (수익률 계산 무관)
DEFAULT_GEN_GOLD_BAL  = 533
DEFAULT_ISA_A_BAL     = 79_099_381
DEFAULT_ISA_B_BAL     = 82_647_962
DEFAULT_BALANCE_VERSION = "2026-05-29-final2"
BALANCE_DEFAULTS = [
    ("bal_gen_kospi", DEFAULT_GEN_KOSPI_BAL),
    ("bal_gen_gold", DEFAULT_GEN_GOLD_BAL),
    ("bal_isa_a", DEFAULT_ISA_A_BAL),
    ("bal_isa_b", DEFAULT_ISA_B_BAL),
]
BALANCE_QUERY_KEYS = {
    "bal_gen_kospi": "gen_k",
    "bal_gen_gold": "gen_g",
    "bal_isa_a": "isaa",
    "bal_isa_b": "isab",
}

# 미확정 예정 입금 — NAV 계산에 반영되지 않음. 확정 시 CONFIRMED 으로 이동.
PERSONAL_CASH_FLOWS_PENDING: dict[str, Any] = {
}
# 확정 입금 — NAV 계산에 반영됨. 입금 확정마다 여기에 추가.
PERSONAL_CASH_FLOWS_CONFIRMED = {
    "2026-04-30": 30_000_000,  # 3천만 원 대출금 Faber 추가투입
}
PERSONAL_CASH_FLOWS = PERSONAL_CASH_FLOWS_CONFIRMED  # 계산에 사용되는 것은 확정분만
APP_DIR = Path(__file__).resolve().parent
LIVE_PORTFOLIO_POLICY_PATH = APP_DIR / "config" / "live_portfolio_policy.json"
MACRO_CYCLE_EVIDENCE_PATH = APP_DIR / "docs" / "macro_cycle" / "latest_evidence.json"
MACRO_CYCLE_REPORT_DIR = APP_DIR / "docs" / "macro_cycle"
MONTHLY_LEDGER_PATHS = [
    APP_DIR / "faber-investment-memory" / "monthly_ledger.md",
    APP_DIR / ".codex-private" / "monthly_ledger.md",
]
MONTHLY_LEDGER_CSV_PATHS = [
    APP_DIR / "faber-investment-memory" / "monthly_ledger.csv",
    APP_DIR / ".codex-private" / "monthly_ledger.csv",
]
MONTHLY_LEDGER_PATH = MONTHLY_LEDGER_PATHS[0]
MONTHLY_LEDGER_CSV_PATH = MONTHLY_LEDGER_CSV_PATHS[0]
MONTHLY_LEDGER_COLUMNS = [
    "month",
    "month_start_date",
    "month_start_assets",
    "month_end_date",
    "month_end_assets",
    "deposit",
    "withdrawal",
    "net_external_cash_flow",
    "official_profit",
    "official_return",
]
DEFAULT_MONTHLY_LEDGER = {
    "2026-04": {
        "month": "2026-04",
        "month_start_date": "2026-03-31",
        "month_start_assets": 249_008_318,
        "month_end_date": "2026-04-30",
        "month_end_assets": 283_565_328,
        "deposit": 30_000_000,
        "withdrawal": 0,
        "net_external_cash_flow": 30_000_000,
        "official_profit": 4_557_010,
        "official_return": 0.0183,
    },
    "2026-05": {
        "month": "2026-05",
        "month_start_date": "2026-04-30",
        "month_start_assets": 283_565_328,
        "month_end_date": "2026-05-29",
        "month_end_assets": 312_411_996,
        "deposit": 0,
        "withdrawal": 0,
        "net_external_cash_flow": 0,
        "official_profit": 28_846_668,
        "official_return": 0.1017,
    },
    "2026-06": {
        "month": "2026-06",
        "month_start_date": "2026-05-29",
        "month_start_assets": 312_411_996,
        "month_end_date": "2026-06-30",
        "month_end_assets": 319_352_259,
        "deposit": 0,
        "withdrawal": 0,
        "net_external_cash_flow": 0,
        "official_profit": 6_940_263,
        "official_return": 0.0222,
    },
}


@st.cache_data(show_spinner=False)
def load_live_portfolio_policy(policy_path: str = str(LIVE_PORTFOLIO_POLICY_PATH)):
    path = Path(policy_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        policy = json.load(f)
    policy.setdefault("summary", {})
    policy.setdefault("rules", {})
    policy.setdefault("asset_classes", [])
    policy.setdefault("accounts", [])
    policy.setdefault("part_summary", [])
    policy.setdefault("holdings", [])
    return policy


def _fmt_won(value, signed=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if signed:
        return f"{value:+,.0f}원"
    return f"{value:,.0f}원"


def _fmt_pct(value, digits=1):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{value * 100:.{digits}f}%"


def _fmt_plain_number(value, digits=2):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{value:,.{digits}f}"


def _display_policy_table(rows, column_map, money_cols=None, signed_money_cols=None, pct_cols=None):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns=column_map)
    money_cols = money_cols or []
    signed_money_cols = signed_money_cols or []
    pct_cols = pct_cols or []
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda v: _fmt_won(v))
    for col in signed_money_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda v: _fmt_won(v, signed=True))
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda v: _fmt_pct(v))
    return df


def render_live_portfolio_policy(policy):
    if not policy:
        return

    summary = policy.get("summary", {})
    rules = policy.get("rules", {})
    st.subheader("🧭 변경 포트폴리오 기준")
    st.caption(
        f"{policy.get('name', '포트폴리오 정책')} | 기준일 {policy.get('as_of', '-')} | "
        f"리밸런싱: {rules.get('rebalance', '수시')} | "
        f"매크로: {rules.get('macro_cycle_layer', '유지')}"
    )

    cols = st.columns(4)
    cols[0].metric("스냅샷 총자산", _fmt_won(summary.get("current_total_assets")))
    cols[1].metric("계좌 기준금액", _fmt_won(summary.get("account_basis_total")))
    cols[2].metric("기준 차이", _fmt_won(summary.get("basis_gap"), signed=True))
    cols[3].metric("USD/KRW", f"{float(summary.get('usd_krw', 0)):,.2f}" if summary.get("usd_krw") is not None else "-")

    st.info(
        "이 섹션은 포트폴리오 스프레드시트의 계산값을 대시보드에 반영한 스냅샷입니다. "
        "자동 주문은 하지 않고, 매크로 판단은 기존처럼 별도 참고/오버레이로 유지합니다."
    )

    part_df = _display_policy_table(
        policy.get("part_summary", []),
        {
            "name": "구분",
            "target_amount": "목표금액",
            "target_weight": "목표비중",
            "current_amount": "현재금액",
            "current_weight": "현재비중",
        },
        money_cols=["목표금액", "현재금액"],
        pct_cols=["목표비중", "현재비중"],
    )
    if not part_df.empty:
        st.markdown("##### 큰 구분")
        st.dataframe(part_df, use_container_width=True, hide_index=True)

    asset_df = _display_policy_table(
        policy.get("asset_classes", []),
        {
            "name": "자산군",
            "target_weight": "파트내 목표비중",
            "target_amount": "목표금액",
            "current_amount": "현재금액",
            "current_total_weight": "현재 총자산비중",
            "gap": "차이",
            "scope": "관리",
            "memo": "메모",
        },
        money_cols=["목표금액", "현재금액"],
        signed_money_cols=["차이"],
        pct_cols=["파트내 목표비중", "현재 총자산비중"],
    )
    if not asset_df.empty:
        st.markdown("##### 자산군 목표 대비 현재")
        st.dataframe(asset_df, use_container_width=True, hide_index=True)

    account_df = _display_policy_table(
        policy.get("accounts", []),
        {
            "name": "계좌",
            "basis_amount": "기준금액",
            "part_weight": "파트내비중",
            "current_amount": "현재금액",
            "current_total_weight": "현재 총자산비중",
            "gap": "차이",
            "input": "입력위치",
            "memo": "메모",
        },
        money_cols=["기준금액", "현재금액"],
        signed_money_cols=["차이"],
        pct_cols=["파트내비중", "현재 총자산비중"],
    )
    if not account_df.empty:
        with st.expander("계좌별 스냅샷", expanded=False):
            st.dataframe(account_df, use_container_width=True, hide_index=True)

    holdings_df = _display_policy_table(
        policy.get("holdings", []),
        {
            "account": "계좌",
            "name": "종목명",
            "ticker": "코드/티커",
            "role": "역할",
            "currency": "통화",
            "price": "현재가",
            "quantity": "보유수량",
            "order": "주문수량",
            "current_amount": "현재평가액",
            "current_internal_weight": "현재내부비중",
            "target_internal_weight": "내부목표비중",
            "target_amount": "내부목표금액",
            "gap": "내부차이",
            "memo": "비고",
        },
        money_cols=["현재평가액", "내부목표금액"],
        signed_money_cols=["내부차이"],
        pct_cols=["현재내부비중", "내부목표비중"],
    )
    if not holdings_df.empty:
        with st.expander("종목별 수시 리밸런싱 주문 참고", expanded=True):
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            st.caption("주문수량은 스프레드시트 기준 참고값입니다. 실제 체결 전 가격, 세금, 수수료, 주문 가능 수량을 별도로 확인하세요.")


@st.cache_data(ttl=300)
def load_macro_cycle_evidence(evidence_path: str = str(MACRO_CYCLE_EVIDENCE_PATH)):
    path = Path(evidence_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_named_indicator(rows, *needles):
    normalized_needles = [str(n).lower() for n in needles]
    for row in rows or []:
        haystack = " ".join(
            str(row.get(key, ""))
            for key in ("name", "ticker", "series_id", "pair")
        ).lower()
        if all(needle in haystack for needle in normalized_needles):
            return row
    return None


def _macro_direction_label(direction):
    if direction == "rising":
        return "상승"
    if direction == "falling":
        return "하락"
    return "불명"


def classify_vix_fear_greed(vix_value, percentile_1y=None, direction=None):
    vix = _to_float(vix_value)
    percentile = _to_float(percentile_1y)
    if vix is None:
        return "데이터 없음"
    if vix >= 40:
        return "극단 공포"
    if vix >= 30:
        return "공포"
    if vix >= 22:
        return "주의"
    if vix <= 14 and (percentile is None or percentile <= 0.35):
        return "탐욕/안도"
    if vix <= 18 and direction == "falling":
        return "중립~탐욕"
    return "중립"


def summarize_macro_cycle_evidence(evidence):
    if not evidence:
        return {
            "as_of": "-",
            "macro_label": "근거팩 없음",
            "sentiment_label": "데이터 없음",
            "rows": [],
        }

    leading = evidence.get("fred", {}).get("leading", [])
    coincident = evidence.get("fred", {}).get("coincident", [])
    sentiment = evidence.get("sentiment", [])
    price_assets = evidence.get("price_assets", [])

    pmi = _find_named_indicator(leading, "pmi")
    new_orders = _find_named_indicator(leading, "new orders")
    sp500 = _find_named_indicator(price_assets, "s&p")
    vix = _find_named_indicator(sentiment, "vix")
    hy_oas = _find_named_indicator(sentiment, "high-yield")
    sp_drawdown = _find_named_indicator(sentiment, "drawdown")

    rising_leading = sum(1 for row in leading if row.get("direction_6m") == "rising")
    falling_leading = sum(1 for row in leading if row.get("direction_6m") == "falling")
    rising_coincident = sum(1 for row in coincident if row.get("direction_6m") == "rising")

    if rising_leading >= max(2, falling_leading) and rising_coincident >= 2:
        macro_label = "성장 우세"
    elif falling_leading > rising_leading:
        macro_label = "둔화 경계"
    else:
        macro_label = "전환/혼재"

    sentiment_label = classify_vix_fear_greed(
        vix.get("latest") if vix else None,
        vix.get("percentile_1y") if vix else None,
        vix.get("direction_3m") if vix else None,
    )

    rows = []
    if pmi:
        rows.append({
            "지표": "매크로 사이클: ISM PMI",
            "최근값": _fmt_plain_number(pmi.get("latest"), digits=1),
            "기준일": pmi.get("latest_date", "-"),
            "판정": "확장" if (_to_float(pmi.get("latest")) or 0) >= 50 else "수축",
            "흐름": _macro_direction_label(pmi.get("direction_6m")),
        })
    if new_orders:
        rows.append({
            "지표": "선행 수요: 제조업 신규주문",
            "최근값": _fmt_plain_number(new_orders.get("latest"), digits=0),
            "기준일": new_orders.get("latest_date", "-"),
            "판정": "개선" if new_orders.get("direction_6m") == "rising" else "약화",
            "흐름": _macro_direction_label(new_orders.get("direction_6m")),
        })
    if sp500:
        rows.append({
            "지표": "가격 선행: S&P 500",
            "최근값": _fmt_plain_number(sp500.get("latest"), digits=1),
            "기준일": sp500.get("latest_date", "-"),
            "판정": "200일선 위" if sp500.get("above_200d_ma") is True else "200일선 아래",
            "흐름": _macro_direction_label(sp500.get("direction_12m")),
        })
    if vix:
        rows.append({
            "지표": "VIX",
            "최근값": _fmt_plain_number(vix.get("latest"), digits=2),
            "기준일": vix.get("latest_date", "-"),
            "판정": sentiment_label,
            "흐름": _macro_direction_label(vix.get("direction_3m")),
        })
    if hy_oas:
        rows.append({
            "지표": "High-Yield OAS",
            "최근값": _fmt_plain_number(hy_oas.get("latest"), digits=2),
            "기준일": hy_oas.get("latest_date", "-"),
            "판정": "신용 스트레스 완화" if hy_oas.get("direction_6m") == "falling" else "신용 스트레스 확대",
            "흐름": _macro_direction_label(hy_oas.get("direction_6m")),
        })
    if sp_drawdown:
        rows.append({
            "지표": "S&P 500 고점 대비",
            "최근값": _fmt_pct(sp_drawdown.get("latest"), digits=1),
            "기준일": sp_drawdown.get("latest_date", "-"),
            "판정": "고점권" if (_to_float(sp_drawdown.get("latest")) or -1) > -0.05 else "낙폭 확대",
            "흐름": _macro_direction_label(sp_drawdown.get("direction_3m")),
        })

    return {
        "as_of": evidence.get("as_of_requested") or evidence.get("generated_at", "-"),
        "macro_label": macro_label,
        "sentiment_label": sentiment_label,
        "rows": rows,
    }


def load_latest_macro_cycle_report_excerpt(report_dir: Path = MACRO_CYCLE_REPORT_DIR):
    try:
        reports = sorted(report_dir.glob("report-*.md"))
        if not reports:
            return None, None
        latest = reports[-1]
        lines = latest.read_text(encoding="utf-8").splitlines()
        picked = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            picked.append(stripped)
            if len(picked) >= 4:
                break
        return latest.name, "\n\n".join(picked)
    except Exception:
        return None, None


def render_macro_cycle_monitor(current_date):
    evidence = load_macro_cycle_evidence()
    summary = summarize_macro_cycle_evidence(evidence)

    st.subheader("매크로 사이클 · VIX · 공포/탐욕")
    st.caption("포트폴리오 운영 참고용 오버레이입니다. 기존 리밸런싱 규칙을 자동으로 덮어쓰지 않습니다.")

    cols = st.columns(4)
    cols[0].metric("근거팩 기준일", summary["as_of"])
    cols[1].metric("매크로 상태", summary["macro_label"])
    vix_row = next((row for row in summary["rows"] if row["지표"] == "VIX"), None)
    cols[2].metric("VIX", vix_row["최근값"] if vix_row else "-")
    cols[3].metric("공포/탐욕 프록시", summary["sentiment_label"])

    if summary["rows"]:
        st.dataframe(pd.DataFrame(summary["rows"]), use_container_width=True, hide_index=True)
    else:
        st.info("매크로 근거팩이 아직 없습니다. `scripts/macro_cycle_evidence.py`로 생성한 JSON을 사용합니다.")

    report_name, report_excerpt = load_latest_macro_cycle_report_excerpt()
    if report_excerpt:
        with st.expander(f"최근 매크로 리포트 요약: {report_name}", expanded=False):
            st.markdown(report_excerpt)

    with st.expander("실시간 데이터/API 필요 여부", expanded=False):
        st.markdown(
            "- 매크로 사이클은 대부분 월간/일간 지표라 완전 실시간 API보다 정기 갱신이 중요합니다. 현재 앱은 로컬 근거팩, FRED 공개 CSV, yfinance 경로를 쓸 수 있습니다.\n"
            "- VIX는 `^VIX` 공개 시세로 갱신할 수 있지만 보통 지연/비공식 데이터입니다. 체결급 실시간성이 필요하면 유료 시장데이터 API가 필요합니다.\n"
            "- CNN Fear & Greed 원지수 그대로를 자동 표시하려면 별도 API 또는 안정적인 스크래핑 경로가 필요합니다. 지금 패널은 VIX, 신용스프레드, S&P 500 낙폭으로 만든 공포/탐욕 프록시입니다."
        )

    if st.button("VIX 최신 공개시세 다시 조회", use_container_width=True):
        with st.spinner("VIX 공개 시세를 다시 조회하는 중..."):
            vix_df = fetch_vix_data(pd.Timestamp(current_date) - pd.Timedelta(days=370), current_date)
        if vix_df is None or vix_df.empty:
            st.warning("VIX 데이터를 다시 조회하지 못했습니다. 기존 근거팩 값을 참고하세요.")
        else:
            vix_prices = pd.to_numeric(vix_df["Close"], errors="coerce").dropna()
            latest_vix = float(vix_prices.iloc[-1])
            percentile = float((vix_prices.tail(252) <= latest_vix).mean()) if len(vix_prices.tail(252)) else None
            label = classify_vix_fear_greed(latest_vix, percentile)
            st.success(
                f"최신 공개 VIX: {latest_vix:.2f} | 1년 백분위: "
                f"{percentile * 100:.0f}% | 판정: {label}"
            )


def _normalize_policy_ticker(ticker):
    ticker = str(ticker or "").strip()
    if ":" in ticker:
        ticker = ticker.split(":")[-1]
    return ticker.upper()


def _is_policy_holding_active(holding):
    current_amount = _to_float(holding.get("current_amount"))
    ticker = _normalize_policy_ticker(holding.get("ticker"))
    return bool(ticker) and current_amount is not None and current_amount > 0


@st.cache_data(ttl=3600)
def load_portfolio_holding_price_data(ticker, currency, start_date, end_date):
    ticker = _normalize_policy_ticker(ticker)
    if not ticker:
        return None
    if ticker.isdigit() and len(ticker) == 6:
        return _read_kr_market_data(ticker, start_date, end_date, prefer_long_history=False)
    return _read_us_market_data(ticker, start_date, end_date)


def _portfolio_fx_value(currency, fx_df, as_of_date):
    currency = str(currency or "KRW").upper()
    if currency == "KRW":
        return 1.0
    if currency == "USD":
        return get_price_at_date(fx_df, as_of_date, price_col="Close")
    return None


def build_live_portfolio_monthly_return_rows(policy, current_date, price_col="Adj Close"):
    """Estimate MTD P/L from the spreadsheet's current values and market price change."""
    if not policy:
        return [], []

    current_ts = pd.Timestamp(current_date)
    month_start = current_ts.replace(day=1)
    data_start = month_start - pd.Timedelta(days=10)
    holdings = [h for h in policy.get("holdings", []) if _is_policy_holding_active(h)]
    needs_usd_fx = any(str(h.get("currency", "")).upper() == "USD" for h in holdings)
    fx_df = get_usdkrw_series(data_start, current_ts) if needs_usd_fx else None

    rows = []
    skipped = []
    for holding in holdings:
        ticker = _normalize_policy_ticker(holding.get("ticker"))
        currency = str(holding.get("currency", "KRW")).upper()
        price_df = load_portfolio_holding_price_data(ticker, currency, data_start, current_ts)
        start_price = get_price_at_date(price_df, month_start, price_col=price_col)
        current_price = get_price_at_date(price_df, current_ts, price_col=price_col)
        start_fx = _portfolio_fx_value(currency, fx_df, month_start)
        current_fx = _portfolio_fx_value(currency, fx_df, current_ts)
        current_amount = _to_float(holding.get("current_amount"))

        if (
            start_price is None or current_price is None
            or start_fx is None or current_fx is None
            or current_amount is None or current_amount <= 0
        ):
            skipped.append({
                "계좌": holding.get("account", "-"),
                "종목명": holding.get("name", "-"),
                "코드/티커": ticker or "-",
                "사유": "월초/현재 가격 또는 환율 데이터 부족",
            })
            continue

        start_krw_price = float(start_price) * float(start_fx)
        current_krw_price = float(current_price) * float(current_fx)
        if start_krw_price <= 0 or current_krw_price <= 0:
            continue

        mtd_return = current_krw_price / start_krw_price - 1.0
        estimated_profit = current_amount * (mtd_return / (1.0 + mtd_return))
        estimated_start_amount = current_amount - estimated_profit
        rows.append({
            "계좌": holding.get("account", "-"),
            "종목명": holding.get("name", "-"),
            "코드/티커": ticker,
            "역할": holding.get("role", "-"),
            "통화": currency,
            "보유수량": _to_float(holding.get("quantity")),
            "월초가격(KRW)": start_krw_price,
            "현재가격(KRW)": current_krw_price,
            "월초추정액": estimated_start_amount,
            "현재평가액": current_amount,
            "이번달수익률": mtd_return,
            "이번달추정손익": estimated_profit,
        })

    rows.sort(key=lambda row: abs(row["이번달추정손익"]), reverse=True)
    return rows, skipped


def get_portfolio_month_start_basis(current_date, ledger=None):
    ledger = ledger or load_structured_monthly_ledger()
    current_ts = pd.Timestamp(current_date)
    month_start = current_ts.replace(day=1)
    current_month = current_ts.strftime("%Y-%m")
    current_row = ledger.get(current_month, {})
    current_start_assets = _to_float(current_row.get("month_start_assets"))
    if current_start_assets is not None and current_start_assets > 0:
        return (
            current_start_assets,
            str(current_row.get("month_start_date") or month_start.date()),
            "이번 달 원장 시작 총자산",
        )

    prev_month_key = (month_start - pd.Timedelta(days=1)).strftime("%Y-%m")
    prev_row = ledger.get(prev_month_key, {})
    prev_end_assets = _to_float(prev_row.get("month_end_assets"))
    if prev_end_assets is not None and prev_end_assets > 0:
        return (
            prev_end_assets,
            str(prev_row.get("month_end_date") or (month_start - pd.Timedelta(days=1)).date()),
            "전월 원장 확정 총자산",
        )

    summary_assets = _to_float((load_live_portfolio_policy() or {}).get("summary", {}).get("account_basis_total"))
    if summary_assets is not None and summary_assets > 0:
        return summary_assets, str(month_start.date()), "포트폴리오 스냅샷 계좌 기준금액"
    return 0.0, str(month_start.date()), "기준 총자산 없음"


def build_monthly_ledger_record(month, start_date, start_assets, end_date, end_assets, deposit, withdrawal):
    start_assets = float(start_assets or 0)
    end_assets = float(end_assets or 0)
    deposit = float(deposit or 0)
    withdrawal = float(withdrawal or 0)
    net_external_cash_flow = deposit - withdrawal
    official_profit = end_assets - start_assets - net_external_cash_flow
    official_return = official_profit / start_assets if start_assets > 0 else None
    return {
        "month": month,
        "month_start_date": str(start_date),
        "month_start_assets": round(start_assets),
        "month_end_date": str(end_date),
        "month_end_assets": round(end_assets),
        "deposit": round(deposit),
        "withdrawal": round(withdrawal),
        "net_external_cash_flow": round(net_external_cash_flow),
        "official_profit": round(official_profit),
        "official_return": official_return,
    }


def get_monthly_ledger_write_path():
    for path in MONTHLY_LEDGER_CSV_PATHS:
        path = Path(path)
        if path.exists():
            return path
    for path in MONTHLY_LEDGER_CSV_PATHS:
        path = Path(path)
        if path.parent.exists():
            return path
    return Path(MONTHLY_LEDGER_CSV_PATH)


def upsert_monthly_ledger_record(record, ledger_path=None):
    path = Path(ledger_path) if ledger_path else get_monthly_ledger_write_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path, dtype={"month": str})
    else:
        df = pd.DataFrame(columns=MONTHLY_LEDGER_COLUMNS)

    for col in MONTHLY_LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[df["month"].astype(str) != str(record["month"])]
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df = df[MONTHLY_LEDGER_COLUMNS].sort_values("month")
    df.to_csv(path, index=False, encoding="utf-8")
    load_structured_monthly_ledger.clear()
    load_confirmed_month_end_navs.clear()
    return path


def render_live_portfolio_monthly_returns(policy, current_date, price_col):
    st.subheader("이번달 포트폴리오 자산 수익")
    rows, skipped = build_live_portfolio_monthly_return_rows(policy, current_date, price_col=price_col)
    if not rows:
        st.info("현재 보유 중인 종목의 월간 수익을 계산할 가격 데이터가 아직 부족합니다.")
        if skipped:
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        return

    total_current = sum(row["현재평가액"] for row in rows)
    total_profit = sum(row["이번달추정손익"] for row in rows)
    total_start = total_current - total_profit
    total_return = total_profit / total_start if total_start > 0 else None

    cols = st.columns(4)
    cols[0].metric("집계 현재평가액", _fmt_won(total_current))
    cols[1].metric("이번달 추정손익", _fmt_won(total_profit, signed=True))
    cols[2].metric("이번달 추정수익률", _fmt_pct(total_return, digits=2) if total_return is not None else "-")
    cols[3].metric("집계 종목 수", f"{len(rows)}개")

    display_rows = []
    for row in rows:
        display_rows.append({
            "계좌": row["계좌"],
            "종목명": row["종목명"],
            "코드/티커": row["코드/티커"],
            "역할": row["역할"],
            "현재평가액": _fmt_won(row["현재평가액"]),
            "월초추정액": _fmt_won(row["월초추정액"]),
            "이번달수익률": _fmt_pct(row["이번달수익률"], digits=2),
            "이번달추정손익": _fmt_won(row["이번달추정손익"], signed=True),
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    st.caption(
        "현재평가액은 포트폴리오 스프레드시트 스냅샷을 기준으로 두고, 월초 대비 가격 변화율로 이번달 손익을 역산한 추정치입니다. "
        "월중 매수/매도와 세금/수수료는 공식 손익 기록에서 총자산 기준으로 별도 확인하세요."
    )

    if skipped:
        with st.expander("가격 데이터가 부족해 제외된 항목", expanded=False):
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)


def build_faber_a_monthly_reference_rows(current_date, price_col="Adj Close"):
    current_ts = pd.Timestamp(current_date)
    month_start = current_ts.replace(day=1)
    data_start = month_start - relativedelta(months=18)
    all_data = load_market_data(data_start, current_ts, hybrid=True)
    if all_data is None:
        return [], [], None, None, 0.0, str(month_start.date()), "시장 데이터 없음"

    trading_dates = build_trading_calendar(all_data, data_start, current_ts)
    if not trading_dates:
        return [], [], None, all_data, 0.0, str(month_start.date()), "거래일 데이터 없음"

    prev_dates = [d for d in trading_dates if pd.Timestamp(d) < month_start]
    rebal_date = prev_dates[-1] if prev_dates else trading_dates[0]
    basis_assets, basis_date, basis_source = get_portfolio_month_start_basis(current_ts)
    rebal_weights = calculate_faber_weights(rebal_date, all_data, mode='A', price_col=price_col)

    rows = []
    skipped = []
    for asset_name, weight in rebal_weights.items():
        if weight < 0.001:
            continue

        alloc_won = basis_assets * weight
        if asset_name == CASH_NAME:
            px_s = get_price_at_date(all_data.get(CASH_NAME), rebal_date, price_col=price_col) or 10000.0
            px_e = get_price_at_date(all_data.get(CASH_NAME), current_ts, price_col=price_col) or px_s
        else:
            px_s = get_price_at_date(all_data.get(asset_name), rebal_date, price_col=price_col)
            px_e = get_price_at_date(all_data.get(asset_name), current_ts, price_col=price_col)

        if not px_s or px_s <= 0 or not px_e or px_e <= 0:
            skipped.append({
                "자산": asset_name,
                "비중": f"{weight*100:.0f}%",
                "사유": "리밸런싱일/현재 가격 데이터 부족",
            })
            continue

        ret = (px_e / px_s) - 1.0
        pnl = alloc_won * ret
        rows.append({
            "자산": asset_name,
            "비중": weight,
            "배분금액": alloc_won,
            "기준가(리밸)": px_s,
            "현재가": px_e,
            "수익률": ret,
            "손익(원)": pnl,
        })

    order = {'코스피200': 0, '미국나스닥100': 1, '한국채30년': 2, '미국채30년': 3, '금현물': 4, CASH_NAME: 5}
    rows.sort(key=lambda row: order.get(row["자산"], 99))
    return rows, skipped, rebal_date, all_data, basis_assets, basis_date, basis_source


def render_faber_a_monthly_reference(current_date, current_total_assets, price_col):
    st.subheader(f"이번 달 성과 ({pd.Timestamp(current_date).strftime('%Y년 %m월')})")
    st.markdown("##### 공식 성과: 실제 계좌 기준")
    rows, skipped, rebal_date, _, basis_assets, basis_date, basis_source = build_faber_a_monthly_reference_rows(
        current_date, price_col=price_col
    )
    if not rows:
        st.info("원조 Faber A 월간 참고 성과를 계산할 가격 데이터가 아직 부족합니다.")
        if skipped:
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        return

    basis_dt = pd.Timestamp(basis_date)
    net_month_cash_flow = calculate_period_cash_flow(PERSONAL_CASH_FLOWS, basis_dt, pd.Timestamp(current_date))
    official_profit = current_total_assets - basis_assets - net_month_cash_flow
    official_return = official_profit / basis_assets if basis_assets > 0 else None
    total_pnl = sum(row["손익(원)"] for row in rows)
    total_return = total_pnl / basis_assets if basis_assets > 0 else None

    official_cols = st.columns(4)
    official_cols[0].metric("기준 총자산", _fmt_won(basis_assets))
    official_cols[1].metric("순외부현금흐름", _fmt_won(net_month_cash_flow, signed=True))
    official_cols[2].metric("현재 운용자산", _fmt_won(current_total_assets))
    official_cols[3].metric(
        "공식 손익",
        _fmt_won(official_profit, signed=True),
        delta=_fmt_pct(official_return, digits=2) if official_return is not None else None,
    )
    st.caption(
        f"공식 성과 = 현재 운용자산 - {basis_date} 기준 총자산 - 순외부현금흐름. "
        f"기준금액 출처: {basis_source}. 아래 자산별 카드는 원조 Faber A(코스피/나스닥 패시브, -5%룰) 신호의 참고 NAV입니다."
    )

    st.markdown("##### 참고: 자산별 가격변동 추정")
    cols_m = st.columns(len(rows) + 1)
    for i, row in enumerate(rows):
        cols_m[i].metric(
            label=row["자산"],
            value=f"{row['수익률']:+.2%}",
            delta=_fmt_won(row["손익(원)"], signed=True),
        )
    cols_m[-1].metric(
        label="📊 참고 합계",
        value=_fmt_won(total_pnl, signed=True),
        delta=_fmt_pct(total_return, digits=2) if total_return is not None else "N/A",
    )

    with st.expander("📋 이번 달 자산별 상세"):
        detail_df = pd.DataFrame([{
            "자산": row["자산"],
            "비중": f"{row['비중']*100:.0f}%",
            "배분금액": _fmt_won(row["배분금액"]),
            "기준가(리밸)": f"{row['기준가(리밸)']:,.2f}",
            "현재가": f"{row['현재가']:,.2f}",
            "수익률": f"{row['수익률']:+.2%}",
            "손익(원)": _fmt_won(row["손익(원)"], signed=True),
        } for row in rows])
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    if skipped:
        with st.expander("가격 데이터가 부족해 제외된 Faber A 슬롯", expanded=False):
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
    if rebal_date is not None:
        st.caption(
            f"※ 기준: {pd.Timestamp(rebal_date).strftime('%Y-%m-%d')} 리밸런싱 당시 Faber A 비중. "
            "각 패시브 자산이 12개월 고점 대비 -5% 이내면 20% ON, 아니면 현금(MMF) 대기."
        )


def render_monthly_profit_recorder(current_date, current_total_assets):
    st.subheader("이번달 공식 수익 기록")
    ledger = load_structured_monthly_ledger()
    month_key = pd.Timestamp(current_date).strftime("%Y-%m")
    existing = ledger.get(month_key, {})
    basis_assets, basis_date, basis_source = get_portfolio_month_start_basis(current_date, ledger=ledger)

    default_start_assets = _to_float(existing.get("month_start_assets")) or basis_assets
    default_end_assets = _to_float(existing.get("month_end_assets")) or current_total_assets
    default_deposit = _to_float(existing.get("deposit")) or 0
    default_withdrawal = _to_float(existing.get("withdrawal")) or 0
    default_start_date = pd.Timestamp(existing.get("month_start_date") or basis_date).date()
    default_end_date = pd.Timestamp(existing.get("month_end_date") or current_date).date()

    with st.form("monthly_profit_record_form"):
        month = st.text_input("기록 월", value=month_key)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작 기준일", value=default_start_date)
            start_assets = st.number_input("시작 총자산", value=float(default_start_assets), step=1_000_000.0)
            deposit = st.number_input("이번달 입금", value=float(default_deposit), step=1_000_000.0)
        with col2:
            end_date = st.date_input("평가 기준일", value=default_end_date)
            end_assets = st.number_input("현재/월말 총자산", value=float(default_end_assets), step=1_000_000.0)
            withdrawal = st.number_input("이번달 출금", value=float(default_withdrawal), step=1_000_000.0)

        preview = build_monthly_ledger_record(month, start_date, start_assets, end_date, end_assets, deposit, withdrawal)
        preview_return = preview["official_return"]
        st.caption(
            f"미리보기: 공식 손익 {_fmt_won(preview['official_profit'], signed=True)}"
            + (f" ({preview_return:+.2%})" if preview_return is not None else "")
            + f" | 시작 기준 출처: {basis_source}"
        )
        submitted = st.form_submit_button("이번달 수익 기록 저장", type="primary")

    if submitted:
        path = upsert_monthly_ledger_record(preview)
        st.success(f"{preview['month']} 기록을 저장했습니다: {path}")


def render_portfolio_operations_dashboard(policy, current_date, current_total_assets, price_col):
    st.markdown("---")
    render_faber_a_monthly_reference(current_date, current_total_assets, price_col)
    st.markdown("---")
    render_monthly_profit_recorder(current_date, current_total_assets)


ASSETS = {
    '코스피200': '294400',
    '미국나스닥100': '133690',
    '한국채30년': '439870',
    '미국채30년': '476760',
    '금현물': '411060'
}

CASH_TICKER = '455890'
CASH_NAME = '현금(MMF)'
NASDAQ100_ASSET_NAME = next((name for name, ticker in ASSETS.items() if ticker == '133690'), None)
KR_STOCK_MIX_ASSET = '코스피200'
KR_BOND_10Y_MIX_ASSET = '한국국고채10년'
KR_BOND_10Y_MIX_TICKER = '148070'
KR_3ASSET_STRATEGY_LABEL = 'KR 3자산 평균모멘텀'
FABER_NASDAQ_ACTIVE_EXEC_LABEL = 'Faber A (나스닥 액티브 집행)'
TIME_NASDAQ_ACTIVE_TICKER = '426030'
KOACT_NASDAQ_GROWTH_ACTIVE_TICKER = '0015B0'
TIME_NASDAQ_ACTIVE_LISTING_DATE = pd.Timestamp('2022-05-11')
KOACT_NASDAQ_GROWTH_ACTIVE_LISTING_DATE = pd.Timestamp('2025-02-25')
FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL = '해남 A'
FABER_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL = '해남 A (한국=삼성전자)'
FABER_ACTIVE_NASDAQ_KR_HYNIX_LABEL = '해남 A (한국=SK하이닉스)'
FABER_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL = '해남 A (한국=삼성전자/하이닉스)'
# 한국 슬롯을 코스피200 패시브(지수 그대로)로 집행하는 변형. 나스닥은 다른 해남 A와 동일하게
# 액티브 집행이라, 나스닥 액티브 집행 데이터셋(build_faber_nasdaq_active_execution_data)이 곧
# 이 변형과 동일한 NAV가 된다(한국 슬롯에 별도 오버레이를 얹지 않으면 코스피200 지수 유지).
FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL = '해남P (-5%룰)'
FABER_PASSIVE_NASDAQ_KR_PASSIVE_LABEL = '해남 A (한국=코스피200TR, 나스닥=패시브)'
HAENAM_M_LABEL = '해남M'
HAENAM_P_LABEL = '해남P'
HAENAM_P_LOCAL_SIGNAL_LABEL = '해남P (현지통화 신호)'
HAENAM_P_VIX70_LABEL = '해남P+VIX (70%상한)'
HAENAM_P_VIX100_LABEL = '해남P+VIX (100%상한)'
HAENAM_S_LABEL = '해남S'
HAENAM_V_MOM_LABEL = '해남V (연속모멘텀)'
HAENAM_V_FABER_LABEL = '해남V (-5%룰)'
HAENAM_V_PASSIVE_MOM_LABEL = '해남V 패시브 (연속모멘텀)'
HAENAM_V_PASSIVE_FABER_LABEL = '해남V 패시브 (-5%룰)'
# 위 5개 해남 A 변형과 한국 슬롯/나스닥(액티브)·나머지 슬롯은 동일하게 두고, 신호(투자 방법)만
# 연속 모멘텀(simulate_daily_nav_with_attribution, 0.2×12개월 모멘텀 점수)으로 바꾼 비교군.
MOM_ACTIVE_NASDAQ_KR_ACTIVE_LABEL = HAENAM_M_LABEL
MOM_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL = HAENAM_S_LABEL
MOM_PASSIVE_NASDAQ_KR_SAMSUNG_LABEL = '연속모멘텀 (한국=삼성전자, 신호=코스피200, 나스닥=패시브)'
MOM_ACTIVE_NASDAQ_KR_SAMSUNG_SELF_SIGNAL_LABEL = '연속모멘텀 (한국=삼성전자, 신호=삼성전자)'
MOM_ACTIVE_NASDAQ_KR_HYNIX_LABEL = '연속모멘텀 (한국=SK하이닉스)'
MOM_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL = '연속모멘텀 (한국=삼성전자/하이닉스)'
MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL = HAENAM_P_LABEL
MOM_PASSIVE_NASDAQ_KR_PASSIVE_LABEL = '연속모멘텀 (한국=코스피200TR, 나스닥=패시브)'
HAENAM_SAMSUNG_NAME = '삼성전자'
HAENAM_HYNIX_NAME = 'SK하이닉스'
HAENAM_TIME_NAME = 'TIME 나스닥100액티브'
HAENAM_KOACT_NAME = 'KoAct 나스닥100액티브'
# 한국주식 슬롯 코스피 액티브 전환 대상 (삼전/하닉 → 코스피 액티브 2종, 슬롯 내 50:50)
TIME_KOSPI_ACTIVE_TICKER = '385720'
KOACT_KOSPI_ACTIVE_TICKER = '0193G0'
HAENAM_KR_TIME_NAME = 'TIME 코스피액티브'
HAENAM_KR_KOACT_NAME = 'KoAct 코스피액티브'
TIME_KOSPI_ACTIVE_LISTING_DATE = pd.Timestamp('2021-05-25')
KOACT_KOSPI_ACTIVE_LISTING_DATE = pd.Timestamp('2026-05-07')  # 효력발생일 근사(상장 직후, 백테스트 영향 미미)
TIME_KOREA_VALUEUP_ACTIVE_TICKER = '495060'
KOACT_KOREA_VALUEUP_ACTIVE_TICKER = '495230'
KOREA_VALUEUP_INDEX_PROXY_TICKER = '495850'
KOREA_VALUEUP_TR_PROXY_TICKER = '495550'
KOREA_VALUEUP_PASSIVE_TICKER = KOREA_VALUEUP_TR_PROXY_TICKER
KOREA_VALUEUP_ACTIVE_LISTING_DATE = pd.Timestamp('2024-11-04')
HAENAM_VALUEUP_TIME_NAME = 'TIME 코리아밸류업액티브'
HAENAM_VALUEUP_KOACT_NAME = 'KoAct 코리아밸류업액티브'
HAENAM_VALUEUP_PASSIVE_NAME = 'SOL 코리아밸류업TR'
# 나스닥100 슬롯 실제 집행 상품 전환일 ('이번 달 성과 참고' 표시 전용 — 공식 손익과는 무관).
# 2026-06-02 TIME/KoAct 나스닥100액티브 50:50로 전환 완료(이전: 패시브 미국나스닥100, 133690).
#   - 전환 게이트: TIME·KoAct 괴리율이 모두 패시브(133690) 기준선 대비 0.3% 이하인 날 집행(2026-06-02 충족).
#
# === 전환 방법 (다른 Claude Code도 따라할 수 있도록) ===
# 사용자가 액티브로 갈아탄 날짜를 주면 아래 None을 그 날짜로 바꾼다. 예: pd.Timestamp('2026-06-30').
#   - 입력값 = '액티브를 매수한 날(리밸런싱일)'. 보통 월말 거래일.
#   - 동작: "이번 달 성과" 계산 시 rebal_date(=전월 마지막 거래일)가 이 날짜보다 '이전'이면 그 달은 패시브로,
#           이 날짜 '이상'이면 액티브(TIME 50% / KoAct 50%)로 참고 성과를 표시한다. (로직: '이번 달 성과' 섹션 참고)
#   - 예) SWITCH='2026-06-30' → 6월 성과(rebal 5/29)는 패시브, 7월 성과(rebal 6/30)부터 액티브.
#   - 코드가 쓰는 가격은 ETF '시장 종가'(데이터피드)이지 사용자의 체결가가 아니다. 따라서 전환에 필요한 건 '날짜'뿐.
#     체결가/체결일은 .codex-private 저널 기록용이며 이 계산에는 들어가지 않는다.
#   - 공식 손익(계좌 총액 − 기준 − 현금흐름)과는 무관. 이 상수는 참고 표시만 바꾼다.
# 상세 운용 메모: .codex-private/investment_memory.md
HAENAM_NASDAQ_ACTIVE_SWITCH_DATE = pd.Timestamp('2026-06-02')
# 한국주식 슬롯 코스피 액티브 전환일 ('이번 달 성과 참고' 표시 전용 — 공식 손익과는 무관).
# 2026-06-01 TIME/KoAct 코스피액티브 50:50로 전환 완료(이전: 삼성전자/SK하이닉스 50:50).
#   - 나스닥 패시브→액티브 전환일과 같은 날 한 번에 옮기는 것을 기준으로 한다.
#   - 사용자가 전환일(=코스피액티브 매수일, 보통 월말 거래일)을 주면 아래 None을 그 날짜로 바꾼다.
#     예: pd.Timestamp('2026-06-30'). rebal_date < 이 날짜면 삼전/하닉, >= 이면 코스피액티브로 참고 표시.
#   - 코스피액티브 괴리율은 둘 다 <0.1%라 자체 게이트 불필요. 공동 전환일의 구속은 나스닥 괴리율(0.3% 이하)뿐.
#   - 체결가/수량은 .codex-private 저널 기록용. 이 상수는 참고 표시만 바꾼다.
#   - 전환 시 이 상수와 함께 아래 캡션의 '삼성전자/SK하이닉스' 문구도 코스피액티브로 같이 바꾼다.
# 상세: .codex-private/investment_memory.md·journal.md (2026-05-31 한국 슬롯 전환)
HAENAM_KR_ACTIVE_SWITCH_DATE = pd.Timestamp('2026-06-01')
SAMSUNG_ELECTRONICS_TICKER = '005930'
SK_HYNIX_TICKER = '000660'
CHINA_CSI300_CNY_ASSET = '중국CSI300(위안화 노출)'
INDIA_NIFTY_INR_ASSET = '인도니프티(루피 노출)'
FABER_EX_BONDS_3_LABEL = 'Faber A (한·미·금 3자산)'
FABER_EX_BONDS_4_LABEL = 'Faber A (한·미·중국·금 4자산)'
FABER_EX_BONDS_5_LABEL = 'Faber A (한·미·중국·인도·금 5자산)'

KR_BOND_DURATION_FACTOR = 2.5

# ── 채권 딥프록시(금리→가격) 합성 파라미터 (실 ETF 검증 기반 보정) ─────────────
# ① 듀레이션 보정: 기존 '10년듀레이션 10 × 2.5배 = 실효 25'는 실 ETF 대비 과다
#    (검증상 변동성 1.11x). KOSEF국고채10년 프록시(실듀레이션 ~8.5 × 2.5 ≈ 21)와
#    일치하도록 30년 실효 수정듀레이션을 ~21로 단일화한다.
# ③ 컨벡시티: C ≈ D·(D+1) 근사로 금리 급변 구간의 1차근사(듀레이션) 오차를
#    양의 볼록성으로 보정한다.
# 채권 합성 듀레이션(실 ETF 검증 기반).
#   - 실제 30년 국고채 금리(ECOS 010230000, 2012-09~)로 합성할 때 D=19 → 실 ETF와
#     일상관 0.97·월상관 0.996·일vol 0.99x (검증). carry(이자) 항 포함 시 총수익 부호도 일치.
#   - 30년물이 없던 2012년 이전 구간은 10년물 금리 변화로 합성하며, 그 변화폭이 30년물보다
#     작아 실효 듀레이션 25가 필요(월vol≈1.0x). 두 구간은 체인링크로 연결.
KR_BOND_30Y_DURATION = 19.0
KR_BOND_30Y_CONVEXITY = KR_BOND_30Y_DURATION * (KR_BOND_30Y_DURATION + 1.0)
# 30년물이 없던 2012년 이전: ECOS 일별 10년물 금리변화에 beta를 곱해 30년물 금리변화를 역산.
#   2012~2026 겹침구간 회귀: corr(d30,d10)=0.92, beta=0.81 (장기물이 10년물의 0.81배 변동).
#   이렇게 만든 연속 30년 금리에 D=19를 단일 적용한다(가격 체인링크 불필요).
KR_BOND_10Y_TO_30Y_BETA = 0.81
KR_BOND_EFFECTIVE_DURATION = 25.0   # 키 없을 때 폴백(FRED OECD 월별 10년물, 변화폭이 작아 보정 큼)
KR_BOND_CONVEXITY = KR_BOND_EFFECTIVE_DURATION * (KR_BOND_EFFECTIVE_DURATION + 1.0)
KR_BOND_10Y_DURATION = 8.5
KR_BOND_10Y_CONVEXITY = KR_BOND_10Y_DURATION * (KR_BOND_10Y_DURATION + 1.0)
US_BOND_EFFECTIVE_DURATION = 21.0   # GS30(실제 30년물) 기반, 시간보간 후 월vol≈1.0x 보정
US_BOND_CONVEXITY = US_BOND_EFFECTIVE_DURATION * (US_BOND_EFFECTIVE_DURATION + 1.0)

PROXY_ASSETS = {
    '코스피200': {
        'ticker': '069500', 'type': 'kr_etf',
        'note': 'KODEX 200 ETF'
    },
    '미국나스닥100': {
        'ticker': 'QQQ', 'fx': 'USD/KRW', 'type': 'us_etf_fx',
        'note': 'QQQ x USD/KRW 합성'
    },
    '한국채30년': {
        'ticker': '148070', 'type': 'kr_etf_duration_adjusted',
        'duration_factor': KR_BOND_DURATION_FACTOR,
        'note': f'KOSEF 국고채10년 x 듀레이션 배수({KR_BOND_DURATION_FACTOR}x) 합성'
    },
    '미국채30년': {
        'ticker': 'TLT', 'fx': 'USD/KRW', 'type': 'us_etf_fx',
        'note': 'TLT x USD/KRW 합성'
    },
    '금현물': {
        'ticker': 'GLD', 'fx': 'USD/KRW', 'type': 'us_etf_fx',
        'note': 'GLD x USD/KRW 합성'
    },
}

PROXY_CASH = {
    'type': 'cd91_cash',
    'spread': 0.0015,
    'fallback_annual_rate': 0.025,
    'note': 'ECOS CD91 - 0.15%p 합성 현금 (실패 시 연 2.5%)'
}

# 보조 벤치마크 ETF
BENCHMARK_ETF = {
    'ticker': '0113D0',
    'name': 'TIME 글로벌 탑픽',
    'note': '자산배분 펀드 (보조 벤치마크)'
}

REQUESTED_STATIC_PORTFOLIOS = [
    {
        "name": "요청 포트폴리오 1",
        "description": "나스닥/코스피 액티브 + ACE 미국30년국채액티브",
        "assets": [
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.35},
            {"name": "TIME 코스피액티브", "ticker": "385720", "weight": 0.35},
            {"name": "ACE 미국30년국채액티브", "ticker": "476760", "weight": 0.30},
        ],
    },
    {
        "name": "요청 포트폴리오 2",
        "description": "액티브 성장/배당 + CD금리",
        "assets": [
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.20},
            {"name": "TIME 글로벌우주테크&방산액티브", "ticker": "478150", "weight": 0.20},
            {"name": "TIME Korea플러스배당액티브", "ticker": "441800", "weight": 0.20},
            {"name": "TIGER 미국배당다우존스", "ticker": "458730", "weight": 0.20},
            {"name": "KODEX CD금리액티브(합성)", "ticker": "459580", "weight": 0.20},
        ],
    },
    {
        "name": "요청 포트폴리오 3",
        "description": "나스닥/코스피/미국30년 + TIGER CD금리",
        "assets": [
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.25},
            {"name": "TIME 코스피액티브", "ticker": "385720", "weight": 0.25},
            {"name": "ACE 미국30년국채액티브", "ticker": "476760", "weight": 0.25},
            {"name": "TIGER CD금리투자KIS(합성)", "ticker": "357870", "weight": 0.25},
        ],
    },
    {
        "name": "요청 포트폴리오 4",
        "description": "나스닥/우주방산/코스피액티브/미국배당/CD 각 20%",
        "assets": [
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.20},
            {"name": "TIME 글로벌우주테크&방산액티브", "ticker": "478150", "weight": 0.20},
            {"name": "TIME 코스피액티브", "ticker": "385720", "weight": 0.20},
            {"name": "TIGER 미국배당다우존스", "ticker": "458730", "weight": 0.20},
            {"name": "KODEX CD금리액티브(합성)", "ticker": "459580", "weight": 0.20},
        ],
    },
    {
        "name": "요청 포트폴리오 5",
        "description": "미국배당 슬롯을 TIME 미국배당다우존스액티브로 교체(상장 전 TIGER 보강)",
        "assets": [
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.20},
            {"name": "TIME 글로벌우주테크&방산액티브", "ticker": "478150", "weight": 0.20},
            {"name": "TIME 코스피액티브", "ticker": "385720", "weight": 0.20},
            {
                "name": "TIME 미국배당다우존스액티브",
                "ticker": "0036D0",
                "fallback_ticker": "458730",
                "fallback_name": "TIGER 미국배당다우존스",
                "weight": 0.20,
            },
            {"name": "KODEX CD금리액티브(합성)", "ticker": "459580", "weight": 0.20},
        ],
    },
    {
        "name": "요청 포트폴리오 7",
        "description": "2번 + TIME 미국배당다우존스액티브(상장 전 TIGER 보강)",
        "assets": [
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.20},
            {"name": "TIME 글로벌우주테크&방산액티브", "ticker": "478150", "weight": 0.20},
            {"name": "TIME Korea플러스배당액티브", "ticker": "441800", "weight": 0.20},
            {
                "name": "TIME 미국배당다우존스액티브",
                "ticker": "0036D0",
                "fallback_ticker": "458730",
                "fallback_name": "TIGER 미국배당다우존스",
                "weight": 0.20,
            },
            {"name": "KODEX CD금리액티브(합성)", "ticker": "459580", "weight": 0.20},
        ],
    },
    {
        "name": "요청 포트폴리오 8",
        "description": "코스피액티브+Korea플러스배당/나스닥/TIME 미국배당/CD 각 20%",
        "assets": [
            {"name": "TIME 코스피액티브", "ticker": "385720", "weight": 0.20},
            {"name": "TIME Korea플러스배당액티브", "ticker": "441800", "weight": 0.20},
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.20},
            {
                "name": "TIME 미국배당다우존스액티브",
                "ticker": "0036D0",
                "fallback_ticker": "458730",
                "fallback_name": "TIGER 미국배당다우존스",
                "weight": 0.20,
            },
            {"name": "KODEX CD금리액티브(합성)", "ticker": "459580", "weight": 0.20},
        ],
    },
    {
        "name": "요청 포트폴리오 9",
        "description": "5번에서 우주방산을 TIME 글로벌AI인공지능액티브로 교체",
        "assets": [
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.20},
            {"name": "TIME 글로벌AI인공지능액티브", "ticker": "456600", "weight": 0.20},
            {"name": "TIME 코스피액티브", "ticker": "385720", "weight": 0.20},
            {
                "name": "TIME 미국배당다우존스액티브",
                "ticker": "0036D0",
                "fallback_ticker": "458730",
                "fallback_name": "TIGER 미국배당다우존스",
                "weight": 0.20,
            },
            {"name": "KODEX CD금리액티브(합성)", "ticker": "459580", "weight": 0.20},
        ],
    },
    {
        "name": "요청 포트폴리오 10",
        "description": "5번에서 우주방산 10% + TIME 글로벌AI 10%",
        "assets": [
            {"name": "TIME 미국나스닥100액티브", "ticker": "426030", "weight": 0.20},
            {"name": "TIME 글로벌우주테크&방산액티브", "ticker": "478150", "weight": 0.10},
            {"name": "TIME 글로벌AI인공지능액티브", "ticker": "456600", "weight": 0.10},
            {"name": "TIME 코스피액티브", "ticker": "385720", "weight": 0.20},
            {
                "name": "TIME 미국배당다우존스액티브",
                "ticker": "0036D0",
                "fallback_ticker": "458730",
                "fallback_name": "TIGER 미국배당다우존스",
                "weight": 0.20,
            },
            {"name": "KODEX CD금리액티브(합성)", "ticker": "459580", "weight": 0.20},
        ],
    },
]
REQUESTED_PORTFOLIO_COMMON_START = pd.Timestamp("2024-04-30")

PREFERRED_ACCOUNT = {
    '코스피200': '일반', '미국나스닥100': 'ISA',
    HAENAM_SAMSUNG_NAME: '일반', HAENAM_HYNIX_NAME: '일반',
    HAENAM_KR_TIME_NAME: '일반', HAENAM_KR_KOACT_NAME: '일반',
    HAENAM_TIME_NAME: 'ISA', HAENAM_KOACT_NAME: 'ISA',
    '한국채30년': 'ISA', '미국채30년': 'ISA', '금현물': '일반'
}
GENERAL_PRIORITY = ['금현물', HAENAM_SAMSUNG_NAME, HAENAM_HYNIX_NAME, HAENAM_KR_TIME_NAME, HAENAM_KR_KOACT_NAME, '코스피200']
ISA_PRIORITY = ['미국채30년', '한국채30년', HAENAM_TIME_NAME, HAENAM_KOACT_NAME, '미국나스닥100']
ACCOUNT_COLUMNS = ["금계좌", "일반계좌", "ISA_A", "ISA_B"]
MIN_VALID_MONTHS = 12

BUY_HOLD_ACCOUNT_COLUMNS = ["일반계좌", "ISA_A", "ISA_B"]
BUY_HOLD_ACCOUNT_DISPLAY = {
    "일반계좌": "일반계좌",
    "ISA_A": "ISA A",
    "ISA_B": "ISA B",
}
BUY_HOLD_BASELINE_WEIGHTS = {
    "코스피": 0.20,
    "나스닥100": 0.20,
    "미국채30년": 0.20,
    "현금": 0.40,
}
BUY_HOLD_ASSET_BUCKETS = list(BUY_HOLD_BASELINE_WEIGHTS.keys())
BUY_HOLD_INSTRUMENT_MAP = {
    HAENAM_KR_TIME_NAME: f"{TIME_KOSPI_ACTIVE_TICKER}",
    HAENAM_KR_KOACT_NAME: f"{KOACT_KOSPI_ACTIVE_TICKER}",
    HAENAM_TIME_NAME: f"{TIME_NASDAQ_ACTIVE_TICKER}",
    HAENAM_KOACT_NAME: f"{KOACT_NASDAQ_GROWTH_ACTIVE_TICKER}",
    "미국채30년": "ACE 미국30년국채액티브 (476760)",
    CASH_NAME: f"{CASH_TICKER}",
}


# ==============================
# 2. 쿼리파라미터 유틸
# ==============================
def _get_query_params():
    if hasattr(st, "query_params"):
        return dict(st.query_params)
    return st.experimental_get_query_params()

def _qp_first(v):
    return v[0] if isinstance(v, list) and len(v) > 0 else v

def _get_qp_int(qp, key):
    raw = _qp_first(qp.get(key))
    if raw is None or raw == "": return None
    try: return int(float(raw))
    except Exception: return None

def _set_query_params(**kwargs):
    if hasattr(st, "query_params"):
        for k, v in kwargs.items(): st.query_params[k] = str(v)
    else:
        st.experimental_set_query_params(**{k: str(v) for k, v in kwargs.items()})


# ==============================
# 3. 날짜/가격 유틸
# ==============================
def normalize_to_date(dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def get_month_end_date(date):
    next_month = date.replace(day=28) + timedelta(days=4)
    return next_month - timedelta(days=next_month.day)


def get_default_backtest_end_date(current_date):
    """Use the last completed month-end as the default reproducible backtest cut-off."""
    return normalize_to_date(get_month_end_date(current_date - relativedelta(months=BACKTEST_DEFAULT_END_LAG_MONTHS)))


def clamp_market_data_to_date(all_data, end_date):
    """Return a copy of market data with every series capped at the requested end date."""
    if all_data is None:
        return None
    end_ts = pd.Timestamp(end_date)
    capped = {}
    for name, df in all_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = df.copy()
            out = out[~out.index.duplicated(keep="last")].sort_index()
            out = out[out.index <= end_ts]
            capped[name] = out
        else:
            capped[name] = df
    return capped


def build_market_data_fingerprint(all_data, price_col="Adj Close"):
    """Build a compact fingerprint so fixed-date backtests can be debugged and reproduced."""
    rows = []
    if not all_data:
        return None, None

    aggregate = hashlib.sha256()
    for name, df in sorted(all_data.items(), key=lambda kv: str(kv[0])):
        if not isinstance(df, pd.DataFrame) or df.empty:
            rows.append({"series": str(name), "rows": 0, "start": "-", "end": "-", "last": None, "hash": "-"})
            aggregate.update(f"{name}|empty".encode("utf-8", errors="ignore"))
            continue

        clean = df[~df.index.duplicated(keep="last")].sort_index()
        col = price_col if price_col in clean.columns else ("Close" if "Close" in clean.columns else clean.columns[0])
        values = pd.to_numeric(clean[col], errors="coerce").dropna()
        if values.empty:
            rows.append({"series": str(name), "rows": int(len(clean)), "start": "-", "end": "-", "last": None, "hash": "-"})
            aggregate.update(f"{name}|no-values".encode("utf-8", errors="ignore"))
            continue

        payload = values.round(10)
        digest = hashlib.sha256(pd.util.hash_pandas_object(payload, index=True).values.tobytes()).hexdigest()[:12]
        aggregate.update(f"{name}|{digest}|{len(values)}|{values.index.min()}|{values.index.max()}".encode("utf-8", errors="ignore"))
        rows.append({
            "series": str(name),
            "rows": int(len(values)),
            "start": values.index.min().strftime("%Y-%m-%d"),
            "end": values.index.max().strftime("%Y-%m-%d"),
            "last": float(values.iloc[-1]),
            "hash": digest,
        })

    return pd.DataFrame(rows), aggregate.hexdigest()[:16]


def render_backtest_reproducibility_status(bt_start_date, bt_end_date, price_col, fingerprint, fingerprint_df):
    """Show whether the current backtest input matches the previous run in this session."""
    key = "last_backtest_reproducibility_snapshot"
    snapshot = {
        "start": pd.Timestamp(bt_start_date).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(bt_end_date).strftime("%Y-%m-%d"),
        "price_col": price_col,
        "fingerprint": fingerprint,
    }
    previous = st.session_state.get(key)

    if previous is None:
        st.info("첫 실행 기준값을 저장했습니다. 다음 실행부터 데이터 변경 여부를 자동 비교합니다.")
    else:
        changed_fields = [
            label for field, label in [
                ("start", "시작일"),
                ("end", "종료일"),
                ("price_col", "가격 기준"),
                ("fingerprint", "입력 데이터"),
            ]
            if previous.get(field) != snapshot.get(field)
        ]
        if not changed_fields:
            st.success("이전 실행과 백테스트 입력이 동일합니다. 같은 결과가 재현되어야 합니다.")
        else:
            st.warning("이전 실행과 백테스트 입력이 변경되었습니다: " + ", ".join(changed_fields))
            st.caption(
                f"이전: {previous.get('start')} ~ {previous.get('end')} | "
                f"{previous.get('price_col')} | {previous.get('fingerprint')}"
            )
            st.caption(
                f"현재: {snapshot['start']} ~ {snapshot['end']} | "
                f"{snapshot['price_col']} | {snapshot['fingerprint']}"
            )

    if fingerprint_df is not None and previous is not None and previous.get("fingerprint_df") is not None:
        prev_df = previous["fingerprint_df"]
        curr_df = fingerprint_df.copy()
        compare = curr_df.merge(
            prev_df[["series", "rows", "end", "last", "hash"]].rename(
                columns={
                    "rows": "prev_rows",
                    "end": "prev_end",
                    "last": "prev_last",
                    "hash": "prev_hash",
                }
            ),
            on="series",
            how="outer",
        )
        changed = compare[
            (compare["hash"] != compare["prev_hash"]) |
            (compare["rows"] != compare["prev_rows"]) |
            (compare["end"] != compare["prev_end"])
        ]
        if not changed.empty:
            display_cols = ["series", "prev_end", "end", "prev_rows", "rows", "prev_last", "last"]
            with st.expander("변경된 입력 데이터 상세", expanded=False):
                st.dataframe(changed[display_cols], use_container_width=True, hide_index=True)

    stored = snapshot.copy()
    stored["fingerprint_df"] = fingerprint_df.copy() if fingerprint_df is not None else None
    st.session_state[key] = stored


# ==============================
# 4. 데이터 로딩
# ==============================
@st.cache_data(ttl=3600)
def fetch_etf_data(ticker, start_date, end_date, is_momentum=False):
    try:
        if ticker == '411060' and is_momentum:
            synthetic_df = None

            # fallback 기반(장기 구간): GLD×USD/KRW 합성
            gld = _read_us_market_data('GLD', start_date, end_date)
            usdkrw = get_usdkrw_series(start_date, end_date)
            if gld is not None and not gld.empty and usdkrw is not None and not usdkrw.empty:
                gld = gld[~gld.index.duplicated(keep='last')]
                usdkrw = usdkrw[~usdkrw.index.duplicated(keep='last')]
                merged = pd.concat([gld['Close'], usdkrw['Close']], axis=1, keys=['GLD', 'USDKRW'])
                merged = merged.ffill().bfill()
                synthetic_df = pd.DataFrame(index=merged.index)
                synthetic_df['Close'] = merged['GLD'] * merged['USDKRW']
                synthetic_df['Adj Close'] = synthetic_df['Close']

            # 1순위(최근 구간): KODEX 금액티브(0064K0)
            kodex_gold = _read_fdr_with_fallback('0064K0', start_date, end_date)
            if kodex_gold is not None and not kodex_gold.empty and 'Close' in kodex_gold.columns:
                kodex_gold = kodex_gold[~kodex_gold.index.duplicated(keep='last')].sort_index()
                momentum_df = pd.DataFrame(index=kodex_gold.index)
                momentum_df['Close'] = kodex_gold['Close'].astype(float)
                if 'Adj Close' in kodex_gold.columns:
                    momentum_df['Adj Close'] = kodex_gold['Adj Close'].astype(float)
                else:
                    momentum_df['Adj Close'] = momentum_df['Close']
                if synthetic_df is not None and not synthetic_df.empty:
                    return _chain_link_series(synthetic_df, momentum_df)
                return momentum_df

            return synthetic_df
        prefer_long_history = ticker in (SAMSUNG_ELECTRONICS_TICKER, SK_HYNIX_TICKER)
        df = _read_kr_market_data(ticker, start_date, end_date, prefer_long_history=prefer_long_history)
        if df is None or len(df) == 0: return None
        df = df[~df.index.duplicated(keep='last')].sort_index()
        if 'Close' not in df.columns: return None
        return df
    except Exception as e:
        st.warning(f"티커 {ticker} 데이터 로딩 오류: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def fetch_proxy_data(asset_name, start_date, end_date):
    try:
        if asset_name == CASH_NAME:
            config = PROXY_CASH
        else:
            config = PROXY_ASSETS.get(asset_name)
        if config is None: return None
        
        if config['type'] == 'cd91_cash':
            rates = fetch_cd91_rate_series(start_date, end_date)
            return build_cash_price_index_from_annual_rates(
                rates,
                start_date,
                end_date,
                spread=config.get('spread', 0.0015),
                fallback_annual_rate=config.get('fallback_annual_rate', 0.025),
            )

        if config['type'] == 'synthetic_cash':
            # 2000-01-01부터 영업일 기준으로 합성 현금을 생성한다.
            # KODEX200 데이터 존재 여부와 무관하게 pd.bdate_range를 직접 사용.
            dates = pd.bdate_range(start=start_date, end=end_date)
            if len(dates) == 0: return None
            annual_rate = config.get('annual_rate', 0.025)
            daily_rate = (1 + annual_rate) ** (1/252) - 1
            base_price = 10000.0
            prices = [base_price]
            for i in range(1, len(dates)):
                prices.append(prices[-1] * (1 + daily_rate))
            return pd.DataFrame({'Close': prices, 'Adj Close': prices}, index=dates)
        
        elif config['type'] == 'kr_etf':
            df = _read_kr_market_data(config['ticker'], start_date, end_date, prefer_long_history=False)
            if df is None or df.empty: return None
            df = df[~df.index.duplicated(keep='last')].sort_index()
            if 'Close' not in df.columns: return None
            return df
        
        elif config['type'] == 'kr_etf_duration_adjusted':
            df = _read_kr_market_data(config['ticker'], start_date, end_date)
            if df is None or df.empty: return None
            df = df[~df.index.duplicated(keep='last')].sort_index()
            if 'Close' not in df.columns: return None
            factor = config.get('duration_factor', KR_BOND_DURATION_FACTOR)
            close = df['Close'].copy()
            daily_returns = close.pct_change().fillna(0.0)
            adjusted_returns = (daily_returns * factor).clip(-0.10, 0.10)
            synthetic_price = (1 + adjusted_returns).cumprod() * close.iloc[0]
            result_df = pd.DataFrame(index=df.index)
            result_df['Close'] = synthetic_price
            result_df['Adj Close'] = synthetic_price
            return result_df
        
        elif config['type'] == 'us_etf_fx':
            us_df = _read_us_market_data(config['ticker'], start_date, end_date)
            fx_df = get_usdkrw_series(start_date, end_date) if config.get('fx') == 'USD/KRW' else fdr.DataReader(config['fx'], start_date, end_date)
            if us_df is None or us_df.empty or fx_df is None or fx_df.empty:
                return None
            us_df = us_df[~us_df.index.duplicated(keep='last')]
            fx_df = fx_df[~fx_df.index.duplicated(keep='last')]
            us_close = us_df['Adj Close'] if 'Adj Close' in us_df.columns else us_df['Close']
            fx_close = fx_df['Close']
            merged = pd.concat([us_close, fx_close], axis=1, keys=['US', 'FX'])
            merged = merged.ffill().bfill().dropna()
            synthetic_df = pd.DataFrame(index=merged.index)
            synthetic_df['Close'] = merged['US'] * merged['FX']
            synthetic_df['Adj Close'] = synthetic_df['Close']
            return synthetic_df
        return None
    except Exception as e:
        st.warning(f"프록시 데이터 로딩 오류 ({asset_name}): {str(e)}")
        return None


@st.cache_data(ttl=86400)
def fetch_deep_proxy_kospi(start_date, end_date):
    """FDR KS11 코스피 지수 → 코스피200 딥프록시 (2000-01-01~).
    KODEX200(069500) 상장 전(2002-10-14) 구간 커버.
    """
    try:
        for symbol in ('KS11', '^KS11', 'KOSPI'):
            try:
                df = fdr.DataReader(symbol, start_date, end_date)
                if df is None or df.empty or 'Close' not in df.columns:
                    continue
                df = df[~df.index.duplicated(keep='last')].sort_index()
                result = pd.DataFrame(index=df.index)
                result['Close'] = df['Close'].astype(float)
                result['Adj Close'] = result['Close']
                return result
            except Exception:
                continue
        return None
    except Exception as e:
        st.warning(f"딥프록시(KOSPI) 로딩 오류: {e}")
        return None


def _synthesize_bond_price_from_yield(yields_raw, duration, convexity, include_carry=True):
    """금리(%) 시계열 → 채권 '총수익' 합성가격(기준 100) 딥프록시.

    개선 포인트(실 ETF 검증 반영):
    - ② 월별 등 저빈도 금리를 일별로 '시간 보간(time interpolation)'해, 한 달치 변화가
        하루에 몰려 일간수익률의 96.7%가 0이 되던 ffill+diff 아티팩트를 제거한다.
    - ① 실효 수정듀레이션(duration)을 단일 적용 — 기존 '듀레이션10 × 2.5배 + clip' 대체.
    - ③ 듀레이션 1차 + 컨벡시티 2차항 + 이자(carry)로 일별 총수익 합성:
        dP/P ≈ carry + (-D·dy/(1+y)) + 0.5·C·dy²
        · carry = y_prev/252 (일별 이자 accrual). 기존엔 가격수익만 계산해 채권 총수익을
          연 ~3~4%p 과소계상했고, 실 ETF 대비 누적수익 부호까지 어긋났다(검증).
    """
    if yields_raw is None or len(yields_raw) == 0:
        return None
    y = pd.to_numeric(yields_raw, errors="coerce").dropna().sort_index()
    y = y[~y.index.duplicated(keep="last")]
    if len(y) < 2:
        return None
    all_dates = pd.bdate_range(start=y.index.min(), end=y.index.max())
    # ② 시간 기반 보간으로 매끄러운 일별 금리 경로 생성(양 끝단은 ffill/bfill).
    y_daily = (
        y.reindex(y.index.union(all_dates))
         .interpolate(method="time")
         .reindex(all_dates)
         .ffill().bfill()
         .dropna()
    )
    y_dec = y_daily / 100.0
    dy = y_dec.diff()
    # ①③ 듀레이션 + 컨벡시티 기반 가격수익 + 이자(carry) = 총수익.
    daily_return = (-duration * dy / (1 + y_dec.shift(1)) + 0.5 * convexity * dy ** 2)
    if include_carry:
        daily_return = daily_return + y_dec.shift(1) / 252.0
    daily_return = daily_return.fillna(0.0)
    # 데이터 오류성 비현실적 점프만 차단(보간 후엔 사실상 발동 안 함).
    daily_return = daily_return.clip(-0.20, 0.20)
    price = (1 + daily_return).cumprod() * 100.0
    result = pd.DataFrame(index=price.index)
    result['Close'] = price.values
    result['Adj Close'] = result['Close'].values
    return result


@st.cache_data(ttl=86400)
def _build_kr_synthetic_30y_yield(start_date, end_date):
    """연속 국고채30년 금리(%) 시계열 구성.
      · 2012-09~ : 실제 국고채30년 금리(ECOS 010230000).
      · ~2012-09 : 30년물 미존재 → 10년물(ECOS 일별 우선) 변화 × beta 로 역산하고,
                   경계 시점에서 실30년물 레벨에 앵커링해 연속성을 보장한다.
    반환: (연속 30년 금리 Series 또는 None, 10년물 금리 Series 또는 None)
          첫 값이 None이면 실30년물이 없는 것(키 미설정) → 호출부에서 10년물 폴백.
    """
    real30 = fetch_kr_30y_bond_yield_series(start_date, end_date)
    long_yield = fetch_kr_long_bond_yield_series(start_date, end_date)
    if real30 is None or len(real30) < 2:
        return None, long_yield
    if long_yield is None or len(long_yield) < 2:
        return real30, None
    t0 = real30.index.min()
    pre10 = long_yield[long_yield.index < t0]
    if len(pre10) < 2:
        return real30, long_yield
    anchor10 = long_yield.asof(t0)
    if pd.isna(anchor10):
        return real30, long_yield
    synth_pre = float(real30.iloc[0]) + KR_BOND_10Y_TO_30Y_BETA * (pre10 - float(anchor10))
    combined = pd.concat([synth_pre, real30]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, long_yield


@st.cache_data(ttl=86400)
def fetch_deep_proxy_kr_bond_ecos(start_date, end_date):
    """국고채30년 합성 총수익가격 딥프록시 (2000-12~).

    ① 실제 국고채30년 금리(ECOS 010230000, 2012-09~) + 2012년 이전은 10년물×beta 역산으로
       만든 '연속 30년 금리'에 실효 듀레이션 D=19를 단일 적용
       → 실 ETF와 일상관 0.97·월상관 0.996·일vol 0.99x, pre-2012 vol≈13%(실물 수준).
    ② 시간보간(lumping 제거)  ③ 컨벡시티  + 이자(carry) = 총수익.
    ECOS 키가 없으면 실30년물을 못 받아 전 구간 10년물(월별) 기반 D=25로 폴백한다.
    """
    try:
        y30, long_yield = _build_kr_synthetic_30y_yield(start_date, end_date)
        if y30 is not None and len(y30) > 1:
            return _synthesize_bond_price_from_yield(
                y30, KR_BOND_30Y_DURATION, KR_BOND_30Y_CONVEXITY
            )
        # 폴백: 실30년물 없음 → 10년물 금리 기반 합성(월별 OECD/FRED 보정 듀레이션).
        if long_yield is not None and len(long_yield) > 1:
            return _synthesize_bond_price_from_yield(
                long_yield, KR_BOND_EFFECTIVE_DURATION, KR_BOND_CONVEXITY
            )
        return None
    except Exception as e:
        st.warning(f"딥프록시(KR채권) 로딩 오류: {e}")
        return None


@st.cache_data(ttl=86400)
def fetch_deep_proxy_kr_bond_10y_fred(start_date, end_date):
    """한국 장기금리(ECOS→OECD/FRED) → 10년 국고채 합성가격 딥프록시.
    ②시간보간 + ①듀레이션(≈8.5) + ③컨벡시티 (30년물과 동일 합성 방식)."""
    try:
        yields_raw = fetch_kr_long_bond_yield_series(start_date, end_date)
        if yields_raw is None or yields_raw.empty:
            return None
        return _synthesize_bond_price_from_yield(
            yields_raw, KR_BOND_10Y_DURATION, KR_BOND_10Y_CONVEXITY
        )
    except Exception as e:
        st.warning(f"딥프록시(KR국고채10년 FRED) 로딩 오류: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_kr_bond_10y_chain_data(start_date, end_date):
    """국고채10년 체인링크: 딥프록시(FRED) → KOSEF국고채10년(148070)."""
    deep = fetch_deep_proxy_kr_bond_10y_fred(start_date, end_date)
    etf_raw = _read_fdr_with_fallback(KR_BOND_10Y_MIX_TICKER, start_date, end_date)
    etf_df = None
    if etf_raw is not None and not etf_raw.empty and 'Close' in etf_raw.columns:
        etf_raw = etf_raw[~etf_raw.index.duplicated(keep='last')].sort_index()
        etf_df = pd.DataFrame(index=etf_raw.index)
        etf_df['Close'] = etf_raw['Close'].astype(float)
        etf_df['Adj Close'] = etf_raw['Adj Close'].astype(float) if 'Adj Close' in etf_raw.columns else etf_df['Close']
    if deep is None or deep.empty:
        return etf_df
    if etf_df is None or etf_df.empty:
        return deep
    return _chain_link_series(deep, etf_df)


def _fetch_price_df_from_fdr_candidates(candidates, start_date, end_date):
    for symbol in candidates:
        try:
            raw = fdr.DataReader(symbol, start_date, end_date)
            if raw is None or raw.empty or "Close" not in raw.columns:
                continue
            raw = raw[~raw.index.duplicated(keep="last")].sort_index()
            out = pd.DataFrame(index=raw.index)
            out["Close"] = raw["Close"].astype(float)
            out["Adj Close"] = raw["Adj Close"].astype(float) if "Adj Close" in raw.columns else out["Close"]
            return out
        except Exception:
            continue
    return None


def _convert_local_price_to_krw(local_price_df, fx_to_krw_df):
    if local_price_df is None or local_price_df.empty or fx_to_krw_df is None or fx_to_krw_df.empty:
        return None
    col_local = "Adj Close" if "Adj Close" in local_price_df.columns else "Close"
    merged = pd.concat(
        [local_price_df[col_local].rename("LOCAL"), fx_to_krw_df["Close"].rename("FX")],
        axis=1
    ).ffill().dropna()
    if merged.empty:
        return None
    out = pd.DataFrame(index=merged.index)
    out["Close"] = (merged["LOCAL"] * merged["FX"]).astype(float)
    out["Adj Close"] = out["Close"]
    return out


@st.cache_data(ttl=86400)
def fetch_china_csi300_cny_krw_chain(start_date, end_date):
    """중국 CSI300(위안화 노출) KRW 시계열: SSEC(딥) → CSI300(000300) 체인."""
    fx = get_fx_series_to_krw("CNY", start_date, end_date)
    if fx is None or fx.empty:
        return None
    deep_cny = _fetch_price_df_from_fdr_candidates(["SSEC"], start_date, end_date)
    csi_cny = _fetch_price_df_from_fdr_candidates(["000300"], start_date, end_date)
    deep_krw = _convert_local_price_to_krw(deep_cny, fx)
    csi_krw = _convert_local_price_to_krw(csi_cny, fx)
    if deep_krw is None or deep_krw.empty:
        return csi_krw
    if csi_krw is None or csi_krw.empty:
        return deep_krw
    return _chain_link_series(deep_krw, csi_krw)


@st.cache_data(ttl=86400)
def fetch_india_nifty_inr_krw_chain(start_date, end_date):
    """인도 NIFTY(루피 노출) KRW 시계열: BSESN(딥) → NIFTY(^NSEI) 체인."""
    fx = get_fx_series_to_krw("INR", start_date, end_date)
    if fx is None or fx.empty:
        return None
    deep_inr = _fetch_price_df_from_fdr_candidates(["^BSESN"], start_date, end_date)
    nifty_inr = _fetch_price_df_from_fdr_candidates(["^NSEI"], start_date, end_date)
    deep_krw = _convert_local_price_to_krw(deep_inr, fx)
    nifty_krw = _convert_local_price_to_krw(nifty_inr, fx)
    if deep_krw is None or deep_krw.empty:
        return nifty_krw
    if nifty_krw is None or nifty_krw.empty:
        return deep_krw
    return _chain_link_series(deep_krw, nifty_krw)


@st.cache_data(ttl=86400)
def fetch_deep_proxy_us_bond_fred(start_date, end_date):
    """FRED GS30(실제 30년물, 월별) → TLT 합성가격(KRW) 딥프록시 (2000-01-01~).
    TLT 상장 전(2002-07-30) 구간 커버.
    ②시간보간 + ①듀레이션(US_BOND_EFFECTIVE_DURATION) + ③컨벡시티로 합성 후 USD→KRW.
    """
    try:
        yields_raw = _fetch_fred_series('GS30', start_date, end_date)
        if yields_raw is None or yields_raw.empty:
            return None
        price_usd_df = _synthesize_bond_price_from_yield(
            yields_raw, US_BOND_EFFECTIVE_DURATION, US_BOND_CONVEXITY
        )
        if price_usd_df is None or price_usd_df.empty:
            return None
        price_usd = price_usd_df['Close']
        # USD → KRW
        usdkrw_df = get_usdkrw_series(start_date, end_date)
        if usdkrw_df is None or usdkrw_df.empty:
            return None
        usdkrw = usdkrw_df['Close']
        usdkrw = usdkrw[~usdkrw.index.duplicated(keep='last')]
        merged = pd.concat([price_usd, usdkrw], axis=1, keys=['price', 'fx'])
        merged = merged.ffill().bfill().dropna()
        price_krw = merged['price'] * merged['fx']
        result = pd.DataFrame(index=price_krw.index)
        result['Close'] = price_krw.values
        result['Adj Close'] = result['Close']
        return result
    except Exception as e:
        st.warning(f"딥프록시(미국채 FRED) 로딩 오류: {e}")
        return None


@st.cache_data(ttl=86400)
def fetch_deep_proxy_gold_fred(start_date, end_date):
    """금 딥프록시 × USD/KRW → KRW 가격 (2000-01-01~).
    소스:
      - GC=F(선물, 장기) + GLD(ETF, 최근) 체인링크 우선
      - 한쪽만 있으면 단일 소스 사용
    fredapi 의존성 없음.
    """
    try:
        usdkrw_df = get_usdkrw_series(start_date, end_date)
        if usdkrw_df is None or usdkrw_df.empty:
            st.warning("USD/KRW 데이터를 가져올 수 없습니다 (금 딥프록시).")
            return None
        usdkrw = usdkrw_df['Close']
        usdkrw = usdkrw[~usdkrw.index.duplicated(keep='last')]

        gld_df, gc_df = None, None
        for ticker in ('GLD', 'GC=F'):
            try:
                raw = _read_us_market_data(ticker, start_date, end_date)
                if raw is None or raw.empty or 'Close' not in raw.columns:
                    continue
                raw = raw[~raw.index.duplicated(keep='last')].sort_index()
                out = pd.DataFrame(index=raw.index)
                out['Close'] = raw['Close'].astype(float)
                out['Adj Close'] = raw['Adj Close'].astype(float) if 'Adj Close' in raw.columns else out['Close']
                if ticker == 'GLD':
                    gld_df = out
                else:
                    gc_df = out
            except Exception:
                continue

        if gc_df is not None and not gc_df.empty and gld_df is not None and not gld_df.empty:
            # 장기(선물) 구간 + ETF 구간을 수익률 연속성 있게 연결.
            gold_df = _chain_link_series(gc_df, gld_df)
        elif gc_df is not None and not gc_df.empty:
            gold_df = gc_df
        elif gld_df is not None and not gld_df.empty:
            gold_df = gld_df
        else:
            st.warning("금 딥프록시 소스(GLD, GC=F) 모두 로딩 실패.")
            return None

        gold_usd = gold_df['Adj Close'] if 'Adj Close' in gold_df.columns else gold_df['Close']
        merged = pd.concat([gold_usd, usdkrw], axis=1, keys=['gold', 'fx'])
        merged = merged.ffill().bfill().dropna()
        price_krw = merged['gold'] * merged['fx']
        result = pd.DataFrame(index=price_krw.index)
        result['Close'] = price_krw.values.astype(float)
        result['Adj Close'] = result['Close']
        return result
    except Exception as e:
        st.warning(f"딥프록시(금) 로딩 오류: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_benchmark_etf(ticker, start_date, end_date):
    """보조 벤치마크 ETF/펀드 데이터 로딩. 실패해도 None 반환."""
    try:
        df = fdr.DataReader(ticker, start_date, end_date)
        if df is None or df.empty: return None
        df = df[~df.index.duplicated(keep='last')].sort_index()
        if 'Close' not in df.columns: return None
        return df
    except Exception:
        return None


def get_price_at_date(df, target_date, price_col="Close"):
    if df is None or df.empty: return None
    col = price_col if price_col in df.columns else "Close"
    if col not in df.columns: return None
    v = df[col].asof(target_date)
    if pd.isna(v): return None
    return float(v)


def harmonize_gold_momentum_scale(all_data, current_date, rt_kodex_price, price_col="Close"):
    """금 모멘텀 시계열 스케일 불일치(캐시 혼선) 자동 보정."""
    if rt_kodex_price is None or rt_kodex_price <= 0:
        return all_data.get('금현물_모멘텀')

    mom_data = all_data.get('금현물_모멘텀')
    if mom_data is None or mom_data.empty:
        return mom_data

    mom_px = get_price_at_date(mom_data, current_date, price_col=price_col)
    if mom_px is None or mom_px <= 0:
        return mom_data

    scale_ratio = max(mom_px, rt_kodex_price) / max(min(mom_px, rt_kodex_price), 1e-9)
    if scale_ratio < 5:
        return mom_data

    # 0064K0 시계열을 즉시 재조회해 금 신호 스케일을 맞춘다.
    fresh = _read_fdr_with_fallback('0064K0', current_date - relativedelta(months=18), current_date)
    if fresh is None or fresh.empty or 'Close' not in fresh.columns:
        return mom_data

    fresh = fresh[~fresh.index.duplicated(keep='last')].sort_index()
    fixed = pd.DataFrame(index=fresh.index)
    fixed['Close'] = fresh['Close'].astype(float)
    fixed['Adj Close'] = fresh['Adj Close'].astype(float) if 'Adj Close' in fresh.columns else fixed['Close']
    # 표시용 계산에서만 스케일 보정을 적용하고, 원본 all_data는 유지한다.
    return fixed

def build_trading_calendar(all_data, start_date, end_date, anchor_name='코스피200'):
    anchor_df = all_data.get(anchor_name)
    if anchor_df is not None and len(anchor_df) > 0:
        dates = [d for d in anchor_df.index if start_date <= d <= end_date]
        if len(dates) > 0: return sorted(dates)
    cash_df = all_data.get(CASH_NAME)
    if cash_df is not None and len(cash_df) > 0:
        dates = [d for d in cash_df.index if start_date <= d <= end_date]
        if len(dates) > 0: return sorted(dates)
    all_dates = set()
    for name, df in all_data.items():
        if df is not None and len(df) > 0 and not name.endswith('_모멘텀'):
            all_dates.update(df.index)
    return sorted(d for d in all_dates if start_date <= d <= end_date)


def _is_month_end_rebalance_day(trading_dates, idx):
    """현재 인덱스가 '월말 거래일(다음 거래일이 다음 달)'인지 판단.

    마지막 인덱스의 경우 next-element 가 없으므로, 해당 날짜 이후
    같은 달의 영업일이 남아있는지를 달력으로 직접 확인한다.
    남은 영업일이 없으면 → 월말 리밸런싱일로 판정.
    """
    if trading_dates is None or idx < 0 or idx >= len(trading_dates):
        return False
    d = trading_dates[idx]
    # 마지막 원소: 다음 달력 영업일이 이미 다음 달인지 확인
    if idx == len(trading_dates) - 1:
        month_end = (d + relativedelta(months=1)).replace(day=1) - timedelta(days=1)
        remaining = pd.bdate_range(d + timedelta(days=1), month_end)
        return len(remaining) == 0
    nd = trading_dates[idx + 1]
    return (nd.month != d.month) or (nd.year != d.year)


def _collect_month_end_dates(trading_dates):
    """완전한 월말 거래일만 반환. 마지막 미완성 월(중간 종료)은 제외."""
    if trading_dates is None or len(trading_dates) < 2:
        return []
    # 마지막 원소도 _is_month_end_rebalance_day 가 올바르게 처리하므로
    # trading_dates[:-1] 슬라이싱 불필요 — 전체 순회
    return [d for i, d in enumerate(trading_dates) if _is_month_end_rebalance_day(trading_dates, i)]


@st.cache_data(ttl=3600)
def load_market_data(start_date, end_date, use_proxy=False, hybrid=False):
    """시장 데이터 로딩.
    - use_proxy=False: 실제 ETF만 (실전 모드)
    - use_proxy=True: 프록시만 (순수 백테스트)
    - hybrid=True: 프록시 + 실제 ETF 체인링크 (ETF 상장 후는 실제 데이터 사용)
    """
    if hybrid:
        return _load_hybrid_data(start_date, end_date)
    
    all_data = {}
    for name, ticker in ASSETS.items():
        if use_proxy:
            proxy_df = fetch_proxy_data(name, start_date, end_date)
            all_data[name] = proxy_df
            all_data[f"{name}_모멘텀"] = proxy_df
        else:
            all_data[name] = fetch_etf_data(ticker, start_date, end_date, is_momentum=False)
            if ticker == '411060':
                all_data[f"{name}_모멘텀"] = fetch_etf_data(ticker, start_date, end_date, is_momentum=True)
            else:
                all_data[f"{name}_모멘텀"] = all_data[name]
    if use_proxy:
        all_data[CASH_NAME] = fetch_proxy_data(CASH_NAME, start_date, end_date)
        all_data[f"{CASH_NAME}_모멘텀"] = all_data[CASH_NAME]
    else:
        all_data[CASH_NAME] = fetch_etf_data(CASH_TICKER, start_date, end_date, is_momentum=False)
        all_data[f"{CASH_NAME}_모멘텀"] = all_data[CASH_NAME]
    return all_data


def _chain_link_series(proxy_df, etf_df):
    """프록시 → 실제 ETF 체인링크.
    Close와 Adj Close만 포함하는 통일된 DataFrame을 반환.

    스케일링 방식:
    - ETF 초기 최대 20거래일의 ratio(ETF가격/프록시가격) 중앙값 사용.
    - 이상치 비율(ratio가 중앙값의 ±50% 초과)은 계산에서 제외.
    - 연결 시점 전후 수익률 연속성 검증: 연결 직전/직후 1일 수익률 차이가
      10%p를 넘으면 경고를 출력(graceful — 데이터는 그대로 반환).
    """
    if proxy_df is None or proxy_df.empty:
        return etf_df
    if etf_df is None or etf_df.empty:
        return proxy_df

    col = 'Close'
    if col not in proxy_df.columns or col not in etf_df.columns:
        return etf_df if etf_df is not None else proxy_df

    # ETF DataFrame을 Close/Adj Close만 남기고 정리
    etf_clean = pd.DataFrame(index=etf_df.index)
    etf_clean['Close'] = etf_df['Close'].astype(float)
    etf_clean['Adj Close'] = (
        etf_df['Adj Close'].astype(float) if 'Adj Close' in etf_df.columns
        else etf_df['Close'].astype(float)
    )

    # 프록시도 동일하게 정리
    proxy_clean = pd.DataFrame(index=proxy_df.index)
    proxy_clean['Close'] = proxy_df['Close'].astype(float)
    proxy_clean['Adj Close'] = (
        proxy_df['Adj Close'].astype(float) if 'Adj Close' in proxy_df.columns
        else proxy_df['Close'].astype(float)
    )

    # ETF 데이터 시작일
    etf_start = etf_clean.index.min()

    # 스케일링 비율: ETF 초기 최대 20거래일의 중앙값
    early_etf = etf_clean.head(20)
    ratios = []
    for d in early_etf.index:
        ep = float(early_etf.loc[d, col])
        pp = proxy_clean[col].asof(d)
        if pd.notna(pp) and pp > 0 and ep > 0:
            ratios.append(ep / pp)

    if len(ratios) == 0:
        return etf_clean

    # 1차 중앙값으로 이상치 필터링 후 재계산
    median_ratio = float(np.median(ratios))
    filtered = [r for r in ratios if 0.5 * median_ratio <= r <= 1.5 * median_ratio]
    ratio = float(np.median(filtered)) if filtered else median_ratio

    # 프록시 가격 스케일링
    proxy_scaled = proxy_clean.copy()
    proxy_scaled['Close'] = proxy_scaled['Close'] * ratio
    proxy_scaled['Adj Close'] = proxy_scaled['Adj Close'] * ratio

    # ETF 시작 전은 스케일링된 프록시, 이후는 실제 ETF
    before = proxy_scaled[proxy_scaled.index < etf_start]
    combined = pd.concat([before, etf_clean]).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]

    # ── 연결 시점 수익률 연속성 검증 ──────────────────────────
    # 동일 연결 시점 경고는 세션당 1회만 출력한다.
    try:
        boundary_rows = combined[col].loc[
            (combined.index >= etf_start - pd.Timedelta(days=10)) &
            (combined.index <= etf_start + pd.Timedelta(days=10))
        ]
        if len(boundary_rows) >= 2:
            boundary_ret = boundary_rows.pct_change().dropna().abs()
            max_jump = float(boundary_ret.max())
            if max_jump > 0.10:
                warn_key = f"_chainlink_jump_{etf_start.date()}"
                if warn_key not in st.session_state:
                    st.session_state[warn_key] = True
                    st.warning(
                        f"⚠️ 체인링크 연결 시점({etf_start.date()}) 근방에서 "
                        f"수익률 점프 감지: 최대 {max_jump*100:.1f}%. "
                        "ratio 재계산에도 불구하고 완전히 제거되지 않을 수 있습니다."
                    )
    except Exception:
        pass

    return combined


def _load_hybrid_data(start_date, end_date):
    """하이브리드 로딩: 3계층 체인링크 (딥프록시 → 프록시 → 실제ETF).
    - 딥프록시: 2000-01-01~ (FRED/ECOS/FDR)
    - 프록시: ETF 상장 전 보완 (KODEX200, QQQ×환율, KOSEF10년 등)
    - 실제 ETF: 상장 후 실거래 데이터
    """
    all_data = {}

    # ── 코스피200 ──────────────────────────────────────────────
    deep_kospi = fetch_deep_proxy_kospi(start_date, end_date)
    proxy_kospi = fetch_proxy_data('코스피200', start_date, end_date)   # KODEX200(069500)
    etf_kospi = fetch_etf_data('294400', start_date, end_date)
    step1 = _chain_link_series(deep_kospi, proxy_kospi)
    all_data['코스피200'] = _chain_link_series(step1, etf_kospi)
    all_data['코스피200_모멘텀'] = all_data['코스피200']

    # ── 미국나스닥100 ─────────────────────────────────────────
    # QQQ 시작일 1999-03-10 → 2000-01-01부터 갭 없음. 딥프록시 불필요.
    proxy_qqq = fetch_proxy_data('미국나스닥100', start_date, end_date)  # QQQ×USD/KRW
    etf_qqq = fetch_etf_data('133690', start_date, end_date)
    all_data['미국나스닥100'] = _chain_link_series(proxy_qqq, etf_qqq)
    all_data['미국나스닥100_모멘텀'] = all_data['미국나스닥100']

    # ── 한국채30년 ────────────────────────────────────────────
    # 실제 30년 국고채 금리(ECOS)가 잡히면 딥프록시가 2012~현재를 정확히 커버하므로,
    # 변동성·수익을 과대계상하던 KOSEF10년×2.5 프록시 tier를 건너뛰고 실 ETF에 바로 연결.
    # (키가 없으면 딥프록시가 10년물 기반이라 KOSEF tier로 보완한다.)
    deep_kr_bond = fetch_deep_proxy_kr_bond_ecos(start_date, end_date)
    etf_kr_bond = fetch_etf_data('439870', start_date, end_date)
    if fetch_kr_30y_bond_yield_series(start_date, end_date) is not None:
        all_data['한국채30년'] = _chain_link_series(deep_kr_bond, etf_kr_bond)
    else:
        proxy_kr_bond = fetch_proxy_data('한국채30년', start_date, end_date)  # KOSEF10년×2.5배
        step1 = _chain_link_series(deep_kr_bond, proxy_kr_bond)
        all_data['한국채30년'] = _chain_link_series(step1, etf_kr_bond)
    all_data['한국채30년_모멘텀'] = all_data['한국채30년']

    # ── 미국채30년 ────────────────────────────────────────────
    deep_us_bond = fetch_deep_proxy_us_bond_fred(start_date, end_date)
    proxy_us_bond = fetch_proxy_data('미국채30년', start_date, end_date)  # TLT×USD/KRW
    etf_us_bond = fetch_etf_data('476760', start_date, end_date)
    step1 = _chain_link_series(deep_us_bond, proxy_us_bond)
    all_data['미국채30년'] = _chain_link_series(step1, etf_us_bond)
    all_data['미국채30년_모멘텀'] = all_data['미국채30년']

    # ── 금현물 ────────────────────────────────────────────────
    deep_gold = fetch_deep_proxy_gold_fred(start_date, end_date)
    proxy_gold = fetch_proxy_data('금현물', start_date, end_date)        # GLD×USD/KRW
    etf_gold = fetch_etf_data('411060', start_date, end_date)
    step1 = _chain_link_series(deep_gold, proxy_gold)
    all_data['금현물'] = _chain_link_series(step1, etf_gold)
    # 모멘텀 신호:
    # 과거(딥프록시→GLD×환율) + 최근(0064K0) 체인으로 구성해
    # 0064K0 상장 이전 구간도 백테스트 신호가 끊기지 않게 한다.
    step1_mom = step1
    kodex_gold = _read_fdr_with_fallback('0064K0', start_date, end_date)
    if kodex_gold is not None and not kodex_gold.empty and 'Close' in kodex_gold.columns:
        kodex_gold = kodex_gold[~kodex_gold.index.duplicated(keep='last')].sort_index()
        kodex_mom = pd.DataFrame(index=kodex_gold.index)
        kodex_mom['Close'] = kodex_gold['Close'].astype(float)
        if 'Adj Close' in kodex_gold.columns:
            kodex_mom['Adj Close'] = kodex_gold['Adj Close'].astype(float)
        else:
            kodex_mom['Adj Close'] = kodex_mom['Close']
        all_data['금현물_모멘텀'] = _chain_link_series(step1_mom, kodex_mom)
    else:
        all_data['금현물_모멘텀'] = step1_mom

    # ── 현금(MMF) ─────────────────────────────────────────────
    proxy_cash = fetch_proxy_data(CASH_NAME, start_date, end_date)
    etf_cash = fetch_etf_data(CASH_TICKER, start_date, end_date, is_momentum=False)
    all_data[CASH_NAME] = _chain_link_series(proxy_cash, etf_cash)
    all_data[f"{CASH_NAME}_모멘텀"] = all_data[CASH_NAME]

    return all_data


@st.cache_data(ttl=3600)
def _fetch_slot_proxy(slot_config, start_date, end_date):
    """슬롯 프록시 데이터 로딩."""
    try:
        ptype = slot_config['proxy_type']
        ticker = slot_config['proxy']
        if ptype == 'kr_etf' or ptype == 'kr_stock':
            df = fdr.DataReader(ticker, start_date, end_date)
            if df is None or df.empty: return None
            df = df[~df.index.duplicated(keep='last')].sort_index()
            if 'Close' not in df.columns: return None
            r = pd.DataFrame(index=df.index)
            r['Close'] = df['Close']
            r['Adj Close'] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            return r
        elif ptype == 'us_etf_fx':
            us = _read_us_market_data(ticker, start_date, end_date)
            fx = get_usdkrw_series(start_date, end_date) if slot_config.get('fx') == 'USD/KRW' else fdr.DataReader(slot_config['fx'], start_date, end_date)
            if us is None or us.empty or fx is None or fx.empty: return None
            us = us[~us.index.duplicated(keep='last')]
            fx = fx[~fx.index.duplicated(keep='last')]
            uc = us['Adj Close'] if 'Adj Close' in us.columns else us['Close']
            m = pd.concat([uc, fx['Close']], axis=1, keys=['US', 'FX']).ffill().bfill().dropna()
            r = pd.DataFrame(index=m.index)
            r['Close'] = m['US'] * m['FX']; r['Adj Close'] = r['Close']
            return r
        elif ptype == 'us_etf':
            us = _read_us_market_data(ticker, start_date, end_date)
            if us is None or us.empty:
                return None
            us = us[~us.index.duplicated(keep='last')].sort_index()
            if 'Close' not in us.columns:
                return None
            r = pd.DataFrame(index=us.index)
            r['Close'] = us['Close']
            r['Adj Close'] = us['Adj Close'] if 'Adj Close' in us.columns else us['Close']
            return r
        return None
    except Exception as e:
        st.warning(f"슬롯 프록시 오류 ({ticker}): {e}")
        return None


# ==============================
# 5. 모멘텀/비중
# ==============================
def calculate_momentum_score_detail_at_date(ticker, as_of_date, historical_data, price_col="Close"):
    try:
        if historical_data is None or len(historical_data) == 0: return None, None, None, 0
        current_price = get_price_at_date(historical_data, as_of_date, price_col=price_col)
        if current_price is None: return None, None, None, 0
        score, valid_months = 0, 0
        for months_ago in range(1, 13):
            past_date = as_of_date - relativedelta(months=months_ago)
            month_end = get_month_end_date(past_date)
            past_price = get_price_at_date(historical_data, month_end, price_col=price_col)
            if past_price is not None:
                if current_price > past_price: score += 1
                valid_months += 1
        if valid_months < MIN_VALID_MONTHS: return current_price, None, score, valid_months
        return current_price, score / valid_months if valid_months > 0 else None, score, valid_months
    except Exception: return None, None, None, 0


def calculate_momentum_score_at_date(ticker, as_of_date, historical_data, price_col="Close"):
    current_price, score, _, _ = calculate_momentum_score_detail_at_date(
        ticker, as_of_date, historical_data, price_col=price_col
    )
    return current_price, score


def format_momentum_score(score, positive_months=None, valid_months=None):
    if pd.isna(score):
        return "-"
    if valid_months is None or pd.isna(valid_months) or int(valid_months) <= 0:
        return f"{score:.2f}"
    valid_months = int(valid_months)
    if positive_months is None or pd.isna(positive_months):
        positive_months = int(round(float(score) * valid_months))
    return f"{score:.2f}({int(positive_months)}/{valid_months})"

def calculate_weights_at_date(as_of_date, all_data, price_col="Close"):
    weights = {}
    for asset_name, ticker in ASSETS.items():
        momentum_data = all_data.get(f"{asset_name}_모멘텀")
        _, score = calculate_momentum_score_at_date(ticker, as_of_date, momentum_data, price_col=price_col)
        weights[asset_name] = 0.20 * score if score is not None else 0.0
    cash_weight = max(0.0, 1.0 - sum(weights.values()))
    weights[CASH_NAME] = cash_weight
    return weights


def build_kr_stock_bond_cash_avg_momentum_data(base_all_data, start_date, end_date):
    """KR 3자산(코스피200/국고채10년/현금) 전략용 데이터셋 구성."""
    if base_all_data is None:
        return None
    stock_df = base_all_data.get(KR_STOCK_MIX_ASSET)
    cash_df = base_all_data.get(CASH_NAME)
    if stock_df is None or stock_df.empty or cash_df is None or cash_df.empty:
        return None
    bond_df = base_all_data.get(KR_BOND_10Y_MIX_ASSET)
    if bond_df is None or bond_df.empty:
        lookback_start = start_date - relativedelta(months=18)
        bond_df = fetch_kr_bond_10y_chain_data(lookback_start, end_date)
    if bond_df is None or bond_df.empty:
        return None
    data = {
        KR_STOCK_MIX_ASSET: stock_df,
        f"{KR_STOCK_MIX_ASSET}_모멘텀": base_all_data.get(f"{KR_STOCK_MIX_ASSET}_모멘텀", stock_df),
        KR_BOND_10Y_MIX_ASSET: bond_df,
        f"{KR_BOND_10Y_MIX_ASSET}_모멘텀": bond_df,
        CASH_NAME: cash_df,
        f"{CASH_NAME}_모멘텀": base_all_data.get(f"{CASH_NAME}_모멘텀", cash_df),
    }
    return data


def calculate_kr_stock_bond_cash_avg_momentum_weights(as_of_date, strategy_data, cash_score=1.0, price_col="Adj Close"):
    """코스피200/국고채10년/현금(점수 1 고정) 비중형 평균 모멘텀."""
    if strategy_data is None:
        return {
            KR_STOCK_MIX_ASSET: 0.0,
            KR_BOND_10Y_MIX_ASSET: 0.0,
            CASH_NAME: 1.0,
        }
    stock_mom = strategy_data.get(f"{KR_STOCK_MIX_ASSET}_모멘텀")
    bond_mom = strategy_data.get(f"{KR_BOND_10Y_MIX_ASSET}_모멘텀")
    _, stock_score = calculate_momentum_score_at_date(
        KR_STOCK_MIX_ASSET, as_of_date, stock_mom, price_col=price_col
    )
    _, bond_score = calculate_momentum_score_at_date(
        KR_BOND_10Y_MIX_ASSET, as_of_date, bond_mom, price_col=price_col
    )
    raw = {
        KR_STOCK_MIX_ASSET: max(0.0, float(stock_score)) if stock_score is not None else 0.0,
        KR_BOND_10Y_MIX_ASSET: max(0.0, float(bond_score)) if bond_score is not None else 0.0,
        CASH_NAME: max(0.0, float(cash_score)),
    }
    total = float(sum(raw.values()))
    if total <= 0:
        return {
            KR_STOCK_MIX_ASSET: 0.0,
            KR_BOND_10Y_MIX_ASSET: 0.0,
            CASH_NAME: 1.0,
        }
    return {k: v / total for k, v in raw.items()}


def simulate_kr_stock_bond_cash_avg_momentum_strategy(start_date, end_date, initial_capital, strategy_data, price_col="Adj Close"):
    """KR 3자산 평균 모멘텀 비중 전략(월말 리밸런싱)."""
    if strategy_data is None:
        return None
    strategy_assets = [KR_STOCK_MIX_ASSET, KR_BOND_10Y_MIX_ASSET]

    def _rebalance_for_strategy_assets(portfolio_value, date, target_weights, holdings):
        cash_price = get_price_at_date(strategy_data.get(CASH_NAME), date, price_col=price_col)
        if cash_price is None or cash_price <= 0:
            cash_price = 10000.0
        cash_target_value = portfolio_value * float(target_weights.get(CASH_NAME, 0.0))
        for asset_name in strategy_assets:
            w = float(target_weights.get(asset_name, 0.0))
            target_value = portfolio_value * w
            px = get_price_at_date(strategy_data.get(asset_name), date, price_col=price_col)
            if px is None or px <= 0:
                holdings[asset_name] = 0.0
                cash_target_value += target_value
                continue
            holdings[asset_name] = target_value / px
        holdings[CASH_NAME] = cash_target_value / cash_price if cash_price > 0 else 0.0

    def _calc_strategy_portfolio_value(holdings, date):
        pv = 0.0
        for asset_name in strategy_assets:
            px = get_price_at_date(strategy_data.get(asset_name), date, price_col=price_col)
            if px is not None and px > 0:
                pv += holdings.get(asset_name, 0.0) * px
        cash_px = get_price_at_date(strategy_data.get(CASH_NAME), date, price_col=price_col)
        if cash_px is None or cash_px <= 0:
            cash_px = 10000.0
        pv += holdings.get(CASH_NAME, 0.0) * cash_px
        return pv

    trading_dates = build_trading_calendar(
        strategy_data, start_date, end_date, anchor_name=KR_STOCK_MIX_ASSET
    )
    if len(trading_dates) == 0:
        return None
    actual_start = trading_dates[0]
    holdings = {
        KR_STOCK_MIX_ASSET: 0.0,
        KR_BOND_10Y_MIX_ASSET: 0.0,
        CASH_NAME: 0.0,
    }
    iw = calculate_kr_stock_bond_cash_avg_momentum_weights(
        actual_start, strategy_data, cash_score=1.0, price_col=price_col
    )
    _rebalance_for_strategy_assets(initial_capital, actual_start, iw, holdings)
    daily_nav = []
    last_valid_nav = float(initial_capital)
    for i, date in enumerate(trading_dates):
        pv_raw = _calc_strategy_portfolio_value(holdings, date)
        pv = _safe_nav_value(pv_raw, last_valid_nav)
        if pv is None:
            pv = float(initial_capital)
        last_valid_nav = pv
        daily_nav.append({"date": date, "nav": pv})
        if _is_month_end_rebalance_day(trading_dates, i) and date != actual_start:
            tw = calculate_kr_stock_bond_cash_avg_momentum_weights(
                date, strategy_data, cash_score=1.0, price_col=price_col
            )
            _rebalance_for_strategy_assets(pv, date, tw, holdings)
    df = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df


def _normalize_component_weights(component_prices, target_weights):
    available = {
        name: weight
        for name, weight in target_weights.items()
        if component_prices.get(name) is not None and component_prices.get(name) > 0 and weight > 0
    }
    total = float(sum(available.values()))
    if total <= 0:
        return {}
    return {name: weight / total for name, weight in available.items()}


def _nasdaq_active_execution_targets(date):
    """나스닥 20% 슬롯 내부 집행 비중. 상장 전 데이터는 만들지 않는다."""
    d = pd.Timestamp(date)
    if d < TIME_NASDAQ_ACTIVE_LISTING_DATE:
        return {"base": 1.0}
    if d < KOACT_NASDAQ_GROWTH_ACTIVE_LISTING_DATE:
        return {"base": 0.5, "time": 0.5}
    return {"time": 0.5, "koact": 0.5}


def _kr_stock_active_execution_targets(as_of_date):
    """한국주식 20% 슬롯 실전 집행 대상.
    코스피 액티브 전환일(HAENAM_KR_ACTIVE_SWITCH_DATE) 전에는 삼성전자/SK하이닉스 50:50,
    전환일 이후에는 TIME/KoAct 코스피액티브 50:50으로 집행한다. 신호(코스피200 -5% 룰)는 불변."""
    if (HAENAM_KR_ACTIVE_SWITCH_DATE is not None
            and pd.Timestamp(as_of_date) >= pd.Timestamp(HAENAM_KR_ACTIVE_SWITCH_DATE)):
        return {"time_kospi": 0.5, "koact_kospi": 0.5}
    return {"samsung": 0.5, "hynix": 0.5}


def build_weighted_execution_series(components, target_weight_fn, trading_dates, price_col="Adj Close"):
    """여러 집행 상품을 월말 리밸런싱하는 합성 가격 시리즈."""
    components = {
        name: df
        for name, df in (components or {}).items()
        if df is not None and not df.empty
    }
    if not components or not trading_dates:
        return None

    holdings = {name: 0.0 for name in components}
    nav = 10000.0
    rows = []

    def _component_prices(date):
        return {
            name: get_price_at_date(df, date, price_col=price_col)
            for name, df in components.items()
            if df is not None and not df.empty
        }

    for i, date in enumerate(trading_dates):
        prices = _component_prices(date)
        if not rows:
            weights = _normalize_component_weights(prices, target_weight_fn(date))
            if not weights:
                continue
            for name, weight in weights.items():
                holdings[name] = (nav * weight) / prices[name]

        pv = 0.0
        for name, units in holdings.items():
            px = prices.get(name)
            if px is not None and px > 0:
                pv += units * px
        safe_nav = _safe_nav_value(pv, nav)
        nav = safe_nav if safe_nav is not None else nav
        rows.append({"date": date, "nav": nav})

        if _is_month_end_rebalance_day(trading_dates, i):
            weights = _normalize_component_weights(prices, target_weight_fn(date))
            if weights:
                holdings = {name: 0.0 for name in components}
                for name, weight in weights.items():
                    holdings[name] = (nav * weight) / prices[name]

    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["Close"] = out["nav"]
    out["Adj Close"] = out["nav"]
    return out[["Close", "Adj Close"]]


def build_nasdaq_active_execution_series(base_nasdaq_df, time_df, koact_df, trading_dates, price_col="Adj Close"):
    """나스닥 신호는 그대로 두고, 나스닥 슬롯의 집행 상품만 상장일 이후 교체한 합성 가격."""
    return build_weighted_execution_series(
        {"base": base_nasdaq_df, "time": time_df, "koact": koact_df},
        _nasdaq_active_execution_targets,
        trading_dates,
        price_col=price_col,
    )


def build_faber_nasdaq_active_execution_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """Faber A와 동일 신호를 쓰되 나스닥 20% 슬롯 집행만 액티브 ETF로 바꾼 데이터셋."""
    if base_all_data is None:
        return None
    nasdaq_name = NASDAQ100_ASSET_NAME
    if not nasdaq_name:
        return None
    base_nasdaq = base_all_data.get(nasdaq_name)
    if base_nasdaq is None or base_nasdaq.empty:
        return None

    time_df = fetch_etf_data(TIME_NASDAQ_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    koact_df = fetch_etf_data(KOACT_NASDAQ_GROWTH_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    trading_dates = build_trading_calendar(base_all_data, start_date, end_date)
    exec_df = build_nasdaq_active_execution_series(
        base_nasdaq, time_df, koact_df, trading_dates, price_col=price_col
    )
    if exec_df is None or exec_df.empty:
        return None

    data = {k: v for k, v in base_all_data.items()}
    data[nasdaq_name] = exec_df
    momentum_key = next((k for k in base_all_data.keys() if k.startswith(f"{nasdaq_name}_")), None)
    if momentum_key:
        data[momentum_key] = base_all_data.get(momentum_key, base_nasdaq)
    return data


def build_faber_kr_stock_overlay_data(base_all_data, start_date, end_date, kr_weights, price_col="Adj Close"):
    """코스피200 신호(_모멘텀 시리즈)는 그대로 유지하고, 한국주식 20% 슬롯의 실제 집행만
    지정한 비중으로 데이터가 있는 구간부터 덮어쓴다.
    kr_weights 예: {"samsung": 0.5, "hynix": 0.5}(해남 A) / {"samsung": 1.0} / {"hynix": 1.0}
    """
    if base_all_data is None:
        return None
    base_slot_df = base_all_data.get(KR_STOCK_MIX_ASSET)
    if base_slot_df is None or base_slot_df.empty:
        return None

    components = {}
    if float(kr_weights.get("samsung", 0.0)) > 0:
        components["samsung"] = fetch_etf_data(SAMSUNG_ELECTRONICS_TICKER, start_date, end_date, is_momentum=False)
    if float(kr_weights.get("hynix", 0.0)) > 0:
        components["hynix"] = fetch_etf_data(SK_HYNIX_TICKER, start_date, end_date, is_momentum=False)
    trading_dates = build_trading_calendar(base_all_data, start_date, end_date)
    exec_df = build_weighted_execution_series(
        components,
        lambda _date: kr_weights,
        trading_dates,
        price_col=price_col,
    )
    if exec_df is None or exec_df.empty:
        return None

    data = {k: v for k, v in base_all_data.items()}
    data[KR_STOCK_MIX_ASSET] = _chain_link_series(base_slot_df, exec_df)
    momentum_key = next((k for k in base_all_data.keys() if k.startswith(f"{KR_STOCK_MIX_ASSET}_")), None)
    if momentum_key:
        data[momentum_key] = base_all_data.get(momentum_key, base_slot_df)
    return data


def build_faber_kr_semiconductor_overlay_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """코스피200 신호를 유지하고, 삼성전자/SK하이닉스 데이터가 있는 구간부터 50:50 집행으로 덮어쓴다."""
    return build_faber_kr_stock_overlay_data(
        base_all_data, start_date, end_date, {"samsung": 0.5, "hynix": 0.5}, price_col=price_col
    )


def build_faber_active_nasdaq_kr_semi_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """나스닥은 액티브 ETF 집행, 한국주식은 삼전/하닉 50:50 집행으로 상장/데이터 이후만 덮어쓴다."""
    nasdaq_active_data = build_faber_nasdaq_active_execution_data(
        base_all_data, start_date, end_date, price_col=price_col
    )
    if nasdaq_active_data is None:
        return None
    return build_faber_kr_semiconductor_overlay_data(
        nasdaq_active_data, start_date, end_date, price_col=price_col
    )

def build_faber_active_nasdaq_kr_single_data(base_all_data, start_date, end_date, kr_weights, price_col="Adj Close"):
    """나스닥은 액티브 ETF 집행, 한국주식 20% 슬롯은 단일 종목(삼전 또는 하닉) 100% 집행.
    신호는 해남 A와 동일하게 코스피200/나스닥100 12개월 고점 -5% 룰을 유지한다."""
    nasdaq_active_data = build_faber_nasdaq_active_execution_data(
        base_all_data, start_date, end_date, price_col=price_col
    )
    if nasdaq_active_data is None:
        return None
    return build_faber_kr_stock_overlay_data(
        nasdaq_active_data, start_date, end_date, kr_weights, price_col=price_col
    )


def build_faber_active_nasdaq_kr_single_self_signal_data(
    base_all_data, start_date, end_date, kr_weights, price_col="Adj Close"
):
    """한국주식 슬롯의 집행 자산과 모멘텀 신호 자산을 같은 종목으로 맞춘 변형."""
    data = build_faber_active_nasdaq_kr_single_data(
        base_all_data, start_date, end_date, kr_weights, price_col=price_col
    )
    if data is None:
        return None

    components = {}
    if float(kr_weights.get("samsung", 0.0)) > 0:
        components["samsung"] = fetch_etf_data(SAMSUNG_ELECTRONICS_TICKER, start_date, end_date, is_momentum=False)
    if float(kr_weights.get("hynix", 0.0)) > 0:
        components["hynix"] = fetch_etf_data(SK_HYNIX_TICKER, start_date, end_date, is_momentum=False)

    trading_dates = build_trading_calendar(base_all_data, start_date, end_date)
    signal_df = build_weighted_execution_series(
        components,
        lambda _date: kr_weights,
        trading_dates,
        price_col=price_col,
    )
    if signal_df is None or signal_df.empty:
        return data

    momentum_key = next((k for k in data.keys() if k.startswith(f"{KR_STOCK_MIX_ASSET}_")), None)
    data[momentum_key or f"{KR_STOCK_MIX_ASSET}_모멘텀"] = signal_df
    return data


def build_haenam_s_strategy_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """해남S: 연속모멘텀 신호 + 한국주식 슬롯 삼성전자 100% + 나스닥 액티브 집행."""
    return build_faber_active_nasdaq_kr_single_data(
        base_all_data, start_date, end_date, {"samsung": 1.0}, price_col=price_col
    )


def expand_haenam_s_execution_weights(base_weights, as_of_date):
    return expand_haenam_execution_weights(base_weights, as_of_date, kr_weights={"samsung": 1.0})


def calculate_haenam_s_weights(as_of_date, strategy_data, price_col="Adj Close"):
    return calculate_weights_at_date(as_of_date, strategy_data, price_col=price_col)


def simulate_haenam_s_strategy(start_date, end_date, initial_capital, strategy_data, price_col="Adj Close"):
    result = simulate_daily_nav_with_attribution(
        start_date, end_date, initial_capital, strategy_data, price_col=price_col
    )
    return result[0] if result is not None else None


def build_haenam_m_strategy_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """해남M: 해남A 집행자산(코스피/나스닥 액티브 2종) + 연속모멘텀 신호."""
    return build_faber_active_nasdaq_kr_active_data(
        base_all_data, start_date, end_date, price_col=price_col
    )


def calculate_haenam_m_weights(as_of_date, strategy_data, price_col="Adj Close"):
    return calculate_weights_at_date(as_of_date, strategy_data, price_col=price_col)


def expand_haenam_m_execution_weights(base_weights, as_of_date):
    return expand_haenam_active_backtest_weights(base_weights, as_of_date)


def simulate_haenam_m_strategy(start_date, end_date, initial_capital, strategy_data, price_col="Adj Close"):
    result = simulate_daily_nav_with_attribution(
        start_date, end_date, initial_capital, strategy_data, price_col=price_col
    )
    return result[0] if result is not None else None


def build_haenam_p_strategy_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """해남P: 한국주식은 코스피200TR 패시브, 나스닥은 TIME/KoAct 액티브 집행, 신호는 연속모멘텀."""
    return build_faber_nasdaq_active_execution_data(
        base_all_data, start_date, end_date, price_col=price_col
    )


def _normalize_us_signal_frame(df):
    if df is None or df.empty or 'Close' not in df.columns:
        return None
    df = df[~df.index.duplicated(keep='last')].sort_index()
    out = pd.DataFrame(index=df.index)
    out['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    out['Adj Close'] = (
        pd.to_numeric(df['Adj Close'], errors='coerce')
        if 'Adj Close' in df.columns else out['Close']
    )
    out = out.dropna(subset=['Close'])
    return out if not out.empty else None


@st.cache_data(ttl=3600)
def fetch_us_local_currency_signal_data(asset_name, start_date, end_date):
    """미국 자산 모멘텀을 환율 제외 현지통화 기준으로 계산하기 위한 신호 시리즈."""
    config = PROXY_ASSETS.get(asset_name)
    if config is None or config.get('type') != 'us_etf_fx':
        return None

    ticker = config.get('ticker')
    etf_signal = _normalize_us_signal_frame(_read_us_market_data(ticker, start_date, end_date))

    if ticker == 'TLT':
        deep_signal = fetch_deep_proxy_us_bond_fred(start_date, end_date)
        return _chain_link_series(deep_signal, etf_signal)
    if ticker == 'GLD':
        deep_signal = fetch_deep_proxy_gold_fred(start_date, end_date)
        return _chain_link_series(deep_signal, etf_signal)
    return etf_signal


def build_haenam_p_local_currency_signal_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """해남P 실행 가격은 유지하고, 미국 자산 신호만 QQQ/TLT/GLD 현지통화 기준으로 교체."""
    data = build_haenam_p_strategy_data(base_all_data, start_date, end_date, price_col=price_col)
    if data is None:
        return None

    for asset_name, config in PROXY_ASSETS.items():
        if config.get('type') != 'us_etf_fx':
            continue
        local_signal = fetch_us_local_currency_signal_data(asset_name, start_date, end_date)
        if local_signal is not None and not local_signal.empty:
            data[f"{asset_name}_모멘텀"] = local_signal
    return data


def calculate_haenam_p_weights(as_of_date, strategy_data, price_col="Adj Close"):
    return calculate_weights_at_date(as_of_date, strategy_data, price_col=price_col)


def expand_haenam_p_execution_weights(base_weights, as_of_date):
    """Expand base signal weights to Haenam P execution assets."""
    return expand_haenam_execution_weights(
        base_weights, as_of_date, kr_weights={}, nasdaq_active=True
    )


def simulate_haenam_p_strategy(start_date, end_date, initial_capital, strategy_data, price_col="Adj Close"):
    result = simulate_daily_nav_with_attribution(
        start_date, end_date, initial_capital, strategy_data, price_col=price_col
    )
    return result[0] if result is not None else None


@st.cache_data(ttl=3600)
def fetch_vix_data(start_date, end_date):
    """VIX daily close for crisis-overlay research."""
    return _read_yfinance_history("^VIX", start_date, end_date)


def calculate_vix_target_equity(vix_value, max_equity=1.0):
    """Map VIX to the stock target used by the research overlay."""
    try:
        vix = float(vix_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(vix) or vix < 25:
        return None
    if vix < 40:
        target = 0.40 + ((vix - 25.0) / 15.0) * 0.30
    elif vix < 80:
        target = 0.70 + ((vix - 40.0) / 40.0) * 0.30
    else:
        target = 1.0
    return min(float(max_equity), target)


def calculate_vix_buy_step(vix_value):
    try:
        vix = float(vix_value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(vix):
        return 0.0
    if vix >= 80:
        return 1.0
    if vix >= 40:
        return 0.10
    if vix >= 25:
        return 0.01
    return 0.0


def _build_vix_signal_series(vix_data, trading_dates, price_col="Close"):
    if vix_data is None or vix_data.empty or len(trading_dates) == 0:
        return pd.Series(index=trading_dates, dtype=float)
    col = price_col if price_col in vix_data.columns else "Close"
    if col not in vix_data.columns:
        return pd.Series(index=trading_dates, dtype=float)
    vix = pd.to_numeric(vix_data[col], errors="coerce")
    vix = vix[~vix.index.duplicated(keep="last")].sort_index()
    # Korean-market execution cannot know the same-date US close, so use the
    # previous available VIX close.
    return vix.reindex(trading_dates).ffill().shift(1)


def _build_stock_recovery_basket(strategy_data, trading_dates, price_col="Adj Close"):
    if strategy_data is None or len(trading_dates) == 0:
        return pd.Series(index=trading_dates, dtype=float)
    kr = strategy_data.get(KR_STOCK_MIX_ASSET)
    us = strategy_data.get(NASDAQ100_ASSET_NAME)
    kr_vals, us_vals = [], []
    for d in trading_dates:
        kr_vals.append(get_price_at_date(kr, d, price_col=price_col))
        us_vals.append(get_price_at_date(us, d, price_col=price_col))
    kr_s = pd.Series(kr_vals, index=trading_dates, dtype=float).ffill()
    us_s = pd.Series(us_vals, index=trading_dates, dtype=float).ffill()
    if kr_s.dropna().empty or us_s.dropna().empty:
        return pd.Series(index=trading_dates, dtype=float)
    kr_base = float(kr_s.dropna().iloc[0])
    us_base = float(us_s.dropna().iloc[0])
    if kr_base <= 0 or us_base <= 0:
        return pd.Series(index=trading_dates, dtype=float)
    return 0.5 * (kr_s / kr_base) + 0.5 * (us_s / us_base)


def _apply_vix_equity_target(base_weights, target_equity):
    if target_equity is None:
        return dict(base_weights)
    out = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
    kr_base = float(base_weights.get(KR_STOCK_MIX_ASSET, 0.0) or 0.0)
    us_base = float(base_weights.get(NASDAQ100_ASSET_NAME, 0.0) or 0.0)
    base_stock = kr_base + us_base
    stock_target = max(base_stock, min(1.0, float(target_equity)))
    overlay = max(0.0, stock_target - base_stock)
    out[KR_STOCK_MIX_ASSET] = kr_base + overlay * 0.5
    out[NASDAQ100_ASSET_NAME] = us_base + overlay * 0.5
    total_stock = out[KR_STOCK_MIX_ASSET] + out[NASDAQ100_ASSET_NAME]
    if total_stock > 1.0:
        out[KR_STOCK_MIX_ASSET] /= total_stock
        out[NASDAQ100_ASSET_NAME] /= total_stock
        total_stock = 1.0
    out[CASH_NAME] = max(0.0, 1.0 - total_stock)
    return out


def simulate_haenam_p_vix_overlay_strategy(
    start_date,
    end_date,
    initial_capital,
    strategy_data,
    vix_data,
    max_equity=1.0,
    price_col="Adj Close",
):
    """HaenamP engine plus a research-only VIX crisis-buying overlay.

    The base HaenamP signal is unchanged. When VIX is active, defensive slots
    are converted to cash and extra stock exposure is split evenly between the
    KOSPI200 and Nasdaq execution slots.
    """
    if strategy_data is None or vix_data is None or vix_data.empty:
        return None, None
    trading_dates = build_trading_calendar(strategy_data, start_date, end_date)
    if len(trading_dates) == 0:
        return None, None

    actual_start = trading_dates[0]
    vix_signal = _build_vix_signal_series(vix_data, trading_dates)
    stock_basket = _build_stock_recovery_basket(strategy_data, trading_dates, price_col=price_col)
    base_weights = calculate_haenam_p_weights(actual_start, strategy_data, price_col=price_col)
    current_vix_target = None
    target_weights = dict(base_weights)
    holdings = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
    current_weights = rebalance_holdings(
        initial_capital, actual_start, target_weights, holdings, strategy_data, price_col=price_col
    )

    daily_nav, overlay_rows = [], []
    monthly_rebalance_dates = [actual_start]
    last_valid_nav = float(initial_capital)
    cycle_peak = None

    for i, date in enumerate(trading_dates):
        portfolio_raw = _calc_portfolio_value(holdings, date, strategy_data, price_col)
        portfolio_value = _safe_nav_value(portfolio_raw, last_valid_nav)
        if portfolio_value is None:
            portfolio_value = last_valid_nav
        last_valid_nav = portfolio_value
        daily_nav.append({"date": date, "nav": portfolio_value})

        is_month_end = _is_month_end_rebalance_day(trading_dates, i)
        if is_month_end and date != actual_start:
            base_weights = calculate_haenam_p_weights(date, strategy_data, price_col=price_col)
            monthly_rebalance_dates.append(date)

        basket_value = stock_basket.loc[date] if date in stock_basket.index else np.nan
        if cycle_peak is not None and np.isfinite(basket_value) and basket_value >= cycle_peak:
            current_vix_target = None
            cycle_peak = None

        vix_value = vix_signal.loc[date] if date in vix_signal.index else np.nan
        desired = calculate_vix_target_equity(vix_value, max_equity=max_equity)
        if desired is not None:
            if cycle_peak is None:
                prior_basket = stock_basket.loc[:date].dropna()
                cycle_peak = float(prior_basket.max()) if not prior_basket.empty else None
            step = calculate_vix_buy_step(vix_value)
            start_target = current_vix_target
            if start_target is None:
                start_target = (
                    float(base_weights.get(KR_STOCK_MIX_ASSET, 0.0) or 0.0)
                    + float(base_weights.get(NASDAQ100_ASSET_NAME, 0.0) or 0.0)
                )
            current_vix_target = min(float(desired), float(start_target) + step)

        if current_vix_target is None:
            next_target_weights = dict(base_weights)
        else:
            next_target_weights = _apply_vix_equity_target(base_weights, current_vix_target)

        should_rebalance = is_month_end or next_target_weights != target_weights
        if should_rebalance and date != trading_dates[-1]:
            target_weights = next_target_weights
            current_weights = rebalance_holdings(
                portfolio_value, date, target_weights, holdings, strategy_data, price_col=price_col
            )

        overlay_rows.append(
            {
                "date": date,
                "vix": vix_value,
                "vix_target_equity": current_vix_target,
                "stock_weight": (
                    float(current_weights.get(KR_STOCK_MIX_ASSET, 0.0) or 0.0)
                    + float(current_weights.get(NASDAQ100_ASSET_NAME, 0.0) or 0.0)
                ),
                "cash_weight": float(current_weights.get(CASH_NAME, 0.0) or 0.0),
                "cycle_active": cycle_peak is not None,
            }
        )

    df_nav = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df_nav["running_max"] = df_nav["nav"].expanding().max()
    df_nav["drawdown"] = (df_nav["nav"] - df_nav["running_max"]) / df_nav["running_max"]
    overlay_df = pd.DataFrame(overlay_rows).set_index("date").sort_index()
    return df_nav, overlay_df


def _kr_valueup_active_backtest_targets(date):
    """해남V 한국주식 슬롯 집행 타깃.
    코리아밸류업 액티브 2종 상장 전에는 코스피200, 상장 후에는 TIME/KoAct 50:50으로 집행한다.
    """
    if pd.Timestamp(date) < KOREA_VALUEUP_ACTIVE_LISTING_DATE:
        return {"base": 1.0}
    return {"time_valueup": 0.5, "koact_valueup": 0.5}


def fetch_korea_valueup_signal_data(base_slot_df, start_date, end_date):
    """해남V 한국 슬롯 모멘텀 신호.
    KRX 코리아밸류업 지수 히스토리를 FDR에서 직접 받기 어려운 환경에서는
    TR 지수 추종 ETF(495550)를 신호 프록시로 체인링크한다. 상장 전 구간은 코스피200 신호를 유지한다.
    """
    signal_df = fetch_etf_data(KOREA_VALUEUP_TR_PROXY_TICKER, start_date, end_date, is_momentum=False)
    if signal_df is None or signal_df.empty:
        return base_slot_df
    return _chain_link_series(base_slot_df, signal_df)


def build_haenam_v_kr_valueup_overlay_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """해남V: 한국주식 슬롯만 코스피200→TIME/KoAct 코리아밸류업액티브로 교체.
    신호는 상장 전 코스피200, 상장 후 코리아밸류업 지수 프록시를 사용한다.
    """
    if base_all_data is None:
        return None
    base_slot_df = base_all_data.get(KR_STOCK_MIX_ASSET)
    if base_slot_df is None or base_slot_df.empty:
        return None

    time_df = fetch_etf_data(TIME_KOREA_VALUEUP_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    koact_df = fetch_etf_data(KOACT_KOREA_VALUEUP_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    trading_dates = build_trading_calendar(base_all_data, start_date, end_date)
    exec_df = build_weighted_execution_series(
        {"base": base_slot_df, "time_valueup": time_df, "koact_valueup": koact_df},
        _kr_valueup_active_backtest_targets,
        trading_dates,
        price_col=price_col,
    )
    if exec_df is None or exec_df.empty:
        return None

    data = {k: v for k, v in base_all_data.items()}
    data[KR_STOCK_MIX_ASSET] = exec_df
    data[f"{KR_STOCK_MIX_ASSET}_모멘텀"] = fetch_korea_valueup_signal_data(
        base_slot_df, start_date, end_date
    )
    return data


def build_haenam_v_strategy_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """해남V: 나스닥은 기존 해남 액티브 집행, 한국은 코리아밸류업액티브 집행/밸류업 신호."""
    nasdaq_active_data = build_faber_nasdaq_active_execution_data(
        base_all_data, start_date, end_date, price_col=price_col
    )
    if nasdaq_active_data is None:
        return None
    return build_haenam_v_kr_valueup_overlay_data(
        nasdaq_active_data, start_date, end_date, price_col=price_col
    )


def _kr_valueup_passive_backtest_targets(date):
    """해남V 패시브 한국주식 슬롯 집행 타깃.
    코리아밸류업 ETF 상장 전에는 코스피200, 상장 후에는 SOL 코리아밸류업TR로 집행한다.
    """
    if pd.Timestamp(date) < KOREA_VALUEUP_ACTIVE_LISTING_DATE:
        return {"base": 1.0}
    return {"valueup_tr": 1.0}


def build_haenam_v_kr_valueup_passive_overlay_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """해남V 패시브: 한국주식 슬롯 집행을 SOL 코리아밸류업TR로 교체하고 밸류업 TR 신호를 사용한다."""
    if base_all_data is None:
        return None
    base_slot_df = base_all_data.get(KR_STOCK_MIX_ASSET)
    if base_slot_df is None or base_slot_df.empty:
        return None

    valueup_tr_df = fetch_etf_data(KOREA_VALUEUP_PASSIVE_TICKER, start_date, end_date, is_momentum=False)
    trading_dates = build_trading_calendar(base_all_data, start_date, end_date)
    exec_df = build_weighted_execution_series(
        {"base": base_slot_df, "valueup_tr": valueup_tr_df},
        _kr_valueup_passive_backtest_targets,
        trading_dates,
        price_col=price_col,
    )
    if exec_df is None or exec_df.empty:
        return None

    data = {k: v for k, v in base_all_data.items()}
    data[KR_STOCK_MIX_ASSET] = exec_df
    data[f"{KR_STOCK_MIX_ASSET}_모멘텀"] = fetch_korea_valueup_signal_data(
        base_slot_df, start_date, end_date
    )
    return data


def build_haenam_v_passive_strategy_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """해남V 패시브: 나스닥은 기존 해남 액티브 집행, 한국은 SOL 코리아밸류업TR 집행/밸류업 TR 신호."""
    nasdaq_active_data = build_faber_nasdaq_active_execution_data(
        base_all_data, start_date, end_date, price_col=price_col
    )
    if nasdaq_active_data is None:
        return None
    return build_haenam_v_kr_valueup_passive_overlay_data(
        nasdaq_active_data, start_date, end_date, price_col=price_col
    )


def expand_haenam_v_backtest_weights(base_weights, as_of_date):
    """Expand base signal weights to the Haenam V active value-up backtest assets."""
    out = {}
    for asset, weight in (base_weights or {}).items():
        w = float(weight or 0.0)
        if w <= 0:
            continue
        if asset == KR_STOCK_MIX_ASSET:
            targets = _kr_valueup_active_backtest_targets(as_of_date)
            out[KR_STOCK_MIX_ASSET] = out.get(KR_STOCK_MIX_ASSET, 0.0) + w * targets.get("base", 0.0)
            out[HAENAM_VALUEUP_TIME_NAME] = out.get(HAENAM_VALUEUP_TIME_NAME, 0.0) + w * targets.get("time_valueup", 0.0)
            out[HAENAM_VALUEUP_KOACT_NAME] = out.get(HAENAM_VALUEUP_KOACT_NAME, 0.0) + w * targets.get("koact_valueup", 0.0)
        elif asset == NASDAQ100_ASSET_NAME:
            targets = _nasdaq_active_execution_targets(as_of_date)
            out[NASDAQ100_ASSET_NAME] = out.get(NASDAQ100_ASSET_NAME, 0.0) + w * targets.get("base", 0.0)
            out[HAENAM_TIME_NAME] = out.get(HAENAM_TIME_NAME, 0.0) + w * targets.get("time", 0.0)
            out[HAENAM_KOACT_NAME] = out.get(HAENAM_KOACT_NAME, 0.0) + w * targets.get("koact", 0.0)
        else:
            out[asset] = out.get(asset, 0.0) + w
    invested = sum(v for k, v in out.items() if k != CASH_NAME)
    out[CASH_NAME] = max(0.0, 1.0 - invested)
    return {k: v for k, v in out.items() if v > 0.000001}


def expand_haenam_v_passive_backtest_weights(base_weights, as_of_date):
    """Expand base signal weights to the Haenam V passive value-up backtest assets."""
    out = {}
    for asset, weight in (base_weights or {}).items():
        w = float(weight or 0.0)
        if w <= 0:
            continue
        if asset == KR_STOCK_MIX_ASSET:
            targets = _kr_valueup_passive_backtest_targets(as_of_date)
            out[KR_STOCK_MIX_ASSET] = out.get(KR_STOCK_MIX_ASSET, 0.0) + w * targets.get("base", 0.0)
            out[HAENAM_VALUEUP_PASSIVE_NAME] = out.get(HAENAM_VALUEUP_PASSIVE_NAME, 0.0) + w * targets.get("valueup_tr", 0.0)
        elif asset == NASDAQ100_ASSET_NAME:
            targets = _nasdaq_active_execution_targets(as_of_date)
            out[NASDAQ100_ASSET_NAME] = out.get(NASDAQ100_ASSET_NAME, 0.0) + w * targets.get("base", 0.0)
            out[HAENAM_TIME_NAME] = out.get(HAENAM_TIME_NAME, 0.0) + w * targets.get("time", 0.0)
            out[HAENAM_KOACT_NAME] = out.get(HAENAM_KOACT_NAME, 0.0) + w * targets.get("koact", 0.0)
        else:
            out[asset] = out.get(asset, 0.0) + w
    invested = sum(v for k, v in out.items() if k != CASH_NAME)
    out[CASH_NAME] = max(0.0, 1.0 - invested)
    return {k: v for k, v in out.items() if v > 0.000001}


def _kr_stock_active_backtest_targets(date):
    """백테스트용 한국주식 슬롯 집행 타깃. 코스피액티브 상장 전엔 코스피200(지수),
    TIME 상장 후 TIME 100%, KoAct 상장 후 TIME/KoAct 50:50으로 스플라이스."""
    d = pd.Timestamp(date)
    if d < TIME_KOSPI_ACTIVE_LISTING_DATE:
        return {"base": 1.0}
    if d < KOACT_KOSPI_ACTIVE_LISTING_DATE:
        return {"time": 1.0}
    return {"time": 0.5, "koact": 0.5}


def build_faber_kr_active_overlay_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """코스피200 신호는 유지하고, 한국주식 20% 슬롯 집행만 코스피200→TIME→TIME/KoAct
    코스피액티브로 상장일 기준 교체한다(나스닥 액티브 집행과 동일 패턴)."""
    if base_all_data is None:
        return None
    base_slot_df = base_all_data.get(KR_STOCK_MIX_ASSET)
    if base_slot_df is None or base_slot_df.empty:
        return None
    time_df = fetch_etf_data(TIME_KOSPI_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    koact_df = fetch_etf_data(KOACT_KOSPI_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    trading_dates = build_trading_calendar(base_all_data, start_date, end_date)
    exec_df = build_weighted_execution_series(
        {"base": base_slot_df, "time": time_df, "koact": koact_df},
        _kr_stock_active_backtest_targets,
        trading_dates,
        price_col=price_col,
    )
    if exec_df is None or exec_df.empty:
        return None
    data = {k: v for k, v in base_all_data.items()}
    data[KR_STOCK_MIX_ASSET] = exec_df
    momentum_key = next((k for k in base_all_data.keys() if k.startswith(f"{KR_STOCK_MIX_ASSET}_")), None)
    if momentum_key:
        data[momentum_key] = base_all_data.get(momentum_key, base_slot_df)
    return data


def build_faber_active_nasdaq_kr_active_data(base_all_data, start_date, end_date, price_col="Adj Close"):
    """나스닥은 액티브 ETF 집행, 한국주식은 코스피200→TIME/KoAct 코스피액티브 집행."""
    nasdaq_active_data = build_faber_nasdaq_active_execution_data(
        base_all_data, start_date, end_date, price_col=price_col
    )
    if nasdaq_active_data is None:
        return None
    return build_faber_kr_active_overlay_data(
        nasdaq_active_data, start_date, end_date, price_col=price_col
    )

def get_haenam_live_price_data(base_all_data, start_date, end_date):
    """실전 화면에서 해남 A 집행 자산의 현재가/월간 성과 계산에 쓸 데이터."""
    data = {k: v for k, v in (base_all_data or {}).items()}
    data[HAENAM_SAMSUNG_NAME] = fetch_etf_data(SAMSUNG_ELECTRONICS_TICKER, start_date, end_date, is_momentum=False)
    data[HAENAM_HYNIX_NAME] = fetch_etf_data(SK_HYNIX_TICKER, start_date, end_date, is_momentum=False)
    data[HAENAM_TIME_NAME] = fetch_etf_data(TIME_NASDAQ_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    data[HAENAM_KOACT_NAME] = fetch_etf_data(KOACT_NASDAQ_GROWTH_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    data[HAENAM_KR_TIME_NAME] = fetch_etf_data(TIME_KOSPI_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    data[HAENAM_KR_KOACT_NAME] = fetch_etf_data(KOACT_KOSPI_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    return data

def expand_haenam_execution_weights(base_weights, as_of_date, kr_weights=None, nasdaq_active=True):
    """Convert base signal weights to Haenam execution assets."""
    if kr_weights is None:
        kr_weights = _kr_stock_active_execution_targets(as_of_date)
    kr_name_map = {
        "samsung": HAENAM_SAMSUNG_NAME,
        "hynix": HAENAM_HYNIX_NAME,
        "time_kospi": HAENAM_KR_TIME_NAME,
        "koact_kospi": HAENAM_KR_KOACT_NAME,
    }
    kr_parts = {
        kr_name_map[k]: float(v)
        for k, v in kr_weights.items()
        if k in kr_name_map and float(v) > 0
    }
    kr_total = sum(kr_parts.values())
    out = {}
    for asset, weight in (base_weights or {}).items():
        w = float(weight or 0.0)
        if w <= 0:
            continue
        if asset == KR_STOCK_MIX_ASSET:
            if kr_total > 0:
                for kr_name, kr_w in kr_parts.items():
                    out[kr_name] = out.get(kr_name, 0.0) + w * (kr_w / kr_total)
            else:
                out[asset] = out.get(asset, 0.0) + w
        elif asset == NASDAQ100_ASSET_NAME:
            if nasdaq_active:
                targets = _nasdaq_active_execution_targets(as_of_date)
                out[NASDAQ100_ASSET_NAME] = out.get(NASDAQ100_ASSET_NAME, 0.0) + w * targets.get("base", 0.0)
                out[HAENAM_TIME_NAME] = out.get(HAENAM_TIME_NAME, 0.0) + w * targets.get("time", 0.0)
                out[HAENAM_KOACT_NAME] = out.get(HAENAM_KOACT_NAME, 0.0) + w * targets.get("koact", 0.0)
            else:
                out[asset] = out.get(asset, 0.0) + w
        else:
            out[asset] = out.get(asset, 0.0) + w
    invested = sum(v for k, v in out.items() if k != CASH_NAME)
    out[CASH_NAME] = max(0.0, 1.0 - invested)
    return {k: v for k, v in out.items() if v > 0.000001}


def expand_haenam_active_backtest_weights(base_weights, as_of_date):
    """Expand base signal weights to the Haenam A active backtest assets."""
    out = {}
    for asset, weight in (base_weights or {}).items():
        w = float(weight or 0.0)
        if w <= 0:
            continue
        if asset == KR_STOCK_MIX_ASSET:
            targets = _kr_stock_active_backtest_targets(as_of_date)
            out[KR_STOCK_MIX_ASSET] = out.get(KR_STOCK_MIX_ASSET, 0.0) + w * targets.get("base", 0.0)
            out[HAENAM_KR_TIME_NAME] = out.get(HAENAM_KR_TIME_NAME, 0.0) + w * targets.get("time", 0.0)
            out[HAENAM_KR_KOACT_NAME] = out.get(HAENAM_KR_KOACT_NAME, 0.0) + w * targets.get("koact", 0.0)
        elif asset == NASDAQ100_ASSET_NAME:
            targets = _nasdaq_active_execution_targets(as_of_date)
            out[NASDAQ100_ASSET_NAME] = out.get(NASDAQ100_ASSET_NAME, 0.0) + w * targets.get("base", 0.0)
            out[HAENAM_TIME_NAME] = out.get(HAENAM_TIME_NAME, 0.0) + w * targets.get("time", 0.0)
            out[HAENAM_KOACT_NAME] = out.get(HAENAM_KOACT_NAME, 0.0) + w * targets.get("koact", 0.0)
        else:
            out[asset] = out.get(asset, 0.0) + w
    invested = sum(v for k, v in out.items() if k != CASH_NAME)
    out[CASH_NAME] = max(0.0, 1.0 - invested)
    return {k: v for k, v in out.items() if v > 0.000001}

def expand_haenam_signal_rows(signal_rows, as_of_date, price_data_map, price_col="Adj Close", kr_weights=None,
                              nasdaq_active=True):
    """신호표의 기준 신호 행을 실전 매수 대상 행으로 확장한다."""
    rows = []
    for row in signal_rows:
        asset = row.get("자산명")
        weight = float(row.get("추천비중", 0.0) or 0.0)
        if asset == KR_STOCK_MIX_ASSET and weight > 0:
            kr_component_map = {
                "samsung": (HAENAM_SAMSUNG_NAME, SAMSUNG_ELECTRONICS_TICKER),
                "hynix": (HAENAM_HYNIX_NAME, SK_HYNIX_TICKER),
                "time_kospi": (HAENAM_KR_TIME_NAME, TIME_KOSPI_ACTIVE_TICKER),
                "koact_kospi": (HAENAM_KR_KOACT_NAME, KOACT_KOSPI_ACTIVE_TICKER),
            }
            components = []
            selected_kr_weights = (
                _kr_stock_active_execution_targets(as_of_date)
                if kr_weights is None else kr_weights
            )
            for kr_key, kr_frac in selected_kr_weights.items():
                if kr_key in kr_component_map and kr_frac > 0:
                    kr_name, kr_ticker = kr_component_map[kr_key]
                    components.append((kr_name, kr_ticker, weight * kr_frac))
            if not components:
                components = [(asset, row.get("티커"), weight)]
        elif asset == NASDAQ100_ASSET_NAME and weight > 0:
            components = []
            if nasdaq_active:
                targets = _nasdaq_active_execution_targets(as_of_date)
                if targets.get("base", 0.0) > 0:
                    components.append((NASDAQ100_ASSET_NAME, ASSETS.get(NASDAQ100_ASSET_NAME), weight * targets["base"]))
                if targets.get("time", 0.0) > 0:
                    components.append((HAENAM_TIME_NAME, TIME_NASDAQ_ACTIVE_TICKER, weight * targets["time"]))
                if targets.get("koact", 0.0) > 0:
                    components.append((HAENAM_KOACT_NAME, KOACT_NASDAQ_GROWTH_ACTIVE_TICKER, weight * targets["koact"]))
            else:
                components.append((NASDAQ100_ASSET_NAME, ASSETS.get(NASDAQ100_ASSET_NAME), weight))
        else:
            components = [(asset, row.get("티커"), weight)]

        for exec_name, ticker, exec_weight in components:
            exec_row = dict(row)
            exec_row["신호자산"] = asset
            exec_row["자산명"] = exec_name
            exec_row["티커"] = ticker
            exec_row["추천비중"] = exec_weight
            exec_df = price_data_map.get(exec_name)
            exec_price = get_price_at_date(exec_df, as_of_date, price_col=price_col)
            if exec_price is not None:
                exec_row["현재가"] = exec_price
            if exec_name != asset:
                exec_row["기준신호"] = f"{row.get('기준신호', '-')} / {asset} 기준"
            rows.append(exec_row)
    return rows


def build_haenam_signal_display_rows(signal_rows):
    """신호표는 집행 ETF가 아니라 기준 신호 자산 단위로 표시한다."""
    rows = []
    for row in signal_rows:
        asset = row.get("자산명")
        display_row = dict(row)
        display_row["신호자산"] = asset
        display_row["자산명"] = asset
        rows.append(display_row)
    return rows


def build_faber_ex_bonds_strategy_data(base_all_data, start_date, end_date, include_china=False, include_india=False):
    """Faber A 변형(채권 제외) 전략용 데이터셋 구성."""
    if base_all_data is None:
        return None
    required = ['코스피200', '미국나스닥100', '금현물', CASH_NAME]
    for name in required:
        df = base_all_data.get(name)
        if df is None or df.empty:
            return None

    data = {
        '코스피200': base_all_data.get('코스피200'),
        '코스피200_모멘텀': base_all_data.get('코스피200_모멘텀', base_all_data.get('코스피200')),
        '미국나스닥100': base_all_data.get('미국나스닥100'),
        '미국나스닥100_모멘텀': base_all_data.get('미국나스닥100_모멘텀', base_all_data.get('미국나스닥100')),
        '금현물': base_all_data.get('금현물'),
        '금현물_모멘텀': base_all_data.get('금현물_모멘텀', base_all_data.get('금현물')),
        CASH_NAME: base_all_data.get(CASH_NAME),
        f'{CASH_NAME}_모멘텀': base_all_data.get(f'{CASH_NAME}_모멘텀', base_all_data.get(CASH_NAME)),
    }

    lookback_start = start_date - relativedelta(months=18)
    if include_china:
        china_df = fetch_china_csi300_cny_krw_chain(lookback_start, end_date)
        if china_df is None or china_df.empty:
            return None
        data[CHINA_CSI300_CNY_ASSET] = china_df
        data[f"{CHINA_CSI300_CNY_ASSET}_모멘텀"] = china_df

    if include_india:
        india_df = fetch_india_nifty_inr_krw_chain(lookback_start, end_date)
        if india_df is None or india_df.empty:
            return None
        data[INDIA_NIFTY_INR_ASSET] = india_df
        data[f"{INDIA_NIFTY_INR_ASSET}_모멘텀"] = india_df

    return data


def calculate_faber_weights_for_assets(as_of_date, strategy_data, asset_names, threshold=0.05, price_col="Adj Close"):
    """선택 자산군에 Faber A(-5%) 룰을 적용한 동일비중 타겟 비중."""
    if strategy_data is None or not asset_names:
        return {CASH_NAME: 1.0}
    n_assets = len(asset_names)
    unit_w = 1.0 / float(n_assets)
    weights = {}
    for asset_name in asset_names:
        signal_data = strategy_data.get(f"{asset_name}_모멘텀", strategy_data.get(asset_name))
        near_high = is_near_12month_high(signal_data, as_of_date, threshold=threshold, price_col=price_col)
        weights[asset_name] = unit_w if near_high else 0.0
    weights[CASH_NAME] = max(0.0, 1.0 - sum(weights.values()))
    return weights


def simulate_faber_subset_strategy(start_date, end_date, initial_capital, strategy_data, asset_names, price_col="Adj Close"):
    """선택 자산군 대상 Faber A(-5%) 동일룰 시뮬레이션."""
    if strategy_data is None or not asset_names:
        return None
    strategy_assets = list(asset_names)

    def _rebalance_subset(portfolio_value, date, target_weights, holdings):
        cash_price = get_price_at_date(strategy_data.get(CASH_NAME), date, price_col=price_col)
        if cash_price is None or cash_price <= 0:
            cash_price = 10000.0
        cash_target_value = portfolio_value * float(target_weights.get(CASH_NAME, 0.0))
        for asset_name in strategy_assets:
            w = float(target_weights.get(asset_name, 0.0))
            target_value = portfolio_value * w
            px = get_price_at_date(strategy_data.get(asset_name), date, price_col=price_col)
            if px is None or px <= 0:
                holdings[asset_name] = 0.0
                cash_target_value += target_value
                continue
            holdings[asset_name] = target_value / px
        holdings[CASH_NAME] = cash_target_value / cash_price if cash_price > 0 else 0.0

    def _calc_subset_nav(holdings, date):
        pv = 0.0
        for asset_name in strategy_assets:
            px = get_price_at_date(strategy_data.get(asset_name), date, price_col=price_col)
            if px is not None and px > 0:
                pv += holdings.get(asset_name, 0.0) * px
        cash_px = get_price_at_date(strategy_data.get(CASH_NAME), date, price_col=price_col)
        if cash_px is None or cash_px <= 0:
            cash_px = 10000.0
        pv += holdings.get(CASH_NAME, 0.0) * cash_px
        return pv

    trading_dates = build_trading_calendar(strategy_data, start_date, end_date, anchor_name='코스피200')
    if len(trading_dates) == 0:
        return None
    actual_start = trading_dates[0]
    holdings = {k: 0.0 for k in strategy_assets + [CASH_NAME]}
    iw = calculate_faber_weights_for_assets(
        actual_start, strategy_data, strategy_assets, threshold=0.05, price_col=price_col
    )
    _rebalance_subset(initial_capital, actual_start, iw, holdings)

    daily_nav = []
    last_valid_nav = float(initial_capital)
    for i, date in enumerate(trading_dates):
        pv_raw = _calc_subset_nav(holdings, date)
        pv = _safe_nav_value(pv_raw, last_valid_nav)
        if pv is None:
            pv = float(initial_capital)
        last_valid_nav = pv
        daily_nav.append({"date": date, "nav": pv})
        if _is_month_end_rebalance_day(trading_dates, i) and date != actual_start:
            tw = calculate_faber_weights_for_assets(
                date, strategy_data, strategy_assets, threshold=0.05, price_col=price_col
            )
            _rebalance_subset(pv, date, tw, holdings)

    df = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df


# ==============================
# 5-2. Faber 12-Month High Switch
# ==============================
def is_near_12month_high(historical_data, as_of_date, threshold=0.05, price_col="Close"):
    """현재 가격이 12개월 고점 대비 threshold(5%) 이내인지 판단."""
    if historical_data is None or historical_data.empty: return None
    col = price_col if price_col in historical_data.columns else "Close"
    if col not in historical_data.columns: return None
    current = get_price_at_date(historical_data, as_of_date, price_col=col)
    if current is None: return None
    # 12개월 고점: 현재 + 과거 11개 월말
    prices = [current]
    for m in range(1, 12):
        me = get_month_end_date(as_of_date - relativedelta(months=m))
        p = get_price_at_date(historical_data, me, price_col=col)
        if p is not None: prices.append(p)
    if len(prices) < 2: return None
    high_12m = max(prices)
    return current >= high_12m * (1 - threshold)


def calculate_faber_weights(as_of_date, all_data, mode='A', buffer_data=None, price_col="Adj Close"):
    """Faber 12-Month High Switch 비중 계산.
    mode A: 고점근처→20%, 나머지→현금
    mode B: 고점근처→20%, 나머지→버퍼자산(항상)
    mode C: 고점근처→20%, 나머지→버퍼자산(버퍼도 12M 고점근접일 때만, 아니면 현금)
    mode D: mode C와 동일(주로 SHV*KRW 버퍼에 사용)
    mode E: mode B와 동일(IEF USD, 환노출 없음 버퍼에 사용)
    mode F: mode B와 동일(한국10년채 버퍼에 사용)
    """
    weights = {}
    BUFFER_KEY = '_faber_buffer_'
    
    for asset_name, ticker in ASSETS.items():
        # 신호는 *_모멘텀 시리즈를 우선 사용한다.
        # 나스닥 액티브 집행형처럼 신호 자산과 실제 집행 자산을 분리할 수 있게 하기 위함이다.
        signal_data = all_data.get(f"{asset_name}_모멘텀", all_data.get(asset_name))
        near_high = is_near_12month_high(signal_data, as_of_date, threshold=0.05, price_col=price_col)
        
        # 이진: 고점 근처면 20%, 아니면 0%
        weights[asset_name] = 0.20 if near_high else 0.0
    
    remainder = max(0.0, 1.0 - sum(weights.values()))
    
    if mode == 'A':
        weights[CASH_NAME] = remainder
    elif mode in ('B', 'E', 'F'):
        weights[BUFFER_KEY] = remainder
        weights[CASH_NAME] = 0.0
    elif mode == 'C':
        # IEF도 고점 체크
        if buffer_data is not None:
            buf_near = is_near_12month_high(buffer_data, as_of_date, threshold=0.05, price_col=price_col)
            if buf_near:
                weights[BUFFER_KEY] = remainder
                weights[CASH_NAME] = 0.0
            else:
                weights[BUFFER_KEY] = 0.0
                weights[CASH_NAME] = remainder
        else:
            weights[CASH_NAME] = remainder
    elif mode == 'D':
        # SHV도 고점 체크
        if buffer_data is not None:
            buf_near = is_near_12month_high(buffer_data, as_of_date, threshold=0.05, price_col=price_col)
            if buf_near:
                weights[BUFFER_KEY] = remainder
                weights[CASH_NAME] = 0.0
            else:
                weights[BUFFER_KEY] = 0.0
                weights[CASH_NAME] = remainder
        else:
            weights[CASH_NAME] = remainder
    else:
        weights[CASH_NAME] = remainder
    
    return weights

def simulate_faber_strategy(start_date, end_date, initial_capital, all_data, mode='A',
                             buffer_df=None, price_col="Adj Close"):
    """Faber 12-Month High Switch 시뮬레이션."""
    BUFFER_KEY = '_faber_buffer_'
    trading_dates = build_trading_calendar(all_data, start_date, end_date)
    if len(trading_dates) == 0: return None
    actual_start = trading_dates[0]
    
    iw = calculate_faber_weights(actual_start, all_data, mode=mode, buffer_data=buffer_df, price_col=price_col)
    all_keys = list(ASSETS.keys()) + [BUFFER_KEY, CASH_NAME]
    holdings = {k: 0.0 for k in all_keys}
    
    # 초기 배분
    cash_px = get_price_at_date(all_data.get(CASH_NAME), actual_start, price_col=price_col)
    if cash_px is None or cash_px <= 0: cash_px = 10000.0
    cash_target = initial_capital * iw.get(CASH_NAME, 0.0)
    
    for an in ASSETS:
        w = iw.get(an, 0.0)
        px = get_price_at_date(all_data.get(an), actual_start, price_col=price_col)
        if px and px > 0: holdings[an] = (initial_capital * w) / px
        else: cash_target += initial_capital * w
    
    buf_w = iw.get(BUFFER_KEY, 0.0)
    if buf_w > 0:
        if buffer_df is not None:
            bpx = get_price_at_date(buffer_df, actual_start, price_col=price_col)
            if bpx and bpx > 0:
                holdings[BUFFER_KEY] = (initial_capital * buf_w) / bpx
            else:
                cash_target += initial_capital * buf_w
        else:
            cash_target += initial_capital * buf_w
    
    holdings[CASH_NAME] = cash_target / cash_px if cash_px > 0 else 0.0
    
    daily_nav = []
    last_valid_nav = float(initial_capital)
    for i, date in enumerate(trading_dates):
        # 포트폴리오 가치 계산
        pv_raw = 0.0
        for an in ASSETS:
            px = get_price_at_date(all_data.get(an), date, price_col=price_col)
            if px and px > 0: pv_raw += holdings.get(an, 0.0) * px
        if buffer_df is not None:
            bpx = get_price_at_date(buffer_df, date, price_col=price_col)
            if bpx and bpx > 0: pv_raw += holdings.get(BUFFER_KEY, 0.0) * bpx
        cpx = get_price_at_date(all_data.get(CASH_NAME), date, price_col=price_col)
        if cpx is None or cpx <= 0: cpx = 10000.0
        pv_raw += holdings.get(CASH_NAME, 0.0) * cpx
        pv = _safe_nav_value(pv_raw, last_valid_nav)
        if pv is None:
            pv = float(initial_capital)
        last_valid_nav = pv
        daily_nav.append({"date": date, "nav": pv})
        
        # 월말 리밸런싱
        if _is_month_end_rebalance_day(trading_dates, i) and date != trading_dates[0]:
            tw = calculate_faber_weights(date, all_data, mode=mode, buffer_data=buffer_df, price_col=price_col)
            ct = pv * tw.get(CASH_NAME, 0.0)
            for an in ASSETS:
                w = tw.get(an, 0.0)
                px = get_price_at_date(all_data.get(an), date, price_col=price_col)
                if px and px > 0: holdings[an] = (pv * w) / px
                else: ct += pv * w; holdings[an] = 0.0
            bw = tw.get(BUFFER_KEY, 0.0)
            if bw > 0:
                if buffer_df is not None:
                    bpx = get_price_at_date(buffer_df, date, price_col=price_col)
                    if bpx and bpx > 0:
                        holdings[BUFFER_KEY] = (pv * bw) / bpx
                    else:
                        ct += pv * bw
                        holdings[BUFFER_KEY] = 0.0
                else:
                    ct += pv * bw
                    holdings[BUFFER_KEY] = 0.0
            else:
                holdings[BUFFER_KEY] = 0.0
            holdings[CASH_NAME] = ct / cpx if cpx > 0 else 0.0
    
    df = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df


# ==============================
# 6. 시뮬레이션
# ==============================
def rebalance_holdings(portfolio_value, date, target_weights, holdings, all_data, price_col="Close"):
    """목표 비중으로 보유 수량을 재계산하고 holdings를 in-place로 업데이트."""
    cash_price = get_price_at_date(all_data.get(CASH_NAME), date, price_col=price_col)
    if cash_price is None or cash_price <= 0: cash_price = 10000.0
    cash_target_value = portfolio_value * float(target_weights.get(CASH_NAME, 0.0))
    new_holdings = {}
    for asset_name in ASSETS.keys():
        w = float(target_weights.get(asset_name, 0.0))
        target_value = portfolio_value * w
        px = get_price_at_date(all_data.get(asset_name), date, price_col=price_col)
        if px is None or px <= 0:
            cash_target_value += target_value
            new_holdings[asset_name] = 0.0
            continue
        new_holdings[asset_name] = target_value / px
    new_holdings[CASH_NAME] = cash_target_value / cash_price if cash_price > 0 else 0.0
    new_weights_actual = {}
    if portfolio_value > 0:
        for asset_name in ASSETS.keys():
            px = get_price_at_date(all_data.get(asset_name), date, price_col=price_col)
            new_weights_actual[asset_name] = (new_holdings[asset_name] * px) / portfolio_value if px and px > 0 else 0.0
        new_weights_actual[CASH_NAME] = (new_holdings[CASH_NAME] * cash_price) / portfolio_value
    else:
        new_weights_actual = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
    holdings.update(new_holdings)
    return new_weights_actual

def _calc_portfolio_value(holdings, date, all_data, price_col):
    pv = 0.0
    for asset_name in ASSETS.keys():
        px = get_price_at_date(all_data.get(asset_name), date, price_col=price_col)
        if px is not None and px > 0:
            pv += holdings.get(asset_name, 0.0) * px
    cash_px = get_price_at_date(all_data.get(CASH_NAME), date, price_col=price_col)
    if cash_px is None or cash_px <= 0: cash_px = 10000.0
    pv += holdings.get(CASH_NAME, 0.0) * cash_px
    return pv


def _safe_nav_value(nav_value, fallback_nav):
    """Invalid NAV(<=0/NaN/inf) 발생 시 직전 유효 NAV를 사용해 연속성 유지."""
    if nav_value is None or not np.isfinite(nav_value) or nav_value <= 0:
        if fallback_nav is not None and np.isfinite(fallback_nav) and fallback_nav > 0:
            return float(fallback_nav)
        return None
    return float(nav_value)

def simulate_daily_nav_with_attribution(start_date, end_date, initial_capital, all_data, price_col="Close"):
    trading_dates = build_trading_calendar(all_data, start_date, end_date)
    if len(trading_dates) == 0: return None, None, None, None
    actual_start = trading_dates[0]
    initial_weights = calculate_weights_at_date(actual_start, all_data, price_col=price_col)
    holdings = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
    current_weights = rebalance_holdings(initial_capital, actual_start, initial_weights, holdings, all_data, price_col=price_col)
    first_nav = _calc_portfolio_value(holdings, actual_start, all_data, price_col)
    if abs(first_nav - initial_capital) / initial_capital > 0.05:
        st.warning(f"⚠️ 첫날 NAV({first_nav:,.0f})가 초기자본({initial_capital:,.0f})과 차이납니다.")

    daily_nav, monthly_attribution, monthly_rebalance_dates = [], [], [actual_start]
    monthly_weights_history = [{"date": actual_start, **{k: v for k, v in initial_weights.items()}}]
    month_start_nav, month_start_date, month_start_weights = initial_capital, actual_start, current_weights.copy()
    last_valid_nav = float(first_nav if first_nav and first_nav > 0 else initial_capital)

    for i, date in enumerate(trading_dates):
        portfolio_raw = _calc_portfolio_value(holdings, date, all_data, price_col)
        portfolio_value = _safe_nav_value(portfolio_raw, last_valid_nav)
        if portfolio_value is None:
            portfolio_value = month_start_nav if month_start_nav > 0 else initial_capital
        last_valid_nav = portfolio_value
        daily_nav.append({"date": date, "nav": portfolio_value})
        is_month_end = _is_month_end_rebalance_day(trading_dates, i)
        if is_month_end and date != trading_dates[0]:
            target_weights = calculate_weights_at_date(date, all_data, price_col=price_col)
            current_weights = rebalance_holdings(portfolio_value, date, target_weights, holdings, all_data, price_col=price_col)
            monthly_rebalance_dates.append(date)
            monthly_weights_history.append({"date": date, **{k: v for k, v in target_weights.items()}})
        if is_month_end and date != actual_start:
            month_end_nav = portfolio_value
            month_total_return = (month_end_nav - month_start_nav) / month_start_nav if month_start_nav > 0 else 0.0
            attr = {"date": date, "total_return": month_total_return}
            for asset_name in list(ASSETS.keys()) + [CASH_NAME]:
                w = float(month_start_weights.get(asset_name, 0.0))
                if abs(w) < 1e-8: attr[asset_name] = 0.0; continue
                sp = get_price_at_date(all_data.get(asset_name), month_start_date, price_col=price_col)
                ep = get_price_at_date(all_data.get(asset_name), date, price_col=price_col)
                if asset_name == CASH_NAME:
                    if sp is None or sp <= 0: sp = 10000.0
                    if ep is None or ep <= 0: ep = sp
                attr[asset_name] = w * ((ep - sp) / sp) if sp and ep and sp > 0 else 0.0
            monthly_attribution.append(attr)
            month_start_nav, month_start_date, month_start_weights = portfolio_value, date, current_weights.copy()

    df_nav = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df_nav["running_max"] = df_nav["nav"].expanding().max()
    df_nav["drawdown"] = (df_nav["nav"] - df_nav["running_max"]) / df_nav["running_max"]
    return df_nav, monthly_attribution, monthly_rebalance_dates, monthly_weights_history


def simulate_static_benchmark(start_date, end_date, initial_capital, all_data, price_col="Close"):
    """정적 동일비중 월별 리밸런싱 벤치마크 시뮬레이션.
    현금 슬롯 없이 데이터가 있는 자산만 균등 분배 (equal_w = 1 / len(available_assets)).
    """
    trading_dates = build_trading_calendar(all_data, start_date, end_date)
    if len(trading_dates) == 0: return None
    actual_start = trading_dates[0]

    # 데이터가 있는 자산만 동일비중 배분 (현금 제외)
    available_assets = []
    for name in ASSETS.keys():
        px = get_price_at_date(all_data.get(name), actual_start, price_col=price_col)
        if px is not None and px > 0:
            available_assets.append(name)

    if len(available_assets) == 0: return None

    equal_w = 1.0 / len(available_assets)  # 현금 없이 균등 분배
    static_weights = {name: equal_w if name in available_assets else 0.0 for name in ASSETS.keys()}
    static_weights[CASH_NAME] = 0.0  # 현금 슬롯 사용 안 함

    holdings = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
    rebalance_holdings(initial_capital, actual_start, static_weights, holdings, all_data, price_col=price_col)

    daily_nav = []
    last_valid_nav = float(initial_capital)
    for i, date in enumerate(trading_dates):
        pv_raw = _calc_portfolio_value(holdings, date, all_data, price_col)
        pv = _safe_nav_value(pv_raw, last_valid_nav)
        if pv is None:
            pv = float(initial_capital)
        last_valid_nav = pv
        daily_nav.append({"date": date, "nav": pv})

        if _is_month_end_rebalance_day(trading_dates, i) and date != trading_dates[0]:
            # 해당 시점에 데이터 있는 자산만 동일비중, 현금 없음
            avail = [n for n in ASSETS.keys() if get_price_at_date(all_data.get(n), date, price_col=price_col) not in (None, 0)]
            if avail:
                ew = 1.0 / len(avail)  # 현금 없이 균등
                sw = {n: ew if n in avail else 0.0 for n in ASSETS.keys()}
            else:
                sw = {n: 0.0 for n in ASSETS.keys()}
            sw[CASH_NAME] = 0.0  # 현금 슬롯 항상 0
            rebalance_holdings(pv, date, sw, holdings, all_data, price_col=price_col)
    
    df = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df


def simulate_equal_weight_no_cash(start_date, end_date, initial_capital, all_data, price_col="Adj Close"):
    """현금 슬롯 없이 5자산 정확히 각 20% 고정, 월말 리밸런싱.
    [S-1] 클로저 의존성 제거: all_data를 인자로 받아 최상위 함수로 정의.
    [S-2] avail이 빈 경우 ZeroDivisionError 방지 가드 포함.
    """
    trading_dates = build_trading_calendar(all_data, start_date, end_date)
    if not trading_dates:
        return None
    eq_w = {name: 0.20 for name in ASSETS.keys()}
    eq_w[CASH_NAME] = 0.0
    holdings = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
    rebalance_holdings(initial_capital, trading_dates[0], eq_w, holdings, all_data, price_col=price_col)
    daily_nav = []
    last_valid_nav = float(initial_capital)
    for i, date in enumerate(trading_dates):
        pv_raw = _calc_portfolio_value(holdings, date, all_data, price_col)
        pv = _safe_nav_value(pv_raw, last_valid_nav)
        if pv is None:
            pv = float(initial_capital)
        last_valid_nav = pv
        daily_nav.append({"date": date, "nav": pv})
        if _is_month_end_rebalance_day(trading_dates, i) and date != trading_dates[0]:
            avail = [n for n in ASSETS.keys()
                     if get_price_at_date(all_data.get(n), date, price_col=price_col) not in (None, 0)]
            if avail:  # [S-2] avail=[] 시 ZeroDivisionError 방지
                sw = {n: (1.0 / len(avail)) if n in avail else 0.0 for n in ASSETS.keys()}
            else:
                sw = {n: 0.0 for n in ASSETS.keys()}
            sw[CASH_NAME] = 0.0
            rebalance_holdings(pv, date, sw, holdings, all_data, price_col=price_col)
    df = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df


def build_benchmark_etf_returns(benchmark_df, strategy_nav_df, initial_capital):
    """보조 벤치마크 ETF의 수익률을 전략 시작일 기준으로 정규화."""
    if benchmark_df is None or benchmark_df.empty or strategy_nav_df is None:
        return None
    
    start_date = strategy_nav_df.index[0]
    col = 'Adj Close' if 'Adj Close' in benchmark_df.columns else 'Close'
    
    bm = benchmark_df[col].copy()
    bm = bm[bm.index >= start_date]
    if len(bm) == 0: return None
    
    base_price = bm.iloc[0]
    if base_price <= 0: return None
    
    bm_nav = (bm / base_price) * initial_capital
    df = pd.DataFrame({"nav": bm_nav})
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df

# ==============================
# 7. 성과 지표
# ==============================
def calculate_cumulative_principal(initial_capital, cash_flows, evaluation_date):
    """평가 기준일에 이미 반영된 외부 입출금을 더한 누적 원금."""
    principal = float(initial_capital)
    if evaluation_date is None:
        return principal

    evaluation_ts = pd.Timestamp(evaluation_date).normalize()
    for date_str, amount in (cash_flows or {}).items():
        cash_flow_ts = pd.Timestamp(date_str).normalize()
        if cash_flow_ts <= evaluation_ts:
            principal += float(amount)
    return principal


@st.cache_data(ttl=60)
def load_structured_monthly_ledger(ledger_path=MONTHLY_LEDGER_CSV_PATH):
    """앱 계산용 월말 원장을 읽는다. 키는 YYYY-MM."""
    ledger = {month: row.copy() for month, row in DEFAULT_MONTHLY_LEDGER.items()}
    ledger_paths = [Path(ledger_path)]
    for fallback_path in MONTHLY_LEDGER_CSV_PATHS:
        if fallback_path not in ledger_paths:
            ledger_paths.append(fallback_path)

    try:
        df = next(
            pd.read_csv(path, dtype={"month": str})
            for path in ledger_paths
            if path.exists()
        )
    except Exception:
        return ledger

    required = {"month", "month_end_assets"}
    if not required.issubset(df.columns):
        return ledger

    numeric_cols = [
        "month_start_assets",
        "month_end_assets",
        "deposit",
        "withdrawal",
        "net_external_cash_flow",
        "official_profit",
        "official_return",
    ]
    for _, row in df.iterrows():
        month = str(row.get("month", "")).strip()
        if not re.match(r"^\d{4}-\d{2}$", month):
            continue
        item = row.to_dict()
        for col in numeric_cols:
            if col in item and pd.notna(item[col]):
                item[col] = _to_float(item[col])
        ledger[month] = item
    return ledger


@st.cache_data(ttl=60)
def load_confirmed_month_end_navs(ledger_path=MONTHLY_LEDGER_PATH):
    """월말 확정 총자산을 읽는다. CSV 원장을 우선 사용하고, 없으면 Markdown으로 fallback한다."""
    confirmed = {
        month: float(row["month_end_assets"])
        for month, row in load_structured_monthly_ledger().items()
        if _to_float(row.get("month_end_assets")) is not None
    }

    ledger_paths = [Path(ledger_path)]
    for fallback_path in MONTHLY_LEDGER_PATHS:
        if fallback_path not in ledger_paths:
            ledger_paths.append(fallback_path)

    for path in ledger_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        current_month = None
        found_nav_in_path = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            month_match = re.match(r"^##\s+(\d{4}-\d{2})\s*$", line)
            if month_match:
                current_month = month_match.group(1)
                continue

            if not current_month:
                continue

            nav_match = re.search(r"(?:월말 총자산 확정|장종료 기준 최종 총액):\s*([0-9,]+)", line)
            if nav_match:
                found_nav_in_path = True
                if current_month not in confirmed:
                    confirmed[current_month] = float(nav_match.group(1).replace(",", ""))

        if found_nav_in_path:
            return confirmed

    return confirmed


def get_rebalance_basis_nav(rebal_date, personal_nav_df):
    """이번 달 성과 계산에 쓸 전월말 기준 NAV와 출처를 반환한다."""
    rebal_ts = pd.Timestamp(rebal_date)
    month_key = rebal_ts.strftime("%Y-%m")
    ledger_navs = load_confirmed_month_end_navs()

    if month_key in ledger_navs:
        return ledger_navs[month_key], "월말 원장 확정 총자산", True

    if month_key == DEFAULT_INVESTMENT_START_DATE.strftime("%Y-%m"):
        return float(DEFAULT_INITIAL_CAPITAL), "초기 확정 총자산", True

    if rebal_date in personal_nav_df.index:
        return float(personal_nav_df.loc[rebal_date, "nav"]), "전략 시뮬레이션 NAV", False
    return float(personal_nav_df["nav"].asof(rebal_date)), "전략 시뮬레이션 NAV", False


def calculate_period_cash_flow(cash_flows, start_date, end_date):
    """start_date 이후부터 end_date까지 확정된 순외부현금흐름."""
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    total = 0.0
    for date_str, amount in (cash_flows or {}).items():
        cash_flow_ts = pd.Timestamp(date_str).normalize()
        if start_ts < cash_flow_ts <= end_ts:
            total += float(amount)
    return total


def build_personal_account_curve(strategy_nav, initial_capital, cash_flows):
    """
    strategy_nav: Faber A 전략 자체의 기준가/NAV Series
    initial_capital: 최초 투입 원금
    cash_flows: {"YYYY-MM-DD": amount} 형태. 입금은 +, 출금은 -
    """
    nav = pd.Series(strategy_nav).dropna().copy()
    nav = nav / nav.iloc[0]

    units = pd.Series(index=nav.index, dtype=float)
    principal = pd.Series(index=nav.index, dtype=float)
    account_value = pd.Series(index=nav.index, dtype=float)
    cash_flow_series = pd.Series(0.0, index=nav.index)

    for date_str, amount in cash_flows.items():
        date = pd.Timestamp(date_str)
        valid_dates = nav.index[nav.index >= date]
        if len(valid_dates) > 0:
            cash_flow_series.loc[valid_dates[0]] += float(amount)

    units.iloc[0] = initial_capital / nav.iloc[0]
    principal.iloc[0] = initial_capital
    account_value.iloc[0] = units.iloc[0] * nav.iloc[0]

    for i in range(1, len(nav)):
        units.iloc[i] = units.iloc[i - 1]
        principal.iloc[i] = principal.iloc[i - 1]

        cf = cash_flow_series.iloc[i]
        if cf != 0:
            units.iloc[i] += cf / nav.iloc[i]
            principal.iloc[i] += cf

        account_value.iloc[i] = units.iloc[i] * nav.iloc[i]

    return pd.DataFrame({
        "strategy_nav": nav,
        "cash_flow": cash_flow_series,
        "units": units,
        "principal": principal,
        "account_value": account_value,
        "profit": account_value - principal,
        "return_on_principal": account_value / principal - 1,
    })


def calculate_performance_metrics(daily_nav_df, initial_capital):
    if daily_nav_df is None or len(daily_nav_df) == 0: return None, None, None, None
    current_value = float(daily_nav_df["nav"].iloc[-1])
    total_return = (current_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0
    mdd = float(daily_nav_df["drawdown"].min())
    days = (daily_nav_df.index[-1] - daily_nav_df.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    cagr = (current_value / initial_capital) ** (1 / years) - 1 if years > 0 else total_return
    return current_value, total_return, mdd, cagr


def slice_nav_period(daily_nav_df, start_date, end_date):
    if daily_nav_df is None or daily_nav_df.empty:
        return None
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    nav = daily_nav_df[(daily_nav_df.index >= start_ts) & (daily_nav_df.index <= end_ts)].copy()
    if len(nav) < 2:
        return None
    nav["running_max"] = nav["nav"].expanding().max()
    nav["drawdown"] = (nav["nav"] - nav["running_max"]) / nav["running_max"]
    return nav


def calculate_period_nav_metrics(daily_nav_df, start_date, end_date):
    nav = slice_nav_period(daily_nav_df, start_date, end_date)
    if nav is None:
        return None
    start_value = float(nav["nav"].iloc[0])
    end_value = float(nav["nav"].iloc[-1])
    days = (nav.index[-1] - nav.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    total_return = end_value / start_value - 1 if start_value > 0 else 0.0
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 and start_value > 0 else total_return
    return {
        "start": nav.index[0],
        "end": nav.index[-1],
        "days": days,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "mdd": float(nav["drawdown"].min()),
    }


def calculate_sharpe_ratio(daily_nav_df, risk_free_annual=0.025):
    """연율화 Sharpe Ratio 계산 (일별 수익률 기준)."""
    if daily_nav_df is None or len(daily_nav_df) < 20: return None
    daily_ret = daily_nav_df["nav"].pct_change().dropna()
    if len(daily_ret) == 0 or daily_ret.std() == 0: return None
    rf_daily = (1 + risk_free_annual) ** (1/252) - 1
    excess = daily_ret - rf_daily
    return float((excess.mean() / excess.std()) * np.sqrt(252))

def calculate_sortino_ratio(daily_nav_df, risk_free_annual=0.025):
    """연율화 Sortino Ratio 계산. 하락 변동성만 분모에 사용."""
    if daily_nav_df is None or len(daily_nav_df) < 20: return None
    daily_ret = daily_nav_df["nav"].pct_change().dropna()
    if len(daily_ret) == 0: return None
    rf_daily = (1 + risk_free_annual) ** (1/252) - 1
    excess = daily_ret - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0: return None
    # Sortino 분모는 음수 초과수익의 RMS(반편차) 사용
    downside_std = float(np.sqrt(np.mean(np.square(downside.values))))
    if downside_std == 0: return None
    return float((excess.mean() / downside_std) * np.sqrt(252))

def calculate_annualized_volatility(daily_nav_df):
    """일별 수익률 표준편차를 연율화한 변동성."""
    if daily_nav_df is None or len(daily_nav_df) < 20:
        return None
    daily_ret = daily_nav_df["nav"].pct_change().dropna()
    if len(daily_ret) == 0:
        return None
    return float(daily_ret.std() * np.sqrt(252))

def calculate_strategy_downside_comparison(base_nav, target_nav):
    """Compare target strategy behavior when the base strategy has weak months."""
    if base_nav is None or target_nav is None or base_nav.empty or target_nav.empty:
        return None

    base_monthly = base_nav["nav"].groupby(base_nav.index.to_period("M")).last().pct_change().dropna()
    target_monthly = target_nav["nav"].groupby(target_nav.index.to_period("M")).last().pct_change().dropna()
    aligned = pd.concat([base_monthly.rename("base"), target_monthly.rename("target")], axis=1).dropna()
    if aligned.empty:
        return None

    aligned["excess"] = aligned["target"] - aligned["base"]
    down = aligned[aligned["base"] < 0]
    up = aligned[aligned["base"] > 0]
    stress = aligned[aligned["base"] <= -0.05]

    def _mean_or_none(series):
        return float(series.mean()) if len(series) > 0 else None

    def _worse_rate(df):
        return float((df["target"] < df["base"]).mean()) if len(df) > 0 else None

    beta = None
    if len(down) >= 2 and float(down["base"].var()) > 0:
        beta = float(down["target"].cov(down["base"]) / down["base"].var())

    capture = None
    if len(up) > 0 and abs(float(up["base"].mean())) > 1e-12:
        capture = float(up["target"].mean() / up["base"].mean())

    return {
        "months": int(len(aligned)),
        "up_months": int(len(up)),
        "down_months": int(len(down)),
        "stress_months": int(len(stress)),
        "avg_excess": _mean_or_none(aligned["excess"]),
        "up_base_avg": _mean_or_none(up["base"]),
        "up_target_avg": _mean_or_none(up["target"]),
        "up_excess_avg": _mean_or_none(up["excess"]),
        "down_base_avg": _mean_or_none(down["base"]),
        "down_target_avg": _mean_or_none(down["target"]),
        "down_excess_avg": _mean_or_none(down["excess"]),
        "stress_base_avg": _mean_or_none(stress["base"]),
        "stress_target_avg": _mean_or_none(stress["target"]),
        "stress_excess_avg": _mean_or_none(stress["excess"]),
        "down_beta": beta,
        "up_capture": capture,
        "down_worse_rate": _worse_rate(down),
        "stress_worse_rate": _worse_rate(stress),
    }

def find_mdd_period(daily_nav_df):
    if daily_nav_df is None or len(daily_nav_df) == 0: return None, None, None
    valley_date = daily_nav_df["drawdown"].idxmin()
    mdd_value = float(daily_nav_df["drawdown"].min())
    running_max_at_valley = float(daily_nav_df.loc[valley_date, "running_max"])
    eps = max(1e-8, abs(running_max_at_valley) * 1e-10)
    peak_dates = daily_nav_df[(daily_nav_df["nav"] - running_max_at_valley).abs() <= eps].index
    peak_date = peak_dates[peak_dates <= valley_date][-1] if len(peak_dates[peak_dates <= valley_date]) > 0 else valley_date
    return peak_date, valley_date, mdd_value

def calculate_monthly_drawdown_series(daily_nav_df):
    if daily_nav_df is None or daily_nav_df.empty: return None
    monthly_nav = daily_nav_df["nav"].groupby(daily_nav_df.index.to_period("M")).last()
    monthly_nav.index = monthly_nav.index.to_timestamp("M")
    running_max = monthly_nav.cummax()
    return (monthly_nav - running_max) / running_max

def calculate_monthly_mdd(daily_nav_df):
    dd = calculate_monthly_drawdown_series(daily_nav_df)
    return float(dd.min()) if dd is not None and not dd.empty else None


def calculate_ulcer_index(daily_nav_df):
    """Ulcer Index: drawdown(%)의 RMS. 높을수록 체감 고통이 큼."""
    if daily_nav_df is None or daily_nav_df.empty or "drawdown" not in daily_nav_df.columns:
        return None
    dd_pct = (daily_nav_df["drawdown"].dropna() * 100.0).values
    if len(dd_pct) == 0:
        return None
    return float(np.sqrt(np.mean(np.square(dd_pct))))


def calculate_martin_ratio(daily_nav_df, initial_capital):
    """Martin Ratio: CAGR(%) / Ulcer Index(%)"""
    if daily_nav_df is None or daily_nav_df.empty:
        return None
    _, _, _, cagr = calculate_performance_metrics(daily_nav_df, initial_capital)
    ui = calculate_ulcer_index(daily_nav_df)
    if cagr is None or ui is None or ui <= 0:
        return None
    return float((cagr * 100.0) / ui)


def calculate_monthly_cvar(daily_nav_df, alpha=0.05):
    """월수익률 기준 CVaR(기대손실). 보통 음수이며 더 낮을수록 꼬리위험 큼."""
    if daily_nav_df is None or daily_nav_df.empty:
        return None
    monthly_nav = daily_nav_df["nav"].groupby(daily_nav_df.index.to_period("M")).last()
    monthly_ret = monthly_nav.pct_change().dropna()
    if len(monthly_ret) < 12:
        return None
    var = monthly_ret.quantile(alpha)
    tail = monthly_ret[monthly_ret <= var]
    if tail.empty:
        return None
    return float(tail.mean())


def calculate_positive_month_ratio(daily_nav_df):
    """월수익률이 +인 달의 비율."""
    if daily_nav_df is None or daily_nav_df.empty:
        return None
    monthly_nav = daily_nav_df["nav"].groupby(daily_nav_df.index.to_period("M")).last()
    monthly_ret = monthly_nav.pct_change().dropna()
    if len(monthly_ret) == 0:
        return None
    return float((monthly_ret > 0).mean())


def _build_union_trading_dates(data_dict, start_date, end_date):
    all_dates = set()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    for df in (data_dict or {}).values():
        if df is not None and not df.empty:
            all_dates.update(df.index)
    return sorted(d for d in all_dates if start_ts <= d <= end_ts)


def _first_common_month_end_start(data_dict, end_date, price_col="Adj Close"):
    if not data_dict or any(df is None or df.empty for df in data_dict.values()):
        return None, None, []

    common_listing_date = max(df.index.min() for df in data_dict.values())
    trading_dates = _build_union_trading_dates(data_dict, common_listing_date, end_date)
    if not trading_dates:
        return None, common_listing_date, []

    month_end_dates = _collect_month_end_dates(trading_dates)
    for d in month_end_dates:
        has_all_prices = all(
            get_price_at_date(df, d, price_col=price_col) is not None
            for df in data_dict.values()
        )
        if has_all_prices:
            return d, common_listing_date, trading_dates
    return None, common_listing_date, trading_dates


def _fetch_requested_portfolio_asset_data(asset, start_date, end_date):
    main_df = fetch_etf_data(asset["ticker"], start_date, end_date, is_momentum=False)
    fallback_ticker = asset.get("fallback_ticker")
    if not fallback_ticker:
        return main_df

    fallback_df = fetch_etf_data(fallback_ticker, start_date, end_date, is_momentum=False)
    if main_df is None or main_df.empty:
        return fallback_df
    if fallback_df is None or fallback_df.empty:
        return main_df
    return _chain_link_series(fallback_df, main_df)


def simulate_fixed_weight_monthly_rebalanced_portfolio(
    portfolio_def, start_date, end_date, initial_capital, price_col="Adj Close", force_start_date=None
):
    """사용자 지정 ETF 바스켓을 공통 상장 이후 첫 월말부터 월말 리밸런싱."""
    raw_data = {}
    for asset in portfolio_def.get("assets", []):
        df = _fetch_requested_portfolio_asset_data(asset, start_date, end_date)
        if df is not None and not df.empty:
            raw_data[asset["name"]] = df[df.index <= pd.Timestamp(end_date)].copy()

    required_names = [asset["name"] for asset in portfolio_def.get("assets", [])]
    if any(name not in raw_data or raw_data[name].empty for name in required_names):
        return None, raw_data, {"status": "데이터 부족"}

    actual_start, common_listing_date, trading_dates = _first_common_month_end_start(
        raw_data, end_date, price_col=price_col
    )
    if actual_start is None:
        return None, raw_data, {"status": "공통 월말 시작일 없음", "common_listing_date": common_listing_date}

    if force_start_date is not None:
        forced_start = pd.Timestamp(force_start_date)
        eligible_dates = [
            d for d in trading_dates
            if d >= forced_start and all(
                get_price_at_date(df, d, price_col=price_col) is not None
                for df in raw_data.values()
            )
        ]
        if not eligible_dates:
            return None, raw_data, {
                "status": "고정 시작일 이후 공통 가격 없음",
                "common_listing_date": common_listing_date,
            }
        actual_start = eligible_dates[0]

    trading_dates = [d for d in trading_dates if d >= actual_start]
    if not trading_dates:
        return None, raw_data, {"status": "거래일 부족", "common_listing_date": common_listing_date}

    weights = {asset["name"]: float(asset["weight"]) for asset in portfolio_def.get("assets", [])}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return None, raw_data, {"status": "비중 오류", "common_listing_date": common_listing_date}
    weights = {name: weight / total_weight for name, weight in weights.items()}

    holdings = {name: 0.0 for name in weights}
    nav = float(initial_capital)
    last_valid_nav = nav
    rows = []

    def _rebalance(date, portfolio_value):
        for name, weight in weights.items():
            px = get_price_at_date(raw_data.get(name), date, price_col=price_col)
            holdings[name] = (portfolio_value * weight / px) if px and px > 0 else 0.0

    _rebalance(actual_start, nav)
    for i, date in enumerate(trading_dates):
        pv_raw = 0.0
        for name, units in holdings.items():
            px = get_price_at_date(raw_data.get(name), date, price_col=price_col)
            if px is None or px <= 0:
                pv_raw = None
                break
            pv_raw += units * px
        pv = _safe_nav_value(pv_raw, last_valid_nav)
        if pv is None:
            pv = last_valid_nav
        nav = pv
        last_valid_nav = nav
        rows.append({"date": date, "nav": nav})

        if _is_month_end_rebalance_day(trading_dates, i) and date != actual_start:
            _rebalance(date, nav)

    nav_df = pd.DataFrame(rows).set_index("date").sort_index()
    nav_df["running_max"] = nav_df["nav"].expanding().max()
    nav_df["drawdown"] = (nav_df["nav"] - nav_df["running_max"]) / nav_df["running_max"]

    meta = {
        "status": "OK",
        "common_listing_date": common_listing_date,
        "actual_start": actual_start,
        "actual_end": nav_df.index.max(),
        "trading_days": int(len(nav_df)),
        "month_count": int(len(nav_df["nav"].groupby(nav_df.index.to_period("M")).last()) - 1),
    }
    return nav_df, raw_data, meta


def calculate_extended_nav_metrics(nav_df):
    if nav_df is None or nav_df.empty:
        return None
    period_initial = float(nav_df["nav"].iloc[0])
    current_value, total_return, mdd, cagr = calculate_performance_metrics(nav_df, period_initial)
    monthly_mdd = calculate_monthly_mdd(nav_df)
    volatility = calculate_annualized_volatility(nav_df)
    sharpe = calculate_sharpe_ratio(nav_df)
    sortino = calculate_sortino_ratio(nav_df)
    cagr_mdd = (cagr / abs(mdd)) if (cagr is not None and cagr > 0 and mdd is not None and mdd < 0) else None
    return {
        "current_value": current_value,
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "monthly_mdd": monthly_mdd,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr_mdd": cagr_mdd,
        "ulcer": calculate_ulcer_index(nav_df),
        "martin": calculate_martin_ratio(nav_df, period_initial),
        "pos_month": calculate_positive_month_ratio(nav_df),
    }


def render_requested_static_portfolio_backtests(
    end_date, initial_capital, price_col="Adj Close", comparison_nav=None, comparison_label="해남 A",
    collapsed=False,
):
    section = (
        st.expander("📎 접어둔 연구/후보 포트폴리오 백테스트", expanded=False)
        if collapsed else st.container()
    )
    with section:
        if not collapsed:
            st.markdown("---")
        common_start = REQUESTED_PORTFOLIO_COMMON_START
        st.subheader("📌 요청 포트폴리오 동일기간 월말 리밸런싱")
        st.caption(
            f"모든 포트폴리오는 {common_start.strftime('%Y-%m-%d')} 월말 거래일 종가에 투자하고, "
            "이후 월말 거래일 종가 기준으로 목표 비중 리밸런싱합니다. "
            "TIME 미국배당다우존스액티브는 상장 전 구간을 TIGER 미국배당다우존스로 보강해 같은 시작일로 비교합니다. "
            "실전 본체가 아니라 비교용 연구/후보 자료입니다."
        )

        data_start = pd.Timestamp("2000-01-01")
        summary_rows = []
        detail_rows = []
        chart_navs = {}

        for portfolio in REQUESTED_STATIC_PORTFOLIOS:
            nav_df, raw_data, meta = simulate_fixed_weight_monthly_rebalanced_portfolio(
                portfolio, data_start, end_date, initial_capital,
                price_col=price_col, force_start_date=common_start
            )
            if nav_df is None or meta.get("status") != "OK":
                summary_rows.append({
                    "포트폴리오": portfolio["name"],
                    "구성": portfolio["description"],
                    "백테스트 기간": "-",
                    "CAGR": "-",
                    "누적수익률": "-",
                    "MDD(일별)": "-",
                    "MDD(월말)": "-",
                    "변동성": "-",
                    "Sharpe": "-",
                    "Sortino": "-",
                    "양(+)월": "-",
                    f"{comparison_label} CAGR": "-",
                    f"{comparison_label} MDD": "-",
                })
                continue

            metrics = calculate_extended_nav_metrics(nav_df)
            comp_metrics = (
                calculate_period_nav_metrics(comparison_nav, meta["actual_start"], meta["actual_end"])
                if comparison_nav is not None and not comparison_nav.empty else None
            )
            chart_navs[portfolio["name"]] = nav_df
            period = f"{meta['actual_start'].strftime('%Y-%m-%d')} ~ {meta['actual_end'].strftime('%Y-%m-%d')}"
            summary_rows.append({
                "포트폴리오": portfolio["name"],
                "구성": portfolio["description"],
                "백테스트 기간": period,
                "개월": meta["month_count"],
                "CAGR": f"{metrics['cagr']:.2%}" if metrics["cagr"] is not None else "-",
                "누적수익률": f"{metrics['total_return']:.2%}" if metrics["total_return"] is not None else "-",
                "MDD(일별)": f"{metrics['mdd']:.2%}" if metrics["mdd"] is not None else "-",
                "MDD(월말)": f"{metrics['monthly_mdd']:.2%}" if metrics["monthly_mdd"] is not None else "-",
                "변동성": f"{metrics['volatility']:.2%}" if metrics["volatility"] is not None else "-",
                "Sharpe": f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "-",
                "Sortino": f"{metrics['sortino']:.2f}" if metrics["sortino"] is not None else "-",
                "CAGR/MDD": f"{metrics['cagr_mdd']:.2f}" if metrics["cagr_mdd"] is not None else "-",
                "양(+)월": f"{metrics['pos_month']:.1%}" if metrics["pos_month"] is not None else "-",
                f"{comparison_label} 기간": (
                    f"{comp_metrics['start'].strftime('%Y-%m-%d')} ~ {comp_metrics['end'].strftime('%Y-%m-%d')}"
                    if comp_metrics is not None else "-"
                ),
                f"{comparison_label} CAGR": (
                    f"{comp_metrics['cagr']:.2%}" if comp_metrics is not None else "-"
                ),
                f"{comparison_label} 누적수익률": (
                    f"{comp_metrics['total_return']:.2%}" if comp_metrics is not None else "-"
                ),
                f"{comparison_label} MDD": (
                    f"{comp_metrics['mdd']:.2%}" if comp_metrics is not None else "-"
                ),
            })

            for asset in portfolio["assets"]:
                df = raw_data.get(asset["name"])
                ticker_label = asset["ticker"]
                if asset.get("fallback_ticker"):
                    ticker_label = f"{asset['ticker']} (상장 전 {asset['fallback_ticker']})"
                detail_rows.append({
                    "포트폴리오": portfolio["name"],
                    "ETF": asset["name"],
                    "티커": ticker_label,
                    "비중": f"{asset['weight']:.0%}",
                    "데이터 시작": df.index.min().strftime("%Y-%m-%d") if df is not None and not df.empty else "-",
                })

        if comparison_nav is not None and not comparison_nav.empty:
            strategy_rows = [
                ("요청 포트폴리오 6", f"{comparison_label} 전략", common_start),
            ]
            for row_name, description, start_ts in strategy_rows:
                strategy_nav = slice_nav_period(comparison_nav, start_ts, end_date)
                if strategy_nav is None or strategy_nav.empty:
                    continue
                metrics = calculate_extended_nav_metrics(strategy_nav)
                actual_start = strategy_nav.index.min()
                actual_end = strategy_nav.index.max()
                month_count = int(len(strategy_nav["nav"].groupby(strategy_nav.index.to_period("M")).last()) - 1)
                period = f"{actual_start.strftime('%Y-%m-%d')} ~ {actual_end.strftime('%Y-%m-%d')}"
                chart_navs[row_name] = strategy_nav
                summary_rows.append({
                    "포트폴리오": row_name,
                    "구성": description,
                    "백테스트 기간": period,
                    "개월": month_count,
                    "CAGR": f"{metrics['cagr']:.2%}" if metrics["cagr"] is not None else "-",
                    "누적수익률": f"{metrics['total_return']:.2%}" if metrics["total_return"] is not None else "-",
                    "MDD(일별)": f"{metrics['mdd']:.2%}" if metrics["mdd"] is not None else "-",
                    "MDD(월말)": f"{metrics['monthly_mdd']:.2%}" if metrics["monthly_mdd"] is not None else "-",
                    "변동성": f"{metrics['volatility']:.2%}" if metrics["volatility"] is not None else "-",
                    "Sharpe": f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "-",
                    "Sortino": f"{metrics['sortino']:.2f}" if metrics["sortino"] is not None else "-",
                    "CAGR/MDD": f"{metrics['cagr_mdd']:.2f}" if metrics["cagr_mdd"] is not None else "-",
                    "양(+)월": f"{metrics['pos_month']:.1%}" if metrics["pos_month"] is not None else "-",
                    f"{comparison_label} 기간": period,
                    f"{comparison_label} CAGR": f"{metrics['cagr']:.2%}" if metrics["cagr"] is not None else "-",
                    f"{comparison_label} 누적수익률": (
                        f"{metrics['total_return']:.2%}" if metrics["total_return"] is not None else "-"
                    ),
                    f"{comparison_label} MDD": f"{metrics['mdd']:.2%}" if metrics["mdd"] is not None else "-",
                })

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        if detail_rows:
            if collapsed:
                st.markdown("**구성 ETF 및 데이터 시작일**")
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
            else:
                with st.expander("구성 ETF 및 데이터 시작일", expanded=False):
                    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        if chart_navs:
            fig = make_subplots(rows=2, cols=1, subplot_titles=("누적수익률 (%)", "Drawdown (%)"),
                                vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)
            colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]
            for idx, (name, nav_df) in enumerate(chart_navs.items()):
                ret_pct = (nav_df["nav"] / nav_df["nav"].iloc[0] - 1.0) * 100.0
                dd_pct = nav_df["drawdown"] * 100.0
                fig.add_trace(go.Scatter(
                    x=nav_df.index, y=ret_pct, mode="lines", name=name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>수익률: %{y:.2f}%<extra></extra>",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=nav_df.index, y=dd_pct, mode="lines", name=f"{name} DD",
                    line=dict(color=colors[idx % len(colors)], width=1.4, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
                    showlegend=False,
                ), row=2, col=1)
            fig.update_layout(height=620, hovermode="x unified", legend=dict(orientation="h", y=1.08))
            fig.update_yaxes(title_text="수익률 (%)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)


def calculate_rolling_outperformance_rate(nav_a, nav_b, window_months=36):
    """A가 B를 이긴 롤링 구간 비율. (월말 수익률 기준)"""
    if nav_a is None or nav_b is None or nav_a.empty or nav_b.empty:
        return None, 0
    ma = nav_a["nav"].groupby(nav_a.index.to_period("M")).last()
    mb = nav_b["nav"].groupby(nav_b.index.to_period("M")).last()
    merged = pd.concat([ma.rename("A"), mb.rename("B")], axis=1).dropna()
    if len(merged) <= window_months:
        return None, 0
    ra = merged["A"].pct_change(window_months)
    rb = merged["B"].pct_change(window_months)
    diff = (ra - rb).dropna()
    if diff.empty:
        return None, 0
    return float((diff > 0).mean()), int(len(diff))


def estimate_turnover_from_weight_series(weight_dicts, asset_keys):
    """연속 목표비중 시계열로 월별 회전율(매수/매도 절반합) 추정."""
    if weight_dicts is None or len(weight_dicts) < 2:
        return None, None, 0
    prev = None
    turns = []
    for w in weight_dicts:
        vec = np.array([float(w.get(k, 0.0)) for k in asset_keys], dtype=float)
        s = float(vec.sum())
        if s > 0:
            vec = vec / s
        if prev is not None:
            turns.append(float(0.5 * np.abs(vec - prev).sum()))
        prev = vec
    if len(turns) == 0:
        return None, None, 0
    return float(np.mean(turns)), float(np.max(turns)), int(len(turns))

def find_monthly_mdd_period(daily_nav_df):
    if daily_nav_df is None or daily_nav_df.empty: return None, None, None
    dd_series = calculate_monthly_drawdown_series(daily_nav_df)
    if dd_series is None or dd_series.empty: return None, None, None
    valley_month = dd_series.idxmin()
    mdd_val = float(dd_series.min())
    if abs(mdd_val) < 1e-10: return None, None, None
    monthly_nav = daily_nav_df["nav"].groupby(daily_nav_df.index.to_period("M")).last()
    monthly_nav.index = monthly_nav.index.to_timestamp("M")
    running_max = monthly_nav.cummax()
    peak_val = running_max.loc[valley_month]
    peak_candidates = monthly_nav[monthly_nav >= peak_val * 0.9999]
    peak_candidates = peak_candidates[peak_candidates.index <= valley_month]
    peak_month = peak_candidates.index[-1] if len(peak_candidates) > 0 else valley_month
    def find_last_td(ts):
        mp = ts.to_period("M")
        md = daily_nav_df[daily_nav_df.index.to_period("M") == mp]
        return md.index[-1] if len(md) > 0 else ts
    return find_last_td(peak_month), find_last_td(valley_month), mdd_val

def calculate_yearly_daily_stats(daily_nav_df, year):
    if daily_nav_df is None or len(daily_nav_df) == 0: return None
    nav = daily_nav_df["nav"].sort_index()
    daily_ret = nav.pct_change()
    df = pd.DataFrame({"nav": nav, "ret": daily_ret})
    df_year = df[df.index.year == year].dropna(subset=["ret"])
    if len(df_year) == 0: return None
    up = df_year[df_year["ret"] > 0]["ret"]
    down = df_year[df_year["ret"] < 0]["ret"]
    prev_year = nav[nav.index < pd.Timestamp(year, 1, 1)]
    year_start_nav = prev_year.iloc[-1] if len(prev_year) > 0 else nav.iloc[0]
    yearly_return = (df_year["nav"].iloc[-1] - year_start_nav) / year_start_nav if year_start_nav > 0 else 0.0
    return {
        "year": year, "yearly_return": float(yearly_return),
        "total_days": int(len(df_year)), "up_days": int(len(up)),
        "down_days": int(len(down)), "flat_days": int(len(df_year) - len(up) - len(down)),
        "up_mean": float(up.mean()) if len(up) > 0 else None,
        "down_mean": float(down.mean()) if len(down) > 0 else None,
        "max_daily_loss": float(down.min()) if len(down) > 0 else None,
    }


def build_comparison_table(strategies_dict, initial_capital):
    """전략 비교 테이블. Sortino 기준 순위."""
    rows = []
    for name, nav_df in strategies_dict.items():
        if nav_df is None or nav_df.empty: continue
        val, ret, mdd, cagr = calculate_performance_metrics(nav_df, initial_capital)
        sharpe = calculate_sharpe_ratio(nav_df)
        sortino = calculate_sortino_ratio(nav_df)
        rows.append({
            "전략": name,
            "CAGR": f"{cagr*100:.2f}%" if cagr is not None else "-",
            "MDD (일별)": f"{mdd*100:.2f}%" if mdd is not None else "-",
            "Sharpe": f"{sharpe:.2f}" if sharpe is not None else "-",
            "Sortino": f"{sortino:.2f}" if sortino is not None else "-",
            "CAGR/MDD": f"{(cagr/abs(mdd)):.2f}" if (cagr is not None and cagr > 0 and mdd is not None and mdd < -0.001) else "-",
            "_sortino_raw": sortino if sortino is not None else -999,
        })
    if not rows: return None
    df = pd.DataFrame(rows).sort_values("_sortino_raw", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "순위"
    df = df.drop(columns=["_sortino_raw"])
    return df


def _normalize_nav_for_compare(nav_df):
    """공정 비교를 위해 nav 시계열을 정리하고 drawdown을 재계산한다."""
    if nav_df is None or nav_df.empty or "nav" not in nav_df.columns:
        return None
    out = nav_df[["nav"]].copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        return None
    out["running_max"] = out["nav"].expanding().max()
    out["drawdown"] = (out["nav"] - out["running_max"]) / out["running_max"]
    return out


def align_strategies_to_common_dates(strategies_dict, min_obs_days=252):
    """전략 NAV를 공통 거래일 교집합으로 맞춰 공정 비교용 데이터와 상태를 반환."""
    normalized = {}
    raw_meta = {}
    for name, nav_df in strategies_dict.items():
        ndf = _normalize_nav_for_compare(nav_df)
        if ndf is None:
            raw_meta[name] = None
            continue
        normalized[name] = ndf
        raw_meta[name] = {
            "start": ndf.index.min(),
            "end": ndf.index.max(),
            "obs": len(ndf),
        }

    common_idx = None
    if normalized:
        for ndf in normalized.values():
            common_idx = ndf.index if common_idx is None else common_idx.intersection(ndf.index)
        common_idx = common_idx.sort_values()

    aligned = {}
    common_start, common_end, common_obs = None, None, 0
    if common_idx is not None and len(common_idx) > 0:
        common_start = common_idx[0]
        common_end = common_idx[-1]
        common_obs = len(common_idx)
        for name, ndf in normalized.items():
            cut = ndf.loc[common_idx].copy()
            cut["running_max"] = cut["nav"].expanding().max()
            cut["drawdown"] = (cut["nav"] - cut["running_max"]) / cut["running_max"]
            aligned[name] = cut

    status_rows = []
    for name in strategies_dict.keys():
        meta = raw_meta.get(name)
        if meta is None:
            status_rows.append({
                "전략": name,
                "원본 기간": "-",
                "원본 거래일": 0,
                "공통 거래일": 0,
                "상태": "제외 (데이터 없음)",
            })
            continue

        aligned_obs = len(aligned.get(name, []))
        if common_obs == 0:
            state = "제외 (공통 거래일 없음)"
        elif aligned_obs < min_obs_days:
            state = f"주의 (공통 거래일 부족: {aligned_obs}일)"
        else:
            state = "비교 가능"

        status_rows.append({
            "전략": name,
            "원본 기간": f"{meta['start'].strftime('%Y-%m-%d')} ~ {meta['end'].strftime('%Y-%m-%d')}",
            "원본 거래일": int(meta["obs"]),
            "공통 거래일": int(aligned_obs),
            "상태": state,
        })

    meta = {
        "common_start": common_start,
        "common_end": common_end,
        "common_obs": int(common_obs),
        "min_obs_days": int(min_obs_days),
    }
    return aligned, meta, pd.DataFrame(status_rows)


# ==============================
# 8. 종목/ETF 분석
# ==============================
def _asset_analysis_pct(current, previous):
    if previous is None or previous == 0 or current is None:
        return None
    return float(current / previous - 1.0)


def _asset_analysis_fmt_pct(value, digits=1):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def _asset_analysis_fmt_price(value):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.2f}"


def _asset_analysis_value_at_or_before(series, target_date):
    if series is None or series.empty:
        return None
    target_ts = pd.Timestamp(target_date)
    eligible = series[series.index <= target_ts].dropna()
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


@st.cache_data(ttl=3600)
def load_asset_analysis_price_data(ticker, start_date, end_date):
    return fetch_etf_data(ticker.strip().upper(), start_date, end_date, is_momentum=False)


def build_asset_analysis_metrics(price_df, price_col="Adj Close"):
    if price_df is None or price_df.empty:
        return None
    col = price_col if price_col in price_df.columns else "Close"
    prices = pd.to_numeric(price_df[col], errors="coerce").dropna()
    prices = prices[prices > 0]
    if len(prices) < 20:
        return None

    latest_price = float(prices.iloc[-1])
    latest_date = pd.Timestamp(prices.index[-1])
    daily_returns = prices.pct_change().dropna()
    running_high = prices.cummax()
    drawdown = prices / running_high - 1.0

    windows = {
        "1개월": 21,
        "3개월": 63,
        "6개월": 126,
        "1년": 252,
        "3년": 756,
    }
    returns = {}
    for label, days in windows.items():
        if len(prices) > days:
            returns[label] = _asset_analysis_pct(latest_price, float(prices.iloc[-days - 1]))
        else:
            returns[label] = None

    moving_averages = {}
    ma_state = {}
    for window in (20, 60, 120, 200):
        if len(prices) >= window:
            ma = float(prices.rolling(window).mean().iloc[-1])
            moving_averages[f"MA{window}"] = ma
            ma_state[f"MA{window} 대비"] = _asset_analysis_pct(latest_price, ma)
        else:
            moving_averages[f"MA{window}"] = None
            ma_state[f"MA{window} 대비"] = None

    one_year_prices = prices.tail(252)
    high_52w = float(one_year_prices.max()) if len(one_year_prices) > 0 else None
    low_52w = float(one_year_prices.min()) if len(one_year_prices) > 0 else None
    distance_from_52w_high = _asset_analysis_pct(latest_price, high_52w)
    distance_from_52w_low = _asset_analysis_pct(latest_price, low_52w)

    volatility_1y = None
    if len(daily_returns.tail(252)) >= 20:
        volatility_1y = float(daily_returns.tail(252).std() * np.sqrt(252))

    metrics = {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "latest_price": latest_price,
        "observations": int(len(prices)),
        "data_start": pd.Timestamp(prices.index[0]).strftime("%Y-%m-%d"),
        "data_end": latest_date.strftime("%Y-%m-%d"),
        "returns": returns,
        "moving_averages": moving_averages,
        "ma_state": ma_state,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "distance_from_52w_high": distance_from_52w_high,
        "distance_from_52w_low": distance_from_52w_low,
        "mdd_full_period": float(drawdown.min()),
        "current_drawdown": float(drawdown.iloc[-1]),
        "volatility_1y": volatility_1y,
    }
    return metrics


def create_asset_price_chart(price_df, ticker, price_col="Adj Close"):
    col = price_col if price_col in price_df.columns else "Close"
    prices = pd.to_numeric(price_df[col], errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices,
        mode="lines",
        name=ticker,
        line=dict(color="#2563eb", width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>가격: %{y:,.2f}<extra></extra>",
    ))
    for window, color in ((20, "#22c55e"), (60, "#f59e0b"), (200, "#ef4444")):
        if len(prices) >= window:
            ma = prices.rolling(window).mean()
            fig.add_trace(go.Scatter(
                x=ma.index,
                y=ma,
                mode="lines",
                name=f"MA{window}",
                line=dict(color=color, width=1.0),
                opacity=0.8,
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>MA{window}: %{{y:,.2f}}<extra></extra>",
            ))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        title=f"{ticker} 가격 추이",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


@st.cache_data(ttl=86400)
def load_asset_analysis_trailing_valuation(ticker):
    """Best-effort current PER/EPS lookup. Historical PER still needs a fundamentals API."""
    if not _YF_AVAILABLE:
        return None, "yfinance가 없어 자동 PER 조회를 사용할 수 없습니다."

    raw = str(ticker or "").strip().upper()
    if not raw:
        return None, "티커가 비어 있습니다."

    symbols = [raw]
    if raw.isdigit() and len(raw) == 6:
        symbols = [f"{raw}.KS", raw]

    last_error = None
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).get_info()
            trailing_pe = _to_float(info.get("trailingPE"))
            forward_pe = _to_float(info.get("forwardPE"))
            trailing_eps = _to_float(info.get("trailingEps"))
            if (trailing_pe is not None and trailing_pe > 0) or (trailing_eps is not None and trailing_eps > 0):
                return {
                    "symbol": symbol,
                    "trailing_pe": trailing_pe,
                    "forward_pe": forward_pe,
                    "trailing_eps": trailing_eps,
                    "source": "yfinance",
                }, None
        except Exception as exc:
            last_error = str(exc)
            continue

    return None, f"자동 PER 조회 실패: {last_error or 'PER/EPS 값 없음'}"


def classify_per_band(current_per, quantiles):
    current = _to_float(current_per)
    if current is None or not quantiles:
        return "판정 불가"
    if current <= quantiles["p10"]:
        return "극단 저평가 영역"
    if current <= quantiles["p25"]:
        return "저평가 영역"
    if current >= quantiles["p90"]:
        return "극단 고평가 영역"
    if current >= quantiles["p75"]:
        return "고평가 영역"
    return "중립 영역"


def build_per_band_analysis(price_df, price_col="Adj Close", current_per=None, current_eps=None):
    if price_df is None or price_df.empty:
        return None
    col = price_col if price_col in price_df.columns else "Close"
    prices = pd.to_numeric(price_df[col], errors="coerce").dropna()
    prices = prices[prices > 0]
    if len(prices) < 60:
        return None

    latest_price = float(prices.iloc[-1])
    eps = _to_float(current_eps)
    source = "사용자 입력 EPS"
    if eps is None or eps <= 0:
        per = _to_float(current_per)
        if per is None or per <= 0:
            return None
        eps = latest_price / per
        source = "현재 PER 역산 EPS"
    if eps <= 0:
        return None

    per_series = (prices / eps).replace([np.inf, -np.inf], np.nan).dropna()
    per_series = per_series[per_series > 0]
    if len(per_series) < 60:
        return None

    quantiles = {
        "p10": float(per_series.quantile(0.10)),
        "p25": float(per_series.quantile(0.25)),
        "p50": float(per_series.quantile(0.50)),
        "p75": float(per_series.quantile(0.75)),
        "p90": float(per_series.quantile(0.90)),
    }
    latest_per = float(per_series.iloc[-1])
    percentile = float((per_series <= latest_per).mean())
    return {
        "latest_price": latest_price,
        "eps": float(eps),
        "latest_per": latest_per,
        "percentile": percentile,
        "quantiles": quantiles,
        "zone": classify_per_band(latest_per, quantiles),
        "data_start": pd.Timestamp(per_series.index[0]).strftime("%Y-%m-%d"),
        "data_end": pd.Timestamp(per_series.index[-1]).strftime("%Y-%m-%d"),
        "source": source,
        "per_series": per_series,
    }


def create_per_band_chart(per_band):
    per_series = per_band["per_series"]
    q = per_band["quantiles"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=per_series.index,
        y=per_series,
        mode="lines",
        name="PER",
        line=dict(color="#2563eb", width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>PER: %{y:.2f}<extra></extra>",
    ))
    band_specs = [
        ("P10", q["p10"], "#16a34a", "dot"),
        ("P25", q["p25"], "#22c55e", "dash"),
        ("P50", q["p50"], "#64748b", "dash"),
        ("P75", q["p75"], "#f59e0b", "dash"),
        ("P90", q["p90"], "#ef4444", "dot"),
    ]
    for label, value, color, dash in band_specs:
        fig.add_hline(
            y=value,
            line_color=color,
            line_dash=dash,
            annotation_text=f"{label} {value:.1f}",
            annotation_position="right",
        )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        title="PER 밴드",
        hovermode="x unified",
    )
    return fig


def build_rule_based_asset_analysis(ticker, asset_kind, metrics):
    ret_1y = metrics["returns"].get("1년")
    ret_3m = metrics["returns"].get("3개월")
    ma200_gap = metrics["ma_state"].get("MA200 대비")
    dd_52w = metrics.get("distance_from_52w_high")
    vol = metrics.get("volatility_1y")

    trend_bits = []
    if ma200_gap is not None:
        trend_bits.append("장기 이동평균 위" if ma200_gap >= 0 else "장기 이동평균 아래")
    if ret_3m is not None:
        trend_bits.append("최근 3개월 플러스" if ret_3m >= 0 else "최근 3개월 마이너스")
    trend = ", ".join(trend_bits) if trend_bits else "추세 판단을 위한 데이터가 부족합니다"

    caution = []
    if dd_52w is not None and dd_52w < -0.2:
        caution.append("52주 고점 대비 낙폭이 커서 하락 추세 지속 여부를 확인해야 합니다")
    if vol is not None and vol > 0.35:
        caution.append("연율화 변동성이 높은 편이라 포지션 크기 관리가 중요합니다")
    if ma200_gap is not None and ma200_gap < 0:
        caution.append("장기 추세가 약한 구간일 수 있습니다")
    if not caution:
        caution.append("가격 데이터만으로는 큰 위험 신호가 단정되지 않습니다")

    return (
        f"### {ticker} 1차 분석\n\n"
        f"- **정체성**: 현재 v0.1은 {asset_kind}로 사용자가 지정한 자산의 가격 상태를 먼저 점검합니다.\n"
        f"- **현재 상태**: {trend}. 1년 수익률은 {_asset_analysis_fmt_pct(ret_1y)}, "
        f"52주 고점 대비 위치는 {_asset_analysis_fmt_pct(dd_52w)}입니다.\n"
        f"- **위험 체크**: {'; '.join(caution)}.\n"
        f"- **다음 확인 질문**: 이 자산을 왜 보유/관심 대상으로 보는지, 비교 대상은 무엇인지, "
        f"가격이 아니라 펀더멘털/구성종목에서 깨질 수 있는 가정은 무엇인지 확인하세요.\n\n"
        "※ GLM API 키가 없어서 규칙 기반 초안으로 표시했습니다."
    )


def call_asset_analysis_llm(ticker, asset_kind, metrics, user_question):
    api_key = _get_config_secret("FABER_LLM_API_KEY", "TOGETHER_API_KEY")
    if not api_key:
        return None, "GLM API 키가 없어 규칙 기반 분석을 사용했습니다."

    base_url = _get_config_secret("FABER_LLM_BASE_URL", "TOGETHER_BASE_URL") or "https://api.together.xyz/v1"
    model = _get_config_secret("FABER_LLM_MODEL") or "zai-org/GLM-5.2"
    url = base_url.rstrip("/") + "/chat/completions"
    system_prompt = (
        "너는 초보 투자자를 위한 종목/ETF 분석 보조자다. "
        "숫자는 제공된 JSON 안의 값만 사용하고, 없는 재무/뉴스/구성종목 정보는 모른다고 말한다. "
        "매수/매도 단정이나 수익 보장을 하지 말고, 위험과 추가 확인 질문을 먼저 정리한다."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"자산: {ticker}\n"
                    f"분류: {asset_kind}\n"
                    f"사용자 질문: {user_question or '이 자산의 현재 상태를 초보자에게 설명해줘.'}\n"
                    f"계산 데이터 JSON:\n{json.dumps(metrics, ensure_ascii=False)}\n\n"
                    "아래 형식으로 한국어 답변:\n"
                    "1. 한 줄 요약\n2. 가격/추세\n3. 위험 체크\n4. 지금 모르는 것\n5. 추가로 확인할 질문 3개"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"], None
    except Exception as exc:
        return None, f"GLM 호출 실패: {exc}"


def mode_asset_analysis(current_dt, current_date, price_col):
    st.title("🔎 종목/ETF 분석")
    st.caption("티커 하나를 넣고 가격 상태, 위험, AI 해석 초안을 확인합니다. v0.1은 가격 기반 분석부터 시작합니다.")
    st.markdown("---")

    input_cols = st.columns([1.2, 1, 1])
    with input_cols[0]:
        ticker = st.text_input("티커", value="005930", help="예: 005930, 000660, QQQ, SPY, GLD")
    with input_cols[1]:
        asset_kind = st.selectbox("분류", ["주식", "ETF", "기타"], index=0)
    with input_cols[2]:
        lookback_years = st.selectbox("가격 기간", [1, 3, 5, 10], index=1)

    with st.expander("PER 밴드 입력", expanded=False):
        st.caption(
            "정확한 과거 PER 밴드는 과거 EPS/컨센서스 데이터 API가 필요합니다. "
            "여기서는 자동 조회 또는 직접 입력한 현재 PER/EPS로 가격을 환산해 1차 밴드를 보여줍니다."
        )
        use_auto_per = st.checkbox("현재 PER/EPS 자동 조회 시도", value=True)
        val_cols = st.columns(2)
        with val_cols[0]:
            manual_current_per = st.number_input("현재 PER 직접 입력", min_value=0.0, value=0.0, step=0.5)
        with val_cols[1]:
            manual_eps_ttm = st.number_input("TTM EPS 직접 입력", min_value=0.0, value=0.0, step=10.0)

    user_question = st.text_input("AI에게 물어볼 질문", value="이 자산 지금 어떤 상태야?")
    run_analysis = st.button("분석 실행", type="primary", use_container_width=True)

    if not run_analysis:
        st.info("티커를 입력하고 분석 실행을 눌러주세요. 처음 버전은 가격과 위험 체크부터 봅니다.")
        return

    ticker = ticker.strip().upper()
    if not ticker:
        st.warning("티커를 입력해 주세요.")
        return

    start_date = current_date - relativedelta(years=int(lookback_years), months=3)
    with st.spinner(f"{ticker} 가격 데이터를 불러오는 중..."):
        price_df = load_asset_analysis_price_data(ticker, start_date, current_date)

    if price_df is None or price_df.empty:
        st.error("가격 데이터를 불러오지 못했습니다. 국내 6자리 코드나 미국 티커 형식으로 다시 시도해 주세요.")
        return

    metrics = build_asset_analysis_metrics(price_df, price_col=price_col)
    if metrics is None:
        st.error("분석에 필요한 가격 데이터가 부족합니다.")
        return

    st.plotly_chart(create_asset_price_chart(price_df, ticker, price_col=price_col), use_container_width=True)

    metric_cols = st.columns(5)
    metric_cols[0].metric("최근 가격", _asset_analysis_fmt_price(metrics["latest_price"]))
    metric_cols[1].metric("1년 수익률", _asset_analysis_fmt_pct(metrics["returns"].get("1년")))
    metric_cols[2].metric("52주 고점 대비", _asset_analysis_fmt_pct(metrics["distance_from_52w_high"]))
    metric_cols[3].metric("현재 낙폭", _asset_analysis_fmt_pct(metrics["current_drawdown"]))
    metric_cols[4].metric("1년 변동성", _asset_analysis_fmt_pct(metrics["volatility_1y"]))

    table_rows = []
    for label, value in metrics["returns"].items():
        table_rows.append({"구분": f"수익률 {label}", "값": _asset_analysis_fmt_pct(value)})
    for label, value in metrics["ma_state"].items():
        table_rows.append({"구분": label, "값": _asset_analysis_fmt_pct(value)})
    table_rows.extend([
        {"구분": "52주 고가", "값": _asset_analysis_fmt_price(metrics["high_52w"])},
        {"구분": "52주 저가", "값": _asset_analysis_fmt_price(metrics["low_52w"])},
        {"구분": "전체 기간 MDD", "값": _asset_analysis_fmt_pct(metrics["mdd_full_period"])},
        {"구분": "데이터 기간", "값": f"{metrics['data_start']} ~ {metrics['data_end']}"},
    ])
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("#### PER 밴드")
    valuation = None
    valuation_warning = None
    if asset_kind == "주식" and use_auto_per:
        valuation, valuation_warning = load_asset_analysis_trailing_valuation(ticker)
        if valuation_warning:
            st.caption(valuation_warning)
    elif asset_kind != "주식":
        st.caption("ETF/기타 자산은 PER 밴드보다 구성종목 밸류에이션 데이터가 더 적합합니다.")

    auto_per = _to_float((valuation or {}).get("trailing_pe"))
    auto_eps = _to_float((valuation or {}).get("trailing_eps"))
    per_input = manual_current_per if manual_current_per > 0 else auto_per
    eps_input = manual_eps_ttm if manual_eps_ttm > 0 else auto_eps
    per_band = build_per_band_analysis(
        price_df,
        price_col=price_col,
        current_per=per_input,
        current_eps=eps_input,
    )
    if per_band:
        per_cols = st.columns(4)
        per_cols[0].metric("현재 PER", _fmt_plain_number(per_band["latest_per"], digits=2))
        per_cols[1].metric("밴드 판정", per_band["zone"])
        per_cols[2].metric("PER 백분위", _fmt_pct(per_band["percentile"], digits=0))
        per_cols[3].metric("사용 EPS", _fmt_plain_number(per_band["eps"], digits=2))
        st.plotly_chart(create_per_band_chart(per_band), use_container_width=True)
        band_rows = [
            {"밴드": "P10", "PER": _fmt_plain_number(per_band["quantiles"]["p10"], digits=2), "의미": "극단 저평가 경계"},
            {"밴드": "P25", "PER": _fmt_plain_number(per_band["quantiles"]["p25"], digits=2), "의미": "저평가 경계"},
            {"밴드": "P50", "PER": _fmt_plain_number(per_band["quantiles"]["p50"], digits=2), "의미": "중앙값"},
            {"밴드": "P75", "PER": _fmt_plain_number(per_band["quantiles"]["p75"], digits=2), "의미": "고평가 경계"},
            {"밴드": "P90", "PER": _fmt_plain_number(per_band["quantiles"]["p90"], digits=2), "의미": "극단 고평가 경계"},
        ]
        st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"{per_band['source']} 기준의 간이 PER 밴드입니다. 실제 역사적 PER 밴드는 연도별/분기별 EPS 변화와 "
            "컨센서스 조정이 반영되어야 하므로 FnGuide, Quantiwise, Bloomberg, Koyfin, FMP 같은 펀더멘털 데이터 API가 필요합니다."
        )
    else:
        st.info("PER 밴드를 보려면 현재 PER 또는 TTM EPS가 필요합니다. 자동 조회가 실패하면 위 입력란에 직접 넣어주세요.")

    st.markdown("#### AI 해석")
    llm_text, llm_warning = call_asset_analysis_llm(ticker, asset_kind, metrics, user_question)
    if llm_warning:
        st.caption(llm_warning)
    if not llm_text:
        llm_text = build_rule_based_asset_analysis(ticker, asset_kind, metrics)
    st.markdown(llm_text)

    with st.expander("분석에 사용한 요약 JSON", expanded=False):
        st.json(metrics)


# ==============================
# 9. 차트
# ==============================
def create_nav_and_drawdown_chart(daily_nav_df, initial_capital, peak_date, valley_date, title,
                                   monthly_peak_date=None, monthly_valley_date=None, monthly_mdd_val=None,
                                   extra_navs=None, primary_label="Faber A"):
    """수익률 + Drawdown 차트. extra_navs = {"이름": (nav_df, color, dash), ...}"""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("일별 수익률 곡선", "Drawdown (낙폭)"),
        vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)
    
    returns = ((daily_nav_df["nav"] / initial_capital) - 1) * 100.0
    fig.add_trace(go.Scatter(x=daily_nav_df.index, y=returns, mode="lines", name=primary_label,
        line=dict(color="#1f77b4", width=2),
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{primary_label}: %{{y:.2f}}%<extra></extra>"), row=1, col=1)

    # 추가 전략/벤치마크 라인
    if extra_navs:
        for ename, (edf, ecolor, edash) in extra_navs.items():
            if edf is None or edf.empty: continue
            er = ((edf["nav"] / initial_capital) - 1) * 100.0
            fig.add_trace(go.Scatter(x=edf.index, y=er, mode="lines", name=ename,
                line=dict(color=ecolor, width=1.2, dash=edash),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{ename}: %{{y:.2f}}%<extra></extra>"), row=1, col=1)
            edd = edf["drawdown"] * 100.0
            fig.add_trace(go.Scatter(x=edf.index, y=edd, mode="lines", name=f"DD {ename}",
                line=dict(color=ecolor, width=1, dash=edash), opacity=0.4, showlegend=False,
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{ename} DD: %{{y:.2f}}%<extra></extra>"), row=2, col=1)

    # MDD 마커들
    if peak_date is not None and valley_date is not None:
        pr = ((daily_nav_df.loc[peak_date, "nav"] / initial_capital) - 1) * 100.0
        fig.add_trace(go.Scatter(x=[peak_date], y=[pr], mode="markers+text", name="고점 (일별)",
            marker=dict(size=10, color="red"), text=["고점"], textposition="top center",
            hovertemplate="고점<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"), row=1, col=1)
        vr = ((daily_nav_df.loc[valley_date, "nav"] / initial_capital) - 1) * 100.0
        fig.add_trace(go.Scatter(x=[valley_date], y=[vr], mode="markers+text", name="MDD 저점 (일별)",
            marker=dict(size=10, color="cyan"), text=["MDD 저점"], textposition="bottom center",
            hovertemplate="MDD 저점<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"), row=1, col=1)

    if monthly_peak_date is not None and monthly_valley_date is not None and monthly_mdd_val is not None:
        if monthly_peak_date in daily_nav_df.index:
            mr = ((daily_nav_df.loc[monthly_peak_date, "nav"] / initial_capital) - 1) * 100.0
            fig.add_trace(go.Scatter(x=[monthly_peak_date], y=[mr], mode="markers+text", name="고점 (월별)",
                marker=dict(size=12, color="darkred", symbol="diamond"), text=["월별 고점"], textposition="top right",
                hovertemplate="월별 고점<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"), row=1, col=1)
        if monthly_valley_date in daily_nav_df.index:
            mvr = ((daily_nav_df.loc[monthly_valley_date, "nav"] / initial_capital) - 1) * 100.0
            fig.add_trace(go.Scatter(x=[monthly_valley_date], y=[mvr], mode="markers+text", name="MDD 저점 (월별)",
                marker=dict(size=12, color="darkblue", symbol="diamond"),
                text=[f"월별 MDD {monthly_mdd_val*100:.1f}%"], textposition="bottom right",
                hovertemplate="월별 MDD<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"), row=1, col=1)

    # Drawdown
    dd_pct = daily_nav_df["drawdown"] * 100.0
    fig.add_trace(go.Scatter(x=daily_nav_df.index, y=dd_pct, mode="lines", name=f"DD {primary_label}",
        fill="tozeroy", hovertemplate="%{x|%Y-%m-%d}<br>낙폭: %{y:.2f}%<extra></extra>"), row=2, col=1)
    if valley_date is not None:
        fig.add_trace(go.Scatter(x=[valley_date], y=[float(daily_nav_df.loc[valley_date, "drawdown"]) * 100.0],
            mode="markers", name="최대 낙폭 (일별)", marker=dict(size=10, color="orange"),
            hovertemplate="최대 낙폭<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"), row=2, col=1)

    mdd_series = calculate_monthly_drawdown_series(daily_nav_df)
    if mdd_series is not None and not mdd_series.empty:
        fig.add_trace(go.Scatter(x=mdd_series.index, y=mdd_series * 100.0, mode="lines+markers",
            name="DD 월별", line=dict(color="darkviolet", width=2, dash="dot"),
            marker=dict(size=5, color="darkviolet"),
            hovertemplate="월별 낙폭<br>%{x|%Y-%m}<br>%{y:.2f}%<extra></extra>"), row=2, col=1)
        if monthly_valley_date is not None and monthly_mdd_val is not None:
            fig.add_trace(go.Scatter(x=[monthly_valley_date], y=[monthly_mdd_val * 100.0],
                mode="markers", name="최대 낙폭 (월별)", marker=dict(size=12, color="darkviolet", symbol="diamond"),
                hovertemplate="최대 낙폭 (월별)<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"), row=2, col=1)

    fig.update_xaxes(title_text="날짜", row=2, col=1)
    fig.update_yaxes(title_text="수익률 (%)", row=1, col=1)
    fig.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
    fig.update_layout(title=title, height=750, showlegend=True, hovermode="x unified")
    return fig


def create_attribution_chart(monthly_attribution, year_start=None, year_end=None):
    if not monthly_attribution: return None
    df_attr = pd.DataFrame(monthly_attribution)
    df_attr["date"] = pd.to_datetime(df_attr["date"])
    if year_start: df_attr = df_attr[df_attr["date"].dt.year >= year_start]
    if year_end: df_attr = df_attr[df_attr["date"].dt.year <= year_end]
    if df_attr.empty: return None
    start_date, end_date = df_attr["date"].min(), df_attr["date"].max()
    all_months = pd.date_range(start=start_date.replace(day=1), end=end_date.replace(day=1), freq="MS")
    df_complete = pd.DataFrame({"date": all_months})
    df_complete["month"] = df_complete["date"].dt.strftime("%Y-%m")
    df_attr["month"] = df_attr["date"].dt.strftime("%Y-%m")
    df_merged = df_complete.merge(df_attr, on="month", how="left", suffixes=("", "_orig"))
    asset_columns = [c for c in df_attr.columns if c not in ["date", "date_orig", "total_return", "month"]]
    for col in asset_columns:
        if col not in df_merged.columns: df_merged[col] = 0.0
        else: df_merged[col] = df_merged[col].fillna(0.0)
    df_merged["total_return"] = df_merged.get("total_return", pd.Series(0.0)).fillna(0.0)
    for col in asset_columns: df_merged[col] = (df_merged[col] * 100.0).clip(-50, 50)
    df_merged["total_pct"] = (df_merged["total_return"] * 100.0).clip(-50, 50)
    n_months = len(df_merged)
    dtick = "M1" if n_months <= 24 else ("M3" if n_months <= 60 else "M6")
    fig = go.Figure()
    for asset in asset_columns:
        fig.add_trace(go.Bar(x=df_merged["month"], y=df_merged[asset], name=asset,
            hovertemplate="<b>%{x}</b><br>" + asset + ": <b>%{y:.2f}pp</b><extra></extra>"))
    fig.update_layout(title="월별 자산 수익 기여도 분석", xaxis_title="월", yaxis_title="기여도 (pp)",
        barmode="stack", height=650, hovermode="x unified",
        xaxis=dict(tickangle=-45, dtick=dtick, tickformat="%Y-%m"), margin=dict(l=60, r=40, t=80, b=120))
    fig.add_hline(y=0, line_width=2)
    if n_months <= 36:
        for _, row in df_merged.iterrows():
            if abs(row["total_pct"]) > 0.1:
                fig.add_annotation(x=row["month"], y=row["total_pct"],
                    text=f"{row['total_pct']:.1f}%", showarrow=False,
                    yshift=12 if row["total_pct"] >= 0 else -18, font=dict(size=9))
    return fig, df_merged


def create_weights_chart(monthly_weights_history, title="월별 자산 배분 비중 변화 (Faber A)"):
    if not monthly_weights_history: return None
    df_w = pd.DataFrame(monthly_weights_history)
    df_w["date"] = pd.to_datetime(df_w["date"])
    df_w = df_w.sort_values("date")
    asset_cols = [c for c in df_w.columns if c != "date"]
    colors = {'코스피200': '#1f77b4', '미국나스닥100': '#ff7f0e', '한국채30년': '#2ca02c',
              '미국채30년': '#d62728', '금현물': '#FFD700', CASH_NAME: '#9467bd',
              HAENAM_SAMSUNG_NAME: '#4c78a8', HAENAM_HYNIX_NAME: '#72b7b2', HAENAM_KR_TIME_NAME: '#54a24b', HAENAM_KR_KOACT_NAME: '#b279a2',
              HAENAM_TIME_NAME: '#f58518', HAENAM_KOACT_NAME: '#e45756'}
    fig = go.Figure()
    for asset in asset_cols:
        fig.add_trace(go.Scatter(x=df_w["date"], y=df_w[asset] * 100, mode="lines", name=asset,
            stackgroup="one", line=dict(width=0.5), fillcolor=colors.get(asset),
            hovertemplate=f"<b>{asset}</b><br>" + "%{x|%Y-%m}<br>비중: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="날짜",
        yaxis_title="비중 (%)", yaxis=dict(range=[0, 100]), height=500, hovermode="x unified",
        xaxis=dict(tickformat="%Y-%m"), margin=dict(l=60, r=40, t=80, b=80))
    return fig


# ==============================
# 9. 3계좌 최적화
# ==============================
def _rebalance_account_priority(asset):
    if asset == '금현물':
        return ["금계좌"]
    if asset in (HAENAM_TIME_NAME, HAENAM_KOACT_NAME, NASDAQ100_ASSET_NAME, '미국나스닥100'):
        return ["ISA_B", "ISA_A", "일반계좌"]
    if asset in ('미국채30년', '한국채30년'):
        return ["ISA_A", "ISA_B", "일반계좌"]
    return ["일반계좌", "ISA_A", "ISA_B"]


def _calculate_tax_optimized_allocation(df_res, account_balances, account_columns):
    account_columns = list(account_columns)
    total = float(sum(max(0.0, float(account_balances.get(col, 0.0))) for col in account_columns))
    rem = {
        account: max(0.0, float(account_balances.get(account, 0.0)))
        for account in account_columns
    }
    weight_map = {}
    if df_res is not None and len(df_res) > 0 and "자산명" in df_res.columns and "추천비중" in df_res.columns:
        weight_map = df_res.set_index("자산명")["추천비중"].to_dict()
    final = {}

    def _asset_bucket(asset):
        if asset not in final:
            final[asset] = {col: 0.0 for col in account_columns}
        return final[asset]

    def _eligible_accounts(asset):
        priority = [account for account in _rebalance_account_priority(asset) if account in account_columns]
        if asset == '금현물':
            if "금계좌" in priority:
                return ["금계좌"]
            if "일반계좌" in account_columns:
                return ["일반계좌"]
            return priority
        for account in account_columns:
            if account not in priority and account != "금계좌":
                priority.append(account)
        return priority

    def _allocate(asset, target):
        left = float(target)
        bucket = _asset_bucket(asset)
        for account in _eligible_accounts(asset):
            if left <= 0:
                break
            if account == "금계좌":
                source_account = "일반계좌"
            else:
                source_account = account
            if source_account not in rem:
                continue
            fill = min(left, rem[source_account])
            rem[source_account] -= fill
            bucket[account] += fill
            left -= fill
        if left > 0.5:
            # 계좌 제약을 넘는 극단 상황에서는 일반계좌에 표시해 총 목표 금액이 사라지지 않게 한다.
            overflow_account = "일반계좌" if "일반계좌" in account_columns else account_columns[0]
            bucket[overflow_account] += left
            rem[overflow_account] -= left

    ordered_assets = []
    for a in (GENERAL_PRIORITY + ISA_PRIORITY):
        if a not in ordered_assets: ordered_assets.append(a)
    for a in weight_map.keys():
        if a != CASH_NAME and a not in ordered_assets: ordered_assets.append(a)
    for a in ASSETS.keys():
        if a not in ordered_assets: ordered_assets.append(a)
    targets = []
    for asset in ordered_assets:
        w = float(weight_map.get(asset, 0.0))
        if w <= 0: continue
        if asset == CASH_NAME:
            continue
        targets.append({"asset": asset, "target": float(total * w)})
    for t in targets:
        _allocate(t["asset"], t["target"])
    final[CASH_NAME] = {
        account: (0.0 if account == "금계좌" else max(0, rem.get(account, 0.0)))
        for account in account_columns
    }
    res_list = []
    output_assets = []
    for k in ordered_assets + [CASH_NAME]:
        if k not in output_assets:
            output_assets.append(k)
    for k in output_assets:
        if k in final:
            v = final[k]
            res_list.append({"자산명": k, "추천비중": float(weight_map.get(k, 0.0)),
                "총목표금액": float(sum(v[col] for col in account_columns)),
                **{col: float(v[col]) for col in account_columns}})
    df_out = pd.DataFrame(res_list)
    for c in ["총목표금액"] + account_columns: df_out[c] = df_out[c].round(0)
    sum_row = {
        "자산명": "합계",
        "추천비중": 1.0,
        "총목표금액": float(df_out["총목표금액"].sum()),
        **{col: float(df_out[col].sum()) for col in account_columns},
    }
    return pd.concat([df_out, pd.DataFrame([sum_row])], ignore_index=True)


def optimize_allocation(df_res, b_gen_kospi, b_gen_gold, b_isa_a, b_isa_b):
    taxable_total = float(b_gen_kospi + b_gen_gold)
    balances = {
        "금계좌": 0.0,
        "일반계좌": taxable_total,
        "ISA_A": float(b_isa_a),
        "ISA_B": float(b_isa_b),
    }
    return _calculate_tax_optimized_allocation(df_res, balances, ACCOUNT_COLUMNS)


def normalize_buy_hold_weights(raw_weights):
    weights = {asset: max(0.0, float(raw_weights.get(asset, 0.0))) for asset in BUY_HOLD_ASSET_BUCKETS}
    total = sum(weights.values())
    if total <= 0:
        return BUY_HOLD_BASELINE_WEIGHTS.copy(), 0.0
    return {asset: weight / total for asset, weight in weights.items()}, total


def expand_buy_hold_weights_to_execution_assets(weights):
    return {
        HAENAM_KR_TIME_NAME: float(weights.get("코스피", 0.0)) / 2.0,
        HAENAM_KR_KOACT_NAME: float(weights.get("코스피", 0.0)) / 2.0,
        HAENAM_TIME_NAME: float(weights.get("나스닥100", 0.0)) / 2.0,
        HAENAM_KOACT_NAME: float(weights.get("나스닥100", 0.0)) / 2.0,
        "미국채30년": float(weights.get("미국채30년", 0.0)),
        CASH_NAME: float(weights.get("현금", 0.0)),
    }


def calculate_buy_hold_allocation(account_balances, target_weights, current_amounts=None, baseline_weights=None):
    """Calculate the isolated buy-and-hold allocation.

    If current_amounts is omitted, the legacy 20/20/20/40 baseline is used
    as an assumed current state.
    """
    baseline_weights = baseline_weights or BUY_HOLD_BASELINE_WEIGHTS
    balances = {
        account: max(0.0, float(account_balances.get(account, 0.0)))
        for account in BUY_HOLD_ACCOUNT_COLUMNS
    }
    total_assets = sum(balances.values())
    normalized_targets, raw_weight_total = normalize_buy_hold_weights(target_weights)
    baseline_normalized, _ = normalize_buy_hold_weights(baseline_weights)
    execution_targets = expand_buy_hold_weights_to_execution_assets(normalized_targets)
    execution_baseline = expand_buy_hold_weights_to_execution_assets(baseline_normalized)
    if current_amounts is None:
        execution_current_amounts = {
            asset: total_assets * weight
            for asset, weight in execution_baseline.items()
        }
    else:
        execution_current_amounts = {
            asset: max(0.0, float(current_amounts.get(asset, 0.0)))
            for asset in execution_targets
        }
    current_total_assets = sum(execution_current_amounts.values())
    target_df = pd.DataFrame(
        [{"자산명": asset, "추천비중": weight} for asset, weight in execution_targets.items()]
    )
    allocation = _calculate_tax_optimized_allocation(target_df, balances, BUY_HOLD_ACCOUNT_COLUMNS)

    rows = []
    allocation_by_asset = allocation[allocation["자산명"] != "합계"].set_index("자산명")
    for asset in execution_targets.keys():
        target_weight = float(execution_targets.get(asset, 0.0))
        current_amount = float(execution_current_amounts.get(asset, 0.0))
        current_weight = current_amount / current_total_assets if current_total_assets > 0 else 0.0
        target_amount = total_assets * target_weight
        row = {
            "자산": asset,
            "투자상품": BUY_HOLD_INSTRUMENT_MAP.get(asset, ""),
            "현재비중": current_weight,
            "목표비중": target_weight,
            "현재금액": current_amount,
            "목표금액": target_amount,
            "추가매수_매도": target_amount - current_amount,
        }
        for account in BUY_HOLD_ACCOUNT_COLUMNS:
            row[account] = float(allocation_by_asset.loc[asset, account]) if asset in allocation_by_asset.index else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    sum_row = {
        "자산": "합계",
        "투자상품": "",
        "현재비중": 1.0 if current_total_assets > 0 else 0.0,
        "목표비중": sum(normalized_targets.values()),
        "현재금액": current_total_assets,
        "목표금액": total_assets,
        "추가매수_매도": float(df["추가매수_매도"].sum()) if not df.empty else 0.0,
    }
    for account in BUY_HOLD_ACCOUNT_COLUMNS:
        sum_row[account] = float(df[account].sum()) if not df.empty else 0.0
    df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)
    money_cols = ["현재금액", "목표금액", "추가매수_매도"] + BUY_HOLD_ACCOUNT_COLUMNS
    for col in money_cols:
        df[col] = df[col].round(0)
    return {
        "total_assets": total_assets,
        "current_total_assets": current_total_assets,
        "raw_weight_total": raw_weight_total,
        "weights": normalized_targets,
        "table": df,
    }


def _set_default_balance_query_params():
    _set_query_params(
        gen_k=DEFAULT_GEN_KOSPI_BAL,
        gen_g=DEFAULT_GEN_GOLD_BAL,
        isaa=DEFAULT_ISA_A_BAL,
        isab=DEFAULT_ISA_B_BAL,
        bal_v=DEFAULT_BALANCE_VERSION,
    )


def _ensure_account_balance_state(account_keys=None, sync_query_params=True):
    selected_keys = list(account_keys) if account_keys is not None else [key for key, _ in BALANCE_DEFAULTS]
    default_by_key = dict(BALANCE_DEFAULTS)
    selected_defaults = [(key, default_by_key[key]) for key in selected_keys]
    qp = _get_query_params()
    qp_balance_version = str(_qp_first(qp.get("bal_v")) or "")
    if st.session_state.get("_balance_defaults_version") != DEFAULT_BALANCE_VERSION and qp_balance_version != DEFAULT_BALANCE_VERSION:
        for key, default in selected_defaults:
            st.session_state[key] = default
        st.session_state["_balance_defaults_version"] = DEFAULT_BALANCE_VERSION
        if sync_query_params:
            _set_default_balance_query_params()
    else:
        for key, default in selected_defaults:
            if key not in st.session_state:
                qp_val = _get_qp_int(qp, BALANCE_QUERY_KEYS[key])
                st.session_state[key] = qp_val if qp_val is not None else default
        st.session_state["_balance_defaults_version"] = DEFAULT_BALANCE_VERSION

    if sum(float(st.session_state.get(key, 0) or 0) for key, _ in selected_defaults) <= 0:
        for key, default in selected_defaults:
            st.session_state[key] = default
        st.session_state["_balance_defaults_version"] = DEFAULT_BALANCE_VERSION
        if sync_query_params:
            _set_default_balance_query_params()


def _get_account_balance_value(session_key, default):
    if session_key in st.session_state:
        return float(st.session_state.get(session_key) or 0)
    qp_val = _get_qp_int(_get_query_params(), BALANCE_QUERY_KEYS[session_key])
    return float(qp_val if qp_val is not None else default)


# ==============================
# 10. UI 모드들
# ==============================
def mode_strategy_backtest(current_dt, current_date, price_col, bt_start_date):
    requested_backtest_end = normalize_to_date(current_date)
    current_date = requested_backtest_end
    st.title("📈 전략 백테스트 & 시장 분석")
    st.markdown(f"**기준시각:** {current_dt.strftime('%Y년 %m월 %d일 %H:%M:%S')}")
    st.caption("※ 월말 종가 기준(같은 날 체결) 12개월 모멘텀 기반 비중 전략입니다.")
    st.markdown("---")
    
    with st.expander("📌 장기 백테스트: 하이브리드 데이터 안내", expanded=False):
        st.markdown(
            "ETF 상장 전 기간은 프록시, **상장 후는 실제 ETF 데이터**를 사용합니다.\n\n"
            "| 자산 | 프록시 (상장 전) | 상장 후 |\n"
            "|---|---|---|\n"
            "| 코스피200 | KODEX 200 (069500) | 실제 ETF (294400) |\n"
            "| 미국나스닥100 | QQQ × USD/KRW | TIGER 미국나스닥100 (133690) |\n"
            "| 한국채30년 | 국고채30년 실금리(ECOS, 2012~)+이자 합성 · 이전은 10년물 기반 | 실제 ETF (439870) |\n"
            "| 미국채30년 | GS30 실금리+이자 합성 → TLT × USD/KRW | 실제 ETF (476760) |\n"
            "| 금현물 | GLD × USD/KRW | ACE KRX금현물 (411060) |\n"
            "| 현금 | ECOS CD91 - 0.15%p 합성 (실패 시 연 2.5%) | 실제 MMF ETF (455890) |\n\n"
            "💡 **체인링크**: 프록시 → ETF 전환 시점에서 가격을 연결하여 수익률 연속성을 유지합니다.\n\n"
            "💡 채권 합성은 듀레이션·컨벡시티에 **이자(carry)**까지 더한 총수익 기준이며, 한국채30년은 "
            "실제 30년 국고채 일별 금리(ECOS, 2012-09~)를 써 실 ETF와 일상관 0.97·월상관 0.996로 추적합니다.\n\n"
            "⚠️ 금현물 Faber 신호는 별도 모멘텀 시리즈(0064K0 우선, 실패 시 GLD×환율 fallback)를 사용합니다.\n\n"
            "⚠️ 최근 기간의 모멘텀/기여도는 실제 ETF 데이터 기반으로 정확합니다."
        )

    st.caption(f"Backtest data cut-off: {requested_backtest_end.strftime('%Y-%m-%d')} (fixed)")
    data_start = bt_start_date - relativedelta(months=18)
    with st.spinner("시장 데이터 로딩 중... (하이브리드: 프록시+실제ETF, 최초 로딩 시 시간 소요)"):
        all_data = load_market_data(data_start, requested_backtest_end, hybrid=True)
        all_data = clamp_market_data_to_date(all_data, requested_backtest_end)

    # 한국채30년 데이터 소스 상태 (ECOS 키 인식 여부 — 모바일/Cloud에서 바로 확인용)
    _kr30_yield = fetch_kr_30y_bond_yield_series(data_start, requested_backtest_end)
    if _kr30_yield is not None and len(_kr30_yield) > 0:
        st.success(
            f"✅ 한국채30년: 실제 국고채30년 금리(ECOS, {_kr30_yield.index.min().date()}~) 기반 합성 사용 중 "
            "— 실 ETF와 일상관 0.97·월상관 0.996"
        )
    else:
        st.warning(
            "⚠️ 한국채30년: ECOS 키 미인식 → 10년물 기반 합성으로 폴백 중입니다. "
            "정확도를 높이려면 Streamlit Secrets에 `ECOS_API_KEY`를 추가하세요."
        )

    # 보조 벤치마크 ETF 로딩
    benchmark_raw = fetch_benchmark_etf(BENCHMARK_ETF['ticker'], bt_start_date, requested_backtest_end)
    if benchmark_raw is not None and not benchmark_raw.empty:
        benchmark_raw = benchmark_raw[benchmark_raw.index <= pd.Timestamp(requested_backtest_end)]
    fingerprint_df, fingerprint = build_market_data_fingerprint(all_data, price_col=price_col)

    with st.expander("📊 데이터 가용 기간 확인 (하이브리드)"):
        DEEP_PROXY_NOTES = {
            '코스피200':    'KOSPI지수(딥) → KODEX200 → 실제ETF: 2000-01-01 ~ 현재',
            '미국나스닥100': 'QQQ × USD/KRW → 실제ETF: 2000-01-01 ~ 현재',
            '한국채30년':   'ECOS 국고채30년 실금리(2012~)+이자 합성, 이전은 10년물×beta → 실제ETF(439870): 2000-12 ~ 현재',
            '미국채30년':   'FRED GS30 실금리+이자 합성 → TLT×환율 → 실제ETF(476760): 2000-01-01 ~ 현재',
            '금현물':       'FRED금현물(딥) → GLD×환율 → 실제ETF: 2000-01-01 ~ 현재 (신호: 0064K0 우선)',
        }
        for name in list(ASSETS.keys()) + [CASH_NAME]:
            df = all_data.get(name)
            status = f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}" if df is not None and len(df) > 0 else "❌ 없음"
            if name in DEEP_PROXY_NOTES:
                note = f" ({DEEP_PROXY_NOTES[name]})"
            elif name == CASH_NAME:
                note = f" ({PROXY_CASH['note']} → 실제MMF)"
            else:
                note = ""
            st.text(f"  {name}{note}: {status}")
        # 보조 벤치마크 상태
        if benchmark_raw is not None and len(benchmark_raw) > 0:
            st.text(f"  {BENCHMARK_ETF['name']}: {benchmark_raw.index.min().strftime('%Y-%m-%d')} ~ {benchmark_raw.index.max().strftime('%Y-%m-%d')}")
        else:
            st.text(f"  {BENCHMARK_ETF['name']} ({BENCHMARK_ETF['ticker']}): ❌ 데이터 없음 (펀드코드가 FDR 미지원일 수 있음)")

    IC = 10_000_000

    st.markdown("---")
    st.subheader(f"📊 원조 Faber A 중심 전략 백테스트 (요청 시작: {bt_start_date.strftime('%Y-%m')})")

    # Faber A 원형을 본선으로 두고, 해남P/액티브 변형은 비교군으로 유지한다.
    if fingerprint_df is not None:
        st.caption(f"Data fingerprint: `{fingerprint}`")
        render_backtest_reproducibility_status(
            bt_start_date, requested_backtest_end, price_col, fingerprint, fingerprint_df
        )
        with st.expander("Backtest input fingerprint", expanded=False):
            st.dataframe(fingerprint_df, use_container_width=True, hide_index=True)

    nav_df = simulate_faber_strategy(bt_start_date, current_date, IC, all_data,
        mode='A', buffer_df=None, price_col=price_col)
    if nav_df is None:
        st.error("백테스트 불가(데이터 부족). 시작일을 더 최근으로 조정해보세요.")
        return
    actual_start = nav_df.index.min()
    if actual_start > bt_start_date + timedelta(days=7):
        st.error(
            f"요청 시작일({bt_start_date.strftime('%Y-%m-%d')})과 실제 계산 시작일({actual_start.strftime('%Y-%m-%d')})이 다릅니다. "
            "이 상태에서는 CAGR/MDD/연도별 성과를 신뢰하면 안 됩니다."
        )
        st.stop()
    st.caption(
        f"✅ 실제 계산 시작일: {actual_start.strftime('%Y-%m-%d')} "
        f"(요청 시작일: {bt_start_date.strftime('%Y-%m-%d')})"
    )
    
    # 기존 연속 모멘텀 (차트 비교 참고용)
    st.caption(
        f"Actual end: {nav_df.index.max().strftime('%Y-%m-%d')} | "
        f"Requested end: {requested_backtest_end.strftime('%Y-%m-%d')} | "
        f"Data fingerprint: {fingerprint}"
    )

    old_nav, _, _, _ = simulate_daily_nav_with_attribution(
        bt_start_date, current_date, IC, all_data, price_col=price_col)
    haenam_p_strategy_data = build_haenam_p_strategy_data(
        all_data, data_start, current_date, price_col=price_col
    )
    faber_nasdaq_active_data = haenam_p_strategy_data
    faber_nasdaq_active_nav = (
        simulate_faber_strategy(
            bt_start_date, current_date, IC, faber_nasdaq_active_data,
            mode='A', buffer_df=None, price_col=price_col
        )
        if faber_nasdaq_active_data is not None else None
    )
    faber_active_nasdaq_kr_semi_data = build_faber_active_nasdaq_kr_semi_data(
        all_data, data_start, current_date, price_col=price_col
    )
    haenam_price_data = get_haenam_live_price_data(all_data, data_start, current_date)
    faber_active_nasdaq_kr_semi_nav = (
        simulate_faber_strategy(
            bt_start_date, current_date, IC, faber_active_nasdaq_kr_semi_data,
            mode='A', buffer_df=None, price_col=price_col
        )
        if faber_active_nasdaq_kr_semi_data is not None else None
    )
    haenam_s_strategy_data = build_haenam_s_strategy_data(
        all_data, data_start, current_date, price_col=price_col
    )
    faber_active_nasdaq_kr_samsung_data = haenam_s_strategy_data
    faber_active_nasdaq_kr_samsung_nav = (
        simulate_faber_strategy(
            bt_start_date, current_date, IC, faber_active_nasdaq_kr_samsung_data,
            mode='A', buffer_df=None, price_col=price_col
        )
        if faber_active_nasdaq_kr_samsung_data is not None else None
    )
    faber_active_nasdaq_kr_hynix_data = build_faber_active_nasdaq_kr_single_data(
        all_data, data_start, current_date, {"hynix": 1.0}, price_col=price_col
    )
    faber_active_nasdaq_kr_hynix_nav = (
        simulate_faber_strategy(
            bt_start_date, current_date, IC, faber_active_nasdaq_kr_hynix_data,
            mode='A', buffer_df=None, price_col=price_col
        )
        if faber_active_nasdaq_kr_hynix_data is not None else None
    )
    faber_active_nasdaq_kr_active_data = build_faber_active_nasdaq_kr_active_data(
        all_data, data_start, current_date, price_col=price_col
    )
    faber_active_nasdaq_kr_active_nav = (
        simulate_faber_strategy(
            bt_start_date, current_date, IC, faber_active_nasdaq_kr_active_data,
            mode='A', buffer_df=None, price_col=price_col
        )
        if faber_active_nasdaq_kr_active_data is not None else None
    )
    haenam_v_strategy_data = build_haenam_v_strategy_data(
        all_data, data_start, current_date, price_col=price_col
    )
    haenam_v_faber_nav = (
        simulate_faber_strategy(
            bt_start_date, current_date, IC, haenam_v_strategy_data,
            mode='A', buffer_df=None, price_col=price_col
        )
        if haenam_v_strategy_data is not None else None
    )
    haenam_v_mom_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, haenam_v_strategy_data,
            price_col=price_col
        )[0]
        if haenam_v_strategy_data is not None else None
    )
    haenam_v_passive_strategy_data = build_haenam_v_passive_strategy_data(
        all_data, data_start, current_date, price_col=price_col
    )
    haenam_v_passive_faber_nav = (
        simulate_faber_strategy(
            bt_start_date, current_date, IC, haenam_v_passive_strategy_data,
            mode='A', buffer_df=None, price_col=price_col
        )
        if haenam_v_passive_strategy_data is not None else None
    )
    haenam_v_passive_mom_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, haenam_v_passive_strategy_data,
            price_col=price_col
        )[0]
        if haenam_v_passive_strategy_data is not None else None
    )
    old_haenam_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, faber_active_nasdaq_kr_semi_data,
            price_col=price_col
        )[0]
        if faber_active_nasdaq_kr_semi_data is not None else None
    )
    # 연속 모멘텀(이전 전략) 신호로 같은 한국 슬롯 변형 + 나스닥 액티브 집행을 돌린 비교군.
    # 한국=삼성/하이닉스 50:50은 위 old_haenam_nav와 동일하므로 재사용한다.
    mom_kr_active_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, faber_active_nasdaq_kr_active_data,
            price_col=price_col
        )[0]
        if faber_active_nasdaq_kr_active_data is not None else None
    )
    mom_kr_samsung_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, haenam_s_strategy_data,
            price_col=price_col
        )[0]
        if haenam_s_strategy_data is not None else None
    )
    mom_passive_nasdaq_kr_samsung_data = build_faber_kr_stock_overlay_data(
        all_data, data_start, current_date, {"samsung": 1.0}, price_col=price_col
    )
    mom_passive_nasdaq_kr_samsung_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, mom_passive_nasdaq_kr_samsung_data,
            price_col=price_col
        )[0]
        if mom_passive_nasdaq_kr_samsung_data is not None else None
    )
    mom_kr_samsung_self_signal_data = build_faber_active_nasdaq_kr_single_self_signal_data(
        all_data, data_start, current_date, {"samsung": 1.0}, price_col=price_col
    )
    mom_kr_samsung_self_signal_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, mom_kr_samsung_self_signal_data,
            price_col=price_col
        )[0]
        if mom_kr_samsung_self_signal_data is not None else None
    )
    mom_kr_hynix_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, faber_active_nasdaq_kr_hynix_data,
            price_col=price_col
        )[0]
        if faber_active_nasdaq_kr_hynix_data is not None else None
    )
    mom_kr_semi_nav = old_haenam_nav
    mom_kr_passive_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, faber_nasdaq_active_data,
            price_col=price_col
        )[0]
        if faber_nasdaq_active_data is not None else None
    )
    haenam_p_local_signal_data = build_haenam_p_local_currency_signal_data(
        all_data, data_start, current_date, price_col=price_col
    )
    haenam_p_local_signal_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, haenam_p_local_signal_data,
            price_col=price_col
        )[0]
        if haenam_p_local_signal_data is not None else None
    )
    vix_data = fetch_vix_data(data_start, current_date)
    haenam_p_vix70_nav, haenam_p_vix70_overlay = (
        simulate_haenam_p_vix_overlay_strategy(
            bt_start_date, current_date, IC, haenam_p_strategy_data, vix_data,
            max_equity=0.70, price_col=price_col
        )
        if haenam_p_strategy_data is not None and vix_data is not None else (None, None)
    )
    haenam_p_vix100_nav, haenam_p_vix100_overlay = (
        simulate_haenam_p_vix_overlay_strategy(
            bt_start_date, current_date, IC, haenam_p_strategy_data, vix_data,
            max_equity=1.00, price_col=price_col
        )
        if haenam_p_strategy_data is not None and vix_data is not None else (None, None)
    )
    primary_nav_df = nav_df
    primary_strategy_data = all_data
    primary_price_data = all_data
    primary_label = "Faber A (원조: 코스피/나스닥 패시브 -5%룰)"
    primary_is_haenam = False
    faber_base_label = "Faber A (원형 신호·ETF 집행)"
    old_haenam_label = "이전 전략(연속모멘텀·해남 A 집행)"
    primary_asset_keys = []
    for asset in [
        KR_STOCK_MIX_ASSET,
        HAENAM_SAMSUNG_NAME,
        HAENAM_KR_TIME_NAME,
        HAENAM_KR_KOACT_NAME,
        NASDAQ100_ASSET_NAME,
        HAENAM_TIME_NAME,
        HAENAM_KOACT_NAME,
        '한국채30년',
        '미국채30년',
        '금현물',
    ]:
        if asset and asset not in primary_asset_keys:
            primary_asset_keys.append(asset)
    if not primary_is_haenam:
        primary_asset_keys = list(ASSETS.keys())
    primary_asset_display_names = {
        KR_STOCK_MIX_ASSET: "코스피200(집행 전/대체)",
        NASDAQ100_ASSET_NAME: "나스닥100(액티브 상장 전)",
        HAENAM_SAMSUNG_NAME: HAENAM_SAMSUNG_NAME,
        HAENAM_HYNIX_NAME: HAENAM_HYNIX_NAME,
        HAENAM_KR_TIME_NAME: HAENAM_KR_TIME_NAME,
        HAENAM_KR_KOACT_NAME: HAENAM_KR_KOACT_NAME,
        HAENAM_TIME_NAME: HAENAM_TIME_NAME,
        HAENAM_KOACT_NAME: HAENAM_KOACT_NAME,
        "한국채30년": "한국채30년",
        "미국채30년": "미국채30년",
        "금현물": "금현물",
        CASH_NAME: CASH_NAME,
    }

    def _display_asset_name(asset):
        return primary_asset_display_names.get(asset, asset)

    def _major_asset_display_name(asset):
        if primary_is_haenam and asset == KR_STOCK_MIX_ASSET:
            return "코스피200TR"
        return "나스닥100" if asset == NASDAQ100_ASSET_NAME else asset

    def _active_contribution_cols(df, cols, threshold=0.005):
        active = []
        for col in cols:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if float(s.abs().sum()) > threshold:
                active.append(col)
        return active

    def _rename_asset_columns_for_display(df):
        return df.rename(columns={c: _display_asset_name(c) for c in df.columns})

    def _aggregate_major_contribution_df(df):
        out = pd.DataFrame(index=df.index)
        out["date"] = df["date"]
        major_groups = {
            KR_STOCK_MIX_ASSET: [KR_STOCK_MIX_ASSET, HAENAM_SAMSUNG_NAME, HAENAM_HYNIX_NAME, HAENAM_KR_TIME_NAME, HAENAM_KR_KOACT_NAME],
            NASDAQ100_ASSET_NAME: [NASDAQ100_ASSET_NAME, HAENAM_TIME_NAME, HAENAM_KOACT_NAME],
            "한국채30년": ["한국채30년"],
            "미국채30년": ["미국채30년"],
            "금현물": ["금현물"],
            CASH_NAME: [CASH_NAME],
        }
        for display_name, source_cols in major_groups.items():
            vals = pd.Series(0.0, index=df.index)
            for col in source_cols:
                if col in df.columns:
                    vals = vals + pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            out[display_name] = vals
        if "합계" in df.columns:
            out["합계"] = df["합계"]
        return out

    st.caption(
        f"메인 전략: {primary_label} | 계산 기간: "
        f"{primary_nav_df.index.min().strftime('%Y-%m-%d')} ~ {primary_nav_df.index.max().strftime('%Y-%m-%d')}"
    )
    faber_ex3_assets = ['코스피200', '미국나스닥100', '금현물']
    faber_ex4_assets = ['코스피200', '미국나스닥100', CHINA_CSI300_CNY_ASSET, '금현물']
    faber_ex5_assets = ['코스피200', '미국나스닥100', CHINA_CSI300_CNY_ASSET, INDIA_NIFTY_INR_ASSET, '금현물']
    faber_ex3_data = build_faber_ex_bonds_strategy_data(
        all_data, bt_start_date, current_date, include_china=False, include_india=False
    )
    faber_ex4_data = build_faber_ex_bonds_strategy_data(
        all_data, bt_start_date, current_date, include_china=True, include_india=False
    )
    faber_ex5_data = build_faber_ex_bonds_strategy_data(
        all_data, bt_start_date, current_date, include_china=True, include_india=True
    )

    # 동일비중 B&H
    static_nav = simulate_static_benchmark(bt_start_date, current_date, IC, all_data, price_col=price_col)

    # ALLW (US ETF) × USD/KRW 벤치마크 로딩
    allw_nav = None
    try:
        allw_raw = fdr.DataReader('ALLW', bt_start_date, current_date)
        allw_fx  = get_usdkrw_series(bt_start_date, current_date)
        if allw_raw is not None and not allw_raw.empty and allw_fx is not None and not allw_fx.empty:
            allw_raw = allw_raw[~allw_raw.index.duplicated(keep='last')].sort_index()
            allw_fx  = allw_fx[~allw_fx.index.duplicated(keep='last')]
            allw_col = 'Adj Close' if 'Adj Close' in allw_raw.columns else 'Close'
            allw_merged = pd.concat([allw_raw[allw_col], allw_fx['Close']], axis=1, keys=['ALLW', 'FX'])
            allw_merged = allw_merged.ffill().dropna()
            allw_price_krw = allw_merged['ALLW'] * allw_merged['FX']
            base_allw = float(allw_price_krw.iloc[0])
            if base_allw > 0:
                allw_series = (allw_price_krw / base_allw) * IC
                allw_df = pd.DataFrame({"nav": allw_series})
                allw_df["running_max"] = allw_df["nav"].expanding().max()
                allw_df["drawdown"] = (allw_df["nav"] - allw_df["running_max"]) / allw_df["running_max"]
                allw_nav = allw_df
    except Exception:
        allw_nav = None

    # 성과 지표 (Faber A 원조 중심)
    s_value, s_return, s_mdd, s_cagr = calculate_performance_metrics(primary_nav_df, IC)
    s_peak, s_valley, _ = find_mdd_period(primary_nav_df)
    s_monthly_mdd = calculate_monthly_mdd(primary_nav_df)
    s_m_peak, s_m_valley, s_m_mdd_val = find_monthly_mdd_period(primary_nav_df)

    st.markdown(f"#### 📊 {primary_label} 전략 성과")
    st.caption(
        "원조 Faber A는 코스피200·미국나스닥100·한국채30년·미국채30년·금현물 5개 패시브 슬롯을 대상으로 "
        "각 자산이 12개월 고점 대비 -5% 이내이면 20% ON, 아니면 현금(MMF)로 대기하는 기준입니다. "
        "해남P/액티브 집행 변형은 아래 비교표와 차트에 보조 비교군으로 유지합니다."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("백테스트 기간", f"{(primary_nav_df.index[-1] - primary_nav_df.index[0]).days}일")
    c2.metric("누적 수익률", f"{s_return*100:.2f}%")
    c3.metric("CAGR", f"{s_cagr*100:.2f}%")
    c4.metric("MDD (일별)", f"{s_mdd*100:.2f}%")
    c5.metric("MDD (월별)", f"{s_monthly_mdd*100:.2f}%" if s_monthly_mdd is not None else "N/A")
    if s_peak and s_valley:
        st.info(f"📉 **MDD (일별)**: {s_peak.strftime('%Y-%m-%d')}(고점) → {s_valley.strftime('%Y-%m-%d')}(저점) | {s_mdd*100:.2f}%")
    if s_m_peak and s_m_valley:
        st.info(f"📉 **MDD (월별)**: {s_m_peak.strftime('%Y-%m-%d')}(고점) → {s_m_valley.strftime('%Y-%m-%d')}(저점) | {s_m_mdd_val*100:.2f}%")

    personal_strategy_nav = simulate_faber_strategy(
        DEFAULT_INVESTMENT_START_DATE, current_date, DEFAULT_INITIAL_CAPITAL, primary_strategy_data,
        mode='A', buffer_df=None, price_col=price_col
    )
    if personal_strategy_nav is not None and not personal_strategy_nav.empty:
        personal_df = build_personal_account_curve(
            strategy_nav=personal_strategy_nav["nav"],
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            cash_flows=PERSONAL_CASH_FLOWS,
        )
        latest_personal = personal_df.iloc[-1]
        latest_principal = calculate_cumulative_principal(
            DEFAULT_INITIAL_CAPITAL,
            PERSONAL_CASH_FLOWS,
            latest_personal.name,
        )
        latest_account_value = float(latest_personal["account_value"])
        latest_profit = latest_account_value - latest_principal
        latest_return_on_principal = latest_account_value / latest_principal - 1 if latest_principal > 0 else 0.0

        st.subheader("개인 계좌 추정(전략 NAV 기준)")
        st.caption(
            f"아래 지표는 실제 계좌 잔고가 아니라 {primary_label} 전략 NAV에 외부 입출금을 반영한 추정값입니다. "
            "실제 성과 판단은 투자 화면의 실제 계좌 입력값 기준 지표를 우선합니다."
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("누적 원금", f"{latest_principal:,.0f}원")
        col2.metric("추정 평가액", f"{latest_account_value:,.0f}원")
        col3.metric("추정 손익", f"{latest_profit:,.0f}원")
        col4.metric("추정 수익률", f"{latest_return_on_principal:.2%}")

    st.markdown("---")
    st.subheader(f"📉 {primary_label} 성과 차트")
    extra = {"이전 전략: 연속 모멘텀": (old_nav, "#ff7f0e", "dash")} if old_nav is not None else {}
    if mom_kr_samsung_nav is not None and primary_label != HAENAM_S_LABEL:
        extra[HAENAM_S_LABEL] = (mom_kr_samsung_nav, "#17becf", "dash")
    if old_haenam_nav is not None:
        extra[old_haenam_label] = (old_haenam_nav, "#8c564b", "dash")
    if primary_is_haenam:
        extra[faber_base_label] = (nav_df, "#1f77b4", "dot")
    if haenam_p_vix70_nav is not None:
        extra[HAENAM_P_VIX70_LABEL] = (haenam_p_vix70_nav, "#9467bd", "dash")
    if haenam_p_vix100_nav is not None:
        extra[HAENAM_P_VIX100_LABEL] = (haenam_p_vix100_nav, "#e377c2", "dashdot")
    if faber_nasdaq_active_nav is not None:
        extra["Faber A 나스닥 액티브 집행"] = (faber_nasdaq_active_nav, "#d62728", "dash")
    extra["동일비중 B&H"] = (static_nav, "gray", "dot")
    if allw_nav is not None:
        extra["ALLW (2025~)"] = (allw_nav, "#9467bd", "dashdot")
    fig = create_nav_and_drawdown_chart(primary_nav_df, IC, s_peak, s_valley,
        f"{primary_label} 전략: 수익률 및 Drawdown",
        monthly_peak_date=s_m_peak, monthly_valley_date=s_m_valley, monthly_mdd_val=s_m_mdd_val,
        extra_navs=extra, primary_label=primary_label)
    st.plotly_chart(fig, use_container_width=True)

    # ── 정량 비교 테이블 ─────────────────────────────────────
    st.markdown("#### 📐 전략 정량 비교")
    # 대부분 변형은 나스닥 슬롯(액티브 집행)·채권·금·현금 슬롯이 동일하고, 아래 두 축만 다르다:
    #   (A) 한국주식 20% 슬롯 집행: 코스피 액티브 2종(현버전)/코리아밸류업 액티브·패시브/코스피200 패시브
    #   (B) 투자 방법(신호): Faber A(코스피200/나스닥100 -5% 룰) vs 연속 모멘텀(0.2×12개월 모멘텀 점수)
    # 개별주(삼성전자/SK하이닉스) 변형은 표에서 숨기고, 지수·ETF 슬롯끼리만 한눈에 비교한다.
    quant_labels = [
        FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL,      # 코스피 액티브 2종 — Faber
        MOM_ACTIVE_NASDAQ_KR_ACTIVE_LABEL,      # 코스피 액티브 2종 — 연속모멘텀
        HAENAM_V_FABER_LABEL,                   # 코리아밸류업 액티브 2종 — Faber
        HAENAM_V_MOM_LABEL,                     # 코리아밸류업 액티브 2종 — 연속모멘텀
        HAENAM_V_PASSIVE_FABER_LABEL,           # 코리아밸류업 패시브 — Faber
        HAENAM_V_PASSIVE_MOM_LABEL,             # 코리아밸류업 패시브 — 연속모멘텀
        FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL,   # 코스피200TR + 나스닥 액티브 — Faber
        MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL,     # 코스피200TR + 나스닥 액티브 — 연속모멘텀
        HAENAM_P_VIX70_LABEL,                   # 해남P + VIX 일별 위기매수 — 주식 70% 상한
        HAENAM_P_VIX100_LABEL,                  # 해남P + VIX 일별 위기매수 — 주식 100% 상한
        FABER_PASSIVE_NASDAQ_KR_PASSIVE_LABEL,  # 코스피200TR + 나스닥 패시브 — Faber
        MOM_PASSIVE_NASDAQ_KR_PASSIVE_LABEL,    # 코스피200TR + 나스닥 패시브 — 연속모멘텀
    ]
    optional_quant_navs = {
        HAENAM_P_VIX70_LABEL: haenam_p_vix70_nav,
        HAENAM_P_VIX100_LABEL: haenam_p_vix100_nav,
    }
    quant_labels = [
        label for label in quant_labels
        if label not in optional_quant_navs or optional_quant_navs[label] is not None
    ]
    quant_labels = [
        FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL,
        MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL,
        HAENAM_P_LOCAL_SIGNAL_LABEL,
    ]
    if haenam_p_local_signal_nav is None:
        quant_labels = [label for label in quant_labels if label != HAENAM_P_LOCAL_SIGNAL_LABEL]
    st.caption(
        "표시 기준: "
        + " / ".join(quant_labels)
        + "만 비교합니다. 현재 본선은 원조 Faber A(코스피/나스닥 패시브 + -5%룰)입니다."
    )
    st.caption(
        "실험 열은 미국 자산 신호만 환율 제외 현지통화(QQQ/TLT/GLD) 기준으로 보고, 실제 수익은 기존 해남P처럼 원화 노출 실행 가격으로 계산합니다."
    )
    st.caption(
        "변동성(위험)은 일별 수익률 표준편차를 연율화한 값입니다. 예를 들어 연 변동성 10%는 "
        "100만원 투자 시 보통 1년 수익 변동 폭을 약 10만원 수준으로 보는 식의 참고 지표이며, "
        "손익 범위를 보장하지는 않습니다."
    )

    def _strategy_metrics(nav, ic):
        """nav DataFrame에서 CAGR/MDD/Sharpe/Sortino를 딕셔너리로 반환."""
        if nav is None or nav.empty:
            return None
        period_initial = float(nav["nav"].iloc[0])
        _, _, mdd, cagr = calculate_performance_metrics(nav, period_initial)
        monthly_mdd = calculate_monthly_mdd(nav)
        sharpe  = calculate_sharpe_ratio(nav)
        sortino = calculate_sortino_ratio(nav)
        volatility = calculate_annualized_volatility(nav)
        ulcer = calculate_ulcer_index(nav)
        martin = calculate_martin_ratio(nav, period_initial)
        cvar_5 = calculate_monthly_cvar(nav, alpha=0.05)
        pos_month = calculate_positive_month_ratio(nav)
        cagr_mdd = (cagr / abs(mdd)) if (cagr is not None and cagr > 0 and mdd is not None and mdd < 0) else None
        return {"cagr": cagr, "mdd": mdd, "monthly_mdd": monthly_mdd, "sharpe": sharpe,
                "sortino": sortino, "volatility": volatility, "cagr_mdd": cagr_mdd,
                "ulcer": ulcer, "martin": martin, "cvar_5": cvar_5,
                "pos_month": pos_month}

    def _fmt(v, fmt):
        return fmt.format(v) if v is not None else "-"

    def _fmt_pp(v):
        return f"{v * 100:+.2f}%p" if v is not None else "-"

    def _event_window_nav(nav, start_date, end_date):
        if nav is None or nav.empty:
            return None
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        window = nav[(nav.index >= start_ts) & (nav.index <= end_ts)].copy()
        if len(window) < 2 or "nav" not in window.columns:
            return None
        window["running_max"] = window["nav"].cummax()
        window["drawdown"] = (window["nav"] - window["running_max"]) / window["running_max"]
        return window

    def _event_nav_metrics(nav, start_date, end_date):
        window = _event_window_nav(nav, start_date, end_date)
        if window is None:
            return None
        daily_ret = window["nav"].pct_change().dropna()
        return {
            "start": window.index[0],
            "end": window.index[-1],
            "return": float(window["nav"].iloc[-1] / window["nav"].iloc[0] - 1.0),
            "mdd": float(window["drawdown"].min()),
            "volatility": calculate_annualized_volatility(window),
            "max_daily_loss": float(daily_ret.min()) if len(daily_ret) > 0 else None,
        }

    quant_strategies = {
        FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL: faber_active_nasdaq_kr_active_nav,
        FABER_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL: faber_active_nasdaq_kr_samsung_nav,
        FABER_ACTIVE_NASDAQ_KR_HYNIX_LABEL: faber_active_nasdaq_kr_hynix_nav,
        FABER_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL: faber_active_nasdaq_kr_semi_nav,
        # 한국=코스피200 패시브는 나스닥 액티브 집행 데이터셋과 NAV가 동일하다(한국 슬롯에 오버레이 미적용).
        FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL: faber_nasdaq_active_nav,
        # 같은 한국 슬롯 조합을 연속 모멘텀 신호로 집행한 짝.
        MOM_ACTIVE_NASDAQ_KR_ACTIVE_LABEL: mom_kr_active_nav,
        HAENAM_V_FABER_LABEL: haenam_v_faber_nav,
        HAENAM_V_MOM_LABEL: haenam_v_mom_nav,
        HAENAM_V_PASSIVE_FABER_LABEL: haenam_v_passive_faber_nav,
        HAENAM_V_PASSIVE_MOM_LABEL: haenam_v_passive_mom_nav,
        MOM_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL: mom_kr_samsung_nav,
        MOM_PASSIVE_NASDAQ_KR_SAMSUNG_LABEL: mom_passive_nasdaq_kr_samsung_nav,
        MOM_ACTIVE_NASDAQ_KR_SAMSUNG_SELF_SIGNAL_LABEL: mom_kr_samsung_self_signal_nav,
        MOM_ACTIVE_NASDAQ_KR_HYNIX_LABEL: mom_kr_hynix_nav,
        MOM_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL: mom_kr_semi_nav,
        MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL: mom_kr_passive_nav,
        HAENAM_P_LOCAL_SIGNAL_LABEL: haenam_p_local_signal_nav,
        HAENAM_P_VIX70_LABEL: haenam_p_vix70_nav,
        HAENAM_P_VIX100_LABEL: haenam_p_vix100_nav,
        FABER_PASSIVE_NASDAQ_KR_PASSIVE_LABEL: nav_df,
        MOM_PASSIVE_NASDAQ_KR_PASSIVE_LABEL: old_nav,
        # 표에는 노출하지 않지만 아래 'Faber A 원형 승률'·'해남P 하락 민감도' 계산에 필요해 정렬 대상에 유지.
        faber_base_label: nav_df,
        "이전 전략(연속 모멘텀)": old_nav,
    }
    display_quant_strategies = {
        name: quant_strategies.get(name)
        for name in quant_labels
        if quant_strategies.get(name) is not None
    }
    quant_aligned, quant_meta, quant_status_df = align_strategies_to_common_dates(
        display_quant_strategies, min_obs_days=252
    )
    if quant_meta["common_obs"] > 0:
        st.caption(
            f"📅 공통 비교 기간: {quant_meta['common_start'].strftime('%Y-%m-%d')} ~ "
            f"{quant_meta['common_end'].strftime('%Y-%m-%d')} "
            f"({quant_meta['common_obs']}거래일)"
        )
    else:
        st.warning(
            "⚠️ "
            + " / ".join(quant_labels)
            + " 간 공통 거래일이 없어 공정 비교가 불가합니다."
        )

    quant_metrics = {name: _strategy_metrics(quant_aligned.get(name), IC) for name in quant_labels}
    turnover_stats = {name: (None, None) for name in quant_labels}
    win3 = n3 = win5 = n5 = None
    if quant_meta["common_start"] is not None and quant_meta["common_end"] is not None:
        td_quant = build_trading_calendar(all_data, quant_meta["common_start"], quant_meta["common_end"])
        me_quant = _collect_month_end_dates(td_quant)
        full_keys = list(ASSETS.keys()) + [CASH_NAME]
        haenam_active_turnover_keys = primary_asset_keys + [CASH_NAME]
        haenam_semi_turnover_keys = [
            HAENAM_SAMSUNG_NAME,
            HAENAM_HYNIX_NAME,
            NASDAQ100_ASSET_NAME,
            HAENAM_TIME_NAME,
            HAENAM_KOACT_NAME,
            '한국채30년',
            '미국채30년',
            '금현물',
            CASH_NAME,
        ]
        haenam_valueup_turnover_keys = [
            KR_STOCK_MIX_ASSET,
            HAENAM_VALUEUP_TIME_NAME,
            HAENAM_VALUEUP_KOACT_NAME,
            NASDAQ100_ASSET_NAME,
            HAENAM_TIME_NAME,
            HAENAM_KOACT_NAME,
            '한국채30년',
            '미국채30년',
            '금현물',
            CASH_NAME,
        ]
        haenam_valueup_passive_turnover_keys = [
            KR_STOCK_MIX_ASSET,
            HAENAM_VALUEUP_PASSIVE_NAME,
            NASDAQ100_ASSET_NAME,
            HAENAM_TIME_NAME,
            HAENAM_KOACT_NAME,
            '한국채30년',
            '미국채30년',
            '금현물',
            CASH_NAME,
        ]
        weight_builders = {
            FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL: (
                lambda d: expand_haenam_active_backtest_weights(
                    calculate_faber_weights(d, all_data, mode='A', price_col=price_col), d
                ) if faber_active_nasdaq_kr_active_nav is not None else None
            ),
            FABER_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_faber_weights(d, all_data, mode='A', price_col=price_col), d,
                    kr_weights={"samsung": 0.5, "hynix": 0.5}
                ) if faber_active_nasdaq_kr_semi_nav is not None else None
            ),
            FABER_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_faber_weights(d, all_data, mode='A', price_col=price_col), d,
                    kr_weights={"samsung": 1.0}
                ) if faber_active_nasdaq_kr_samsung_nav is not None else None
            ),
            FABER_ACTIVE_NASDAQ_KR_HYNIX_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_faber_weights(d, all_data, mode='A', price_col=price_col), d,
                    kr_weights={"hynix": 1.0}
                ) if faber_active_nasdaq_kr_hynix_nav is not None else None
            ),
            FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL: (
                lambda d: calculate_faber_weights(
                    d, faber_nasdaq_active_data, mode='A', price_col=price_col
                ) if faber_nasdaq_active_data is not None else None
            ),
            MOM_ACTIVE_NASDAQ_KR_ACTIVE_LABEL: (
                lambda d: expand_haenam_active_backtest_weights(
                    calculate_weights_at_date(d, all_data, price_col=price_col), d
                ) if mom_kr_active_nav is not None else None
            ),
            HAENAM_V_FABER_LABEL: (
                lambda d: expand_haenam_v_backtest_weights(
                    calculate_faber_weights(d, haenam_v_strategy_data, mode='A', price_col=price_col), d
                ) if haenam_v_strategy_data is not None else None
            ),
            HAENAM_V_MOM_LABEL: (
                lambda d: expand_haenam_v_backtest_weights(
                    calculate_weights_at_date(d, haenam_v_strategy_data, price_col=price_col), d
                ) if haenam_v_strategy_data is not None else None
            ),
            HAENAM_V_PASSIVE_FABER_LABEL: (
                lambda d: expand_haenam_v_passive_backtest_weights(
                    calculate_faber_weights(d, haenam_v_passive_strategy_data, mode='A', price_col=price_col), d
                ) if haenam_v_passive_strategy_data is not None else None
            ),
            HAENAM_V_PASSIVE_MOM_LABEL: (
                lambda d: expand_haenam_v_passive_backtest_weights(
                    calculate_weights_at_date(d, haenam_v_passive_strategy_data, price_col=price_col), d
                ) if haenam_v_passive_strategy_data is not None else None
            ),
            MOM_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_weights_at_date(d, all_data, price_col=price_col), d,
                    kr_weights={"samsung": 1.0}
                ) if mom_kr_samsung_nav is not None else None
            ),
            MOM_PASSIVE_NASDAQ_KR_SAMSUNG_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_weights_at_date(d, all_data, price_col=price_col), d,
                    kr_weights={"samsung": 1.0},
                    nasdaq_active=False
                ) if mom_passive_nasdaq_kr_samsung_nav is not None else None
            ),
            MOM_ACTIVE_NASDAQ_KR_SAMSUNG_SELF_SIGNAL_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_weights_at_date(d, mom_kr_samsung_self_signal_data, price_col=price_col), d,
                    kr_weights={"samsung": 1.0}
                ) if mom_kr_samsung_self_signal_data is not None else None
            ),
            MOM_ACTIVE_NASDAQ_KR_HYNIX_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_weights_at_date(d, all_data, price_col=price_col), d,
                    kr_weights={"hynix": 1.0}
                ) if mom_kr_hynix_nav is not None else None
            ),
            MOM_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_weights_at_date(d, all_data, price_col=price_col), d,
                    kr_weights={"samsung": 0.5, "hynix": 0.5}
                ) if mom_kr_semi_nav is not None else None
            ),
            MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL: (
                lambda d: calculate_weights_at_date(d, all_data, price_col=price_col)
                if mom_kr_passive_nav is not None else None
            ),
            HAENAM_P_LOCAL_SIGNAL_LABEL: (
                lambda d: calculate_weights_at_date(d, haenam_p_local_signal_data, price_col=price_col)
                if haenam_p_local_signal_data is not None else None
            ),
            FABER_PASSIVE_NASDAQ_KR_PASSIVE_LABEL: (
                lambda d: calculate_faber_weights(d, all_data, mode='A', price_col=price_col)
                if nav_df is not None else None
            ),
            MOM_PASSIVE_NASDAQ_KR_PASSIVE_LABEL: (
                lambda d: calculate_weights_at_date(d, all_data, price_col=price_col)
                if old_nav is not None else None
            ),
            FABER_EX_BONDS_3_LABEL: (
                lambda d: calculate_faber_weights_for_assets(
                    d, faber_ex3_data, faber_ex3_assets, threshold=0.05, price_col=price_col
                ) if faber_ex3_data is not None else None
            ),
            FABER_EX_BONDS_4_LABEL: (
                lambda d: calculate_faber_weights_for_assets(
                    d, faber_ex4_data, faber_ex4_assets, threshold=0.05, price_col=price_col
                ) if faber_ex4_data is not None else None
            ),
            FABER_EX_BONDS_5_LABEL: (
                lambda d: calculate_faber_weights_for_assets(
                    d, faber_ex5_data, faber_ex5_assets, threshold=0.05, price_col=price_col
                ) if faber_ex5_data is not None else None
            ),
        }
        turnover_keys = {
            FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL: haenam_active_turnover_keys,
            FABER_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL: haenam_semi_turnover_keys,
            FABER_ACTIVE_NASDAQ_KR_HYNIX_LABEL: haenam_semi_turnover_keys,
            FABER_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL: haenam_semi_turnover_keys,
            FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL: full_keys,
            MOM_ACTIVE_NASDAQ_KR_ACTIVE_LABEL: haenam_active_turnover_keys,
            HAENAM_V_FABER_LABEL: haenam_valueup_turnover_keys,
            HAENAM_V_MOM_LABEL: haenam_valueup_turnover_keys,
            HAENAM_V_PASSIVE_FABER_LABEL: haenam_valueup_passive_turnover_keys,
            HAENAM_V_PASSIVE_MOM_LABEL: haenam_valueup_passive_turnover_keys,
            MOM_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL: haenam_semi_turnover_keys,
            MOM_PASSIVE_NASDAQ_KR_SAMSUNG_LABEL: haenam_semi_turnover_keys,
            MOM_ACTIVE_NASDAQ_KR_SAMSUNG_SELF_SIGNAL_LABEL: haenam_semi_turnover_keys,
            MOM_ACTIVE_NASDAQ_KR_HYNIX_LABEL: haenam_semi_turnover_keys,
            MOM_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL: haenam_semi_turnover_keys,
            MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL: full_keys,
            HAENAM_P_LOCAL_SIGNAL_LABEL: full_keys,
            FABER_PASSIVE_NASDAQ_KR_PASSIVE_LABEL: full_keys,
            MOM_PASSIVE_NASDAQ_KR_PASSIVE_LABEL: full_keys,
        }
        for name in quant_labels:
            builder = weight_builders.get(name)
            if builder is None:
                continue
            ws = []
            for d in me_quant:
                try:
                    w = builder(d)
                    if w is not None:
                        ws.append(w)
                except Exception:
                    pass
            avg_turn, max_turn, _ = estimate_turnover_from_weight_series(ws, turnover_keys.get(name, full_keys))
            turnover_stats[name] = (avg_turn, max_turn)

        qf = quant_aligned.get(faber_base_label)
        qo = quant_aligned.get("이전 전략(연속 모멘텀)")
        win3, n3 = calculate_rolling_outperformance_rate(qf, qo, window_months=36)
        win5, n5 = calculate_rolling_outperformance_rate(qf, qo, window_months=60)

    if all(quant_metrics.get(name) is not None for name in quant_labels):
        # (지표명, quant_metrics 키, 표시 포맷, 좋은 방향) — 방향은 행별 최우수 셀 강조에 쓴다.
        metric_specs = [
            ("CAGR", "cagr", "{:.2%}", "max"),
            ("MDD (일별)", "mdd", "{:.2%}", "max"),          # 덜 빠질수록(0에 가까울수록) 우수
            ("MDD (월말)", "monthly_mdd", "{:.2%}", "max"),  # 월말 종가 기준 최대 낙폭
            ("변동성 (위험)", "volatility", "{:.2%}", "min"),
            ("Sharpe", "sharpe", "{:.2f}", "max"),
            ("Sortino", "sortino", "{:.2f}", "max"),
            ("CAGR / MDD", "cagr_mdd", "{:.2f}", "max"),
            ("Ulcer Index", "ulcer", "{:.2f}", "min"),
            ("Martin Ratio", "martin", "{:.2f}", "max"),
            ("CVaR 5% (월)", "cvar_5", "{:.2%}", "max"),     # 덜 손실일수록(0에 가까울수록) 우수
            ("양(+)월 비율", "pos_month", "{:.1%}", "max"),
        ]
        row_directions = {}
        comparison_rows = []
        for row_label, key, fmt, direction in metric_specs:
            row_directions[row_label] = direction
            comparison_rows.append(
                tuple([row_label] + [_fmt(quant_metrics[name].get(key), fmt) for name in quant_labels])
            )
        row_directions["평균 월회전율(추정)"] = "min"
        comparison_rows.append(
            tuple(["평균 월회전율(추정)"] + [_fmt(turnover_stats[name][0], "{:.1%}") for name in quant_labels])
        )
        row_directions["최대 월회전율(추정)"] = "min"
        comparison_rows.append(
            tuple(["최대 월회전율(추정)"] + [_fmt(turnover_stats[name][1], "{:.1%}") for name in quant_labels])
        )
        df_cmp = pd.DataFrame(comparison_rows, columns=["지표"] + quant_labels)

        def _parse_metric_cell(s):
            if s is None or s == "-":
                return None
            try:
                return float(str(s).replace("%", "").replace(",", "").strip())
            except ValueError:
                return None

        def _highlight_best_in_row(row):
            """각 지표 행에서 '좋은 방향' 기준 최우수 셀(동률 포함)에 초록 배경을 입힌다."""
            styles = ["" for _ in row.index]
            direction = row_directions.get(row["지표"])
            if direction is None:
                return styles
            vals = {col: _parse_metric_cell(row[col]) for col in quant_labels}
            nums = [v for v in vals.values() if v is not None]
            if not nums:
                return styles
            best = max(nums) if direction == "max" else min(nums)
            for i, col in enumerate(row.index):
                if col in quant_labels and vals.get(col) is not None and abs(vals[col] - best) < 1e-9:
                    styles[i] = "background-color: #c6efce; color: #006100; font-weight: 600"
            return styles

        st.caption("🟩 각 지표(행)에서 가장 좋은 값에 초록색 표시 — 한눈에 우열 비교용(동률이면 함께 표시).")
        st.dataframe(
            df_cmp.style.apply(_highlight_best_in_row, axis=1),
            use_container_width=True, hide_index=True,
        )
        if win3 is not None or win5 is not None:
            rel_rows = [{
                "상대지표": "Faber A 원형 승률 (3년 롤링, 누적수익률 기준)",
                "값": f"{win3*100:.1f}% ({n3}구간)" if win3 is not None else "-",
            }, {
                "상대지표": "Faber A 원형 승률 (5년 롤링, 누적수익률 기준)",
                "값": f"{win5*100:.1f}% ({n5}구간)" if win5 is not None else "-",
            }]
            st.dataframe(pd.DataFrame(rel_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ 전략 정량 비교를 위한 공통 기간 데이터가 부족합니다.")

    event_active_nav = quant_aligned.get(MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL)
    event_passive_nav = quant_aligned.get(MOM_PASSIVE_NASDAQ_KR_PASSIVE_LABEL)
    event_rows = []
    event_windows = [
        ("2025년 4월 관세 이슈", pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30")),
        ("2026년 3월 전쟁 이슈", pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31")),
    ]
    for event_name, event_start, event_end in event_windows:
        active_stats = _event_nav_metrics(event_active_nav, event_start, event_end)
        passive_stats = _event_nav_metrics(event_passive_nav, event_start, event_end)
        if active_stats is None or passive_stats is None:
            continue
        mdd_gap = abs(active_stats["mdd"]) - abs(passive_stats["mdd"])
        vol_gap = (
            active_stats["volatility"] - passive_stats["volatility"]
            if active_stats["volatility"] is not None and passive_stats["volatility"] is not None else None
        )
        loss_gap = (
            abs(active_stats["max_daily_loss"]) - abs(passive_stats["max_daily_loss"])
            if active_stats["max_daily_loss"] is not None and passive_stats["max_daily_loss"] is not None else None
        )
        event_rows.append({
            "이벤트": event_name,
            "실제 비교 거래일": f"{active_stats['start'].strftime('%Y-%m-%d')}~{active_stats['end'].strftime('%Y-%m-%d')}",
            "액티브 MDD": _fmt(active_stats["mdd"], "{:.2%}"),
            "패시브 MDD": _fmt(passive_stats["mdd"], "{:.2%}"),
            "액티브 MDD 더 큼": _fmt_pp(mdd_gap),
            "액티브 위험": _fmt(active_stats["volatility"], "{:.2%}"),
            "패시브 위험": _fmt(passive_stats["volatility"], "{:.2%}"),
            "액티브 위험 더 큼": _fmt_pp(vol_gap),
            "액티브 최대 일손실 더 큼": _fmt_pp(loss_gap),
            "액티브 누적수익": _fmt(active_stats["return"], "{:+.2%}"),
            "패시브 누적수익": _fmt(passive_stats["return"], "{:+.2%}"),
        })
    if event_rows:
        st.markdown("#### 🔥 해남P 나스닥 액티브 vs 패시브 이벤트 위험")
        st.caption(
            "해남P(코스피200TR+나스닥 액티브)를 해남P 나스닥 패시브 버전과 비교합니다. "
            "MDD는 각 이벤트 월의 첫 거래일 NAV부터 새로 계산했고, 위험은 해당 구간 일수익률 변동성의 연율화 값입니다."
        )
        st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("⚠️ 해남P 나스닥 액티브/패시브 이벤트 비교에 필요한 공통 거래일 데이터가 부족합니다.")

    haenam_downside = calculate_strategy_downside_comparison(
        quant_aligned.get(faber_base_label),
        quant_aligned.get(MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL),
    )
    if haenam_downside is not None:
        st.markdown("#### 🔎 해남P 초과수익 vs 하락 민감도")
        st.caption(
            "Faber A가 오른 달과 빠진 달을 나눠서 해남P가 추가 수익을 냈는지, "
            "아니면 하락월 손실만 더 키웠는지 보는 표입니다."
        )
        stress_note = (
            f"{haenam_downside['stress_months']}개월"
            if haenam_downside["stress_months"] > 0 else "해당 없음"
        )
        downside_rows = [
            {
                "구분": "전체 월",
                "월수": f"{haenam_downside['months']}개월",
                "Faber A 평균": "-",
                "해남P 평균": "-",
                "해남P 초과": _fmt(haenam_downside["avg_excess"], "{:+.2%}"),
                "해석": "월평균 초과수익이 양수면 장기 알파 후보",
            },
            {
                "구분": "Faber A 상승월",
                "월수": f"{haenam_downside['up_months']}개월",
                "Faber A 평균": _fmt(haenam_downside["up_base_avg"], "{:.2%}"),
                "해남P 평균": _fmt(haenam_downside["up_target_avg"], "{:.2%}"),
                "해남P 초과": _fmt(haenam_downside["up_excess_avg"], "{:+.2%}"),
                "해석": "상승장에서 집행 알파가 붙는지 확인",
            },
            {
                "구분": "Faber A 하락월",
                "월수": f"{haenam_downside['down_months']}개월",
                "Faber A 평균": _fmt(haenam_downside["down_base_avg"], "{:.2%}"),
                "해남P 평균": _fmt(haenam_downside["down_target_avg"], "{:.2%}"),
                "해남P 초과": _fmt(haenam_downside["down_excess_avg"], "{:+.2%}"),
                "해석": "음수가 클수록 방어 비용이 큼",
            },
            {
                "구분": "Faber A -5% 이하 월",
                "월수": stress_note,
                "Faber A 평균": _fmt(haenam_downside["stress_base_avg"], "{:.2%}"),
                "해남P 평균": _fmt(haenam_downside["stress_target_avg"], "{:.2%}"),
                "해남P 초과": _fmt(haenam_downside["stress_excess_avg"], "{:+.2%}"),
                "해석": "큰 하락장에서 더 깨지는지 확인",
            },
            {
                "구분": "하락월 베타",
                "월수": f"{haenam_downside['down_months']}개월 기준",
                "Faber A 평균": "1.00",
                "해남P 평균": _fmt(haenam_downside["down_beta"], "{:.2f}"),
                "해남P 초과": "-",
                "해석": "1보다 크면 Faber A 하락에 더 민감",
            },
            {
                "구분": "하락월 더 손실 빈도",
                "월수": f"{haenam_downside['down_months']}개월 기준",
                "Faber A 평균": "-",
                "해남P 평균": _fmt(haenam_downside["down_worse_rate"], "{:.1%}"),
                "해남P 초과": "-",
                "해석": "하락월에 해남P가 더 나빴던 비율",
            },
            {
                "구분": "상승월 캡처",
                "월수": f"{haenam_downside['up_months']}개월 기준",
                "Faber A 평균": "1.00",
                "해남P 평균": _fmt(haenam_downside["up_capture"], "{:.2f}"),
                "해남P 초과": "-",
                "해석": "1보다 크면 상승장에서 더 강하게 따라감",
            },
        ]
        st.dataframe(pd.DataFrame(downside_rows), use_container_width=True, hide_index=True)

    quant_warn_df = quant_status_df[quant_status_df["상태"] != "비교 가능"]
    if not quant_warn_df.empty:
        st.caption("⚠️ 공정 비교 주의/제외 전략")
        st.dataframe(quant_warn_df, use_container_width=True, hide_index=True)

    # ── 정적 자산배분(20% 고정) vs 해남P 비교 ─────────────

    st.markdown("---")
    st.subheader(f"📊 정적 자산배분 (20% 균등) vs {primary_label}")
    st.caption(f"현금 없이 5자산 각 20%를 고정 후 월말 리밸런싱. {primary_label}의 타이밍/집행 조합이 단순 분산 대비 얼마나 유효한지 확인합니다.")

    eq_nav = simulate_equal_weight_no_cash(
        bt_start_date, current_date, IC, all_data, price_col=price_col)

    static_compare_map = {
        f'{primary_label} (실전 집행) ⭐': primary_nav_df,
        faber_base_label: nav_df,
        '정적 균등 (20%×5, 현금無)': eq_nav,
        '동일비중 B&H (현금포함)': static_nav,
    }
    static_aligned, static_meta, static_status_df = align_strategies_to_common_dates(
        static_compare_map, min_obs_days=252
    )
    if static_meta["common_obs"] > 0:
        st.caption(
            f"📅 공통 비교 기간: {static_meta['common_start'].strftime('%Y-%m-%d')} ~ "
            f"{static_meta['common_end'].strftime('%Y-%m-%d')} "
            f"({static_meta['common_obs']}거래일)"
        )
    else:
        st.warning(f"⚠️ {primary_label}와 정적 전략 간 공통 거래일이 없어 공정 비교가 불가합니다.")

    primary_static = static_aligned.get(f'{primary_label} (실전 집행) ⭐')
    faber_static = static_aligned.get(faber_base_label)
    eq_static = static_aligned.get('정적 균등 (20%×5, 현금無)')
    bh_static = static_aligned.get('동일비중 B&H (현금포함)')

    if primary_static is not None and eq_static is not None:
        cmp_input = {
            f'{primary_label} (실전 집행) ⭐': primary_static,
            '정적 균등 (20%×5, 현금無)': eq_static,
        }
        if faber_static is not None:
            cmp_input[faber_base_label] = faber_static
        if bh_static is not None:
            cmp_input['동일비중 B&H (현금포함)'] = bh_static
        static_cmp = build_comparison_table(cmp_input, IC)
        if static_cmp is not None:
            st.dataframe(static_cmp, use_container_width=True)

        # 비교 차트
        fig_static = make_subplots(rows=2, cols=1,
            subplot_titles=("수익률 (%)", "Drawdown (%)"),
            vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)
        primary_pct = ((primary_static['nav'] / IC) - 1) * 100
        eq_pct = ((eq_static['nav'] / IC) - 1) * 100
        fig_static.add_trace(go.Scatter(x=primary_static.index, y=primary_pct, mode='lines',
            name=f'{primary_label} ⭐', line=dict(color='#1f77b4', width=2),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{primary_label}: %{{y:.1f}}%<extra></extra>"), row=1, col=1)
        if faber_static is not None:
            faber_pct = ((faber_static['nav'] / IC) - 1) * 100
            fig_static.add_trace(go.Scatter(x=faber_static.index, y=faber_pct, mode='lines',
                name=faber_base_label, line=dict(color='#2ca02c', width=1.2, dash='dot'),
                hovertemplate="%{x|%Y-%m-%d}<br>Faber A 원형: %{y:.1f}%<extra></extra>"), row=1, col=1)
        fig_static.add_trace(go.Scatter(x=eq_static.index, y=eq_pct, mode='lines',
            name='정적 균등 20%×5', line=dict(color='#d62728', width=2, dash='dash'),
            hovertemplate="%{x|%Y-%m-%d}<br>정적: %{y:.1f}%<extra></extra>"), row=1, col=1)
        if bh_static is not None:
            st_pct = ((bh_static['nav'] / IC) - 1) * 100
            fig_static.add_trace(go.Scatter(x=bh_static.index, y=st_pct, mode='lines',
                name='동일비중+현금', line=dict(color='gray', width=1, dash='dot'),
                hovertemplate="%{x|%Y-%m-%d}<br>B&H: %{y:.1f}%<extra></extra>"), row=1, col=1)
        fig_static.add_trace(go.Scatter(x=primary_static.index, y=primary_static['drawdown']*100,
            mode='lines', name=f'DD {primary_label}', fill='tozeroy',
            line=dict(color='#1f77b4', width=1)), row=2, col=1)
        fig_static.add_trace(go.Scatter(x=eq_static.index, y=eq_static['drawdown']*100,
            mode='lines', name='DD 정적',
            line=dict(color='#d62728', width=1.5, dash='dash')), row=2, col=1)
        fig_static.update_yaxes(title_text="수익률 (%)", row=1, col=1)
        fig_static.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
        fig_static.update_layout(
            title=f"{primary_label} vs 정적 균등 자산배분 (20%×5)",
            height=650, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_static, use_container_width=True)
        st.caption("💡 **정적 균등**: 시장 상황에 관계없이 5자산 20%씩 유지. "
                   f"{primary_label}보다 MDD가 크지만 CAGR도 높을 수 있음 — 타이밍 비용 vs 하락 방어 트레이드오프.")

        static_warn_df = static_status_df[static_status_df["상태"] != "비교 가능"]
        if not static_warn_df.empty:
            st.caption("⚠️ 공정 비교 주의/제외 전략")
            st.dataframe(static_warn_df, use_container_width=True, hide_index=True)

        # ── 회복력 분석 ─────────────────────────────────────
        st.markdown("#### 📉 회복력 분석")
        st.caption("MDD 회복기간, 평균/최장 회복기간, Underwater 비율 비교.")

        def _calc_recovery(nav_s):
            if nav_s is None or nav_s.empty: return None
            running_max = nav_s.expanding().max()
            underwater_ratio = (nav_s < running_max).sum() / len(nav_s)
            periods, period_infos, in_dd, peak_idx = [], [], False, 0
            nav_arr = nav_s.values
            dates = nav_s.index
            rmax_arr = running_max.values
            for i in range(len(nav_arr)):
                if nav_arr[i] < rmax_arr[i]:
                    if not in_dd:
                        in_dd = True
                        # peak는 현재 running_max가 처음 달성된 시점
                        peak_idx = i
                        for j in range(i, -1, -1):
                            if rmax_arr[j] < rmax_arr[i]:
                                peak_idx = j + 1
                                break
                            if j == 0:
                                peak_idx = 0
                else:
                    if in_dd:
                        days = (dates[i] - dates[peak_idx]).days
                        if days > 0:
                            periods.append(days)
                            period_infos.append({
                                "days": days,
                                "start": dates[peak_idx],
                                "end": dates[i],
                            })
                        in_dd = False
            def fmt(d):
                if d is None or d == 0: return "-"
                y, m = int(d//365), int((d%365)//30)
                if y > 0 and m > 0: return f"{y}년 {m}개월"
                if y > 0: return f"{y}년"
                if m > 0: return f"{m}개월"
                return f"{d}일"
            max_info = max(period_infos, key=lambda x: x["days"]) if period_infos else None
            return {
                'Underwater 비율': f"{underwater_ratio*100:.1f}%",
                '평균 회복기간': fmt(int(sum(periods)/len(periods))) if periods else "-",
                '최장 회복기간': fmt(max(periods)) if periods else "-",
                '최장 회복 구간': (
                    f"{max_info['start'].strftime('%Y-%m-%d')} → {max_info['end'].strftime('%Y-%m-%d')}"
                    if max_info else "-"
                ),
            }

        recovery_map = {
            f'{primary_label} ⭐': primary_nav_df,
            faber_base_label: nav_df,
            '정적 균등 (20%×5)': eq_nav,
            '이전 전략(연속 모멘텀)': old_nav,
        }
        rec_aligned, rec_meta, rec_status_df = align_strategies_to_common_dates(
            recovery_map, min_obs_days=252
        )
        if rec_meta["common_obs"] > 0:
            st.caption(
                f"📅 공통 비교 기간: {rec_meta['common_start'].strftime('%Y-%m-%d')} ~ "
                f"{rec_meta['common_end'].strftime('%Y-%m-%d')} "
                f"({rec_meta['common_obs']}거래일)"
            )

        rec_rows = []
        for lbl in [f'{primary_label} ⭐', faber_base_label, '정적 균등 (20%×5)', '이전 전략(연속 모멘텀)']:
            nav_aligned = rec_aligned.get(lbl)
            r = _calc_recovery(nav_aligned["nav"]) if nav_aligned is not None else None
            if r: rec_rows.append({'전략': lbl, **r})
        if rec_rows:
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
            st.caption("💡 **Underwater 비율**: 전체 기간 중 고점 아래 있던 비중. **최장 회복기간**: 한 번 꺾인 후 회복까지 최대 시간.")

        rec_warn_df = rec_status_df[rec_status_df["상태"] != "비교 가능"]
        if not rec_warn_df.empty:
            st.caption("⚠️ 회복력 분석 주의/제외 전략")
            st.dataframe(rec_warn_df, use_container_width=True, hide_index=True)
    else:
        st.warning("정적 균등 전략 데이터가 부족해 공정 비교를 수행할 수 없습니다.")

    # Faber A 원조 월별 비중 변화
    st.markdown("---")
    st.subheader(f"📊 {primary_label} 월별 자산 배분 비중")
    st.caption("💡 **기준신호**: 12개월 고점 대비 -5% 이내면 해당 패시브 자산 20% ON. OFF 슬롯은 현금(MMF)로 대기합니다.")
    
    trading_dates_all = build_trading_calendar(all_data, bt_start_date, current_date)
    faber_month_ends = _collect_month_end_dates(trading_dates_all)
    
    primary_weight_records = []
    for d in faber_month_ends:
        base_w = calculate_haenam_p_weights(d, primary_strategy_data, price_col=price_col) if primary_is_haenam else calculate_faber_weights(d, all_data, mode='A', price_col=price_col)
        w = expand_haenam_p_execution_weights(base_w, d) if primary_is_haenam else base_w
        row = {"date": d}
        for an in primary_asset_keys:
            row[an] = w.get(an, 0.0)
        row[CASH_NAME] = w.get(CASH_NAME, 0.0)
        primary_weight_records.append(row)

    if primary_weight_records:
        df_fw = pd.DataFrame(primary_weight_records)
        df_fw["date"] = pd.to_datetime(df_fw["date"])
        df_fw = df_fw.sort_values("date")
        visible_weight_cols = _active_contribution_cols(df_fw, primary_asset_keys, threshold=0.000001)
        if CASH_NAME in df_fw.columns:
            visible_weight_cols = visible_weight_cols + [CASH_NAME]
        # 비중 차트 (stacked area)
        asset_colors = {'코스피200': '#1f77b4', '미국나스닥100': '#ff7f0e', '한국채30년': '#2ca02c',
                       '미국채30년': '#d62728', '금현물': '#FFD700', CASH_NAME: '#9467bd',
                       HAENAM_SAMSUNG_NAME: '#4c78a8', HAENAM_HYNIX_NAME: '#72b7b2', HAENAM_KR_TIME_NAME: '#54a24b', HAENAM_KR_KOACT_NAME: '#b279a2',
                       HAENAM_TIME_NAME: '#f58518', HAENAM_KOACT_NAME: '#e45756'}
        fig_fw = go.Figure()
        acols = visible_weight_cols
        for ac in acols:
            fig_fw.add_trace(go.Scatter(x=df_fw["date"], y=df_fw[ac]*100, mode="lines",
                name=ac, stackgroup="one", line=dict(width=0),
                fillcolor=asset_colors.get(ac, "#7f7f7f")))
        fig_fw.update_layout(title=f"{primary_label} 월별 자산 배분 비중", xaxis_title="날짜",
            yaxis_title="비중 (%)", yaxis_range=[0, 100], height=400, hovermode="x unified")
        st.plotly_chart(fig_fw, use_container_width=True)

        with st.expander(f"📋 {primary_label} 월별 비중 상세"):
            disp_fw = df_fw.copy()
            disp_fw["월"] = disp_fw["date"].dt.strftime("%Y-%m")
            visible_asset_cols = [c for c in visible_weight_cols if c != CASH_NAME]
            for an in visible_asset_cols:
                disp_fw[an] = disp_fw[an].apply(lambda x: f"●{x*100:.0f}%" if x > 0.01 else "○0%")
            disp_fw[CASH_NAME] = (disp_fw[CASH_NAME]*100).round(0).astype(int).astype(str) + "%"
            disp_fw = _rename_asset_columns_for_display(disp_fw)
            st.dataframe(disp_fw[["월"] + [_display_asset_name(c) for c in visible_weight_cols]],
                         use_container_width=True, hide_index=True, height=400)

    # 연도별 성과 요약 (해남P 기준)
    if primary_nav_df is not None and not primary_nav_df.empty:
        years = sorted(primary_nav_df.index.year.unique())
        if years:
            st.markdown("---")
            st.subheader(f"📅 {primary_label} 연도별 성과 요약")
            stats_list = [calculate_yearly_daily_stats(primary_nav_df, y) for y in years]
            stats_list = [s for s in stats_list if s is not None]
            if stats_list:
                dfy = pd.DataFrame(stats_list)
                st.dataframe(pd.DataFrame({
                    "연도": dfy["year"], "연간 수익률": dfy["yearly_return"].apply(lambda x: f"{x*100:.2f}%"),
                    "거래일": dfy["total_days"], "상승일": dfy["up_days"], "하락일": dfy["down_days"],
                    "상승 평균": dfy["up_mean"].apply(lambda x: f"{x*100:.3f}%" if pd.notna(x) else "-"),
                    "하락 평균": dfy["down_mean"].apply(lambda x: f"{x*100:.3f}%" if pd.notna(x) else "-"),
                    "최대 일손실": dfy["max_daily_loss"].apply(lambda x: f"{x*100:.3f}%" if pd.notna(x) else "-"),
                }), use_container_width=True, hide_index=True)

    # 해남P 월별 자산 수익 기여도 분석
    primary_attr_list = []
    if primary_weight_records and len(primary_weight_records) > 1:
        st.markdown("---")
        st.subheader(f"🔍 {primary_label} 월별 자산 수익 기여도 분석")

        # 월말 비중 + 다음달 자산 수익률 → 기여도
        for i in range(len(primary_weight_records) - 1):
            w_rec = primary_weight_records[i]
            next_rec = primary_weight_records[i + 1]
            d_start = w_rec["date"]
            d_end = next_rec["date"]
            
            attr = {"date": d_end.strftime("%Y-%m")}
            total = 0.0
            for an in primary_asset_keys + [CASH_NAME]:
                wt = w_rec.get(an, 0.0)
                p1 = get_price_at_date(primary_price_data.get(an), d_start, price_col=price_col)
                p2 = get_price_at_date(primary_price_data.get(an), d_end, price_col=price_col)
                if p1 and p2 and p1 > 0 and wt > 0:
                    ret = (p2 / p1) - 1
                    contrib = wt * ret * 100  # pp
                else:
                    contrib = 0.0
                attr[an] = round(contrib, 2)
                total += contrib
            attr["합계"] = round(total, 2)
            primary_attr_list.append(attr)

        if primary_attr_list:
            df_fa = pd.DataFrame(primary_attr_list)
            all_yrs = sorted(set(int(d[:4]) for d in df_fa["date"]))
            if len(all_yrs) > 3:
                c1, c2 = st.columns(2)
                with c1: yr_s = st.selectbox("시작 연도", all_yrs, index=max(0, len(all_yrs)-3), key="fattr_yr_s")
                with c2:
                    opts = [y for y in all_yrs if y >= yr_s]
                    yr_e = st.selectbox("종료 연도", opts, index=len(opts)-1, key="fattr_yr_e")
            else: yr_s, yr_e = all_yrs[0], all_yrs[-1]
            
            mask = df_fa["date"].apply(lambda x: yr_s <= int(x[:4]) <= yr_e)
            df_filt = df_fa[mask].copy()
            
            if len(df_filt) > 0:
                # 차트
                asset_cols = _active_contribution_cols(
                    df_filt, [c for c in df_filt.columns if c not in ["date", "합계"]]
                )
                fig_fa = go.Figure()
                attr_colors = {'코스피200': '#1f77b4', '미국나스닥100': '#ff7f0e', '한국채30년': '#2ca02c',
                              '미국채30년': '#d62728', '금현물': '#FFD700', CASH_NAME: '#9467bd',
                              HAENAM_SAMSUNG_NAME: '#4c78a8', HAENAM_HYNIX_NAME: '#72b7b2', HAENAM_KR_TIME_NAME: '#54a24b', HAENAM_KR_KOACT_NAME: '#b279a2',
                              HAENAM_TIME_NAME: '#f58518', HAENAM_KOACT_NAME: '#e45756'}
                for ac in asset_cols:
                    display_ac = _display_asset_name(ac)
                    fig_fa.add_trace(go.Bar(x=df_filt["date"], y=df_filt[ac], name=display_ac,
                        marker_color=attr_colors.get(ac, "#7f7f7f"),
                        text=[f"{v:+.2f}pp" if abs(v) > 0.005 else "" for v in df_filt[ac]],
                        textposition="inside", textfont=dict(size=8),
                        hovertemplate="%{x}<br>" + display_ac + ": %{y:.2f}pp<extra></extra>"))
                # 합계 라인 (바 아래쪽에 표시)
                fig_fa.add_trace(go.Scatter(x=df_filt["date"], y=df_filt["합계"], mode="lines+markers",
                    name="월 수익률", line=dict(color="black", width=1.5),
                    hovertemplate="%{x}<br>월 합계: %{y:.2f}pp<extra></extra>"))
                # 월 수익률 라벨: 항상 바 위에 표시 (별도 trace)
                # 각 월의 양수 바 합계 계산해서 그 위에 표시
                top_positions = []
                for _, row in df_filt.iterrows():
                    pos_sum = sum(max(0, row[ac]) for ac in asset_cols)
                    top_positions.append(max(pos_sum, 0.5) + 0.6)  # 바 위 여백
                fig_fa.add_trace(go.Scatter(x=df_filt["date"], y=top_positions, mode="text",
                    text=[f"{v:+.2f}pp" for v in df_filt["합계"]], textposition="top center",
                    textfont=dict(size=10, color="black"), showlegend=False, hoverinfo="skip"))
                fig_fa.update_layout(title=f"{primary_label} 월별 자산 수익 기여도", xaxis_title="월", yaxis_title="기여도 (pp)",
                    barmode="relative", height=550, hovermode="x unified")
                st.plotly_chart(fig_fa, use_container_width=True)

                with st.expander("📊 월별 수익률 상세"):
                    st.dataframe(_rename_asset_columns_for_display(df_filt), use_container_width=True, hide_index=True, height=400)

    # ==============================
    # 자산별 역할 분석
    # ==============================
    if primary_attr_list and len(primary_attr_list) > 0:
        st.markdown("---")
        st.subheader(f"🎯 자산별 역할 분석 ({primary_label})")
        st.caption("코스피200·나스닥100 기준 자산군으로 합산해 각 자산이 포트폴리오에 얼마나 기여했는지 확인합니다.")

        df_all_attr = pd.DataFrame(primary_attr_list)
        df_major_attr = _aggregate_major_contribution_df(df_all_attr)
        major_asset_names = [
            KR_STOCK_MIX_ASSET,
            NASDAQ100_ASSET_NAME,
            "한국채30년",
            "미국채30년",
            "금현물",
            CASH_NAME,
        ]
        major_asset_names = _active_contribution_cols(df_major_attr, major_asset_names, threshold=0.005)

        # 총 분석 기간 표시
        total_analysis_months = len(df_all_attr)
        total_portfolio_return = df_all_attr["합계"].sum()
        positive_portfolio_months = (df_all_attr["합계"] > 0.01).sum()
        negative_portfolio_months = (df_all_attr["합계"] < -0.01).sum()
        # 포트폴리오 손익비 (평균 수익 / |평균 손실|)
        port_gains = df_all_attr["합계"][df_all_attr["합계"] > 0.01]
        port_losses = df_all_attr["합계"][df_all_attr["합계"] < -0.01]
        avg_port_gain = float(port_gains.mean()) if len(port_gains) > 0 else 0
        avg_port_loss = float(port_losses.mean()) if len(port_losses) > 0 else 0
        port_plr = abs(avg_port_gain / avg_port_loss) if abs(avg_port_loss) > 0.001 else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("총 분석 기간", f"{total_analysis_months}개월")
        c2.metric("포트폴리오 누적 기여", f"{total_portfolio_return:.1f}pp")
        c3.metric("수익 월 / 손실 월", f"{positive_portfolio_months} / {negative_portfolio_months}")
        c4.metric("포트폴리오 승률", f"{positive_portfolio_months/max(positive_portfolio_months+negative_portfolio_months,1)*100:.0f}%")
        c5.metric("손익비", f"{port_plr:.2f}", help=f"평균수익 {avg_port_gain:+.2f}pp / 평균손실 {avg_port_loss:.2f}pp")

        major_group_sources = {
            KR_STOCK_MIX_ASSET: [KR_STOCK_MIX_ASSET, HAENAM_SAMSUNG_NAME, HAENAM_HYNIX_NAME, HAENAM_KR_TIME_NAME, HAENAM_KR_KOACT_NAME],
            NASDAQ100_ASSET_NAME: [NASDAQ100_ASSET_NAME, HAENAM_TIME_NAME, HAENAM_KOACT_NAME],
            "한국채30년": ["한국채30년"],
            "미국채30년": ["미국채30년"],
            "금현물": ["금현물"],
            CASH_NAME: [CASH_NAME],
        }
        major_held_counts = {an: 0 for an in major_asset_names}
        for w_rec in primary_weight_records[:-1]:
            for an in major_asset_names:
                held_weight = sum(float(w_rec.get(src, 0.0) or 0.0) for src in major_group_sources.get(an, [an]))
                if held_weight > 0.000001:
                    major_held_counts[an] = major_held_counts.get(an, 0) + 1

        role_rows = []
        for an in major_asset_names:
            vals = df_major_attr[an].values
            total_months = len(vals)
            positive_months = sum(1 for v in vals if v > 0.01)
            negative_months = sum(1 for v in vals if v < -0.01)
            zero_months = total_months - positive_months - negative_months
            held_months = major_held_counts.get(an, total_months - zero_months)
            win_rate = positive_months / max(held_months, 1) * 100
            cumulative = sum(vals)
            avg_gain = np.mean([v for v in vals if v > 0.01]) if positive_months > 0 else 0
            avg_loss = np.mean([v for v in vals if v < -0.01]) if negative_months > 0 else 0
            # 위기 방어: 전체 포트폴리오가 마이너스인 달에 이 자산이 플러스인 횟수
            crisis_months = [i for i, v in enumerate(df_major_attr["합계"].values) if v < -0.5]
            defense_count = sum(1 for i in crisis_months if vals[i] > 0.01) if crisis_months else 0
            defense_rate = defense_count / max(len(crisis_months), 1) * 100
            crisis_invested = sum(1 for i in crisis_months if abs(vals[i]) > 0.01) if crisis_months else 0
            held_defense_rate = defense_count / max(crisis_invested, 1) * 100

            role_rows.append({
                "자산": _major_asset_display_name(an),
                "투자월": f"{held_months}개월",
                "기여월": f"{total_months - zero_months}개월",
                "수익월": f"{positive_months}",
                "손실월": f"{negative_months}",
                "승률": f"{win_rate:.0f}%",
                "누적 기여": f"{cumulative:.1f}pp",
                "평균 수익": f"+{avg_gain:.2f}pp",
                "평균 손실": f"{avg_loss:.2f}pp",
                "위기방어": f"{defense_count}/{len(crisis_months)} ({defense_rate:.0f}%)",
                "보유시 위기방어": f"{defense_count}/{crisis_invested} ({held_defense_rate:.0f}%)" if crisis_invested > 0 else "-",
            })

        df_role = pd.DataFrame(role_rows)
        st.dataframe(df_role, use_container_width=True, hide_index=True)
        st.caption("💡 **투자월**: 월초 비중이 있었던 달. **기여월**: 기여도가 ±0.01pp를 넘은 달. **승률**: 투자월 중 플러스 기여 월 비율. **위기방어**: 포트폴리오 전체가 -0.5pp 이상 손실인 달에 해당 자산이 플러스 기여한 횟수.")

        # 누적 기여도 차트: 실전 집행 종목은 6개 큰 자산군으로 합산해 표시한다.
        cumul_data = {}
        for an in major_asset_names:
            cumul_data[an] = np.cumsum(df_major_attr[an].values)

        fig_role = go.Figure()
        role_colors = {'코스피200': '#1f77b4', '미국나스닥100': '#ff7f0e', '한국채30년': '#2ca02c',
                      '미국채30년': '#d62728', '금현물': '#FFD700', CASH_NAME: '#9467bd'}
        for an in major_asset_names:
            display_an = _major_asset_display_name(an)
            fig_role.add_trace(go.Scatter(x=df_major_attr["date"], y=cumul_data[an], mode="lines",
                name=display_an, line=dict(color=role_colors.get(an, "#7f7f7f"), width=2),
                hovertemplate=f"%{{x}}<br>{display_an}: %{{y:.1f}}pp<extra></extra>"))
        fig_role.update_layout(title="자산별 누적 기여도 (pp)", xaxis_title="월", yaxis_title="누적 기여 (pp)",
            height=450, hovermode="x unified")
        st.plotly_chart(fig_role, use_container_width=True)

    st.markdown("---")
    st.subheader(f"📊 {primary_label} 휩소(Whipsaw) 분석")
    st.caption("코스피200·나스닥100 등 Faber A 패시브 슬롯의 -5%룰 ON/OFF 변화 빈도를 분석합니다.")

    # 휩소 분석은 집행 종목이 아니라 원 신호 슬롯 기준으로 본다.
    with st.expander(f"📊 {primary_label} 휩소(Whipsaw) 분석: 월별 신호 변화"):
        trading_dates_for_whipsaw = build_trading_calendar(all_data, bt_start_date, current_date)
        month_ends = _collect_month_end_dates(trading_dates_for_whipsaw)
        whipsaw_asset_keys = [
            KR_STOCK_MIX_ASSET,
            NASDAQ100_ASSET_NAME,
            "한국채30년",
            "미국채30년",
            "금현물",
        ]
        whipsaw_asset_keys = [an for an in whipsaw_asset_keys if an]

        whipsaw_records = []
        for d in month_ends:
            base_w = calculate_haenam_p_weights(d, primary_strategy_data, price_col=price_col) if primary_is_haenam else calculate_faber_weights(d, all_data, mode='A', price_col=price_col)
            w = base_w
            row = {"월": d.strftime("%Y-%m")}
            for an in whipsaw_asset_keys:
                row[an] = "●" if w.get(an, 0) > 0.01 else "○"
            row["투자자산수"] = sum(1 for an in whipsaw_asset_keys if w.get(an, 0) > 0.01)
            row["현금비중"] = f"{w.get(CASH_NAME, 0)*100:.0f}%"
            whipsaw_records.append(row)
        
        df_whipsaw = pd.DataFrame(whipsaw_records)
        
        # 휩소 횟수 계산
        flip_counts = {}
        for an in whipsaw_asset_keys:
            col_vals = ["●" if r.get(an) == "●" else "○" for r in whipsaw_records]
            flips = sum(1 for i in range(1, len(col_vals)) if col_vals[i] != col_vals[i-1])
            flip_counts[an] = flips
        
        total_months = len(whipsaw_records)
        st.write(f"**총 {total_months}개월 중 자산별 전환(flip) 횟수:**")
        for an, fc in flip_counts.items():
            st.write(f"  {an}: **{fc}회** (평균 {total_months/max(fc,1):.0f}개월에 1번 전환)")

        avg_invested = np.mean([r["투자자산수"] for r in whipsaw_records])
        st.write(f"**평균 투자 자산 수: {avg_invested:.1f}개** / {len(whipsaw_asset_keys)}개")
        
        st.dataframe(df_whipsaw, use_container_width=True, hide_index=True, height=400)
    
def mode_live_and_rebalance(current_dt, current_date, price_col, inv_start_date, init_capital, hist_profit, bt_start_date):
    st.title("MAIN")
    st.subheader("Faber A 실전 & 리밸런싱")
    st.caption("원조 Faber A(코스피/나스닥 패시브, -5%룰)를 기준으로 실전 성과, MDD, 이번 달 참고 성과, 추천 비중을 확인합니다.")
    st.markdown("---")

    live_policy = load_live_portfolio_policy()
    _ensure_account_balance_state()

    st.sidebar.markdown("### 💰 계좌 잔고 입력")
    st.sidebar.warning(
        "사이드바 계좌 잔고 변경분은 입출금인지 수익인지 자동 구분할 수 없습니다. "
        "외부 입출금은 반드시 PERSONAL_CASH_FLOWS에 기록해 주세요."
    )
    if st.sidebar.button("🔄 잔고 기본값으로 초기화"):
        for k, v in BALANCE_DEFAULTS:
            st.session_state[k] = v
        st.session_state["_balance_defaults_version"] = DEFAULT_BALANCE_VERSION
        _set_query_params(
            gen_k=DEFAULT_GEN_KOSPI_BAL, gen_g=DEFAULT_GEN_GOLD_BAL,
            isaa=DEFAULT_ISA_A_BAL, isab=DEFAULT_ISA_B_BAL,
            bal_v=DEFAULT_BALANCE_VERSION,
        )
        st.rerun()

    bal_gen_kospi = st.sidebar.number_input("일반 계좌 (코스피 등)", key="bal_gen_kospi", step=1_000_000)
    st.sidebar.markdown(f"**확인:** {bal_gen_kospi:,.0f}원 (약 {bal_gen_kospi/10000:,.0f}만 원)")
    bal_gen_gold = st.sidebar.number_input("KRX 금현물 계좌", key="bal_gen_gold", step=1_000_000)
    st.sidebar.markdown(f"**확인:** {bal_gen_gold:,.0f}원 (약 {bal_gen_gold/10000:,.0f}만 원)")
    bal_isa_a = st.sidebar.number_input("ISA A 계좌", key="bal_isa_a", step=1_000_000)
    st.sidebar.markdown(f"**확인:** {bal_isa_a:,.0f}원 (약 {bal_isa_a/10000:,.0f}만 원)")
    bal_isa_b = st.sidebar.number_input("ISA B 계좌", key="bal_isa_b", step=1_000_000)
    st.sidebar.markdown(f"**확인:** {bal_isa_b:,.0f}원 (약 {bal_isa_b/10000:,.0f}만 원)")
    try: _set_query_params(gen_k=int(bal_gen_kospi), gen_g=int(bal_gen_gold), isaa=int(bal_isa_a), isab=int(bal_isa_b), bal_v=DEFAULT_BALANCE_VERSION)
    except Exception: pass

    bal_gen = bal_gen_kospi + bal_gen_gold
    manual_total_assets = float(bal_gen + bal_isa_a + bal_isa_b)
    use_live_policy_assets = False
    show_changed_portfolio_snapshot = False
    if live_policy:
        show_changed_portfolio_snapshot = st.sidebar.checkbox(
            "변경 포트폴리오 스냅샷 섹션 표시",
            value=False,
            help="최근 스프레드시트 기반 변경 포트폴리오 표를 보조 섹션으로만 표시합니다.",
        )
        use_live_policy_assets = st.sidebar.checkbox(
            "변경 포트폴리오 스냅샷 총자산 사용",
            value=False,
            help="켜면 config/live_portfolio_policy.json의 현재 총자산을 성과 계산 기준으로 사용합니다.",
        )
    current_total_assets = (
        float(live_policy.get("summary", {}).get("current_total_assets", manual_total_assets))
        if use_live_policy_assets else manual_total_assets
    )
    st.sidebar.markdown("---")
    st.sidebar.metric("총 운용 자산", f"{current_total_assets:,.0f}원")
    if use_live_policy_assets:
        st.sidebar.caption("출처: config/live_portfolio_policy.json")
    show_legacy_haenam_tools = True
    gold_stable_mode = True
    if show_legacy_haenam_tools:
        gold_rt_mode_label = st.sidebar.radio(
            "금 실시간 소스",
            ["안정모드 (권장)", "엄격모드 (0064K0만)"],
            index=0,
            key="gold_rt_mode",
        )
        st.sidebar.caption("안정모드: 0064K0 실패 시 최근 성공값(최대 120분) → GC=F×환율 fallback")
        gold_stable_mode = (gold_rt_mode_label == "안정모드 (권장)")

    if show_changed_portfolio_snapshot and live_policy:
        render_live_portfolio_policy(live_policy)
        render_macro_cycle_monitor(current_date)
        st.markdown("---")
        render_portfolio_operations_dashboard(live_policy, current_date, current_total_assets, price_col)
        st.markdown("---")

    # data_start는 bt_start_date/inv_start_date 중 더 이른 날 - 18개월이므로
    # all_data 단일 로딩으로 역대 MDD 계산까지 커버 가능 (M-1: 이중 호출 제거)
    data_start = min(bt_start_date, inv_start_date) - relativedelta(months=18)
    with st.spinner("📊 데이터를 불러오는 중..."):
        all_data = load_market_data(data_start, current_date, hybrid=True)
        haenam_strategy_data = all_data
        haenam_price_data = all_data
    if haenam_strategy_data is None:
        st.error("Faber A 시장 데이터가 부족해 실전 성과 NAV를 계산할 수 없습니다.")
        return

    # 역대 백테스트 MDD 계산 (Faber A 원조 기준, 위에서 로딩한 데이터 재사용)
    live_backtest_initial_capital = 10_000_000
    with st.spinner("📊 역대 MDD 계산 중 (Faber A 원조/연속모멘텀)..."):
        bt_nav_full = simulate_faber_strategy(
            bt_start_date, current_date, live_backtest_initial_capital, haenam_strategy_data,
            mode='A', buffer_df=None, price_col=price_col
        )
        bt_mom_nav_full = simulate_daily_nav_with_attribution(
            bt_start_date, current_date, live_backtest_initial_capital, haenam_strategy_data, price_col=price_col
        )[0]
        bt_mdd_historical = calculate_performance_metrics(bt_nav_full, live_backtest_initial_capital)[2] if bt_nav_full is not None else None
        bt_monthly_mdd_historical = calculate_monthly_mdd(bt_nav_full) if bt_nav_full is not None else None
        bt_mom_mdd_historical = calculate_performance_metrics(bt_mom_nav_full, live_backtest_initial_capital)[2] if bt_mom_nav_full is not None else None
        bt_mom_monthly_mdd_historical = calculate_monthly_mdd(bt_mom_nav_full) if bt_mom_nav_full is not None else None

    st.subheader("📊 성과 분석")
    st.markdown("#### 💼 나의 투자 성과")
    personal_nav_df = simulate_faber_strategy(
        inv_start_date, current_date, init_capital, haenam_strategy_data,
        mode='A', buffer_df=None, price_col=price_col
    )
    personal_mom_nav_df = simulate_daily_nav_with_attribution(
        inv_start_date, current_date, init_capital, haenam_strategy_data, price_col=price_col
    )[0]
    performance_base_date = personal_nav_df.index[-1] if personal_nav_df is not None and len(personal_nav_df) > 0 else current_date
    cumulative_principal = calculate_cumulative_principal(
        init_capital,
        PERSONAL_CASH_FLOWS,
        performance_base_date,
    )
    performance_base_date_str = pd.Timestamp(performance_base_date).strftime('%Y-%m-%d')
    recorded_principal_gap = current_total_assets - cumulative_principal
    recorded_return_on_principal = recorded_principal_gap / cumulative_principal if cumulative_principal > 0 else 0.0
    p_mdd_daily, p_mdd_monthly, p_peak, p_valley = None, None, None, None
    if personal_nav_df is not None and len(personal_nav_df) > 0:
        _, _, p_mdd_daily, _ = calculate_performance_metrics(personal_nav_df, init_capital)
        p_mdd_monthly = calculate_monthly_mdd(personal_nav_df)
        p_peak, p_valley, _ = find_mdd_period(personal_nav_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재 운용자산", f"{current_total_assets:,.0f}원")
    c2.metric("누적 원금", f"{cumulative_principal:,.0f}원")
    c3.metric("실제 손익", f"{recorded_principal_gap:,.0f}원", delta=f"{recorded_return_on_principal:.2%}")
    c4.metric("MDD (일별)", f"{p_mdd_daily*100:.2f}%" if p_mdd_daily is not None else "N/A")
    c5.metric("MDD (월별)", f"{p_mdd_monthly*100:.2f}%" if p_mdd_monthly is not None else "N/A")
    strategy_compare_rows = []
    for label, strategy_nav in [
        ("Faber A -5%룰", personal_nav_df),
        ("연속모멘텀", personal_mom_nav_df),
    ]:
        if strategy_nav is None or len(strategy_nav) == 0:
            continue
        _, _, strat_mdd_daily, _ = calculate_performance_metrics(strategy_nav, init_capital)
        strat_mdd_monthly = calculate_monthly_mdd(strategy_nav)
        latest_nav = float(strategy_nav["nav"].iloc[-1])
        strategy_compare_rows.append({
            "전략": label,
            "전략 NAV": f"{latest_nav:,.0f}원",
            "누적 원금 대비 전략 손익": f"{latest_nav - cumulative_principal:+,.0f}원",
            "전략 수익률": f"{latest_nav / cumulative_principal - 1:+.2%}" if cumulative_principal > 0 else "-",
            "MDD(일별)": f"{strat_mdd_daily*100:.2f}%" if strat_mdd_daily is not None else "-",
            "MDD(월별)": f"{strat_mdd_monthly*100:.2f}%" if strat_mdd_monthly is not None else "-",
        })
    if strategy_compare_rows:
        st.markdown("##### 전략 기준 비교")
        st.dataframe(pd.DataFrame(strategy_compare_rows), use_container_width=True, hide_index=True)
    st.warning(
        "입출금이 PERSONAL_CASH_FLOWS에 기록되지 않으면 현재 운용자산과 누적 원금의 차이가 수익처럼 보일 수 있습니다. "
        "사이드바 계좌 잔고 변경분은 자동으로 입출금과 투자 성과를 구분할 수 없으므로, 외부 입출금은 반드시 PERSONAL_CASH_FLOWS에 기록해 주세요."
    )
    st.info("실제 성과 판단은 위의 현재 운용자산, 누적 원금, 실제 손익을 기준으로 확인하세요.")
    st.caption(
        "참고: 현재 실제 손익률은 추가투입금까지 포함한 누적 원금 대비로 계산되어, 월간 수익률보다 낮게 보일 수 있습니다."
    )
    if p_peak and p_valley:
        st.info(f"📉 **최대 낙폭 (일별)**: {p_peak.strftime('%Y-%m-%d')}(고점) → {p_valley.strftime('%Y-%m-%d')}(저점)")
    p_m_peak, p_m_valley, _ = find_monthly_mdd_period(personal_nav_df) if personal_nav_df is not None and len(personal_nav_df) > 0 else (None, None, None)
    if p_m_peak and p_m_valley:
        st.info(f"📉 **최대 낙폭 (월별)**: {p_m_peak.strftime('%Y-%m-%d')}(고점) → {p_m_valley.strftime('%Y-%m-%d')}(저점)")
    # 현재 고점 대비 하락률
    if personal_nav_df is not None and len(personal_nav_df) > 0:
        current_dd = float(personal_nav_df["drawdown"].iloc[-1])
        if abs(current_dd) < 1e-8:
            st.success("📊 **현재 상태: 신고점 갱신 중** (고점 대비 하락 없음)")
        else:
            # 현재 고점 날짜 찾기
            peak_val = float(personal_nav_df["running_max"].iloc[-1])
            peak_date_candidates = personal_nav_df[personal_nav_df["nav"] >= peak_val * 0.9999].index
            peak_date_str = peak_date_candidates[-1].strftime('%Y-%m-%d') if len(peak_date_candidates) > 0 else "?"
            # 역대 백테스트 MDD 기준으로 비교
            ref_mdd = bt_mdd_historical if bt_mdd_historical and abs(bt_mdd_historical) > 0.001 else p_mdd_daily
            ref_label = "역대MDD(일별)" if bt_mdd_historical and abs(bt_mdd_historical) > 0.001 else "투자기간MDD(일별)"
            if ref_mdd and abs(ref_mdd) > 0.001:
                st.warning(f"📊 **현재 고점 대비 하락률 (일별): {current_dd*100:.2f}%** | "
                           f"고점: {peak_date_str} → 현재: {performance_base_date_str} | "
                           f"{ref_label}({ref_mdd*100:.2f}%) 대비 {abs(current_dd/ref_mdd)*100:.0f}% 수준")
            else:
                st.warning(f"📊 **현재 고점 대비 하락률 (일별): {current_dd*100:.2f}%** | "
                           f"고점: {peak_date_str} → 현재: {performance_base_date_str}")
            monthly_dd_series = calculate_monthly_drawdown_series(personal_nav_df)
            if monthly_dd_series is not None and not monthly_dd_series.empty:
                current_monthly_dd = float(monthly_dd_series.iloc[-1])
                monthly_nav = personal_nav_df["nav"].groupby(personal_nav_df.index.to_period("M")).last()
                monthly_nav.index = monthly_nav.index.to_timestamp("M")
                monthly_running_max = monthly_nav.cummax()
                current_month = monthly_nav.index[-1]
                current_monthly_peak_val = float(monthly_running_max.iloc[-1])
                monthly_peak_candidates = monthly_nav[monthly_nav >= current_monthly_peak_val * 0.9999]
                monthly_peak_candidates = monthly_peak_candidates[monthly_peak_candidates.index <= current_month]
                monthly_peak_date = monthly_peak_candidates.index[-1] if len(monthly_peak_candidates) > 0 else current_month
                monthly_ref_mdd = (
                    bt_monthly_mdd_historical
                    if bt_monthly_mdd_historical and abs(bt_monthly_mdd_historical) > 0.001
                    else p_mdd_monthly
                )
                monthly_ref_label = (
                    "역대MDD(월별)"
                    if bt_monthly_mdd_historical and abs(bt_monthly_mdd_historical) > 0.001
                    else "투자기간MDD(월별)"
                )
                if monthly_ref_mdd and abs(monthly_ref_mdd) > 0.001:
                    st.warning(f"📊 **현재 고점 대비 하락률 (월별): {current_monthly_dd*100:.2f}%** | "
                               f"고점: {monthly_peak_date.strftime('%Y-%m-%d')} → 현재: {performance_base_date_str} | "
                               f"{monthly_ref_label}({monthly_ref_mdd*100:.2f}%) 대비 {abs(current_monthly_dd/monthly_ref_mdd)*100:.0f}% 수준")
                else:
                    st.warning(f"📊 **현재 고점 대비 하락률 (월별): {current_monthly_dd*100:.2f}%** | "
                               f"고점: {monthly_peak_date.strftime('%Y-%m-%d')} → 현재: {performance_base_date_str}")
    st.caption(
        f"📅 투자 시작일: {inv_start_date.strftime('%Y-%m-%d')} | "
        f"평가 기준일: {performance_base_date_str} | "
        f"누적 원금: {cumulative_principal:,.0f}원"
    )

    # ── 이번 달 공식 성과 + 자산별 참고 성과 ──
    st.markdown("---")
    st.markdown(f"#### 📅 이번 달 성과 ({current_date.strftime('%Y년 %m월')})")
    try:
        # 이번 달 첫 거래일 찾기 (= 지난달 말 리밸런싱 다음날)
        month_start = current_date.replace(day=1)
        # 리밸런싱 기준일: 전월 마지막 거래일
        if personal_nav_df is not None and len(personal_nav_df) > 1:
            prev_month_rows = personal_nav_df[personal_nav_df.index < month_start]
            if len(prev_month_rows) == 0:
                rebal_date = personal_nav_df.index[0]
            else:
                rebal_date = prev_month_rows.index[-1]
            # 리밸런싱일 기준금액은 월말 장부 확정 총자산을 우선 사용한다.
            # 장부에 아직 없는 달은 전략 시뮬레이션 NAV로 fallback한다.
            nav_at_rebal, nav_source, nav_is_confirmed = get_rebalance_basis_nav(rebal_date, personal_nav_df)

            net_month_cash_flow = calculate_period_cash_flow(PERSONAL_CASH_FLOWS, rebal_date, current_date)
            official_month_profit = current_total_assets - nav_at_rebal - net_month_cash_flow
            official_month_return = official_month_profit / nav_at_rebal if nav_at_rebal > 0 else None
            official_profit_label = "0원" if abs(official_month_profit) < 0.5 else f"{official_month_profit:+,.0f}원"
            official_return_label = None
            if official_month_return is not None and abs(official_month_profit) >= 0.5:
                official_return_label = f"{official_month_return:+.2%}"

            st.markdown("##### 공식 성과: 실제 계좌 기준")
            if not nav_is_confirmed:
                st.warning(
                    f"{rebal_date.strftime('%Y-%m')} 월말 확정 원장이 없어 전략 시뮬레이션 NAV로 임시 계산 중입니다. "
                    "월말 원장을 기록하면 이 기준금액이 자동으로 바뀝니다."
                )
            official_cols = st.columns(4)
            official_cols[0].metric("기준 총자산", f"{nav_at_rebal:,.0f}원")
            official_cols[1].metric("순외부현금흐름", f"{net_month_cash_flow:+,.0f}원")
            official_cols[2].metric("현재 운용자산", f"{current_total_assets:,.0f}원")
            official_cols[3].metric(
                "공식 손익",
                official_profit_label,
                delta=official_return_label,
            )
            st.caption(
                f"공식 성과 = 현재 운용자산 - {rebal_date.strftime('%Y-%m-%d')} 기준 총자산 - 순외부현금흐름. "
                f"기준금액 출처: {nav_source}."
            )

            def _month_period_mdd(strategy_nav):
                if strategy_nav is None or strategy_nav.empty:
                    return None
                period = strategy_nav[
                    (strategy_nav.index >= pd.Timestamp(rebal_date))
                    & (strategy_nav.index <= pd.Timestamp(current_date))
                ].copy()
                if len(period) < 2:
                    return None
                running_max = period["nav"].cummax()
                dd = (period["nav"] - running_max) / running_max
                return float(dd.min())

            def _build_monthly_reference_rows(strategy_label, strategy_weights, basis_nav):
                rows = []
                total = 0.0
                for an in [k for k, v in strategy_weights.items() if v >= 0.001]:
                    w = float(strategy_weights.get(an, 0.0))
                    if w < 0.001:
                        continue
                    alloc_won = basis_nav * w
                    if an == CASH_NAME:
                        px_s = get_price_at_date(all_data.get(CASH_NAME), rebal_date, price_col=price_col) or 10000.0
                        px_e = get_price_at_date(all_data.get(CASH_NAME), current_date, price_col=price_col) or px_s
                    else:
                        px_s = get_price_at_date(haenam_price_data.get(an), rebal_date, price_col=price_col)
                        px_e = get_price_at_date(haenam_price_data.get(an), current_date, price_col=price_col)
                    if not px_s or px_s <= 0 or not px_e or px_e <= 0:
                        continue
                    ret = (px_e / px_s) - 1.0
                    pnl = alloc_won * ret
                    total += pnl
                    rows.append({
                        "전략": strategy_label,
                        "자산": an,
                        "_raw": an,
                        "비중": f"{w*100:.1f}%",
                        "배분금액": f"{alloc_won:,.0f}원",
                        "기준가(리밸)": f"{px_s:,.2f}",
                        "현재가": f"{px_e:,.2f}",
                        "수익률": ret * 100,
                        "손익(원)": pnl,
                    })
                return rows, total

            def _cat_rank(raw):
                if raw == KR_STOCK_MIX_ASSET: return 0
                if raw == NASDAQ100_ASSET_NAME: return 1
                if raw == '미국채30년': return 2
                if raw == '한국채30년': return 3
                if raw == '금현물': return 4
                if raw == CASH_NAME: return 6
                return 5

            strategy_refs = []
            faber_weights = calculate_faber_weights(rebal_date, haenam_strategy_data, mode='A', price_col=price_col)
            momentum_weights = calculate_weights_at_date(rebal_date, haenam_strategy_data, price_col=price_col)
            for label, strategy_nav, weights, bt_daily, bt_monthly in [
                ("Faber A -5%룰", personal_nav_df, faber_weights, bt_mdd_historical, bt_monthly_mdd_historical),
                ("연속모멘텀", personal_mom_nav_df, momentum_weights, bt_mom_mdd_historical, bt_mom_monthly_mdd_historical),
            ]:
                if strategy_nav is None or len(strategy_nav) == 0:
                    continue
                basis_nav, basis_source, _ = get_rebalance_basis_nav(rebal_date, strategy_nav)
                strategy_flow = calculate_period_cash_flow(PERSONAL_CASH_FLOWS, rebal_date, current_date)
                basis_profit = current_total_assets - basis_nav - strategy_flow
                basis_return = basis_profit / basis_nav if basis_nav > 0 else None
                month_mdd = _month_period_mdd(strategy_nav)
                rows, total_pnl = _build_monthly_reference_rows(label, weights, basis_nav)
                strategy_refs.append({
                    "label": label,
                    "basis_nav": basis_nav,
                    "basis_source": basis_source,
                    "basis_profit": basis_profit,
                    "basis_return": basis_return,
                    "month_mdd": month_mdd,
                    "bt_daily_mdd": bt_daily,
                    "bt_monthly_mdd": bt_monthly,
                    "rows": sorted(rows, key=lambda r: _cat_rank(r["_raw"])),
                    "total_pnl": total_pnl,
                })

            if strategy_refs:
                st.markdown("##### 전략별 이번 달 기준 손익/MDD")
                compare_df = pd.DataFrame([{
                    "전략": ref["label"],
                    "기준 총자산": f"{ref['basis_nav']:,.0f}원",
                    "기준금액 출처": ref["basis_source"],
                    "기준 손익": f"{ref['basis_profit']:+,.0f}원",
                    "기준 수익률": f"{ref['basis_return']:+.2%}" if ref["basis_return"] is not None else "-",
                    "이번달 MDD": f"{ref['month_mdd']*100:.2f}%" if ref["month_mdd"] is not None else "-",
                    "역대 MDD(일별)": f"{ref['bt_daily_mdd']*100:.2f}%" if ref["bt_daily_mdd"] is not None else "-",
                    "역대 MDD(월별)": f"{ref['bt_monthly_mdd']*100:.2f}%" if ref["bt_monthly_mdd"] is not None else "-",
                } for ref in strategy_refs])
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

                st.markdown("##### 참고: 자산별 가격변동 추정")
                tabs = st.tabs([ref["label"] for ref in strategy_refs])
                for tab, ref in zip(tabs, strategy_refs):
                    with tab:
                        current_rows = ref["rows"]
                        if not current_rows:
                            st.info("표시할 자산별 참고 성과가 없습니다.")
                            continue
                        cols_m = st.columns(len(current_rows) + 1)
                        for i, row in enumerate(current_rows):
                            cols_m[i].metric(
                                label=row["자산"],
                                value=f"{row['수익률']:+.2f}%",
                                delta=f"{row['손익(원)']:+,.0f}원",
                            )
                        cols_m[-1].metric(
                            label="📊 참고 합계",
                            value=f"{ref['total_pnl']:+,.0f}원",
                            delta=f"{ref['total_pnl']/ref['basis_nav']*100:+.2f}%" if ref["basis_nav"] > 0 else "N/A",
                        )
                        with st.expander(f"📋 {ref['label']} 이번 달 자산별 상세"):
                            detail_df = pd.DataFrame([{
                                "자산": r["자산"],
                                "비중": r["비중"],
                                "배분금액": r["배분금액"],
                                "기준가(리밸)": r["기준가(리밸)"],
                                "현재가": r["현재가"],
                                "수익률": f"{r['수익률']:+.2f}%",
                                "손익(원)": f"{r['손익(원)']:+,.0f}원",
                            } for r in current_rows])
                            st.dataframe(detail_df, use_container_width=True, hide_index=True)
                        st.caption(
                            f"※ 기준: {rebal_date.strftime('%Y-%m-%d')} 리밸런싱 당시 {ref['label']} NAV "
                            f"{ref['basis_nav']:,.0f}원 기준({ref['basis_source']}). 자산별 가격변동으로 추정한 값이며 실제와 차이 있을 수 있음."
                        )
    except Exception as e:
        st.warning(f"이번 달 성과 계산 오류: {e}")

    st.markdown("---")
    st.info(f"📅 기준일: {current_dt.strftime('%Y년 %m월 %d일 %H시 %M분')}")
    gold_rt = resolve_gold_signal_runtime(current_dt, stable_mode=gold_stable_mode, sticky_minutes=120)
    gold_source = gold_rt["source"]
    rt_kodex = gold_rt["kodex"]
    rt_kodex_px = gold_rt["price"] if gold_source in ("KODEX_REALTIME", "KODEX_STICKY") else None
    rt_gold_krw = gold_rt["price"] if gold_source == "GC_FX_REALTIME" else None
    rt_gc = gold_rt["gc"]
    rt_fx = gold_rt["fx"]

    if gold_source == "KODEX_REALTIME":
        traded_at = ((rt_kodex or {}).get("traded_at") or "").replace("T", " ")[:16]
        traded_txt = f" | 체결시각: {traded_at}" if traded_at else ""
        st.caption(
            f"**Faber A 원조 룰**: 코스피200·미국나스닥100·한국채30년·미국채30년·금현물은 "
            f"12개월 고점 대비 -5% 이내면 각 20% ON, 아니면 현금(MMF) 대기. 코스피/나스닥은 패시브 ETF 기준입니다. "
            f"금현물은 KODEX 금액티브(0064K0) 실시간 기준. "
            f"(현재가: ₩{rt_kodex_px:,.0f}{traded_txt})"
        )
    elif gold_source == "KODEX_STICKY":
        age_min = gold_rt.get("sticky_age_min")
        age_txt = f"{int(age_min):d}분 전 값" if age_min is not None else "최근 성공값"
        st.caption(
            f"**Faber A 원조 룰**: 코스피200·미국나스닥100·한국채30년·미국채30년·금현물은 "
            f"12개월 고점 대비 -5% 이내면 각 20% ON, 아니면 현금(MMF) 대기. 코스피/나스닥은 패시브 ETF 기준입니다. "
            f"금현물은 0064K0 최근 성공값({age_txt}) 기준. "
            f"(현재가: ₩{rt_kodex_px:,.0f})"
        )
    elif gold_source == "GC_FX_REALTIME":
        st.caption(
            f"**Faber A 원조 룰**: 코스피200·미국나스닥100·한국채30년·미국채30년·금현물은 "
            f"12개월 고점 대비 -5% 이내면 각 20% ON, 아니면 현금(MMF) 대기. 코스피/나스닥은 패시브 ETF 기준입니다. "
            f"금현물은 GC=F 실시간 보정(GLD 스케일 환산) 기준(0064K0 fallback). "
            f"(GLD 환산가: ${rt_gc:,.2f} | USD/KRW: ₩{rt_fx:,.0f} | 원화: ₩{rt_gold_krw:,.0f})"
        )
    elif gold_stable_mode:
        st.caption("**Faber A 원조 룰**: 5개 패시브 슬롯은 12개월 고점 대비 -5% 이내면 각 20% ON, 아니면 현금(MMF) 대기. 금현물은 0064K0 종가 기준 (실시간 로딩 실패).")
    else:
        st.caption("**Faber A 원조 룰**: 5개 패시브 슬롯은 12개월 고점 대비 -5% 이내면 각 20% ON, 아니면 현금(MMF) 대기. 엄격모드(0064K0만)에서 실시간을 못 받아 종가 기준으로 계산합니다.")
    col_rt1, col_rt2 = st.columns([1, 4])
    with col_rt1:
        if st.button("🔄 신호표 새로고침", help="Faber A 신호 및 추천 비중 섹션만 새로 계산"):
            # 전체 데이터 캐시는 유지하고, 신호표 계산에 필요한 실시간 소스만 갱신
            get_realtime_kodex_gold_active.clear()
            get_realtime_gold_krw.clear()
            st.rerun()
    st.subheader("📋 Faber A 원조 신호 및 추천 비중")
    results = []
    for asset_name, ticker in ASSETS.items():
        price_data = all_data.get(asset_name)
        mom_data = all_data.get(f"{asset_name}_모멘텀")
        if ticker == '411060':
            mom_data = harmonize_gold_momentum_scale(all_data, current_date, rt_kodex_px, price_col=price_col)
        curr_price = get_price_at_date(price_data, current_date, price_col=price_col)
        _, score, positive_months, valid_months = calculate_momentum_score_detail_at_date(
            ticker, current_date, mom_data, price_col=price_col
        )
        signal_data = mom_data if ticker == '411060' else price_data
        near_high = is_near_12month_high(signal_data, current_date, threshold=0.05, price_col=price_col)
        high_12m = None
        if signal_data is not None and not signal_data.empty:
            col = price_col if price_col in signal_data.columns else "Close"
            prices_list = []
            sp = get_price_at_date(signal_data, current_date, price_col=col)
            if sp is not None: prices_list.append(sp)
            for m in range(1, 12):
                me = get_month_end_date(current_date - relativedelta(months=m))
                p = get_price_at_date(signal_data, me, price_col=col)
                if p is not None: prices_list.append(p)
            if prices_list: high_12m = max(prices_list)
        signal_px = get_price_at_date(signal_data, current_date, price_col=price_col) if signal_data is not None else curr_price

        # 금현물: 1순위 0064K0 실시간, 2순위 GC=F×환율 실시간 fallback
        if ticker == '411060' and rt_kodex_px:
            signal_px = rt_kodex_px
            # 오늘 가격을 실시간으로 교체하여 12M 고점 재계산
            if signal_data is not None and not signal_data.empty:
                _col = price_col if price_col in signal_data.columns else "Close"
                rt_prices = [rt_kodex_px]
                for m in range(1, 12):
                    me = get_month_end_date(current_date - relativedelta(months=m))
                    p = get_price_at_date(signal_data, me, price_col=_col)
                    if p is not None: rt_prices.append(p)
                if rt_prices: high_12m = max(rt_prices)
            near_high = (signal_px / high_12m - 1) >= -0.05 if high_12m and high_12m > 0 else near_high
        elif ticker == '411060' and rt_gold_krw:
            signal_px = rt_gold_krw
            # 오늘 가격을 실시간으로 교체하여 12M 고점 재계산
            if signal_data is not None and not signal_data.empty:
                _col = price_col if price_col in signal_data.columns else "Close"
                rt_prices = [rt_gold_krw]
                for m in range(1, 12):
                    me = get_month_end_date(current_date - relativedelta(months=m))
                    p = get_price_at_date(signal_data, me, price_col=_col)
                    if p is not None: rt_prices.append(p)
                if rt_prices: high_12m = max(rt_prices)
            near_high = (signal_px / high_12m - 1) >= -0.05 if high_12m and high_12m > 0 else near_high

        dist_from_high = ((signal_px / high_12m) - 1) if signal_px and high_12m and high_12m > 0 else None
        signal_weight = 0.20 if near_high else 0.0
        display_price = signal_px if ticker == '411060' else curr_price
        results.append({
            "자산명": asset_name, "티커": ("0064K0" if ticker == '411060' else ticker), "현재가": display_price,
            "12M고점": high_12m, "고점대비": dist_from_high,
            "모멘텀": score,
            "모멘텀상승개월": positive_months,
            "모멘텀유효개월": valid_months,
            "기준신호": f"● {signal_weight:.0%}" if signal_weight > 0 else "○ 0%",
            "추천비중": signal_weight,
            "_is_gold": ticker == '411060'
        })
    df_results = pd.DataFrame(build_haenam_signal_display_rows(results))
    df_rebalance_results = pd.DataFrame(
        expand_haenam_signal_rows(
            results, current_date, haenam_price_data, price_col=price_col, kr_weights={},
            nasdaq_active=False
        )
    )
    cash_weight = max(0.0, 1.0 - float(df_results["추천비중"].sum()))
    cash_price = get_price_at_date(all_data.get(CASH_NAME), current_date, price_col=price_col) or 10000.0
    cash_row = {
        "신호자산": CASH_NAME, "자산명": CASH_NAME, "티커": CASH_TICKER, "현재가": cash_price,
        "12M고점": None, "고점대비": None, "모멘텀": None,
        "모멘텀상승개월": None, "모멘텀유효개월": None,
        "기준신호": "-", "추천비중": cash_weight, "_is_gold": False
    }
    df_results = pd.concat([df_results, pd.DataFrame([cash_row])], ignore_index=True)
    df_rebalance_results = pd.concat([df_rebalance_results, pd.DataFrame([cash_row])], ignore_index=True)
    df_results_orig = df_rebalance_results.copy()  # 리밸런싱용
    df_display = df_rebalance_results.copy()
    # 금현물 표시명 변경
    if gold_source == "KODEX_REALTIME":
        gold_display_name = "금현물 (0064K0🔴실시간)"
    elif gold_source == "KODEX_STICKY":
        gold_display_name = "금현물 (0064K0🟡지연값)"
    elif gold_source == "GC_FX_REALTIME":
        gold_display_name = "금현물 (GC=F×환율🔴실시간)"
    else:
        gold_display_name = "금현물 (0064K0 종가기준)"
    df_display.loc[df_display["_is_gold"] == True, "자산명"] = gold_display_name
    df_display = df_display.drop(columns=["_is_gold"])
    df_display["현재가"] = df_display["현재가"].apply(lambda x: f"{x:,.0f}원" if pd.notna(x) else "-")
    df_display["12M고점"] = df_display["12M고점"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
    df_display["고점대비"] = df_display["고점대비"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df_display["모멘텀"] = df_display.apply(
        lambda row: format_momentum_score(
            row["모멘텀"], row.get("모멘텀상승개월"), row.get("모멘텀유효개월")
        ),
        axis=1,
    )
    df_display["추천비중"] = df_display["추천비중"].apply(lambda x: f"{x*100:.0f}%")
    df_display = df_display[["신호자산", "자산명", "티커", "현재가", "12M고점", "고점대비", "모멘텀", "기준신호", "추천비중"]]
    df_display.columns = ["신호자산", "집행자산", "티커", "현재가", "12M고점", "고점대비", "모멘텀", "신호", "추천비중"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # 금현물 참고: GLD * USD/KRW
    with st.expander("🥇 금현물 참고 데이터 (GLD 전일종가 × USD/KRW 실시간)"):
        try:
            gld_raw = fdr.DataReader('GLD', current_date - relativedelta(months=18), current_date)
            fx_raw = fdr.DataReader('USD/KRW', current_date - relativedelta(months=18), current_date)
            if gld_raw is not None and fx_raw is not None and not gld_raw.empty and not fx_raw.empty:
                gld_raw = gld_raw[~gld_raw.index.duplicated(keep='last')]
                fx_raw = fx_raw[~fx_raw.index.duplicated(keep='last')]
                gld_price = float(gld_raw['Close'].iloc[-1])
                fx_price = float(fx_raw['Close'].iloc[-1])
                gld_krw = gld_price * fx_price
                gld_date = gld_raw.index[-1].strftime('%Y-%m-%d')
                fx_date = fx_raw.index[-1].strftime('%Y-%m-%d')
                c1g, c2g, c3g = st.columns(3)
                c1g.metric("GLD (USD)", f"${gld_price:,.2f}", help=f"기준일: {gld_date}")
                c2g.metric("USD/KRW", f"₩{fx_price:,.0f}", help=f"기준일: {fx_date}")
                c3g.metric("GLD×환율 (원화)", f"₩{gld_krw:,.0f}", help="GLD × USD/KRW")
                gld_adj = gld_raw['Adj Close'] if 'Adj Close' in gld_raw.columns else gld_raw['Close']
                gld_fx = pd.concat([gld_adj, fx_raw['Close']], axis=1, keys=['G', 'F']).ffill().dropna()
                gld_fx['KRW'] = gld_fx['G'] * gld_fx['F']
                krw_series = gld_fx['KRW']

                # 메인 Faber와 동일 기준: 현재 + 과거 11개 월말(총 12포인트)
                current_ref = krw_series.asof(current_date)
                prices_12m = [float(current_ref)] if pd.notna(current_ref) else []
                for m in range(1, 12):
                    me = get_month_end_date(current_date - relativedelta(months=m))
                    p = krw_series.asof(me)
                    if pd.notna(p):
                        prices_12m.append(float(p))

                if len(prices_12m) >= 2:
                    gld_high = max(prices_12m)
                    gld_dist = (prices_12m[0] / gld_high - 1) * 100
                    gld_near = gld_dist >= -5.0
                    st.info(f"📊 GLD×환율 12M고점: ₩{gld_high:,.0f} | 고점대비: {gld_dist:.1f}% | "
                            f"Faber 판단: {'● 투자' if gld_near else '○ 현금'}")
                    st.caption("💡 위 Faber 신호는 0064K0(우선) 기준이고, 이 참고 데이터는 GLD×환율 비교용입니다.")
        except Exception as e:
            st.warning(f"GLD 데이터 오류: {e}")

    st.markdown("---")
    st.subheader("🏦 3계좌 절세 최적화 리밸런싱")
    st.info("👇 우선순위 배치: 금=금계좌 고정 / 일반=코스피200 우선 / ISA_A=채권 우선 / ISA_B=미국나스닥100 패시브 우선")
    if st.button("🚀 리밸런싱 목표 계산하기", type="primary"):
        with st.spinner("계산 중..."):
            final_df = optimize_allocation(
                df_results_orig[["자산명","추천비중"]].copy(),
                bal_gen_kospi,
                bal_gen_gold,
                bal_isa_a,
                bal_isa_b,
            )
            st.success("✅ 계산 완료!")
            disp = final_df.copy()
            disp["추천비중"] = disp["추천비중"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
            for c in ["총목표금액"] + ACCOUNT_COLUMNS: disp[c] = disp[c].apply(lambda x: f"{x:,.0f}")
            st.dataframe(disp.style
                .map(lambda x: "background-color: #e6f3ff" if x != "0" else "", subset=["ISA_A","ISA_B"])
                .map(lambda x: "background-color: #fff5e6" if x != "0" else "", subset=["일반계좌"])
                .map(lambda x: "background-color: #fff8d6" if x != "0" else "", subset=["금계좌"]),
                use_container_width=True, height=400)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                pd.DataFrame([{"항목":"기준일","값":current_dt.strftime("%Y-%m-%d %H:%M:%S")},
                    {"항목":"총금액","값":current_total_assets},{"항목":"일반(코스피)","값":bal_gen_kospi},
                    {"항목":"KRX금","값":bal_gen_gold},{"항목":"일반합계","값":bal_gen},
                    {"항목":"ISA_A","값":bal_isa_a},{"항목":"ISA_B","값":bal_isa_b}]).to_excel(writer, sheet_name="Summary", index=False)
                t_df = final_df.copy(); t_df["추천비중(%)"] = (t_df["추천비중"]*100).round(2); t_df.drop(columns=["추천비중"]).to_excel(writer, sheet_name="Targets", index=False)
            st.download_button("📥 엑셀 다운로드", output.getvalue(), f"리밸런싱_{current_dt.strftime('%Y%m%d_%H%M')}.xlsx")

    st.markdown("---")
    st.subheader("💾 결과 다운로드")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ex = df_results_orig.drop(columns=["_is_gold"], errors="ignore").copy()
        ex["모멘텀"] = ex.apply(
            lambda row: format_momentum_score(
                row["모멘텀"], row.get("모멘텀상승개월"), row.get("모멘텀유효개월")
            ),
            axis=1,
        )
        ex["고점대비"] = ex["고점대비"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
        ex["추천비중"] = ex["추천비중"]*100
        ex = ex[["신호자산", "자산명", "티커", "현재가", "12M고점", "고점대비", "모멘텀", "기준신호", "추천비중"]]
        ex.columns = ["신호자산","집행자산","티커","현재가","12M고점","고점대비","모멘텀","신호","추천비중(%)"]
        ex.to_excel(writer, sheet_name="FaberA_리밸런싱", index=False)
    st.download_button("📥 엑셀 파일 다운로드", output.getvalue(), f"FaberA_리밸런싱_{current_dt.strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


# ==============================
# 11. main
# ==============================
def mode_monte_carlo(current_dt, current_date, price_col, bt_start_date, init_capital):
    st.title("🎲 몬테카를로 시뮬레이션")
    st.caption("Faber A 실제 월별 수익률을 부트스트랩하여 미래 경로를 시뮬레이션합니다.")
    
    # 데이터 로딩 + Faber A 시뮬레이션
    data_start = bt_start_date - relativedelta(months=18)
    with st.spinner("📊 Faber A 백테스트 실행 중..."):
        all_data = load_market_data(data_start, current_date, hybrid=True)
    
    IC = 10_000_000
    nav_df = simulate_faber_strategy(bt_start_date, current_date, IC, all_data,
        mode='A', buffer_df=None, price_col=price_col)
    
    if nav_df is None or len(nav_df) < 2:
        st.error("백테스트 데이터 부족")
        return
    actual_start = nav_df.index.min()
    if actual_start > bt_start_date + timedelta(days=7):
        st.error(
            f"요청 시작일({bt_start_date.strftime('%Y-%m-%d')}) 대비 "
            f"실제 계산 시작일({actual_start.strftime('%Y-%m-%d')})이 늦습니다. "
            "이 상태에서는 몬테카를로 입력 분포 신뢰도가 낮습니다."
        )
        st.stop()
    st.caption(
        f"✅ 실제 계산 시작일: {actual_start.strftime('%Y-%m-%d')} "
        f"(요청 시작일: {bt_start_date.strftime('%Y-%m-%d')})"
    )
    
    # 실제 월별 수익률 추출
    monthly_nav = nav_df['nav'].resample('ME').last().dropna()
    monthly_returns = monthly_nav.pct_change().dropna().values
    
    st.markdown("---")
    st.subheader("📋 Faber A 실제 월별 수익률 분포")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("데이터 수", f"{len(monthly_returns)}개월")
    c2.metric("월 평균", f"{np.mean(monthly_returns)*100:.2f}%")
    c3.metric("월 표준편차", f"{np.std(monthly_returns)*100:.2f}%")
    c4.metric("최악의 달", f"{np.min(monthly_returns)*100:.2f}%")
    c5.metric("최고의 달", f"{np.max(monthly_returns)*100:.2f}%")
    
    # 파라미터
    st.markdown("---")
    st.subheader("⚙️ 시뮬레이션 설정")
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_capital = st.number_input("초기 자산 (원)", value=init_capital, step=10_000_000)
    with col2:
        sim_years = st.selectbox("시뮬레이션 기간", [2, 3, 5, 7, 10], index=1)
    with col3:
        n_sims = st.selectbox("경로 수", [1000, 5000, 10000], index=2)
    
    sim_months = sim_years * 12
    
    if st.button("🎲 시뮬레이션 실행", type="primary"):
        with st.spinner(f"🎲 {n_sims:,}개 경로 시뮬레이션 중..."):
            np.random.seed(None)  # 매번 다른 결과
            
            final_values = []
            path_matrix = []
            max_drawdowns = []
            
            for i in range(n_sims):
                sampled = np.random.choice(monthly_returns, size=sim_months, replace=True)
                path = [sim_capital]
                peak = sim_capital
                max_dd = 0
                for r in sampled:
                    new_val = path[-1] * (1 + r)
                    path.append(new_val)
                    if new_val > peak: peak = new_val
                    dd = (new_val / peak) - 1
                    if dd < max_dd: max_dd = dd
                
                final_values.append(path[-1])
                max_drawdowns.append(max_dd)
                path_matrix.append(path)
            
            final_values = np.array(final_values)
            max_drawdowns = np.array(max_drawdowns)
            paths_arr = np.asarray(path_matrix, dtype=float)
            
            # 퍼센타일 경로
            percentile_paths = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                percentile_paths[p] = np.percentile(paths_arr, p, axis=0).tolist()
        
        # 결과 표시
        st.markdown("---")
        st.subheader(f"📊 {sim_years}년 후 자산 분포 ({n_sims:,}개 경로)")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("하위 5%", f"{np.percentile(final_values, 5)/100_000_000:.2f}억",
                  delta=f"{(np.percentile(final_values, 5)/sim_capital - 1)*100:.1f}%")
        c2.metric("하위 10%", f"{np.percentile(final_values, 10)/100_000_000:.2f}억",
                  delta=f"{(np.percentile(final_values, 10)/sim_capital - 1)*100:.1f}%")
        c3.metric("중간값", f"{np.percentile(final_values, 50)/100_000_000:.2f}억",
                  delta=f"{(np.percentile(final_values, 50)/sim_capital - 1)*100:.1f}%")
        c4.metric("상위 10%", f"{np.percentile(final_values, 90)/100_000_000:.2f}억",
                  delta=f"{(np.percentile(final_values, 90)/sim_capital - 1)*100:.1f}%")
        c5.metric("상위 5%", f"{np.percentile(final_values, 95)/100_000_000:.2f}억",
                  delta=f"{(np.percentile(final_values, 95)/sim_capital - 1)*100:.1f}%")
        
        # 확률 테이블
        st.markdown("---")
        st.subheader("🎯 핵심 확률")
        targets = [
            (sim_capital, "원금 이상", "원금 손실 확률"),
            (sim_capital * 1.2, "1.2배 이상", None),
            (sim_capital * 1.5, "1.5배 이상", None),
            (sim_capital * 2.0, "2.0배 이상", None),
            (sim_capital * 3.0, "3.0배 이상", None),
        ]
        
        prob_data = []
        for target, label, alt_label in targets:
            prob = (final_values >= target).mean() * 100
            if alt_label:
                prob_data.append({"목표": alt_label, "확률": f"{100-prob:.1f}%"})
            else:
                prob_data.append({"목표": f"{sim_years}년 후 {label}", "확률": f"{prob:.1f}%"})
        st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)
        
        # MDD 분포
        st.markdown("---")
        st.subheader("📉 경로 중 최대 낙폭(MDD) 분포")
        c1, c2, c3 = st.columns(3)
        c1.metric("MDD 중간값", f"{np.percentile(max_drawdowns, 50)*100:.1f}%")
        c2.metric("MDD 하위 10% (심한 경우)", f"{np.percentile(max_drawdowns, 10)*100:.1f}%")
        c3.metric("MDD 최악", f"{np.min(max_drawdowns)*100:.1f}%")
        
        # 팬 차트
        st.markdown("---")
        st.subheader("📈 경로 팬 차트")
        
        months_labels = list(range(sim_months + 1))
        fig_fan = go.Figure()
        
        # 밴드: 5-95, 10-90, 25-75
        bands = [
            (5, 95, "5%~95% 범위", "rgba(248,113,113,0.1)"),
            (10, 90, "10%~90% 범위", "rgba(250,204,21,0.1)"),
            (25, 75, "25%~75% 범위", "rgba(96,165,250,0.15)"),
        ]
        for lo, hi, name, color in bands:
            fig_fan.add_trace(go.Scatter(x=months_labels, y=[v/100_000_000 for v in percentile_paths[hi]],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_fan.add_trace(go.Scatter(x=months_labels, y=[v/100_000_000 for v in percentile_paths[lo]],
                mode='lines', line=dict(width=0), fill='tonexty', fillcolor=color, name=name))
        
        # 중간값 라인
        fig_fan.add_trace(go.Scatter(x=months_labels, y=[v/100_000_000 for v in percentile_paths[50]],
            mode='lines', name='중간값 (50%)', line=dict(color='white', width=2.5)))
        
        # 원금 라인
        fig_fan.add_trace(go.Scatter(x=months_labels, y=[sim_capital/100_000_000]*len(months_labels),
            mode='lines', name='원금', line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash')))
        
        fig_fan.update_layout(
            title=f"Faber A {sim_years}년 경로 ({n_sims:,}개 시뮬레이션)",
            xaxis_title="개월", yaxis_title="자산 (억원)",
            height=500, hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#888')
        )
        st.plotly_chart(fig_fan, use_container_width=True)
        
        # 탈레브의 경고
        worst = np.min(final_values)
        worst_pct = (worst / sim_capital - 1) * 100
        st.warning(f"⚠️ **탈레브의 경고** — {n_sims:,}개 경로 중 최악: "
                   f"{worst/100_000_000:.2f}억 ({worst_pct:.1f}%). "
                   f"그리고 이 시뮬레이션에 포함되지 않은 것: 2008년급 금융위기, 하이퍼인플레이션, "
                   f"전쟁. **역사에 없었던 일은 몬테카를로에도 없습니다.**")
        
        st.info(f"💡 **현실적 기대치** — 가장 가능성 높은 시나리오(중간값): "
                f"{sim_years}년 후 약 {np.percentile(final_values, 50)/100_000_000:.1f}억. "
                f"운이 나빠도(하위 10%): {np.percentile(final_values, 10)/100_000_000:.1f}억. "
                f"**\"잘 되면 얼마 버나\"가 아니라 \"안 되어도 안 죽는다\"가 이 전략의 본질입니다.**")


def mode_buy_hold_sandbox(current_dt):
    st.title("Buy & Hold")
    st.caption(
        "기존 전략을 바꾸지 않는 검토용 계산기입니다. "
        "목표비중, 계좌금액, 현재 보유금액을 입력하면 실전 리밸런싱과 같은 계좌 우선순위로 목표 배치를 계산합니다."
    )
    st.markdown("---")

    st.markdown(f"**기준시각:** {current_dt.strftime('%Y년 %m월 %d일 %H:%M:%S')}")

    st.markdown("#### 계좌금액")
    balance_input_cols = st.columns(3)
    account_balances = {
        "일반계좌": balance_input_cols[0].number_input(
            "일반계좌",
            min_value=0,
            value=int(_get_account_balance_value("bal_gen_kospi", DEFAULT_GEN_KOSPI_BAL)),
            step=1_000_000,
            key="buy_hold_balance_general",
        ),
        "ISA_A": balance_input_cols[1].number_input(
            "ISA A",
            min_value=0,
            value=int(_get_account_balance_value("bal_isa_a", DEFAULT_ISA_A_BAL)),
            step=1_000_000,
            key="buy_hold_balance_isa_a",
        ),
        "ISA_B": balance_input_cols[2].number_input(
            "ISA B",
            min_value=0,
            value=int(_get_account_balance_value("bal_isa_b", DEFAULT_ISA_B_BAL)),
            step=1_000_000,
            key="buy_hold_balance_isa_b",
        ),
    }
    total_assets = sum(float(v) for v in account_balances.values())
    st.metric("계산 대상 합계", f"{total_assets:,.0f}원")

    st.markdown("#### 목표 비중")
    st.caption("코스피와 나스닥100은 각각 TIME/KoAct 액티브 50:50 집행자산으로 펼쳐서 배치합니다.")
    weight_cols = st.columns(4)
    target_weights = {}
    for idx, asset in enumerate(BUY_HOLD_ASSET_BUCKETS):
        default_pct = BUY_HOLD_BASELINE_WEIGHTS[asset] * 100
        target_weights[asset] = (
            weight_cols[idx].number_input(
                asset,
                min_value=0.0,
                max_value=100.0,
                value=float(default_pct),
                step=1.0,
                key=f"buy_hold_target_{asset}",
            )
            / 100.0
        )

    raw_total_pct = sum(target_weights.values()) * 100
    if abs(raw_total_pct - 100.0) > 0.01:
        st.warning(f"목표 비중 합계가 {raw_total_pct:.1f}%입니다. 계산표는 합계 100%로 정규화해 표시합니다.")

    normalized_targets, _ = normalize_buy_hold_weights(target_weights)
    default_current_amounts = {
        asset: total_assets * weight
        for asset, weight in expand_buy_hold_weights_to_execution_assets(BUY_HOLD_BASELINE_WEIGHTS).items()
    }
    st.markdown("#### 현재 보유금액")
    st.caption("실제 현재 비중을 계산하려면 각 집행자산의 평가금액을 입력하세요.")
    current_amounts = {}
    current_input_cols = st.columns(3)
    for idx, asset in enumerate(expand_buy_hold_weights_to_execution_assets(normalized_targets).keys()):
        current_amounts[asset] = current_input_cols[idx % 3].number_input(
            asset,
            min_value=0,
            value=int(round(default_current_amounts.get(asset, 0.0))),
            step=1_000_000,
            key=f"buy_hold_current_amount_{idx}",
        )
    current_total_assets = sum(float(v) for v in current_amounts.values())
    current_gap = current_total_assets - total_assets
    if abs(current_gap) > 0.5:
        st.warning(
            f"현재 보유금액 합계({current_total_assets:,.0f}원)와 계좌금액 합계({total_assets:,.0f}원)가 "
            f"{current_gap:+,.0f}원 차이납니다. 누락된 현금/자산이 있으면 현재 보유금액에 반영하세요."
        )

    result = calculate_buy_hold_allocation(account_balances, target_weights, current_amounts=current_amounts)
    table = result["table"].copy()
    display_table = table.rename(columns=BUY_HOLD_ACCOUNT_DISPLAY)
    for col in ["현재비중", "목표비중"]:
        display_table[col] = display_table[col].map(lambda v: f"{float(v) * 100:.1f}%")
    for col in ["현재금액", "목표금액", "추가매수_매도", "일반계좌", "ISA A", "ISA B"]:
        display_table[col] = display_table[col].map(lambda v: f"{float(v):+,.0f}원" if col == "추가매수_매도" else f"{float(v):,.0f}원")

    st.markdown("#### 계좌별 목표 배치 및 추가매수/매도")
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    st.info("계좌 배치는 실전 3계좌 절세 최적화 리밸런싱과 같은 우선순위입니다. 추가매수/매도는 목표금액에서 입력한 현재금액을 뺀 값입니다.")
    st.caption("이 화면은 계산 보조 도구이며 투자 권유나 자동 주문 기능이 아닙니다.")


def main():
    KST = pytz.timezone('Asia/Seoul')
    current_dt = datetime.now(KST).replace(tzinfo=None)
    current_date = normalize_to_date(current_dt)

    st.sidebar.title("⚙️ 메뉴 및 설정")
    if st.sidebar.button("🔄 최신 데이터 새로고침"):
        st.cache_data.clear()
        st.sidebar.success("캐시 초기화 완료")
    st.sidebar.markdown("---")

    with st.sidebar.expander("🛠 기본 투자 정보 설정", expanded=False):
        inv_start_date = datetime.combine(
            st.date_input("투자 시작일", DEFAULT_INVESTMENT_START_DATE),
            datetime.min.time(),
        )
        init_capital = st.number_input("초기 투자 원금", value=DEFAULT_INITIAL_CAPITAL, step=1_000_000)
        st.markdown(f"**확인:** {init_capital:,.0f}원")
        hist_profit = st.number_input("과거 누적 실현손익", value=DEFAULT_HISTORICAL_REALIZED_PROFIT, step=100_000)
        st.markdown(f"**확인:** {hist_profit:,.0f}원")
        bt_start_date = datetime.combine(
            st.date_input(
                "백테스트 시작일",
                DEFAULT_BACKTEST_START_DATE,
                help="하이브리드 모드: 2000-01-01~. FRED/ECOS 딥프록시 → 프록시 → 실제ETF 3계층 체인링크.",
            ),
            datetime.min.time(),
        )
        auto_bt_end = st.checkbox(
            "백테스트 종료일을 오늘로 자동 갱신",
            value=True,
            help="켜두면 오늘까지 자동 포함합니다. 과거 결과를 그대로 재현하려면 끄고 종료일을 고정하세요.",
        )
        if auto_bt_end:
            bt_end_date = current_date
            st.caption(f"자동 종료일: {bt_end_date.strftime('%Y-%m-%d')}")
        else:
            default_bt_end_date = get_default_backtest_end_date(current_date)
            bt_end_date = datetime.combine(
                st.date_input(
                    "백테스트 종료일",
                    default_bt_end_date,
                    help="재현성을 위해 종료일을 명시 고정합니다.",
                ),
                datetime.min.time(),
            )
        if bt_end_date > current_date:
            st.warning(f"백테스트 종료일이 오늘({current_date.strftime('%Y-%m-%d')})보다 늦어 오늘로 제한합니다.")
            bt_end_date = current_date
        if bt_end_date < bt_start_date:
            st.warning("백테스트 종료일이 시작일보다 빨라 시작일로 제한합니다.")
            bt_end_date = bt_start_date
    st.sidebar.markdown("---")

    use_adj = st.sidebar.checkbox("수정주가 사용", value=True)
    price_col = "Adj Close" if use_adj else "Close"
    options = [
        "1. MAIN",
        "2. 전략 백테스트 (시장 분석)",
        "3. 몬테카를로 시뮬레이션",
        "4. Buy & Hold",
        "5. 종목/ETF 분석",
    ]
    if "mode_select" not in st.session_state or st.session_state["mode_select"] not in options:
        st.session_state["mode_select"] = options[0]
    mode = st.sidebar.radio("기능 선택", options, key="mode_select")

    if mode.startswith("1."):
        mode_live_and_rebalance(current_dt, current_date, price_col, inv_start_date, init_capital, hist_profit, bt_start_date)
    elif mode.startswith("2."):
        mode_strategy_backtest(current_dt, bt_end_date, price_col, bt_start_date)
    elif mode.startswith("3."):
        mode_monte_carlo(current_dt, current_date, price_col, bt_start_date, init_capital)
    elif mode.startswith("4."):
        mode_buy_hold_sandbox(current_dt)
    else:
        mode_asset_analysis(current_dt, current_date, price_col)

    st.markdown("---")
    st.caption("ℹ️ 본 대시보드는 과거 데이터 기반이며 투자 권유가 아닙니다.")
    st.caption(f"📌 데이터: FinanceDataReader / ECOS | 현금: {CASH_NAME} ({CASH_TICKER}, 상장 전 CD91 프록시)")

if __name__ == "__main__":
    main()
