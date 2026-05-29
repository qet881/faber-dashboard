import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import pytz
import requests
import hashlib
import os
import re
from pathlib import Path

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
            df = fdr.DataReader(symbol, start_date, end_date)
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


st.set_page_config(page_title="통합 투자 솔루션 (분석 & 실행)", page_icon="💎", layout="wide")

# ==============================
# 1) 기본 설정값
# ==============================
DEFAULT_INVESTMENT_START_DATE = datetime(2026, 3, 31)
DEFAULT_INITIAL_CAPITAL = 249_008_318  # 3/31 종가 확정 총자산 — 수익률 계산 기준점, 수정 금지
DEFAULT_HISTORICAL_REALIZED_PROFIT = 67_571_303  # (249,008,318 - 226,356,552) + 44,919,537
DEFAULT_BACKTEST_START_DATE = datetime(2000, 1, 1)
BACKTEST_DEFAULT_END_LAG_MONTHS = 1

DEFAULT_GEN_KOSPI_BAL = 150_548_909  # 사이드바 기본값 (수익률 계산 무관)
DEFAULT_GEN_GOLD_BAL  = 533
DEFAULT_ISA_A_BAL     = 79_047_466
DEFAULT_ISA_B_BAL     = 82_489_989
DEFAULT_BALANCE_VERSION = "2026-05-29-close"

# 미확정 예정 입금 — NAV 계산에 반영되지 않음. 확정 시 CONFIRMED 으로 이동.
PERSONAL_CASH_FLOWS_PENDING = {
}
# 확정 입금 — NAV 계산에 반영됨. 입금 확정마다 여기에 추가.
PERSONAL_CASH_FLOWS_CONFIRMED = {
    "2026-04-30": 30_000_000,  # 3천만 원 대출금 Faber 추가투입
}
PERSONAL_CASH_FLOWS = PERSONAL_CASH_FLOWS_CONFIRMED  # 계산에 사용되는 것은 확정분만
APP_DIR = Path(__file__).resolve().parent
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
}

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
HAENAM_SAMSUNG_NAME = '삼성전자'
HAENAM_HYNIX_NAME = 'SK하이닉스'
HAENAM_TIME_NAME = 'TIME 나스닥100액티브'
HAENAM_KOACT_NAME = 'KoAct 나스닥100액티브'
SAMSUNG_ELECTRONICS_TICKER = '005930'
SK_HYNIX_TICKER = '000660'
CHINA_CSI300_CNY_ASSET = '중국CSI300(위안화 노출)'
INDIA_NIFTY_INR_ASSET = '인도니프티(루피 노출)'
FABER_EX_BONDS_3_LABEL = 'Faber A (한·미·금 3자산)'
FABER_EX_BONDS_4_LABEL = 'Faber A (한·미·중국·금 4자산)'
FABER_EX_BONDS_5_LABEL = 'Faber A (한·미·중국·인도·금 5자산)'

KR_BOND_DURATION_FACTOR = 2.5

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

PREFERRED_ACCOUNT = {
    '코스피200': '일반', '미국나스닥100': 'ISA',
    HAENAM_SAMSUNG_NAME: '일반', HAENAM_HYNIX_NAME: '일반',
    HAENAM_TIME_NAME: 'ISA', HAENAM_KOACT_NAME: 'ISA',
    '한국채30년': 'ISA', '미국채30년': 'ISA', '금현물': '일반'
}
GENERAL_PRIORITY = ['금현물', HAENAM_SAMSUNG_NAME, HAENAM_HYNIX_NAME, '코스피200']
ISA_PRIORITY = ['미국채30년', '한국채30년', HAENAM_TIME_NAME, HAENAM_KOACT_NAME, '미국나스닥100']
ACCOUNT_COLUMNS = ["금계좌", "일반계좌", "ISA_A", "ISA_B"]
MIN_VALID_MONTHS = 12


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


@st.cache_data(ttl=86400)
def fetch_deep_proxy_kr_bond_ecos(start_date, end_date):
    """FRED IRLTLT01KRM156N(한국 장기금리 월별) → 30년채 합성가격 딥프록시 (2000-01-01~).
    KOSEF국고채10년(148070) 상장 전(2009-08-27) 구간 커버.
    듀레이션 배수(KR_BOND_DURATION_FACTOR=2.5) 적용.
    ※ 과거 ECOS API 방식에서 FRED로 대체.
    """
    try:
        yields_raw = fetch_kr_long_bond_yield_series(start_date, end_date)
        if yields_raw is None or yields_raw.empty:
            return None
        yields_raw = yields_raw.dropna()
        # 월별 → 영업일 리샘플링 + ffill
        all_dates = pd.bdate_range(start=yields_raw.index.min(), end=yields_raw.index.max())
        yields = yields_raw.reindex(all_dates).ffill().dropna()
        # 수익률(%) → 일별 채권가격 (10년 국고채 듀레이션)
        daily_yield_change = yields.diff() / 100
        duration_10y = 10.0
        daily_return = -duration_10y / (1 + yields.shift(1) / 100) * daily_yield_change
        daily_return = daily_return.fillna(0.0)
        price_10y = (1 + daily_return).cumprod() * 100
        # KR_BOND_DURATION_FACTOR(현재값=2.5) 로 10년채 일간수익률을 레버리지해 30년채 합성.
        # ✅ 확인: adjusted_ret = daily_ret_10y × 2.5 → price_30y 는 사실상 10년채×2.5배 가격.
        #    KOSEF국고채10년×2.5배 프록시와 동일 방식으로 처리되어 체인링크 연결 일관성 확보.
        daily_ret_10y = price_10y.pct_change().fillna(0.0)
        adjusted_ret = (daily_ret_10y * KR_BOND_DURATION_FACTOR).clip(-0.15, 0.15)
        price_30y = (1 + adjusted_ret).cumprod() * 100
        result = pd.DataFrame(index=price_30y.index)
        result['Close'] = price_30y.values
        result['Adj Close'] = result['Close']
        return result
    except Exception as e:
        st.warning(f"딥프록시(KR채권 FRED) 로딩 오류: {e}")
        return None


@st.cache_data(ttl=86400)
def fetch_deep_proxy_kr_bond_10y_fred(start_date, end_date):
    """FRED IRLTLT01KRM156N(한국 장기금리 월별) → 10년 국고채 합성가격 딥프록시."""
    try:
        yields_raw = fetch_kr_long_bond_yield_series(start_date, end_date)
        if yields_raw is None or yields_raw.empty:
            return None
        yields_raw = yields_raw.dropna()
        all_dates = pd.bdate_range(start=yields_raw.index.min(), end=yields_raw.index.max())
        yields = yields_raw.reindex(all_dates).ffill().dropna()
        daily_yield_change = yields.diff() / 100
        duration_10y = 10.0
        daily_return = -duration_10y / (1 + yields.shift(1) / 100) * daily_yield_change
        daily_return = daily_return.fillna(0.0)
        price_10y = (1 + daily_return).cumprod() * 100
        result = pd.DataFrame(index=price_10y.index)
        result['Close'] = price_10y.values
        result['Adj Close'] = result['Close']
        return result
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
    """FRED GS30 → TLT 합성가격(KRW) 딥프록시 (2000-01-01~).
    TLT 상장 전(2002-07-30) 구간 커버.
    """
    try:
        yields_raw = _fetch_fred_series('GS30', start_date, end_date)
        if yields_raw is None or yields_raw.empty:
            return None
        # 영업일 기준 리인덱싱 + ffill
        all_dates = pd.bdate_range(start=yields_raw.index.min(), end=yields_raw.index.max())
        yields = yields_raw.reindex(all_dates).ffill().dropna()
        # 수익률(%) → TLT 합성가격 (USD)
        daily_yield_change = yields.diff() / 100
        duration_tlt = 18.0
        daily_return = -duration_tlt / (1 + yields.shift(1) / 100) * daily_yield_change
        daily_return = daily_return.fillna(0.0)
        price_usd = (1 + daily_return).cumprod() * 100
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
    deep_kr_bond = fetch_deep_proxy_kr_bond_ecos(start_date, end_date)
    proxy_kr_bond = fetch_proxy_data('한국채30년', start_date, end_date)  # KOSEF10년×2.5배
    etf_kr_bond = fetch_etf_data('439870', start_date, end_date)
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
def calculate_momentum_score_at_date(ticker, as_of_date, historical_data, price_col="Close"):
    try:
        if historical_data is None or len(historical_data) == 0: return None, None
        current_price = get_price_at_date(historical_data, as_of_date, price_col=price_col)
        if current_price is None: return None, None
        score, valid_months = 0, 0
        for months_ago in range(1, 13):
            past_date = as_of_date - relativedelta(months=months_ago)
            month_end = get_month_end_date(past_date)
            past_price = get_price_at_date(historical_data, month_end, price_col=price_col)
            if past_price is not None:
                if current_price > past_price: score += 1
                valid_months += 1
        if valid_months < MIN_VALID_MONTHS: return current_price, None
        return current_price, score / valid_months if valid_months > 0 else None
    except Exception: return None, None

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

def get_haenam_live_price_data(base_all_data, start_date, end_date):
    """실전 화면에서 해남 A 집행 자산의 현재가/월간 성과 계산에 쓸 데이터."""
    data = {k: v for k, v in (base_all_data or {}).items()}
    data[HAENAM_SAMSUNG_NAME] = fetch_etf_data(SAMSUNG_ELECTRONICS_TICKER, start_date, end_date, is_momentum=False)
    data[HAENAM_HYNIX_NAME] = fetch_etf_data(SK_HYNIX_TICKER, start_date, end_date, is_momentum=False)
    data[HAENAM_TIME_NAME] = fetch_etf_data(TIME_NASDAQ_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    data[HAENAM_KOACT_NAME] = fetch_etf_data(KOACT_NASDAQ_GROWTH_ACTIVE_TICKER, start_date, end_date, is_momentum=False)
    return data

def expand_haenam_execution_weights(base_weights, as_of_date, kr_weights=None):
    """Faber A 신호 비중을 해남 A 실제 집행 비중으로 변환한다.
    kr_weights로 한국주식 슬롯 집행 비중을 바꿀 수 있다(기본: 삼전/하닉 50:50)."""
    if kr_weights is None:
        kr_weights = {"samsung": 0.5, "hynix": 0.5}
    kr_samsung = float(kr_weights.get("samsung", 0.0))
    kr_hynix = float(kr_weights.get("hynix", 0.0))
    kr_total = kr_samsung + kr_hynix
    out = {}
    for asset, weight in (base_weights or {}).items():
        w = float(weight or 0.0)
        if w <= 0:
            continue
        if asset == KR_STOCK_MIX_ASSET:
            if kr_total > 0:
                if kr_samsung > 0:
                    out[HAENAM_SAMSUNG_NAME] = out.get(HAENAM_SAMSUNG_NAME, 0.0) + w * (kr_samsung / kr_total)
                if kr_hynix > 0:
                    out[HAENAM_HYNIX_NAME] = out.get(HAENAM_HYNIX_NAME, 0.0) + w * (kr_hynix / kr_total)
            else:
                out[asset] = out.get(asset, 0.0) + w
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

def expand_haenam_signal_rows(signal_rows, as_of_date, price_data_map, price_col="Adj Close"):
    """신호표의 기준 신호 행을 해남 A 실제 매수 대상 행으로 확장한다."""
    rows = []
    for row in signal_rows:
        asset = row.get("자산명")
        weight = float(row.get("추천비중", 0.0) or 0.0)
        if asset == KR_STOCK_MIX_ASSET and weight > 0:
            components = [
                (HAENAM_SAMSUNG_NAME, SAMSUNG_ELECTRONICS_TICKER, weight * 0.5),
                (HAENAM_HYNIX_NAME, SK_HYNIX_TICKER, weight * 0.5),
            ]
        elif asset == NASDAQ100_ASSET_NAME and weight > 0:
            targets = _nasdaq_active_execution_targets(as_of_date)
            components = []
            if targets.get("base", 0.0) > 0:
                components.append((NASDAQ100_ASSET_NAME, ASSETS.get(NASDAQ100_ASSET_NAME), weight * targets["base"]))
            if targets.get("time", 0.0) > 0:
                components.append((HAENAM_TIME_NAME, TIME_NASDAQ_ACTIVE_TICKER, weight * targets["time"]))
            if targets.get("koact", 0.0) > 0:
                components.append((HAENAM_KOACT_NAME, KOACT_NASDAQ_GROWTH_ACTIVE_TICKER, weight * targets["koact"]))
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
        m_mdd = calculate_monthly_mdd(nav_df)
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
# 8. 차트
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
              HAENAM_SAMSUNG_NAME: '#4c78a8', HAENAM_HYNIX_NAME: '#72b7b2',
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
def optimize_allocation(df_res, b_gen_kospi, b_gen_gold, b_isa_a, b_isa_b):
    taxable_total = float(b_gen_kospi + b_gen_gold)
    total = float(taxable_total + b_isa_a + b_isa_b)
    rem = {
        "일반계좌": taxable_total,
        "금계좌": 0.0,
        "ISA_A": float(b_isa_a),
        "ISA_B": float(b_isa_b),
    }
    weight_map = {}
    if df_res is not None and len(df_res) > 0 and "자산명" in df_res.columns and "추천비중" in df_res.columns:
        weight_map = df_res.set_index("자산명")["추천비중"].to_dict()
    final = {}

    def _asset_bucket(asset):
        if asset not in final:
            final[asset] = {col: 0.0 for col in ACCOUNT_COLUMNS}
        return final[asset]

    def _account_priority(asset):
        if asset == '금현물':
            return ["금계좌"]
        if asset in (HAENAM_TIME_NAME, HAENAM_KOACT_NAME, NASDAQ100_ASSET_NAME, '미국나스닥100'):
            return ["ISA_B", "ISA_A", "일반계좌"]
        if asset in ('미국채30년', '한국채30년'):
            return ["ISA_A", "ISA_B", "일반계좌"]
        return ["일반계좌", "ISA_A", "ISA_B"]

    def _allocate(asset, target):
        left = float(target)
        bucket = _asset_bucket(asset)
        for account in _account_priority(asset):
            if left <= 0:
                break
            if account == "금계좌":
                fill = min(left, rem["일반계좌"])
                rem["일반계좌"] -= fill
            else:
                fill = min(left, rem[account])
                rem[account] -= fill
            bucket[account] += fill
            left -= fill
        if left > 0.5:
            # 계좌 제약을 넘는 극단 상황에서는 일반계좌에 표시해 총 목표 금액이 사라지지 않게 한다.
            bucket["일반계좌"] += left
            rem["일반계좌"] -= left

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
        "금계좌": 0.0,
        "일반계좌": max(0, rem["일반계좌"]),
        "ISA_A": max(0, rem["ISA_A"]),
        "ISA_B": max(0, rem["ISA_B"]),
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
                "총목표금액": float(sum(v[col] for col in ACCOUNT_COLUMNS)),
                "금계좌": float(v["금계좌"]), "일반계좌": float(v["일반계좌"]),
                "ISA_A": float(v["ISA_A"]), "ISA_B": float(v["ISA_B"])})
    df_out = pd.DataFrame(res_list)
    for c in ["총목표금액"] + ACCOUNT_COLUMNS: df_out[c] = df_out[c].round(0)
    sum_row = {"자산명": "합계", "추천비중": 1.0, "총목표금액": float(df_out["총목표금액"].sum()),
        "금계좌": float(df_out["금계좌"].sum()), "일반계좌": float(df_out["일반계좌"].sum()),
        "ISA_A": float(df_out["ISA_A"].sum()), "ISA_B": float(df_out["ISA_B"].sum())}
    return pd.concat([df_out, pd.DataFrame([sum_row])], ignore_index=True)


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
            f"| 한국채30년 | KOSEF 국고채10년 × {KR_BOND_DURATION_FACTOR}배 | 실제 ETF (439870) |\n"
            "| 미국채30년 | TLT × USD/KRW | 실제 ETF (476760) |\n"
            "| 금현물 | GLD × USD/KRW | ACE KRX금현물 (411060) |\n"
            "| 현금 | ECOS CD91 - 0.15%p 합성 (실패 시 연 2.5%) | 실제 MMF ETF (455890) |\n\n"
            "💡 **체인링크**: 프록시 → ETF 전환 시점에서 가격을 연결하여 수익률 연속성을 유지합니다.\n\n"
            "⚠️ 금현물 Faber 신호는 별도 모멘텀 시리즈(0064K0 우선, 실패 시 GLD×환율 fallback)를 사용합니다.\n\n"
            "⚠️ 최근 기간의 모멘텀/기여도는 실제 ETF 데이터 기반으로 정확합니다."
        )

    st.caption(f"Backtest data cut-off: {requested_backtest_end.strftime('%Y-%m-%d')} (fixed)")
    data_start = bt_start_date - relativedelta(months=18)
    with st.spinner("시장 데이터 로딩 중... (하이브리드: 프록시+실제ETF, 최초 로딩 시 시간 소요)"):
        all_data = load_market_data(data_start, requested_backtest_end, hybrid=True)
        all_data = clamp_market_data_to_date(all_data, requested_backtest_end)

    # 보조 벤치마크 ETF 로딩
    benchmark_raw = fetch_benchmark_etf(BENCHMARK_ETF['ticker'], bt_start_date, requested_backtest_end)
    if benchmark_raw is not None and not benchmark_raw.empty:
        benchmark_raw = benchmark_raw[benchmark_raw.index <= pd.Timestamp(requested_backtest_end)]
    fingerprint_df, fingerprint = build_market_data_fingerprint(all_data, price_col=price_col)

    with st.expander("📊 데이터 가용 기간 확인 (하이브리드)"):
        DEEP_PROXY_NOTES = {
            '코스피200':    'KOSPI지수(딥) → KODEX200 → 실제ETF: 2000-01-01 ~ 현재',
            '미국나스닥100': 'QQQ × USD/KRW → 실제ETF: 2000-01-01 ~ 현재',
            '한국채30년':   'ECOS국고채10년×2.5배(딥) → KOSEF국고채10년×2.5배 → 실제ETF: 2000-01-01 ~ 현재',
            '미국채30년':   'FRED GS30(딥) → TLT×환율 → 실제ETF: 2000-01-01 ~ 현재',
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

    st.markdown("---")
    st.subheader(f"📊 해남 A 중심 전략 백테스트 (요청 시작: {bt_start_date.strftime('%Y-%m')})")
    
    # Faber A 원형은 기준 신호/비교군으로 보존하고, 해남 A를 실전 집행 기준으로 둔다.
    if fingerprint_df is not None:
        st.caption(f"Data fingerprint: `{fingerprint}`")
        render_backtest_reproducibility_status(
            bt_start_date, requested_backtest_end, price_col, fingerprint, fingerprint_df
        )
        with st.expander("Backtest input fingerprint", expanded=False):
            st.dataframe(fingerprint_df, use_container_width=True, hide_index=True)

    IC = 10_000_000
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
    kr_3asset_data = build_kr_stock_bond_cash_avg_momentum_data(all_data, bt_start_date, current_date)
    kr_3asset_nav = simulate_kr_stock_bond_cash_avg_momentum_strategy(
        bt_start_date, current_date, IC, kr_3asset_data, price_col=price_col
    )
    faber_nasdaq_active_data = build_faber_nasdaq_active_execution_data(
        all_data, data_start, current_date, price_col=price_col
    )
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
    faber_active_nasdaq_kr_samsung_data = build_faber_active_nasdaq_kr_single_data(
        all_data, data_start, current_date, {"samsung": 1.0}, price_col=price_col
    )
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
    old_haenam_nav = (
        simulate_daily_nav_with_attribution(
            bt_start_date, current_date, IC, faber_active_nasdaq_kr_semi_data,
            price_col=price_col
        )[0]
        if faber_active_nasdaq_kr_semi_data is not None else None
    )
    primary_nav_df = faber_active_nasdaq_kr_semi_nav if faber_active_nasdaq_kr_semi_nav is not None else nav_df
    primary_strategy_data = faber_active_nasdaq_kr_semi_data if faber_active_nasdaq_kr_semi_data is not None else all_data
    primary_price_data = haenam_price_data if faber_active_nasdaq_kr_semi_nav is not None else all_data
    primary_label = FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL if faber_active_nasdaq_kr_semi_nav is not None else "Faber A"
    primary_is_haenam = primary_label == FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL
    faber_base_label = "Faber A (원형 신호·ETF 집행)"
    old_haenam_label = "이전 전략(연속모멘텀·해남 A 집행)"
    primary_asset_keys = []
    for asset in [
        HAENAM_SAMSUNG_NAME,
        HAENAM_HYNIX_NAME,
        KR_STOCK_MIX_ASSET,
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
            return "삼성전자/하이닉스(코스피 신호)"
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
            KR_STOCK_MIX_ASSET: [KR_STOCK_MIX_ASSET, HAENAM_SAMSUNG_NAME, HAENAM_HYNIX_NAME],
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

    if faber_active_nasdaq_kr_semi_nav is None:
        st.warning("해남 A 집행 데이터가 부족해 이 화면의 메인 백테스트를 Faber A 원형으로 임시 표시합니다.")
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
    faber_ex3_nav = simulate_faber_subset_strategy(
        bt_start_date, current_date, IC, faber_ex3_data, faber_ex3_assets, price_col=price_col
    )
    faber_ex4_nav = simulate_faber_subset_strategy(
        bt_start_date, current_date, IC, faber_ex4_data, faber_ex4_assets, price_col=price_col
    )
    faber_ex5_nav = simulate_faber_subset_strategy(
        bt_start_date, current_date, IC, faber_ex5_data, faber_ex5_assets, price_col=price_col
    )

    # 동일비중 B&H
    static_nav = simulate_static_benchmark(bt_start_date, current_date, IC, all_data, price_col=price_col)
    benchmark_nav = build_benchmark_etf_returns(benchmark_raw, nav_df, IC)

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

    # 성과 지표 (해남 A 중심)
    s_value, s_return, s_mdd, s_cagr = calculate_performance_metrics(primary_nav_df, IC)
    s_peak, s_valley, _ = find_mdd_period(primary_nav_df)
    s_monthly_mdd = calculate_monthly_mdd(primary_nav_df)
    s_m_peak, s_m_valley, s_m_mdd_val = find_monthly_mdd_period(primary_nav_df)

    st.markdown(f"#### 📊 {primary_label} 전략 성과")
    st.caption(
        "해남 A는 Faber A의 12개월 고점 -5% 기준신호를 쓰되, "
        "코스피200 슬롯은 삼성전자/SK하이닉스, 나스닥100 슬롯은 TIME/KoAct로 집행하는 실전 기준입니다. "
        "Faber A 원형 데이터는 아래 비교표와 차트에 별도 유지합니다."
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
    extra = {"이전 전략: 연속 모멘텀 (참고)": (old_nav, "#ff7f0e", "dash")} if old_nav is not None else {}
    if old_haenam_nav is not None:
        extra[old_haenam_label] = (old_haenam_nav, "#8c564b", "dash")
    if primary_is_haenam:
        extra[faber_base_label] = (nav_df, "#1f77b4", "dot")
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
    quant_labels = [
        FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL,
        FABER_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL,
        FABER_ACTIVE_NASDAQ_KR_HYNIX_LABEL,
        faber_base_label,
        old_haenam_label,
        "이전 전략(연속 모멘텀)",
        FABER_NASDAQ_ACTIVE_EXEC_LABEL,
    ]
    st.caption(
        "✅ 기준 확인: "
        + " / ".join(quant_labels)
        + " 모두 월말 거래일(`_is_month_end_rebalance_day`) 기준으로만 리밸런싱합니다."
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
        sharpe  = calculate_sharpe_ratio(nav)
        sortino = calculate_sortino_ratio(nav)
        volatility = calculate_annualized_volatility(nav)
        ulcer = calculate_ulcer_index(nav)
        martin = calculate_martin_ratio(nav, period_initial)
        cvar_5 = calculate_monthly_cvar(nav, alpha=0.05)
        pos_month = calculate_positive_month_ratio(nav)
        cagr_mdd = (cagr / abs(mdd)) if (cagr is not None and cagr > 0 and mdd is not None and mdd < 0) else None
        return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe,
                "sortino": sortino, "volatility": volatility, "cagr_mdd": cagr_mdd,
                "ulcer": ulcer, "martin": martin, "cvar_5": cvar_5,
                "pos_month": pos_month}

    def _fmt(v, fmt):
        return fmt.format(v) if v is not None else "-"

    quant_strategies = {
        FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL: faber_active_nasdaq_kr_semi_nav,
        FABER_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL: faber_active_nasdaq_kr_samsung_nav,
        FABER_ACTIVE_NASDAQ_KR_HYNIX_LABEL: faber_active_nasdaq_kr_hynix_nav,
        faber_base_label: nav_df,
        old_haenam_label: old_haenam_nav,
        "이전 전략(연속 모멘텀)": old_nav,
        FABER_NASDAQ_ACTIVE_EXEC_LABEL: faber_nasdaq_active_nav,
    }
    quant_aligned, quant_meta, quant_status_df = align_strategies_to_common_dates(
        quant_strategies, min_obs_days=252
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
        haenam_turnover_keys = primary_asset_keys + [CASH_NAME]
        weight_builders = {
            FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL: (
                lambda d: expand_haenam_execution_weights(
                    calculate_faber_weights(d, all_data, mode='A', price_col=price_col), d
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
            faber_base_label: lambda d: calculate_faber_weights(d, all_data, mode='A', price_col=price_col),
            old_haenam_label: (
                lambda d: expand_haenam_execution_weights(
                    calculate_weights_at_date(d, all_data, price_col=price_col), d
                ) if old_haenam_nav is not None else None
            ),
            "이전 전략(연속 모멘텀)": lambda d: calculate_weights_at_date(d, all_data, price_col=price_col),
            FABER_NASDAQ_ACTIVE_EXEC_LABEL: (
                lambda d: calculate_faber_weights(
                    d, faber_nasdaq_active_data, mode='A', price_col=price_col
                ) if faber_nasdaq_active_data is not None else None
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
            FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL: haenam_turnover_keys,
            FABER_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL: haenam_turnover_keys,
            FABER_ACTIVE_NASDAQ_KR_HYNIX_LABEL: haenam_turnover_keys,
            faber_base_label: full_keys,
            old_haenam_label: haenam_turnover_keys,
            "이전 전략(연속 모멘텀)": full_keys,
            FABER_NASDAQ_ACTIVE_EXEC_LABEL: full_keys,
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
        metric_specs = [
            ("CAGR", "cagr", "{:.2%}"),
            ("MDD (일별)", "mdd", "{:.2%}"),
            ("변동성 (위험)", "volatility", "{:.2%}"),
            ("Sharpe", "sharpe", "{:.2f}"),
            ("Sortino", "sortino", "{:.2f}"),
            ("CAGR / MDD", "cagr_mdd", "{:.2f}"),
            ("Ulcer Index", "ulcer", "{:.2f}"),
            ("Martin Ratio", "martin", "{:.2f}"),
            ("CVaR 5% (월)", "cvar_5", "{:.2%}"),
            ("양(+)월 비율", "pos_month", "{:.1%}"),
        ]
        comparison_rows = []
        for row_label, key, fmt in metric_specs:
            comparison_rows.append(
                tuple([row_label] + [_fmt(quant_metrics[name].get(key), fmt) for name in quant_labels])
            )
        comparison_rows.append(
            tuple(["평균 월회전율(추정)"] + [_fmt(turnover_stats[name][0], "{:.1%}") for name in quant_labels])
        )
        comparison_rows.append(
            tuple(["최대 월회전율(추정)"] + [_fmt(turnover_stats[name][1], "{:.1%}") for name in quant_labels])
        )
        df_cmp = pd.DataFrame(comparison_rows, columns=["지표"] + quant_labels)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)
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

    haenam_downside = calculate_strategy_downside_comparison(
        quant_aligned.get(faber_base_label),
        quant_aligned.get(FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL),
    )
    if haenam_downside is not None:
        st.markdown("#### 🔎 해남 A 초과수익 vs 하락 민감도")
        st.caption(
            "Faber A가 오른 달과 빠진 달을 나눠서 해남 A가 추가 수익을 냈는지, "
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
                "해남 A 평균": "-",
                "해남 A 초과": _fmt(haenam_downside["avg_excess"], "{:+.2%}"),
                "해석": "월평균 초과수익이 양수면 장기 알파 후보",
            },
            {
                "구분": "Faber A 상승월",
                "월수": f"{haenam_downside['up_months']}개월",
                "Faber A 평균": _fmt(haenam_downside["up_base_avg"], "{:.2%}"),
                "해남 A 평균": _fmt(haenam_downside["up_target_avg"], "{:.2%}"),
                "해남 A 초과": _fmt(haenam_downside["up_excess_avg"], "{:+.2%}"),
                "해석": "상승장에서 집행 알파가 붙는지 확인",
            },
            {
                "구분": "Faber A 하락월",
                "월수": f"{haenam_downside['down_months']}개월",
                "Faber A 평균": _fmt(haenam_downside["down_base_avg"], "{:.2%}"),
                "해남 A 평균": _fmt(haenam_downside["down_target_avg"], "{:.2%}"),
                "해남 A 초과": _fmt(haenam_downside["down_excess_avg"], "{:+.2%}"),
                "해석": "음수가 클수록 방어 비용이 큼",
            },
            {
                "구분": "Faber A -5% 이하 월",
                "월수": stress_note,
                "Faber A 평균": _fmt(haenam_downside["stress_base_avg"], "{:.2%}"),
                "해남 A 평균": _fmt(haenam_downside["stress_target_avg"], "{:.2%}"),
                "해남 A 초과": _fmt(haenam_downside["stress_excess_avg"], "{:+.2%}"),
                "해석": "큰 하락장에서 더 깨지는지 확인",
            },
            {
                "구분": "하락월 베타",
                "월수": f"{haenam_downside['down_months']}개월 기준",
                "Faber A 평균": "1.00",
                "해남 A 평균": _fmt(haenam_downside["down_beta"], "{:.2f}"),
                "해남 A 초과": "-",
                "해석": "1보다 크면 Faber A 하락에 더 민감",
            },
            {
                "구분": "하락월 더 손실 빈도",
                "월수": f"{haenam_downside['down_months']}개월 기준",
                "Faber A 평균": "-",
                "해남 A 평균": _fmt(haenam_downside["down_worse_rate"], "{:.1%}"),
                "해남 A 초과": "-",
                "해석": "하락월에 해남 A가 더 나빴던 비율",
            },
            {
                "구분": "상승월 캡처",
                "월수": f"{haenam_downside['up_months']}개월 기준",
                "Faber A 평균": "1.00",
                "해남 A 평균": _fmt(haenam_downside["up_capture"], "{:.2f}"),
                "해남 A 초과": "-",
                "해석": "1보다 크면 상승장에서 더 강하게 따라감",
            },
        ]
        st.dataframe(pd.DataFrame(downside_rows), use_container_width=True, hide_index=True)

    quant_warn_df = quant_status_df[quant_status_df["상태"] != "비교 가능"]
    if not quant_warn_df.empty:
        st.caption("⚠️ 공정 비교 주의/제외 전략")
        st.dataframe(quant_warn_df, use_container_width=True, hide_index=True)

    # ── 정적 자산배분(20% 고정) vs 해남 A 비교 ─────────────

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

    # 해남 A 월별 비중 변화
    st.markdown("---")
    st.subheader(f"📊 {primary_label} 월별 자산 배분 비중")
    st.caption("💡 **기준신호**: Faber A와 동일하게 12개월 고점 -5% 이내 → ON. 해남 A는 ON 슬롯을 실전 집행자산으로 변환해 표시합니다.")
    
    trading_dates_all = build_trading_calendar(all_data, bt_start_date, current_date)
    faber_month_ends = _collect_month_end_dates(trading_dates_all)
    
    primary_weight_records = []
    for d in faber_month_ends:
        base_w = calculate_faber_weights(d, all_data, mode='A', price_col=price_col)
        w = expand_haenam_execution_weights(base_w, d) if primary_is_haenam else base_w
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
                       HAENAM_SAMSUNG_NAME: '#4c78a8', HAENAM_HYNIX_NAME: '#72b7b2',
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

    # 연도별 성과 요약 (해남 A 기준)
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

    # 해남 A 월별 자산 수익 기여도 분석
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
                              HAENAM_SAMSUNG_NAME: '#4c78a8', HAENAM_HYNIX_NAME: '#72b7b2',
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
                    neg_sum = abs(sum(min(0, row[ac]) for ac in asset_cols))
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
            KR_STOCK_MIX_ASSET: [KR_STOCK_MIX_ASSET, HAENAM_SAMSUNG_NAME, HAENAM_HYNIX_NAME],
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
    st.caption("실제 매수 종목이 아니라 코스피200·나스닥100 기준신호 슬롯이 매월 투자↔현금 전환되는 빈도를 분석합니다.")

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
            base_w = calculate_faber_weights(d, all_data, mode='A', price_col=price_col)
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
    st.title("투자")
    st.caption("※ 월말 종가 기준(같은 날 체결) 가정. 금현물 Faber 신호는 0064K0 기준(실시간 실패 시 GC=F×환율 fallback).")
    st.markdown("---")

    qp = _get_query_params()
    balance_defaults = [
        ("bal_gen_kospi", DEFAULT_GEN_KOSPI_BAL),
        ("bal_gen_gold", DEFAULT_GEN_GOLD_BAL),
        ("bal_isa_a", DEFAULT_ISA_A_BAL),
        ("bal_isa_b", DEFAULT_ISA_B_BAL),
    ]
    qp_balance_version = str(_qp_first(qp.get("bal_v")) or "")
    if st.session_state.get("_balance_defaults_version") != DEFAULT_BALANCE_VERSION and qp_balance_version != DEFAULT_BALANCE_VERSION:
        for key, default in balance_defaults:
            st.session_state[key] = default
        st.session_state["_balance_defaults_version"] = DEFAULT_BALANCE_VERSION
        _set_query_params(
            gen_k=DEFAULT_GEN_KOSPI_BAL, gen_g=DEFAULT_GEN_GOLD_BAL,
            isaa=DEFAULT_ISA_A_BAL, isab=DEFAULT_ISA_B_BAL,
            bal_v=DEFAULT_BALANCE_VERSION,
        )
    else:
        for key, default in balance_defaults:
            qp_key = {"bal_gen_kospi":"gen_k","bal_gen_gold":"gen_g","bal_isa_a":"isaa","bal_isa_b":"isab"}[key]
            if key not in st.session_state:
                qp_val = _get_qp_int(qp, qp_key)
                st.session_state[key] = qp_val if qp_val is not None else default
        st.session_state["_balance_defaults_version"] = DEFAULT_BALANCE_VERSION

    st.sidebar.markdown("### 💰 계좌 잔고 입력")
    st.sidebar.warning(
        "사이드바 계좌 잔고 변경분은 입출금인지 수익인지 자동 구분할 수 없습니다. "
        "외부 입출금은 반드시 PERSONAL_CASH_FLOWS에 기록해 주세요."
    )
    if st.sidebar.button("🔄 잔고 기본값으로 초기화"):
        for k, v in balance_defaults:
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
    current_total_assets = float(bal_gen + bal_isa_a + bal_isa_b)
    st.sidebar.markdown("---")
    st.sidebar.metric("총 운용 자산", f"{current_total_assets:,.0f}원")
    gold_rt_mode_label = st.sidebar.radio(
        "금 실시간 소스",
        ["안정모드 (권장)", "엄격모드 (0064K0만)"],
        index=0,
        key="gold_rt_mode",
    )
    st.sidebar.caption("안정모드: 0064K0 실패 시 최근 성공값(최대 120분) → GC=F×환율 fallback")
    gold_stable_mode = (gold_rt_mode_label == "안정모드 (권장)")

    # data_start는 bt_start_date/inv_start_date 중 더 이른 날 - 18개월이므로
    # all_data 단일 로딩으로 역대 MDD 계산까지 커버 가능 (M-1: 이중 호출 제거)
    data_start = min(bt_start_date, inv_start_date) - relativedelta(months=18)
    with st.spinner("📊 데이터를 불러오는 중..."):
        all_data = load_market_data(data_start, current_date, hybrid=True)
        haenam_strategy_data = build_faber_active_nasdaq_kr_semi_data(
            all_data, data_start, current_date, price_col=price_col
        )
        haenam_price_data = get_haenam_live_price_data(all_data, data_start, current_date)
    if haenam_strategy_data is None:
        st.warning("해남 A 집행 데이터가 부족해 실전 성과 NAV는 Faber A 기준으로 임시 계산합니다.")
        haenam_strategy_data = all_data

    # 역대 백테스트 MDD 계산 (해남 A 기준, 위에서 로딩한 데이터 재사용)
    with st.spinner("📊 역대 MDD 계산 중 (해남 A)..."):
        bt_nav_full = simulate_faber_strategy(bt_start_date, current_date, 10_000_000, haenam_strategy_data,
            mode='A', buffer_df=None, price_col=price_col)
        bt_mdd_historical = calculate_performance_metrics(bt_nav_full, 10_000_000)[2] if bt_nav_full is not None else None

    st.subheader("📊 성과 분석")
    st.markdown("#### 💼 나의 투자 성과")
    personal_nav_df = simulate_faber_strategy(inv_start_date, current_date, init_capital, haenam_strategy_data,
        mode='A', buffer_df=None, price_col=price_col)
    performance_base_date = personal_nav_df.index[-1] if personal_nav_df is not None and len(personal_nav_df) > 0 else current_date
    cumulative_principal = calculate_cumulative_principal(
        init_capital,
        PERSONAL_CASH_FLOWS,
        performance_base_date,
    )
    performance_base_date_str = pd.Timestamp(performance_base_date).strftime('%Y-%m-%d')
    realized_return_pct = (hist_profit / init_capital) * 100 if init_capital > 0 else 0.0
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
            ref_label = "역대MDD" if bt_mdd_historical and abs(bt_mdd_historical) > 0.001 else "투자기간MDD"
            if ref_mdd and abs(ref_mdd) > 0.001:
                st.warning(f"📊 **현재 고점 대비 하락률: {current_dd*100:.2f}%** | "
                           f"고점: {peak_date_str} → 현재: {performance_base_date_str} | "
                           f"{ref_label}({ref_mdd*100:.2f}%) 대비 {abs(current_dd/ref_mdd)*100:.0f}% 수준")
            else:
                st.warning(f"📊 **현재 고점 대비 하락률: {current_dd*100:.2f}%** | "
                           f"고점: {peak_date_str} → 현재: {performance_base_date_str}")
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

            # 리밸런싱 당시 해남 A 실제 집행 비중
            base_rebal_weights = calculate_faber_weights(rebal_date, all_data, mode='A', price_col=price_col)
            rebal_weights = expand_haenam_execution_weights(base_rebal_weights, rebal_date)

            monthly_rows = []
            total_pnl = 0.0
            asset_labels = [k for k, v in rebal_weights.items() if v >= 0.001]
            for an in asset_labels:
                w = rebal_weights.get(an, 0.0)
                if w < 0.001:
                    continue
                alloc_won = nav_at_rebal * w
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
                total_pnl += pnl
                monthly_rows.append({
                    "자산": an,
                    "비중": f"{w*100:.0f}%",
                    "배분금액": f"{alloc_won:,.0f}원",
                    "기준가(리밸)": f"{px_s:,.2f}",
                    "현재가": f"{px_e:,.2f}",
                    "수익률": ret * 100,
                    "손익(원)": pnl,
                })

            if monthly_rows:
                st.markdown("##### 참고: 자산별 가격변동 추정")
                cols_m = st.columns(len(monthly_rows) + 1)
                for i, row in enumerate(monthly_rows):
                    delta_color = "normal"
                    cols_m[i].metric(
                        label=row["자산"],
                        value=f"{row['수익률']:+.2f}%",
                        delta=f"{row['손익(원)']:+,.0f}원",
                    )
                total_color = "🟢" if total_pnl >= 0 else "🔴"
                cols_m[-1].metric(
                    label="📊 참고 합계",
                    value=f"{total_pnl:+,.0f}원",
                    delta=f"{total_pnl/nav_at_rebal*100:+.2f}%" if nav_at_rebal > 0 else "N/A",
                )
                with st.expander("📋 이번 달 자산별 상세"):
                    detail_df = pd.DataFrame([{
                        "자산": r["자산"],
                        "비중": r["비중"],
                        "배분금액": r["배분금액"],
                        "기준가(리밸)": r["기준가(리밸)"],
                        "현재가": r["현재가"],
                        "수익률": f"{r['수익률']:+.2f}%",
                        "손익(원)": f"{r['손익(원)']:+,.0f}원",
                    } for r in monthly_rows])
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                st.caption(f"※ 기준: {rebal_date.strftime('%Y-%m-%d')} 리밸런싱 당시 NAV {nav_at_rebal:,.0f}원 기준({nav_source}). 자산별 가격변동으로 추정한 값이며 실제와 차이 있을 수 있음.")
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
            f"**해남 A 룰**: 신호는 Faber A와 동일하게 12개월 고점 -5% 이내면 ON. "
            f"코스피200 ON은 삼성전자/SK하이닉스, 나스닥100 ON은 TIME/KoAct로 집행합니다. "
            f"금현물은 KODEX 금액티브(0064K0) 실시간 기준. "
            f"(현재가: ₩{rt_kodex_px:,.0f}{traded_txt})"
        )
    elif gold_source == "KODEX_STICKY":
        age_min = gold_rt.get("sticky_age_min")
        age_txt = f"{int(age_min):d}분 전 값" if age_min is not None else "최근 성공값"
        st.caption(
            f"**해남 A 룰**: 신호는 Faber A와 동일하게 12개월 고점 -5% 이내면 ON. "
            f"코스피200 ON은 삼성전자/SK하이닉스, 나스닥100 ON은 TIME/KoAct로 집행합니다. "
            f"금현물은 0064K0 최근 성공값({age_txt}) 기준. "
            f"(현재가: ₩{rt_kodex_px:,.0f})"
        )
    elif gold_source == "GC_FX_REALTIME":
        st.caption(
            f"**해남 A 룰**: 신호는 Faber A와 동일하게 12개월 고점 -5% 이내면 ON. "
            f"코스피200 ON은 삼성전자/SK하이닉스, 나스닥100 ON은 TIME/KoAct로 집행합니다. "
            f"금현물은 GC=F 실시간 보정(GLD 스케일 환산) 기준(0064K0 fallback). "
            f"(GLD 환산가: ${rt_gc:,.2f} | USD/KRW: ₩{rt_fx:,.0f} | 원화: ₩{rt_gold_krw:,.0f})"
        )
    elif gold_stable_mode:
        st.caption("**해남 A 룰**: 신호는 Faber A와 동일하게 12개월 고점 -5% 이내면 ON. 코스피200 ON은 삼성전자/SK하이닉스, 나스닥100 ON은 TIME/KoAct로 집행합니다. 금현물은 0064K0 종가 기준 (실시간 로딩 실패).")
    else:
        st.caption("**해남 A 룰**: 신호는 Faber A와 동일하게 12개월 고점 -5% 이내면 ON. 코스피200 ON은 삼성전자/SK하이닉스, 나스닥100 ON은 TIME/KoAct로 집행합니다. 엄격모드(0064K0만)에서 실시간을 못 받아 종가 기준으로 계산합니다.")
    col_rt1, col_rt2 = st.columns([1, 4])
    with col_rt1:
        if st.button("🔄 신호표 새로고침", help="해남 A 신호 및 추천 비중 섹션만 새로 계산"):
            # 전체 데이터 캐시는 유지하고, 신호표 계산에 필요한 실시간 소스만 갱신
            get_realtime_kodex_gold_active.clear()
            get_realtime_gold_krw.clear()
            st.rerun()
    st.subheader("📋 해남 A 신호 및 추천 비중")
    results = []
    for asset_name, ticker in ASSETS.items():
        price_data = all_data.get(asset_name)
        mom_data = all_data.get(f"{asset_name}_모멘텀")
        if ticker == '411060':
            mom_data = harmonize_gold_momentum_scale(all_data, current_date, rt_kodex_px, price_col=price_col)
        curr_price = get_price_at_date(price_data, current_date, price_col=price_col)
        _, score = calculate_momentum_score_at_date(ticker, current_date, mom_data, price_col=price_col)
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
        faber_w = 0.20 if near_high else 0.0
        display_price = signal_px if ticker == '411060' else curr_price
        results.append({
            "자산명": asset_name, "티커": ("0064K0" if ticker == '411060' else ticker), "현재가": display_price,
            "12M고점": high_12m, "고점대비": dist_from_high,
            "모멘텀": score,
            "기준신호": "● 투자 (20%)" if near_high else "○ 현금 (0%)",
            "추천비중": faber_w,
            "_is_gold": ticker == '411060'
        })
    df_results = pd.DataFrame(
        expand_haenam_signal_rows(results, current_date, haenam_price_data, price_col=price_col)
    )
    cash_weight = max(0.0, 1.0 - float(df_results["추천비중"].sum()))
    cash_price = get_price_at_date(all_data.get(CASH_NAME), current_date, price_col=price_col) or 10000.0
    df_results = pd.concat([df_results, pd.DataFrame([{
        "신호자산": CASH_NAME, "자산명": CASH_NAME, "티커": CASH_TICKER, "현재가": cash_price,
        "12M고점": None, "고점대비": None, "모멘텀": None,
        "기준신호": "-", "추천비중": cash_weight, "_is_gold": False
    }])], ignore_index=True)
    df_results_orig = df_results.copy()  # 리밸런싱용
    df_display = df_results.copy()
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
    df_display["모멘텀"] = df_display["모멘텀"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    df_display["추천비중"] = df_display["추천비중"].apply(lambda x: f"{x*100:.0f}%")
    df_display = df_display[["신호자산", "자산명", "티커", "현재가", "12M고점", "고점대비", "모멘텀", "기준신호", "추천비중"]]
    df_display.columns = ["신호자산", "집행자산", "티커", "현재가", "12M고점", "고점대비", "모멘텀(참고)", "신호", "추천비중"]
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
    st.info("👇 우선순위 배치: 금=금계좌 고정 / 일반=삼성전자→SK하이닉스 / ISA_A=채권 우선 / ISA_B=나스닥 액티브 우선")
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
        ex = df_results.drop(columns=["_is_gold"], errors="ignore").copy()
        ex["모멘텀"] = ex["모멘텀"].apply(lambda x: x if pd.notna(x) else "-")
        ex["고점대비"] = ex["고점대비"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
        ex["추천비중"] = ex["추천비중"]*100
        ex = ex[["신호자산", "자산명", "티커", "현재가", "12M고점", "고점대비", "모멘텀", "기준신호", "추천비중"]]
        ex.columns = ["신호자산","집행자산","티커","현재가","12M고점","고점대비","모멘텀(참고)","신호","추천비중(%)"]
        ex.to_excel(writer, sheet_name="Haenam_A_리밸런싱", index=False)
    st.download_button("📥 엑셀 파일 다운로드", output.getvalue(), f"HaenamA_리밸런싱_{current_dt.strftime('%Y%m%d')}.xlsx",
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
        loss_prob = (final_values < sim_capital).mean() * 100
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
        inv_start_date = datetime.combine(st.date_input("투자 시작일", DEFAULT_INVESTMENT_START_DATE), datetime.min.time())
        init_capital = st.number_input("초기 투자 원금", value=DEFAULT_INITIAL_CAPITAL, step=1000000)
        st.markdown(f"**확인:** {init_capital:,.0f}원")
        hist_profit = st.number_input("과거 누적 실현손익", value=DEFAULT_HISTORICAL_REALIZED_PROFIT, step=100000)
        st.markdown(f"**확인:** {hist_profit:,.0f}원")
        bt_start_date = datetime.combine(st.date_input("백테스트 시작일", DEFAULT_BACKTEST_START_DATE,
            help="하이브리드 모드: 2000-01-01~. FRED/ECOS 딥프록시 → 프록시 → 실제ETF 3계층 체인링크."), datetime.min.time())
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

    with st.sidebar.expander("🥇 금 괴리율 차익거래 계산기", expanded=False):
        st.markdown("""
**📌 매매 타이밍 룰**

**진입** (KRX → KODEX 금액티브)
- 매일 장 마감 후 종가 기준 괴리율 확인
- 3% 이상이면 **다음날** 계단식 비중으로 전환
- 괴리율이 더 오르면 다음 계단에서 추가 전환

**청산** (KODEX 금액티브 → KRX)
- 매일 종가 기준 괴리율 **0.5% 이하** → 다음날 **전량 한 번에** KRX 복귀
- 0.5% 초과면 KODEX 금액티브 유지 (월말 리밸런싱과 무관)
- 괴리율 거품은 한 번에 꺼지는 특성 → 계단식 청산 X

**관망**
- 0.5%~3% 구간은 대기 (진입도 청산도 안 함)

⚠️ Faber A 금 신호 OFF 시 → 괴리율 무관하게 월말에 전액 청산
        """)
        st.markdown("---")
        st.caption("계단식 비중 룰 (종가 기준 괴리율 입력)")
        krx_val = st.number_input("KRX 금 평가액", value=47998800, step=1000000, key="krx")
        sol_val = st.number_input("KODEX 금액티브 평가액", value=0, step=1000000, key="sol")
        premium = st.number_input("괴리율 (%)", value=3.0, step=0.5, key="prem")
        if st.button("매매 금액 계산", type="primary", use_container_width=True):
            total_gold = krx_val + sol_val
            if premium >= 15: tr = 1.0
            elif premium >= 12: tr = 0.8
            elif premium >= 9: tr = 0.6
            elif premium >= 6: tr = 0.4
            elif premium >= 3: tr = 0.2
            elif premium <= 0.5: tr = 0.0
            else: tr = None
            if tr is None:
                st.info("⏸️ 관망 구간 (0.5%~3%)")
            else:
                trade = total_gold * tr - sol_val
                st.write(f"**총 금:** {total_gold:,.0f}원 | **목표 KODEX 금액티브:** {tr*100:.0f}%")
                if trade > 0: st.success(f"✅ 다음날 매매 | KRX 매도 → KODEX 금액티브 매수: {trade:,.0f}원")
                elif trade < 0: st.warning(f"✅ 괴리율 0.5% 이하 → 다음날 | KODEX 금액티브 매도 → KRX 매수: {abs(trade):,.0f}원")
                else: st.info("거래 불필요")
    with st.sidebar.expander("🏠 부동산 매수 신호 (이현철 전세가율)", expanded=False):
        st.caption("전세가율 기반 매수 시점 판단 (이현철 공식)")
        re_sale = st.number_input("매매가 (만원)", value=50000, step=1000, key="re_sale")
        re_jeon = st.number_input("전세가 (만원)", value=38000, step=1000, key="re_jeon")
        re_trend = st.radio("전세가율 추이", ["상승중", "보합", "하락중"], index=1,
                            horizontal=True, key="re_trend")
        if re_sale > 0:
            jeon_rate = re_jeon / re_sale * 100
            # 기본 신호 단계
            if jeon_rate >= 80:
                base_signal, base_color = "🔵 강력 매수", "blue"
            elif jeon_rate >= 75:
                base_signal, base_color = "🟢 적극 매수", "green"
            elif jeon_rate >= 70:
                base_signal, base_color = "🟡 매수 고려 가능", "orange"
            else:
                base_signal, base_color = "🔴 매수 금지", "red"

            # 추이 상승중이면 한 단계 상향
            signal_labels = ["🔴 매수 금지", "🟡 매수 고려 가능", "🟢 적극 매수", "🔵 강력 매수"]
            base_idx = signal_labels.index(base_signal)
            if re_trend == "상승중" and base_idx < len(signal_labels) - 1:
                final_signal = signal_labels[base_idx + 1]
                upgraded = True
            else:
                final_signal = base_signal
                upgraded = False

            st.metric("전세가율", f"{jeon_rate:.1f}%")
            st.markdown(f"**매수 신호:** {final_signal}")
            if upgraded:
                st.caption(f"↑ 전세가율 상승 추이로 인해 {base_signal} → {final_signal} 상향")
    st.sidebar.markdown("---")

    use_adj = st.sidebar.checkbox("수정주가 사용", value=True)
    price_col = "Adj Close" if use_adj else "Close"
    options = ["1. 내 자산 & 리밸런싱 (실전)", "2. 전략 백테스트 (시장 분석)", "3. 몬테카를로 시뮬레이션"]
    if "mode_select" not in st.session_state or st.session_state["mode_select"] not in options:
        st.session_state["mode_select"] = options[0]
    mode = st.sidebar.radio("기능 선택", options, key="mode_select")

    if mode.startswith("1."): mode_live_and_rebalance(current_dt, current_date, price_col, inv_start_date, init_capital, hist_profit, bt_start_date)
    elif mode.startswith("2."): mode_strategy_backtest(current_dt, bt_end_date, price_col, bt_start_date)
    else: mode_monte_carlo(current_dt, bt_end_date, price_col, bt_start_date, init_capital)

    st.markdown("---")
    st.caption("ℹ️ 본 대시보드는 과거 데이터 기반이며 투자 권유가 아닙니다.")
    st.caption(f"📌 데이터: FinanceDataReader / ECOS | 현금: {CASH_NAME} ({CASH_TICKER}, 상장 전 CD91 프록시)")

if __name__ == "__main__":
    main()
