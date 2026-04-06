import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import pytz

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="통합 투자 솔루션 (분석 & 실행)", page_icon="💎", layout="wide")

# ==============================
# 1) 기본 설정값
# ==============================
DEFAULT_INVESTMENT_START_DATE = datetime(2026, 3, 31)
DEFAULT_INITIAL_CAPITAL = 249008318  # 3/31 종가 확정 총자산
DEFAULT_HISTORICAL_REALIZED_PROFIT = 67571303  # (249,008,318 - 226,356,552) + 44,919,537
DEFAULT_BACKTEST_START_DATE = datetime(2000, 1, 1)

DEFAULT_GEN_KOSPI_BAL = 100_870_700
DEFAULT_GEN_GOLD_BAL = 0
DEFAULT_ISA_A_BAL = 74_281_904
DEFAULT_ISA_B_BAL = 73_855_714

ASSETS = {
    '코스피200': '294400',
    '미국나스닥100': '133690',
    '한국채30년': '439870',
    '미국채30년': '476760',
    '금현물': '411060'
}

CASH_TICKER = '455890'
CASH_NAME = '현금(MMF)'

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
    'type': 'synthetic_cash',
    'annual_rate': 0.025,
    'note': '합성 현금 (연 2.5% 가정)'
}

# 보조 벤치마크 ETF
BENCHMARK_ETF = {
    'ticker': '0113D0',
    'name': 'TIME 글로벌 탑픽',
    'note': '자산배분 펀드 (보조 벤치마크)'
}

# 주식 슬롯 교체 실험: 한국주식 3종 × 미국주식 3종
KR_STOCK_SLOTS = {
    '코스피200': {'etf': '294400', 'proxy': '069500', 'proxy_type': 'kr_etf'},
    '삼성전자': {'etf': '005930', 'proxy': '005930', 'proxy_type': 'kr_stock'},
    '코스피전체': {'etf': '226490', 'proxy': '069500', 'proxy_type': 'kr_etf', 'note': 'KODEX 코스피'},
    '코스닥150': {'etf': '233740', 'proxy': '233740', 'proxy_type': 'kr_etf', 'note': 'TIGER 코스닥150'},
}
US_STOCK_SLOTS = {
    'S&P500': {'proxy': 'SPY', 'proxy_type': 'us_etf_fx', 'fx': 'USD/KRW'},
    '나스닥100': {'proxy': 'QQQ', 'proxy_type': 'us_etf_fx', 'fx': 'USD/KRW'},
    '미국배당다우존스': {'proxy': 'SCHD', 'proxy_type': 'us_etf_fx', 'fx': 'USD/KRW'},
}
# 미국주식 슬롯은 전 구간 미국 ETF × USD/KRW로 통일 (한국 ETF 미사용)
# 비교 전략 조합: 한국 3종 × 미국 3종 = 9개 전체
SLOT_STRATEGIES = [
    (kr, us)
    for kr in KR_STOCK_SLOTS.keys()
    for us in US_STOCK_SLOTS.keys()
]  # 3×3 = 9개 조합 전부

PREFERRED_ACCOUNT = {
    '코스피200': '일반', '미국나스닥100': 'ISA',
    '한국채30년': 'ISA', '미국채30년': 'ISA', '금현물': '일반'
}
GENERAL_PRIORITY = ['금현물', '코스피200']
ISA_PRIORITY = ['미국채30년', '한국채30년', '미국나스닥100']
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


# ==============================
# 4. 데이터 로딩
# ==============================
@st.cache_data(ttl=3600)
def fetch_etf_data(ticker, start_date, end_date, is_momentum=False):
    try:
        if ticker == '411060' and is_momentum:
            gld = fdr.DataReader('GLD', start_date, end_date)
            usdkrw = fdr.DataReader('USD/KRW', start_date, end_date)
            if gld is None or gld.empty or usdkrw is None or usdkrw.empty:
                return None
            gld = gld[~gld.index.duplicated(keep='last')]
            usdkrw = usdkrw[~usdkrw.index.duplicated(keep='last')]
            merged = pd.concat([gld['Close'], usdkrw['Close']], axis=1, keys=['GLD', 'USDKRW'])
            merged = merged.ffill().bfill()
            synthetic_df = pd.DataFrame(index=merged.index)
            synthetic_df['Close'] = merged['GLD'] * merged['USDKRW']
            synthetic_df['Adj Close'] = synthetic_df['Close']
            return synthetic_df
        df = fdr.DataReader(ticker, start_date, end_date)
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
            df = fdr.DataReader(config['ticker'], start_date, end_date)
            if df is None or df.empty: return None
            df = df[~df.index.duplicated(keep='last')].sort_index()
            if 'Close' not in df.columns: return None
            return df
        
        elif config['type'] == 'kr_etf_duration_adjusted':
            df = fdr.DataReader(config['ticker'], start_date, end_date)
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
            us_df = fdr.DataReader(config['ticker'], start_date, end_date)
            fx_df = fdr.DataReader(config['fx'], start_date, end_date)
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
        df = fdr.DataReader('KS11', start_date, end_date)
        if df is None or df.empty:
            return None
        df = df[~df.index.duplicated(keep='last')].sort_index()
        if 'Close' not in df.columns:
            return None
        result = pd.DataFrame(index=df.index)
        result['Close'] = df['Close'].astype(float)
        result['Adj Close'] = result['Close']
        return result
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
        from fredapi import Fred
        try:
            fred_key = st.secrets["FRED_API_KEY"]
            fred = Fred(api_key=fred_key)
        except Exception:
            fred = None
            st.warning("FRED API 키 없음 — 한국채30년 딥프록시 건너뜀.")
            return None
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        yields_raw = fred.get_series(
            'IRLTLT01KRM156N',
            observation_start=start_ts,
            observation_end=end_ts,
        )
        if yields_raw is None or yields_raw.empty:
            st.warning("FRED IRLTLT01KRM156N 데이터를 가져올 수 없습니다 (한국채30년 딥프록시).")
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
def fetch_deep_proxy_us_bond_fred(start_date, end_date):
    """FRED GS30 → TLT 합성가격(KRW) 딥프록시 (2000-01-01~).
    TLT 상장 전(2002-07-30) 구간 커버.
    """
    try:
        from fredapi import Fred
        try:
            fred_key = st.secrets["FRED_API_KEY"]
            fred = Fred(api_key=fred_key)
        except Exception:
            fred = None
            st.warning("FRED API 키 없음 — 미국채30년 딥프록시 건너뜀.")
            return None
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        yields_raw = fred.get_series('GS30', observation_start=start_ts, observation_end=end_ts)
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
        usdkrw_df = fdr.DataReader('USD/KRW', start_date, end_date)
        if usdkrw_df is None or usdkrw_df.empty:
            return None
        usdkrw = usdkrw_df['Close']
        usdkrw = usdkrw[~usdkrw.index.duplicated(keep='last')]
        merged = pd.concat([price_usd, usdkrw], axis=1, keys=['price', 'fx'])
        merged = merged.ffill().dropna()
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
    소스 우선순위:
      1순위: fdr.DataReader('GLD', ...)  — 2004-11-18~ (ETF 상장일)
      2순위: fdr.DataReader('GC=F', ...) — 금 선물, 더 긴 히스토리
    fredapi 의존성 없음.
    """
    try:
        usdkrw_df = fdr.DataReader('USD/KRW', start_date, end_date)
        if usdkrw_df is None or usdkrw_df.empty:
            st.warning("USD/KRW 데이터를 가져올 수 없습니다 (금 딥프록시).")
            return None
        usdkrw = usdkrw_df['Close']
        usdkrw = usdkrw[~usdkrw.index.duplicated(keep='last')]

        gold_usd = None
        for ticker in ('GLD', 'GC=F'):
            try:
                df = fdr.DataReader(ticker, start_date, end_date)
                if df is None or df.empty:
                    continue
                df = df[~df.index.duplicated(keep='last')].sort_index()
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                s = df[col].dropna()
                if len(s) > 0:
                    gold_usd = s
                    break
            except Exception:
                continue

        if gold_usd is None or gold_usd.empty:
            st.warning("금 딥프록시 소스(GLD, GC=F) 모두 로딩 실패.")
            return None

        merged = pd.concat([gold_usd, usdkrw], axis=1, keys=['gold', 'fx'])
        merged = merged.ffill().dropna()
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
    etf_gold_mom = fetch_etf_data('411060', start_date, end_date, is_momentum=True)
    step1 = _chain_link_series(deep_gold, proxy_gold)
    all_data['금현물'] = _chain_link_series(step1, etf_gold)
    # 모멘텀: GLD×환율 기준 유지 (실전과 동일한 설계)
    step1_mom = _chain_link_series(deep_gold, proxy_gold)
    all_data['금현물_모멘텀'] = _chain_link_series(step1_mom, etf_gold_mom)

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
            us = fdr.DataReader(ticker, start_date, end_date)
            fx = fdr.DataReader(slot_config['fx'], start_date, end_date)
            if us is None or us.empty or fx is None or fx.empty: return None
            us = us[~us.index.duplicated(keep='last')]
            fx = fx[~fx.index.duplicated(keep='last')]
            uc = us['Adj Close'] if 'Adj Close' in us.columns else us['Close']
            m = pd.concat([uc, fx['Close']], axis=1, keys=['US', 'FX']).ffill().bfill().dropna()
            r = pd.DataFrame(index=m.index)
            r['Close'] = m['US'] * m['FX']; r['Adj Close'] = r['Close']
            return r
        return None
    except Exception as e:
        st.warning(f"슬롯 프록시 오류 ({ticker}): {e}")
        return None


def build_slot_strategy_data(base_all_data, kr_slot_name, us_slot_name, start_date, end_date):
    """기존 all_data에서 한국주식/미국주식 슬롯만 교체한 데이터 생성.

    한국주식 슬롯은 3계층 체인링크로 2000-01-01부터 통일:
      KS11 딥프록시 → 슬롯 프록시 → 실제 ETF
    코스피200 슬롯은 base_all_data에 이미 3계층이 적용되어 있으므로 그대로 사용.
    """
    vdata = {k: v for k, v in base_all_data.items()}

    # 한국주식 슬롯 교체
    if kr_slot_name != '코스피200':  # 기존과 다를 때만
        kr_cfg = KR_STOCK_SLOTS[kr_slot_name]
        # 1계층: KS11 딥프록시 (2000-01-01~)
        deep_kospi = fetch_deep_proxy_kospi(start_date, end_date)
        # 2계층: 슬롯 자체 프록시 (상장일~)
        kr_proxy = _fetch_slot_proxy(kr_cfg, start_date, end_date)
        # 3계층: 실제 ETF
        kr_etf = fetch_etf_data(kr_cfg['etf'], start_date, end_date, is_momentum=False)
        step1 = _chain_link_series(deep_kospi, kr_proxy)
        kr_data = _chain_link_series(step1, kr_etf)
        if kr_data is None or kr_data.empty: return None
        vdata['코스피200'] = kr_data
        vdata['코스피200_모멘텀'] = kr_data
    
    # 미국주식 슬롯 교체 (전 구간 미국 ETF × 환율)
    if us_slot_name != '나스닥100':  # 기존과 다를 때만
        us_cfg = US_STOCK_SLOTS[us_slot_name]
        us_data = _fetch_slot_proxy(us_cfg, start_date, end_date)
        if us_data is None or us_data.empty: return None
        vdata['미국나스닥100'] = us_data
        vdata['미국나스닥100_모멘텀'] = us_data
    
    return vdata


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


def is_above_10m_sma(historical_data, as_of_date, price_col="Close"):
    """GTAA: 현재 가격이 10개월 이동평균선 위인지 판단."""
    if historical_data is None or historical_data.empty: return None
    col = price_col if price_col in historical_data.columns else "Close"
    if col not in historical_data.columns: return None
    current = get_price_at_date(historical_data, as_of_date, price_col=col)
    if current is None: return None
    # 10개월 월말 종가의 평균
    monthly_prices = []
    for m in range(1, 11):
        me = get_month_end_date(as_of_date - relativedelta(months=m))
        p = get_price_at_date(historical_data, me, price_col=col)
        if p is not None: monthly_prices.append(p)
    if len(monthly_prices) < 8: return None  # 최소 8개월 데이터
    sma_10m = np.mean(monthly_prices)
    return current > sma_10m


def calculate_gtaa_weights(as_of_date, all_data, price_col="Adj Close"):
    """GTAA: 10개월 이동평균선 위면 20%, 아래면 0%. 나머지 현금."""
    weights = {}
    for asset_name, ticker in ASSETS.items():
        if ticker == '411060':
            signal_data = all_data.get(f"{asset_name}_모멘텀")
        else:
            signal_data = all_data.get(asset_name)
        above = is_above_10m_sma(signal_data, as_of_date, price_col=price_col)
        weights[asset_name] = 0.20 if above else 0.0
    weights[CASH_NAME] = max(0.0, 1.0 - sum(weights.values()))
    return weights


def simulate_gtaa_strategy(start_date, end_date, initial_capital, all_data, price_col="Adj Close"):
    """GTAA 10개월 이동평균 전략 시뮬레이션."""
    trading_dates = build_trading_calendar(all_data, start_date, end_date)
    if len(trading_dates) == 0: return None
    actual_start = trading_dates[0]
    iw = calculate_gtaa_weights(actual_start, all_data, price_col=price_col)
    holdings = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
    cash_px = get_price_at_date(all_data.get(CASH_NAME), actual_start, price_col=price_col)
    if cash_px is None or cash_px <= 0: cash_px = 10000.0
    cash_target = initial_capital * iw.get(CASH_NAME, 0.0)
    for an in ASSETS:
        w = iw.get(an, 0.0)
        px = get_price_at_date(all_data.get(an), actual_start, price_col=price_col)
        if px and px > 0: holdings[an] = (initial_capital * w) / px
        else: cash_target += initial_capital * w
    holdings[CASH_NAME] = cash_target / cash_px if cash_px > 0 else 0.0
    daily_nav = []
    for i, date in enumerate(trading_dates):
        pv = _calc_portfolio_value(holdings, date, all_data, price_col)
        if pv <= 0: pv = initial_capital
        daily_nav.append({"date": date, "nav": pv})
        is_last = (i == len(trading_dates) - 1)
        if not is_last:
            nd = trading_dates[i + 1]
            if nd.month != date.month or nd.year != date.year: is_last = True
        if is_last and date != trading_dates[0]:
            tw = calculate_gtaa_weights(date, all_data, price_col=price_col)
            cpx = get_price_at_date(all_data.get(CASH_NAME), date, price_col=price_col)
            if cpx is None or cpx <= 0: cpx = 10000.0
            ct = pv * tw.get(CASH_NAME, 0.0)
            for an in ASSETS:
                w = tw.get(an, 0.0)
                px = get_price_at_date(all_data.get(an), date, price_col=price_col)
                if px and px > 0: holdings[an] = (pv * w) / px
                else: ct += pv * w; holdings[an] = 0.0
            holdings[CASH_NAME] = ct / cpx if cpx > 0 else 0.0
    df = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df

def calculate_faber_weights(as_of_date, all_data, mode='A', buffer_data=None, price_col="Adj Close"):
    """Faber 12-Month High Switch 비중 계산.
    mode A: 고점근처→20%, 나머지→현금
    mode B: 고점근처→20%, 나머지→IEF×환율 (무조건)
    mode C: 고점근처→20%, 나머지→IEF×환율 (IEF도 고점체크, 아니면 현금)
    mode D: 고점근처→20%, 나머지→SHV×환율 (SHV도 고점체크, 아니면 현금)
    mode E: 고점근처 AND 모멘텀>0 → 20%*모멘텀, 나머지→현금 (하이브리드)
    """
    weights = {}
    BUFFER_KEY = '_faber_buffer_'
    
    for asset_name, ticker in ASSETS.items():
        # 금현물: GLD×환율(모멘텀 데이터)로 Faber 신호 판단 (ETF 괴리율 배제)
        # 나머지: 가격 데이터(실제 ETF)로 판단
        if ticker == '411060':
            signal_data = all_data.get(f"{asset_name}_모멘텀")
        else:
            signal_data = all_data.get(asset_name)
        near_high = is_near_12month_high(signal_data, as_of_date, threshold=0.05, price_col=price_col)
        
        if mode == 'E':
            # 하이브리드: 고점 근처일 때만 모멘텀 비중 적용
            mom_data = all_data.get(f"{asset_name}_모멘텀")
            _, score = calculate_momentum_score_at_date(ticker, as_of_date, mom_data, price_col=price_col)
            if near_high and score is not None and score > 0:
                weights[asset_name] = 0.20 * score
            else:
                weights[asset_name] = 0.0
        else:
            # 이진: 고점 근처면 20%, 아니면 0%
            weights[asset_name] = 0.20 if near_high else 0.0
    
    remainder = max(0.0, 1.0 - sum(weights.values()))
    
    if mode == 'A' or mode == 'E':
        weights[CASH_NAME] = remainder
    elif mode == 'B':
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
    if buf_w > 0 and buffer_df is not None:
        bpx = get_price_at_date(buffer_df, actual_start, price_col=price_col)
        if bpx and bpx > 0: holdings[BUFFER_KEY] = (initial_capital * buf_w) / bpx
        else: cash_target += initial_capital * buf_w
    
    holdings[CASH_NAME] = cash_target / cash_px if cash_px > 0 else 0.0
    
    daily_nav = []
    for i, date in enumerate(trading_dates):
        # 포트폴리오 가치 계산
        pv = 0.0
        for an in ASSETS:
            px = get_price_at_date(all_data.get(an), date, price_col=price_col)
            if px and px > 0: pv += holdings.get(an, 0.0) * px
        if buffer_df is not None:
            bpx = get_price_at_date(buffer_df, date, price_col=price_col)
            if bpx and bpx > 0: pv += holdings.get(BUFFER_KEY, 0.0) * bpx
        cpx = get_price_at_date(all_data.get(CASH_NAME), date, price_col=price_col)
        if cpx is None or cpx <= 0: cpx = 10000.0
        pv += holdings.get(CASH_NAME, 0.0) * cpx
        if pv <= 0: pv = initial_capital
        daily_nav.append({"date": date, "nav": pv})
        
        # 월말 리밸런싱
        is_last = (i == len(trading_dates) - 1)
        if not is_last:
            nd = trading_dates[i + 1]
            if nd.month != date.month or nd.year != date.year: is_last = True
        if is_last and date != trading_dates[0]:
            tw = calculate_faber_weights(date, all_data, mode=mode, buffer_data=buffer_df, price_col=price_col)
            ct = pv * tw.get(CASH_NAME, 0.0)
            for an in ASSETS:
                w = tw.get(an, 0.0)
                px = get_price_at_date(all_data.get(an), date, price_col=price_col)
                if px and px > 0: holdings[an] = (pv * w) / px
                else: ct += pv * w; holdings[an] = 0.0
            bw = tw.get(BUFFER_KEY, 0.0)
            if bw > 0 and buffer_df is not None:
                bpx = get_price_at_date(buffer_df, date, price_col=price_col)
                if bpx and bpx > 0: holdings[BUFFER_KEY] = (pv * bw) / bpx
                else: ct += pv * bw; holdings[BUFFER_KEY] = 0.0
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

    for i, date in enumerate(trading_dates):
        portfolio_value = _calc_portfolio_value(holdings, date, all_data, price_col)
        if portfolio_value <= 0: portfolio_value = month_start_nav if month_start_nav > 0 else initial_capital
        daily_nav.append({"date": date, "nav": portfolio_value})
        is_last_day = (i == len(trading_dates) - 1)
        if not is_last_day:
            nd = trading_dates[i + 1]
            if nd.month != date.month or nd.year != date.year: is_last_day = True
        if is_last_day and date != trading_dates[0]:
            target_weights = calculate_weights_at_date(date, all_data, price_col=price_col)
            current_weights = rebalance_holdings(portfolio_value, date, target_weights, holdings, all_data, price_col=price_col)
            monthly_rebalance_dates.append(date)
            monthly_weights_history.append({"date": date, **{k: v for k, v in target_weights.items()}})
        if is_last_day and date != actual_start:
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
    """정적 동일비중(각 자산 20%) 월별 리밸런싱 벤치마크 시뮬레이션."""
    trading_dates = build_trading_calendar(all_data, start_date, end_date)
    if len(trading_dates) == 0: return None
    actual_start = trading_dates[0]
    
    # 데이터가 있는 자산만 동일비중 배분
    available_assets = []
    for name in ASSETS.keys():
        px = get_price_at_date(all_data.get(name), actual_start, price_col=price_col)
        if px is not None and px > 0:
            available_assets.append(name)
    
    if len(available_assets) == 0: return None
    
    equal_w = 1.0 / (len(available_assets) + 1)  # +1 for cash
    static_weights = {name: equal_w if name in available_assets else 0.0 for name in ASSETS.keys()}
    static_weights[CASH_NAME] = 1.0 - sum(static_weights.values())
    
    holdings = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
    rebalance_holdings(initial_capital, actual_start, static_weights, holdings, all_data, price_col=price_col)
    
    daily_nav = []
    for i, date in enumerate(trading_dates):
        pv = _calc_portfolio_value(holdings, date, all_data, price_col)
        if pv <= 0: pv = initial_capital
        daily_nav.append({"date": date, "nav": pv})
        
        is_last_day = (i == len(trading_dates) - 1)
        if not is_last_day:
            nd = trading_dates[i + 1]
            if nd.month != date.month or nd.year != date.year: is_last_day = True
        
        if is_last_day and date != trading_dates[0]:
            # 정적 비중 유지 리밸런싱 (모멘텀 없이 동일비중)
            # 해당 시점에 데이터 있는 자산만 동일비중
            avail = [n for n in ASSETS.keys() if get_price_at_date(all_data.get(n), date, price_col=price_col) not in (None, 0)]
            if len(avail) > 0:
                ew = 1.0 / (len(avail) + 1)
                sw = {n: ew if n in avail else 0.0 for n in ASSETS.keys()}
                sw[CASH_NAME] = 1.0 - sum(sw.values())
            else:
                sw = {n: 0.0 for n in ASSETS.keys()}
                sw[CASH_NAME] = 1.0
            rebalance_holdings(pv, date, sw, holdings, all_data, price_col=price_col)
    
    df = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df


def simulate_single_asset_faber(asset_name, start_date, end_date, initial_capital, all_data, price_col="Adj Close"):
    """단일 자산 Faber A: -5% 룰 → 해당 자산 100% or 현금 100%, 월말 리밸런싱."""
    trading_dates = build_trading_calendar(all_data, start_date, end_date)
    if not trading_dates:
        return None
    actual_start = trading_dates[0]
    asset_df = all_data.get(asset_name)
    cash_df = all_data.get(CASH_NAME)

    cash_px0 = get_price_at_date(cash_df, actual_start, price_col=price_col) or 10000.0
    asset_px0 = get_price_at_date(asset_df, actual_start, price_col=price_col)
    near_high0 = is_near_12month_high(asset_df, actual_start, threshold=0.05, price_col=price_col)

    holdings_asset = 0.0
    holdings_cash = 0.0
    if near_high0 and asset_px0 and asset_px0 > 0:
        holdings_asset = initial_capital / asset_px0
    else:
        holdings_cash = initial_capital / cash_px0

    daily_nav = []
    for i, date in enumerate(trading_dates):
        asset_px = get_price_at_date(asset_df, date, price_col=price_col)
        cash_px = get_price_at_date(cash_df, date, price_col=price_col) or 10000.0
        pv = (holdings_asset * asset_px if asset_px else 0.0) + holdings_cash * cash_px
        if pv <= 0:
            pv = initial_capital
        daily_nav.append({"date": date, "nav": pv})

        is_last = (i == len(trading_dates) - 1) or (
            trading_dates[i + 1].month != date.month or trading_dates[i + 1].year != date.year)
        if is_last and date != actual_start:
            near_high = is_near_12month_high(asset_df, date, threshold=0.05, price_col=price_col)
            asset_px = get_price_at_date(asset_df, date, price_col=price_col)
            cash_px = get_price_at_date(cash_df, date, price_col=price_col) or 10000.0
            if near_high and asset_px and asset_px > 0:
                holdings_asset = pv / asset_px
                holdings_cash = 0.0
            else:
                holdings_asset = 0.0
                holdings_cash = pv / cash_px

    df = pd.DataFrame(daily_nav).set_index("date").sort_index()
    df["running_max"] = df["nav"].expanding().max()
    df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
    return df


def simulate_single_asset_bh(asset_name, start_date, end_date, initial_capital, all_data, price_col="Adj Close"):
    """단일 자산 100% 보유 (Buy & Hold)."""
    trading_dates = build_trading_calendar(all_data, start_date, end_date)
    if not trading_dates:
        return None
    asset_df = all_data.get(asset_name)
    px0 = get_price_at_date(asset_df, trading_dates[0], price_col=price_col)
    if not px0 or px0 <= 0:
        return None
    holdings = initial_capital / px0

    daily_nav = []
    for date in trading_dates:
        px = get_price_at_date(asset_df, date, price_col=price_col)
        if px and px > 0:
            daily_nav.append({"date": date, "nav": holdings * px})
    if not daily_nav:
        return None
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
def calculate_performance_metrics(daily_nav_df, initial_capital):
    if daily_nav_df is None or len(daily_nav_df) == 0: return None, None, None, None
    current_value = float(daily_nav_df["nav"].iloc[-1])
    total_return = (current_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0
    mdd = float(daily_nav_df["drawdown"].min())
    days = (daily_nav_df.index[-1] - daily_nav_df.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    cagr = (current_value / initial_capital) ** (1 / years) - 1 if years > 0 else total_return
    return current_value, total_return, mdd, cagr

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
    if len(downside) == 0 or downside.std() == 0: return None
    downside_std = float(downside.std())
    return float((excess.mean() / downside_std) * np.sqrt(252))

def find_mdd_period(daily_nav_df):
    if daily_nav_df is None or len(daily_nav_df) == 0: return None, None, None
    valley_date = daily_nav_df["drawdown"].idxmin()
    mdd_value = float(daily_nav_df["drawdown"].min())
    running_max_at_valley = float(daily_nav_df.loc[valley_date, "running_max"])
    peak_dates = daily_nav_df[daily_nav_df["nav"] == running_max_at_valley].index
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
            "CAGR/MDD": f"{abs(cagr/mdd):.2f}" if cagr and mdd and abs(mdd) > 0.001 else "-",
            "_sortino_raw": sortino if sortino is not None else -999,
        })
    if not rows: return None
    df = pd.DataFrame(rows).sort_values("_sortino_raw", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "순위"
    df = df.drop(columns=["_sortino_raw"])
    return df


# ==============================
# 8. 차트
# ==============================
def create_nav_and_drawdown_chart(daily_nav_df, initial_capital, peak_date, valley_date, title,
                                   monthly_peak_date=None, monthly_valley_date=None, monthly_mdd_val=None,
                                   extra_navs=None):
    """수익률 + Drawdown 차트. extra_navs = {"이름": (nav_df, color, dash), ...}"""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("일별 수익률 곡선", "Drawdown (낙폭)"),
        vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)
    
    returns = ((daily_nav_df["nav"] / initial_capital) - 1) * 100.0
    fig.add_trace(go.Scatter(x=daily_nav_df.index, y=returns, mode="lines", name="Faber A",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>기존: %{y:.2f}%<extra></extra>"), row=1, col=1)

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
    fig.add_trace(go.Scatter(x=daily_nav_df.index, y=dd_pct, mode="lines", name="DD Faber A",
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


def create_weights_chart(monthly_weights_history):
    if not monthly_weights_history: return None
    df_w = pd.DataFrame(monthly_weights_history)
    df_w["date"] = pd.to_datetime(df_w["date"])
    df_w = df_w.sort_values("date")
    asset_cols = [c for c in df_w.columns if c != "date"]
    colors = {'코스피200': '#1f77b4', '미국나스닥100': '#ff7f0e', '한국채30년': '#2ca02c',
              '미국채30년': '#d62728', '금현물': '#FFD700', CASH_NAME: '#9467bd'}
    fig = go.Figure()
    for asset in asset_cols:
        fig.add_trace(go.Scatter(x=df_w["date"], y=df_w[asset] * 100, mode="lines", name=asset,
            stackgroup="one", line=dict(width=0.5), fillcolor=colors.get(asset),
            hovertemplate=f"<b>{asset}</b><br>" + "%{x|%Y-%m}<br>비중: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="월별 자산 배분 비중 변화 (Faber A)", xaxis_title="날짜",
        yaxis_title="비중 (%)", yaxis=dict(range=[0, 100]), height=500, hovermode="x unified",
        xaxis=dict(tickformat="%Y-%m"), margin=dict(l=60, r=40, t=80, b=80))
    return fig


# ==============================
# 9. 3계좌 최적화
# ==============================
def optimize_allocation(df_res, b_gen, b_isa_a, b_isa_b):
    total = float(b_gen + b_isa_a + b_isa_b)
    rem_gen, rem_isa_a, rem_isa_b = float(b_gen), float(b_isa_a), float(b_isa_b)
    weight_map = {}
    if df_res is not None and len(df_res) > 0 and "자산명" in df_res.columns and "추천비중" in df_res.columns:
        weight_map = df_res.set_index("자산명")["추천비중"].to_dict()
    final = {}
    ordered_assets = []
    for a in (GENERAL_PRIORITY + ISA_PRIORITY):
        if a in ASSETS and a not in ordered_assets: ordered_assets.append(a)
    for a in ASSETS.keys():
        if a not in ordered_assets: ordered_assets.append(a)
    targets = []
    for asset in ordered_assets:
        w = float(weight_map.get(asset, 0.0))
        if w <= 0: continue
        targets.append({"asset": asset, "target": float(total * w), "pref": PREFERRED_ACCOUNT.get(asset, "일반")})
    for t in targets:
        if t["pref"] != "일반": continue
        tgt = t["target"]
        f_gen = min(tgt, rem_gen); rem_gen -= f_gen; tgt -= f_gen
        f_isa_a = min(tgt, rem_isa_a); rem_isa_a -= f_isa_a; tgt -= f_isa_a
        f_isa_b = min(tgt, rem_isa_b); rem_isa_b -= f_isa_b
        final[t["asset"]] = {"일반": f_gen, "ISA_A": f_isa_a, "ISA_B": f_isa_b}
    for t in targets:
        if t["pref"] != "ISA": continue
        tgt = t["target"]
        f_isa_a = min(tgt, rem_isa_a); rem_isa_a -= f_isa_a; tgt -= f_isa_a
        f_isa_b = min(tgt, rem_isa_b); rem_isa_b -= f_isa_b; tgt -= f_isa_b
        f_gen = min(tgt, rem_gen); rem_gen -= f_gen
        final[t["asset"]] = {"일반": f_gen, "ISA_A": f_isa_a, "ISA_B": f_isa_b}
    final[CASH_NAME] = {"일반": max(0, rem_gen), "ISA_A": max(0, rem_isa_a), "ISA_B": max(0, rem_isa_b)}
    res_list = []
    for k in list(ASSETS.keys()) + [CASH_NAME]:
        if k in final:
            v = final[k]
            res_list.append({"자산명": k, "추천비중": float(weight_map.get(k, 0.0)),
                "총목표금액": float(v["일반"]+v["ISA_A"]+v["ISA_B"]),
                "일반계좌": float(v["일반"]), "ISA_A": float(v["ISA_A"]), "ISA_B": float(v["ISA_B"])})
    df_out = pd.DataFrame(res_list)
    for c in ["총목표금액","일반계좌","ISA_A","ISA_B"]: df_out[c] = df_out[c].round(0)
    sum_row = {"자산명": "합계", "추천비중": 1.0, "총목표금액": float(df_out["총목표금액"].sum()),
        "일반계좌": float(df_out["일반계좌"].sum()), "ISA_A": float(df_out["ISA_A"].sum()), "ISA_B": float(df_out["ISA_B"].sum())}
    return pd.concat([df_out, pd.DataFrame([sum_row])], ignore_index=True)


# ==============================
# 10. UI 모드들
# ==============================
def mode_strategy_backtest(current_dt, current_date, price_col, bt_start_date):
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
            "| 현금 | 합성 (연 2.5%) | 실제 MMF ETF (455890) |\n\n"
            "💡 **체인링크**: 프록시 → ETF 전환 시점에서 가격을 연결하여 수익률 연속성을 유지합니다.\n\n"
            "⚠️ 최근 기간의 모멘텀/기여도는 실제 ETF 데이터 기반으로 정확합니다."
        )

    data_start = bt_start_date - relativedelta(months=18)
    with st.spinner("시장 데이터 로딩 중... (하이브리드: 프록시+실제ETF, 최초 로딩 시 시간 소요)"):
        all_data = load_market_data(data_start, current_date, hybrid=True)

    # 보조 벤치마크 ETF 로딩
    benchmark_raw = fetch_benchmark_etf(BENCHMARK_ETF['ticker'], bt_start_date, current_date)

    with st.expander("📊 데이터 가용 기간 확인 (하이브리드)"):
        DEEP_PROXY_NOTES = {
            '코스피200':    'KOSPI지수(딥) → KODEX200 → 실제ETF: 2000-01-01 ~ 현재',
            '미국나스닥100': 'QQQ × USD/KRW → 실제ETF: 2000-01-01 ~ 현재',
            '한국채30년':   'ECOS국고채10년×2.5배(딥) → KOSEF국고채10년×2.5배 → 실제ETF: 2000-01-01 ~ 현재',
            '미국채30년':   'FRED GS30(딥) → TLT×환율 → 실제ETF: 2000-01-01 ~ 현재',
            '금현물':       'FRED금현물(딥) → GLD×환율 → 실제ETF: 2000-01-01 ~ 현재',
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

    st.subheader("📋 현재 시장 신호 및 Faber A 추천 비중")
    st.caption("💡 **Faber A 룰**: 12개월 고점 대비 -5% 이내 → 20%, 그 외 → 0%. 나머지 현금. 금현물은 GLD×환율 기준.")
    results = []
    for asset_name, ticker in ASSETS.items():
        price_data = all_data.get(asset_name)
        mom_data = all_data.get(f"{asset_name}_모멘텀")
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
        dist_from_high = ((signal_px / high_12m) - 1) if signal_px and high_12m and high_12m > 0 else None
        faber_w = 0.20 if near_high else 0.0
        display_name = f"{asset_name} (GLD×환율)" if ticker == '411060' else asset_name
        display_price = signal_px if ticker == '411060' else curr_price
        results.append({
            "자산명": display_name, "현재가": display_price,
            "12M고점": high_12m, "고점대비": dist_from_high,
            "모멘텀": score,
            "Faber신호": "● 투자" if near_high else "○ 현금",
            "추천비중": faber_w
        })
    df_res = pd.DataFrame(results)
    cash_w = max(0.0, 1.0 - float(df_res["추천비중"].sum()))
    cash_p = get_price_at_date(all_data.get(CASH_NAME), current_date, price_col=price_col) or 10000.0
    df_res = pd.concat([df_res, pd.DataFrame([{
        "자산명": CASH_NAME, "현재가": cash_p, "12M고점": None, "고점대비": None,
        "모멘텀": None, "Faber신호": "-", "추천비중": cash_w
    }])], ignore_index=True)
    df_disp = df_res.copy()
    df_disp["현재가"] = df_disp["현재가"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
    df_disp["12M고점"] = df_disp["12M고점"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
    df_disp["고점대비"] = df_disp["고점대비"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df_disp["모멘텀"] = df_disp["모멘텀"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    df_disp["추천비중"] = df_disp["추천비중"].apply(lambda x: f"{x*100:.0f}%")
    st.dataframe(df_disp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"📊 Faber A 전략 과거 성과 (Since {bt_start_date.strftime('%Y-%m')})")
    
    # 메인 전략: Faber A
    IC = 10_000_000
    nav_df = simulate_faber_strategy(bt_start_date, current_date, IC, all_data,
        mode='A', buffer_df=None, price_col=price_col)
    if nav_df is None:
        st.error("백테스트 불가(데이터 부족). 시작일을 더 최근으로 조정해보세요.")
        return
    
    # 기존 연속 모멘텀 (차트 비교 참고용)
    old_nav, _, _, _ = simulate_daily_nav_with_attribution(
        bt_start_date, current_date, IC, all_data, price_col=price_col)

    # 동일비중 B&H
    static_nav = simulate_static_benchmark(bt_start_date, current_date, IC, all_data, price_col=price_col)
    benchmark_nav = build_benchmark_etf_returns(benchmark_raw, nav_df, IC)

    # ALLW (US ETF) × USD/KRW 벤치마크 로딩
    allw_nav = None
    try:
        allw_raw = fdr.DataReader('ALLW', bt_start_date, current_date)
        allw_fx  = fdr.DataReader('USD/KRW', bt_start_date, current_date)
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

    # 성과 지표 (Faber A)
    s_value, s_return, s_mdd, s_cagr = calculate_performance_metrics(nav_df, IC)
    s_peak, s_valley, _ = find_mdd_period(nav_df)
    s_monthly_mdd = calculate_monthly_mdd(nav_df)
    s_m_peak, s_m_valley, s_m_mdd_val = find_monthly_mdd_period(nav_df)

    st.markdown("#### 📊 Faber A 전략 성과")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("백테스트 기간", f"{(current_date - bt_start_date).days}일")
    c2.metric("누적 수익률", f"{s_return*100:.2f}%")
    c3.metric("CAGR", f"{s_cagr*100:.2f}%")
    c4.metric("MDD (일별)", f"{s_mdd*100:.2f}%")
    c5.metric("MDD (월별)", f"{s_monthly_mdd*100:.2f}%" if s_monthly_mdd is not None else "N/A")
    if s_peak and s_valley:
        st.info(f"📉 **MDD (일별)**: {s_peak.strftime('%Y-%m-%d')}(고점) → {s_valley.strftime('%Y-%m-%d')}(저점) | {s_mdd*100:.2f}%")
    if s_m_peak and s_m_valley:
        st.info(f"📉 **MDD (월별)**: {s_m_peak.strftime('%Y-%m-%d')}(고점) → {s_m_valley.strftime('%Y-%m-%d')}(저점) | {s_m_mdd_val*100:.2f}%")

    st.markdown("---")
    st.subheader("📉 Faber A 성과 차트")
    extra = {"이전 전략: 연속 모멘텀 (참고)": (old_nav, "#ff7f0e", "dash")} if old_nav is not None else {}
    extra["동일비중 B&H"] = (static_nav, "gray", "dot")
    if allw_nav is not None:
        extra["ALLW (2025~)"] = (allw_nav, "#9467bd", "dashdot")
    fig = create_nav_and_drawdown_chart(nav_df, IC, s_peak, s_valley,
        "Faber A 전략: 수익률 및 Drawdown",
        monthly_peak_date=s_m_peak, monthly_valley_date=s_m_valley, monthly_mdd_val=s_m_mdd_val,
        extra_navs=extra)
    st.plotly_chart(fig, use_container_width=True)

    # ── 정량 비교 테이블 ─────────────────────────────────────
    st.markdown("#### 📐 전략 정량 비교")

    def _strategy_metrics(nav, ic):
        """nav DataFrame에서 CAGR/MDD/Sharpe/Sortino를 딕셔너리로 반환."""
        if nav is None or nav.empty:
            return None
        _, _, mdd, cagr = calculate_performance_metrics(nav, ic)
        sharpe  = calculate_sharpe_ratio(nav)
        sortino = calculate_sortino_ratio(nav)
        cagr_mdd = (cagr / abs(mdd)) if (mdd is not None and mdd < 0) else None
        return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe,
                "sortino": sortino, "cagr_mdd": cagr_mdd}

    def _fmt(v, fmt):
        return fmt.format(v) if v is not None else "-"

    m_faber = _strategy_metrics(nav_df, IC)
    m_old   = _strategy_metrics(old_nav, IC) if old_nav is not None else None

    comparison_rows = [
        ("CAGR",        _fmt(m_faber["cagr"]    if m_faber else None, "{:.2%}"),
                        _fmt(m_old["cagr"]       if m_old   else None, "{:.2%}")),
        ("MDD (일별)",  _fmt(m_faber["mdd"]      if m_faber else None, "{:.2%}"),
                        _fmt(m_old["mdd"]        if m_old   else None, "{:.2%}")),
        ("Sharpe",      _fmt(m_faber["sharpe"]   if m_faber else None, "{:.2f}"),
                        _fmt(m_old["sharpe"]     if m_old   else None, "{:.2f}")),
        ("Sortino",     _fmt(m_faber["sortino"]  if m_faber else None, "{:.2f}"),
                        _fmt(m_old["sortino"]    if m_old   else None, "{:.2f}")),
        ("CAGR / MDD",  _fmt(m_faber["cagr_mdd"] if m_faber else None, "{:.2f}"),
                        _fmt(m_old["cagr_mdd"]   if m_old   else None, "{:.2f}")),
    ]

    df_cmp = pd.DataFrame(comparison_rows, columns=["지표", "Faber A ⭐", "이전 전략(연속 모멘텀)"])
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    # ── 정적 자산배분(20% 고정) vs Faber A 비교 ─────────────
    st.markdown("---")
    st.subheader("📊 정적 자산배분 (20% 균등) vs Faber A")
    st.caption("현금 없이 5자산 각 20%를 고정 후 월말 리밸런싱. Faber A의 타이밍 능력이 단순 분산 대비 얼마나 유효한지 확인합니다.")

    # simulate_static_equal_weight: 현금 슬롯 없이 5자산 정확히 20% × 5
    @st.cache_data(ttl=3600)
    def _simulate_equal_weight_no_cash(start_date, end_date, ic, _all_data_keys, price_col="Adj Close"):
        """현금 슬롯 없음. 5자산 정확히 각 20% 고정, 월말 리밸런싱."""
        # _all_data_keys는 캐시 키 역할 (실제 데이터는 all_data 클로저)
        trading_dates = build_trading_calendar(all_data, start_date, end_date)
        if not trading_dates: return None
        eq_w = {name: 0.20 for name in ASSETS.keys()}
        eq_w[CASH_NAME] = 0.0
        holdings = {k: 0.0 for k in list(ASSETS.keys()) + [CASH_NAME]}
        rebalance_holdings(ic, trading_dates[0], eq_w, holdings, all_data, price_col=price_col)
        daily_nav = []
        for i, date in enumerate(trading_dates):
            pv = _calc_portfolio_value(holdings, date, all_data, price_col)
            if pv <= 0: pv = ic
            daily_nav.append({"date": date, "nav": pv})
            is_last = (i == len(trading_dates) - 1) or (
                trading_dates[i+1].month != date.month or trading_dates[i+1].year != date.year)
            if is_last and date != trading_dates[0]:
                avail = [n for n in ASSETS.keys()
                         if get_price_at_date(all_data.get(n), date, price_col=price_col) not in (None, 0)]
                sw = {n: (1.0 / len(avail)) if n in avail else 0.0 for n in ASSETS.keys()}
                sw[CASH_NAME] = 0.0
                rebalance_holdings(pv, date, sw, holdings, all_data, price_col=price_col)
        df = pd.DataFrame(daily_nav).set_index("date").sort_index()
        df["running_max"] = df["nav"].expanding().max()
        df["drawdown"] = (df["nav"] - df["running_max"]) / df["running_max"]
        return df

    eq_nav = _simulate_equal_weight_no_cash(
        bt_start_date, current_date, IC,
        tuple(sorted(all_data.keys())), price_col=price_col)

    if eq_nav is not None and nav_df is not None:
        static_cmp = build_comparison_table({
            'Faber A (5자산 타이밍) ⭐': nav_df,
            '정적 균등 (20%×5, 현금無)': eq_nav,
            '동일비중 B&H (현금포함)': static_nav,
        }, IC)
        if static_cmp is not None:
            st.dataframe(static_cmp, use_container_width=True)

        # 비교 차트
        fig_static = make_subplots(rows=2, cols=1,
            subplot_titles=("수익률 (%)", "Drawdown (%)"),
            vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)
        faber_pct = ((nav_df['nav'] / IC) - 1) * 100
        eq_pct    = ((eq_nav['nav']  / IC) - 1) * 100
        fig_static.add_trace(go.Scatter(x=nav_df.index, y=faber_pct, mode='lines',
            name='Faber A ⭐', line=dict(color='#1f77b4', width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Faber A: %{y:.1f}%<extra></extra>"), row=1, col=1)
        fig_static.add_trace(go.Scatter(x=eq_nav.index, y=eq_pct, mode='lines',
            name='정적 균등 20%×5', line=dict(color='#d62728', width=2, dash='dash'),
            hovertemplate="%{x|%Y-%m-%d}<br>정적: %{y:.1f}%<extra></extra>"), row=1, col=1)
        if static_nav is not None:
            st_pct = ((static_nav['nav'] / IC) - 1) * 100
            fig_static.add_trace(go.Scatter(x=static_nav.index, y=st_pct, mode='lines',
                name='동일비중+현금', line=dict(color='gray', width=1, dash='dot'),
                hovertemplate="%{x|%Y-%m-%d}<br>B&H: %{y:.1f}%<extra></extra>"), row=1, col=1)
        fig_static.add_trace(go.Scatter(x=nav_df.index, y=nav_df['drawdown']*100,
            mode='lines', name='DD Faber A', fill='tozeroy',
            line=dict(color='#1f77b4', width=1)), row=2, col=1)
        fig_static.add_trace(go.Scatter(x=eq_nav.index, y=eq_nav['drawdown']*100,
            mode='lines', name='DD 정적',
            line=dict(color='#d62728', width=1.5, dash='dash')), row=2, col=1)
        fig_static.update_yaxes(title_text="수익률 (%)", row=1, col=1)
        fig_static.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
        fig_static.update_layout(
            title="Faber A vs 정적 균등 자산배분 (20%×5)",
            height=650, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_static, use_container_width=True)
        st.caption("💡 **정적 균등**: 시장 상황에 관계없이 5자산 20%씩 유지. "
                   "Faber A보다 MDD가 크지만 CAGR도 높을 수 있음 — 타이밍 비용 vs 하락 방어 트레이드오프.")

    # Faber A 월별 비중 변화
    st.markdown("---")
    st.subheader("📊 Faber A 월별 자산 배분 비중")
    st.caption("💡 **Faber A 룰**: 12개월 고점 -5% 이내 → 20%, 아님 → 0%. ● = 투자, ○ = 현금.")
    
    trading_dates_all = build_trading_calendar(all_data, bt_start_date, current_date)
    faber_month_ends = []
    for i, d in enumerate(trading_dates_all):
        if i == len(trading_dates_all) - 1:
            faber_month_ends.append(d)
        elif trading_dates_all[i+1].month != d.month or trading_dates_all[i+1].year != d.year:
            faber_month_ends.append(d)
    
    faber_weight_records = []
    for d in faber_month_ends:
        w = calculate_faber_weights(d, all_data, mode='A', price_col=price_col)
        row = {"date": d}
        for an in ASSETS:
            row[an] = w.get(an, 0.0)
        row[CASH_NAME] = w.get(CASH_NAME, 0.0)
        faber_weight_records.append(row)
    
    if faber_weight_records:
        df_fw = pd.DataFrame(faber_weight_records)
        df_fw["date"] = pd.to_datetime(df_fw["date"])
        df_fw = df_fw.sort_values("date")
        # 비중 차트 (stacked area)
        asset_colors = {'코스피200': '#1f77b4', '미국나스닥100': '#ff7f0e', '한국채30년': '#2ca02c',
                       '미국채30년': '#d62728', '금현물': '#FFD700', CASH_NAME: '#9467bd'}
        fig_fw = go.Figure()
        acols = [c for c in df_fw.columns if c != "date"]
        for ac in acols:
            fig_fw.add_trace(go.Scatter(x=df_fw["date"], y=df_fw[ac]*100, mode="lines",
                name=ac, stackgroup="one", line=dict(width=0),
                fillcolor=asset_colors.get(ac, "#7f7f7f")))
        fig_fw.update_layout(title="Faber A 월별 자산 배분 비중", xaxis_title="날짜",
            yaxis_title="비중 (%)", yaxis_range=[0, 100], height=400, hovermode="x unified")
        st.plotly_chart(fig_fw, use_container_width=True)
        
        with st.expander("📋 Faber A 월별 비중 상세"):
            disp_fw = df_fw.copy()
            disp_fw["월"] = disp_fw["date"].dt.strftime("%Y-%m")
            for an in ASSETS:
                disp_fw[an] = disp_fw[an].apply(lambda x: "●20%" if x > 0.01 else "○0%")
            disp_fw[CASH_NAME] = (disp_fw[CASH_NAME]*100).round(0).astype(int).astype(str) + "%"
            st.dataframe(disp_fw[["월"] + list(ASSETS.keys()) + [CASH_NAME]], 
                         use_container_width=True, hide_index=True, height=400)

    # 연도별 성과 요약 (Faber A 기준)
    if nav_df is not None and not nav_df.empty:
        years = sorted(nav_df.index.year.unique())
        if years:
            st.markdown("---")
            st.subheader("📅 Faber A 연도별 성과 요약")
            stats_list = [calculate_yearly_daily_stats(nav_df, y) for y in years]
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

    # Faber A 월별 자산 수익 기여도 분석
    if faber_weight_records and len(faber_weight_records) > 1:
        st.markdown("---")
        st.subheader("🔍 Faber A 월별 자산 수익 기여도 분석")
        
        # 월말 비중 + 다음달 자산 수익률 → 기여도
        faber_attr_list = []
        for i in range(len(faber_weight_records) - 1):
            w_rec = faber_weight_records[i]
            next_rec = faber_weight_records[i + 1]
            d_start = w_rec["date"]
            d_end = next_rec["date"]
            
            attr = {"date": d_end.strftime("%Y-%m")}
            total = 0.0
            for an in list(ASSETS.keys()) + [CASH_NAME]:
                wt = w_rec.get(an, 0.0)
                p1 = get_price_at_date(all_data.get(an), d_start, price_col=price_col)
                p2 = get_price_at_date(all_data.get(an), d_end, price_col=price_col)
                if p1 and p2 and p1 > 0 and wt > 0:
                    ret = (p2 / p1) - 1
                    contrib = wt * ret * 100  # pp
                else:
                    contrib = 0.0
                attr[an] = round(contrib, 2)
                total += contrib
            attr["합계"] = round(total, 2)
            faber_attr_list.append(attr)
        
        if faber_attr_list:
            df_fa = pd.DataFrame(faber_attr_list)
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
                asset_cols = [c for c in df_filt.columns if c not in ["date", "합계"]]
                fig_fa = go.Figure()
                attr_colors = {'코스피200': '#1f77b4', '미국나스닥100': '#ff7f0e', '한국채30년': '#2ca02c',
                              '미국채30년': '#d62728', '금현물': '#FFD700', CASH_NAME: '#9467bd'}
                for ac in asset_cols:
                    fig_fa.add_trace(go.Bar(x=df_filt["date"], y=df_filt[ac], name=ac,
                        marker_color=attr_colors.get(ac, "#7f7f7f"),
                        text=[f"{v:+.2f}pp" if abs(v) > 0.005 else "" for v in df_filt[ac]],
                        textposition="inside", textfont=dict(size=8),
                        hovertemplate="%{x}<br>" + ac + ": %{y:.2f}pp<extra></extra>"))
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
                fig_fa.update_layout(title="Faber A 월별 자산 수익 기여도", xaxis_title="월", yaxis_title="기여도 (pp)",
                    barmode="relative", height=550, hovermode="x unified")
                st.plotly_chart(fig_fa, use_container_width=True)
                
                with st.expander("📊 월별 수익률 상세"):
                    st.dataframe(df_filt, use_container_width=True, hide_index=True, height=400)

    # ==============================
    # 자산별 역할 분석
    # ==============================
    if faber_attr_list and len(faber_attr_list) > 0:
        st.markdown("---")
        st.subheader("🎯 자산별 역할 분석 (Faber A)")
        st.caption("각 자산이 포트폴리오에 얼마나 기여했는지. 채권과 금의 방어 역할을 확인할 수 있습니다.")
        
        df_all_attr = pd.DataFrame(faber_attr_list)
        asset_names = [c for c in df_all_attr.columns if c not in ["date", "합계"]]
        
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
        
        role_rows = []
        for an in asset_names:
            vals = df_all_attr[an].values
            total_months = len(vals)
            positive_months = sum(1 for v in vals if v > 0.01)
            negative_months = sum(1 for v in vals if v < -0.01)
            zero_months = total_months - positive_months - negative_months
            win_rate = positive_months / max(total_months - zero_months, 1) * 100
            cumulative = sum(vals)
            avg_gain = np.mean([v for v in vals if v > 0.01]) if positive_months > 0 else 0
            avg_loss = np.mean([v for v in vals if v < -0.01]) if negative_months > 0 else 0
            # 위기 방어: 전체 포트폴리오가 마이너스인 달에 이 자산이 플러스인 횟수
            crisis_months = [i for i, v in enumerate(df_all_attr["합계"].values) if v < -0.5]
            defense_count = sum(1 for i in crisis_months if vals[i] > 0.01) if crisis_months else 0
            defense_rate = defense_count / max(len(crisis_months), 1) * 100
            
            role_rows.append({
                "자산": an,
                "투자월": f"{total_months - zero_months}개월",
                "수익월": f"{positive_months}",
                "손실월": f"{negative_months}",
                "승률": f"{win_rate:.0f}%",
                "누적 기여": f"{cumulative:.1f}pp",
                "평균 수익": f"+{avg_gain:.2f}pp",
                "평균 손실": f"{avg_loss:.2f}pp",
                "위기방어": f"{defense_count}/{len(crisis_months)} ({defense_rate:.0f}%)",
            })
        
        df_role = pd.DataFrame(role_rows)
        st.dataframe(df_role, use_container_width=True, hide_index=True)
        st.caption("💡 **승률**: 투자한 달 중 수익이 난 비율. **위기방어**: 포트폴리오 전체가 -0.5pp 이상 손실인 달에 해당 자산이 플러스 기여한 횟수.")
        
        # 누적 기여도 차트
        cumul_data = {}
        for an in asset_names:
            cumul_data[an] = np.cumsum(df_all_attr[an].values)
        
        fig_role = go.Figure()
        role_colors = {'코스피200': '#1f77b4', '미국나스닥100': '#ff7f0e', '한국채30년': '#2ca02c',
                      '미국채30년': '#d62728', '금현물': '#FFD700', CASH_NAME: '#9467bd'}
        for an in asset_names:
            fig_role.add_trace(go.Scatter(x=df_all_attr["date"], y=cumul_data[an], mode="lines",
                name=an, line=dict(color=role_colors.get(an, "#7f7f7f"), width=2),
                hovertemplate=f"%{{x}}<br>{an}: %{{y:.1f}}pp<extra></extra>"))
        fig_role.update_layout(title="자산별 누적 기여도 (pp)", xaxis_title="월", yaxis_title="누적 기여 (pp)",
            height=450, hovermode="x unified")
        st.plotly_chart(fig_role, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Faber A 휩소(Whipsaw) 분석")
    st.caption("Faber A 룰에서 자산이 매월 투자↔현금 전환되는 빈도를 분석합니다.")

    # 휩소 분석 (Faber A)
    with st.expander("📊 Faber A 휩소(Whipsaw) 분석: 월별 비중 변화"):
        trading_dates_for_whipsaw = build_trading_calendar(all_data, bt_start_date, current_date)
        month_ends = []
        for i, d in enumerate(trading_dates_for_whipsaw):
            if i == len(trading_dates_for_whipsaw) - 1:
                month_ends.append(d)
            elif trading_dates_for_whipsaw[i+1].month != d.month or trading_dates_for_whipsaw[i+1].year != d.year:
                month_ends.append(d)
        
        whipsaw_records = []
        for d in month_ends:
            w = calculate_faber_weights(d, all_data, mode='A', price_col=price_col)
            row = {"월": d.strftime("%Y-%m")}
            for an in ASSETS:
                row[an] = "●" if w.get(an, 0) > 0.01 else "○"
            row["투자자산수"] = sum(1 for an in ASSETS if w.get(an, 0) > 0.01)
            row["현금비중"] = f"{w.get(CASH_NAME, 0)*100:.0f}%"
            whipsaw_records.append(row)
        
        df_whipsaw = pd.DataFrame(whipsaw_records)
        
        # 휩소 횟수 계산
        flip_counts = {}
        for an in ASSETS:
            col_vals = ["●" if r.get(an) == "●" else "○" for r in whipsaw_records]
            flips = sum(1 for i in range(1, len(col_vals)) if col_vals[i] != col_vals[i-1])
            flip_counts[an] = flips
        
        total_months = len(whipsaw_records)
        st.write(f"**총 {total_months}개월 중 자산별 전환(flip) 횟수:**")
        for an, fc in flip_counts.items():
            st.write(f"  {an}: **{fc}회** (평균 {total_months/max(fc,1):.0f}개월에 1번 전환)")
        
        avg_invested = np.mean([r["투자자산수"] for r in whipsaw_records])
        st.write(f"**평균 투자 자산 수: {avg_invested:.1f}개** / 5개 (평균 투자비중 {avg_invested*20:.0f}%)")
        
        st.dataframe(df_whipsaw, use_container_width=True, hide_index=True, height=400)
    
    # ==============================
    # Faber A 룰 × 주식 슬롯 비교 (3×3)
    # ==============================
    st.markdown("---")
    st.subheader("⚔️ Faber A 룰 × 주식 슬롯 비교 (Sortino 순위)")
    st.caption("Faber A(-5% 이진, 현금) 룰을 고정한 채, 한국주식 3종 × 미국주식 3종 = 9개 조합 비교.")
    
    with st.spinner("⚔️ Faber A × 9개 슬롯 시뮬레이션 중..."):
        faber_slot_navs = {}
        bh_slot_navs = {}  # B&H도 함께 추적
        for kr_name, us_name in SLOT_STRATEGIES:
            label = f"{kr_name} + {us_name}"
            if kr_name == '코스피200' and us_name == '나스닥100':
                faber_slot_navs[label + " ⭐"] = nav_df
                bh_slot_navs[label + " ⭐"] = static_nav  # 이미 계산된 동일비중 B&H
                continue
            if kr_name == '코스피200' and us_name == 'S&P500':
                vdata_sp = build_slot_strategy_data(all_data, kr_name, us_name, data_start, current_date)
                if vdata_sp is not None:
                    fnav_sp = simulate_faber_strategy(bt_start_date, current_date, IC, vdata_sp,
                        mode='A', buffer_df=None, price_col=price_col)
                    bh_sp = simulate_static_benchmark(bt_start_date, current_date, IC, vdata_sp, price_col=price_col)
                    if fnav_sp is not None:
                        faber_slot_navs[label] = fnav_sp
                        bh_slot_navs[label] = bh_sp
                continue
            
            vdata = build_slot_strategy_data(all_data, kr_name, us_name, data_start, current_date)
            if vdata is not None:
                fnav = simulate_faber_strategy(bt_start_date, current_date, IC, vdata,
                    mode='A', buffer_df=None, price_col=price_col)
                bh_nav_slot = simulate_static_benchmark(bt_start_date, current_date, IC, vdata, price_col=price_col)
                if fnav is not None:
                    faber_slot_navs[label] = fnav
                    bh_slot_navs[label] = bh_nav_slot
    
    if faber_slot_navs:
        # 기존 비교 테이블 + Faber 궁합 지표 추가
        slot_rows = []
        for name, fnav_df in faber_slot_navs.items():
            if fnav_df is None or fnav_df.empty: continue
            fv, fr, fm, fc = calculate_performance_metrics(fnav_df, IC)
            sharpe = calculate_sharpe_ratio(fnav_df)
            sortino = calculate_sortino_ratio(fnav_df)
            # B&H 성과
            bh_df = bh_slot_navs.get(name)
            bv, br, bm, bc = calculate_performance_metrics(bh_df, IC) if bh_df is not None and not bh_df.empty else (None, None, None, None)
            # Faber 궁합 지표
            faber_alpha = (fc - bc) * 100 if fc is not None and bc is not None else None
            mdd_improve = abs(bm / fm) if fm and bm and abs(fm) > 0.001 else None
            slot_rows.append({
                "전략": name,
                "CAGR": f"{fc*100:.2f}%" if fc is not None else "-",
                "MDD (일별)": f"{fm*100:.2f}%" if fm is not None else "-",
                "Sharpe": f"{sharpe:.2f}" if sharpe is not None else "-",
                "Sortino": f"{sortino:.2f}" if sortino is not None else "-",
                "↓ CAGR/MDD": f"{abs(fc/fm):.2f}" if fc and fm and abs(fm) > 0.001 else "-",
                "Faber α": f"{faber_alpha:+.2f}%p" if faber_alpha is not None else "-",
                "MDD 개선": f"{mdd_improve:.1f}×" if mdd_improve is not None else "-",
                "_sort": sortino if sortino is not None else -999,
            })
        if slot_rows:
            df_slot = pd.DataFrame(slot_rows).sort_values("_sort", ascending=False).reset_index(drop=True)
            df_slot.index = df_slot.index + 1
            df_slot.index.name = "순위"
            df_slot = df_slot.drop(columns=["_sort"])
            st.dataframe(df_slot, use_container_width=True)
            st.caption("💡 ⭐ = Faber A 기본 (코스피200+나스닥100). **Faber α** = Faber CAGR - B&H CAGR (높을수록 Faber가 잘 먹힘). "
                       "**MDD 개선** = B&H MDD ÷ Faber MDD (높을수록 Faber가 위험을 많이 줄임).")

    # 3개 핵심 조합 자산별 역할 비교
    st.markdown("---")
    st.subheader("🎯 미국주식 슬롯별 자산 역할 비교 (Faber A)")
    st.caption("코스피200 고정, 미국주식만 S&P500/나스닥100/배당다우존스로 교체. 각 자산이 어떤 역할을 하는지 비교.")
    
    role_combos = [
        ('코스피200', 'S&P500', 'S&P500'),
        ('코스피200', '나스닥100', '나스닥100'),
        ('코스피200', '미국배당다우존스', '배당다우존스'),
    ]
    
    with st.spinner("🎯 역할 분석 중..."):
        for kr_name, us_name, us_label in role_combos:
            vdata = build_slot_strategy_data(all_data, kr_name, us_name, data_start, current_date) if us_name != '나스닥100' else all_data
            if vdata is None: continue
            
            # 월말 비중 + 기여도 계산
            td_all = build_trading_calendar(vdata, bt_start_date, current_date)
            me_list = []
            for i, d in enumerate(td_all):
                if i == len(td_all) - 1: me_list.append(d)
                elif td_all[i+1].month != d.month or td_all[i+1].year != d.year: me_list.append(d)
            
            combo_attr = []
            for i in range(len(me_list) - 1):
                d_s, d_e = me_list[i], me_list[i+1]
                w = calculate_faber_weights(d_s, vdata, mode='A', price_col=price_col)
                attr = {}
                total = 0.0
                for an in list(ASSETS.keys()) + [CASH_NAME]:
                    wt = w.get(an, 0.0)
                    p1 = get_price_at_date(vdata.get(an), d_s, price_col=price_col)
                    p2 = get_price_at_date(vdata.get(an), d_e, price_col=price_col)
                    if p1 and p2 and p1 > 0 and wt > 0:
                        contrib = wt * ((p2/p1) - 1) * 100
                    else:
                        contrib = 0.0
                    attr[an] = contrib
                    total += contrib
                attr["합계"] = total
                combo_attr.append(attr)
            
            if not combo_attr: continue
            
            # 역할 요약
            role_summary = {}
            for an in list(ASSETS.keys()) + [CASH_NAME]:
                vals = [a.get(an, 0) for a in combo_attr]
                totals = [a.get("합계", 0) for a in combo_attr]
                pos = sum(1 for v in vals if v > 0.01)
                neg = sum(1 for v in vals if v < -0.01)
                zero = len(vals) - pos - neg
                invested = len(vals) - zero
                cumul = sum(vals)
                crisis = [i for i, t in enumerate(totals) if t < -0.5]
                defense = sum(1 for i in crisis if vals[i] > 0.01)
                # 미국주식 슬롯 표시명 변경
                display_name = us_label if an == '미국나스닥100' else an
                role_summary[display_name] = {
                    "누적": f"{cumul:.1f}pp",
                    "승률": f"{pos/max(invested,1)*100:.0f}%",
                    "방어": f"{defense}/{len(crisis)}"
                }
            
            # 미국주식 슬롯만 하이라이트
            us_vals = [a.get('미국나스닥100', 0) for a in combo_attr]
            us_cumul = sum(us_vals)
            us_pos = sum(1 for v in us_vals if v > 0.01)
            us_invested = sum(1 for v in us_vals if abs(v) > 0.01)
            
            st.markdown(f"**{kr_name} + {us_name}** — 미국주식 슬롯({us_label}): "
                        f"누적 {us_cumul:.1f}pp | 승률 {us_pos/max(us_invested,1)*100:.0f}% | "
                        f"투자 {us_invested}개월")
    
    # 요약 테이블
    st.info("💡 **해석**: 누적 기여가 높을수록 수익에 기여, 승률이 높을수록 안정적, "
            "투자 개월이 많을수록 Faber 룰에 자주 들어감(=고점 근처 오래 유지).")

    # ==============================
    # 🔄 채권 슬롯 교체 비교: 미국채30년 vs 미국배당다우존스
    # ==============================
    st.markdown("---")
    st.subheader("🔄 채권 슬롯 교체: 미국채30년 → 미국배당다우존스?")
    st.caption("미국채30년 자리에 SCHD×USD/KRW를 넣으면 결과가 어떻게 바뀌는지 비교합니다. "
               "나머지 4자산(코스피200, 나스닥100, 한국채30년, 금현물)은 동일.")

    with st.spinner("🔄 채권 슬롯 교체 시뮬레이션 중..."):
        try:
            # SCHD × USD/KRW 데이터 로딩
            schd_proxy = _fetch_slot_proxy(
                {'proxy': 'SCHD', 'proxy_type': 'us_etf_fx', 'fx': 'USD/KRW'},
                bt_start_date - relativedelta(months=18), current_date)

            if schd_proxy is not None and not schd_proxy.empty:
                # 기존 all_data 복사 후 미국채30년 → SCHD 교체
                alt_data = {k: v for k, v in all_data.items()}
                alt_data['미국채30년'] = schd_proxy
                alt_data['미국채30년_모멘텀'] = schd_proxy

                # 기존 전략 (미국채30년 = TLT×환율)
                orig_nav = simulate_faber_strategy(bt_start_date, current_date, IC, all_data,
                    mode='A', buffer_df=None, price_col=price_col)
                # 교체 전략 (미국채30년 → SCHD×환율)
                alt_nav = simulate_faber_strategy(bt_start_date, current_date, IC, alt_data,
                    mode='A', buffer_df=None, price_col=price_col)

                if orig_nav is not None and alt_nav is not None:
                    # 비교 테이블
                    bond_comp = build_comparison_table({
                        '기존 (미국채30년=TLT)': orig_nav,
                        '교체 (미국채30년→SCHD)': alt_nav,
                    }, IC)
                    if bond_comp is not None:
                        st.dataframe(bond_comp, use_container_width=True)

                    # 성과 요약
                    ov_, or_, om_, oc_ = calculate_performance_metrics(orig_nav, IC)
                    av_, ar_, am_, ac_ = calculate_performance_metrics(alt_nav, IC)
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("기존 CAGR", f"{oc_*100:.2f}%")
                    bc2.metric("교체 CAGR", f"{ac_*100:.2f}%", delta=f"{(ac_-oc_)*100:+.2f}%p")
                    bc3.metric("기존 MDD", f"{om_*100:.2f}%")
                    bc4.metric("교체 MDD", f"{am_*100:.2f}%", delta=f"{(am_-om_)*100:+.2f}%p")

                    # 비교 차트
                    fig_bc = make_subplots(rows=2, cols=1, subplot_titles=("수익률 (%)", "Drawdown (%)"),
                        vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)
                    or_pct = ((orig_nav['nav'] / IC) - 1) * 100
                    ar_pct = ((alt_nav['nav'] / IC) - 1) * 100
                    fig_bc.add_trace(go.Scatter(x=orig_nav.index, y=or_pct, mode='lines',
                        name='기존 (미국채30년)', line=dict(color='#1f77b4', width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>기존: %{y:.1f}%<extra></extra>"), row=1, col=1)
                    fig_bc.add_trace(go.Scatter(x=alt_nav.index, y=ar_pct, mode='lines',
                        name='교체 (SCHD)', line=dict(color='#ff7f0e', width=2, dash='dash'),
                        hovertemplate="%{x|%Y-%m-%d}<br>SCHD: %{y:.1f}%<extra></extra>"), row=1, col=1)
                    fig_bc.add_trace(go.Scatter(x=orig_nav.index, y=orig_nav['drawdown']*100,
                        mode='lines', name='DD 기존', fill='tozeroy',
                        line=dict(color='#1f77b4', width=1),
                        hovertemplate="%{x|%Y-%m-%d}<br>기존 DD: %{y:.1f}%<extra></extra>"), row=2, col=1)
                    fig_bc.add_trace(go.Scatter(x=alt_nav.index, y=alt_nav['drawdown']*100,
                        mode='lines', name='DD SCHD',
                        line=dict(color='#ff7f0e', width=1.5, dash='dash'),
                        hovertemplate="%{x|%Y-%m-%d}<br>SCHD DD: %{y:.1f}%<extra></extra>"), row=2, col=1)
                    fig_bc.update_yaxes(title_text="수익률 (%)", row=1, col=1)
                    fig_bc.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
                    fig_bc.update_layout(title="채권 슬롯 교체: 미국채30년(TLT) vs 미국배당다우존스(SCHD)",
                        height=650, hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_bc, use_container_width=True)

                    st.caption("💡 SCHD는 미국 배당 우량주 ETF. 채권과 성격이 완전히 다름 — "
                               "금리 인상기(2022)에 채권은 급락했지만 SCHD는 상대적 방어. "
                               "반면 주식 하락장에서는 채권의 헤지 역할을 못함.")
                else:
                    st.warning("시뮬레이션 실패 (데이터 부족).")
            else:
                st.warning("SCHD×USD/KRW 데이터를 가져올 수 없습니다.")
        except Exception as e:
            st.warning(f"채권 슬롯 비교 오류: {e}")

    # ==============================
    # 🛡️ 채권 슬롯 교체: 미국채30년 → TIPS 비교
    # ==============================
    st.markdown("---")
    st.subheader("🛡️ 채권 슬롯 교체: 미국채30년(TLT) → TIPS(TIP ETF) 비교")
    st.caption("미국 물가연동채(TIPS) ETF × USD/KRW를 미국채30년 자리에 대체했을 때의 성과를 비교합니다. "
               "인플레이션 헤지 효과가 있는지, Faber A 전략과 궁합은 어떤지 확인합니다.")

    with st.spinner("🛡️ TIPS 슬롯 교체 시뮬레이션 중..."):
        try:
            # TIP ETF × USD/KRW 로딩 (상장일: 2003-12-05)
            tip_raw = fdr.DataReader('TIP', data_start, current_date)
            usdkrw_tip = fdr.DataReader('USD/KRW', data_start, current_date)
            tip_nav_data = None
            if tip_raw is not None and not tip_raw.empty and usdkrw_tip is not None and not usdkrw_tip.empty:
                tip_raw = tip_raw[~tip_raw.index.duplicated(keep='last')].sort_index()
                usdkrw_tip = usdkrw_tip[~usdkrw_tip.index.duplicated(keep='last')]
                tip_col = 'Adj Close' if 'Adj Close' in tip_raw.columns else 'Close'
                merged_tip = pd.concat([tip_raw[tip_col], usdkrw_tip['Close']], axis=1, keys=['TIP', 'FX'])
                merged_tip = merged_tip.ffill().dropna()
                tip_krw = merged_tip['TIP'] * merged_tip['FX']
                tip_nav_data = pd.DataFrame(index=tip_krw.index)
                tip_nav_data['Close'] = tip_krw.values.astype(float)
                tip_nav_data['Adj Close'] = tip_nav_data['Close']

            if tip_nav_data is not None and not tip_nav_data.empty:
                tips_data = {k: v for k, v in all_data.items()}
                tips_data['미국채30년'] = tip_nav_data
                tips_data['미국채30년_모멘텀'] = tip_nav_data

                tips_nav = simulate_faber_strategy(bt_start_date, current_date, IC, tips_data,
                    mode='A', buffer_df=None, price_col=price_col)
                tips_bh  = simulate_single_asset_bh('미국채30년', bt_start_date, current_date, IC, tips_data, price_col=price_col)
                orig_bh  = simulate_single_asset_bh('미국채30년', bt_start_date, current_date, IC, all_data, price_col=price_col)

                if tips_nav is not None:
                    tips_comp = build_comparison_table({
                        '기존 Faber A (미국채30년=TLT)': nav_df,
                        '교체 Faber A (미국채30년→TIPS)': tips_nav,
                    }, IC)
                    if orig_bh is not None: tips_comp = build_comparison_table({
                        '기존 Faber A (TLT) ⭐': nav_df,
                        '교체 Faber A (TIPS)': tips_nav,
                        'TLT B&H': orig_bh,
                        'TIPS B&H': tips_bh,
                    }, IC) if tips_bh is not None else None
                    if tips_comp is not None:
                        st.dataframe(tips_comp, use_container_width=True)

                    tv_, tr_, tm_, tc_ = calculate_performance_metrics(tips_nav, IC)
                    ov_, or2_, om_, oc_ = calculate_performance_metrics(nav_df, IC)
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("기존 CAGR (TLT)", f"{oc_*100:.2f}%")
                    tc2.metric("TIPS CAGR", f"{tc_*100:.2f}%", delta=f"{(tc_-oc_)*100:+.2f}%p")
                    tc3.metric("기존 MDD (TLT)", f"{om_*100:.2f}%")
                    tc4.metric("TIPS MDD", f"{tm_*100:.2f}%", delta=f"{(tm_-om_)*100:+.2f}%p")

                    # 비교 차트
                    fig_tips = make_subplots(rows=2, cols=1,
                        subplot_titles=("수익률 (%)", "Drawdown (%)"),
                        vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)
                    orig_pct = ((nav_df['nav'] / IC) - 1) * 100
                    tips_pct = ((tips_nav['nav'] / IC) - 1) * 100
                    fig_tips.add_trace(go.Scatter(x=nav_df.index, y=orig_pct, mode='lines',
                        name='기존 (TLT) ⭐', line=dict(color='#1f77b4', width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>TLT: %{y:.1f}%<extra></extra>"), row=1, col=1)
                    fig_tips.add_trace(go.Scatter(x=tips_nav.index, y=tips_pct, mode='lines',
                        name='교체 (TIPS)', line=dict(color='#2ca02c', width=2, dash='dash'),
                        hovertemplate="%{x|%Y-%m-%d}<br>TIPS: %{y:.1f}%<extra></extra>"), row=1, col=1)
                    fig_tips.add_trace(go.Scatter(x=nav_df.index, y=nav_df['drawdown']*100,
                        mode='lines', name='DD TLT', fill='tozeroy',
                        line=dict(color='#1f77b4', width=1)), row=2, col=1)
                    fig_tips.add_trace(go.Scatter(x=tips_nav.index, y=tips_nav['drawdown']*100,
                        mode='lines', name='DD TIPS',
                        line=dict(color='#2ca02c', width=1.5, dash='dash')), row=2, col=1)
                    fig_tips.update_yaxes(title_text="수익률 (%)", row=1, col=1)
                    fig_tips.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
                    fig_tips.update_layout(
                        title="채권 슬롯 교체: 미국채30년(TLT) vs TIPS ETF",
                        height=650, hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_tips, use_container_width=True)
                    st.caption("💡 TIPS는 원금이 CPI에 연동되어 인플레이션에 강하지만, "
                               "금리 민감도가 TLT보다 낮아 디플레이션/금리 급락 국면에서 TLT 대비 방어력이 약할 수 있습니다.")
                else:
                    st.warning("TIPS 슬롯 시뮬레이션 실패 (데이터 부족).")
            else:
                st.warning("TIP ETF 데이터를 가져올 수 없습니다. (TIP 티커 FDR 미지원 가능성)")
        except Exception as e:
            st.warning(f"TIPS 슬롯 비교 오류: {e}")

    # ==============================
    # 🔬 자산별 단독 전략 비교
    # ==============================
    st.markdown("---")
    st.subheader("🔬 자산별 단독 전략 비교")
    st.caption("각 자산을 단독으로 Faber A(-5% 룰) 또는 Buy & Hold 했을 때의 성과를 Faber A 통합 전략과 비교합니다.")

    with st.spinner("🔬 자산별 단독 전략 시뮬레이션 중..."):
        asset_cmp_rows = []

        # Faber A 통합 전략 (이미 계산된 nav_df)
        def _asset_row(label, nav, ic):
            if nav is None or nav.empty:
                return None
            _, _, mdd, cagr = calculate_performance_metrics(nav, ic)
            sharpe  = calculate_sharpe_ratio(nav)
            sortino = calculate_sortino_ratio(nav)
            cagr_mdd = (cagr / abs(mdd)) if (mdd and mdd < 0) else None
            return {
                "전략": label,
                "CAGR": f"{cagr*100:.2f}%" if cagr is not None else "-",
                "MDD (일별)": f"{mdd*100:.2f}%" if mdd is not None else "-",
                "Sharpe": f"{sharpe:.2f}" if sharpe is not None else "-",
                "Sortino": f"{sortino:.2f}" if sortino is not None else "-",
                "CAGR/MDD": f"{cagr_mdd:.2f}" if cagr_mdd is not None else "-",
                "_sort": sortino if sortino is not None else -999,
            }

        r = _asset_row("Faber A 통합 (5자산) ⭐", nav_df, IC)
        if r: asset_cmp_rows.append(r)

        for aname in ASSETS.keys():
            faber1 = simulate_single_asset_faber(aname, bt_start_date, current_date, IC, all_data, price_col=price_col)
            bh1    = simulate_single_asset_bh(aname, bt_start_date, current_date, IC, all_data, price_col=price_col)
            r = _asset_row(f"{aname} Faber 단독", faber1, IC)
            if r: asset_cmp_rows.append(r)
            r = _asset_row(f"{aname} B&H", bh1, IC)
            if r: asset_cmp_rows.append(r)

    if asset_cmp_rows:
        df_acmp = (pd.DataFrame(asset_cmp_rows)
                   .sort_values("_sort", ascending=False)
                   .reset_index(drop=True))
        df_acmp.index = df_acmp.index + 1
        df_acmp.index.name = "순위"
        df_acmp = df_acmp.drop(columns=["_sort"])
        st.dataframe(df_acmp, use_container_width=True)
        st.caption("💡 **Faber 단독** = 해당 자산만 -5% 룰로 on/off. **B&H** = 해당 자산 100% 보유. "
                   "통합 전략이 단독 대비 얼마나 분산 효과를 내는지 확인할 수 있습니다.")

    # ==============================
    # 🆚 GTAA(10개월 이동평균) vs Faber A 비교
    # ==============================
    st.markdown("---")
    st.subheader("🆚 GTAA (10개월 이동평균) vs Faber A (-5% 룰)")
    st.caption("Meb Faber의 대표 전략 GTAA: 가격 > 10개월 SMA → 20%, 아래 → 0%. 같은 5자산, 같은 기간으로 직접 비교.")

    with st.expander("📌 두 전략의 차이", expanded=False):
        st.markdown(
            "| | **Faber A (-5% 룰)** ⭐ | **GTAA (10개월 SMA)** |\n"
            "|---|---|---|\n"
            "| 기준 | 12개월 고점 대비 -5% 이내 | 10개월 이동평균선 위 |\n"
            "| 보는 것 | 추세의 **건강함** (고점에서 얼마나 떨어졌나) | 추세의 **방향** (평균보다 위인가) |\n"
            "| 반응 속도 | 빠름 (고점에서 -5%면 즉시 퇴출) | 느림 (평균이 천천히 움직임) |\n"
            "| 횡보장 휩소 | 많을 수 있음 (-5% 경계선 왔다갔다) | 적음 (SMA가 부드러움) |\n"
            "| 급락장 방어 | 빠른 퇴출 | 늦은 퇴출 |\n"
        )

    with st.spinner("🆚 GTAA 시뮬레이션 중..."):
        gtaa_nav = simulate_gtaa_strategy(bt_start_date, current_date, IC, all_data, price_col=price_col)

    if gtaa_nav is not None and nav_df is not None:
        # 비교 테이블
        compare_dict = {
            'Faber A (-5% 룰) ⭐': nav_df,
            'GTAA (10개월 SMA)': gtaa_nav,
        }
        if static_nav is not None:
            compare_dict['동일비중 B&H'] = static_nav

        gtaa_comp = build_comparison_table(compare_dict, IC)
        if gtaa_comp is not None:
            st.dataframe(gtaa_comp, use_container_width=True)

        # 성과 요약
        fv_, fr_, fm_, fc_ = calculate_performance_metrics(nav_df, IC)
        gv_, gr_, gm_, gc_ = calculate_performance_metrics(gtaa_nav, IC)
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Faber A CAGR", f"{fc_*100:.2f}%")
        gc2.metric("GTAA CAGR", f"{gc_*100:.2f}%", delta=f"{(gc_-fc_)*100:+.2f}%p")
        gc3.metric("Faber A MDD", f"{fm_*100:.2f}%")
        gc4.metric("GTAA MDD", f"{gm_*100:.2f}%", delta=f"{(gm_-fm_)*100:+.2f}%p")

        # 비교 차트
        fig_gtaa = make_subplots(rows=2, cols=1, subplot_titles=("수익률 (%)", "Drawdown (%)"),
            vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)
        fr_pct = ((nav_df['nav'] / IC) - 1) * 100
        gr_pct = ((gtaa_nav['nav'] / IC) - 1) * 100
        fig_gtaa.add_trace(go.Scatter(x=nav_df.index, y=fr_pct, mode='lines',
            name='Faber A (-5%) ⭐', line=dict(color='#1f77b4', width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Faber A: %{y:.1f}%<extra></extra>"), row=1, col=1)
        fig_gtaa.add_trace(go.Scatter(x=gtaa_nav.index, y=gr_pct, mode='lines',
            name='GTAA (10M SMA)', line=dict(color='#e377c2', width=2, dash='dash'),
            hovertemplate="%{x|%Y-%m-%d}<br>GTAA: %{y:.1f}%<extra></extra>"), row=1, col=1)
        if static_nav is not None:
            sr_pct = ((static_nav['nav'] / IC) - 1) * 100
            fig_gtaa.add_trace(go.Scatter(x=static_nav.index, y=sr_pct, mode='lines',
                name='B&H', line=dict(color='gray', width=1, dash='dot'),
                hovertemplate="%{x|%Y-%m-%d}<br>B&H: %{y:.1f}%<extra></extra>"), row=1, col=1)
        # Drawdown
        fig_gtaa.add_trace(go.Scatter(x=nav_df.index, y=nav_df['drawdown']*100,
            mode='lines', name='DD Faber A', fill='tozeroy',
            line=dict(color='#1f77b4', width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>Faber DD: %{y:.1f}%<extra></extra>"), row=2, col=1)
        fig_gtaa.add_trace(go.Scatter(x=gtaa_nav.index, y=gtaa_nav['drawdown']*100,
            mode='lines', name='DD GTAA',
            line=dict(color='#e377c2', width=1.5, dash='dash'),
            hovertemplate="%{x|%Y-%m-%d}<br>GTAA DD: %{y:.1f}%<extra></extra>"), row=2, col=1)
        fig_gtaa.update_yaxes(title_text="수익률 (%)", row=1, col=1)
        fig_gtaa.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
        fig_gtaa.update_layout(title="Faber A (-5% 룰) vs GTAA (10개월 SMA) vs Buy & Hold",
            height=650, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_gtaa, use_container_width=True)

        # 월별 신호 비교
        with st.expander("📊 월별 신호 비교: Faber A vs GTAA"):
            trading_dates_cmp = build_trading_calendar(all_data, bt_start_date, current_date)
            cmp_month_ends = []
            for i, d in enumerate(trading_dates_cmp):
                if i == len(trading_dates_cmp) - 1:
                    cmp_month_ends.append(d)
                elif trading_dates_cmp[i+1].month != d.month or trading_dates_cmp[i+1].year != d.year:
                    cmp_month_ends.append(d)
            signal_rows = []
            agree_count, disagree_count = 0, 0
            for d in cmp_month_ends:
                fw = calculate_faber_weights(d, all_data, mode='A', price_col=price_col)
                gw = calculate_gtaa_weights(d, all_data, price_col=price_col)
                row = {"월": d.strftime("%Y-%m")}
                month_agree = True
                for an in ASSETS:
                    f_on = fw.get(an, 0) > 0.01
                    g_on = gw.get(an, 0) > 0.01
                    if f_on and g_on: row[an] = "●●"  # 둘 다 ON
                    elif f_on and not g_on: row[an] = "●○"  # Faber만 ON
                    elif not f_on and g_on: row[an] = "○●"  # GTAA만 ON
                    else: row[an] = "○○"  # 둘 다 OFF
                    if f_on != g_on: month_agree = False
                if month_agree: agree_count += 1
                else: disagree_count += 1
                signal_rows.append(row)
            st.write(f"**신호 일치율: {agree_count}/{agree_count+disagree_count} "
                     f"({agree_count/max(agree_count+disagree_count,1)*100:.0f}%)**")
            st.caption("●● = 둘 다 투자 | ○○ = 둘 다 현금 | ●○ = Faber만 투자 | ○● = GTAA만 투자")
            st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True, height=400)
    else:
        st.warning("GTAA 시뮬레이션 실패.")

    # ==============================
    # 🎯 급락장 분할매수 전략 백테스트
    # ==============================
    st.markdown("---")
    st.subheader("🎯 급락장 분할매수 전략 백테스트")
    st.caption("사상최고가 대비 -N% 하락 시 매수 시작, 하락일에만 매수, 전고점 회복 시 전량 매도. "
               "\"바닥을 잡지 않고 기계적으로 사는\" 전략의 실제 성과를 확인합니다.")

    with st.expander("📌 전략 룰 상세", expanded=False):
        st.markdown(
            "1. **진입 조건**: 종가가 역대 사상최고가(ATH) 대비 N% 이상 하락\n"
            "2. **매수 룰**: 진입 조건 충족 구간에서 **전일 대비 하락 마감한 날**에만 1단위 매수\n"
            "3. **매도 룰**: 종가가 직전 ATH를 회복하면 **전량 매도** (1사이클 완료)\n"
            "4. **반복**: 새로운 ATH 갱신 후 다시 -N% 하락하면 새 사이클 시작\n\n"
            "⚠️ 전제: 해당 종목이 **결국 전고점을 회복한다**. LG생활건강처럼 회복 못하면 물린다."
        )

    DIP_TICKERS = {
        '삼성전자': '005930',
        'SK하이닉스': '000660',
        'LG생활건강 ⚠️': '051900',
        'KODEX 200': '069500',
        'TIGER 나스닥100': '133690',
        'TIGER 차이나CSI300': '192090',
    }

    dc1, dc2 = st.columns(2)
    with dc1:
        dip_selected = st.selectbox("종목 선택", list(DIP_TICKERS.keys()), index=0, key="dip_ticker")
    with dc2:
        dip_threshold = st.slider("ATH 대비 하락 진입선 (%)", min_value=10, max_value=40, value=20, step=5, key="dip_thresh")

    dip_ticker = DIP_TICKERS[dip_selected]
    dip_threshold_pct = dip_threshold / 100.0

    with st.spinner(f"🎯 {dip_selected} 급락장 분할매수 백테스트 중..."):
        try:
            dip_df = fdr.DataReader(dip_ticker, '2005-01-01', current_date)
            if dip_df is None or dip_df.empty:
                st.warning(f"{dip_selected} 데이터를 가져올 수 없습니다.")
            else:
                dip_df = dip_df[~dip_df.index.duplicated(keep='last')].sort_index()
                col = 'Adj Close' if 'Adj Close' in dip_df.columns else 'Close'
                prices = dip_df[col].dropna()
                if len(prices) < 100:
                    st.warning("데이터 부족")
                else:
                    # 시뮬레이션
                    ath = float(prices.iloc[0])
                    cycles = []  # 완료된 사이클
                    current_cycle = None  # 진행 중인 사이클
                    buy_log = []  # 전체 매수 기록

                    prev_close = float(prices.iloc[0])
                    for date, price in prices.items():
                        price = float(price)

                        # ATH 갱신 (매수 중이 아닐 때)
                        if current_cycle is None:
                            if price > ath:
                                ath = price

                        # 진입 체크
                        if current_cycle is None:
                            if price <= ath * (1 - dip_threshold_pct):
                                current_cycle = {
                                    'ath': ath, 'entry_date': date, 'entry_price': price,
                                    'buys': [], 'total_shares': 0, 'total_cost': 0,
                                }

                        # 매수 체크 (진입 상태)
                        if current_cycle is not None:
                            # 전고점 회복 체크 (매도)
                            if price >= current_cycle['ath']:
                                # 전량 매도
                                if current_cycle['total_shares'] > 0:
                                    avg_cost = current_cycle['total_cost'] / current_cycle['total_shares']
                                    profit_pct = (price / avg_cost - 1) * 100
                                    cycles.append({
                                        'ath': current_cycle['ath'],
                                        'entry_date': current_cycle['entry_date'],
                                        'exit_date': date,
                                        'entry_price': current_cycle['entry_price'],
                                        'exit_price': price,
                                        'num_buys': len(current_cycle['buys']),
                                        'avg_cost': avg_cost,
                                        'total_invested': current_cycle['total_cost'],
                                        'total_value': current_cycle['total_shares'] * price,
                                        'profit_pct': profit_pct,
                                        'duration_days': (date - current_cycle['entry_date']).days,
                                        'max_drawdown_from_ath': min(
                                            (b['price'] / current_cycle['ath'] - 1) * 100
                                            for b in current_cycle['buys']
                                        ) if current_cycle['buys'] else 0,
                                    })
                                ath = price  # 새 ATH
                                current_cycle = None
                            else:
                                # 하락일 매수
                                if price < prev_close:
                                    current_cycle['buys'].append({'date': date, 'price': price})
                                    current_cycle['total_shares'] += 1
                                    current_cycle['total_cost'] += price
                                    buy_log.append({'date': date, 'price': price, 'cycle': len(cycles) + 1})

                        prev_close = price

                    # 결과 표시
                    st.markdown(f"##### 📊 {dip_selected} — ATH 대비 -{dip_threshold}% 분할매수 결과")
                    st.caption(f"데이터: {prices.index[0].strftime('%Y-%m-%d')} ~ {prices.index[-1].strftime('%Y-%m-%d')} "
                               f"({len(prices):,}일)")

                    if cycles:
                        # 완료된 사이클 테이블
                        cycle_rows = []
                        for i, c in enumerate(cycles):
                            cycle_rows.append({
                                '사이클': f"#{i+1}",
                                '진입일': c['entry_date'].strftime('%Y-%m-%d'),
                                '청산일': c['exit_date'].strftime('%Y-%m-%d'),
                                '기간': f"{c['duration_days']}일",
                                '매수횟수': f"{c['num_buys']}회",
                                '평균단가': f"{c['avg_cost']:,.0f}",
                                '청산가': f"{c['exit_price']:,.0f}",
                                '수익률': f"{c['profit_pct']:+.1f}%",
                                '최대낙폭': f"{c['max_drawdown_from_ath']:.1f}%",
                            })
                        st.dataframe(pd.DataFrame(cycle_rows), use_container_width=True, hide_index=True)

                        # 요약
                        avg_profit = np.mean([c['profit_pct'] for c in cycles])
                        avg_duration = np.mean([c['duration_days'] for c in cycles])
                        total_buys = sum(c['num_buys'] for c in cycles)
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("완료 사이클", f"{len(cycles)}회")
                        mc2.metric("평균 수익률", f"{avg_profit:+.1f}%")
                        mc3.metric("평균 소요기간", f"{avg_duration:.0f}일")
                        mc4.metric("총 매수 횟수", f"{total_buys}회")
                    else:
                        st.info("완료된 사이클이 없습니다.")

                    # 진행 중 사이클
                    if current_cycle is not None and current_cycle['total_shares'] > 0:
                        avg_c = current_cycle['total_cost'] / current_cycle['total_shares']
                        curr_p = float(prices.iloc[-1])
                        unrealized = (curr_p / avg_c - 1) * 100
                        days_in = (prices.index[-1] - current_cycle['entry_date']).days
                        recovery_needed = (current_cycle['ath'] / curr_p - 1) * 100
                        st.warning(
                            f"🔴 **진행 중 사이클** — 진입: {current_cycle['entry_date'].strftime('%Y-%m-%d')} | "
                            f"매수 {len(current_cycle['buys'])}회 | 평균단가 {avg_c:,.0f} | "
                            f"현재가 {curr_p:,.0f} | 평가손익 **{unrealized:+.1f}%** | "
                            f"{days_in}일 경과 | 전고점({current_cycle['ath']:,.0f})까지 **+{recovery_needed:.1f}%** 필요"
                        )

                    # 차트: 가격 + ATH + 매수 포인트
                    fig_dip = go.Figure()
                    fig_dip.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines',
                        name='종가', line=dict(color='#1f77b4', width=1.5),
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>"))
                    # ATH 라인
                    ath_series = prices.expanding().max()
                    fig_dip.add_trace(go.Scatter(x=ath_series.index, y=ath_series.values, mode='lines',
                        name='ATH', line=dict(color='red', width=1, dash='dot'), opacity=0.5))
                    # -N% 라인
                    threshold_series = ath_series * (1 - dip_threshold_pct)
                    fig_dip.add_trace(go.Scatter(x=threshold_series.index, y=threshold_series.values, mode='lines',
                        name=f'-{dip_threshold}% 진입선', line=dict(color='orange', width=1, dash='dash'), opacity=0.5))
                    # 매수 포인트
                    if buy_log:
                        bl_df = pd.DataFrame(buy_log)
                        fig_dip.add_trace(go.Scatter(x=bl_df['date'], y=bl_df['price'], mode='markers',
                            name='매수', marker=dict(size=4, color='green', symbol='triangle-up'),
                            hovertemplate="%{x|%Y-%m-%d}<br>매수: %{y:,.0f}<extra></extra>"))
                    # 청산 포인트
                    if cycles:
                        exit_dates = [c['exit_date'] for c in cycles]
                        exit_prices = [c['exit_price'] for c in cycles]
                        fig_dip.add_trace(go.Scatter(x=exit_dates, y=exit_prices, mode='markers',
                            name='청산 (전고점 회복)', marker=dict(size=10, color='red', symbol='star'),
                            hovertemplate="%{x|%Y-%m-%d}<br>청산: %{y:,.0f}<extra></extra>"))
                    fig_dip.update_layout(
                        title=f"{dip_selected}: ATH -{dip_threshold}% 분할매수 전략",
                        xaxis_title="날짜", yaxis_title="가격",
                        height=500, hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_dip, use_container_width=True)

                    # Faber A 대비 코멘트
                    if dip_selected == 'LG생활건강 ⚠️' and current_cycle is not None:
                        st.error("💀 **경각심**: 이 종목은 전고점 회복이 안 되고 있어서 전략이 작동하지 않습니다. "
                                 "\"결국 회복한다\"는 전제가 깨지면 이렇게 됩니다.")
        except Exception as e:
            st.warning(f"백테스트 오류: {e}")


def mode_live_and_rebalance(current_dt, current_date, price_col, inv_start_date, init_capital, hist_profit, bt_start_date):
    st.title("💎 내 자산 & 실전 리밸런싱")
    st.caption("※ 월말 종가 기준(같은 날 체결) 가정. 금현물 Faber 신호는 GLD×환율 기준.")
    st.markdown("---")

    qp = _get_query_params()
    for key, default in [("bal_gen_kospi", DEFAULT_GEN_KOSPI_BAL), ("bal_gen_gold", DEFAULT_GEN_GOLD_BAL),
                          ("bal_isa_a", DEFAULT_ISA_A_BAL), ("bal_isa_b", DEFAULT_ISA_B_BAL)]:
        qp_key = {"bal_gen_kospi":"gen_k","bal_gen_gold":"gen_g","bal_isa_a":"isaa","bal_isa_b":"isab"}[key]
        if key not in st.session_state:
            qp_val = _get_qp_int(qp, qp_key)
            st.session_state[key] = qp_val if qp_val is not None else default

    st.sidebar.markdown("### 💰 계좌 잔고 입력")
    if st.sidebar.button("🔄 잔고 기본값으로 초기화"):
        for k, v in [("bal_gen_kospi", DEFAULT_GEN_KOSPI_BAL), ("bal_gen_gold", DEFAULT_GEN_GOLD_BAL),
                      ("bal_isa_a", DEFAULT_ISA_A_BAL), ("bal_isa_b", DEFAULT_ISA_B_BAL)]:
            st.session_state[k] = v
        _set_query_params(gen_k=DEFAULT_GEN_KOSPI_BAL, gen_g=DEFAULT_GEN_GOLD_BAL, isaa=DEFAULT_ISA_A_BAL, isab=DEFAULT_ISA_B_BAL)
        st.rerun()

    bal_gen_kospi = st.sidebar.number_input("일반 계좌 (코스피 등)", key="bal_gen_kospi", step=1_000_000)
    st.sidebar.markdown(f"**확인:** {bal_gen_kospi:,.0f}원 (약 {bal_gen_kospi/10000:,.0f}만 원)")
    bal_gen_gold = st.sidebar.number_input("KRX 금현물 계좌", key="bal_gen_gold", step=1_000_000)
    st.sidebar.markdown(f"**확인:** {bal_gen_gold:,.0f}원 (약 {bal_gen_gold/10000:,.0f}만 원)")
    bal_isa_a = st.sidebar.number_input("ISA A 계좌", key="bal_isa_a", step=1_000_000)
    st.sidebar.markdown(f"**확인:** {bal_isa_a:,.0f}원 (약 {bal_isa_a/10000:,.0f}만 원)")
    bal_isa_b = st.sidebar.number_input("ISA B 계좌", key="bal_isa_b", step=1_000_000)
    st.sidebar.markdown(f"**확인:** {bal_isa_b:,.0f}원 (약 {bal_isa_b/10000:,.0f}만 원)")
    try: _set_query_params(gen_k=int(bal_gen_kospi), gen_g=int(bal_gen_gold), isaa=int(bal_isa_a), isab=int(bal_isa_b))
    except Exception: pass

    bal_gen = bal_gen_kospi + bal_gen_gold
    current_total_assets = float(bal_gen + bal_isa_a + bal_isa_b)
    st.sidebar.markdown("---")
    st.sidebar.metric("총 운용 자산", f"{current_total_assets:,.0f}원")

    data_start = min(bt_start_date, inv_start_date) - relativedelta(months=18)
    with st.spinner("📊 데이터를 불러오는 중..."):
        all_data = load_market_data(data_start, current_date, hybrid=True)

    # 역대 백테스트 MDD 계산 (Faber A 기준, 하이브리드 데이터)
    with st.spinner("📊 역대 MDD 계산 중 (Faber A)..."):
        hybrid_data = load_market_data(bt_start_date - relativedelta(months=18), current_date, hybrid=True)
        bt_nav_full = simulate_faber_strategy(bt_start_date, current_date, 10_000_000, hybrid_data,
            mode='A', buffer_df=None, price_col=price_col)
        bt_mdd_historical = calculate_performance_metrics(bt_nav_full, 10_000_000)[2] if bt_nav_full is not None else None

    st.subheader("📊 성과 분석")
    st.markdown("#### 💼 나의 투자 성과")
    personal_nav_df = simulate_faber_strategy(inv_start_date, current_date, init_capital, all_data,
        mode='A', buffer_df=None, price_col=price_col)
    realized_return_pct = (hist_profit / init_capital) * 100 if init_capital > 0 else 0.0
    acc_return = (current_total_assets - init_capital) / init_capital if init_capital > 0 else 0.0
    delta_won = current_total_assets - init_capital
    p_mdd_daily, p_mdd_monthly, p_peak, p_valley = None, None, None, None
    if personal_nav_df is not None and len(personal_nav_df) > 0:
        _, _, p_mdd_daily, _ = calculate_performance_metrics(personal_nav_df, init_capital)
        p_mdd_monthly = calculate_monthly_mdd(personal_nav_df)
        p_peak, p_valley, _ = find_mdd_period(personal_nav_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("누적 실현수익", f"{hist_profit:,.0f}원", delta=f"{realized_return_pct:.2f}%")
    c2.metric("현재 내 돈", f"{current_total_assets:,.0f}원", delta=f"{delta_won:,.0f}원")
    c3.metric("누적 수익률", f"{acc_return*100:.2f}%")
    c4.metric("MDD (일별)", f"{p_mdd_daily*100:.2f}%" if p_mdd_daily else "N/A")
    c5.metric("MDD (월별)", f"{p_mdd_monthly*100:.2f}%" if p_mdd_monthly else "N/A")
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
                           f"고점: {peak_date_str} → 현재: {current_date.strftime('%Y-%m-%d')} | "
                           f"{ref_label}({ref_mdd*100:.2f}%) 대비 {abs(current_dd/ref_mdd)*100:.0f}% 수준")
            else:
                st.warning(f"📊 **현재 고점 대비 하락률: {current_dd*100:.2f}%** | "
                           f"고점: {peak_date_str} → 현재: {current_date.strftime('%Y-%m-%d')}")
    st.caption(f"📅 투자 시작일: {inv_start_date.strftime('%Y-%m-%d')} | 초기 투자금: {init_capital:,.0f}원")

    st.markdown("---")
    st.info(f"📅 기준일: {current_dt.strftime('%Y년 %m월 %d일 %H시 %M분')}")
    st.subheader("📋 Faber A 신호 및 추천 비중")
    st.caption("**Faber A 룰**: 12개월 고점(수정주가 월말 기준) 대비 -5% 이내 → 20%, 그 외 → 0%. 나머지 현금(MMF). 금현물은 GLD×환율 기준.")
    results = []
    for asset_name, ticker in ASSETS.items():
        price_data = all_data.get(asset_name)
        mom_data = all_data.get(f"{asset_name}_모멘텀")
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
        dist_from_high = ((signal_px / high_12m) - 1) if signal_px and high_12m and high_12m > 0 else None
        faber_w = 0.20 if near_high else 0.0
        display_price = signal_px if ticker == '411060' else curr_price
        results.append({
            "자산명": asset_name, "티커": ticker, "현재가": display_price,
            "12M고점": high_12m, "고점대비": dist_from_high,
            "모멘텀": score,
            "Faber신호": "● 투자 (20%)" if near_high else "○ 현금 (0%)",
            "추천비중": faber_w,
            "_is_gold": ticker == '411060'
        })
    df_results = pd.DataFrame(results)
    # 금현물 자산명에 (GLD×환율) 표시 (리밸런싱용 원본은 보존)
    df_results_orig = df_results.copy()  # 리밸런싱용
    cash_weight = max(0.0, 1.0 - float(df_results["추천비중"].sum()))
    cash_price = get_price_at_date(all_data.get(CASH_NAME), current_date, price_col=price_col) or 10000.0
    df_results = pd.concat([df_results, pd.DataFrame([{
        "자산명": CASH_NAME, "티커": CASH_TICKER, "현재가": cash_price,
        "12M고점": None, "고점대비": None, "모멘텀": None,
        "Faber신호": "-", "추천비중": cash_weight, "_is_gold": False
    }])], ignore_index=True)
    df_display = df_results.copy()
    # 금현물 표시명 변경
    df_display.loc[df_display["_is_gold"] == True, "자산명"] = "금현물 (GLD×환율)"
    df_display = df_display.drop(columns=["_is_gold"])
    df_display["현재가"] = df_display["현재가"].apply(lambda x: f"{x:,.0f}원" if pd.notna(x) else "-")
    df_display["12M고점"] = df_display["12M고점"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
    df_display["고점대비"] = df_display["고점대비"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    df_display["모멘텀"] = df_display["모멘텀"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    df_display["추천비중"] = df_display["추천비중"].apply(lambda x: f"{x*100:.0f}%")
    df_display.columns = ["자산명", "티커", "현재가", "12M고점", "고점대비", "모멘텀(참고)", "Faber신호", "추천비중"]
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # 금현물 참고: GLD * USD/KRW
    with st.expander("🥇 금현물 참고 데이터 (GLD × USD/KRW)"):
        try:
            gld_raw = fdr.DataReader('GLD', current_date - relativedelta(months=14), current_date)
            fx_raw = fdr.DataReader('USD/KRW', current_date - relativedelta(months=14), current_date)
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
                gld_fx = pd.concat([gld_adj, fx_raw['Close']], axis=1, keys=['G', 'F']).ffill().bfill().dropna()
                gld_fx['KRW'] = gld_fx['G'] * gld_fx['F']
                gld_monthly = gld_fx['KRW'].resample('ME').last().dropna()
                if len(gld_monthly) > 0:
                    gld_high = float(gld_monthly.max())
                    gld_dist = (gld_krw / gld_high - 1) * 100
                    gld_near = gld_dist >= -5.0
                    st.info(f"📊 GLD×환율 12M고점: ₩{gld_high:,.0f} | 고점대비: {gld_dist:.1f}% | "
                            f"Faber 판단: {'● 투자' if gld_near else '○ 현금'}")
                    st.caption("💡 위 Faber 신호는 ETF(411060) 가격 기준이고, 이 참고 데이터는 GLD×환율 기준입니다.")
        except Exception as e:
            st.warning(f"GLD 데이터 오류: {e}")

    st.markdown("---")
    st.subheader("🏦 3계좌 절세 최적화 리밸런싱")
    st.info("👇 우선순위 배치: 일반=금→코스피 / ISA=미국채→한국채→나스닥100")
    if st.button("🚀 리밸런싱 목표 계산하기", type="primary"):
        with st.spinner("계산 중..."):
            final_df = optimize_allocation(df_results_orig[["자산명","추천비중"]].copy(), bal_gen, bal_isa_a, bal_isa_b)
            st.success("✅ 계산 완료!")
            disp = final_df.copy()
            disp["추천비중"] = disp["추천비중"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
            for c in ["총목표금액","일반계좌","ISA_A","ISA_B"]: disp[c] = disp[c].apply(lambda x: f"{x:,.0f}")
            st.dataframe(disp.style
                .map(lambda x: "background-color: #e6f3ff" if x != "0" else "", subset=["ISA_A","ISA_B"])
                .map(lambda x: "background-color: #fff5e6" if x != "0" else "", subset=["일반계좌"]),
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
        ex.columns = ["자산명","티커","현재가","12M고점","고점대비","모멘텀(참고)","Faber신호","추천비중(%)"]
        ex.to_excel(writer, sheet_name="Faber_A_리밸런싱", index=False)
    st.download_button("📥 엑셀 파일 다운로드", output.getvalue(), f"FaberA_나스닥100_{current_dt.strftime('%Y%m%d')}.xlsx",
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
            all_paths = []
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
                if i < 300:  # 시각화용 샘플
                    all_paths.append(path)
            
            final_values = np.array(final_values)
            max_drawdowns = np.array(max_drawdowns)
            
            # 퍼센타일 경로
            percentile_paths = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                ppath = []
                for m in range(sim_months + 1):
                    vals = [all_paths[i][m] for i in range(len(all_paths))]
                    ppath.append(np.percentile(vals, p))
                percentile_paths[p] = ppath
        
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
            (300_000_000, "3억 이상", None),
            (350_000_000, "3.5억 이상", None),
            (400_000_000, "4억 이상", None),
            (500_000_000, "5억 이상", None),
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
    st.sidebar.markdown("---")

    with st.sidebar.expander("🥇 금 괴리율 차익거래 계산기", expanded=False):
        st.caption("계단식 룰(3~15%) 및 청산 룰(0.5%)")
        krx_val = st.number_input("KRX 금 평가액", value=47998800, step=1000000, key="krx")
        sol_val = st.number_input("SOL 국제금 평가액", value=0, step=1000000, key="sol")
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
                st.write(f"**총 금:** {total_gold:,.0f}원 | **목표 SOL:** {tr*100:.0f}%")
                if trade > 0: st.success(f"KRX 매도 → SOL 매수: {trade:,.0f}원")
                elif trade < 0: st.warning(f"SOL 매도 → KRX 매수: {abs(trade):,.0f}원")
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

    KST = pytz.timezone('Asia/Seoul')
    current_dt = datetime.now(KST).replace(tzinfo=None)
    current_date = normalize_to_date(current_dt)

    if mode.startswith("1."): mode_live_and_rebalance(current_dt, current_date, price_col, inv_start_date, init_capital, hist_profit, bt_start_date)
    elif mode.startswith("2."): mode_strategy_backtest(current_dt, current_date, price_col, bt_start_date)
    else: mode_monte_carlo(current_dt, current_date, price_col, bt_start_date, init_capital)

    st.markdown("---")
    st.caption("ℹ️ 본 대시보드는 과거 데이터 기반이며 투자 권유가 아닙니다.")
    st.caption(f"📌 데이터: FinanceDataReader | 현금: {CASH_NAME} ({CASH_TICKER})")

if __name__ == "__main__":
    main()
