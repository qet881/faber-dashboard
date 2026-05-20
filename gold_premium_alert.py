#!/usr/bin/env python3
"""Telegram alert for KRX gold spot vs international gold premium.

This script calculates the same kind of premium shown on Samsung Securities'
gold spot screen: domestic KRX gold spot KRW/g compared with international gold
converted to KRW/g. It does not use gold ETF prices, Samsung login-only data, or
GoldKimp automated scraping. GoldKimp and Samsung screens are reference/check
values only.

Data timestamps, the USD/KRW quote basis, rounding, and delayed KRX domestic
gold executions can make this result differ slightly from Samsung/GoldKimp.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
import yfinance as yf


TROY_OZ_TO_GRAM = 31.1034768
KST = ZoneInfo("Asia/Seoul")
UTC = ZoneInfo("UTC")
DEFAULT_STATE_FILE = "gold_premium_alert_state.json"
# This caveat is printed in dry-run output and Telegram messages because
# quote timestamps, the USD/KRW source basis, rounding, and delayed domestic
# KRX gold executions can make this calculation slightly different from
# Samsung Securities or GoldKimp reference screens.
NOTE = (
    "참고: 데이터 시각, 환율 기준, 반올림, 국내금 체결가 지연 때문에 "
    "삼성증권/GoldKimp와 소폭 차이날 수 있습니다."
)


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Quote:
    value: float
    source: str
    symbol: str
    timestamp: datetime
    basis: str


@dataclass(frozen=True)
class Calculation:
    domestic: Quote
    international_raw: Quote
    fx: Quote
    international_krw_per_g: float
    premium_pct: float
    segment_key: str
    segment_label: str
    decision: str
    calculated_at: datetime


@dataclass(frozen=True)
class AlertDecision:
    should_send: bool
    reason: str
    previous_segment: str
    current_segment: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KRX domestic gold vs international gold KRW/g premium alert."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print result without Telegram send or state write.")
    parser.add_argument("--force", action="store_true", help="Send even if the segment did not change; also bypass market-hours filter.")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="State JSON path.")
    parser.add_argument(
        "--domestic-source",
        choices=["auto", "krx", "naver", "manual"],
        default="auto",
        help="Domestic KRX gold KRW/g provider.",
    )
    parser.add_argument("--domestic-price", type=float, help="Manual domestic KRX gold KRW/g price.")
    parser.add_argument("--ignore-market-hours", action="store_true", help="Run alert logic outside KST market hours.")
    return parser.parse_args()


def now_kst() -> datetime:
    return datetime.now(KST)


def is_market_hours(dt: datetime) -> bool:
    local_dt = dt.astimezone(KST)
    return local_dt.weekday() < 5 and time(9, 0) <= local_dt.time() <= time(15, 30)


def parse_number(value: Any) -> float:
    if value is None:
        raise ProviderError("가격 값이 비어 있습니다.")
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace(",", "")
    if not cleaned or cleaned in {"-", "N/A", "nan"}:
        raise ProviderError(f"가격 값을 숫자로 해석할 수 없습니다: {value!r}")
    return float(cleaned)


def latest_yahoo_quote(symbols: list[str], source_name: str, basis: str) -> Quote:
    errors: list[str] = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            for period, interval in [("1d", "1m"), ("5d", "15m"), ("10d", "1d")]:
                hist = ticker.history(period=period, interval=interval)
                if hist is None or hist.empty or "Close" not in hist:
                    continue
                closes = hist["Close"].dropna()
                if closes.empty:
                    continue
                ts = closes.index[-1]
                if hasattr(ts, "to_pydatetime"):
                    dt = ts.to_pydatetime()
                else:
                    dt = datetime.now(UTC)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return Quote(
                    value=float(closes.iloc[-1]),
                    source=source_name,
                    symbol=symbol,
                    timestamp=dt.astimezone(KST),
                    basis=f"{basis}; yfinance history {period}/{interval}",
                )
            errors.append(f"{symbol}: yfinance history가 비어 있습니다.")
        except Exception as exc:  # yfinance raises several non-public exception types.
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
    raise ProviderError("; ".join(errors))


def fetch_international_yahoo() -> Quote:
    return latest_yahoo_quote(
        ["GC=F", "XAUUSD=X"],
        source_name="yahoo",
        basis="국제금 원자료 USD/troy oz",
    )


def fetch_fx_yahoo() -> Quote:
    return latest_yahoo_quote(
        ["USDKRW=X"],
        source_name="yahoo",
        basis="USD/KRW Yahoo Finance 환율",
    )


def fetch_domestic_manual(price: float | None) -> Quote:
    if price is None:
        raise ProviderError("--domestic-source manual 사용 시 --domestic-price가 필요합니다.")
    if price <= 0:
        raise ProviderError("--domestic-price는 0보다 커야 합니다.")
    return Quote(
        value=float(price),
        source="manual",
        symbol="KRD040200002",
        timestamp=now_kst(),
        basis="수동 입력 KRX 금현물 원/g 가격",
    )


def fetch_domestic_krx() -> Quote:
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201060201",
    }
    payload = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT15201",
        "locale": "ko_KR",
        "isuCd": "KRD040200002",
        "csvxls_isNo": "false",
    }
    try:
        session.get("https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd", headers=headers, timeout=15)
        response = session.post(
            "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
            headers=headers,
            data=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"KRX 요청 실패: {type(exc).__name__}: {exc}") from exc

    body = response.text.strip()
    if response.status_code != 200:
        raise ProviderError(f"KRX HTTP {response.status_code}: {body[:200]}")
    if body.upper() == "LOGOUT" or "로그인" in body:
        raise ProviderError("KRX 공개 JSON 엔드포인트가 로그인/세션 요구로 차단되었습니다.")

    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(f"KRX JSON 파싱 실패: {body[:200]}") from exc

    price = parse_number(data.get("TDD_CLSPRC") or data.get("CLSPRC") or data.get("TRADE_PRICE"))
    timestamp = now_kst()
    traded_at = data.get("TRD_DD") or data.get("BAS_DD")
    basis = "KRX 정보데이터시스템 금 99.99_1Kg(KRD040200002) 원/g"
    if traded_at:
        basis += f"; 거래일 {traded_at}"
    return Quote(
        value=price,
        source="krx",
        symbol="KRD040200002",
        timestamp=timestamp,
        basis=basis,
    )


def fetch_domestic_naver() -> Quote:
    """Try Naver without accepting ETF prices as a substitute.

    Naver frequently exposes stock/ETF endpoints, but this alert requires the KRX
    gold spot KRW/g instrument. If Naver only returns ETFs such as ACE KRX금현물
    or KODEX 금액티브, this provider intentionally fails.
    """
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8"}
    queries = ["금 99.99_1Kg", "금현물 04020000", "KRD040200002"]
    errors: list[str] = []
    for query in queries:
        try:
            response = requests.get(
                "https://m.stock.naver.com/api/json/search/searchListJson.nhn",
                headers=headers,
                params={"keyword": query},
                timeout=15,
            )
            if response.status_code != 200:
                errors.append(f"{query}: HTTP {response.status_code}")
                continue
            data = response.json()
            rows = data.get("result", {}).get("d", [])
            for row in rows:
                code = str(row.get("cd", ""))
                name = str(row.get("nm", ""))
                is_etf = bool(row.get("etf"))
                if is_etf or "ETF" in name.upper() or "ETN" in name.upper():
                    continue
                if code in {"04020000", "KRD040200002"} or ("금" in name and "99.99" in name):
                    return Quote(
                        value=parse_number(row.get("nv")),
                        source="naver",
                        symbol=code or "KRD040200002",
                        timestamp=now_kst(),
                        basis=f"Naver search result for KRX gold spot; query={query}",
                    )
            errors.append(f"{query}: KRX 금현물 원/g 종목을 찾지 못했습니다(ETF/ETN은 제외).")
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{query}: {type(exc).__name__}: {exc}")
    raise ProviderError("; ".join(errors))


def fetch_domestic(source: str, manual_price: float | None) -> Quote:
    if source == "manual":
        return fetch_domestic_manual(manual_price)
    if source == "krx":
        return fetch_domestic_krx()
    if source == "naver":
        return fetch_domestic_naver()

    errors: list[str] = []
    for provider_name, fetcher in [
        ("krx", lambda: fetch_domestic_krx()),
        ("naver", lambda: fetch_domestic_naver()),
    ]:
        try:
            return fetcher()
        except ProviderError as exc:
            errors.append(f"{provider_name}: {exc}")
    raise ProviderError(
        "domestic-source auto 실패. "
        + " | ".join(errors)
        + " | 공개 국내금 소스가 막혀 있으면 "
        "--domestic-source manual --domestic-price 145000 처럼 수동 입력으로 테스트하세요."
    )


def classify_premium(premium_pct: float) -> tuple[str, str, str]:
    if premium_pct <= 0.5:
        return "clear", "청산 구간(<=0.5%)", "청산"
    if premium_pct < 3:
        return "watch", "관망 구간(0.5~3%)", "관망"
    if premium_pct < 6:
        return "convert_20", "전환 구간(3~6%)", "20% 전환"
    if premium_pct < 9:
        return "convert_40", "전환 구간(6~9%)", "40% 전환"
    if premium_pct < 12:
        return "convert_60", "전환 구간(9~12%)", "60% 전환"
    if premium_pct < 15:
        return "convert_80", "전환 구간(12~15%)", "80% 전환"
    return "convert_100", "전환 구간(15% 이상)", "100% 전환"


def calculate(args: argparse.Namespace) -> Calculation:
    domestic = fetch_domestic(args.domestic_source, args.domestic_price)
    international_raw = fetch_international_yahoo()
    fx = fetch_fx_yahoo()
    international_krw_per_g = international_raw.value * fx.value / TROY_OZ_TO_GRAM
    if international_krw_per_g <= 0 or not math.isfinite(international_krw_per_g):
        raise ProviderError(f"국제금 원/g 환산가격이 비정상입니다: {international_krw_per_g}")
    premium_pct = (domestic.value / international_krw_per_g - 1) * 100
    segment_key, segment_label, decision = classify_premium(premium_pct)
    return Calculation(
        domestic=domestic,
        international_raw=international_raw,
        fx=fx,
        international_krw_per_g=international_krw_per_g,
        premium_pct=premium_pct,
        segment_key=segment_key,
        segment_label=segment_label,
        decision=decision,
        calculated_at=now_kst(),
    )


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"상태 파일을 읽을 수 없습니다: {path} ({exc})") from exc


def write_state(path: Path, state: dict[str, Any]) -> bool:
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    rendered = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if previous == rendered:
        return False
    path.write_text(rendered, encoding="utf-8")
    return True


def decide_alert(calc: Calculation, state: dict[str, Any], force: bool) -> AlertDecision:
    previous_segment = str(state.get("last_segment_label") or "없음")
    previous_key = state.get("last_segment_key")

    if force:
        return AlertDecision(True, "--force 지정", previous_segment, calc.segment_label)
    if previous_key is None:
        if calc.segment_key == "watch":
            return AlertDecision(False, "초기 상태가 관망 구간이라 상태만 기록", previous_segment, calc.segment_label)
        return AlertDecision(True, "초기 상태가 알림 대상 구간", previous_segment, calc.segment_label)
    if previous_key != calc.segment_key:
        return AlertDecision(True, "알림 구간 변경", previous_segment, calc.segment_label)
    return AlertDecision(False, "같은 구간 중복 알림 방지", previous_segment, calc.segment_label)


def build_state(calc: Calculation) -> dict[str, Any]:
    return {
        "last_segment_key": calc.segment_key,
        "last_segment_label": calc.segment_label,
        "last_premium_pct": round(calc.premium_pct, 4),
        "last_domestic_krw_per_g": round(calc.domestic.value, 2),
        "last_international_krw_per_g": round(calc.international_krw_per_g, 2),
        "updated_at_kst": calc.calculated_at.isoformat(),
        "data_sources": {
            "domestic": calc.domestic.source,
            "international": calc.international_raw.source,
            "fx": calc.fx.source,
        },
    }


def fmt_krw(value: float) -> str:
    return f"{value:,.0f}원/g"


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def build_message(calc: Calculation, alert: AlertDecision) -> str:
    data_sources = (
        f"국내={calc.domestic.source}({calc.domestic.symbol}), "
        f"국제금={calc.international_raw.source}({calc.international_raw.symbol}), "
        f"환율={calc.fx.source}({calc.fx.symbol})"
    )
    return "\n".join(
        [
            "[국내금-국제금 괴리율 알림]",
            f"국내금 가격: {fmt_krw(calc.domestic.value)}",
            f"국제금 환산가격: {fmt_krw(calc.international_krw_per_g)}",
            f"국제금 원자료: {fmt_num(calc.international_raw.value, 2)} USD/troy oz",
            f"USD/KRW 환율: {fmt_num(calc.fx.value, 2)}",
            f"현재 괴리율: {calc.premium_pct:.2f}%",
            f"이전 알림 구간: {alert.previous_segment}",
            f"현재 구간: {alert.current_segment}",
            f"차익거래 판단: {calc.decision}",
            f"데이터 소스: {data_sources}",
            f"기준 시각: 계산={fmt_dt(calc.calculated_at)}, 국내={fmt_dt(calc.domestic.timestamp)}, 국제={fmt_dt(calc.international_raw.timestamp)}, 환율={fmt_dt(calc.fx.timestamp)}",
            f"발송 사유: {alert.reason}",
            NOTE,
        ]
    )


def print_calculation(calc: Calculation, alert: AlertDecision, will_send: bool) -> None:
    print("[계산 결과]")
    print(f"- 국내금 가격: {fmt_krw(calc.domestic.value)}")
    print(f"- 국제금 환산가격: {fmt_krw(calc.international_krw_per_g)}")
    print(f"- 국제금 원자료: {fmt_num(calc.international_raw.value, 2)} USD/troy oz")
    print(f"- USD/KRW 환율: {fmt_num(calc.fx.value, 2)}")
    print(f"- 괴리율: {calc.premium_pct:.2f}%")
    print(f"- 이전 알림 구간: {alert.previous_segment}")
    print(f"- 현재 구간: {alert.current_segment}")
    print(f"- 차익거래 판단: {calc.decision}")
    print(f"- 데이터 소스: 국내={calc.domestic.source}, 국제금={calc.international_raw.source}, 환율={calc.fx.source}")
    print(f"- 기준 시각: 계산={fmt_dt(calc.calculated_at)}, 국내={fmt_dt(calc.domestic.timestamp)}, 국제={fmt_dt(calc.international_raw.timestamp)}, 환율={fmt_dt(calc.fx.timestamp)}")
    print(f"- 예정 동작: {'텔레그램 발송' if will_send else '발송 없음'} ({alert.reason})")
    print(f"- {NOTE}")
    if will_send:
        print("\n[발송 예정 메시지]")
        print(build_message(calc, alert))


def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "텔레그램 발송 모드에는 TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID 환경변수가 필요합니다. "
            "GitHub Actions에서는 두 값을 Secrets로 설정하세요."
        )
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"텔레그램 API 요청 실패: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"텔레그램 API HTTP {response.status_code}: {response.text[:500]}")


def main() -> int:
    args = parse_args()
    state_path = Path(args.state_file)
    current_time = now_kst()
    bypass_market_hours = args.dry_run or args.force or args.ignore_market_hours

    if not bypass_market_hours and not is_market_hours(current_time):
        print(f"KST 장중(평일 09:00~15:30)이 아니므로 알림 없이 정상 종료합니다. 현재: {fmt_dt(current_time)}")
        return 0

    try:
        state = read_state(state_path)
        calc = calculate(args)
        alert = decide_alert(calc, state, args.force)
        print_calculation(calc, alert, alert.should_send)

        if args.dry_run:
            print("\n--dry-run: 텔레그램 발송과 상태 파일 저장을 건너뜁니다.")
            return 0

        if alert.should_send:
            send_telegram(build_message(calc, alert))
            print("텔레그램 알림을 발송했습니다.")

        new_state = build_state(calc)
        changed = write_state(state_path, new_state)
        print(f"상태 파일: {state_path} ({'변경됨' if changed else '변경 없음'})")
        return 0
    except Exception as exc:
        print(f"실패: {exc}", file=sys.stderr)
        print(
            "국내금 공개 데이터 소스가 막혀 있으면 "
            "--domestic-source manual --domestic-price 145000 옵션으로 계산/알림 로직을 테스트할 수 있습니다.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
