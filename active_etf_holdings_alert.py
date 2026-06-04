#!/usr/bin/env python3
"""Telegram report for active ETF holding changes from official issuer data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, time
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


KST = ZoneInfo("Asia/Seoul")
DEFAULT_STATE_FILE = "active_etf_holdings_alert_state.json"
FOOTER = "공식 운용사 PDF/엑셀 공시 기준 자동 요약이며, 투자 권유가 아닙니다."
DEFAULT_OPENROUTER_MODEL = "google/gemma-3-12b-it:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class EtfConfig:
    ticker: str
    name: str
    issuer: str
    source_kind: str
    source_id: str
    page_url: str


@dataclass(frozen=True)
class Holding:
    code: str
    name: str
    weight: float
    quantity: str = ""
    value: str = ""
    rank: int = 0

    @property
    def key(self) -> str:
        code = normalize_code(self.code)
        if code:
            return code
        return normalize_name(self.name)


@dataclass(frozen=True)
class Snapshot:
    ticker: str
    name: str
    issuer: str
    disclosure_date: str
    source_url: str
    source_hash: str
    holdings: list[Holding]


@dataclass(frozen=True)
class SourceResult:
    ok: bool
    status: str
    config: EtfConfig
    snapshot: Snapshot | None = None
    error: str = ""


@dataclass(frozen=True)
class DiffResult:
    added: list[Holding]
    removed: list[Holding]
    top10_entries: list[Holding]
    top10_exits: list[Holding]
    rank_changes: list[tuple[Holding, Holding]]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.top10_entries or self.top10_exits or self.rank_changes)


ETFS = [
    EtfConfig(
        ticker="0015B0",
        name="KoAct 미국나스닥성장기업액티브",
        issuer="KoAct",
        source_kind="koact",
        source_id="2ETFQ1",
        page_url="https://www.samsungactive.co.kr/etf/view.do?id=2ETFQ1",
    ),
    EtfConfig(
        ticker="426030",
        name="TIME 미국나스닥100액티브",
        issuer="TIME",
        source_kind="time",
        source_id="2",
        page_url="https://timeetf.co.kr/m11_view.php?idx=2",
    ),
    EtfConfig(
        ticker="0193G0",
        name="KOACT 코스피액티브",
        issuer="KoAct",
        source_kind="koact",
        source_id="2ETFV2",
        page_url="https://www.samsungactive.co.kr/etf/view.do?id=2ETFV2",
    ),
    EtfConfig(
        ticker="385720",
        name="TIME 코스피액티브",
        issuer="TIME",
        source_kind="time",
        source_id="11",
        page_url="https://timeetf.co.kr/m11_view.php?idx=11",
    ),
]


class SourceError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active ETF holdings change Telegram report.")
    parser.add_argument("--dry-run", action="store_true", help="Print report without Telegram send or state write.")
    parser.add_argument("--force-send", action="store_true", help="Send even if the rendered report was already sent.")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="State JSON path.")
    parser.add_argument("--fixture-dir", help="Load <ticker>_latest.json fixtures instead of live sources.")
    parser.add_argument("--ignore-time-gate", action="store_true", help="Run outside scheduled KST report windows.")
    return parser.parse_args(argv)


def now_kst() -> datetime:
    return datetime.now(KST)


def is_report_window(dt: datetime) -> bool:
    local_dt = dt.astimezone(KST)
    return local_dt.weekday() < 5 and local_dt.time().replace(second=0, microsecond=0) in {
        time(8, 10),
        time(8, 40),
        time(9, 10),
    }


def normalize_code(value: str) -> str:
    value = str(value or "").strip()
    if not value or value.lower() == "nan":
        return ""
    value = re.sub(r"\s+", " ", value)
    return value.upper()


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def parse_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace(",", "").replace("%", "")
    if not cleaned or cleaned in {"-", "nan", "None"}:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def stable_hash(data: Any) -> str:
    rendered = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def normalize_date(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def canonical_holdings(holdings: list[Holding]) -> list[Holding]:
    filtered = [h for h in holdings if h.name and h.key]
    sorted_holdings = sorted(filtered, key=lambda h: (-h.weight, h.name, h.code))
    return [
        Holding(h.code, h.name, h.weight, h.quantity, h.value, rank=index)
        for index, h in enumerate(sorted_holdings, start=1)
    ]


def snapshot_to_json(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "ticker": snapshot.ticker,
        "name": snapshot.name,
        "issuer": snapshot.issuer,
        "disclosure_date": snapshot.disclosure_date,
        "source_url": snapshot.source_url,
        "source_hash": snapshot.source_hash,
        "holdings": [
            {
                "code": h.code,
                "name": h.name,
                "weight": h.weight,
                "quantity": h.quantity,
                "value": h.value,
                "rank": h.rank,
            }
            for h in snapshot.holdings
        ],
    }


def snapshot_from_json(data: dict[str, Any], config: EtfConfig) -> Snapshot:
    holdings = canonical_holdings(
        [
            Holding(
                code=str(row.get("code") or ""),
                name=str(row.get("name") or ""),
                weight=parse_float(row.get("weight")),
                quantity=str(row.get("quantity") or ""),
                value=str(row.get("value") or ""),
            )
            for row in data.get("holdings", [])
        ]
    )
    if not holdings:
        raise SourceError("fixture holdings are empty")
    source_url = str(data.get("source_url") or config.page_url)
    disclosure_date = normalize_date(str(data.get("disclosure_date") or ""))
    return Snapshot(
        ticker=config.ticker,
        name=str(data.get("name") or config.name),
        issuer=str(data.get("issuer") or config.issuer),
        disclosure_date=disclosure_date,
        source_url=source_url,
        source_hash=str(data.get("source_hash") or stable_hash([h.__dict__ for h in holdings])),
        holdings=holdings,
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"snapshots": {}, "last_messages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict[str, Any]) -> bool:
    rendered = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous == rendered:
        return False
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(rendered, encoding="utf-8")
    tmp_path.replace(path)
    return True


def request_get(url: str, *, accept: str = "*/*") -> requests.Response:
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": accept,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
        },
        timeout=25,
    )
    if response.status_code != 200:
        raise SourceError(f"HTTP {response.status_code}: {response.text[:200]}")
    return response


def fetch_time(config: EtfConfig) -> Snapshot:
    excel_url = f"https://timeetf.co.kr/pdf_excel.php?idx={config.source_id}&"
    response = request_get(
        excel_url,
        accept="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
    )
    df = pd.read_excel(BytesIO(response.content), header=None)
    holdings = parse_time_dataframe(df)
    if not holdings:
        raise SourceError("TIME holdings table parsed empty")
    source_hash = stable_hash(
        {
            "url": excel_url,
            "bytes": hashlib.sha256(response.content).hexdigest(),
            "holdings": [h.__dict__ for h in holdings],
        }
    )
    return Snapshot(
        ticker=config.ticker,
        name=config.name,
        issuer=config.issuer,
        disclosure_date=normalize_date(extract_time_disclosure_date(config)),
        source_url=excel_url,
        source_hash=source_hash,
        holdings=holdings,
    )


def parse_time_dataframe(df: pd.DataFrame) -> list[Holding]:
    header_row = None
    for idx, row in df.iterrows():
        labels = [str(value).strip() for value in row.tolist()]
        if "종목명" in labels and "비중(%)" in labels:
            header_row = idx
            break
    if header_row is None:
        raise SourceError("TIME holdings header not found")

    labels = [str(value).strip() for value in df.iloc[header_row].tolist()]
    code_col = labels.index("종목코드")
    name_col = labels.index("종목명")
    qty_col = labels.index("수량")
    value_col = labels.index("평가금액(원)")
    weight_col = labels.index("비중(%)")
    holdings: list[Holding] = []
    for _, row in df.iloc[header_row + 1 :].iterrows():
        name = str(row.iloc[name_col] or "").strip()
        if not name or name.lower() == "nan":
            continue
        holdings.append(
            Holding(
                code=normalize_code(str(row.iloc[code_col] or "")),
                name=name,
                weight=parse_float(row.iloc[weight_col]),
                quantity=str(row.iloc[qty_col] or "").strip(),
                value=str(row.iloc[value_col] or "").strip(),
            )
        )
    return canonical_holdings(holdings)


def extract_time_disclosure_date(config: EtfConfig) -> str:
    try:
        html = request_get(config.page_url, accept="text/html,*/*").text
    except SourceError:
        return ""
    match = re.search(r'id=["\']pdfDate["\'][^>]*value=["\']([^"\']+)["\']', html)
    return match.group(1) if match else ""


def fetch_koact(config: EtfConfig) -> Snapshot:
    product_url = f"https://www.samsungactive.co.kr/api/v1/product/etf/{config.source_id}.do"
    product = request_get(product_url, accept="application/json,*/*").json()
    pdf = product.get("pdf") or {}
    disclosure_date = normalize_date(str(pdf.get("gijunYMD") or ""))
    if not disclosure_date:
        raise SourceError("KoAct pdf.gijunYMD missing")

    detail_url = (
        "https://www.samsungactive.co.kr"
        f"/api/v1/product/etf-pdf/{config.source_id}.do?gijunYMD={disclosure_date}"
    )
    detail = request_get(detail_url, accept="application/json,*/*").json()
    pdf_detail = detail.get("pdf") or {}
    rows = pdf_detail.get("list") or pdf.get("list") or []
    holdings = parse_koact_rows(rows)
    if not holdings:
        raise SourceError("KoAct holdings table parsed empty")
    download_url = pdf_detail.get("pdfExcelDownloadUrl") or pdf.get("pdfExcelDownloadUrl") or detail_url
    if str(download_url).startswith("/"):
        download_url = "https://www.samsungactive.co.kr" + str(download_url)
    return Snapshot(
        ticker=config.ticker,
        name=config.name,
        issuer=config.issuer,
        disclosure_date=disclosure_date,
        source_url=str(download_url),
        source_hash=stable_hash({"url": detail_url, "pdf": pdf_detail}),
        holdings=holdings,
    )


def parse_koact_rows(rows: list[dict[str, Any]]) -> list[Holding]:
    holdings: list[Holding] = []
    for row in rows:
        name = str(row.get("secNm") or "").strip()
        code = normalize_code(str(row.get("itmNo") or row.get("secId") or ""))
        if not name or name in {"설정현금액"}:
            continue
        holdings.append(
            Holding(
                code=code,
                name=name,
                weight=parse_float(row.get("ratio") or row.get("wgt")),
                quantity=str(row.get("applyQ") or ""),
                value=str(row.get("evalA") or ""),
            )
        )
    return canonical_holdings(holdings)


def fetch_live(config: EtfConfig) -> SourceResult:
    try:
        if config.source_kind == "time":
            return SourceResult(True, "ok", config, fetch_time(config))
        if config.source_kind == "koact":
            return SourceResult(True, "ok", config, fetch_koact(config))
        raise SourceError(f"unknown source kind: {config.source_kind}")
    except Exception as exc:
        return SourceResult(False, "source_error", config, error=str(exc))


def fetch_fixture(config: EtfConfig, fixture_dir: Path, suffix: str = "latest") -> SourceResult:
    path = fixture_dir / f"{config.ticker}_{suffix}.json"
    if not path.exists():
        return SourceResult(False, "fixture_missing", config, error=f"{path} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SourceResult(True, "ok", config, snapshot_from_json(data, config))
    except Exception as exc:
        return SourceResult(False, "fixture_error", config, error=str(exc))


def holdings_by_key(holdings: list[Holding]) -> dict[str, Holding]:
    return {holding.key: holding for holding in holdings if holding.key}


def diff_snapshots(previous: Snapshot | None, latest: Snapshot) -> DiffResult:
    if previous is None:
        return DiffResult(added=latest.holdings, removed=[], top10_entries=latest.holdings[:10], top10_exits=[], rank_changes=[])

    prev_map = holdings_by_key(previous.holdings)
    latest_map = holdings_by_key(latest.holdings)
    added = [latest_map[key] for key in latest_map.keys() - prev_map.keys()]
    removed = [prev_map[key] for key in prev_map.keys() - latest_map.keys()]

    prev_top = {h.key: h for h in previous.holdings[:10]}
    latest_top = {h.key: h for h in latest.holdings[:10]}
    top10_entries = [latest_top[key] for key in latest_top.keys() - prev_top.keys()]
    top10_exits = [prev_top[key] for key in prev_top.keys() - latest_top.keys()]
    rank_changes = [
        (prev_top[key], latest_top[key])
        for key in latest_top.keys() & prev_top.keys()
        if latest_top[key].rank != prev_top[key].rank
    ]
    return DiffResult(
        added=sorted(added, key=lambda h: h.rank),
        removed=sorted(removed, key=lambda h: h.rank),
        top10_entries=sorted(top10_entries, key=lambda h: h.rank),
        top10_exits=sorted(top10_exits, key=lambda h: h.rank),
        rank_changes=sorted(rank_changes, key=lambda pair: pair[1].rank),
    )


THEME_RULES = [
    ("AI/반도체", ["NVIDIA", "NVDA", "AMD", "BROADCOM", "AVGO", "MICRON", "MU", "SK하이닉스", "삼성전자", "반도체"]),
    ("클라우드/빅테크", ["MICROSOFT", "MSFT", "ALPHABET", "GOOGL", "AMAZON", "AMZN", "META", "APPLE", "AAPL"]),
    ("전력/인프라", ["LS", "효성중공업", "HD현대중공업", "두산", "산일전기", "BLOOM"]),
    ("자동차/전장", ["현대차", "현대모비스", "TESLA", "TSLA", "삼성전기"]),
    ("금융", ["KB금융", "삼성화재", "은행", "FINANCIAL"]),
    ("현금", ["현금", "예금", "CASH", "KRD0100"]),
]


def infer_themes(holdings: list[Holding]) -> list[str]:
    text = " ".join(f"{h.code} {h.name}".upper() for h in holdings)
    themes = [theme for theme, needles in THEME_RULES if any(needle.upper() in text for needle in needles)]
    return themes[:3]


def fmt_holding(holding: Holding) -> str:
    code = f"({holding.code})" if holding.code else ""
    return f"{holding.rank}위 {holding.name}{code} {holding.weight:.2f}%"


def changed_holdings_for_comment(diff: DiffResult) -> list[Holding]:
    return diff.added + diff.removed + diff.top10_entries + diff.top10_exits + [new for _, new in diff.rank_changes]


def fallback_one_liner(diff: DiffResult) -> str:
    themes = infer_themes(changed_holdings_for_comment(diff))
    if themes:
        return "변화 종목 기준으로 " + ", ".join(themes) + " 쪽 노출 조정이 보입니다."
    return "특정 테마로 단정하기 어려운 보유종목 재배치입니다."


def diff_prompt_payload(snapshot: Snapshot, diff: DiffResult) -> dict[str, Any]:
    return {
        "etf": {"ticker": snapshot.ticker, "name": snapshot.name, "date": snapshot.disclosure_date},
        "added": [fmt_holding(h) for h in diff.added[:8]],
        "removed": [fmt_holding(h) for h in diff.removed[:8]],
        "top10_entries": [fmt_holding(h) for h in diff.top10_entries[:8]],
        "top10_exits": [fmt_holding(h) for h in diff.top10_exits[:8]],
        "rank_changes": [f"{new.name} {old.rank}->{new.rank}위" for old, new in diff.rank_changes[:8]],
    }


def sanitize_one_liner(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    cleaned = re.sub(r"^(한줄평|해석)\s*[:：]\s*", "", cleaned)
    if len(cleaned) > 90:
        cleaned = cleaned[:87].rstrip() + "..."
    return cleaned


def generate_gemma_one_liner(snapshot: Snapshot, diff: DiffResult) -> str | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/qet881/faber-dashboard",
                "X-OpenRouter-Title": "faber-dashboard active ETF report",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "너는 액티브 ETF 보유종목 변화 리포트를 쓰는 한국어 애널리스트다. "
                            "제공된 변화만 근거로 ETF 매니저가 어디에 무게를 둔 것처럼 보이는지 "
                            "한 문장으로만 말한다. 투자 권유, 확정적 단정, 상투적인 표현은 피한다."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "아래 ETF 보유종목 변화만 보고 45자 안팎의 한국어 한줄평 1개만 써줘. "
                            "접두어 없이 문장만 출력해.\n"
                            + json.dumps(diff_prompt_payload(snapshot, diff), ensure_ascii=False)
                        ),
                    },
                ],
                "temperature": 0.4,
                "max_tokens": 90,
            },
            timeout=15,
        )
        if response.status_code != 200:
            return None
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        return sanitize_one_liner(content) or None
    except Exception:
        return None


def one_liner(snapshot: Snapshot, diff: DiffResult) -> str:
    return generate_gemma_one_liner(snapshot, diff) or fallback_one_liner(diff)


def render_etf_section(snapshot: Snapshot, diff: DiffResult) -> str:
    lines = [f"[{snapshot.ticker}] {snapshot.name}", f"기준일: {snapshot.disclosure_date or '확인 불가'}"]
    lines.append("한줄평: " + one_liner(snapshot, diff))

    if diff.added:
        lines.append("신규 편입: " + "; ".join(fmt_holding(h) for h in diff.added[:5]))
    if diff.removed:
        lines.append("제외/소멸: " + "; ".join(fmt_holding(h) for h in diff.removed[:5]))
    if diff.top10_entries:
        lines.append("Top10 진입: " + "; ".join(fmt_holding(h) for h in diff.top10_entries[:5]))
    if diff.top10_exits:
        lines.append("Top10 이탈: " + "; ".join(fmt_holding(h) for h in diff.top10_exits[:5]))
    if diff.rank_changes:
        changes = [f"{new.name} {old.rank}->{new.rank}위" for old, new in diff.rank_changes[:5]]
        lines.append("Top10 순위 변화: " + "; ".join(changes))
    lines.append(f"출처: {snapshot.source_url}")
    return "\n".join(lines)


def build_report(
    latest_results: list[SourceResult],
    previous_snapshots: dict[str, Snapshot],
    run_dt: datetime,
) -> tuple[str | None, dict[str, DiffResult], list[SourceResult], bool]:
    valid_results = [result for result in latest_results if result.ok and result.snapshot is not None]
    failed_results = [result for result in latest_results if not result.ok]
    if not valid_results:
        details = "; ".join(f"{r.config.ticker} {r.status}: {r.error}" for r in failed_results)
        return f"액티브 ETF 공식 소스가 모두 유효하지 않아 스냅샷을 갱신하지 않았습니다.\n{details}\n{FOOTER}", {}, failed_results, False

    diffs: dict[str, DiffResult] = {}
    changed_sections: list[str] = []
    for result in valid_results:
        snapshot = result.snapshot
        assert snapshot is not None
        previous = previous_snapshots.get(snapshot.ticker)
        diff = diff_snapshots(previous, snapshot)
        diffs[snapshot.ticker] = diff
        if diff.has_changes:
            changed_sections.append(render_etf_section(snapshot, diff))

    all_four_valid = len(valid_results) == len(ETFS)
    if changed_sections:
        header = f"액티브 ETF 보유종목 변화 리포트 ({run_dt.astimezone(KST):%Y-%m-%d %H:%M KST})"
        status_lines = []
        if failed_results:
            status_lines.append("소스 실패: " + "; ".join(f"{r.config.ticker} {r.status}" for r in failed_results))
        return "\n\n".join([header, *changed_sections, *status_lines, FOOTER]), diffs, failed_results, True
    if all_four_valid:
        return "4개 모두 변화없음.", diffs, failed_results, True
    return None, diffs, failed_results, True


def previous_snapshots_from_state(state: dict[str, Any]) -> dict[str, Snapshot]:
    snapshots: dict[str, Snapshot] = {}
    raw = state.get("snapshots") or {}
    configs = {config.ticker: config for config in ETFS}
    for ticker, data in raw.items():
        config = configs.get(ticker)
        if config and isinstance(data, dict):
            try:
                snapshots[ticker] = snapshot_from_json(data, config)
            except Exception:
                continue
    return snapshots


def update_state(
    state: dict[str, Any],
    latest_results: list[SourceResult],
    report: str | None,
    message_key: str | None,
    run_dt: datetime,
    write_snapshots: bool,
) -> dict[str, Any]:
    new_state = dict(state)
    snapshots = dict(new_state.get("snapshots") or {})
    statuses = {}
    for result in latest_results:
        statuses[result.config.ticker] = {"status": result.status, "error": result.error}
        if write_snapshots and result.ok and result.snapshot:
            snapshots[result.config.ticker] = snapshot_to_json(result.snapshot)
    new_state["snapshots"] = snapshots
    new_state["last_status"] = statuses
    new_state["updated_at_kst"] = run_dt.astimezone(KST).isoformat()
    if report and message_key:
        new_state["last_message_key"] = message_key
        new_state["last_message_hash"] = stable_hash(report)
        new_state["last_sent_at_kst"] = run_dt.astimezone(KST).isoformat()
    return new_state


def build_message_key(report: str, latest_results: list[SourceResult], run_dt: datetime) -> str:
    source_hashes = [
        f"{r.config.ticker}:{r.snapshot.source_hash}"
        for r in latest_results
        if r.ok and r.snapshot is not None
    ]
    stable_report = re.sub(
        r"액티브 ETF 보유종목 변화 리포트 \([^)]+\)",
        "액티브 ETF 보유종목 변화 리포트",
        report,
    )
    return stable_hash(
        {
            "date": run_dt.astimezone(KST).date().isoformat(),
            "sources": sorted(source_hashes),
            "report": stable_report,
        }
    )


def split_telegram_message(message: str, limit: int = 3900) -> list[str]:
    if len(message) <= limit:
        return [message]
    chunks: list[str] = []
    current = ""
    for block in message.split("\n\n"):
        candidate = block if not current else current + "\n\n" + block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(block) <= limit:
            current = block
            continue
        for index in range(0, len(block), limit):
            chunks.append(block[index : index + limit])
    if current:
        chunks.append(current)
    return chunks


def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID 환경변수가 필요합니다.")
    chunks = split_telegram_message(message)
    for index, chunk in enumerate(chunks, start=1):
        text = chunk if len(chunks) == 1 else f"({index}/{len(chunks)})\n{chunk}"
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=20,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Telegram HTTP {response.status_code}: {response.text[:500]}")


def collect_results(args: argparse.Namespace) -> list[SourceResult]:
    if args.fixture_dir:
        fixture_dir = Path(args.fixture_dir)
        return [fetch_fixture(config, fixture_dir) for config in ETFS]
    return [fetch_live(config) for config in ETFS]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dt = now_kst()
    if not (args.dry_run or args.ignore_time_gate or is_report_window(run_dt)):
        print(f"정해진 발송 시각이 아니므로 종료합니다. 현재: {run_dt:%Y-%m-%d %H:%M:%S %Z}")
        return 0

    state_path = Path(args.state_file)
    state = load_state(state_path)
    results = collect_results(args)
    previous = previous_snapshots_from_state(state)
    report, _, failures, should_update_snapshots = build_report(results, previous, run_dt)

    for result in results:
        if result.ok and result.snapshot:
            print(f"{result.config.ticker}: ok {result.snapshot.disclosure_date} holdings={len(result.snapshot.holdings)}")
        else:
            print(f"{result.config.ticker}: {result.status} {result.error}")

    if not report:
        print("변화가 없는 ETF는 조용히 생략합니다.")
        if not args.dry_run:
            new_state = update_state(state, results, None, None, run_dt, should_update_snapshots)
            changed = write_state(state_path, new_state)
            print(f"상태 파일: {state_path} ({'변경됨' if changed else '변경 없음'})")
        return 0

    message_key = build_message_key(report, results, run_dt)
    duplicate = state.get("last_message_key") == message_key
    print("\n[발송 예정 메시지]")
    print(report)

    if args.dry_run:
        print("\n--dry-run: 텔레그램 발송과 상태 파일 저장을 건너뜁니다.")
        return 0
    sent = False
    if duplicate and not args.force_send:
        print("동일 메시지 중복 발송을 건너뜁니다.")
    else:
        send_telegram(report)
        sent = True
        print("텔레그램 알림을 발송했습니다.")

    if failures and not any(result.ok for result in results):
        should_update_snapshots = False
    new_state = update_state(state, results, report if sent else None, message_key if sent else None, run_dt, should_update_snapshots)
    changed = write_state(state_path, new_state)
    print(f"상태 파일: {state_path} ({'변경됨' if changed else '변경 없음'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
