from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import macro_cycle_evidence as macro  # noqa: E402


def test_change_since_uses_nearest_prior_observation():
    series = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2025-01-31", "2025-07-31", "2026-01-31"]),
    )

    assert macro._change_since(series, 6, pct=True) == pytest.approx(0.1)
    assert macro._change_since(series, 12, pct=False) == 21.0


def test_price_summary_keeps_long_and_short_trend_context():
    dates = pd.bdate_range("2021-01-01", periods=1400)
    series = pd.Series(range(100, 1500), index=dates, dtype=float)

    summary = macro.summarize_price_asset({"name": "Test Index", "ticker": "TEST"}, series)

    assert summary["status"] == "ok"
    assert summary["return_3m"] is not None
    assert summary["return_6m"] is not None
    assert summary["return_12m"] is not None
    assert summary["return_5y"] is not None
    assert summary["above_200d_ma"] is True


def test_render_markdown_preserves_agent_judgment_boundary():
    evidence = {
        "generated_at": "2026-06-18 12:00:00",
        "as_of_requested": "2026-06-18",
        "price_assets": [],
        "fred": {"leading": [], "coincident": [], "lagging": []},
        "rotation": [],
        "sector_rotation": [],
    }

    rendered = macro.render_markdown(evidence)

    assert "evidence package for GPT-5.5/Codex judgment" in rendered
    assert "Do not score mechanically" in rendered
    assert "It must not override Haenam P" in rendered
    assert "국면 위치" in rendered
    assert "100% 기준 매크로 포트폴리오" in rendered
    assert "주식 내부 100% 배분" in rendered
    assert "해남P와의 관계" in rendered


def test_judgment_prompt_requires_phase_location_and_full_allocations():
    prompt = (ROOT / "docs" / "macro_cycle" / "JUDGMENT_PROMPT.md").read_text(encoding="utf-8")

    assert "Phase age is a range estimate" in prompt
    assert "국면 위치" in prompt
    assert "100% 기준 매크로 포트폴리오" in prompt
    assert "주식 내부 100% 배분" in prompt
    assert "not a personalized trade instruction" in prompt


def test_tradingeconomics_pmi_snapshot_parser_handles_latest_previous_sentence():
    html = (
        "Business Confidence in the United States increased to 54 points in May "
        "from 52.70 points in April of 2026."
    )

    parsed = macro.parse_tradingeconomics_pmi_snapshot(html)

    assert parsed["latest"] == 54.0
    assert parsed["previous"] == 52.7
    assert parsed["latest_date"] == "May 2026"
    assert parsed["delta_latest"] == pytest.approx(1.3)
