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
    assert parsed["previous_date"] == "April 2026"
    assert parsed["delta_latest"] == pytest.approx(1.3)


def test_merge_pmi_history_upserts_and_sorts():
    existing = pd.Series(
        [50.0, 51.0],
        index=pd.to_datetime(["2026-03-01", "2026-04-01"]),
    )
    updates = {
        pd.Timestamp("2026-04-01"): 51.5,  # overwrite
        pd.Timestamp("2026-05-01"): 54.0,  # new
    }

    merged = macro.merge_pmi_history(existing, updates)

    assert list(merged.index) == list(pd.to_datetime(["2026-03-01", "2026-04-01", "2026-05-01"]))
    assert merged.loc[pd.Timestamp("2026-04-01")] == 51.5  # overwritten
    assert merged.loc[pd.Timestamp("2026-05-01")] == 54.0


def test_pmi_history_round_trip_and_overlay(tmp_path):
    series = pd.Series(
        [48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
        index=pd.date_range("2025-11-01", periods=7, freq="MS"),
    )
    macro.save_pmi_history(tmp_path, series)
    loaded = macro.load_pmi_history(tmp_path)

    assert loaded is not None
    assert loaded.iloc[-1] == 54.0

    summary = {
        "name": "ISM Manufacturing PMI", "status": "ok", "latest": 54.0,
        "delta_3m": None, "delta_6m": None, "delta_12m": None,
        "direction_6m": "rising", "note": "snapshot",
    }
    overlaid = macro.apply_pmi_history(summary, loaded)

    assert overlaid["delta_6m"] == pytest.approx(6.0)  # 54 (2026-05) - 48 (2025-11)
    assert overlaid["history_points"] == 7
    assert "Accumulated locally" in overlaid["note"]


def test_drawdown_stats_measures_distance_from_trailing_high():
    idx = pd.date_range("2025-01-01", periods=5, freq="MS")
    series = pd.Series([100.0, 120.0, 110.0, 105.0, 90.0], index=idx)

    stats = macro._drawdown_stats(series)

    assert stats["peak_value"] == 120.0
    assert stats["peak_date"] == "2025-02-01"
    assert stats["drawdown"] == pytest.approx(90.0 / 120.0 - 1.0)
    assert stats["at_high"] is False


def test_summarize_peak_asset_flags_at_high():
    idx = pd.date_range("2025-01-01", periods=4, freq="MS")
    rising = pd.Series([90.0, 95.0, 100.0, 105.0], index=idx)

    summary = macro.summarize_peak_asset({"name": "X", "ticker": "X", "note": "n"}, rising)

    assert summary["status"] == "ok"
    assert summary["drawdown"] == pytest.approx(0.0)
    assert summary["at_high"] is True


def test_percentile_rank_handles_extremes():
    series = pd.Series([10.0, 20.0, 30.0, 40.0])

    assert macro._percentile_rank(series, 40.0) == pytest.approx(1.0)
    assert macro._percentile_rank(series, 5.0) == pytest.approx(0.0)
    assert macro._percentile_rank(series, None) is None


def test_render_markdown_includes_sentiment_peak_and_qualitative_sections():
    evidence = {
        "generated_at": "2026-06-23 12:00:00",
        "as_of_requested": "2026-06-23",
        "price_assets": [],
        "fred": {"leading": [], "coincident": [], "lagging": []},
        "rotation": [],
        "sector_rotation": [],
        "sentiment": [
            {"name": "VIX (equity fear gauge)", "ticker": "^VIX", "status": "ok",
             "latest_date": "2026-06-22", "latest": 18.0, "change_3m": 0.1,
             "percentile_1y": 0.4, "direction_3m": "rising", "note": "fear gauge"},
            {"name": "S&P 500 drawdown from trailing high", "ticker": "^GSPC",
             "status": "ok", "latest_date": "2026-06-22", "latest": -0.03,
             "change_3m": None, "percentile_1y": None, "direction_3m": "unknown", "note": "dd"},
        ],
        "asset_peaks": [
            {"name": "US Equity (S&P 500)", "ticker": "^GSPC", "status": "ok",
             "peak_date": "2026-06-18", "drawdown": -0.03, "months_since_peak": 0.2,
             "at_high": True, "note": "leads"},
        ],
        "qualitative_top_signals": [
            {"name": "서점 지표", "prompt": "가치투자서 vs 차트서?", "reading": "(수동 입력)"},
        ],
    }

    rendered = macro.render_markdown(evidence)

    assert "Investor Sentiment (Fear / Greed)" in rendered
    assert "Asset Peak Order" in rendered
    assert "Qualitative Top-Signals (manual input)" in rendered
    assert "서점 지표" in rendered
    # boundary preserved alongside new sections
    assert "It must not override Haenam P" in rendered
    assert "투자자 심리 · 고점 신호" in rendered
