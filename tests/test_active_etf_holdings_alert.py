import json
from datetime import datetime

import active_etf_holdings_alert as alert


def make_snapshot(ticker="0015B0", holdings=None, date="2026-06-03"):
    config = next(config for config in alert.ETFS if config.ticker == ticker)
    rows = holdings or [
        {"code": "AAA", "name": "Alpha Semiconductor", "weight": 30},
        {"code": "BBB", "name": "Beta Cloud", "weight": 20},
        {"code": "CCC", "name": "Gamma Cash", "weight": 10},
    ]
    return alert.snapshot_from_json(
        {
            "ticker": ticker,
            "name": config.name,
            "issuer": config.issuer,
            "disclosure_date": date,
            "source_url": "https://official.example/report.pdf",
            "holdings": rows,
        },
        config,
    )


def test_diff_detects_added_removed_top10_and_rank_changes():
    previous = make_snapshot(
        holdings=[
            {"code": "AAA", "name": "NVIDIA Corp", "weight": 30},
            {"code": "BBB", "name": "Microsoft Corp", "weight": 20},
            {"code": "CCC", "name": "Old Holding", "weight": 10},
            {"code": "DDD", "name": "Fourth", "weight": 9},
            {"code": "EEE", "name": "Fifth", "weight": 8},
            {"code": "FFF", "name": "Sixth", "weight": 7},
            {"code": "GGG", "name": "Seventh", "weight": 6},
            {"code": "HHH", "name": "Eighth", "weight": 5},
            {"code": "III", "name": "Ninth", "weight": 4},
            {"code": "JJJ", "name": "Tenth", "weight": 3},
            {"code": "KKK", "name": "Eleventh", "weight": 2},
        ]
    )
    latest = make_snapshot(
        holdings=[
            {"code": "BBB", "name": "Microsoft Corp", "weight": 31},
            {"code": "AAA", "name": "NVIDIA Corp", "weight": 29},
            {"code": "NEW", "name": "New AI Holding", "weight": 11},
            {"code": "DDD", "name": "Fourth", "weight": 9},
            {"code": "EEE", "name": "Fifth", "weight": 8},
            {"code": "FFF", "name": "Sixth", "weight": 7},
            {"code": "GGG", "name": "Seventh", "weight": 6},
            {"code": "HHH", "name": "Eighth", "weight": 5},
            {"code": "III", "name": "Ninth", "weight": 4},
            {"code": "KKK", "name": "Eleventh", "weight": 2},
        ]
    )

    diff = alert.diff_snapshots(previous, latest)

    assert [holding.code for holding in diff.added] == ["NEW"]
    assert [holding.code for holding in diff.removed] == ["CCC", "JJJ"]
    assert [holding.code for holding in diff.top10_entries] == ["NEW", "KKK"]
    assert [holding.code for holding in diff.top10_exits] == ["CCC", "JJJ"]
    assert ("AAA", 1, 2) in [(old.code, old.rank, new.rank) for old, new in diff.rank_changes]


def test_report_includes_changed_etfs_only_and_all_unchanged_message():
    config1, config2 = alert.ETFS[0], alert.ETFS[1]
    previous_1 = make_snapshot(config1.ticker)
    latest_1 = make_snapshot(
        config1.ticker,
        holdings=[
            {"code": "NVDA", "name": "NVIDIA Corp", "weight": 35},
            {"code": "BBB", "name": "Beta Cloud", "weight": 20},
        ],
        date="2026-06-04",
    )
    previous_2 = make_snapshot(config2.ticker)
    latest_2 = make_snapshot(config2.ticker)
    results = [
        alert.SourceResult(True, "ok", config1, latest_1),
        alert.SourceResult(True, "ok", config2, latest_2),
    ]

    report, _, _, should_update = alert.build_report(
        results,
        {config1.ticker: previous_1, config2.ticker: previous_2},
        alert.now_kst(),
    )

    assert should_update is True
    assert report is not None
    assert config1.ticker in report
    assert config2.ticker not in report
    assert "AI/반도체" in report

    all_results = [
        alert.SourceResult(True, "ok", config, make_snapshot(config.ticker))
        for config in alert.ETFS
    ]
    previous = {result.snapshot.ticker: result.snapshot for result in all_results}
    unchanged_report, _, _, should_update = alert.build_report(all_results, previous, alert.now_kst())

    assert should_update is True
    assert unchanged_report == "4개 모두 변화없음."


def test_all_source_failures_do_not_update_snapshots():
    results = [
        alert.SourceResult(False, "source_error", config, error="blocked")
        for config in alert.ETFS
    ]

    report, diffs, failures, should_update = alert.build_report(results, {}, alert.now_kst())

    assert report is not None
    assert "모두 유효하지 않아 스냅샷을 갱신하지 않았습니다" in report
    assert diffs == {}
    assert len(failures) == 4
    assert should_update is False


def test_cli_fixture_dry_run_skips_state_write(tmp_path, capsys):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    state_path = tmp_path / "state.json"
    for config in alert.ETFS:
        snapshot = make_snapshot(config.ticker)
        (fixture_dir / f"{config.ticker}_latest.json").write_text(
            json.dumps(alert.snapshot_to_json(snapshot), ensure_ascii=False),
            encoding="utf-8",
        )

    exit_code = alert.main(
        [
            "--dry-run",
            "--ignore-time-gate",
            "--fixture-dir",
            str(fixture_dir),
            "--state-file",
            str(state_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[발송 예정 메시지]" in output
    assert "--dry-run" in output
    assert not state_path.exists()


def test_message_key_ignores_dynamic_report_time():
    config = alert.ETFS[0]
    snapshot = make_snapshot(config.ticker)
    result = alert.SourceResult(True, "ok", config, snapshot)
    report_18 = "액티브 ETF 보유종목 변화 리포트 (2026-06-04 18:00 KST)\n\n본문"
    report_20 = "액티브 ETF 보유종목 변화 리포트 (2026-06-04 20:00 KST)\n\n본문"

    key_18 = alert.build_message_key(report_18, [result], datetime(2026, 6, 4, 18, tzinfo=alert.KST))
    key_20 = alert.build_message_key(report_20, [result], datetime(2026, 6, 4, 20, tzinfo=alert.KST))

    assert key_18 == key_20
