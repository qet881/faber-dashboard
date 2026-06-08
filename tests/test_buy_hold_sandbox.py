import importlib.util
from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("faber_app_for_buy_hold_tests", APP_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_buy_hold_allocation_excludes_gold_account():
    app = _load_app_module()

    result = app.calculate_buy_hold_allocation(
        {"일반계좌": 100, "ISA_A": 200, "ISA_B": 300},
        {"코스피": 0.25, "나스닥100": 0.25, "미국채30년": 0.20, "현금": 0.30},
    )

    assert "금계좌" not in result["table"].columns
    assert {"일반계좌", "ISA_A", "ISA_B"} <= set(result["table"].columns)


def test_buy_hold_target_amounts_and_deltas_use_baseline():
    app = _load_app_module()

    result = app.calculate_buy_hold_allocation(
        {"일반계좌": 400, "ISA_A": 300, "ISA_B": 300},
        {"코스피": 0.25, "나스닥100": 0.25, "미국채30년": 0.20, "현금": 0.30},
    )
    table = result["table"].set_index("자산")

    assert table.loc["합계", "목표금액"] == 1000
    assert table.loc[app.HAENAM_KR_TIME_NAME, "추가매수_매도"] == 25
    assert table.loc[app.HAENAM_KR_KOACT_NAME, "추가매수_매도"] == 25
    assert table.loc[app.HAENAM_TIME_NAME, "추가매수_매도"] == 25
    assert table.loc[app.HAENAM_KOACT_NAME, "추가매수_매도"] == 25
    assert table.loc["미국채30년", "추가매수_매도"] == 0
    assert table.loc[app.CASH_NAME, "추가매수_매도"] == -100


def test_buy_hold_current_amounts_override_baseline_for_real_weights():
    app = _load_app_module()

    result = app.calculate_buy_hold_allocation(
        {"일반계좌": 400, "ISA_A": 300, "ISA_B": 300},
        {"코스피": 0.25, "나스닥100": 0.25, "미국채30년": 0.20, "현금": 0.30},
        current_amounts={
            app.HAENAM_KR_TIME_NAME: 100,
            app.HAENAM_KR_KOACT_NAME: 100,
            app.HAENAM_TIME_NAME: 300,
            app.HAENAM_KOACT_NAME: 100,
            "미국채30년": 200,
            app.CASH_NAME: 200,
        },
    )
    table = result["table"].set_index("자산")

    assert table.loc["합계", "현재금액"] == 1000
    assert table.loc[app.HAENAM_TIME_NAME, "현재비중"] == 0.30
    assert table.loc[app.HAENAM_TIME_NAME, "추가매수_매도"] == -175
    assert table.loc[app.CASH_NAME, "추가매수_매도"] == 100


def test_buy_hold_account_priority_is_deterministic():
    app = _load_app_module()

    result = app.calculate_buy_hold_allocation(
        {"일반계좌": 300, "ISA_A": 300, "ISA_B": 300},
        {"코스피": 1 / 3, "나스닥100": 1 / 3, "미국채30년": 1 / 3, "현금": 0},
    )
    table = result["table"].set_index("자산")

    assert table.loc[app.HAENAM_KR_TIME_NAME, "일반계좌"] == 150
    assert table.loc[app.HAENAM_KR_KOACT_NAME, "일반계좌"] == 150
    assert table.loc[app.HAENAM_TIME_NAME, "ISA_B"] == 150
    assert table.loc[app.HAENAM_KOACT_NAME, "ISA_B"] == 150
    assert table.loc["미국채30년", "ISA_A"] == 300


def test_buy_hold_uses_live_rebalance_allocation_engine():
    app = _load_app_module()

    account_balances = {"일반계좌": 300, "ISA_A": 300, "ISA_B": 300}
    target_weights = {"코스피": 1 / 3, "나스닥100": 1 / 3, "미국채30년": 1 / 3, "현금": 0}
    result = app.calculate_buy_hold_allocation(account_balances, target_weights)

    execution_weights = app.expand_buy_hold_weights_to_execution_assets(result["weights"])
    expected = app._calculate_tax_optimized_allocation(
        app.pd.DataFrame(
            [{"자산명": asset, "추천비중": weight} for asset, weight in execution_weights.items()]
        ),
        account_balances,
        app.BUY_HOLD_ACCOUNT_COLUMNS,
    )

    actual = result["table"].set_index("자산")
    expected = expected.set_index("자산명")
    for asset in execution_weights:
        for account in app.BUY_HOLD_ACCOUNT_COLUMNS:
            assert actual.loc[asset, account] == expected.loc[asset, account]


def test_buy_hold_zero_weights_fall_back_to_baseline():
    app = _load_app_module()

    result = app.calculate_buy_hold_allocation(
        {"일반계좌": 100, "ISA_A": 100, "ISA_B": 100},
        {"코스피": 0, "나스닥100": 0, "미국채30년": 0, "현금": 0},
    )

    assert result["weights"] == app.BUY_HOLD_BASELINE_WEIGHTS
    assert result["raw_weight_total"] == 0


def test_buy_hold_mode_is_added_without_removing_existing_modes():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "1. 내 자산 & 리밸런싱 (실전)" in source
    assert "2. 전략 백테스트 (시장 분석)" in source
    assert "3. 몬테카를로 시뮬레이션" in source
    assert "4. Buy & Hold" in source
    assert "Buy & Hold 샌드박스" not in source


def test_buy_hold_balance_inputs_use_only_three_accounts():
    source = APP_SOURCE.read_text(encoding="utf-8")
    start = source.index("def mode_buy_hold_sandbox")
    end = source.index("def main", start)
    mode_source = source[start:end]

    assert '_get_account_balance_value("bal_gen_kospi"' in mode_source
    assert '_get_account_balance_value("bal_isa_a"' in mode_source
    assert '_get_account_balance_value("bal_isa_b"' in mode_source
    assert '"bal_gen_gold"' not in mode_source
    assert "_ensure_account_balance_state()" not in mode_source
