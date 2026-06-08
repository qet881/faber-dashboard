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
    assert table.loc["코스피", "추가매수_매도"] == 50
    assert table.loc["나스닥100", "추가매수_매도"] == 50
    assert table.loc["미국채30년", "추가매수_매도"] == 0
    assert table.loc["현금", "추가매수_매도"] == -100


def test_buy_hold_account_priority_is_deterministic():
    app = _load_app_module()

    result = app.calculate_buy_hold_allocation(
        {"일반계좌": 300, "ISA_A": 300, "ISA_B": 300},
        {"코스피": 1 / 3, "나스닥100": 1 / 3, "미국채30년": 1 / 3, "현금": 0},
    )
    table = result["table"].set_index("자산")

    assert table.loc["코스피", "일반계좌"] == 300
    assert table.loc["나스닥100", "ISA_B"] == 300
    assert table.loc["미국채30년", "ISA_A"] == 300


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
    assert "4. Buy & Hold 샌드박스" in source


def test_buy_hold_balance_read_uses_only_three_accounts():
    source = APP_SOURCE.read_text(encoding="utf-8")
    start = source.index("def mode_buy_hold_sandbox")
    end = source.index("def main", start)
    mode_source = source[start:end]

    assert '_get_account_balance_value("bal_gen_kospi"' in mode_source
    assert '_get_account_balance_value("bal_isa_a"' in mode_source
    assert '_get_account_balance_value("bal_isa_b"' in mode_source
    assert '"bal_gen_gold"' not in mode_source
    assert "_ensure_account_balance_state()" not in mode_source
