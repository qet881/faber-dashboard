from pathlib import Path
import re
import symtable


APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _function_symbol_table(function_name: str) -> symtable.SymbolTable:
    root = symtable.symtable(APP_SOURCE.read_text(encoding="utf-8"), str(APP_SOURCE), "exec")
    for child in root.get_children():
        if child.get_name() == function_name:
            return child
    raise AssertionError(f"{function_name} was not found")


def test_live_mode_static_portfolio_backtest_uses_live_scope_values():
    live_mode = _function_symbol_table("mode_live_and_rebalance")
    referenced_globals = {
        symbol.get_name()
        for symbol in live_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }

    assert not {
        "requested_backtest_end",
        "IC",
        "primary_nav_df",
        "primary_label",
    } & referenced_globals


def test_live_mode_haenam_mdd_uses_same_active_builder_as_backtest():
    live_mode = _function_symbol_table("mode_live_and_rebalance")
    referenced_globals = {
        symbol.get_name()
        for symbol in live_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }

    assert "build_faber_active_nasdaq_kr_active_data" in referenced_globals
    assert "build_faber_active_nasdaq_kr_semi_data" not in referenced_globals


def test_backtest_haenam_display_uses_active_backtest_weights():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def expand_haenam_active_backtest_weights" in source
    assert re.search(
        r"FABER_ACTIVE_NASDAQ_KR_SEMI_LABEL:\s*\(\s*lambda d: "
        r"expand_haenam_active_backtest_weights",
        source,
    )
    assert re.search(
        r"w = expand_haenam_active_backtest_weights\(base_w, d\) "
        r"if primary_is_haenam else base_w",
        source,
    )


def test_live_balance_defaults_recover_from_zero_state():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def _ensure_account_balance_state" in source
    assert "sum(float(st.session_state.get(key, 0) or 0) for key, _ in selected_defaults) <= 0" in source
    assert "_ensure_account_balance_state()" in source


def test_live_signal_display_stays_on_base_assets_before_rebalance_expansion():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def build_haenam_signal_display_rows" in source
    assert "df_results = pd.DataFrame(build_haenam_signal_display_rows(results))" in source
    assert re.search(
        r"df_rebalance_results = pd\.DataFrame\(\s*"
        r"expand_haenam_signal_rows\(results, current_date, haenam_price_data, price_col=price_col\)",
        source,
    )
    assert "df_results_orig = df_rebalance_results.copy()  # 리밸런싱용" in source


def test_live_current_drawdown_reports_daily_and_monthly_reference_levels():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "bt_monthly_mdd_historical = calculate_monthly_mdd(bt_nav_full)" in source
    assert "현재 고점 대비 하락률 (일별)" in source
    assert "현재 고점 대비 하락률 (월별)" in source
    assert "역대MDD(일별)" in source
    assert "역대MDD(월별)" in source
