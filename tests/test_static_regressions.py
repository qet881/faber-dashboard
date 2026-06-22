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


def test_live_mode_haenam_p_mdd_uses_same_builder_as_backtest():
    live_mode = _function_symbol_table("mode_live_and_rebalance")
    referenced_globals = {
        symbol.get_name()
        for symbol in live_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }

    assert "build_haenam_p_strategy_data" in referenced_globals
    assert "simulate_haenam_p_strategy" in referenced_globals
    assert "calculate_haenam_p_weights" in referenced_globals
    assert "expand_haenam_p_execution_weights" in referenced_globals
    assert "build_haenam_s_strategy_data" not in referenced_globals
    assert "simulate_haenam_s_strategy" not in referenced_globals
    assert "build_faber_active_nasdaq_kr_semi_data" not in referenced_globals


def test_backtest_haenam_p_display_uses_passive_kospi_execution_weights():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def expand_haenam_p_execution_weights" in source
    assert re.search(
        r"w = expand_haenam_p_execution_weights\(base_w, d\) "
        r"if primary_is_haenam else base_w",
        source,
    )


def test_strategy_backtest_primary_path_uses_haenam_p_builder():
    backtest_mode = _function_symbol_table("mode_strategy_backtest")
    referenced_globals = {
        symbol.get_name()
        for symbol in backtest_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "build_faber_active_nasdaq_kr_active_data" in referenced_globals
    assert "build_haenam_p_strategy_data" in referenced_globals
    assert "haenam_p_strategy_data = build_haenam_p_strategy_data" in source
    assert "primary_strategy_data = haenam_p_strategy_data if mom_kr_passive_nav is not None else all_data" in source


def test_strategy_backtest_includes_haenam_p_vix_overlay_candidates():
    backtest_mode = _function_symbol_table("mode_strategy_backtest")
    referenced_globals = {
        symbol.get_name()
        for symbol in backtest_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "HAENAM_P_VIX70_LABEL = '해남P+VIX (70%상한)'" in source
    assert "HAENAM_P_VIX100_LABEL = '해남P+VIX (100%상한)'" in source
    assert "fetch_vix_data" in referenced_globals
    assert "simulate_haenam_p_vix_overlay_strategy" in referenced_globals
    assert "max_equity=0.70" in source
    assert "max_equity=1.00" in source
    assert "HAENAM_P_VIX70_LABEL: haenam_p_vix70_nav" in source
    assert "HAENAM_P_VIX100_LABEL: haenam_p_vix100_nav" in source


def test_vix_overlay_rules_keep_thresholds_and_daily_steps():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def calculate_vix_target_equity" in source
    assert "if not np.isfinite(vix) or vix < 25:" in source
    assert "0.40 + ((vix - 25.0) / 15.0) * 0.30" in source
    assert "0.70 + ((vix - 40.0) / 40.0) * 0.30" in source
    assert "if vix >= 80:" in source
    assert "return 1.0" in source
    assert "if vix >= 40:" in source
    assert "return 0.10" in source
    assert "if vix >= 25:" in source
    assert "return 0.01" in source


def test_strategy_quant_comparison_includes_haenam_v_variants():
    backtest_mode = _function_symbol_table("mode_strategy_backtest")
    referenced_globals = {
        symbol.get_name()
        for symbol in backtest_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "build_haenam_v_strategy_data" in referenced_globals
    assert "expand_haenam_v_backtest_weights" in source
    assert "HAENAM_V_FABER_LABEL = '해남V (-5%룰)'" in source
    assert "HAENAM_V_MOM_LABEL = '해남V (연속모멘텀)'" in source
    assert "HAENAM_V_PASSIVE_FABER_LABEL = '해남V 패시브 (-5%룰)'" in source
    assert "HAENAM_V_PASSIVE_MOM_LABEL = '해남V 패시브 (연속모멘텀)'" in source
    assert "HAENAM_V_FABER_LABEL: haenam_v_faber_nav" in source
    assert "HAENAM_V_MOM_LABEL: haenam_v_mom_nav" in source
    assert "HAENAM_V_PASSIVE_FABER_LABEL: haenam_v_passive_faber_nav" in source
    assert "HAENAM_V_PASSIVE_MOM_LABEL: haenam_v_passive_mom_nav" in source


def test_strategy_quant_comparison_uses_tr_passive_execution_etfs():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "KOREA_VALUEUP_TR_PROXY_TICKER = '495550'" in source
    assert "KOREA_VALUEUP_PASSIVE_TICKER = KOREA_VALUEUP_TR_PROXY_TICKER" in source
    assert "HAENAM_VALUEUP_PASSIVE_NAME = 'SOL 코리아밸류업TR'" in source
    assert "signal_df = fetch_etf_data(KOREA_VALUEUP_TR_PROXY_TICKER" in source
    assert "valueup_tr_df = fetch_etf_data(KOREA_VALUEUP_PASSIVE_TICKER" in source
    assert "etf_kospi = fetch_etf_data('294400', start_date, end_date)" in source


def test_strategy_quant_comparison_shows_haenam_p_faber_and_momentum_only():
    source = APP_SOURCE.read_text(encoding="utf-8")
    quant_labels_block = re.search(
        r"quant_labels = \[\s*FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL,\s*MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL,\s*HAENAM_P_LOCAL_SIGNAL_LABEL,\s*\]",
        source,
        flags=re.S,
    )

    assert quant_labels_block is not None
    block = quant_labels_block.group(0)
    assert "FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL" in block
    assert "MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL" in block
    assert "HAENAM_P_LOCAL_SIGNAL_LABEL" in block
    assert "FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL = '해남P (-5%룰)'" in source
    assert "HAENAM_P_LABEL = '해남P'" in source
    assert "HAENAM_P_LOCAL_SIGNAL_LABEL = '해남P (현지통화 신호)'" in source
    assert "MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL = HAENAM_P_LABEL" in source
    assert "docs/aggressive_no_bond/nav.csv" not in source
    assert "Aggressive no-bond" not in source
    assert "공격형 무채권" not in source
    assert "docs/haenam_p_us_dividend_replacements/nav.csv" not in source
    assert "display_quant_strategies" in source


def test_strategy_quant_comparison_hides_single_stock_variants():
    source = APP_SOURCE.read_text(encoding="utf-8")
    quant_labels_block = re.search(
        r"quant_labels = \[\s*FABER_ACTIVE_NASDAQ_KR_PASSIVE_LABEL,\s*MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL,\s*HAENAM_P_LOCAL_SIGNAL_LABEL,\s*\]",
        source,
        flags=re.S,
    )

    assert quant_labels_block is not None
    block = quant_labels_block.group(0)
    assert "FABER_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL" not in block
    assert "MOM_ACTIVE_NASDAQ_KR_SAMSUNG_LABEL" not in block
    assert "MOM_PASSIVE_NASDAQ_KR_SAMSUNG_LABEL" not in block
    assert "MOM_ACTIVE_NASDAQ_KR_SAMSUNG_SELF_SIGNAL_LABEL" not in block
    assert "FABER_ACTIVE_NASDAQ_KR_HYNIX_LABEL" not in block
    assert "MOM_ACTIVE_NASDAQ_KR_HYNIX_LABEL" not in block
    assert "FABER_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL" not in block
    assert "MOM_ACTIVE_NASDAQ_KR_SAMHYNIX_LABEL" not in block
    assert "us_dividend_experiment_labels" not in source
    assert "aggressive_no_bond_labels" not in source


def test_strategy_quant_comparison_includes_event_risk_windows():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "해남P 나스닥 액티브 vs 패시브 이벤트 위험" in source
    assert '"2025년 4월 관세 이슈", pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30")' in source
    assert '"2026년 3월 전쟁 이슈", pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31")' in source
    assert 'event_active_nav = quant_aligned.get(MOM_ACTIVE_NASDAQ_KR_PASSIVE_LABEL)' in source
    assert 'event_passive_nav = quant_aligned.get(MOM_PASSIVE_NASDAQ_KR_PASSIVE_LABEL)' in source
    assert "액티브 MDD 더 큼" in source
    assert "액티브 위험 더 큼" in source


def test_live_monthly_reference_shows_haenam_a_kr_liquidation_for_haenam_p():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "한국 슬롯: 해남 A 실거래 흐름처럼 삼전/하닉 청산 후 해남P의 코스피200TR 보유분으로 표시." in source
    assert "freeze_px[HAENAM_SAMSUNG_NAME] = 349500.0" in source
    assert "freeze_px[HAENAM_HYNIX_NAME] = 2364000.0" in source
    assert "rebal_weights[KR_STOCK_MIX_ASSET] = kr_slot_w" in source
    assert "active_entry_px[KR_STOCK_MIX_ASSET] = kr_entry_px" in source
    assert '"자산": f"{an}(매도청산)" if an in freeze_px else an' in source


def test_active_backtest_weight_expansion_keeps_nasdaq_active():
    source = APP_SOURCE.read_text(encoding="utf-8")
    execution_block = re.search(
        r"def expand_haenam_execution_weights\(.*?\n(?=def expand_haenam_active_backtest_weights)",
        source,
        flags=re.S,
    )
    active_backtest_block = re.search(
        r"def expand_haenam_active_backtest_weights\(.*?\n(?=def expand_haenam_signal_rows)",
        source,
        flags=re.S,
    )

    assert execution_block is not None
    assert active_backtest_block is not None
    assert re.search(
        r"elif asset == NASDAQ100_ASSET_NAME:\s*"
        r"if nasdaq_active:\s*"
        r"targets = _nasdaq_active_execution_targets\(as_of_date\)",
        execution_block.group(0),
    )
    assert "else:\n                out[asset] = out.get(asset, 0.0) + w" in execution_block.group(0)
    assert "if nasdaq_active" not in active_backtest_block.group(0)
    assert "_nasdaq_active_execution_targets(as_of_date)" in active_backtest_block.group(0)


def test_live_balance_defaults_recover_from_zero_state():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def _ensure_account_balance_state" in source
    assert "sum(float(st.session_state.get(key, 0) or 0) for key, _ in selected_defaults) <= 0" in source
    assert "_ensure_account_balance_state()" in source


def test_live_signal_display_uses_haenam_p_execution_assets():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def build_haenam_signal_display_rows" in source
    assert "df_results = pd.DataFrame(build_haenam_signal_display_rows(results))" in source
    assert re.search(
        r"df_rebalance_results = pd\.DataFrame\(\s*"
        r"expand_haenam_signal_rows\(\s*"
        r"results, current_date, haenam_price_data, price_col=price_col, kr_weights=\{\}\s*"
        r"\)\s*"
        r"\)",
        source,
    )
    assert "df_results_orig = df_rebalance_results.copy()  # 리밸런싱용" in source
    assert "df_display = df_rebalance_results.copy()" in source


def test_live_current_drawdown_reports_daily_and_monthly_reference_levels():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "bt_monthly_mdd_historical = calculate_monthly_mdd(bt_nav_full)" in source
    assert "현재 고점 대비 하락률 (일별)" in source
    assert "현재 고점 대비 하락률 (월별)" in source
    assert "역대MDD(일별)" in source
    assert "역대MDD(월별)" in source
