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


def test_live_mode_faber_a_mdd_uses_original_faber_path():
    live_mode = _function_symbol_table("mode_live_and_rebalance")
    referenced_globals = {
        symbol.get_name()
        for symbol in live_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }

    assert "simulate_faber_strategy" in referenced_globals
    assert "calculate_faber_weights" in referenced_globals
    assert "build_haenam_p_strategy_data" not in referenced_globals
    assert "simulate_haenam_p_strategy" not in referenced_globals
    assert "calculate_haenam_p_weights" not in referenced_globals
    assert "expand_haenam_p_execution_weights" not in referenced_globals
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


def test_strategy_backtest_primary_path_uses_original_faber_a():
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
    assert "primary_nav_df = nav_df" in source
    assert "primary_strategy_data = all_data" in source
    assert "primary_label = \"Faber A (원조: 코스피/나스닥 패시브 -5%룰)\"" in source
    assert "현재 본선은 원조 Faber A(코스피/나스닥 패시브 + -5%룰)입니다." in source


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


def test_live_monthly_reference_uses_original_faber_a_passive_weights():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "faber_weights = calculate_faber_weights(rebal_date, haenam_strategy_data, mode='A', price_col=price_col)" in source
    assert "momentum_weights = calculate_weights_at_date(rebal_date, haenam_strategy_data, price_col=price_col)" in source
    assert '("Faber A -5%룰", personal_nav_df, faber_weights' in source
    assert '("연속모멘텀", personal_mom_nav_df, momentum_weights' in source
    assert "전략별 이번 달 기준 손익/MDD" in source
    assert '"기준 손익"' in source
    assert '"이번달 MDD"' in source
    assert "freeze_px[HAENAM_SAMSUNG_NAME] = 349500.0" not in source
    assert "freeze_px[HAENAM_HYNIX_NAME] = 2364000.0" not in source


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


def test_live_signal_display_keeps_original_faber_passive_execution():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def build_haenam_signal_display_rows" in source
    assert "df_results = pd.DataFrame(build_haenam_signal_display_rows(results))" in source
    assert re.search(
        r"df_rebalance_results = pd\.DataFrame\(\s*"
        r"expand_haenam_signal_rows\(\s*"
        r"results, current_date, haenam_price_data, price_col=price_col, kr_weights=\{\}\s*"
        r",\s*nasdaq_active=False\s*"
        r"\)\s*"
        r"\)",
        source,
    )
    assert "FaberA_리밸런싱" in source
    assert "df_results_orig = df_rebalance_results.copy()  # 리밸런싱용" in source
    assert "df_display = df_rebalance_results.copy()" in source


def test_live_current_drawdown_reports_daily_and_monthly_reference_levels():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "bt_monthly_mdd_historical = calculate_monthly_mdd(bt_nav_full)" in source
    assert "현재 고점 대비 하락률 (일별)" in source
    assert "현재 고점 대비 하락률 (월별)" in source
    assert "역대MDD(일별)" in source
    assert "역대MDD(월별)" in source


def test_live_portfolio_policy_snapshot_is_loaded_without_macro_override():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "LIVE_PORTFOLIO_POLICY_PATH = APP_DIR / \"config\" / \"live_portfolio_policy.json\"" in source
    assert "def load_live_portfolio_policy" in source
    assert "if not path.exists():" in source
    assert "return None" in source
    assert "render_live_portfolio_policy(live_policy)" in source
    assert "변경 포트폴리오 스냅샷 총자산 사용" in source
    assert "show_changed_portfolio_snapshot = st.sidebar.checkbox" in source
    assert "value=False" in source


def test_faber_a_live_mode_keeps_legacy_mdd_and_optional_portfolio_snapshot():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "MONTHLY_LEDGER_COLUMNS = [" in source
    assert "def build_live_portfolio_monthly_return_rows" in source
    assert "def render_live_portfolio_monthly_returns" in source
    assert "def render_monthly_profit_recorder" in source
    assert "def upsert_monthly_ledger_record" in source
    assert "이번달 포트폴리오 자산 수익" in source
    assert "이번달 공식 수익 기록" in source
    assert "이번달 수익 기록 저장" in source
    assert "show_legacy_haenam_tools = True" in source
    assert "show_changed_portfolio_snapshot = st.sidebar.checkbox" in source
    assert "render_macro_cycle_monitor(current_date)" in source
    assert 'st.set_page_config(page_title="MAIN"' in source
    assert 'st.title("MAIN")' in source
    assert 'st.subheader("Faber A 실전 & 리밸런싱")' in source
    assert '"1. MAIN"' in source
    assert "signal_weight = 0.20 if near_high else 0.0" in source


def test_main_menu_restores_legacy_backtest_surfaces_centered_on_faber_a():
    source = APP_SOURCE.read_text(encoding="utf-8")
    main_block = re.search(r"def main\(\):.*?if __name__ == \"__main__\":", source, flags=re.S)

    assert main_block is not None
    block = main_block.group(0)
    assert '"1. MAIN"' in block
    assert '"2. 전략 백테스트 (시장 분석)"' in block
    assert '"3. 몬테카를로 시뮬레이션"' in block
    assert '"4. Buy & Hold"' in block
    assert '"5. 종목/ETF 분석"' in block
    assert "금 괴리율 차익거래 계산기" not in block
    assert "부동산 매수 신호" not in block
    assert "mode_strategy_backtest(current_dt, bt_end_date, price_col, bt_start_date)" in block
    assert "mode_monte_carlo(current_dt, current_date, price_col, bt_start_date, init_capital)" in block
    assert "mode_buy_hold_sandbox(current_dt)" in block


def test_portfolio_mode_exposes_macro_cycle_vix_and_fear_greed_monitor():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "MACRO_CYCLE_EVIDENCE_PATH = APP_DIR / \"docs\" / \"macro_cycle\" / \"latest_evidence.json\"" in source
    assert "def load_macro_cycle_evidence" in source
    assert "def summarize_macro_cycle_evidence" in source
    assert "def classify_vix_fear_greed" in source
    assert "매크로 사이클 · VIX · 공포/탐욕" in source
    assert "CNN Fear & Greed 원지수 그대로를 자동 표시하려면 별도 API" in source


def test_asset_analysis_includes_per_band_with_api_boundary():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "def load_asset_analysis_trailing_valuation" in source
    assert "def build_per_band_analysis" in source
    assert "def classify_per_band" in source
    assert "PER 밴드 입력" in source
    assert "저평가 영역" in source
    assert "고평가 영역" in source
    assert "실제 역사적 PER 밴드는 연도별/분기별 EPS 변화" in source
