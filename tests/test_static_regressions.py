import ast
from pathlib import Path
import re
import symtable


APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _module_assignment(name: str):
    tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} was not found")


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
    assert "simulate_faber_strategy" in referenced_globals
    assert "simulate_daily_nav_with_attribution" in referenced_globals
    assert "align_strategies_to_common_dates" in referenced_globals
    assert "build_haenam_p_strategy_data" not in referenced_globals
    assert "build_faber_active_nasdaq_kr_active_data" not in referenced_globals


def test_strategy_backtest_excludes_unrelated_variant_builders():
    backtest_mode = _function_symbol_table("mode_strategy_backtest")
    referenced_globals = {
        symbol.get_name()
        for symbol in backtest_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }
    assert "fetch_vix_data" not in referenced_globals
    assert "simulate_haenam_p_vix_overlay_strategy" not in referenced_globals
    assert "build_haenam_v_strategy_data" not in referenced_globals
    assert "build_haenam_p_local_currency_signal_data" not in referenced_globals


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


def test_strategy_quant_comparison_uses_only_original_faber_and_momentum():
    backtest_mode = _function_symbol_table("mode_strategy_backtest")
    referenced_globals = {
        symbol.get_name()
        for symbol in backtest_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }
    assert "simulate_faber_strategy" in referenced_globals
    assert "simulate_daily_nav_with_attribution" in referenced_globals
    assert "simulate_static_benchmark" not in referenced_globals
    assert "build_haenam_v_strategy_data" not in referenced_globals


def test_strategy_quant_comparison_uses_tr_passive_execution_etfs():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "KOREA_VALUEUP_TR_PROXY_TICKER = '495550'" in source
    assert "KOREA_VALUEUP_PASSIVE_TICKER = KOREA_VALUEUP_TR_PROXY_TICKER" in source
    assert "HAENAM_VALUEUP_PASSIVE_NAME = 'SOL 코리아밸류업TR'" in source
    assert "signal_df = fetch_etf_data(KOREA_VALUEUP_TR_PROXY_TICKER" in source
    assert "valueup_tr_df = fetch_etf_data(KOREA_VALUEUP_PASSIVE_TICKER" in source
    assert "etf_kospi = fetch_etf_data('294400', start_date, end_date)" in source


def test_strategy_quant_comparison_shows_faber_a_and_continuous_momentum_only():
    source = APP_SOURCE.read_text(encoding="utf-8")
    strategy_block = re.search(
        r"strategy_navs = \{\s*\"Faber A\": faber_nav,\s*\"연속모멘텀\": momentum_nav,\s*\}",
        source,
        flags=re.S,
    )

    assert strategy_block is not None
    assert "MDD (일별)" in source
    assert "MDD (월말)" in source
    assert "CAGR / MDD" in source


def test_strategy_quant_comparison_hides_single_stock_variants():
    backtest_mode = _function_symbol_table("mode_strategy_backtest")
    referenced_globals = {
        symbol.get_name()
        for symbol in backtest_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }

    assert not any("HAENAM" in name for name in referenced_globals)
    assert not any("SAMSUNG" in name for name in referenced_globals)
    assert not any("HYNIX" in name for name in referenced_globals)


def test_strategy_quant_comparison_has_two_strategy_mdd_periods():
    backtest_mode = _function_symbol_table("mode_strategy_backtest")
    referenced_globals = {
        symbol.get_name()
        for symbol in backtest_mode.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }

    assert "find_mdd_period" in referenced_globals
    assert "find_monthly_mdd_period" in referenced_globals


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


def test_default_monthly_ledger_has_confirmed_june_2026_basis():
    ledger = _module_assignment("DEFAULT_MONTHLY_LEDGER")
    june = ledger["2026-06"]

    assert june["month_end_date"] == "2026-06-30"
    assert june["month_end_assets"] == 319_352_259
    assert june["official_profit"] == 6_940_263


def test_july_2026_month_end_and_deposit_are_confirmed_for_august_benchmark():
    ledger = _module_assignment("DEFAULT_MONTHLY_LEDGER")
    cash_flows = _module_assignment("PERSONAL_CASH_FLOWS_CONFIRMED")
    july = ledger["2026-07"]

    assert july["month_start_assets"] == 319_352_259
    assert july["month_end_date"] == "2026-07-31"
    assert july["month_end_assets"] == 299_356_616
    assert july["deposit"] == 15_795_862
    assert july["net_external_cash_flow"] == 15_795_862
    assert july["official_profit"] == -35_791_505
    assert cash_flows["2026-07-31"] == 15_795_862


def test_main_menu_only_exposes_main_and_strategy_backtest():
    source = APP_SOURCE.read_text(encoding="utf-8")
    main_block = re.search(r"def main\(\):.*?if __name__ == \"__main__\":", source, flags=re.S)

    assert main_block is not None
    block = main_block.group(0)
    assert '"1. MAIN"' in block
    assert '"2. 전략 백테스트 (시장 분석)"' in block
    assert '"3. 몬테카를로 시뮬레이션"' not in block
    assert '"4. Buy & Hold"' not in block
    assert '"5. 종목/ETF 분석"' not in block
    assert "금 괴리율 차익거래 계산기" not in block
    assert "부동산 매수 신호" not in block
    assert "mode_strategy_backtest(current_dt, bt_end_date, price_col, bt_start_date)" in block
    assert "mode_monte_carlo(current_dt, current_date, price_col, bt_start_date, init_capital)" not in block
    assert "mode_buy_hold_sandbox(current_dt)" not in block


def test_strategy_backtest_displays_previous_calendar_month_return():
    backtest_mode = _function_symbol_table("mode_strategy_backtest")
    local_names = {symbol.get_name() for symbol in backtest_mode.get_symbols()}
    source = APP_SOURCE.read_text(encoding="utf-8")
    start = source.index("def mode_strategy_backtest")
    end = source.index("\ndef ", start + 1)
    mode_source = source[start:end]

    assert "previous_month" in local_names
    assert "previous_month_return" in local_names
    assert "직전 달 수익률" in mode_source


def test_strategy_backtest_displays_asset_and_recent_monthly_returns():
    source = APP_SOURCE.read_text(encoding="utf-8")
    start = source.index("def mode_strategy_backtest")
    end = source.index("\ndef ", start + 1)
    mode_source = source[start:end]

    assert "calculate_monthly_return_series" in mode_source
    assert "직전 달 자산별 수익률" in mode_source
    assert "최근 12개월 월별 수익률" in mode_source
    assert "[*ASSETS.keys(), CASH_NAME]" in mode_source


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
