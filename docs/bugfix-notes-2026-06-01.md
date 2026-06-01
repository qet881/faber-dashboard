# Bugfix Notes - 2026-06-01

## Important Decisions
- Prioritized the reproducible `NameError` class found by `ruff --select F821`.
- Fixed `mode_live_and_rebalance` so the collapsed static portfolio backtest uses values that exist in live mode instead of variables copied from the strategy backtest mode.
- Reused the already computed long-run live-mode NAV (`bt_nav_full`) for the comparison series to avoid an extra expensive simulation.
- Kept the full default `ruff check .` style debt out of scope because it reports broad pre-existing E701/E702/E402 style issues across `app.py`.

## Modified Files
- `app.py`
  - Added live-mode comparison variables for the static portfolio backtest.
  - Replaced undefined `requested_backtest_end`, `IC`, `primary_nav_df`, and `primary_label` references.
  - Added narrow type annotations and removed unused intermediate variables reported by `ruff --select F`.
- `scripts/fear_overlay_backtest.py`
  - Added explicit `Any`-based annotations for mixed metric rows so mypy can typecheck the script.
- `tests/test_static_regressions.py`
  - Added a regression test that prevents live mode from referencing the copied backtest-mode local variables again.

## Validation
- `python -m pytest` -> 1 passed.
- `python -m mypy --ignore-missing-imports --follow-imports=skip app.py gold_premium_alert.py scripts\fear_overlay_backtest.py tests` -> passed.
- `python -m ruff check --select F app.py gold_premium_alert.py scripts\fear_overlay_backtest.py tests` -> passed.
- `python -m ruff check --select F821 app.py gold_premium_alert.py scripts\fear_overlay_backtest.py tests` -> passed.
- `python -m compileall app.py gold_premium_alert.py scripts\fear_overlay_backtest.py tests` -> passed.
- `python -c "import app; print('app import ok')"` -> passed.
- `python gold_premium_alert.py --dry-run --ignore-market-hours --domestic-source manual --domestic-price 150000` -> passed.
- Streamlit smoke test on local port 8507 -> HTTP response received, then the process was stopped.

## Remaining Risks
- Full default `python -m ruff check .` still fails on pre-existing style-only rules such as E701/E702/E402 and a pandas boolean comparison warning. The bug-class lint checks now pass.
- The Streamlit smoke test verifies startup only; it does not click through every expensive data-loading path.
