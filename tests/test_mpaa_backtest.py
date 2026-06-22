import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mpaa_backtest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mpaa_backtest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_average_momentum_score_counts_positive_months():
    bt = _load_module()
    index = pd.date_range("2020-01-01", periods=8, freq="MS")
    prices = pd.DataFrame({"asset": [100, 110, 105, 120, 130, 125, 140, 150]}, index=index)

    score = bt.average_momentum_score(prices, lookback=3)

    assert score["asset"].iloc[-1] == 1.0


def test_simulate_weighted_portfolio_uses_existing_weights_without_hidden_fill():
    bt = _load_module()
    index = pd.date_range("2020-01-01", periods=3, freq="MS")
    prices = pd.DataFrame({"asset": [100, 200, 100], "현금": [100, 100, 100]}, index=index)
    weights = pd.DataFrame({"asset": [0.0, 1.0, 0.0], "현금": [1.0, 0.0, 1.0]}, index=index)

    nav = bt.simulate_weighted_portfolio(prices, weights)

    assert nav["nav"].iloc[1] == 200.0
    assert nav["nav"].iloc[2] == 200.0


def test_calculate_mpaa_weights_keeps_monthly_weights_near_fully_invested():
    bt = _load_module()
    index = pd.date_range("2020-01-01", periods=18, freq="MS")
    data = {}
    for group in bt.ASSET_GROUPS:
        frame = pd.DataFrame(index=index)
        for offset, asset in enumerate(bt.TICKERS[group]):
            frame[asset] = [100 + offset + month * (offset + 1) for month in range(len(index))]
        data[group] = frame
    data[bt.CASH_GROUP] = pd.DataFrame({bt.CASH_ASSET: [100.0] * len(index)}, index=index)

    weights = bt.calculate_mpaa_weights(data)
    active = weights[weights.sum(axis=1) > 0]

    assert not active.empty
    assert (active.sum(axis=1).sub(1.0).abs() < 1e-9).all()
    assert active.drop(columns=[bt.CASH_ASSET]).sum(axis=1).iloc[-1] > 0
