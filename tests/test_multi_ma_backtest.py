import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "multi_ma_backtest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("multi_ma_backtest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"), dtype=float)


def test_ma5_and_ma4_use_expected_weight_steps():
    bt = _load_module()
    price = _series([100] * 240 + [200] * 9 + [150])

    ma5 = bt.moving_average_signal(price, (10, 20, 60, 120, 200))
    ma4 = bt.moving_average_signal(price, (20, 60, 120, 200))

    assert ma5["exposure"].iloc[-1] == 0.8
    assert ma5["above_count"].iloc[-1] == 4
    assert ma4["exposure"].iloc[-1] == 1.0
    assert ma4["above_count"].iloc[-1] == 4


def test_stack_hard_filter_removes_non_stacked_exposure():
    bt = _load_module()
    price = _series([200] * 240 + [100] * 9 + [150])

    plain = bt.moving_average_signal(price, (10, 20, 60, 120, 200), "none")
    hard = bt.moving_average_signal(price, (10, 20, 60, 120, 200), "hard")

    assert plain["exposure"].iloc[-1] == 0.2
    assert bool(plain["stacked"].iloc[-1]) is False
    assert hard["exposure"].iloc[-1] == 0.0


def test_signal_applies_to_next_day_return_without_lookahead():
    bt = _load_module()
    price = _series([100, 200, 100])
    signal = _series([0, 1, 0])

    nav, position = bt.simulate_strategy(price, signal, rebalance="daily")

    assert position.iloc[1] == 0
    assert position.iloc[2] == 1
    assert nav["nav"].iloc[-1] == 50


def test_52w_high_filter_requires_new_closing_high():
    bt = _load_module()
    not_high = _series([100 + i * 0.1 for i in range(252)] + [120, 119])
    new_high = _series([100 + i * 0.1 for i in range(252)] + [120, 130])

    filtered_not_high = bt.moving_average_signal(not_high, (20, 60, 120, 200), require_52w_high=True)
    filtered_new_high = bt.moving_average_signal(new_high, (20, 60, 120, 200), require_52w_high=True)

    assert bool(filtered_not_high["high_52w"].iloc[-1]) is False
    assert filtered_not_high["exposure"].iloc[-1] == 0.0
    assert bool(filtered_new_high["high_52w"].iloc[-1]) is True
    assert filtered_new_high["exposure"].iloc[-1] == 1.0


def test_52w_near_high_filter_accepts_within_five_percent():
    bt = _load_module()
    price = _series([100 + i * 0.1 for i in range(252)] + [130, 126])

    exact = bt.moving_average_signal(price, (20, 60, 120, 200), require_52w_high=True)
    near5 = bt.moving_average_signal(
        price,
        (20, 60, 120, 200),
        require_52w_high=True,
        high_52w_tolerance=0.05,
    )

    assert bool(exact["high_52w"].iloc[-1]) is False
    assert exact["exposure"].iloc[-1] == 0.0
    assert bool(near5["near_52w_high"].iloc[-1]) is True
    assert near5["exposure"].iloc[-1] == 1.0
