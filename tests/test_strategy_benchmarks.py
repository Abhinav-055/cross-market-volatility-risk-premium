

import pandas as pd
import pytest

from src.analysis.strategy_backtest import CostParams, buy_and_hold_pnl, cost_sensitivity
from tests.test_strategy_backtest_orchestration import _build_synthetic_panel


def test_buy_and_hold_pnl_matches_diff():
    spot = pd.Series([100.0, 102.0, 99.0, 101.0])
    result = buy_and_hold_pnl(spot, notional=10.0)
    assert result.iloc[0] != result.iloc[0]  # NaN on first day (no prior close)
    assert result.iloc[1] == pytest.approx(20.0)
    assert result.iloc[2] == pytest.approx(-30.0)
    assert result.iloc[3] == pytest.approx(20.0)


def test_cost_sensitivity_higher_costs_reduce_mean_pnl():
    panel = _build_synthetic_panel()
    base_costs = CostParams(hedge_bps_notional=0.5, option_roundtrip_pct_premium=0.03)
    result = cost_sensitivity(panel, base_costs, multipliers=[0.5, 1.0, 1.5], target_dte=30)

    assert list(result["cost_multiplier"]) == [0.5, 1.0, 1.5]
    assert result["total_transaction_cost"].is_monotonic_increasing
    # higher costs should (weakly) reduce mean daily pnl, all else equal
    assert result["mean_daily_pnl"].iloc[0] >= result["mean_daily_pnl"].iloc[-1]
