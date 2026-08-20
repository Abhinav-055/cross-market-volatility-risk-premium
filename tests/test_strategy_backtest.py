
import numpy as np
import pandas as pd
import pytest

from src.analysis.strategy_backtest import CostParams, performance_metrics, rolling_sharpe, single_cycle_pnl


def test_single_cycle_pnl_matches_hand_calculation():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    straddle_price = np.array([10.0, 8.0, 0.0])
    position_delta = np.array([0.1, -0.05, 0.0])  # net delta of the SHORT position
    underlying = np.array([100.0, 101.0, 99.0])
    costs = CostParams(hedge_bps_notional=1.0, option_roundtrip_pct_premium=0.02)

    result = single_cycle_pnl(dates, straddle_price, position_delta, underlying, costs)

    assert result["option_pnl"].iloc[0] == pytest.approx(0.0)
    assert result["option_pnl"].iloc[1] == pytest.approx(2.0)  # short, price fell 10->8: gain
    assert result["option_pnl"].iloc[2] == pytest.approx(8.0)  # 8->0: gain

    assert result["hedge_position"].iloc[1] == pytest.approx(-0.1)  # -delta[0]
    assert result["hedge_position"].iloc[2] == pytest.approx(0.05)  # -delta[1]
    assert result["hedge_pnl"].iloc[1] == pytest.approx(-0.1 * (101 - 100))
    assert result["hedge_pnl"].iloc[2] == pytest.approx(0.05 * (99 - 101))

    assert result["transaction_cost"].iloc[0] == pytest.approx(0.01 * 10.0)  # entry: half of 2% * 10
    assert result["transaction_cost"].iloc[1] == pytest.approx(abs(-0.1 - 0.0) * 100.0 * 1.0 / 10_000)
    expected_exit_hedge_cost = abs(0.05 - (-0.1)) * 101.0 * 1.0 / 10_000
    assert result["transaction_cost"].iloc[2] == pytest.approx(expected_exit_hedge_cost + 0.01 * 0.0)

    assert result["net_pnl"].iloc[0] == pytest.approx(-0.1)
    assert result["net_pnl"].sum() == pytest.approx(
        result["option_pnl"].sum() + result["hedge_pnl"].sum() - result["transaction_cost"].sum()
    )


def test_single_cycle_pnl_no_hedge_reduces_to_option_pnl_minus_costs():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    straddle_price = np.array([5.0, 4.0, 3.0])
    zero_delta = np.zeros(3)
    underlying = np.array([100.0, 100.0, 100.0])  # unchanged, and delta always 0 -> no hedge P&L
    costs = CostParams(hedge_bps_notional=0.5, option_roundtrip_pct_premium=0.0)

    result = single_cycle_pnl(dates, straddle_price, zero_delta, underlying, costs)
    assert (result["hedge_pnl"] == 0).all()
    assert (result["transaction_cost"] == 0).all()
    assert result["net_pnl"].iloc[1] == pytest.approx(1.0)  # 5->4, short gains 1
    assert result["net_pnl"].iloc[2] == pytest.approx(1.0)  # 4->3


def test_performance_metrics_sharpe_matches_sample_moments_formula():
    # Formula-correctness check (not a statistical-convergence check, which would be
    # flaky here: with mu=50 and sigma=1000, the sample mean's own sampling error at
    # n=3000 is comparable to mu, so the *sample* Sharpe is what our function must match.
    rng = np.random.default_rng(30)
    n = 3000
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    pnl = pd.Series(rng.normal(50.0, 1000.0, n), index=idx)

    m = performance_metrics(pnl)
    expected_sharpe = pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)
    assert m.sharpe_annualized == pytest.approx(expected_sharpe)
    assert m.mean_daily_pnl == pytest.approx(pnl.mean())


def test_performance_metrics_max_drawdown_on_known_path():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    pnl = pd.Series([10.0, -30.0, 5.0, -5.0, 20.0], index=idx)
    # cum: 10, -20, -15, -20, 0 ; running max: 10,10,10,10,10 ; dd: 0,-30,-25,-30,-10
    m = performance_metrics(pnl)
    assert m.max_drawdown == pytest.approx(-30.0)


def test_performance_metrics_negative_skew_short_vol_signature():
    # Left-fat-tail distribution: mostly small gains, rare large losses (short-vol signature).
    rng = np.random.default_rng(31)
    n = 5000
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    gains = rng.uniform(1, 3, n)
    crash_mask = rng.uniform(size=n) < 0.02
    pnl = pd.Series(np.where(crash_mask, -50.0, gains), index=idx)
    m = performance_metrics(pnl)
    assert m.skewness < 0


def test_rolling_sharpe_matches_manual_window_calc():
    rng = np.random.default_rng(32)
    n = 500
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    pnl = pd.Series(rng.normal(10, 100, n), index=idx)
    window = 252
    rs = rolling_sharpe(pnl, window=window, annualization_factor=252)

    window_slice = pnl.iloc[0:window]
    expected = window_slice.mean() / window_slice.std(ddof=1) * np.sqrt(252)
    assert rs.iloc[window - 1] == pytest.approx(expected)
    assert rs.iloc[: window - 1].isna().all()
