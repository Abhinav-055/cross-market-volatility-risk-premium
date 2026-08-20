
import pandas as pd
import pytest

from src.analysis.strategy_backtest import CostParams, _atm_strike_on, _select_cycles, run_backtest


def _build_synthetic_panel():
    
    rows = []

    def add_day(trade_date, expiry_date, forward, dte, strikes):
        for strike in strikes:
            off = strike - forward
            for opt_type, base_price, delta_sign in [("CE", 5.0 - off * 0.3, 1), ("PE", 5.0 + off * 0.3, -1)]:
                price = max(base_price, 0.5)
                delta = delta_sign * (0.5 - off * 0.02)
                rows.append(
                    {
                        "trade_date": trade_date,
                        "expiry_date": expiry_date,
                        "strike": strike,
                        "OPTION_TYP": opt_type,
                        "forward": forward,
                        "dte": dte,
                        "settle_p": price,
                        "delta": delta,
                    }
                )

    strikes_a = [95.0, 100.0, 105.0]
    strikes_b = [98.0, 103.0, 108.0]

    # Cycle A: expiry 2020-01-30, entered 2020-01-02, forward drifts 100 -> 102.9
    exp_a = pd.Timestamp("2020-01-30")
    days_a = pd.bdate_range("2020-01-02", exp_a)
    for i, d in enumerate(days_a):
        forward = 100.0 + i * 0.1
        add_day(d, exp_a, forward, (exp_a - d).days, strikes_a)
        # also list the "next month" expiry B in parallel, as NSE does
        exp_b_dte = (pd.Timestamp("2020-02-27") - d).days
        if exp_b_dte > 0:
            add_day(d, pd.Timestamp("2020-02-27"), 100.0 + i * 0.05, exp_b_dte, strikes_b)

    # Cycle B: expiry 2020-02-27, entered the day after 2020-01-30
    exp_b = pd.Timestamp("2020-02-27")
    days_b = pd.bdate_range(exp_a + pd.Timedelta(days=1), exp_b)
    for i, d in enumerate(days_b):
        forward = 103.0 + i * 0.1
        add_day(d, exp_b, forward, (exp_b - d).days, strikes_b)

    df = pd.DataFrame(rows).drop_duplicates(subset=["trade_date", "expiry_date", "strike", "OPTION_TYP"])
    return df


def test_select_cycles_finds_two_monthly_rolls():
    panel = _build_synthetic_panel()
    cycles = _select_cycles(panel, target_dte=30)

    assert len(cycles) == 2
    entry1, expiry1 = cycles[0]
    entry2, expiry2 = cycles[1]

    assert expiry1 == pd.Timestamp("2020-01-30")
    assert entry1 == pd.Timestamp("2020-01-02")  # first date in the panel, before any prior expiry
    assert expiry2 == pd.Timestamp("2020-02-27")
    assert entry2 == pd.Timestamp("2020-01-31")  # first trading day after cycle A's expiry


def test_atm_strike_picks_closest_to_forward():
    panel = _build_synthetic_panel()
    strike = _atm_strike_on(panel, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-30"))
    assert strike == pytest.approx(100.0)  # forward=100.0 on day 0, strikes are 95/100/105


def test_run_backtest_produces_continuous_pnl_across_cycles():
    panel = _build_synthetic_panel()
    costs = CostParams(hedge_bps_notional=0.5, option_roundtrip_pct_premium=0.03)

    pnl_df, cycles = run_backtest(panel, costs, target_dte=30)

    assert len(cycles) == 2
    assert cycles[0].atm_strike == pytest.approx(100.0)
    assert cycles[1].atm_strike == pytest.approx(103.0)
    assert not pnl_df["net_pnl"].isna().any()
    assert len(pnl_df) == cycles[0].n_days + cycles[1].n_days
    # index should be strictly increasing (no overlap between cycles)
    assert pnl_df.index.is_monotonic_increasing


def test_run_backtest_empty_panel_returns_empty_frame():
    empty = pd.DataFrame(columns=["trade_date", "expiry_date", "strike", "OPTION_TYP", "forward", "dte", "settle_p", "delta"])
    costs = CostParams(hedge_bps_notional=0.5, option_roundtrip_pct_premium=0.03)
    pnl_df, cycles = run_backtest(empty, costs)
    assert pnl_df.empty
    assert cycles == []
