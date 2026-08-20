

from datetime import date

import pandas as pd
import pytest

from src.analysis.strategy_backtest import CostParams
from src.live.paper_trade import advance_one_day, load_state


def _chain(expiry, strike, forward, dte, ce_price, pe_price, ce_delta, pe_delta):
    return pd.DataFrame(
        [
            {"expiry_date": expiry, "strike": strike, "OPTION_TYP": "CE", "forward": forward, "dte": dte, "settle_p": ce_price, "delta": ce_delta},
            {"expiry_date": expiry, "strike": strike, "OPTION_TYP": "PE", "forward": forward, "dte": dte, "settle_p": pe_price, "delta": pe_delta},
        ]
    )


def test_full_cycle_entry_mark_expiry(tmp_path):
    state_path = tmp_path / "state.json"
    log_path = tmp_path / "log.csv"
    costs = CostParams(hedge_bps_notional=1.0, option_roundtrip_pct_premium=0.02)
    expiry = date(2026, 9, 30)

    # Day 1: entry. CE price=5, delta=0.5; PE price=5, delta=-0.4 -> position_delta=-0.1
    day1 = _chain(expiry, 100.0, 100.0, 30, 5.0, 5.0, 0.5, -0.4)
    advance_one_day(date(2026, 9, 1), day1, state_path, log_path, costs, target_dte=30)

    state = load_state(state_path)
    assert state is not None
    assert state.strike == 100.0
    assert state.cumulative_transaction_cost == pytest.approx(0.1)  # 0.02/2 * 10.0

    log = pd.read_csv(log_path)
    assert log.iloc[0]["event"] == "entry"
    assert log.iloc[0]["net_pnl"] == pytest.approx(-0.1)

    # Day 2: mark. CE price=4, delta=0.45; PE price=4, delta=-0.3 -> position_delta=-0.15
    day2 = _chain(expiry, 100.0, 101.0, 29, 4.0, 4.0, 0.45, -0.3)
    advance_one_day(date(2026, 9, 2), day2, state_path, log_path, costs, target_dte=30)

    state = load_state(state_path)
    log = pd.read_csv(log_path)
    row2 = log.iloc[1]
    assert row2["event"] == "mark"
    assert row2["option_pnl"] == pytest.approx(2.0)  # short, 10->8: gain of 2
    assert row2["hedge_pnl"] == pytest.approx(0.1 * (101 - 100))  # hedge_position = -(-0.1) = 0.1
    expected_cost_day2 = abs(0.1 - 0.0) * 100.0 * 1.0 / 10_000
    assert row2["transaction_cost"] == pytest.approx(expected_cost_day2)
    assert state.last_position_delta == pytest.approx(-0.15)

    # Day 3: expiry, straddle worthless. CE price=0, delta=0; PE price=0, delta=0.
    day3 = _chain(expiry, 100.0, 99.0, 0, 0.0, 0.0, 0.0, 0.0)
    advance_one_day(expiry, day3, state_path, log_path, costs, target_dte=30)

    state = load_state(state_path)
    assert state is None  # flat again after settlement

    log = pd.read_csv(log_path)
    row3 = log.iloc[2]
    assert row3["event"] == "expiry_settle"
    assert row3["option_pnl"] == pytest.approx(8.0)  # 8 -> 0: gain of 8
    # hedge_position = -state.last_position_delta (from day 2) = -(-0.15) = 0.15
    assert row3["hedge_pnl"] == pytest.approx(0.15 * (99 - 101))
    assert row3["transaction_cost"] == pytest.approx(abs(0.15 - 0.1) * 101.0 * 1.0 / 10_000 + 0.0)


def test_no_entry_signal_when_chain_empty(tmp_path):
    state_path = tmp_path / "state.json"
    log_path = tmp_path / "log.csv"
    costs = CostParams(hedge_bps_notional=1.0, option_roundtrip_pct_premium=0.02)

    advance_one_day(date(2026, 9, 1), pd.DataFrame(), state_path, log_path, costs, target_dte=30)

    assert load_state(state_path) is None
    log = pd.read_csv(log_path)
    assert log.iloc[0]["event"] == "no_position"


def test_missing_mark_when_leg_disappears(tmp_path):
    state_path = tmp_path / "state.json"
    log_path = tmp_path / "log.csv"
    costs = CostParams(hedge_bps_notional=1.0, option_roundtrip_pct_premium=0.02)
    expiry = date(2026, 9, 30)

    day1 = _chain(expiry, 100.0, 100.0, 30, 5.0, 5.0, 0.5, -0.4)
    advance_one_day(date(2026, 9, 1), day1, state_path, log_path, costs, target_dte=30)

    # Next day: today's chain has no rows for the held (expiry, strike) at all.
    empty_chain = pd.DataFrame(columns=["expiry_date", "strike", "OPTION_TYP", "forward", "dte", "settle_p", "delta"])
    advance_one_day(date(2026, 9, 2), empty_chain, state_path, log_path, costs, target_dte=30)

    log = pd.read_csv(log_path)
    assert log.iloc[1]["event"] == "missing_mark"
    # position should remain open/unchanged, ready to resume marking when data returns
    state = load_state(state_path)
    assert state is not None
    assert state.strike == 100.0
