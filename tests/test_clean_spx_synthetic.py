
import pandas as pd
import pytest

from src.data.clean_spx import _third_fridays, build_synthetic_priced_panel
from src.vol.black76 import black76_delta, black76_price


def test_third_fridays_known_dates():
    # August 2024: Fridays are 2,9,16,23,30 -> 3rd Friday is Aug 16.
    # September 2024: Fridays are 6,13,20,27 -> 3rd Friday is Sep 20.
    expiries = _third_fridays("2024-08-01", "2024-09-30")
    assert pd.Timestamp("2024-08-16") in expiries
    assert pd.Timestamp("2024-09-20") in expiries


def test_synthetic_panel_schema_and_fixed_strike_per_expiry():
    spx_df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=5, freq="B").date,
            "spx_spot": [4700.0, 4710.0, 4695.0, 4720.0, 4705.0],
            "vix": [13.5, 13.2, 14.0, 13.8, 13.6],
        }
    )
    panel = build_synthetic_priced_panel(spx_df, max_dte=75)

    assert set(panel.columns) >= {
        "trade_date", "expiry_date", "strike", "OPTION_TYP", "forward", "dte", "settle_p", "delta",
    }
    assert not panel.empty
    assert (panel["dte"] > 0).all() and (panel["dte"] <= 75).all()
    assert set(panel["OPTION_TYP"].unique()) == {"CE", "PE"}

    # A real listed strike never moves: every (expiry_date) should map to exactly
    # one strike across all trade_dates it appears on, even as forward drifts.
    strikes_per_expiry = panel.groupby("expiry_date")["strike"].nunique()
    assert (strikes_per_expiry == 1).all()

    # And that single strike must actually recur across multiple trading days for
    # at least one expiry (otherwise run_backtest could never track a position).
    days_per_expiry = panel.groupby("expiry_date")["trade_date"].nunique()
    assert (days_per_expiry > 1).any()


def test_synthetic_panel_prices_match_scalar_black76_reference():
    spx_df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-02", periods=3, freq="B").date,
            "spx_spot": [4700.0, 4710.0, 4695.0],
            "vix": [13.5, 13.2, 14.0],
        }
    )
    panel = build_synthetic_priced_panel(spx_df, max_dte=75)

    vix_by_date = dict(zip(spx_df["trade_date"], spx_df["vix"], strict=True))
    for _, row in panel.head(20).iterrows():
        vol = vix_by_date[row["trade_date"]] / 100.0
        tau = row["dte"] / 365.0
        opt_type = "C" if row["OPTION_TYP"] == "CE" else "P"
        expected_price = black76_price(row["forward"], row["strike"], vol, tau, 1.0, opt_type)
        expected_delta = black76_delta(row["forward"], row["strike"], vol, tau, 1.0, opt_type)
        assert row["settle_p"] == pytest.approx(expected_price, rel=1e-9)
        assert row["delta"] == pytest.approx(expected_delta, rel=1e-9)


def test_synthetic_panel_call_and_put_atm_delta_symmetry():
    # Exact ATM (strike == forward): call and put delta should differ by ~1 (df=1 here).
    spx_df = pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02").date()], "spx_spot": [4700.0], "vix": [15.0]})
    panel = build_synthetic_priced_panel(spx_df, max_dte=75)
    row_pair = panel[panel["expiry_date"] == panel["expiry_date"].iloc[0]]
    call_delta = row_pair[row_pair["OPTION_TYP"] == "CE"]["delta"].iloc[0]
    put_delta = row_pair[row_pair["OPTION_TYP"] == "PE"]["delta"].iloc[0]
    assert call_delta - put_delta == pytest.approx(1.0, abs=1e-6)
