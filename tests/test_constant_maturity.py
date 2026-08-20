
import numpy as np
import pandas as pd
import pytest

from src.vol.constant_maturity import atm_iv_by_expiry, constant_maturity_1m


def test_atm_iv_by_expiry_picks_closest_strike_and_averages_ce_pe():
    forward = 100.0
    rows = [
        # trade_date, expiry_date, strike, OPTION_TYP, forward, dte, tau, iv
        ("2020-01-01", "e1", 95.0, "CE", forward, 20, 20 / 365, 0.20),
        ("2020-01-01", "e1", 95.0, "PE", forward, 20, 20 / 365, 0.22),
        ("2020-01-01", "e1", 100.0, "CE", forward, 20, 20 / 365, 0.18),  # closest to F=100
        ("2020-01-01", "e1", 100.0, "PE", forward, 20, 20 / 365, 0.20),  # closest to F=100
        ("2020-01-01", "e1", 105.0, "CE", forward, 20, 20 / 365, 0.19),
    ]
    df = pd.DataFrame(rows, columns=["trade_date", "expiry_date", "strike", "OPTION_TYP", "forward", "dte", "tau", "iv"])

    atm = atm_iv_by_expiry(df)
    assert len(atm) == 1
    assert atm["atm_iv"].iloc[0] == pytest.approx((0.18 + 0.20) / 2)


def test_constant_maturity_interpolates_total_variance():
    # Near: 20 DTE, IV=0.20. Far: 40 DTE, IV=0.30. Target: 30 DTE.
    atm = pd.DataFrame(
        {
            "trade_date": ["2020-01-01", "2020-01-01"],
            "expiry_date": ["near", "far"],
            "atm_iv": [0.20, 0.30],
            "forward": [100.0, 100.0],
            "dte": [20, 40],
            "tau": [20 / 365, 40 / 365],
        }
    )
    result = constant_maturity_1m(atm, target_days=30)

    tau_near, tau_far, tau_t = 20 / 365, 40 / 365, 30 / 365
    var_near, var_far = 0.20**2 * tau_near, 0.30**2 * tau_far
    w = (tau_t - tau_near) / (tau_far - tau_near)
    expected_iv = np.sqrt((var_near + w * (var_far - var_near)) / tau_t)

    assert result.loc["2020-01-01"] == pytest.approx(expected_iv)
    # sanity: interpolated IV should sit strictly between the two endpoints
    assert 0.20 < result.loc["2020-01-01"] < 0.30


def test_constant_maturity_exact_match_returns_directly():
    atm = pd.DataFrame(
        {
            "trade_date": ["2020-01-01"],
            "expiry_date": ["e1"],
            "atm_iv": [0.25],
            "forward": [100.0],
            "dte": [30],
            "tau": [30 / 365],
        }
    )
    result = constant_maturity_1m(atm, target_days=30)
    assert result.loc["2020-01-01"] == pytest.approx(0.25)


def test_constant_maturity_nan_when_only_one_side_available():
    atm = pd.DataFrame(
        {
            "trade_date": ["2020-01-01"],
            "expiry_date": ["e1"],
            "atm_iv": [0.25],
            "forward": [100.0],
            "dte": [10],
            "tau": [10 / 365],
        }
    )
    result = constant_maturity_1m(atm, target_days=30)
    assert np.isnan(result.loc["2020-01-01"])
