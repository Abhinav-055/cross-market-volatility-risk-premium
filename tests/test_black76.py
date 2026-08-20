
import numpy as np
import pytest

from src.vol.black76 import (
    black76_delta,
    black76_delta_vectorized,
    black76_price,
    black76_price_vectorized,
    black76_vega,
)
from src.vol.iv_inversion import implied_vol


def test_atm_call_put_parity():
    forward, strike, vol, tau, df = 100.0, 100.0, 0.20, 0.25, 0.995
    call = black76_price(forward, strike, vol, tau, df, "C")
    put = black76_price(forward, strike, vol, tau, df, "P")
    # Black-76 put-call parity: C - P = DF * (F - K)
    assert call - put == pytest.approx(df * (forward - strike), abs=1e-10)


def test_known_value_atm_call():
    # Reference value from a standard Black-76 calculator: F=100, K=100, vol=0.2,
    # T=1, DF=1 (r=0) -> C = F*(2*N(0.1)-1) = 100*(2*0.5398278-1) approx 7.9656
    forward, strike, vol, tau, df = 100.0, 100.0, 0.20, 1.0, 1.0
    call = black76_price(forward, strike, vol, tau, df, "C")
    assert call == pytest.approx(7.9655675, abs=1e-5)


def test_deep_itm_call_converges_to_intrinsic():
    forward, strike, vol, tau, df = 200.0, 100.0, 0.10, 0.5, 1.0
    call = black76_price(forward, strike, vol, tau, df, "C")
    assert call == pytest.approx(df * (forward - strike), rel=1e-3)


def test_zero_tau_is_intrinsic():
    assert black76_price(105.0, 100.0, 0.3, 0.0, 1.0, "C") == pytest.approx(5.0)
    assert black76_price(95.0, 100.0, 0.3, 0.0, 1.0, "P") == pytest.approx(5.0)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        black76_price(100.0, 100.0, 0.2, 1.0, 1.0, "X")


@pytest.mark.parametrize("true_vol", [0.05, 0.15, 0.30, 0.60, 1.20])
@pytest.mark.parametrize("moneyness", [0.85, 1.0, 1.2])
@pytest.mark.parametrize("option_type", ["C", "P"])
def test_iv_round_trip(true_vol, moneyness, option_type):
    forward, tau, df = 100.0, 30 / 365, 0.998
    strike = forward * moneyness
    price = black76_price(forward, strike, true_vol, tau, df, option_type)
    recovered = implied_vol(price, forward, strike, tau, df, option_type)

    # Same parity conversion implied_vol applies internally, to judge whether
    # the OTM-equivalent price is above the numerical noise floor.
    parity = df * (forward - strike)
    if option_type == "C" and forward > strike:
        otm_price = price - parity
    elif option_type == "P" and strike > forward:
        otm_price = price + parity
    else:
        otm_price = price

    if otm_price <= 1e-8:
        # Pathological corner: e.g. a 15%-OTM option at 5% vol and 30 DTE has
        # |d1| ~ 11, so its true price underflows below double precision and
        # is fundamentally unrecoverable -- implied_vol must say so, not guess.
        assert np.isnan(recovered)
    else:
        assert recovered == pytest.approx(true_vol, abs=1e-6)


def test_iv_returns_nan_for_arbitrage_violation():
    # Price above the upper no-arb bound (DF * F for a call) is unquotable.
    forward, strike, tau, df = 100.0, 100.0, 0.1, 1.0
    bad_price = df * forward * 1.5
    assert np.isnan(implied_vol(bad_price, forward, strike, tau, df, "C"))


def test_vega_positive_and_symmetric_with_finite_difference():
    forward, strike, vol, tau, df = 100.0, 95.0, 0.25, 0.5, 0.99
    analytic = black76_vega(forward, strike, vol, tau, df)
    bump = 1e-5
    fd = (
        black76_price(forward, strike, vol + bump, tau, df, "C")
        - black76_price(forward, strike, vol - bump, tau, df, "C")
    ) / (2 * bump)
    assert analytic > 0
    assert analytic == pytest.approx(fd, rel=1e-3)


def test_delta_bounds():
    forward, strike, vol, tau, df = 100.0, 100.0, 0.2, 0.5, 1.0
    call_delta = black76_delta(forward, strike, vol, tau, df, "C")
    put_delta = black76_delta(forward, strike, vol, tau, df, "P")
    assert 0.0 < call_delta < df
    assert -df < put_delta < 0.0
    # ATM call/put delta symmetry (r=0 case, df cancels): call_delta - put_delta ~= df
    assert call_delta - put_delta == pytest.approx(df, abs=1e-6)


def test_price_vectorized_matches_scalar():
    rng = np.random.default_rng(5)
    n = 200
    forward = rng.uniform(50, 200, n)
    strike = rng.uniform(50, 200, n)
    vol = rng.uniform(0.05, 1.0, n)
    tau = rng.uniform(0.02, 1.0, n)
    df = rng.uniform(0.9, 1.0, n)
    is_call = rng.uniform(size=n) < 0.5

    vec = black76_price_vectorized(forward, strike, vol, tau, df, is_call)
    scalar = np.array(
        [black76_price(forward[i], strike[i], vol[i], tau[i], df[i], "C" if is_call[i] else "P") for i in range(n)]
    )
    np.testing.assert_allclose(vec, scalar, rtol=1e-10)


def test_delta_vectorized_matches_scalar():
    rng = np.random.default_rng(6)
    n = 200
    forward = rng.uniform(50, 200, n)
    strike = rng.uniform(50, 200, n)
    vol = rng.uniform(0.05, 1.0, n)
    tau = rng.uniform(0.02, 1.0, n)
    df = rng.uniform(0.9, 1.0, n)
    is_call = rng.uniform(size=n) < 0.5

    vec = black76_delta_vectorized(forward, strike, vol, tau, df, is_call)
    scalar = np.array(
        [black76_delta(forward[i], strike[i], vol[i], tau[i], df[i], "C" if is_call[i] else "P") for i in range(n)]
    )
    np.testing.assert_allclose(vec, scalar, rtol=1e-10)
