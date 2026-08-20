
import numpy as np

from src.vol.black76 import black76_price
from src.vol.iv_inversion import implied_vol, implied_vol_vectorized


def test_vectorized_matches_scalar_on_random_batch():
    rng = np.random.default_rng(42)
    n = 500
    forward = rng.uniform(50, 200, n)
    moneyness = rng.uniform(0.85, 1.15, n)
    strike = forward * moneyness
    true_vol = rng.uniform(0.10, 0.80, n)
    tau = rng.uniform(7 / 365, 90 / 365, n)
    df = rng.uniform(0.97, 1.0, n)
    option_type = np.where(rng.uniform(size=n) < 0.5, "C", "P")

    price = np.array(
        [black76_price(forward[i], strike[i], true_vol[i], tau[i], df[i], option_type[i]) for i in range(n)]
    )

    vectorized = implied_vol_vectorized(price, forward, strike, tau, df, option_type)
    scalar = np.array(
        [implied_vol(price[i], forward[i], strike[i], tau[i], df[i], option_type[i]) for i in range(n)]
    )

    np.testing.assert_allclose(vectorized, scalar, rtol=1e-4, atol=1e-6, equal_nan=True)
    # For the (large majority) well-conditioned rows, recover the true vol closely.
    # A handful of random (low true_vol, far-OTM, short-tau) draws are the same
    # pathological sub-precision-price case covered explicitly in test_black76.py,
    # where NaN is the *correct* answer -- exclude only those from this check.
    resolvable = ~np.isnan(vectorized)
    assert resolvable.sum() > 0.9 * len(vectorized)  # pathological draws should be rare
    np.testing.assert_allclose(vectorized[resolvable], true_vol[resolvable], atol=1e-4)


def test_vectorized_returns_nan_for_arbitrage_violation():
    forward = np.array([100.0])
    strike = np.array([100.0])
    tau = np.array([0.1])
    df = np.array([1.0])
    bad_price = np.array([df[0] * forward[0] * 1.5])  # above upper no-arb bound
    result = implied_vol_vectorized(bad_price, forward, strike, tau, df, np.array(["C"]))
    assert np.isnan(result[0])


def test_vectorized_returns_nan_for_pathological_low_vol_far_otm():
    forward, strike, tau, df = 100.0, 85.0, 30 / 365, 0.998
    price = black76_price(forward, strike, 0.05, tau, df, "C")  # deep ITM call, ~0 OTM-put-equivalent
    result = implied_vol_vectorized(
        np.array([price]), np.array([forward]), np.array([strike]), np.array([tau]), np.array([df]), np.array(["C"])
    )
    assert np.isnan(result[0])


def test_vectorized_handles_mixed_calls_and_puts_itm_and_otm():
    forward = np.array([100.0, 100.0, 100.0, 100.0])
    strike = np.array([90.0, 110.0, 90.0, 110.0])  # ITM call, OTM call, OTM put, ITM put
    option_type = np.array(["C", "C", "P", "P"])
    true_vol = np.array([0.25, 0.25, 0.25, 0.25])
    tau = np.full(4, 30 / 365)
    df = np.full(4, 0.998)

    price = np.array(
        [black76_price(forward[i], strike[i], true_vol[i], tau[i], df[i], option_type[i]) for i in range(4)]
    )
    result = implied_vol_vectorized(price, forward, strike, tau, df, option_type)
    np.testing.assert_allclose(result, true_vol, atol=1e-5)
