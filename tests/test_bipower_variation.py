

import numpy as np
import pandas as pd
import pytest

from src.vol.bipower_variation import bipower_variation, forward_bipower_variation, jump_component, jump_fraction
from src.vol.realized_vol import log_returns, realized_vol


def _simulate_merton_jump_diffusion(n, sigma, p_jump, sigma_jump, seed):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    diffusive = rng.normal(0, sigma * np.sqrt(dt), n)
    jumps = (rng.uniform(size=n) < p_jump) * rng.normal(0, sigma_jump, n)
    returns = diffusive + jumps
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


@pytest.fixture(scope="module")
def jump_diffusion_series():
    return _simulate_merton_jump_diffusion(n=10_000, sigma=0.20, p_jump=0.02, sigma_jump=0.05, seed=0)


def test_bipower_variation_recovers_diffusive_variance(jump_diffusion_series):
    true_diffusive_var = 0.20**2
    ret = log_returns(jump_diffusion_series).dropna()
    window = 9000
    bv = bipower_variation(ret, window, annualization_factor=252).dropna()

    rel_err = abs(bv.iloc[-1] - true_diffusive_var) / true_diffusive_var
    assert rel_err < 0.20  # daily-frequency BV is noisy; directional recovery, not exact


def test_realized_variance_is_inflated_by_jumps_relative_to_bipower(jump_diffusion_series):
    ret = log_returns(jump_diffusion_series).dropna()
    window = 9000
    rv = realized_vol(ret, window, annualization_factor=252).dropna()
    bv = bipower_variation(ret, window, annualization_factor=252).dropna()

    assert rv.iloc[-1] ** 2 > bv.iloc[-1]  # RV picks up jump variance, BV is jump-robust


def test_jump_component_nonnegative_and_positive_here(jump_diffusion_series):
    ret = log_returns(jump_diffusion_series).dropna()
    window = 9000
    rv = realized_vol(ret, window, annualization_factor=252).dropna()
    bv = bipower_variation(ret, window, annualization_factor=252).dropna()

    jump = jump_component(rv, bv)
    assert (jump.dropna() >= 0).all()
    assert jump.dropna().iloc[-1] > 0


def test_jump_component_zero_for_pure_diffusion():
    pure_diffusion = _simulate_merton_jump_diffusion(n=5000, sigma=0.15, p_jump=0.0, sigma_jump=0.0, seed=1)
    ret = log_returns(pure_diffusion).dropna()
    window = 4500
    rv = realized_vol(ret, window, annualization_factor=252).dropna()
    bv = bipower_variation(ret, window, annualization_factor=252).dropna()

    jump = jump_component(rv, bv).dropna()
    # No jumps: RV^2 and BV should be close, so the clipped jump component is tiny.
    assert jump.iloc[-1] < 0.15 * rv.iloc[-1] ** 2


def test_jump_fraction_between_zero_and_one(jump_diffusion_series):
    ret = log_returns(jump_diffusion_series).dropna()
    window = 9000
    rv = realized_vol(ret, window, annualization_factor=252).dropna()
    bv = bipower_variation(ret, window, annualization_factor=252).dropna()

    frac = jump_fraction(rv, bv).dropna()
    assert ((frac >= 0) & (frac <= 1)).all()


def test_forward_bipower_variation_no_lookahead_and_matches_trailing_shifted():
    # forward_bipower_variation(t) over [t+1, t+window] should equal trailing
    # bipower_variation(t+window) over the same window of returns.
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 200))
    window = 21
    ann = 252

    fwd = forward_bipower_variation(r, window, ann)
    trailing = bipower_variation(r, window, ann)

    # fwd[t] summarizes (t, t+window]; trailing[t+window] summarizes the same span.
    for t in [0, 10, 50, 100]:
        assert fwd.iloc[t] == pytest.approx(trailing.iloc[t + window])

    assert fwd.iloc[-1:].isna().all() or fwd.iloc[-(window):].isna().any()


def test_forward_bipower_variation_unaffected_by_returns_beyond_window():
    # fwd(1) with window=3 sums cross-terms through index 4; index 5 must not matter.
    r1 = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    r2 = r1.copy()
    r2.iloc[5] = 999.0  # a return strictly beyond fwd(1)'s window
    window = 3
    fwd1 = forward_bipower_variation(r1, window, 1.0)
    fwd2 = forward_bipower_variation(r2, window, 1.0)
    assert fwd1.iloc[1] == pytest.approx(fwd2.iloc[1])
