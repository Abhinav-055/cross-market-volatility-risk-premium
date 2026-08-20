
import numpy as np
import pandas as pd
import pytest

from src.vol.realized_vol import forward_realized_variance, log_returns, realized_variance, realized_vol


def test_log_returns_known_values():
    prices = pd.Series([100.0, 110.0, 99.0])
    r = log_returns(prices)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(1.10))
    assert r.iloc[2] == pytest.approx(np.log(99 / 110))


def test_realized_variance_matches_hand_calc():
    # Constant daily return of 1% for 5 days: RV = ann_factor * r^2 (constant window sum / window * ann).
    r = pd.Series([0.01] * 5)
    window = 3
    rv = realized_variance(r, window, annualization_factor=252)
    expected = 252 * 0.01**2  # (ann/window) * window * r^2 = ann * r^2 when all returns equal
    assert rv.iloc[-1] == pytest.approx(expected)
    assert rv.iloc[:2].isna().all()  # first window-1 obs are NaN


def test_realized_vol_is_sqrt_of_variance():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 100))
    rv_var = realized_variance(r, 21, 252)
    rv_vol = realized_vol(r, 21, 252)
    pd.testing.assert_series_equal(rv_vol, np.sqrt(rv_var), check_names=False)


def test_forward_realized_variance_no_lookahead():
    # r_i = i (index 0..9); forward_realized_variance(t) should equal ann/window * sum_{i=t+1}^{t+window} r_i^2
    r = pd.Series(np.arange(10, dtype=float))
    window = 3
    ann = 1.0
    fwd = forward_realized_variance(r, window, ann)

    # t=0: sum of r_1^2+r_2^2+r_3^2 = 1+4+9=14, / window
    assert fwd.iloc[0] == pytest.approx(14 / window)
    # t=6: sum of r_7^2+r_8^2+r_9^2 = 49+64+81=194, /window -- last valid index (10-window-1=6)
    assert fwd.iloc[6] == pytest.approx(194 / window)
    # t=7 onward: not enough future data -> NaN
    assert fwd.iloc[7:].isna().all()


def test_forward_realized_variance_uses_strictly_future_returns():
    # Perturbing r_t itself must not change forward_realized_variance(t).
    r1 = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04])
    r2 = r1.copy()
    r2.iloc[1] = 999.0  # blow up r_1; should not affect fwd(1) which sums r_2, r_3
    window = 2
    fwd1 = forward_realized_variance(r1, window, 1.0)
    fwd2 = forward_realized_variance(r2, window, 1.0)
    assert fwd1.iloc[1] == pytest.approx(fwd2.iloc[1])
