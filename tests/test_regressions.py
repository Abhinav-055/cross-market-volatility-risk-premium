
import numpy as np
import pandas as pd
import pytest

from src.analysis.regressions import encompassing_regression, mincer_zarnowitz


def _make_index(n):
    return pd.date_range("2010-01-01", periods=n, freq="B")


def test_mz_unbiased_forecast_not_rejected():
    rng = np.random.default_rng(10)
    n = 1000
    idx = _make_index(n)
    iv = pd.Series(rng.uniform(0.02, 0.06, n), index=idx)  # variance units
    rv = iv + 0.002 * rng.normal(size=n)  # alpha=0, beta=1 + small noise
    rv = pd.Series(rv.values, index=idx)

    result = mincer_zarnowitz(iv, rv, nw_lag=5)
    assert result.alpha == pytest.approx(0.0, abs=0.005)
    assert result.beta == pytest.approx(1.0, abs=0.1)
    assert result.p_value_joint > 0.05


def test_mz_biased_forecast_rejected():
    rng = np.random.default_rng(11)
    n = 1000
    idx = _make_index(n)
    iv = pd.Series(rng.uniform(0.02, 0.06, n), index=idx)
    rv = pd.Series(2.0 * iv.values + 0.01 + 0.002 * rng.normal(size=n), index=idx)  # beta=2, biased

    result = mincer_zarnowitz(iv, rv, nw_lag=5)
    assert result.p_value_joint < 0.01


def test_mz_drops_misaligned_nans():
    idx = _make_index(50)
    iv = pd.Series(np.linspace(0.02, 0.05, 50), index=idx)
    rv = iv.copy()
    rv.iloc[-21:] = np.nan  # forward RV unavailable near the end, as in real use
    result = mincer_zarnowitz(iv, rv, nw_lag=5)
    assert result.nobs == 29


def test_encompassing_iv_dominates_when_rv_uninformative():
    rng = np.random.default_rng(12)
    n = 1000
    idx = _make_index(n)
    iv = pd.Series(rng.uniform(0.02, 0.06, n), index=idx)
    trailing_rv = pd.Series(rng.uniform(0.02, 0.06, n), index=idx)  # independent noise, no info
    forward_rv = pd.Series(iv.values + 0.002 * rng.normal(size=n), index=idx)  # depends only on iv

    result = encompassing_regression(iv, trailing_rv, forward_rv, nw_lag=5)
    assert result.beta_iv == pytest.approx(1.0, abs=0.15)
    assert result.beta_rv == pytest.approx(0.0, abs=0.15)


def test_encompassing_both_informative():
    rng = np.random.default_rng(13)
    n = 1000
    idx = _make_index(n)
    iv = pd.Series(rng.uniform(0.02, 0.06, n), index=idx)
    trailing_rv = pd.Series(rng.uniform(0.02, 0.06, n), index=idx)
    forward_rv = pd.Series(
        0.5 * iv.values + 0.5 * trailing_rv.values + 0.001 * rng.normal(size=n), index=idx
    )

    result = encompassing_regression(iv, trailing_rv, forward_rv, nw_lag=5)
    assert result.beta_iv == pytest.approx(0.5, abs=0.1)
    assert result.beta_rv == pytest.approx(0.5, abs=0.1)
