
import numpy as np
import pandas as pd
import pytest

from src.analysis.vrp_decomposition import (
    har_rv_forecast,
    regime_split_summary,
    summarize_vrp,
    vrp_diffusive,
    vrp_jump,
    vrp_total,
)


def _idx(n):
    return pd.date_range("2015-01-01", periods=n, freq="B")


def test_vrp_total_diffusive_jump_additive_when_no_clipping():
    idx = _idx(200)
    rng = np.random.default_rng(20)
    iv = pd.Series(rng.uniform(0.15, 0.30, 200), index=idx)
    bv = pd.Series(rng.uniform(0.02, 0.05, 200), index=idx)
    raw_jump = pd.Series(rng.uniform(0.001, 0.01, 200), index=idx)  # always positive, so no clipping
    rv_forecast = bv + raw_jump  # RV^2 = BV + J identity, unclipped

    total = vrp_total(iv, rv_forecast)
    diffusive = vrp_diffusive(iv, bv)
    jump = vrp_jump(raw_jump)

    pd.testing.assert_series_equal(diffusive + jump, total, check_names=False)


def test_vrp_jump_is_negative_of_forecast():
    idx = _idx(10)
    jump_forecast = pd.Series(np.linspace(0.001, 0.01, 10), index=idx)
    result = vrp_jump(jump_forecast)
    np.testing.assert_allclose(result.to_numpy(), -jump_forecast.to_numpy())


def test_summarize_vrp_recovers_known_mean():
    idx = _idx(500)
    rng = np.random.default_rng(21)
    series = pd.Series(0.02 + 0.001 * rng.normal(size=500), index=idx)  # mean ~0.02, tiny noise
    result = summarize_vrp(series, nw_lag=21)
    assert result.mean == pytest.approx(0.02, abs=0.002)
    assert result.p_value < 0.01  # clearly nonzero mean


def test_regime_split_summary_splits_correctly():
    idx = _idx(400)
    rng = np.random.default_rng(22)
    series = pd.Series(rng.normal(0, 0.001, 400), index=idx)
    series[idx < "2015-06-01"] += 0.01  # pre-split mean shifted up
    result = regime_split_summary(series, split_date="2015-06-01", nw_lag=5)
    assert result["pre"].mean > result["post"].mean
    assert result["pre"].nobs + result["post"].nobs == 400


def test_har_rv_forecast_recovers_known_coefficients():
    idx = _idx(2000)
    rng = np.random.default_rng(23)
    d = pd.Series(rng.uniform(0.01, 0.08, 2000), index=idx)
    w = pd.Series(rng.uniform(0.01, 0.06, 2000), index=idx)
    m = pd.Series(rng.uniform(0.01, 0.05, 2000), index=idx)
    true_fwd = 0.3 * d + 0.3 * w + 0.4 * m + 0.0005 * rng.normal(size=2000)
    forward_rv2 = pd.Series(true_fwd.values, index=idx)

    forecast = har_rv_forecast(d, w, m, forward_rv2, nw_lag=21)
    assert forecast.index.equals(d.index)
    # forecast should track the (noiseless) true combination closely in-sample
    true_noiseless = 0.3 * d + 0.3 * w + 0.4 * m
    corr = np.corrcoef(forecast, true_noiseless)[0, 1]
    assert corr > 0.99
