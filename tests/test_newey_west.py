
import numpy as np
import pytest
import statsmodels.api as sm

from src.utils.newey_west import newey_west_ols, wald_test


@pytest.fixture
def overlapping_data():
    rng = np.random.default_rng(42)
    n = 500
    x_raw = rng.normal(size=n)
    eps = rng.normal(size=n)
    # induce MA(20)-style overlap so HAC actually matters
    u = np.convolve(eps, np.ones(21) / np.sqrt(21), mode="full")[:n]
    y = 1.0 + 0.5 * x_raw + u
    x = np.column_stack([np.ones(n), x_raw])
    return x, y


def test_matches_statsmodels_hac(overlapping_data):
    x, y = overlapping_data
    lag = 21

    ours = newey_west_ols(x, y, lag)

    ref = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False})

    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-8)
    np.testing.assert_allclose(ours.se, ref.bse, rtol=1e-6)
    np.testing.assert_allclose(ours.cov, ref.cov_params(), rtol=1e-6)


def test_nw_se_larger_than_ols_se_under_positive_autocorrelation():
    # AR(1) residuals with strong positive persistence: textbook case where
    # ignoring autocorrelation understates the true sampling variance.
    rng = np.random.default_rng(7)
    n = 1000
    rho = 0.85
    eps = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + eps[t]
    x_raw = rng.normal(size=n)
    y = 1.0 + 0.5 * x_raw + u
    x = np.column_stack([np.ones(n), x_raw])

    nw = newey_west_ols(x, y, lag=21)
    ols_cov = np.linalg.inv(x.T @ x) * np.var(nw.resid, ddof=x.shape[1])
    ols_se = np.sqrt(np.diag(ols_cov))
    assert nw.se[1] > ols_se[1]


def test_wald_joint_test_alpha0_beta1():
    # Construct y = x exactly (alpha=0, beta=1) plus tiny noise -> should not reject.
    rng = np.random.default_rng(1)
    n = 300
    x_raw = rng.normal(scale=1.0, size=n)
    y = x_raw + 0.01 * rng.normal(size=n)
    x = np.column_stack([np.ones(n), x_raw])
    result = newey_west_ols(x, y, lag=5)

    restriction = np.array([[1.0, 0.0], [0.0, 1.0]])
    value = np.array([0.0, 1.0])
    _f_stat, p_value = wald_test(result, restriction, value)
    assert p_value > 0.05


def test_wald_joint_test_rejects_false_restriction():
    rng = np.random.default_rng(2)
    n = 300
    x_raw = rng.normal(size=n)
    y = 5.0 + 2.0 * x_raw + 0.05 * rng.normal(size=n)
    x = np.column_stack([np.ones(n), x_raw])
    result = newey_west_ols(x, y, lag=5)

    restriction = np.array([[1.0, 0.0], [0.0, 1.0]])
    value = np.array([0.0, 1.0])
    _f_stat, p_value = wald_test(result, restriction, value)
    assert p_value < 0.01
