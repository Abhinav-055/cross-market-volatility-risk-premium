

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class NWResult:
    params: np.ndarray
    cov: np.ndarray
    se: np.ndarray
    tstats: np.ndarray
    pvalues: np.ndarray
    nobs: int
    resid: np.ndarray


def newey_west_ols(x: np.ndarray, y: np.ndarray, lag: int) -> NWResult:
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, _k = x.shape

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta

    # Meat matrix: Omega = sum_{j=-L}^{L} w_j * Gamma_j, Gamma_j = sum_t u_t u_{t-j} x_t x_{t-j}'
    scores = x * resid[:, None]  # (n, k), row t is u_t * x_t
    omega = scores.T @ scores  # lag 0 term (Gamma_0)
    for lag_j in range(1, lag + 1):
        weight = 1.0 - lag_j / (lag + 1)
        gamma_j = scores[lag_j:].T @ scores[:-lag_j]
        omega += weight * (gamma_j + gamma_j.T)

    cov = xtx_inv @ omega @ xtx_inv
    se = np.sqrt(np.diag(cov))
    tstats = beta / se
    pvalues = 2 * (1 - stats.norm.cdf(np.abs(tstats)))

    return NWResult(params=beta, cov=cov, se=se, tstats=tstats, pvalues=pvalues, nobs=n, resid=resid)


def wald_test(result: NWResult, restriction: np.ndarray, value: np.ndarray) -> tuple[float, float]:
    
    r_beta = restriction @ result.params - value
    r_cov_rt = restriction @ result.cov @ restriction.T
    q = restriction.shape[0]
    k = restriction.shape[1]
    wald_stat = float(r_beta.T @ np.linalg.inv(r_cov_rt) @ r_beta)
    f_stat = wald_stat / q
    dof2 = result.nobs - k
    p_value = float(stats.f.sf(f_stat, q, dof2))
    return f_stat, p_value
