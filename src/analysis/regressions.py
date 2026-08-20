

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.newey_west import NWResult, newey_west_ols, wald_test


@dataclass
class MZResult:
    nw: NWResult
    alpha: float
    beta: float
    f_stat_joint: float
    p_value_joint: float
    r_squared: float
    nobs: int


def mincer_zarnowitz(iv: pd.Series, forward_rv: pd.Series, nw_lag: int) -> MZResult:
    data = pd.concat([iv.rename("iv"), forward_rv.rename("rv")], axis=1).dropna()
    x = np.column_stack([np.ones(len(data)), data["iv"].to_numpy()])
    y = data["rv"].to_numpy()

    nw = newey_west_ols(x, y, lag=nw_lag)
    restriction = np.array([[1.0, 0.0], [0.0, 1.0]])
    value = np.array([0.0, 1.0])
    f_stat, p_value = wald_test(nw, restriction, value)

    y_hat = x @ nw.params
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return MZResult(
        nw=nw,
        alpha=nw.params[0],
        beta=nw.params[1],
        f_stat_joint=f_stat,
        p_value_joint=p_value,
        r_squared=r_squared,
        nobs=nw.nobs,
    )


@dataclass
class EncompassingResult:
    nw: NWResult
    alpha: float
    beta_iv: float
    beta_rv: float
    r_squared: float
    nobs: int


def encompassing_regression(iv: pd.Series, trailing_rv: pd.Series, forward_rv: pd.Series, nw_lag: int) -> EncompassingResult:
    data = pd.concat(
        [iv.rename("iv"), trailing_rv.rename("trailing_rv"), forward_rv.rename("rv")], axis=1
    ).dropna()
    x = np.column_stack([np.ones(len(data)), data["iv"].to_numpy(), data["trailing_rv"].to_numpy()])
    y = data["rv"].to_numpy()

    nw = newey_west_ols(x, y, lag=nw_lag)

    y_hat = x @ nw.params
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return EncompassingResult(
        nw=nw, alpha=nw.params[0], beta_iv=nw.params[1], beta_rv=nw.params[2], r_squared=r_squared, nobs=nw.nobs
    )
