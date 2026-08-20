

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.newey_west import newey_west_ols


def har_rv_forecast(
    daily_rv2: pd.Series, weekly_rv2: pd.Series, monthly_rv2: pd.Series, forward_rv2: pd.Series, nw_lag: int
) -> pd.Series:
    data = pd.concat(
        [daily_rv2.rename("d"), weekly_rv2.rename("w"), monthly_rv2.rename("m"), forward_rv2.rename("y")], axis=1
    ).dropna()
    x = np.column_stack([np.ones(len(data)), data["d"], data["w"], data["m"]])
    y = data["y"].to_numpy()
    fit = newey_west_ols(x, y, lag=nw_lag)

    x_full = np.column_stack(
        [np.ones(len(daily_rv2)), daily_rv2.to_numpy(), weekly_rv2.to_numpy(), monthly_rv2.to_numpy()]
    )
    fitted = x_full @ fit.params
    return pd.Series(fitted, index=daily_rv2.index, name="har_rv_forecast")


def vrp_total(iv: pd.Series, rv_forecast: pd.Series) -> pd.Series:
    return (iv**2 - rv_forecast).rename("vrp_total")


def vrp_diffusive(iv: pd.Series, bv_forecast: pd.Series) -> pd.Series:
    return (iv**2 - bv_forecast).rename("vrp_diffusive")


def vrp_jump(jump_forecast: pd.Series) -> pd.Series:
    return (-jump_forecast).rename("vrp_jump")


@dataclass
class VRPSummary:
    mean: float
    t_stat: float
    p_value: float
    nobs: int


def summarize_vrp(series: pd.Series, nw_lag: int) -> VRPSummary:
    clean = series.dropna()
    x = np.ones((len(clean), 1))
    y = clean.to_numpy()
    fit = newey_west_ols(x, y, lag=nw_lag)
    return VRPSummary(mean=fit.params[0], t_stat=fit.tstats[0], p_value=fit.pvalues[0], nobs=fit.nobs)


def regime_split_summary(series: pd.Series, split_date: str, nw_lag: int) -> dict[str, VRPSummary]:
    split = pd.Timestamp(split_date)
    idx = pd.to_datetime(series.index)
    pre = series[idx < split]
    post = series[idx >= split]
    return {"pre": summarize_vrp(pre, nw_lag), "post": summarize_vrp(post, nw_lag)}
