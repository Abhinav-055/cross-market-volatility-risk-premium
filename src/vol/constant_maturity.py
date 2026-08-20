

from __future__ import annotations

import numpy as np
import pandas as pd


def atm_iv_by_expiry(options: pd.DataFrame) -> pd.DataFrame:
    df = options.copy()
    df["abs_log_moneyness"] = np.log(df["strike"] / df["forward"]).abs()

    group_cols = ["trade_date", "expiry_date"]
    atm_strike = (
        df.loc[df.groupby(group_cols)["abs_log_moneyness"].idxmin(), [*group_cols, "strike"]]
        .rename(columns={"strike": "atm_strike"})
    )
    at_strike = df.merge(atm_strike, on=group_cols)
    at_strike = at_strike[at_strike["strike"] == at_strike["atm_strike"]]

    atm = (
        at_strike.groupby(group_cols)
        .agg(atm_iv=("iv", "mean"), forward=("forward", "first"), dte=("dte", "first"), tau=("tau", "first"))
        .reset_index()
    )
    return atm


def constant_maturity_1m(atm_by_expiry: pd.DataFrame, target_days: int = 30) -> pd.Series:
    results = {}
    for trade_date, day_df in atm_by_expiry.groupby("trade_date"):
        day_df = day_df.sort_values("dte")
        exact = day_df[day_df["dte"] == target_days]
        if len(exact) > 0:
            results[trade_date] = exact["atm_iv"].iloc[0]
            continue

        near = day_df[day_df["dte"] < target_days].tail(1)
        far = day_df[day_df["dte"] > target_days].head(1)
        if len(near) == 0 or len(far) == 0:
            results[trade_date] = np.nan
            continue

        tau_near, iv_near = near["tau"].iloc[0], near["atm_iv"].iloc[0]
        tau_far, iv_far = far["tau"].iloc[0], far["atm_iv"].iloc[0]
        tau_target = target_days / 365.0

        var_near, var_far = iv_near**2 * tau_near, iv_far**2 * tau_far
        weight_far = (tau_target - tau_near) / (tau_far - tau_near)
        var_target = var_near + weight_far * (var_far - var_near)
        results[trade_date] = np.sqrt(var_target / tau_target)

    out = pd.Series(results, name="iv_1m_atm").sort_index()
    out.index.name = "trade_date"
    return out
