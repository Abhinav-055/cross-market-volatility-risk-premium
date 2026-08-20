

from __future__ import annotations

import numpy as np
import pandas as pd

MU1_INV_SQ = np.pi / 2  # mu1^-2, mu1 = E|Z| = sqrt(2/pi)


def bipower_variation(returns: pd.Series, window: int, annualization_factor: int) -> pd.Series:
    
    abs_r = returns.abs()
    cross_term = abs_r * abs_r.shift(1)
    return cross_term.rolling(window).sum() * MU1_INV_SQ * (annualization_factor / window)


def forward_bipower_variation(returns: pd.Series, window: int, annualization_factor: int) -> pd.Series:
    
    abs_r = returns.abs()
    cross_term = (abs_r * abs_r.shift(1)) * MU1_INV_SQ
    cumsum = cross_term.fillna(0.0).cumsum().to_numpy()
    n = len(cross_term)
    valid_upto = max(n - window, 0)
    out = np.full(n, np.nan)
    out[:valid_upto] = cumsum[window : window + valid_upto] - cumsum[:valid_upto]
    return pd.Series(out * (annualization_factor / window), index=returns.index)


def jump_component(realized_vol: pd.Series, bipower_var: pd.Series) -> pd.Series:
    return (realized_vol.pow(2) - bipower_var).clip(lower=0)


def jump_fraction(realized_vol: pd.Series, bipower_var: pd.Series) -> pd.Series:
    rv_var = realized_vol.pow(2)
    jump = jump_component(realized_vol, bipower_var)
    return (jump / rv_var).where(rv_var > 0)
