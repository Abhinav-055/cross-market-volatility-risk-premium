

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def realized_variance(returns: pd.Series, window: int, annualization_factor: int) -> pd.Series:
    return returns.pow(2).rolling(window).sum() * (annualization_factor / window)


def realized_vol(returns: pd.Series, window: int, annualization_factor: int) -> pd.Series:
    
    return np.sqrt(realized_variance(returns, window, annualization_factor))


def forward_realized_variance(returns: pd.Series, window: int, annualization_factor: int) -> pd.Series:
    r2 = returns.pow(2)
    cumsum = r2.fillna(0.0).cumsum().to_numpy()
    n = len(r2)
    valid_upto = max(n - window, 0)
    out = np.full(n, np.nan)
    # sum_{i=t+1}^{t+window} r2_i = cumsum[t+window] - cumsum[t], 0-indexed positions
    out[:valid_upto] = cumsum[window : window + valid_upto] - cumsum[:valid_upto]
    return pd.Series(out * (annualization_factor / window), index=returns.index)


def forward_realized_vol(returns: pd.Series, window: int, annualization_factor: int) -> pd.Series:
    return np.sqrt(forward_realized_variance(returns, window, annualization_factor))
