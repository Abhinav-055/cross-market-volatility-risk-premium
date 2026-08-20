

from __future__ import annotations

import numpy as np
from scipy.stats import norm

Number = float | np.ndarray


def _d1_d2(forward: Number, strike: Number, vol: Number, tau: Number) -> tuple[Number, Number]:
    vol_sqrt_tau = vol * np.sqrt(tau)
    d1 = (np.log(forward / strike) + 0.5 * vol * vol * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    return d1, d2


def black76_price(
    forward: Number,
    strike: Number,
    vol: Number,
    tau: Number,
    discount_factor: Number,
    option_type: str,
) -> Number:
    if tau <= 0 or vol <= 0:
        intrinsic = max(forward - strike, 0.0) if option_type == "C" else max(strike - forward, 0.0)
        return discount_factor * intrinsic

    d1, d2 = _d1_d2(forward, strike, vol, tau)
    if option_type == "C":
        undiscounted = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    elif option_type == "P":
        undiscounted = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")
    return discount_factor * undiscounted


def black76_vega(forward: Number, strike: Number, vol: Number, tau: Number, discount_factor: Number) -> Number:

    if tau <= 0 or vol <= 0:
        return 0.0
    d1, _ = _d1_d2(forward, strike, vol, tau)
    return discount_factor * forward * norm.pdf(d1) * np.sqrt(tau)


def black76_delta(forward: Number, strike: Number, vol: Number, tau: Number, discount_factor: Number, option_type: str) -> Number:
    
    if tau <= 0 or vol <= 0:
        if option_type == "C":
            return discount_factor * float(forward > strike)
        return -discount_factor * float(forward < strike)
    d1, _ = _d1_d2(forward, strike, vol, tau)
    if option_type == "C":
        return discount_factor * norm.cdf(d1)
    if option_type == "P":
        return -discount_factor * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")


def black76_price_vectorized(
    forward: np.ndarray, strike: np.ndarray, vol: np.ndarray, tau: np.ndarray, discount_factor: np.ndarray, is_call: np.ndarray
) -> np.ndarray:
    d1, d2 = _d1_d2(forward, strike, vol, tau)
    call = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    put = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    return discount_factor * np.where(is_call, call, put)


def black76_delta_vectorized(
    forward: np.ndarray, strike: np.ndarray, vol: np.ndarray, tau: np.ndarray, discount_factor: np.ndarray, is_call: np.ndarray
) -> np.ndarray:
    d1, _ = _d1_d2(forward, strike, vol, tau)
    return discount_factor * np.where(is_call, norm.cdf(d1), -norm.cdf(-d1))
