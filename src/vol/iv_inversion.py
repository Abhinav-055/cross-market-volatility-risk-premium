
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src.vol.black76 import black76_price


def implied_vol(
    price: float,
    forward: float,
    strike: float,
    tau: float,
    discount_factor: float,
    option_type: str,
    vol_bounds: tuple[float, float] = (0.01, 3.0),
    price_tol: float = 1e-8,
) -> float:
    lo, hi = vol_bounds
    parity = discount_factor * (forward - strike)
    if option_type == "C" and forward > strike:
        otm_price, otm_type = price - parity, "P"
    elif option_type == "P" and strike > forward:
        otm_price, otm_type = price + parity, "C"
    else:
        otm_price, otm_type = price, option_type

    intrinsic = 0.0  # by construction the OTM leg's intrinsic value is 0
    upper_bound = discount_factor * min(forward, strike)
    if not (intrinsic - price_tol <= otm_price <= upper_bound + price_tol):
        return np.nan
    if otm_price <= price_tol:
        # Below this the price is numerically indistinguishable from zero (e.g. a
        # far-OTM, low-vol, short-tenor option with |d1| in the double digits) and
        # a wide range of vols would round-trip to ~0: no vol is identifiable.
        # Real quote feeds drop these anyway via the zero volume/OI filter.
        return np.nan

    def objective(vol: float) -> float:
        return black76_price(forward, strike, vol, tau, discount_factor, otm_type) - otm_price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        return np.nan
    return brentq(objective, lo, hi, xtol=price_tol)


def _price_otm(forward: np.ndarray, strike: np.ndarray, vol: np.ndarray, tau: np.ndarray, df: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_sqrt_tau = vol * np.sqrt(tau)
        d1 = (np.log(forward / strike) + 0.5 * vol * vol * tau) / vol_sqrt_tau
        d2 = d1 - vol_sqrt_tau
        call = forward * norm.cdf(d1) - strike * norm.cdf(d2)
        put = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    return df * np.where(is_call, call, put)


def _vega_vectorized(forward: np.ndarray, strike: np.ndarray, vol: np.ndarray, tau: np.ndarray, df: np.ndarray) -> np.ndarray:
    
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_sqrt_tau = vol * np.sqrt(tau)
        d1 = (np.log(forward / strike) + 0.5 * vol * vol * tau) / vol_sqrt_tau
    return df * forward * norm.pdf(d1) * np.sqrt(tau)


def implied_vol_vectorized(
    price: np.ndarray,
    forward: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    discount_factor: np.ndarray,
    option_type: np.ndarray,
    vol_bounds: tuple[float, float] = (0.01, 3.0),
    price_tol: float = 1e-8,
    n_iter: int = 60,
) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    discount_factor = np.asarray(discount_factor, dtype=float)
    is_call = np.asarray(option_type) == "C"

    lo_bound, hi_bound = vol_bounds
    parity = discount_factor * (forward - strike)
    itm_call = is_call & (forward > strike)
    itm_put = (~is_call) & (strike > forward)

    otm_price = np.where(itm_call, price - parity, np.where(itm_put, price + parity, price))
    otm_is_call = np.where(itm_call, False, np.where(itm_put, True, is_call))

    upper_bound = discount_factor * np.minimum(forward, strike)
    valid = (otm_price >= -price_tol) & (otm_price <= upper_bound + price_tol) & (otm_price > price_tol)

    lo = np.full_like(forward, lo_bound)
    hi = np.full_like(forward, hi_bound)
    # price is increasing in vol, so f(lo) should be <= target <= f(hi); if not,
    # no root exists in-bounds (e.g. a genuinely mispriced/stale quote) -> invalid.
    f_lo = _price_otm(forward, strike, lo, tau, discount_factor, otm_is_call) - otm_price
    f_hi = _price_otm(forward, strike, hi, tau, discount_factor, otm_is_call) - otm_price
    valid = valid & (f_lo <= price_tol) & (f_hi >= -price_tol)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _price_otm(forward, strike, mid, tau, discount_factor, otm_is_call) - otm_price
        below = f_mid < 0
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)

    vol = 0.5 * (lo + hi)
    return np.where(valid, vol, np.nan)
