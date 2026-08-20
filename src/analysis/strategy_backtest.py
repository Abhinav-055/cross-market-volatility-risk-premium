
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostParams:
    hedge_bps_notional: float
    option_roundtrip_pct_premium: float


def single_cycle_pnl(
    dates: pd.DatetimeIndex,
    straddle_price: np.ndarray,
    straddle_position_delta: np.ndarray,
    underlying_price: np.ndarray,
    costs: CostParams,
) -> pd.DataFrame:
    n = len(dates)
    option_pnl = np.zeros(n)
    hedge_pnl = np.zeros(n)
    transaction_cost = np.zeros(n)
    hedge_position = np.zeros(n)

    entry_cost = costs.option_roundtrip_pct_premium / 2 * abs(straddle_price[0])
    transaction_cost[0] = entry_cost

    for t in range(1, n):
        # Short straddle: price falling is a gain.
        option_pnl[t] = -(straddle_price[t] - straddle_price[t - 1])

        # Hedge sized off yesterday's close delta -> held over [t-1, t].
        hedge_position[t] = -straddle_position_delta[t - 1]
        hedge_pnl[t] = hedge_position[t] * (underlying_price[t] - underlying_price[t - 1])

        rebalance_notional = abs(hedge_position[t] - hedge_position[t - 1]) * underlying_price[t - 1]
        transaction_cost[t] = rebalance_notional * costs.hedge_bps_notional / 10_000

    exit_cost = costs.option_roundtrip_pct_premium / 2 * abs(straddle_price[-1])
    transaction_cost[-1] += exit_cost

    net_pnl = option_pnl + hedge_pnl - transaction_cost

    return pd.DataFrame(
        {
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "transaction_cost": transaction_cost,
            "net_pnl": net_pnl,
            "hedge_position": hedge_position,
        },
        index=dates,
    )


@dataclass
class PerformanceMetrics:
    mean_daily_pnl: float
    std_daily_pnl: float
    sharpe_annualized: float
    sortino_annualized: float
    max_drawdown: float
    worst_month: float
    skewness: float
    kurtosis: float
    tail_ratio: float


def performance_metrics(daily_pnl: pd.Series, annualization_factor: int = 252) -> PerformanceMetrics:
    pnl = daily_pnl.dropna()
    mean, std = pnl.mean(), pnl.std(ddof=1)
    sharpe = mean / std * np.sqrt(annualization_factor) if std > 0 else np.nan

    downside = pnl[pnl < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = mean / downside_std * np.sqrt(annualization_factor) if downside_std and downside_std > 0 else np.nan

    cum_pnl = pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    monthly = pnl.resample("ME").sum() if isinstance(pnl.index, pd.DatetimeIndex) else pnl
    worst_month = monthly.min()

    skewness = pnl.skew()
    kurt = pnl.kurtosis()

    right_tail = pnl[pnl > pnl.quantile(0.95)].mean()
    left_tail = pnl[pnl < pnl.quantile(0.05)].mean()
    tail_ratio = abs(right_tail / left_tail) if left_tail != 0 else np.nan

    return PerformanceMetrics(
        mean_daily_pnl=mean,
        std_daily_pnl=std,
        sharpe_annualized=sharpe,
        sortino_annualized=sortino,
        max_drawdown=max_dd,
        worst_month=worst_month,
        skewness=skewness,
        kurtosis=kurt,
        tail_ratio=tail_ratio,
    )


def rolling_sharpe(daily_pnl: pd.Series, window: int = 252, annualization_factor: int = 252) -> pd.Series:
    mean = daily_pnl.rolling(window).mean()
    std = daily_pnl.rolling(window).std(ddof=1)
    return (mean / std * np.sqrt(annualization_factor)).rename("rolling_sharpe")


@dataclass
class Cycle:
    entry_date: object
    expiry_date: object
    atm_strike: float
    n_days: int


def _select_cycles(priced_panel: pd.DataFrame, target_dte: int) -> list[tuple]:
    
    dates = np.sort(priced_panel["trade_date"].unique())
    expiries = np.sort(priced_panel["expiry_date"].unique())

    dte_by_date_expiry = priced_panel.groupby(["trade_date", "expiry_date"])["dte"].first()

    cycles = []
    prev_expiry = None
    for exp in expiries:
        candidates = dates[dates > prev_expiry] if prev_expiry is not None else dates[dates < exp]
        if len(candidates) == 0:
            prev_expiry = exp
            continue
        entry_date = candidates[0]
        if entry_date >= exp:
            prev_expiry = exp
            continue

        try:
            day_dte = dte_by_date_expiry.loc[entry_date]
        except KeyError:
            prev_expiry = exp
            continue
        positive = day_dte[day_dte > 0]
        if len(positive) == 0:
            prev_expiry = exp
            continue
        chosen_expiry = (positive - target_dte).abs().idxmin()

        if not cycles or cycles[-1] != (entry_date, chosen_expiry):
            cycles.append((entry_date, chosen_expiry))
        prev_expiry = exp

    return cycles


def _atm_strike_on(priced_panel: pd.DataFrame, entry_date, expiry_date) -> float | None:
    day = priced_panel[(priced_panel["trade_date"] == entry_date) & (priced_panel["expiry_date"] == expiry_date)]
    if day.empty:
        return None
    idx = (np.log(day["strike"] / day["forward"])).abs().idxmin()
    return day.loc[idx, "strike"]


def run_backtest(priced_panel: pd.DataFrame, costs: CostParams, target_dte: int = 30) -> tuple[pd.DataFrame, list[Cycle]]:
    cycle_frames = []
    cycle_meta = []

    for entry_date, expiry_date in _select_cycles(priced_panel, target_dte):
        strike = _atm_strike_on(priced_panel, entry_date, expiry_date)
        if strike is None:
            continue

        leg = priced_panel[
            (priced_panel["expiry_date"] == expiry_date)
            & (priced_panel["strike"] == strike)
            & (priced_panel["trade_date"] >= entry_date)
            & (priced_panel["trade_date"] <= expiry_date)
        ]
        calls = leg[leg["OPTION_TYP"] == "CE"].set_index("trade_date")
        puts = leg[leg["OPTION_TYP"] == "PE"].set_index("trade_date")
        common_dates = calls.index.intersection(puts.index).sort_values()
        if len(common_dates) < 2:
            continue

        straddle_price = (calls.loc[common_dates, "settle_p"] + puts.loc[common_dates, "settle_p"]).to_numpy()
        position_delta = -(calls.loc[common_dates, "delta"] + puts.loc[common_dates, "delta"]).to_numpy()
        underlying_price = calls.loc[common_dates, "forward"].to_numpy()

        cycle_pnl = single_cycle_pnl(
            pd.DatetimeIndex(common_dates), straddle_price, position_delta, underlying_price, costs
        )
        cycle_frames.append(cycle_pnl)
        cycle_meta.append(Cycle(entry_date, expiry_date, strike, len(common_dates)))

    if not cycle_frames:
        return pd.DataFrame(columns=["option_pnl", "hedge_pnl", "transaction_cost", "net_pnl", "hedge_position"]), []

    full = pd.concat(cycle_frames).sort_index()
    return full, cycle_meta


def buy_and_hold_pnl(spot: pd.Series, notional: float = 1.0) -> pd.Series:
    return (spot.diff() * notional).rename("buy_and_hold_pnl")


def cost_sensitivity(
    priced_panel: pd.DataFrame, base_costs: CostParams, multipliers: list[float], target_dte: int = 30
) -> pd.DataFrame:
    rows = []
    for mult in multipliers:
        scaled = CostParams(
            hedge_bps_notional=base_costs.hedge_bps_notional * mult,
            option_roundtrip_pct_premium=base_costs.option_roundtrip_pct_premium * mult,
        )
        pnl_df, _ = run_backtest(priced_panel, scaled, target_dte)
        if pnl_df.empty:
            continue
        metrics = performance_metrics(pnl_df["net_pnl"])
        rows.append(
            {
                "cost_multiplier": mult,
                "mean_daily_pnl": metrics.mean_daily_pnl,
                "sharpe_annualized": metrics.sharpe_annualized,
                "max_drawdown": metrics.max_drawdown,
                "total_transaction_cost": pnl_df["transaction_cost"].sum(),
            }
        )
    return pd.DataFrame(rows)
