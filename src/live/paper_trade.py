

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.strategy_backtest import CostParams


@dataclass
class OpenPosition:
    entry_date: str
    expiry_date: str
    strike: float
    last_straddle_price: float
    last_position_delta: float
    last_underlying_price: float
    last_hedge_position: float
    cumulative_option_pnl: float = 0.0
    cumulative_hedge_pnl: float = 0.0
    cumulative_transaction_cost: float = 0.0


def load_state(state_path: Path) -> OpenPosition | None:
    if not state_path.exists():
        return None
    with open(state_path, encoding="utf-8") as f:
        data = json.load(f)
    return OpenPosition(**data) if data else None


def save_state(state_path: Path, position: OpenPosition | None) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(asdict(position) if position is not None else None, f, indent=2)


def select_atm_straddle(day_chain: pd.DataFrame, target_dte: int) -> tuple[str, float] | None:
    if day_chain.empty:
        return None
    dte_by_expiry = day_chain.groupby("expiry_date")["dte"].first()
    positive = dte_by_expiry[dte_by_expiry > 0]
    if positive.empty:
        return None
    chosen_expiry = (positive - target_dte).abs().idxmin()

    chain = day_chain[day_chain["expiry_date"] == chosen_expiry]
    idx = (np.log(chain["strike"] / chain["forward"])).abs().idxmin()
    return chosen_expiry, chain.loc[idx, "strike"]


def advance_one_day(
    today: date,
    day_chain: pd.DataFrame,
    state_path: Path,
    log_path: Path,
    costs: CostParams,
    target_dte: int,
) -> None:
    
    position = load_state(state_path)
    log_row = {"date": today.isoformat(), "event": "no_position", "net_pnl": 0.0}

    if position is None:
        picked = select_atm_straddle(day_chain, target_dte)
        if picked is not None:
            chosen_expiry, strike = picked
            leg = day_chain[(day_chain["expiry_date"] == chosen_expiry) & (day_chain["strike"] == strike)]
            calls = leg[leg["OPTION_TYP"] == "CE"]
            puts = leg[leg["OPTION_TYP"] == "PE"]
            if not calls.empty and not puts.empty:
                straddle_price = float(calls["settle_p"].iloc[0] + puts["settle_p"].iloc[0])
                position_delta = -float(calls["delta"].iloc[0] + puts["delta"].iloc[0])
                underlying_price = float(calls["forward"].iloc[0])
                entry_cost = costs.option_roundtrip_pct_premium / 2 * abs(straddle_price)

                position = OpenPosition(
                    entry_date=today.isoformat(),
                    expiry_date=str(chosen_expiry),
                    strike=float(strike),
                    last_straddle_price=straddle_price,
                    last_position_delta=position_delta,
                    last_underlying_price=underlying_price,
                    last_hedge_position=0.0,
                    cumulative_transaction_cost=entry_cost,
                )
                log_row = {
                    "date": today.isoformat(),
                    "event": "entry",
                    "expiry_date": str(chosen_expiry),
                    "strike": float(strike),
                    "option_pnl": 0.0,
                    "hedge_pnl": 0.0,
                    "transaction_cost": entry_cost,
                    "net_pnl": -entry_cost,
                }
    else:
        leg = day_chain[
            (day_chain["expiry_date"].astype(str) == position.expiry_date) & (day_chain["strike"] == position.strike)
        ]
        calls = leg[leg["OPTION_TYP"] == "CE"]
        puts = leg[leg["OPTION_TYP"] == "PE"]
        if calls.empty or puts.empty:
            log_row = {"date": today.isoformat(), "event": "missing_mark", "net_pnl": 0.0}
        else:
            straddle_price = float(calls["settle_p"].iloc[0] + puts["settle_p"].iloc[0])
            position_delta = -float(calls["delta"].iloc[0] + puts["delta"].iloc[0])
            underlying_price = float(calls["forward"].iloc[0])

            option_pnl = -(straddle_price - position.last_straddle_price)
            hedge_position = -position.last_position_delta
            hedge_pnl = hedge_position * (underlying_price - position.last_underlying_price)
            rebalance_notional = abs(hedge_position - position.last_hedge_position) * position.last_underlying_price
            transaction_cost = rebalance_notional * costs.hedge_bps_notional / 10_000

            is_expiry = today.isoformat() == position.expiry_date
            if is_expiry:
                transaction_cost += costs.option_roundtrip_pct_premium / 2 * abs(straddle_price)

            net_pnl = option_pnl + hedge_pnl - transaction_cost

            position.cumulative_option_pnl += option_pnl
            position.cumulative_hedge_pnl += hedge_pnl
            position.cumulative_transaction_cost += transaction_cost
            position.last_straddle_price = straddle_price
            position.last_position_delta = position_delta
            position.last_underlying_price = underlying_price
            position.last_hedge_position = hedge_position

            log_row = {
                "date": today.isoformat(),
                "event": "expiry_settle" if is_expiry else "mark",
                "expiry_date": position.expiry_date,
                "strike": position.strike,
                "option_pnl": option_pnl,
                "hedge_pnl": hedge_pnl,
                "transaction_cost": transaction_cost,
                "net_pnl": net_pnl,
            }
            if is_expiry:
                position = None  # flat again, ready for the next entry

    save_state(state_path, position)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame([log_row])
    log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
