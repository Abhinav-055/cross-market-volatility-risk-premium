

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.analysis.strategy_backtest import CostParams
from src.data.clean_nifty import build_futures_forward_curve, merge_spot_and_rate, standardize_legacy_schema
from src.data.download_bhavcopy import fetch_one_day
from src.data.fetch_yfinance import fetch_daily_close
from src.live.paper_trade import advance_one_day
from src.utils.config import REPO_ROOT, load_config
from src.vol.black76 import black76_delta_vectorized
from src.vol.iv_inversion import implied_vol_vectorized


def _already_logged(log_path: Path, today: date) -> bool:
    if not log_path.exists():
        return False
    last_line = pd.read_csv(log_path).tail(1)
    return not last_line.empty and str(last_line["date"].iloc[0]) == today.isoformat()


def build_today_priced_chain(today: date, cfg: dict) -> pd.DataFrame:
    
    fo_symbol = cfg["markets"]["nifty"]["fo_symbol"]
    raw = fetch_one_day(today)
    if raw is None:
        return pd.DataFrame()  # holiday

    std = standardize_legacy_schema(raw, fo_symbol)
    options = std[std["INSTRUMENT"] == "OPTIDX"].copy()
    futures = std[std["INSTRUMENT"] == "FUTIDX"].copy()
    if options.empty or futures.empty:
        return pd.DataFrame()

    fwd_curve = build_futures_forward_curve(futures)
    options_fwd = options.merge(fwd_curve, on=["trade_date", "expiry_date"], how="inner")
    if options_fwd.empty:
        return pd.DataFrame()

    spot_ticker = cfg["markets"]["nifty"]["spot_ticker"]
    spot = fetch_daily_close(spot_ticker, (today - pd.Timedelta(days=10)).isoformat(), (today + pd.Timedelta(days=1)).isoformat())
    merged = merge_spot_and_rate(options_fwd, spot)
    if merged.empty:
        return pd.DataFrame()

    iv_cfg = cfg["iv"]
    is_call = (merged["OPTION_TYP"] == "CE").to_numpy()
    merged["iv"] = implied_vol_vectorized(
        price=merged["settle_p"].to_numpy(),
        forward=merged["forward"].to_numpy(),
        strike=merged["strike"].to_numpy(),
        tau=merged["tau"].to_numpy(),
        discount_factor=merged["discount_factor"].to_numpy(),
        option_type=(merged["OPTION_TYP"] == "CE").map({True: "C", False: "P"}).to_numpy(),
        vol_bounds=tuple(iv_cfg["vol_bounds"]),
        price_tol=iv_cfg["price_tol"],
    )
    merged = merged.dropna(subset=["iv"])
    is_call = (merged["OPTION_TYP"] == "CE").to_numpy()
    merged["delta"] = black76_delta_vectorized(
        merged["forward"].to_numpy(), merged["strike"].to_numpy(), merged["iv"].to_numpy(),
        merged["tau"].to_numpy(), merged["discount_factor"].to_numpy(), is_call,
    )
    return merged


def run(today: date | None = None, config_path=None) -> None:
    cfg = load_config(config_path)
    today = today or date.today()

    processed_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    state_path = processed_dir / "live_paper_trade_state.json"
    log_path = processed_dir / "live_paper_trade_log.csv"

    if _already_logged(log_path, today):
        print(f"{today.isoformat()} already logged, skipping.")
        return

    chain = build_today_priced_chain(today, cfg)
    strat_cfg = cfg["strategy"]["costs"]["nifty"]
    costs = CostParams(
        hedge_bps_notional=strat_cfg["hedge_bps_notional"],
        option_roundtrip_pct_premium=strat_cfg["option_roundtrip_pct_premium"],
    )
    advance_one_day(today, chain, state_path, log_path, costs, cfg["strategy"]["target_dte"])
    print(f"Logged {today.isoformat()} -> {log_path}")


if __name__ == "__main__":
    run()
