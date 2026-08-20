

from __future__ import annotations

import pandas as pd

from src.analysis.regressions import encompassing_regression, mincer_zarnowitz
from src.analysis.strategy_backtest import (
    CostParams,
    cost_sensitivity,
    performance_metrics,
    rolling_sharpe,
    run_backtest,
)
from src.analysis.vrp_decomposition import (
    har_rv_forecast,
    regime_split_summary,
    summarize_vrp,
    vrp_diffusive,
    vrp_jump,
    vrp_total,
)
from src.data import clean_nifty, clean_spx
from src.data.fetch_yfinance import fetch_daily_close
from src.utils.config import REPO_ROOT, load_config
from src.vol.bipower_variation import bipower_variation, forward_bipower_variation, jump_component, jump_fraction
from src.vol.constant_maturity import atm_iv_by_expiry, constant_maturity_1m
from src.vol.realized_vol import (
    forward_realized_variance,
    forward_realized_vol,
    log_returns,
    realized_variance,
    realized_vol,
)

CFG = load_config()
OUT = REPO_ROOT / CFG["paths"]["processed_dir"]
FIG = REPO_ROOT / "figures"
ANN = CFG["realized_vol"]["annualization_factor"]
W = CFG["realized_vol"]["window_days"]
NW_LAG = CFG["newey_west"]["lag"]


def stage1_data() -> None:
    clean_nifty.run()
    clean_spx.run()


def build_nifty_market_series() -> pd.DataFrame:
    clean = pd.read_parquet(OUT / "nifty_options_clean.parquet")
    atm = atm_iv_by_expiry(clean)
    iv_1m = constant_maturity_1m(atm, target_days=CFG["iv"]["constant_maturity_target_days"])
    iv_1m.index = pd.to_datetime(iv_1m.index)

    spot = fetch_daily_close(
        CFG["markets"]["nifty"]["spot_ticker"], CFG["sample"]["start_date"], CFG["sample"]["end_date"]
    )
    spot.index = pd.to_datetime(spot.index)
    ret = log_returns(spot)

    df = pd.DataFrame({"spot": spot, "log_return": ret})
    df["iv_1m_atm"] = iv_1m.reindex(df.index)

    df["rv_trailing"] = realized_vol(ret, W, ANN)
    df["rv_forward"] = forward_realized_vol(ret, W, ANN)
    df["rv_forward_var"] = forward_realized_variance(ret, W, ANN)
    df["bv_trailing"] = bipower_variation(ret, W, ANN)
    df["bv_forward"] = forward_bipower_variation(ret, W, ANN)
    df["jump_trailing"] = jump_component(df["rv_trailing"], df["bv_trailing"])
    df["jump_forward"] = (df["rv_forward_var"] - df["bv_forward"]).clip(lower=0)
    df["jump_fraction_trailing"] = jump_fraction(df["rv_trailing"], df["bv_trailing"])

    df["rv_daily_var"] = realized_variance(ret, 1, ANN)
    df["rv_weekly_var"] = realized_variance(ret, 5, ANN)
    df["rv_monthly_var"] = realized_variance(ret, 21, ANN)

    return df


def build_spx_market_series() -> pd.DataFrame:
    spx = pd.read_parquet(OUT / "spx_data_clean.parquet")
    spx["trade_date"] = pd.to_datetime(spx["trade_date"])
    spx = spx.set_index("trade_date").sort_index()

    ret = log_returns(spx["spx_spot"])
    df = pd.DataFrame({"spot": spx["spx_spot"], "log_return": ret})
    df["iv_1m_atm"] = spx["iv_1m_atm_proxy"]

    df["rv_trailing"] = realized_vol(ret, W, ANN)
    df["rv_forward"] = forward_realized_vol(ret, W, ANN)
    df["rv_forward_var"] = forward_realized_variance(ret, W, ANN)
    df["bv_trailing"] = bipower_variation(ret, W, ANN)
    df["bv_forward"] = forward_bipower_variation(ret, W, ANN)
    df["jump_trailing"] = jump_component(df["rv_trailing"], df["bv_trailing"])
    df["jump_forward"] = (df["rv_forward_var"] - df["bv_forward"]).clip(lower=0)
    df["jump_fraction_trailing"] = jump_fraction(df["rv_trailing"], df["bv_trailing"])

    df["rv_daily_var"] = realized_variance(ret, 1, ANN)
    df["rv_weekly_var"] = realized_variance(ret, 5, ANN)
    df["rv_monthly_var"] = realized_variance(ret, 21, ANN)

    return df


def stage2_and_3_market_series() -> dict[str, pd.DataFrame]:
    series = {"nifty": build_nifty_market_series(), "spx": build_spx_market_series()}
    for name, df in series.items():
        df.to_parquet(OUT / f"{name}_market_series.parquet")
    return series


def run_q1_regressions(df: pd.DataFrame) -> dict:
    iv_var = df["iv_1m_atm"] ** 2
    mz = mincer_zarnowitz(iv_var, df["rv_forward_var"], nw_lag=NW_LAG)
    enc = encompassing_regression(iv_var, df["rv_trailing"] ** 2, df["rv_forward_var"], nw_lag=NW_LAG)
    return {"mz": mz, "encompassing": enc}


def run_q2_vrp(df: pd.DataFrame, regime_split_date: str) -> dict:
    iv = df["iv_1m_atm"]
    har = har_rv_forecast(df["rv_daily_var"], df["rv_weekly_var"], df["rv_monthly_var"], df["rv_forward_var"], NW_LAG)

    vrp_ex_post = vrp_total(iv, df["rv_forward_var"])
    vrp_har = vrp_total(iv, har)
    vrp_diff = vrp_diffusive(iv, df["bv_forward"])
    vrp_j = vrp_jump(df["jump_forward"])

    return {
        "vrp_ex_post": vrp_ex_post,
        "vrp_har": vrp_har,
        "vrp_diffusive": vrp_diff,
        "vrp_jump": vrp_j,
        "summary_ex_post": summarize_vrp(vrp_ex_post, NW_LAG),
        "summary_diffusive": summarize_vrp(vrp_diff, NW_LAG),
        "summary_jump": summarize_vrp(vrp_j, NW_LAG),
        "regime_split": regime_split_summary(vrp_ex_post, regime_split_date, NW_LAG),
    }


def run_q3_strategy(panel: pd.DataFrame, costs: CostParams, target_dte: int) -> dict:
    pnl_df, cycles = run_backtest(panel, costs, target_dte)
    metrics = performance_metrics(pnl_df["net_pnl"]) if not pnl_df.empty else None
    rolling = rolling_sharpe(pnl_df["net_pnl"]) if not pnl_df.empty else pd.Series(dtype=float)
    sensitivity = cost_sensitivity(panel, costs, CFG["strategy"]["costs"]["sensitivity_multipliers"], target_dte)
    return {"pnl": pnl_df, "cycles": cycles, "metrics": metrics, "rolling_sharpe": rolling, "cost_sensitivity": sensitivity}


def stage4_analysis(series: dict[str, pd.DataFrame]) -> dict:
    results = {}
    for market in ["nifty", "spx"]:
        df = series[market]
        results[f"{market}_q1"] = run_q1_regressions(df)
        results[f"{market}_q2"] = run_q2_vrp(df, CFG["vrp"]["regime_split_dates"][market])

    nifty_panel = pd.read_parquet(OUT / "nifty_priced_panel.parquet")
    spx_panel = pd.read_parquet(OUT / "spx_synthetic_priced_panel.parquet")
    target_dte = CFG["strategy"]["target_dte"]

    nifty_costs = CFG["strategy"]["costs"]["nifty"]
    spx_costs = CFG["strategy"]["costs"]["spx"]
    results["nifty_q3"] = run_q3_strategy(
        nifty_panel, CostParams(nifty_costs["hedge_bps_notional"], nifty_costs["option_roundtrip_pct_premium"]), target_dte
    )
    results["spx_q3"] = run_q3_strategy(
        spx_panel, CostParams(spx_costs["hedge_bps_notional"], spx_costs["option_roundtrip_pct_premium"]), target_dte
    )
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("Stage 1: data pipelines...")
    stage1_data()

    print("Stage 2-3: IV construction + RV/jump decomposition...")
    series = stage2_and_3_market_series()

    print("Stage 4: Q1/Q2/Q3 analysis...")
    results = stage4_analysis(series)

    for market in ["nifty", "spx"]:
        mz = results[f"{market}_q1"]["mz"]
        print(f"[{market}] MZ: alpha={mz.alpha:.5f} beta={mz.beta:.3f} R2={mz.r_squared:.3f} p_joint={mz.p_value_joint:.4f}")
        vrp_summary = results[f"{market}_q2"]["summary_ex_post"]
        print(f"[{market}] VRP mean={vrp_summary.mean:.6f} t={vrp_summary.t_stat:.2f}")
        m = results[f"{market}_q3"]["metrics"]
        if m is not None:
            print(f"[{market}] Strategy Sharpe={m.sharpe_annualized:.2f} max_dd={m.max_drawdown:.2f} skew={m.skewness:.2f}")

    print("Done. See data/processed/ for tables and figures/ for charts (charts built in analysis.ipynb).")


if __name__ == "__main__":
    main()
