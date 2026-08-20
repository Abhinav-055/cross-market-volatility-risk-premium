
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

NIFTY_COLOR = "#1f77b4"
SPX_COLOR = "#d62728"
DIFFUSIVE_COLOR = "#1f77b4"
JUMP_COLOR = "#d62728"

CRISES = {
    "2013 taper tantrum": "2013-06-01",
    "2015 China deval.": "2015-08-24",
    "2018 Volmageddon": "2018-02-05",
    "2020 COVID": "2020-03-01",
    "2022 rate hikes": "2022-01-01",
    "2024 India election shock": "2024-06-04",
}


def _annotate_crises(ax, index_min, index_max):
    for label, date_str in CRISES.items():
        d = pd.Timestamp(date_str)
        if index_min <= d <= index_max:
            ax.axvline(d, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.text(d, ax.get_ylim()[1], label, rotation=90, fontsize=6, va="top", ha="right", color="grey")


def fig1_iv_vs_rv(nifty_series: pd.DataFrame, spx_series: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, series, name, color in [
        (axes[0], nifty_series, "Nifty", NIFTY_COLOR),
        (axes[1], spx_series, "SPX", SPX_COLOR),
    ]:
        ax.plot(series.index, series["iv_1m_atm"], label=f"{name} IV (1M ATM)", color=color, linewidth=0.9)
        ax.plot(series.index, series["rv_trailing"], label=f"{name} RV (21d trailing)", color=color, alpha=0.5, linewidth=0.9, linestyle="--")
        ax.set_ylabel("Annualized vol")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(name)
        _annotate_crises(ax, series.index.min(), series.index.max())
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.suptitle("Implied vs Realized Volatility", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig2_vrp_decomposition(nifty_vrp: dict, spx_vrp: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, vrp, name in [(axes[0], nifty_vrp, "Nifty"), (axes[1], spx_vrp, "SPX")]:
        diff = vrp["vrp_diffusive"].dropna()
        jump = vrp["vrp_jump"].dropna()
        ax.plot(diff.index, diff, label="Diffusive VRP", color=DIFFUSIVE_COLOR, linewidth=0.8)
        ax.plot(jump.index, jump, label="Jump VRP", color=JUMP_COLOR, linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Variance units")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(name)
        _annotate_crises(ax, diff.index.min(), diff.index.max())
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.suptitle("VRP Decomposition: Diffusive vs Jump", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig3_strategy_cumulative_pnl(nifty_pnl: pd.DataFrame, spx_pnl: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for ax, pnl_df, name, color in [
        (axes[0], nifty_pnl, "Nifty (real backtest)", NIFTY_COLOR),
        (axes[1], spx_pnl, "SPX (model-implied, VIX-proxy)", SPX_COLOR),
    ]:
        if pnl_df.empty:
            continue
        cum = pnl_df["net_pnl"].cumsum()
        ax.plot(cum.index, cum, color=color, linewidth=1.0)
        ax.set_ylabel("Cumulative net P&L")
        ax.set_title(name)
        _annotate_crises(ax, cum.index.min(), cum.index.max())
    fig.suptitle("Delta-Hedged Short Straddle: Cumulative P&L", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig4_live_vs_backtest(
    backtest_pnl: pd.Series, live_log: pd.DataFrame | None, out_path: Path,
    n_bootstrap: int = 2000, horizon_days: int = 21, seed: int = 42,
    recent_window_pnl: float | None = None, recent_window_label: str | None = None,
) -> None:
    """`recent_window_pnl` is a distinct thing from `live_log`: it's a retrospective
    mechanical re-check (see src/live -- same advance_one_day engine, real NSE data
    for the most recent real trading days available) run once to sanity-check the
    strategy against current data, NOT the forward-only, no-hindsight official live
    paper-trade record. Plotted separately and labeled as such -- conflating the two
    would defeat the entire point of Part 5's honesty requirement.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    daily = backtest_pnl.dropna().to_numpy()
    sims = rng.choice(daily, size=(n_bootstrap, horizon_days), replace=True).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=60, color=NIFTY_COLOR, alpha=0.6, label="Backtest: bootstrapped 21d P&L distribution")

    has_live = live_log is not None and not live_log.empty and "net_pnl" in live_log.columns and (live_log["event"] != "no_position").any()
    if has_live:
        live_cum = live_log["net_pnl"].sum()
        ax.axvline(live_cum, color="black", linewidth=2, label=f"Live paper-trade realization ({live_cum:.1f})")

    if recent_window_pnl is not None:
        label = f"Recent 21 real trading days, retrospective check ({recent_window_pnl:.1f})"
        if recent_window_label:
            label = f"{recent_window_label}: {recent_window_pnl:.1f} (retrospective check)"
        ax.axvline(recent_window_pnl, color="darkorange", linewidth=2, linestyle="--", label=label)

    ax.set_xlabel("21-trading-day net P&L")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Live Paper-Trade vs Backtest-Implied Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
