"""
Visualisation: all charts for the momentum alpha signal study.

All figures are saved to figures/ as PNG for embedding in the README
and as pre-rendered outputs in the notebook.
"""
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

FIGURE_DIR = Path("figures")

# Diverging colour palette: red (weak) → blue (strong)
QUINTILE_COLORS = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
LS_COLOR = "#1a1a2e"
BENCHMARK_COLOR = "#888888"
IC_COLOR = "#2c7bb6"
IC_NEGATIVE_COLOR = "#d73027"
ROLLING_COLOR = "#d7191c"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.grid": True,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})


def _save(fig: plt.Figure, name: str, dpi: int = 150) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {path}")


def plot_ic_series(ic: pd.Series, rolling_window: int = 12, save: bool = True) -> plt.Figure:
    """
    IC bar chart with 12-month rolling mean overlaid.

    A consistently positive rolling mean is the primary diagnostic for
    a tradeable signal. Single-month IC is noisy by nature.
    """
    fig, ax = plt.subplots(figsize=(13, 4))

    bar_colors = [IC_COLOR if v >= 0 else IC_NEGATIVE_COLOR for v in ic]
    ax.bar(ic.index, ic.values, color=bar_colors, alpha=0.55, width=20, label="Monthly IC")
    rolling = ic.rolling(rolling_window).mean()
    ax.plot(ic.index, rolling, color=ROLLING_COLOR, linewidth=2.0,
            label=f"{rolling_window}m Rolling Mean")
    ax.axhline(0, color="black", linewidth=0.8, zorder=3)

    mean_ic = ic.mean()
    icir = ic.mean() / ic.std()
    pct_positive = (ic > 0).mean()

    ax.set_title(
        f"Information Coefficient (Spearman ρ)   |   "
        f"Mean IC: {mean_ic:.3f}   ICIR: {icir:.2f}   % Positive: {pct_positive:.0%}"
    )
    ax.set_ylabel("IC")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save:
        _save(fig, "ic_series")
    return fig


def plot_quintile_returns(quintile_ann_returns: pd.Series, save: bool = True) -> plt.Figure:
    """
    Bar chart of annualised net return by momentum quintile.
    Q1 = lowest momentum (short leg), Q5 = highest (long leg).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        quintile_ann_returns.index,
        quintile_ann_returns.values * 100,
        color=QUINTILE_COLORS,
        edgecolor="white",
        linewidth=1.5,
        width=0.6,
        zorder=3,
    )
    for bar, val in zip(bars, quintile_ann_returns.values):
        y_pos = bar.get_height() + (0.3 if val >= 0 else -1.2)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_title(
        "Annualised Net Return by Momentum Quintile\n"
        "Q1 = lowest momentum  ·  Q5 = highest momentum  ·  10bps one-way cost"
    )
    ax.set_ylabel("Annualised Return (%)")
    ax.set_xlabel("Quintile")
    fig.tight_layout()

    if save:
        _save(fig, "quintile_returns")
    return fig


def plot_cumulative_with_drawdown(
    ls_returns: pd.Series,
    benchmark_returns: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """
    Two-panel chart: cumulative L/S portfolio vs SPY benchmark (top),
    L/S drawdown series (bottom).
    """
    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_ret = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_ret)

    common = ls_returns.index.intersection(benchmark_returns.index)
    ls = ls_returns.loc[common].dropna()
    bm = benchmark_returns.loc[common].dropna()

    ls_cum = (1 + ls).cumprod()
    bm_cum = (1 + bm).cumprod()

    ax_ret.plot(ls_cum.index, ls_cum.values, color=LS_COLOR, linewidth=2.0,
                label="L/S Momentum (Q5 − Q1)", zorder=3)
    ax_ret.plot(bm_cum.index, bm_cum.values, color=BENCHMARK_COLOR, linewidth=1.5,
                linestyle="--", label="SPY Benchmark", zorder=2)
    ax_ret.axhline(1, color="black", linewidth=0.5, zorder=1)
    ax_ret.set_ylabel("Cumulative Return  (1 = start)")
    ax_ret.set_title("Long/Short Momentum Portfolio  vs  SPY Benchmark")
    ax_ret.legend(loc="upper left")
    plt.setp(ax_ret.get_xticklabels(), visible=False)

    # Drawdown panel
    drawdown = ls_cum / ls_cum.cummax() - 1
    ax_dd.fill_between(drawdown.index, drawdown.values * 100, 0,
                       color=LS_COLOR, alpha=0.45, zorder=2)
    ax_dd.plot(drawdown.index, drawdown.values * 100, color=LS_COLOR,
               linewidth=0.8, zorder=3)
    ax_dd.axhline(0, color="black", linewidth=0.5)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("")

    fig.tight_layout()
    if save:
        _save(fig, "cumulative_returns")
    return fig


def plot_rolling_icir(ic: pd.Series, window: int = 12, save: bool = True) -> plt.Figure:
    """
    Rolling ICIR over time. A practitioner threshold of 0.5 is shown as a dashed line.
    Periods below zero indicate the signal was not adding value.
    """
    rolling_icir = ic.rolling(window).mean() / ic.rolling(window).std()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(rolling_icir.index, rolling_icir.values, color=IC_COLOR, linewidth=1.8, zorder=3)
    ax.fill_between(rolling_icir.index, rolling_icir.values.clip(min=0), 0,
                    color="green", alpha=0.15, zorder=2)
    ax.fill_between(rolling_icir.index, rolling_icir.values.clip(max=0), 0,
                    color="red", alpha=0.15, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax.axhline(0.5, color="green", linewidth=1.0, linestyle="--", alpha=0.7,
               label="ICIR = 0.5 (practitioner threshold)", zorder=3)
    ax.axhline(-0.5, color="red", linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)

    ax.set_title(f"{window}-Month Rolling ICIR")
    ax.set_ylabel("ICIR")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save:
        _save(fig, "rolling_icir")
    return fig


def generate_all(
    ic: pd.Series,
    quintile_returns: pd.DataFrame,
    benchmark_monthly: pd.Series,
) -> None:
    """Generate and save all four figures. Called from the report notebook."""
    print("Generating figures...")

    ls_returns = quintile_returns["Q5"] - quintile_returns["Q1"]

    n = len(quintile_returns)
    ann_factor = 12 / n
    quintile_ann = quintile_returns.apply(
        lambda col: (1 + col.dropna()).prod() ** (12 / len(col.dropna())) - 1
    )

    plot_ic_series(ic)
    plot_quintile_returns(quintile_ann)
    plot_cumulative_with_drawdown(ls_returns, benchmark_monthly)
    plot_rolling_icir(ic)

    print("Done. All figures saved to figures/")
