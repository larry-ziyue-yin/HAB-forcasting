#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization for ILW CI_cyano after QC.
Generates:
- CONUS daily & monthly time-series plots
- Per-lake (Great Lakes) daily & monthly time series with coverage
- Faceted (5-lake) small multiples
- Validity heatmap (lake x date)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
ROOT = Path("/dkucc/home/zy166/HAB-forecasting")
QC_DIR = ROOT / "datasets/processed/qc"
FIG_DIR = ROOT / "visualization/ILW"
FIG_DIR.mkdir(parents=True, exist_ok=True)

P_CONUS_DAILY_CLEAN   = QC_DIR / "conus_daily_clean.csv"
P_CONUS_MONTHLY_CLEAN = QC_DIR / "conus_monthly_clean.csv"
P_GL_DAILY_CLEAN      = QC_DIR / "greatlakes_daily_clean.parquet"
P_GL_MONTHLY_CLEAN    = QC_DIR / "greatlakes_monthly_clean.parquet"

# -----------------------------
# Helpers
# -----------------------------
def _rolling(s, win=7):
    """Simple rolling mean with center window, preserving NaNs at edges."""
    try:
        return s.rolling(win, center=True, min_periods=max(1, win // 2)).mean()
    except Exception:
        return s

def _nice_ax(ax, title=None, xlabel=None, ylabel=None, grid=True):
    """Minimal axis styling."""
    if title:  ax.set_title(title, fontsize=12)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if grid:   ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

def _lake_name_map(df):
    """Resolve a friendly lake name column."""
    for c in ["lake_name", "Lake_name", "name", "Name"]:
        if c in df.columns:
            return c
    return None

# -----------------------------
# 1) CONUS time series
# -----------------------------
def plot_conus_timeseries():
    """Plot CONUS daily & monthly CI time series."""
    if not P_CONUS_DAILY_CLEAN.exists() and not P_CONUS_MONTHLY_CLEAN.exists():
        print("[SKIP] No CONUS files found.")
        return

    # Daily
    if P_CONUS_DAILY_CLEAN.exists():
        d = pd.read_csv(P_CONUS_DAILY_CLEAN, parse_dates=["date"])
        d = d.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        if "CI_mean" in d.columns:
            ax.plot(d["date"], d["CI_mean"], linewidth=1.0, label="CI_mean (daily)")
            ax.plot(d["date"], _rolling(d["CI_mean"], 7), linewidth=2.0, label="CI_mean 7D MA")
        if "CI_p90" in d.columns:
            ax.plot(d["date"], _rolling(d["CI_p90"], 7), linewidth=1.5, linestyle=":", label="CI_p90 7D MA")
        _nice_ax(ax, title="CONUS — Daily CI_cyano (with 7D moving average)", xlabel="Date", ylabel="CI")
        ax.legend()
        fig.tight_layout()
        out = FIG_DIR / "conus_daily_timeseries.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print("[OK] Saved", out)

    # Monthly
    if P_CONUS_MONTHLY_CLEAN.exists():
        m = pd.read_csv(P_CONUS_MONTHLY_CLEAN, parse_dates=["date"])
        m = m.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        if "CI_mean" in m.columns:
            ax.plot(m["date"], m["CI_mean"], marker="o", linewidth=2.0, label="CI_mean (monthly)")
        if "CI_p90" in m.columns:
            ax.plot(m["date"], m["CI_p90"], marker="s", linewidth=1.8, linestyle="--", label="CI_p90 (monthly)")
        _nice_ax(ax, title="CONUS — Monthly CI_cyano", xlabel="Month", ylabel="CI")
        ax.legend()
        fig.tight_layout()
        out = FIG_DIR / "conus_monthly_timeseries.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print("[OK] Saved", out)

# -----------------------------
# 2) Great Lakes time series
# -----------------------------
def plot_lake_timeseries_daily():
    """Per-lake time series with coverage as secondary axis."""
    if not P_GL_DAILY_CLEAN.exists():
        print("[SKIP] No Great Lakes daily parquet.")
        return
    df = pd.read_parquet(P_GL_DAILY_CLEAN)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    name_col = _lake_name_map(df)
    lakes = sorted(df["lake_id"].unique().tolist())

    for lid in lakes:
        g = df[df["lake_id"] == lid].sort_values("date")
        title = f"Lake {g[name_col].iloc[0] if name_col else lid} — Daily CI & Coverage"
        fig, ax1 = plt.subplots(figsize=(10, 4))
        # CI curves
        has_mean = "CI_mean" in g.columns and g["CI_mean"].notna().any()
        has_p90  = "CI_p90" in g.columns and g["CI_p90"].notna().any()
        if not (has_mean or has_p90):
            ax1.text(0.5, 0.5, "no finite CI values", ha="center", va="center", transform=ax1.transAxes)
        if has_mean:
            ax1.plot(g["date"], g["CI_mean"], linewidth=1.0, label="CI_mean")
            ax1.plot(g["date"], _rolling(g["CI_mean"], 7), linewidth=2.0, label="CI_mean 7D MA")
        if has_p90:
            ax1.plot(g["date"], _rolling(g["CI_p90"], 7), linewidth=1.5, linestyle=":", label="CI_p90 7D MA")
        _nice_ax(ax1, title=title, xlabel="Date", ylabel="CI")
        # Coverage on twin axis
        ax2 = ax1.twinx()
        if "pct_valid_geom" in g.columns and g["pct_valid_geom"].notna().any():
            ax2.plot(g["date"], g["pct_valid_geom"], linewidth=1.0, alpha=0.6, label="Coverage (geom)")
            ax2.set_ylabel("Coverage ratio")
            ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        out = FIG_DIR / f"lake_{lid}_daily_timeseries.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print("[OK] Saved", out)

def plot_lake_timeseries_monthly():
    """Plot monthly CI time series for each Great Lake with robust checks."""
    if not P_GL_MONTHLY_CLEAN.exists():
        print("[SKIP] No Great Lakes monthly parquet.")
        return
    df = pd.read_parquet(P_GL_MONTHLY_CLEAN)
    if "date" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "qc_is_valid" in df.columns:
        before = len(df)
        df = df[df["qc_is_valid"] == 1]
        print(f"[INFO] Filtered by qc_is_valid==1: {before} -> {len(df)} rows")

    has_mean = "CI_mean" in df.columns
    has_p90 = "CI_p90" in df.columns
    if not (has_mean or has_p90):
        print("[WARN] Neither CI_mean nor CI_p90 present.")
        return

    name_col = _lake_name_map(df)
    lakes = sorted(df["lake_id"].dropna().unique().tolist())

    for lid in lakes:
        g = df[df["lake_id"] == lid].sort_values("date")
        title_name = g[name_col].iloc[0] if name_col and len(g[name_col].dropna()) else lid
        fig, ax = plt.subplots(figsize=(10, 4))
        plotted = False
        if has_mean and g["CI_mean"].notna().any():
            ax.plot(g["date"], g["CI_mean"], marker="o", linewidth=2.0, label="CI_mean (monthly)")
            plotted = True
        if has_p90 and g["CI_p90"].notna().any():
            ax.plot(g["date"], g["CI_p90"], marker="s", linewidth=1.6, linestyle="--", label="CI_p90 (monthly)")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "no finite CI values", ha="center", va="center", transform=ax.transAxes)
        _nice_ax(ax, title=f"Lake {title_name} — Monthly CI", xlabel="Month", ylabel="CI")
        if plotted:
            ax.legend()
        fig.tight_layout()
        out = FIG_DIR / f"lake_{lid}_monthly_timeseries.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print("[OK] Saved", out)

# -----------------------------
# 3) Faceted small multiples & validity heatmap
# -----------------------------
def plot_lake_facets_daily():
    """Five-lake small multiples (CI_mean 7D MA + coverage shading)."""
    if not P_GL_DAILY_CLEAN.exists():
        print("[SKIP] No Great Lakes daily parquet.")
        return
    df = pd.read_parquet(P_GL_DAILY_CLEAN)
    df["date"] = pd.to_datetime(df["date"])
    name_col = _lake_name_map(df)
    desired = ["Superior", "Michigan", "Huron", "Erie", "Ontario"]
    if name_col:
        order = []
        for nm in desired:
            subset = df[df[name_col].str.fullmatch(nm, case=False, na=False)]
            if not subset.empty:
                order.extend(subset["lake_id"].unique().tolist())
        lakes = order or sorted(df["lake_id"].unique().tolist())[:5]
    else:
        lakes = sorted(df["lake_id"].unique().tolist())[:5]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    axes = axes.ravel()
    for i, lid in enumerate(lakes[:5]):
        ax = axes[i]
        g = df[df["lake_id"] == lid].sort_values("date")
        label = g[name_col].iloc[0] if name_col else str(lid)
        if "pct_valid_geom" in g.columns:
            ycov = (g["pct_valid_geom"].clip(0, 1.0) * (g["CI_mean"].max() * 0.2)).fillna(0.0)
            ax.fill_between(g["date"], 0, ycov, alpha=0.15, step="pre")
        if "CI_mean" in g.columns:
            ax.plot(g["date"], _rolling(g["CI_mean"], 7), linewidth=2.0, label="CI_mean 7D MA")
        if "CI_p90" in g.columns:
            ax.plot(g["date"], _rolling(g["CI_p90"], 7), linewidth=1.2, linestyle="--", label="CI_p90 7D MA")
        _nice_ax(ax, title=label, xlabel=None, ylabel="CI")
        if i == 0:
            ax.legend(fontsize=9)
    for j in range(len(lakes), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Great Lakes — Daily CI (7D MA) with coverage shading", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = FIG_DIR / "greatlakes_daily_facets.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("[OK] Saved", out)

def plot_validity_heatmap_daily():
    """Heatmap of qc_is_valid by (lake x date)."""
    if not P_GL_DAILY_CLEAN.exists():
        print("[SKIP] No Great Lakes daily parquet.")
        return
    df = pd.read_parquet(P_GL_DAILY_CLEAN)
    df["date"] = pd.to_datetime(df["date"])
    name_col = _lake_name_map(df)
    if name_col:
        id2name = (df.dropna(subset=[name_col])
                     .drop_duplicates(subset=["lake_id"])[["lake_id", name_col]]
                     .set_index("lake_id")[name_col].to_dict())
    else:
        id2name = {}
    mat = df.pivot_table(index="lake_id", columns="date", values="qc_is_valid", aggfunc="max")
    if id2name:
        mat.index = [f"{lid} - {id2name.get(lid, '')}" for lid in mat.index]
    mat = mat.sort_index()
    fig, ax = plt.subplots(figsize=(12, 3 + 0.35 * len(mat)))
    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=9)
    dates = mat.columns.to_pydatetime()
    step = max(1, len(dates) // 12)
    xticks = np.arange(0, len(dates), step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([dates[i].strftime("%Y-%m-%d") for i in xticks], rotation=45, ha="right", fontsize=8)
    _nice_ax(ax, title="Great Lakes — Daily validity (qc_is_valid) heatmap", xlabel="Date", ylabel="Lake")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("qc_is_valid (1=valid, 0=invalid)", rotation=270, labelpad=12)
    fig.tight_layout()
    out = FIG_DIR / "greatlakes_daily_validity_heatmap.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("[OK] Saved", out)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    plot_conus_timeseries()
    plot_lake_timeseries_daily()
    plot_lake_facets_daily()
    plot_validity_heatmap_daily()
    plot_lake_timeseries_monthly()