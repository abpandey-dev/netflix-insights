"""Charts for the README. One hue for single-series charts; two fixed hues (blue, orange) when comparing two."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbars(labels, values, title, xlabel, fmt="{:.0f}", color=BLUE, note=None):
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(labels) + 1.5))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, color=color, height=0.62)
    span = max(values) or 1
    for yi, v in zip(y, values):
        ax.text(v + span * 0.01, yi, fmt.format(v), va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, span * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if note:
        ax.text(0, -0.13 - 0.02 * (8 - min(len(labels), 8)), note, transform=ax.transAxes, fontsize=8.5, color=MUTED)
    _grid(ax, "x")
    fig.tight_layout()
    return fig


def decay(d: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for col, color, label in [("show", ORANGE, "series"), ("movie", BLUE, "films")]:
        ax.plot(d.index, d[col] * 100, color=color, linewidth=2, marker="o", markersize=7, label=label)
        ax.annotate(f"{label}: {d[col].iloc[3] * 100:.0f}% in week 4", (4, d[col].iloc[3] * 100),
                    textcoords="offset points", xytext=(-8, 12) if col == "show" else (-70, -22), fontsize=9,
                    color=INK2)
    ax.set_xticks(d.index)
    ax.set_xlabel("week in the global Top 10")
    ax.set_ylabel("weekly views, % of the title's best week")
    ax.set_ylim(0, 110)
    ax.set_title("Interest fades fast after release (median title)")
    ax.legend(frameon=False, loc="upper right")
    _grid(ax)
    fig.tight_layout()
    return fig


def weekday(w: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    days = [d[:3] for d in w.index]
    for ax, col, title, fmt in [(axes[0], "share_of_releases", "Share of Netflix releases", "{:.0%}"),
                                (axes[1], "median_views", "Median views per release (millions)", "{:.1f}")]:
        vals = w[col] / (1e6 if col == "median_views" else 1)
        ax.bar(days, vals, color=BLUE, width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + vals.max() * 0.02, fmt.format(v), ha="center", fontsize=8.5, color=INK2)
        ax.set_title(title)
        ax.set_ylim(0, vals.max() * 1.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.suptitle("Release day: Jan-Jun 2026 releases", x=0.01, ha="left", fontsize=9, color=MUTED, y=0.02)
    fig.tight_layout()
    return fig


def heatmap(m: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(m)), m.columns, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(m)), m.index, fontsize=8.5)
    for i in range(len(m)):
        for j in range(len(m)):
            v = m.values[i, j]
            if i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else INK2)
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("share of Top 10 titles in common (Jaccard)", color=INK2)
    fig.tight_layout()
    return fig


def recommenders(ev: pd.DataFrame):
    order = ["content", "popularity", "collaborative"]
    labels = {"content": "Content: similar genre,\ndirector, decade",
              "popularity": "Most popular\n(baseline)",
              "collaborative": "Collaborative: watched by\nthe same audiences"}
    vals = [ev.loc[o, "recall_at_k"] * 100 for o in order]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    colors = [BLUE if o == "collaborative" else "#a9c8ee" for o in order]
    ax.bar([labels[o] for o in order], vals, color=colors, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=10, color=INK2)
    ax.set_ylabel("hidden titles recovered (recall@20, %)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Which recommender predicts what a country watches?")
    _grid(ax)
    fig.tight_layout()
    return fig


def rating_views(r: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    vals = r["median_views"] / 1e6
    ax.bar(r.index.astype(str), vals, color=BLUE, width=0.55)
    for i, (v, n) in enumerate(zip(vals, r["titles"])):
        ax.text(i, v + vals.max() * 0.02, f"{v:.1f}M\n({n:,} titles)", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("IMDb rating")
    ax.set_ylabel("median views (millions)")
    ax.set_ylim(0, vals.max() * 1.35)
    ax.set_title(title)
    _grid(ax)
    fig.tight_layout()
    return fig


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
