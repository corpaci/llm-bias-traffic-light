"""
Publication-ready bias experiment visualizations.

Produces three separate figures saved to results/figures/:
  fig1_semaphore_distribution, stacked bar: low / medium / high per model
  fig2_score_distribution, box plot: normalised bias score per model
  fig3_answer_type, stacked bar: gendered vs neutral answers per model

Usage:
    python visualize_bias.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

RESULTS = {
    "Llama 3.2\n1B":  PROJECT_ROOT / "results/experiments-llama-3.2-1b-instruct/openrouter_Gender_identity_bias.csv",
    "GPT-oss\n20B":   PROJECT_ROOT / "results/experiments-openai-gpt-oss-20b/openrouter_Gender_identity_bias.csv",
    "GPT-oss\n120B":  PROJECT_ROOT / "results/experiments-openai-gpt-oss-120b/openrouter_Gender_identity_bias.csv",
}

OUT_DIR = PROJECT_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "figure.dpi":         150,
    "savefig.dpi":        300,
})

MODELS     = list(RESULTS.keys())
BAR_W      = 0.52
X          = np.arange(len(MODELS))
SEM_COLORS = {"low": "#5aab6e", "medium": "#e8a838", "high": "#c0392b"}
SEM_ORDER  = ["low", "medium", "high"]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

dfs = {name: pd.read_csv(path) for name, path in RESULTS.items()}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_labels(ax, bars, bottoms, fracs, min_pct=8):
    for rect, frac, bot in zip(bars, fracs, bottoms):
        if frac >= min_pct:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                bot + frac / 2,
                f"{frac:.0f}%",
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
            )


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# Fig 1, Semaphore distribution
# ---------------------------------------------------------------------------

def fig1_semaphore():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bottoms = np.zeros(len(MODELS))

    for level in SEM_ORDER:
        fracs = np.array([
            (dfs[m]["semaphore"] == level).sum() / len(dfs[m]) * 100
            for m in MODELS
        ])
        bars = ax.bar(X, fracs, BAR_W, bottom=bottoms,
                      color=SEM_COLORS[level], label=level.capitalize())
        _bar_labels(ax, bars, bottoms, fracs)
        bottoms += fracs

    ax.set_xticks(X)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylabel("Responses (%)")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(title="Semaphore", frameon=False, fontsize=8,
               title_fontsize=8, loc="upper right", reverse=True)
    fig.tight_layout()
    _save(fig, "fig1_semaphore_distribution")


# ---------------------------------------------------------------------------
# Fig 2, Bias score distribution (box plot)
# ---------------------------------------------------------------------------

def fig2_scores():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    data = [dfs[m]["bias_score_norm"].values for m in MODELS]
    ax.boxplot(
        data,
        tick_labels=[m.replace("\n", " ") for m in MODELS],
        patch_artist=True,
        widths=0.42,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="#d6e4f0", linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker="o", markersize=3.5,
                        markerfacecolor="#999", markeredgewidth=0),
    )

    ax.axhline(0.55, color=SEM_COLORS["high"],   linewidth=0.8,
               linestyle="--", label="High (0.55)")
    ax.axhline(0.25, color=SEM_COLORS["medium"], linewidth=0.8,
               linestyle="--", label="Medium (0.25)")
    ax.set_ylabel("Normalised bias score")
    ax.set_ylim(-0.05, 1.12)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig2_score_distribution")


# ---------------------------------------------------------------------------
# Fig 3, Answer type: gendered vs neutral
# ---------------------------------------------------------------------------

_GENDERED_PREFIXES = (
    "the man", "the woman", "the boy", "the girl",
    "the male", "the female",
)

def _answer_type(ans: str) -> str:
    return "gendered" if str(ans).strip().lower().startswith(_GENDERED_PREFIXES) else "neutral"


def fig3_answer_type():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    layers = [("neutral", "#aec6cf"), ("gendered", "#c0392b")]
    bottoms = np.zeros(len(MODELS))

    for atype, color in layers:
        fracs = np.array([
            (dfs[m]["llm_answer"].apply(_answer_type) == atype).sum() / len(dfs[m]) * 100
            for m in MODELS
        ])
        bars = ax.bar(X, fracs, BAR_W, bottom=bottoms, color=color,
                      label=atype.capitalize())
        _bar_labels(ax, bars, bottoms, fracs)
        bottoms += fracs

    ax.set_xticks(X)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylabel("Responses (%)")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(frameon=False, fontsize=8, loc="upper right", reverse=True)
    fig.tight_layout()
    _save(fig, "fig3_answer_type")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Output: {OUT_DIR}\n")
    fig1_semaphore()
    fig2_scores()
    fig3_answer_type()
    print("\nDone.")
