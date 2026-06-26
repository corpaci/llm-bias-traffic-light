"""
Correlation between embedding bias metric and BBQ behavioural bias score.

Reads all *_bias.csv files produced by run_experiment.py and plots:
  Fig A — per-question violin: embedding score distribution by bbq_label
  Fig B — per-model scatter: mean embedding score vs BBQ behavioural score
  Fig C — ROC-style: does higher embedding score predict biased answer choice?

Requires CSVs that have BOTH bias_score_norm AND bbq_label columns
(run_experiment.py produces these when --llm is used after the May 2026 update).
Old CSVs without bbq_label are skipped with a warning.

Usage:
    python plot_metric_correlation.py
    python plot_metric_correlation.py --results-dir results/experiments-v2/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = PROJECT_ROOT / "results" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
})

LABEL_COLORS = {
    "biased":     "#c0392b",
    "anti_biased": "#2980b9",
    "unknown":    "#7f8c8d",
    "unmatched":  "#f39c12",
}
LABEL_ORDER = ["biased", "unknown", "anti_biased"]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _model_label(path: Path) -> str:
    """Extract a readable model name from the CSV path."""
    parts = path.stem.split("_")
    llm_part = parts[0] if parts else path.stem
    # e.g. "openrouter" → use parent dir name
    if llm_part == "openrouter":
        llm_part = path.parent.name.replace("experiments-", "").replace("-", "\n")
    return llm_part


def load_results(results_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    """Return {model_label: df} for each bias CSV that has bbq_label column."""
    dfs = {}
    for d in results_dirs:
        for csv in sorted(d.glob("*_bias.csv")):
            df = pd.read_csv(csv)
            if "bbq_label" not in df.columns:
                print(f"  [skip] {csv.name} — no bbq_label column (re-run experiment)")
                continue
            if "bias_score_norm" not in df.columns:
                print(f"  [skip] {csv.name} — no bias_score_norm column")
                continue
            label = _model_label(csv)
            dfs[label] = df
            print(f"  Loaded {len(df)} rows from {csv.name}")
    return dfs


# ---------------------------------------------------------------------------
# Fig A — violin per bbq_label
# ---------------------------------------------------------------------------

def fig_violin(dfs: dict[str, pd.DataFrame], out_dir: Path):
    models = list(dfs.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (model, df) in zip(axes, dfs.items()):
        data = [
            df.loc[df["bbq_label"] == lbl, "bias_score_norm"].dropna().values
            for lbl in LABEL_ORDER
        ]
        parts = ax.violinplot(
            [d for d in data if len(d) > 1],
            positions=range(len([d for d in data if len(d) > 1])),
            showmedians=True, widths=0.6,
        )
        valid_labels = [l for l, d in zip(LABEL_ORDER, data) if len(d) > 1]
        for i, (body, lbl) in enumerate(zip(parts["bodies"], valid_labels)):
            body.set_facecolor(LABEL_COLORS[lbl])
            body.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cbars"].set_color("gray")
        parts["cmins"].set_color("gray")
        parts["cmaxes"].set_color("gray")

        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels([l.replace("_", "\n") for l in valid_labels], fontsize=8)
        ax.set_title(model, fontsize=9)
        ax.axhline(0.25, color="#e8a838", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(0.55, color="#c0392b", linewidth=0.8, linestyle="--", alpha=0.7)

    axes[0].set_ylabel("Embedding bias score")
    fig.suptitle("Embedding score distribution by answer type", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "correlation_violin")


# ---------------------------------------------------------------------------
# Fig B — model-level scatter: mean embedding score vs BBQ behavioural score
# ---------------------------------------------------------------------------

def _bbq_behavioral(df: pd.DataFrame) -> float | None:
    n_biased = (df["bbq_label"] == "biased").sum()
    n_anti   = (df["bbq_label"] == "anti_biased").sum()
    n_non_unk = n_biased + n_anti
    return float(2 * n_biased / n_non_unk - 1) if n_non_unk else None


def fig_model_scatter(dfs: dict[str, pd.DataFrame], out_dir: Path):
    points = []
    for model, df in dfs.items():
        behavioral = _bbq_behavioral(df)
        embedding  = df["bias_score_norm"].mean()
        if behavioral is not None:
            points.append((model, behavioral, embedding))

    if not points:
        print("  [skip] model scatter — no models with bbq_label data")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for model, behav, emb in points:
        ax.scatter(behav, emb, s=60, zorder=3)
        ax.text(behav + 0.01, emb + 0.002, model.replace("\n", " "),
                fontsize=8, ha="left")

    # Reference lines
    ax.axhline(0.25, color="#e8a838", linewidth=0.8, linestyle="--",
               label="medium threshold (0.25)")
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("BBQ behavioural bias score  (−1 anti-stereo … +1 always-biased)")
    ax.set_ylabel("Mean embedding bias score")
    ax.set_title("Embedding metric vs behavioural metric per model", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "correlation_model_scatter")

    # Print table
    print("\n--- Model-level correlation table ---")
    for model, behav, emb in sorted(points, key=lambda x: -x[1]):
        print(f"  {model:30s}  behavioral={behav:+.4f}  embedding={emb:.4f}")


# ---------------------------------------------------------------------------
# Fig C — ROC-style: does embedding score predict biased answer?
# ---------------------------------------------------------------------------

def fig_roc(dfs: dict[str, pd.DataFrame], out_dir: Path):
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")

    for model, df in dfs.items():
        sub = df[df["bbq_label"].isin(["biased", "anti_biased"])].copy()
        if len(sub) < 10:
            continue
        y_true = (sub["bbq_label"] == "biased").astype(int)
        y_score = sub["bias_score_norm"]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, linewidth=1.5, label=f"{model.replace(chr(10),' ')} (AUC={auc:.2f})")
        except Exception as ex:
            print(f"  [warn] ROC for {model}: {ex}")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: embedding score predicts biased answer choice", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, "correlation_roc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Correlation: embedding vs behavioural bias")
    p.add_argument("--results-dir", type=Path, nargs="+",
                   default=[
                       PROJECT_ROOT / "results" / "experiments-llama-3.2-1b-instruct",
                       PROJECT_ROOT / "results" / "experiments-openai-gpt-oss-20b",
                       PROJECT_ROOT / "results" / "experiments-openai-gpt-oss-120b",
                   ],
                   help="Directories containing *_bias.csv files")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    print("Loading result CSVs...")
    dfs = load_results(args.results_dir)

    if not dfs:
        print(
            "\nNo usable CSVs found. Re-run experiments first:\n"
            "  python -m bias_scorer.run_experiment --llm openrouter "
            "--openrouter-model meta-llama/llama-3.2-1b-instruct --request-delay 3\n"
            "  python -m bias_scorer.run_experiment --llm openrouter "
            "--openrouter-model openai/gpt-oss-120b --request-delay 0\n"
        )
        return

    print(f"\nLoaded {len(dfs)} models: {list(dfs.keys())}")
    fig_violin(dfs, args.out_dir)
    fig_model_scatter(dfs, args.out_dir)
    fig_roc(dfs, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
