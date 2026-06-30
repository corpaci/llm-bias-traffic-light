"""
Batch runner: geometry analysis across all BBQ categories.

Runs run_bbq_geometry for every category, both raw and normalised,
then merges results into a single summary CSV + heatmap for the paper.

Usage:
    python run_all_geometry.py                   # all 11 categories
    python run_all_geometry.py --categories Gender_identity Age Religion
    python run_all_geometry.py --templates answer_only cq_answer --no-plots
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bias_scorer.embedder import Embedder
from run_bbq_geometry import (
    TEMPLATES, run_category,
    BBQ_DIR, OUT_DIR,
)

ALL_CATEGORIES = [p.stem for p in sorted((BBQ_DIR).glob("*.jsonl"))]


def parse_args():
    p = argparse.ArgumentParser(description="Batch BBQ geometry analysis")
    p.add_argument("--categories", nargs="+", default=ALL_CATEGORIES)
    p.add_argument("--templates", nargs="+", default=["answer_only", "cq_answer"],
                   choices=TEMPLATES,
                   help="Subset of templates (default: answer_only + cq_answer)")
    p.add_argument("--model-name", default="all-MiniLM-L6-v2")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def make_paper_table(df: pd.DataFrame, out_dir: Path):
    """Produce the geometry summary table used in the paper."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })

    metrics = ["accuracy", "roc_auc", "cohens_d_pc1", "centroid_sim_bia_ant"]
    labels  = ["Accuracy", "AUC", "|Cohen's d|", "Centroid sim (b/ab)"]

    for tmpl in df["template"].unique():
        sub = df[df["template"] == tmpl].copy()
        sub = sub.set_index("category")[metrics]

        fig, axes = plt.subplots(1, 4, figsize=(13, max(3, len(sub) * 0.45 + 1.5)))
        for ax, col, lbl in zip(axes, metrics, labels):
            vals = sub[col].abs() if col == "cohens_d_pc1" else sub[col]
            bars = ax.barh(sub.index, vals, color="#2980b9", alpha=0.75)
            if col in ("accuracy", "roc_auc"):
                ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", label="chance")
            ax.set_title(lbl, fontsize=9)
            ax.set_xlim(0, 1.05)
            ax.tick_params(labelsize=8)
            for bar, v in zip(bars, vals):
                ax.text(min(v + 0.01, 1.0), bar.get_y() + bar.get_height() / 2,
                        f"{v:.2f}", va="center", fontsize=7)

        fig.suptitle(f"BBQ geometry, template: {tmpl}", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"geometry_all_{tmpl}.{ext}", bbox_inches="tight")
        plt.close(fig)
        print(f"  → geometry_all_{tmpl}.pdf / .png")


def main():
    args = parse_args()
    print(f"Loading embedder: {args.model_name}")
    embedder = Embedder(args.model_name)

    all_raw  = []
    all_norm = []

    for cat in args.categories:
        bbq_file = BBQ_DIR / f"{cat}.jsonl"
        if not bbq_file.exists():
            print(f"  [skip] {cat}, file not found")
            continue

        # Raw answers
        df_raw = run_category(
            category=cat, templates=args.templates,
            max_examples=args.max_examples, no_plots=args.no_plots,
            embedder=embedder, binary=True, normalize=False,
        )
        if not df_raw.empty:
            df_raw["variant"] = "raw"
            all_raw.append(df_raw)

        # Normalised answers
        df_norm = run_category(
            category=cat, templates=args.templates,
            max_examples=args.max_examples, no_plots=args.no_plots,
            embedder=embedder, binary=True, normalize=True,
        )
        if not df_norm.empty:
            df_norm["variant"] = "normalized"
            all_norm.append(df_norm)

    if not all_raw:
        print("No results produced.")
        return

    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(all_raw + all_norm, ignore_index=True)
    csv_path = OUT_DIR / "all_categories_geometry.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nCombined results → {csv_path}")

    # Paper table: normalized variant only
    norm_df = pd.concat(all_norm, ignore_index=True) if all_norm else pd.DataFrame()
    if not norm_df.empty:
        make_paper_table(norm_df, figures_dir)

    # Print summary
    print("\n--- Normalised geometry summary (cq_answer template) ---")
    sub = combined[(combined["variant"] == "normalized") &
                   (combined["template"] == "cq_answer")]
    if not sub.empty:
        cols = ["category", "accuracy", "roc_auc", "cohens_d_pc1", "centroid_sim_bia_ant"]
        available = [c for c in cols if c in sub.columns]
        print(sub[available].to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
