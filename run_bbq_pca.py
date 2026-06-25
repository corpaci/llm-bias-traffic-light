"""
BBQ embedding space PCA — visualising bias regions and anchor directions.

Implements the Apr 17 idea: project BBQ embeddings into 2D and show whether
biased/anti_biased regions are spatially separable, and where the anchor
vectors point in that space.

What it shows
-------------
1. Fit PCA on all (cq + normalised_answer) embeddings for the category.
2. Plot biased / anti_biased / unknown as coloured scatter.
3. Overlay the gender (or general) anchor vectors projected into the same 2D space.
4. If an LLM results CSV is given (--llm-csv), also plot where LLM answers land.

This visualisation is Figure X in the paper: it shows geometrically that the
anchor direction aligns with the bias direction in the embedding space.

Usage
-----
    python run_bbq_pca.py --category Gender_identity
    python run_bbq_pca.py --category Age --no-anchor
    python run_bbq_pca.py --category Gender_identity \\
        --llm-csv results/experiments-openai-gpt-oss-120b/openrouter_Gender_identity_bias.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bias_scorer.embedder import Embedder
from bias_scorer.anchors import compute_anchors, _answer_roles
from run_bbq_geometry import load_bbq_rows, embed_rows, LABEL_COLORS, LABEL_MARKERS, LABELS
import json

OUT_DIR = PROJECT_ROOT / "results" / "bbq_geometry"
BBQ_DIR = PROJECT_ROOT / "BBQ_Data"

# Colours / markers for gender-role mode
ROLE_COLORS  = {"male": "#2980b9", "female": "#c0392b", "unknown": "#7f8c8d"}
ROLE_MARKERS = {"male": "o",       "female": "s",       "unknown": "^"}


def load_role_rows(category: str, max_examples: int | None = None) -> list[dict]:
    """
    Load BBQ rows labelled by GENDER ROLE (male / female / unknown) rather
    than by biased / anti_biased.  Role is read directly from answer_info.
    Only works for Gender_identity (or categories with male/female labels).
    """
    path = BBQ_DIR / f"{category}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line.strip())
            if e.get("context_condition") != "ambig":
                continue
            roles = _answer_roles(e)
            if roles is None:
                continue
            cq = e["context"] + " " + e["question"]
            base = {"entry_id": e["example_id"], "context": e["context"],
                    "question": e["question"], "cq": cq}
            for role in ("male", "female", "unknown"):
                rows.append({**base, "answer": roles[role], "label": role})
            if max_examples and len(rows) // 3 >= max_examples:
                break
    return rows


# ---------------------------------------------------------------------------
# Anchor projection helpers
# ---------------------------------------------------------------------------

def _project_anchor(anchor: torch.Tensor, pca_components: np.ndarray) -> np.ndarray:
    """Project a unit anchor vector into 2D PCA space."""
    v = anchor.float().numpy()
    return pca_components @ v   # shape (2,)


# ---------------------------------------------------------------------------
# Main PCA visualisation
# ---------------------------------------------------------------------------

def _plot_bias_axis_1d(
    embs: torch.Tensor,
    labels: list[str],
    anchors,
    category: str,
    save_dir: Path,
    suffix: str = "",
):
    """
    1D projection onto the male-female anchor axis (= the actual bias metric).
    Shows what the scorer sees, colored by label.
    This is the most honest visualisation: PCA/t-SNE can't separate
    biased/anti_biased because the label is context-dependent, but the
    bias axis projection shows the full score distribution per class.
    """
    import matplotlib.pyplot as plt

    embs_n = Fn.normalize(embs.float(), dim=1)
    scores = ((embs_n @ anchors.male_anchor.float())
              - (embs_n @ anchors.female_anchor.float())
              - anchors.baseline_mean).numpy()
    label_arr = np.array(labels)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    # Left: histogram by label
    ax = axes[0]
    colors = {**LABEL_COLORS, **ROLE_COLORS}
    for lab in sorted(set(labels)):
        mask = label_arr == lab
        if mask.sum() < 2:
            continue
        ax.hist(scores[mask], bins=40, density=True, alpha=0.45,
                color=colors.get(lab, "#888"), label=lab.replace("_", " "),
                histtype="stepfilled")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("bias_corrected  (+ = male lean, − = female lean)")
    ax.set_ylabel("Density")
    ax.set_title(f"{category} — bias axis projection by label", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    # Right: violin / strip
    ax2 = axes[1]
    unique_labs = sorted(set(labels))
    data = [scores[label_arr == lab] for lab in unique_labs]
    parts = ax2.violinplot(data, positions=range(len(unique_labs)),
                           showmedians=True, widths=0.6)
    for body, lab in zip(parts["bodies"], unique_labs):
        body.set_facecolor(colors.get(lab, "#888"))
        body.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("gray")
    ax2.set_xticks(range(len(unique_labs)))
    ax2.set_xticklabels([l.replace("_", "\n") for l in unique_labs], fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("bias_corrected")
    ax2.set_title(f"{category} — bias score violin by label", fontsize=9)

    fig.tight_layout()
    stem = f"{category}_bias_axis_1d{suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(save_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {stem}.pdf / .png")

    # Print summary stats
    print(f"  bias_corrected stats by label:")
    for lab in sorted(set(labels)):
        s = scores[label_arr == lab]
        if len(s) > 0:
            print(f"    {lab:14s}: mean={s.mean():+.5f}  std={s.std():.5f}  "
                  f"median={np.median(s):+.5f}")


def run_pca_plot(
    category: str,
    embedder: Embedder,
    max_examples: int | None,
    show_anchor: bool,
    llm_csv: Path | None,
    save_dir: Path,
    binary: bool = False,
    by_role: bool = False,
):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })

    suffix = ("_role" if by_role else "") + ("_binary" if binary else "")

    if by_role:
        role_desc = "male/female only" if binary else "male/female/unknown"
        print(f"\nLoading BBQ rows for {category} (by gender role: {role_desc})...")
        rows = load_role_rows(category, max_examples=max_examples)
        if binary:
            rows = [r for r in rows if r["label"] != "unknown"]
        if not rows:
            print("  No rows — check category name and that it has male/female answer labels.")
            return
        n_entries = len(rows) // (2 if binary else 3)
        print(f"  {n_entries} entries → {len(rows)} rows")
        # Residual embedding: E(cq + answer) - E(cq)
        # Subtracting the shared context isolates what the answer adds,
        # making the gender direction visible in PCA despite context dominating.
        cq_texts  = [r["cq"]                    for r in rows]
        ans_texts = [r["cq"] + " " + r["answer"] for r in rows]
        e_cq  = embedder.encode(cq_texts).float()
        e_ans = embedder.encode(ans_texts).float()
        embs  = e_ans - e_cq          # residual: answer contribution only
        labels = [r["label"] for r in rows]
        label_colors  = ROLE_COLORS
        label_markers = ROLE_MARKERS
        plot_labels   = ["male", "female", "unknown"]
    else:
        mode_desc = "biased/anti_biased only" if binary else "all labels"
        print(f"\nLoading BBQ rows for {category} ({mode_desc})...")
        rows = load_bbq_rows(category, max_examples=max_examples, normalize=True)
        if binary:
            rows = [r for r in rows if r["label"] != "unknown"]
        if not rows:
            print("  No rows — check category name.")
            return
        n_entries = len(rows) // (2 if binary else 3)
        print(f"  {n_entries} entries → {len(rows)} rows")
        # Residual embedding — same approach as by-role mode.
        # Subtracts shared context so the answer's contribution is visible.
        cq_texts  = [r["cq"]                    for r in rows]
        ans_texts = [r["cq"] + " " + r["answer"] for r in rows]
        e_cq  = embedder.encode(cq_texts).float()
        e_ans = embedder.encode(ans_texts).float()
        embs  = e_ans - e_cq
        labels = [r["label"] for r in rows]
        label_colors  = LABEL_COLORS
        label_markers = LABEL_MARKERS
        plot_labels   = LABELS

    X = embs.float().numpy()
    label_arr = np.array(labels)

    # ---- Fit PCA ----
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # ---- BBQ scatter ----
    for lab in plot_labels:
        mask = label_arr == lab
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=label_colors[lab], marker=label_markers[lab],
            alpha=0.35, s=12, linewidths=0,
            label=lab.replace("_", " "),
            zorder=2,
        )

    # ---- Anchor arrows (Gender_identity only, or if anchor loads) ----
    if show_anchor:
        try:
            anchors = compute_anchors(embedder)
            scale = np.abs(coords).max() * 0.45

            # When the PCA is on residuals (E(cq+ans) - E(cq)), projecting
            # raw anchors is wrong — they live in full embedding space.
            # Project the DIFFERENCE (male - female) instead: it cancels the
            # shared context component, leaving only the gender direction,
            # which IS in the same space as the residuals.
            diff = (anchors.male_anchor - anchors.female_anchor).float().numpy()
            d2d  = pca.components_ @ diff
            d2d  = d2d / (np.linalg.norm(d2d) + 1e-9) * scale
            # Draw one double-headed arrow: female ← → male
            ax.annotate("", xy=d2d, xytext=-d2d,
                        arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=2.0),
                        zorder=5)
            ax.text( d2d[0]*1.2,  d2d[1]*1.2, "male",   color="#2ecc71", fontsize=8, ha="center", zorder=5)
            ax.text(-d2d[0]*1.2, -d2d[1]*1.2, "female", color="#e74c3c", fontsize=8, ha="center", zorder=5)
        except Exception as ex:
            print(f"  [warn] Could not load anchors: {ex}")

    # ---- LLM answer overlay ----
    if llm_csv and llm_csv.exists():
        import pandas as pd
        df_llm = pd.read_csv(llm_csv)
        if "llm_answer" in df_llm.columns:
            print(f"  Embedding {len(df_llm)} LLM answers from {llm_csv.name}...")
            llm_texts = [
                row.get("context", "") + " " + row.get("question", "") + " " + row["llm_answer"]
                for _, row in df_llm.iterrows()
            ]
            e_llm = embedder.encode(llm_texts).float().numpy()
            c_llm = pca.transform(e_llm)

            bbq_labels = df_llm.get("bbq_label", pd.Series(["unknown"] * len(df_llm)))
            llm_colors = {
                "biased":     "#c0392b",
                "anti_biased": "#2980b9",
                "unknown":    "#7f8c8d",
                "unmatched":  "#f39c12",
            }
            for lbl, col in llm_colors.items():
                mask = bbq_labels == lbl
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    c_llm[mask, 0], c_llm[mask, 1],
                    c=col, marker="D", s=40, linewidths=0.5,
                    edgecolors="white", alpha=0.85,
                    label=f"LLM {lbl}", zorder=4,
                )

    # Annotate the two gender clusters so the reader understands the structure.
    # Cluster centres are the mean PC1 coordinate for each answer text.
    if not by_role and show_anchor:
        from run_bbq_geometry import _CANONICAL
        # "The man" residuals land right of 0, "The woman" left of 0 (after residual PCA)
        right_mask = coords[:, 0] > coords[:, 0].mean()
        left_mask  = ~right_mask
        for mask, lbl, col in [(right_mask, '"The man" answers',   "#2980b9"),
                                (left_mask,  '"The woman" answers', "#c0392b")]:
            cx = coords[mask, 0].mean()
            cy = coords[mask, 1].mean()
            ax.text(cx, cy + np.abs(coords[:, 1]).max() * 0.18, lbl,
                    color=col, fontsize=8, ha="center", style="italic", zorder=6)

    title_tag = "biased vs anti-biased" if binary else "all labels"
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var.)")
    ax.set_title(f"{category} — PCA ({title_tag}, cq + normalised answer)", fontsize=9)
    ax.axhline(0, color="lightgray", linewidth=0.5, zorder=1)
    ax.axvline(0, color="lightgray", linewidth=0.5, zorder=1)
    ax.legend(frameon=False, fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{category}_embedding_space_pca{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(save_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {stem}.pdf / .png")

    # ---- PC1 projection histogram ----
    _plot_pc1_histogram(coords, label_arr, category, ev[0], save_dir, suffix)

    # ---- t-SNE scatter ----
    _plot_tsne(X, label_arr, category, title_tag, show_anchor,
               pca, save_dir, suffix)

    # ---- 1D bias-axis projection (the metric itself) ----
    if show_anchor:
        try:
            anchors = compute_anchors(embedder)
            _plot_bias_axis_1d(embs, labels, anchors, category, save_dir, suffix)
        except Exception as ex:
            print(f"  [warn] bias axis plot skipped: {ex}")


def _plot_pc1_histogram(
    coords: np.ndarray,
    label_arr: np.ndarray,
    category: str,
    ev0: float,
    save_dir: Path,
    suffix: str = "",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for lab in LABELS:
        mask = label_arr == lab
        if mask.sum() < 2:
            continue
        ax.hist(
            coords[mask, 0], bins=40, density=True, alpha=0.45,
            color=LABEL_COLORS[lab], label=lab.replace("_", " "),
            histtype="stepfilled",
        )
    ax.set_xlabel(f"PC1 ({ev0*100:.1f}% var.)")
    ax.set_ylabel("Density")
    ax.set_title(f"{category} — PC1 distribution by label", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    stem = f"{category}_pc1_histogram{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(save_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {stem}.pdf / .png")


def _plot_tsne(
    X: np.ndarray,
    label_arr: np.ndarray,
    category: str,
    title_tag: str,
    show_anchor: bool,
    pca,           # fitted PCA object — used to project anchors into same orientation
    save_dir: Path,
    suffix: str = "",
):
    """t-SNE scatter coloured by label, with optional anchor direction overlay."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    MAX = 800
    if len(X) > MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), MAX, replace=False)
        X_plot, lab_plot = X[idx], label_arr[idx]
    else:
        X_plot, lab_plot = X, label_arr

    perp = min(30, max(5, len(X_plot) // 10))
    print(f"  Running t-SNE (n={len(X_plot)}, perplexity={perp})...")
    coords = TSNE(n_components=2, perplexity=perp, random_state=42,
                  max_iter=1000).fit_transform(X_plot)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lab in LABELS:
        mask = lab_plot == lab
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=LABEL_COLORS[lab], marker=LABEL_MARKERS[lab],
            alpha=0.5, s=15, linewidths=0,
            label=lab.replace("_", " "), zorder=2,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"{category} — t-SNE ({title_tag}, cq + normalised answer)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout()

    stem = f"{category}_embedding_space_tsne{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(save_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BBQ embedding space PCA")
    p.add_argument("--category", default="Gender_identity")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--model-name", default="all-MiniLM-L6-v2")
    p.add_argument("--binary", action="store_true",
                   help="Only biased vs anti_biased — drop unknown rows")
    p.add_argument("--by-role", action="store_true",
                   help="Colour by gender role (male/female/unknown) instead of "
                        "biased/anti_biased. Produces visible PCA separation because "
                        "the embedder can distinguish 'The man' from 'The woman'.")
    p.add_argument("--no-anchor", action="store_true",
                   help="Skip anchor vector overlay (for non-gender categories)")
    p.add_argument("--llm-csv", type=Path, default=None,
                   help="Optional: LLM results CSV to overlay on the PCA plot")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading embedder: {args.model_name}")
    embedder = Embedder(args.model_name)

    run_pca_plot(
        category=args.category,
        embedder=embedder,
        max_examples=args.max_examples,
        show_anchor=not args.no_anchor,
        llm_csv=args.llm_csv,
        save_dir=args.out_dir,
        binary=args.binary,
        by_role=args.by_role,
    )


if __name__ == "__main__":
    main()
