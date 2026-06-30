"""
SAIR Embedding Geometry Experiment
===================================
Tests whether TRUE vs FALSE mathematical implications (from SAIR/Lean-verified data)
land in different regions of sentence embedding space, without the embedder understanding
the underlying math.

Uses sentence-transformers (Windows compatible).

Usage examples:
    # Full pipeline on normal split
    python sair_experiment.py --data normal --steps embed cluster classify plot

    # Just embed and cache (expensive), then analyse later
    python sair_experiment.py --data normal --steps embed
    python sair_experiment.py --data normal --steps cluster classify plot

    # Compare all text templates on hard split
    python sair_experiment.py --data hard --template all --steps embed cluster classify plot

    # Run on every split, natural language template, skip plots
    python sair_experiment.py --data all --template natural --steps embed cluster classify

    # Use a different sentence-transformers model
    python sair_experiment.py --data normal --model all-mpnet-base-v2 --steps embed cluster classify plot
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "SAIR-competition"
OUTPUT_DIR = PROJECT_DIR / "results" / "sair"
CACHE_DIR = PROJECT_DIR / "cache" / "sair"

SPLITS = ["normal", "hard", "hard1", "hard2"]
DEFAULT_MODEL = "all-MiniLM-L6-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPLATES = {
    "raw":       lambda e1, e2: f"{e1} implies {e2}",
    "natural":   lambda e1, e2: f"If {e1}, then {e2}.",
    "conjoined": lambda e1, e2: f"{e1} and {e2}",
    "eq1_only":  lambda e1, e2: e1,
    "eq2_only":  lambda e1, e2: e2,
    # "separate" is handled specially, embeds eq1 and eq2 independently
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(split: str, max_examples: int | None = None) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            rows.append(json.loads(line.strip()))
    return rows


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def format_texts(rows: list[dict], template: str) -> list[str]:
    """
    Returns one text string per row. For 'separate', returns two parallel
    lists (eq1_texts, eq2_texts) instead, handled by the caller.
    """
    if template == "separate":
        raise ValueError("Use format_texts_separate() for the 'separate' template.")
    fn = TEMPLATES[template]
    return [fn(r["equation1"], r["equation2"]) for r in rows]


def format_texts_separate(rows: list[dict]) -> tuple[list[str], list[str]]:
    return (
        [r["equation1"] for r in rows],
        [r["equation2"] for r in rows],
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedder(model_name: str = DEFAULT_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=DEVICE)
        print(f"Embedder ready: {model_name} on {DEVICE}")
        return model
    except ImportError as exc:
        print(f"ERROR: Could not import sentence-transformers: {exc}")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)


def embed_texts(model, texts: list[str], batch_size: int = 32) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def cache_path(split: str, template: str, model_name: str, suffix: str = "") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    name = f"{split}_{template}_{safe_model}{suffix}.pt"
    return CACHE_DIR / name


def run_embed(split: str, template: str, rows: list[dict],
              batch_size: int, force: bool,
              model_name: str = DEFAULT_MODEL) -> dict[str, torch.Tensor]:
    """
    Compute (or load from cache) embeddings for the given split+template.
    Returns a dict of tensors keyed by role: {'main', 'eq1', 'eq2'}.
    For 'separate', 'main' is absent; for others, 'eq1'/'eq2' are absent.
    """
    labels = torch.tensor([1 if r["answer"] else 0 for r in rows])

    if template == "separate":
        cp_eq1 = cache_path(split, template, model_name, "_eq1")
        cp_eq2 = cache_path(split, template, model_name, "_eq2")
        cp_lbl = cache_path(split, template, model_name, "_labels")

        if not force and cp_eq1.exists() and cp_eq2.exists():
            print(f"  Loading cached embeddings from {CACHE_DIR}")
            return {
                "eq1": torch.load(cp_eq1),
                "eq2": torch.load(cp_eq2),
                "labels": torch.load(cp_lbl),
            }

        embedder = get_embedder(model_name)
        eq1_texts, eq2_texts = format_texts_separate(rows)
        print(f"  Embedding {len(rows)} eq1 strings...")
        emb_eq1 = embed_texts(embedder, eq1_texts, batch_size)
        print(f"  Embedding {len(rows)} eq2 strings...")
        emb_eq2 = embed_texts(embedder, eq2_texts, batch_size)

        torch.save(emb_eq1, cp_eq1)
        torch.save(emb_eq2, cp_eq2)
        torch.save(labels, cp_lbl)
        print(f"  Cached to {CACHE_DIR}")
        return {"eq1": emb_eq1, "eq2": emb_eq2, "labels": labels}

    else:
        cp_main = cache_path(split, template, model_name, "_main")
        cp_lbl  = cache_path(split, template, model_name, "_labels")

        if not force and cp_main.exists():
            print(f"  Loading cached embeddings from {cp_main}")
            return {
                "main": torch.load(cp_main),
                "labels": torch.load(cp_lbl),
            }

        embedder = get_embedder(model_name)
        texts = format_texts(rows, template)
        print(f"  Embedding {len(rows)} texts (template={template!r})...")
        emb = embed_texts(embedder, texts, batch_size)

        torch.save(emb, cp_main)
        torch.save(labels, cp_lbl)
        print(f"  Cached to {CACHE_DIR}")
        return {"main": emb, "labels": labels}


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(emb_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert embedding dict to (X, y) numpy arrays.
    For 'separate', X is the element-wise cosine similarity vector
    (a scalar per pair, but we also expose [emb_eq1, emb_eq2, diff, prod] variants).
    """
    labels = emb_dict["labels"].numpy()

    if "main" in emb_dict:
        X = emb_dict["main"].cpu().float().numpy()
    else:
        # separate: feature = [eq1, eq2, eq1-eq2, eq1*eq2] concatenated
        e1 = emb_dict["eq1"].cpu().float()
        e2 = emb_dict["eq2"].cpu().float()
        diff = (e1 - e2).numpy()
        prod = (e1 * e2).numpy()
        X = np.concatenate([e1.numpy(), e2.numpy(), diff, prod], axis=1)

    return X, labels


# ---------------------------------------------------------------------------
# Step: cluster analysis
# ---------------------------------------------------------------------------

def run_cluster(emb_dict: dict, split: str, template: str) -> pd.DataFrame:
    """
    Measures:
    - Centroid cosine similarity (TRUE centroid vs FALSE centroid)
    - Mean pairwise intra-class cosine similarity
    - Mean pairwise inter-class cosine similarity
    - Cohen's d on the first PC
    """
    X, y = build_feature_matrix(emb_dict)
    X_t = torch.tensor(X)

    true_mask  = y == 1
    false_mask = y == 0
    X_true  = X_t[true_mask]
    X_false = X_t[false_mask]

    # Centroids
    c_true  = X_true.mean(dim=0, keepdim=True)
    c_false = X_false.mean(dim=0, keepdim=True)
    centroid_sim = F.cosine_similarity(c_true, c_false).item()

    # Mean intra-class similarity (sample up to 500 per class for speed)
    def mean_intra_sim(X_class, max_n=500):
        n = min(len(X_class), max_n)
        idx = torch.randperm(len(X_class))[:n]
        sub = X_class[idx]
        # Normalise
        sub = F.normalize(sub, dim=1)
        gram = sub @ sub.T
        # Upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        return gram[mask].mean().item()

    intra_true  = mean_intra_sim(X_true)
    intra_false = mean_intra_sim(X_false)

    # Mean inter-class similarity
    n = min(len(X_true), len(X_false), 500)
    idx_t = torch.randperm(len(X_true))[:n]
    idx_f = torch.randperm(len(X_false))[:n]
    nt = F.normalize(X_true[idx_t], dim=1)
    nf = F.normalize(X_false[idx_f], dim=1)
    inter_sim = (nt @ nf.T).mean().item()

    # Cohen's d on first PC projection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    proj = pca.fit_transform(X).flatten()
    proj_true  = proj[true_mask]
    proj_false = proj[false_mask]
    pooled_std = np.sqrt((proj_true.var() + proj_false.var()) / 2)
    cohens_d = (proj_true.mean() - proj_false.mean()) / (pooled_std + 1e-9)

    stats = {
        "split": split,
        "template": template,
        "n_true": int(true_mask.sum()),
        "n_false": int(false_mask.sum()),
        "centroid_cosine_sim": round(centroid_sim, 6),
        "intra_true_sim": round(intra_true, 6),
        "intra_false_sim": round(intra_false, 6),
        "inter_class_sim": round(inter_sim, 6),
        "cohens_d_pc1": round(cohens_d, 6),
    }

    print("\n--- Cluster Analysis ---")
    for k, v in stats.items():
        print(f"  {k:<30} {v}")

    return pd.DataFrame([stats])


# ---------------------------------------------------------------------------
# Step: linear classification
# ---------------------------------------------------------------------------

def run_classify(emb_dict: dict, split: str, template: str) -> pd.DataFrame:
    """
    Logistic regression (L2) with 5-fold stratified CV.
    Balanced accuracy accounts for class imbalance in hard splits.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X, y = build_feature_matrix(emb_dict)

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "balanced_accuracy", "roc_auc"],
        return_train_score=False,
    )

    result = {
        "split": split,
        "template": template,
        "acc_mean":  round(scores["test_accuracy"].mean(), 4),
        "acc_std":   round(scores["test_accuracy"].std(), 4),
        "bacc_mean": round(scores["test_balanced_accuracy"].mean(), 4),
        "bacc_std":  round(scores["test_balanced_accuracy"].std(), 4),
        "auc_mean":  round(scores["test_roc_auc"].mean(), 4),
        "auc_std":   round(scores["test_roc_auc"].std(), 4),
        "chance":    round(max(y.mean(), 1 - y.mean()), 4),
    }

    print("\n--- Linear Classification (5-fold CV) ---")
    for k, v in result.items():
        print(f"  {k:<30} {v}")

    return pd.DataFrame([result])


# ---------------------------------------------------------------------------
# Step: plot
# ---------------------------------------------------------------------------

def run_plot(emb_dict: dict, split: str, template: str, output_dir: Path):
    """
    Generates:
    1. PCA 2D scatter (TRUE vs FALSE)
    2. t-SNE 2D scatter
    3. Cosine similarity distributions (intra-TRUE, intra-FALSE, inter)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # headless
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X, y = build_feature_matrix(emb_dict)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {1: "#2196F3", 0: "#F44336"}
    labels_str = {1: "TRUE", 0: "FALSE"}

    # --- PCA ---
    print("  Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[cls], label=labels_str[cls],
                   alpha=0.4, s=15, linewidths=0)
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(f"PCA, {split} / {template}")
    ax.legend()
    plt.tight_layout()
    path = output_dir / f"{split}_{template}_pca.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved {path}")

    # --- t-SNE (subsample to 800 for speed) ---
    n_tsne = min(len(X), 800)
    idx = np.random.default_rng(42).choice(len(X), n_tsne, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    print("  Computing t-SNE (may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in [0, 1]:
        mask = y_sub == cls
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=colors[cls], label=labels_str[cls],
                   alpha=0.4, s=15, linewidths=0)
    ax.set_title(f"t-SNE, {split} / {template}")
    ax.legend()
    plt.tight_layout()
    path = output_dir / f"{split}_{template}_tsne.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved {path}")

    # --- Cosine similarity distribution ---
    X_t = torch.tensor(X).float()
    X_norm = F.normalize(X_t, dim=1)

    true_mask  = torch.tensor(y == 1)
    false_mask = torch.tensor(y == 0)

    def sample_sims(A, B, max_n=400):
        n = min(len(A), len(B), max_n)
        ia = torch.randperm(len(A))[:n]
        ib = torch.randperm(len(B))[:n]
        return (A[ia] * B[ib]).sum(dim=1).numpy()

    sims_intra_true  = sample_sims(X_norm[true_mask],  X_norm[true_mask])
    sims_intra_false = sample_sims(X_norm[false_mask], X_norm[false_mask])
    sims_inter       = sample_sims(X_norm[true_mask],  X_norm[false_mask])

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(sims_intra_true.min(), sims_intra_false.min(), sims_inter.min()),
        max(sims_intra_true.max(), sims_intra_false.max(), sims_inter.max()),
        50,
    )
    ax.hist(sims_intra_true,  bins=bins, alpha=0.5, color="#2196F3", label="Intra-TRUE")
    ax.hist(sims_intra_false, bins=bins, alpha=0.5, color="#F44336", label="Intra-FALSE")
    ax.hist(sims_inter,       bins=bins, alpha=0.5, color="#9C27B0", label="Inter (T vs F)")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Pairwise Similarity Distributions, {split} / {template}")
    ax.legend()
    plt.tight_layout()
    path = output_dir / f"{split}_{template}_sim_dist.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAIR embedding geometry experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=["normal"],
        choices=SPLITS + ["all"],
        help="Which SAIR split(s) to use. 'all' expands to all four splits.",
    )
    parser.add_argument(
        "--template",
        nargs="+",
        default=["natural"],
        choices=list(TEMPLATES.keys()) + ["separate", "all"],
        help=(
            "Text formatting template(s). "
            "'raw': 'eq1 implies eq2', "
            "'natural': 'If eq1, then eq2.', "
            "'conjoined': 'eq1 and eq2', "
            "'eq1_only' / 'eq2_only': control baselines, "
            "'separate': embed eq1 and eq2 independently then concatenate features. "
            "'all' runs every template."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["embed", "cluster", "classify", "plot"],
        choices=["embed", "cluster", "classify", "plot"],
        help="Which pipeline steps to run.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap number of examples per split (default: use all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Sentence-transformers model name "
            f"(default: {DEFAULT_MODEL}). "
            "Any model from https://www.sbert.net/docs/pretrained_models.html works."
        ),
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Re-compute embeddings even if a cache file exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Where to save results (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Expand 'all'
    splits = SPLITS if "all" in args.data else args.data
    all_template_keys = list(TEMPLATES.keys()) + ["separate"]
    templates = all_template_keys if "all" in args.template else args.template

    steps = set(args.steps)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dependency: cluster/classify/plot require embeddings
    needs_embeddings = steps & {"cluster", "classify", "plot"}
    if needs_embeddings and "embed" not in steps:
        # Will try to load from cache; embed step will be run implicitly
        pass

    all_cluster_rows  = []
    all_classify_rows = []

    for split in splits:
        print(f"\n{'='*60}")
        print(f"SPLIT: {split}")
        print(f"{'='*60}")

        rows = load_split(split, max_examples=args.max_examples)
        print(f"Loaded {len(rows)} rows  (TRUE={sum(r['answer'] for r in rows)}, FALSE={sum(not r['answer'] for r in rows)})")

        for template in templates:
            print(f"\n  -- template: {template} --")

            # Always run embed (or load from cache) if any downstream step needs it
            if "embed" in steps or needs_embeddings:
                emb_dict = run_embed(
                    split, template, rows,
                    batch_size=args.batch_size,
                    force=args.force_embed,
                    model_name=args.model,
                )
            else:
                continue  # Nothing to do

            if "cluster" in steps:
                df_c = run_cluster(emb_dict, split, template)
                all_cluster_rows.append(df_c)

            if "classify" in steps:
                df_cl = run_classify(emb_dict, split, template)
                all_classify_rows.append(df_cl)

            if "plot" in steps:
                print("  Generating plots...")
                run_plot(emb_dict, split, template, output_dir)

    # Save aggregate results
    if all_cluster_rows:
        df_cluster = pd.concat(all_cluster_rows, ignore_index=True)
        path = output_dir / "cluster_results.csv"
        df_cluster.to_csv(path, index=False)
        print(f"\nCluster results saved to {path}")
        print(df_cluster.to_string(index=False))

    if all_classify_rows:
        df_classify = pd.concat(all_classify_rows, ignore_index=True)
        path = output_dir / "classify_results.csv"
        df_classify.to_csv(path, index=False)
        print(f"\nClassification results saved to {path}")
        print(df_classify.to_string(index=False))


if __name__ == "__main__":
    main()
