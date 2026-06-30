"""
BBQ embedding geometry analysis, SAIR-style.

For each ambiguous BBQ entry, creates three rows:
    (context + question + stereo_ans,      "biased")
    (context + question + anti_stereo_ans, "anti_biased")
    (context + question + unknown_ans,     "unknown")

Then tests whether label classes separate in embedding space:
    1. Clustering, pairwise centroid cosine sim + Cohen's d on PC1
    2. Classification, logistic regression, 5-fold stratified CV
    3. Visualisation, PCA scatter, t-SNE scatter, cosine similarity distributions

Templates tested (analogous to SAIR's raw / natural / eq1_only):
    answer_only, just the answer text
    cq_answer, context + question + answer  (what the current anchors use)
    question_answer, question + answer, no context
    context_only, control: same text for all 3 labels from the same entry
    separate, [E(cq), E(answer), E(cq)−E(answer), E(cq)⊙E(answer)] concatenated

Usage:
    python run_bbq_geometry.py
    python run_bbq_geometry.py --category Nationality --max-examples 300
    python run_bbq_geometry.py --templates answer_only cq_answer --no-plots
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from bias_scorer.embedder import Embedder

# Gender answer_info labels that need aliasing (other categories use direct label match)
_GENDER_ALIASES: dict[str, set[str]] = {
    "f":                 {"f", "female", "woman", "girl", "trans_f"},
    "m":                 {"m", "male", "man", "boy", "trans_m"},
    "transgender women": {"trans_f"},
}

# Canonical answer texts used when --normalize is set.
# Maps answer_info label → a consistent natural-language phrase.
# Unknown-type labels all map to the same "can't determine" string.
_CANONICAL: dict[str, str] = {
    # --- gender ---
    "f": "The woman", "female": "The woman", "woman": "The woman",
    "girl": "The woman", "trans_f": "The woman", "nontrans_f": "The woman",
    "m": "The man", "male": "The man", "man": "The man",
    "boy": "The man", "trans_m": "The man", "nontrans_m": "The man",
    "trans": "The transgender person",
    "nontrans": "The cisgender person",
    # --- age ---
    "old": "The elderly person", "nonold": "The younger person",
    # --- disability ---
    "disabled": "The disabled person", "nondisabled": "The non-disabled person",
    # --- SES ---
    "highses": "The wealthy person", "lowses": "The low-income person",
    # --- sexual orientation ---
    "bisexual": "The bisexual person", "gay": "The gay man",
    "lesbian": "The lesbian woman", "pansexual": "The pansexual person",
    "straight": "The straight person",
    # --- religion ---
    "atheist": "The atheist", "buddhist": "The Buddhist person",
    "catholic": "The Catholic person", "christian": "The Christian person",
    "hindu": "The Hindu person", "jewish": "The Jewish person",
    "mormon": "The Mormon person", "muslim": "The Muslim person",
    "protestant": "The Protestant person", "sikh": "The Sikh person",
    # --- nationality (BBQ uses world regions) ---
    "africa": "The African person", "arabstates": "The Arab person",
    "asiapacific": "The Asian person", "europe": "The European person",
    "latinsouthamerica": "The Latin American person",
    "middleeast": "The Middle Eastern person",
    "northamerica": "The North American person",
    # --- physical appearance ---
    "obese": "The overweight person", "nonobese": "The non-overweight person",
    "short": "The short person", "tall": "The tall person",
    "pregnant": "The pregnant person", "notpregnant": "The non-pregnant person",
    "visibledifference": "The person with a visible difference",
    "novisibledifference": "The person without a visible difference",
    "posdress": "The well-dressed person", "negdress": "The poorly-dressed person",
    # --- race/ethnicity ---
    "african": "The African person",
    "african american": "The African American person",
    "arab": "The Arab person", "asian": "The Asian person",
    "black": "The Black person", "caucasian": "The White person",
    "european": "The European person", "hispanic": "The Hispanic person",
    "latino": "The Latino person", "middle eastern": "The Middle Eastern person",
    "native american": "The Native American person",
    "roma": "The Roma person", "south american": "The South American person",
    "white": "The White person",
    # race with gender prefix (Race_ethnicity + Race_x_gender)
    "f-arab": "The Arab woman", "f-asian": "The Asian woman",
    "f-black": "The Black woman", "f-latino": "The Latina woman",
    "f-white": "The White woman", "f-african": "The African woman",
    "f-african american": "The African American woman",
    "f-caucasian": "The White woman", "f-european": "The European woman",
    "f-hispanic": "The Hispanic woman", "f-jewish": "The Jewish woman",
    "f-middle eastern": "The Middle Eastern woman",
    "f-native american": "The Native American woman",
    "f-roma": "The Roma woman", "f-south american": "The South American woman",
    "m-arab": "The Arab man", "m-asian": "The Asian man",
    "m-black": "The Black man", "m-latino": "The Latino man",
    "m-white": "The White man", "m-african": "The African man",
    "m-african american": "The African American man",
    "m-caucasian": "The White man", "m-european": "The European man",
    "m-hispanic": "The Hispanic man", "m-jewish": "The Jewish man",
    "m-middle eastern": "The Middle Eastern man",
    "m-native american": "The Native American man",
    "m-roma": "The Roma man", "m-south american": "The South American man",
    # --- universal ---
    "unknown": "Cannot be determined",
}

# Race_x_SES uses compound labels like "highses-african american".
# Parse them on the fly rather than enumerate all 50 combinations.
def _canonicalize(label: str, raw_text: str) -> str:
    """Return canonical phrase for an answer_info label, or raw_text if unknown."""
    l = label.lower().strip()
    if not l:
        return raw_text   # empty label in BBQ data → keep raw
    if l in _CANONICAL:
        return _CANONICAL[l]
    # compound SES label: "highses-<group>" / "lowses-<group>"
    for prefix, ses_phrase in (("highses-", "The wealthy "), ("lowses-", "The low-income ")):
        if l.startswith(prefix):
            group = l[len(prefix):]
            group_phrase = _CANONICAL.get(group, group)
            # strip "The " prefix from group phrase for composition
            group_word = group_phrase.replace("The ", "").replace("the ", "")
            return f"The {ses_phrase.strip().split()[-1]} {group_word}"
    return raw_text  # unmapped label keeps raw text


def _matches_sg(answer_label: str, sg: str) -> bool:
    """Return True if an answer_info label corresponds to the stereotyped group sg."""
    al = answer_label.lower()
    sg_l = sg.lower()
    if al == sg_l:
        return True
    return al in _GENDER_ALIASES.get(sg_l, set())


def _load_row_labels(entry: dict, normalize: bool = False) -> tuple[str, str, str] | None:
    """
    Return (stereo_text, anti_stereo_text, unknown_text) for one BBQ entry,
    accounting for question_polarity.  Works for all BBQ categories.
    When normalize=True, replaces raw answer text with canonical role phrases
    so that all male-role answers embed as "The man", female as "The woman", etc.
    Returns None if roles can't be fully determined.
    """
    info = entry["answer_info"]
    sg_list = entry.get("additional_metadata", {}).get("stereotyped_groups", [])
    sg = sg_list[0] if sg_list else None
    if not sg:
        return None

    unknown_text = sg_text = other_text = None
    for key in ("ans0", "ans1", "ans2"):
        label = info[key][1].lower()
        text  = _canonicalize(label, entry[key]) if normalize else entry[key]
        if label == "unknown":
            unknown_text = text
        elif _matches_sg(label, sg):
            sg_text = text
        else:
            other_text = text

    if None in (unknown_text, sg_text, other_text):
        return None

    # neg: picking stereotyped group = biased
    # nonneg: picking the OTHER group = biased (implies sg lacks the positive trait)
    if entry.get("question_polarity", "neg") == "neg":
        return sg_text, other_text, unknown_text
    return other_text, sg_text, unknown_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BBQ_DIR     = PROJECT_ROOT / "BBQ_Data"
CACHE_DIR   = PROJECT_ROOT / "results" / "bbq_geometry" / "cache"
OUT_DIR     = PROJECT_ROOT / "results" / "bbq_geometry"
MODEL_NAME  = "all-MiniLM-L6-v2"

TEMPLATES = ["answer_only", "cq_answer", "question_answer", "context_only", "separate"]
LABELS    = ["biased", "anti_biased", "unknown"]

LABEL_COLORS = {"biased": "#c0392b", "anti_biased": "#2980b9", "unknown": "#7f8c8d"}
LABEL_MARKERS = {"biased": "o", "anti_biased": "s", "unknown": "^"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bbq_rows(
    category: str,
    max_examples: int | None = None,
    normalize: bool = False,
) -> list[dict]:
    """Return one row per (entry, answer_role) triple, labelled biased/anti_biased/unknown.
    Works for all BBQ categories.
    normalize=True replaces raw answer text with canonical role phrases (e.g. 'The man',
    'The woman') so embeddings reflect the role rather than the specific name or phrase."""
    path = BBQ_DIR / f"{category}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line.strip())
            if e.get("context_condition") != "ambig":
                continue
            result = _load_row_labels(e, normalize=normalize)
            if result is None:
                continue
            stereo_ans, anti_stereo_ans, unknown_ans = result
            cq = e["context"] + " " + e["question"]
            base = {"entry_id": e["example_id"], "context": e["context"],
                    "question": e["question"], "cq": cq}
            rows.append({**base, "answer": stereo_ans,     "label": "biased"})
            rows.append({**base, "answer": anti_stereo_ans,"label": "anti_biased"})
            rows.append({**base, "answer": unknown_ans,    "label": "unknown"})
            if max_examples and len(rows) // 3 >= max_examples:
                break
    return rows


# ---------------------------------------------------------------------------
# Embedding with caching
# ---------------------------------------------------------------------------

def _cache_path(category: str, template: str, normalize: bool = False) -> Path:
    suffix = "_norm" if normalize else ""
    return CACHE_DIR / f"{category}_{template}{suffix}.pt"


def embed_rows(
    rows: list[dict],
    template: str,
    embedder: Embedder,
    category: str,
    force: bool = False,
    normalize: bool = False,
) -> tuple[torch.Tensor, list[str]]:
    """Return (embeddings [N, dim], labels [N])."""
    cache = _cache_path(category, template, normalize)
    labels = [r["label"] for r in rows]

    if not force and cache.exists():
        saved = torch.load(cache, weights_only=False)
        if saved["template"] == template and len(saved["labels"]) == len(rows):
            print(f"  [cache] {template}, loaded from {cache.name}")
            return saved["embeddings"], saved["labels"]

    if template == "answer_only":
        texts = [r["answer"] for r in rows]
        embs = embedder.encode(texts)

    elif template == "cq_answer":
        texts = [r["cq"] + " " + r["answer"] for r in rows]
        embs = embedder.encode(texts)

    elif template == "question_answer":
        texts = [r["question"] + " " + r["answer"] for r in rows]
        embs = embedder.encode(texts)

    elif template == "context_only":
        # Control: one unique text per entry, same for all 3 labels
        texts = [r["context"] for r in rows]
        embs = embedder.encode(texts)

    elif template == "separate":
        cq_texts  = [r["cq"]     for r in rows]
        ans_texts = [r["answer"] for r in rows]
        e_cq  = Fn.normalize(embedder.encode(cq_texts).float(),  dim=1)
        e_ans = Fn.normalize(embedder.encode(ans_texts).float(), dim=1)
        embs = torch.cat([e_cq, e_ans, e_cq - e_ans, e_cq * e_ans], dim=1)

    else:
        raise ValueError(f"Unknown template: {template}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"template": template, "embeddings": embs, "labels": labels}, cache)
    return embs, labels


# ---------------------------------------------------------------------------
# Clustering analysis
# ---------------------------------------------------------------------------

def compute_cluster_stats(embs: torch.Tensor, labels: list[str]) -> dict:
    embs_n = Fn.normalize(embs.float(), dim=1).numpy()
    label_arr = np.array(labels)
    stats = {}

    # Pairwise centroid cosine similarity
    centroids = {}
    for lab in LABELS:
        mask = label_arr == lab
        if mask.sum() == 0:
            continue
        c = embs_n[mask].mean(axis=0)
        centroids[lab] = c / (np.linalg.norm(c) + 1e-9)

    pairs = [("biased", "anti_biased"), ("biased", "unknown"), ("anti_biased", "unknown")]
    for a, b in pairs:
        if a in centroids and b in centroids:
            sim = float(centroids[a] @ centroids[b])
            stats[f"centroid_sim_{a[:3]}_{b[:3]}"] = sim

    # Mean intra-class cosine similarity (sampled)
    for lab in LABELS:
        mask = label_arr == lab
        pts = embs_n[mask]
        if len(pts) < 2:
            continue
        pts = pts[:500]
        sims = pts @ pts.T
        np.fill_diagonal(sims, np.nan)
        stats[f"intra_{lab[:3]}"] = float(np.nanmean(sims))

    # Cohen's d on PC1 for biased vs anti_biased
    from sklearn.decomposition import PCA
    mask_b  = label_arr == "biased"
    mask_ab = label_arr == "anti_biased"
    if mask_b.sum() > 1 and mask_ab.sum() > 1:
        pca = PCA(n_components=1)
        combined = np.vstack([embs_n[mask_b], embs_n[mask_ab]])
        pc1 = pca.fit_transform(combined).ravel()
        n_b = mask_b.sum()
        g1, g2 = pc1[:n_b], pc1[n_b:]
        pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
        stats["cohens_d_pc1"] = float((g1.mean() - g2.mean()) / (pooled_std + 1e-9))

    return stats


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(embs: torch.Tensor, labels: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import LabelEncoder

    X = embs.float().numpy()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(np.unique(y))

    n_splits = min(5, min(np.bincount(y)))
    if n_splits < 2:
        return {"accuracy": None, "balanced_accuracy": None, "roc_auc_ovr": None}

    clf = LogisticRegression(max_iter=1000, C=1.0)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "balanced_accuracy"]
    if n_classes == 2:
        scoring.append("roc_auc")
    else:
        scoring.append("roc_auc_ovr_weighted")

    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    result = {
        "accuracy":          float(scores["test_accuracy"].mean()),
        "balanced_accuracy": float(scores["test_balanced_accuracy"].mean()),
    }
    auc_key = "test_roc_auc" if n_classes == 2 else "test_roc_auc_ovr_weighted"
    result["roc_auc"] = float(scores[auc_key].mean())
    result["n_splits"] = n_splits
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_pca(embs: torch.Tensor, labels: list[str], title: str, save_path: Path):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })

    X = embs.float().numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    label_arr = np.array(labels)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for lab in LABELS:
        mask = label_arr == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=LABEL_COLORS[lab], marker=LABEL_MARKERS[lab],
            alpha=0.55, s=18, linewidths=0, label=lab.replace("_", " "),
        )
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8, markerscale=1.2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(embs: torch.Tensor, labels: list[str], title: str, save_path: Path):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })

    X = embs.float().numpy()
    # Subsample for speed
    MAX = 600
    if len(X) > MAX:
        idx = np.random.default_rng(42).choice(len(X), MAX, replace=False)
        X_plot = X[idx]
        lab_plot = np.array(labels)[idx]
    else:
        X_plot = X
        lab_plot = np.array(labels)

    perp = min(30, max(5, len(X_plot) // 10))
    coords = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000).fit_transform(X_plot)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for lab in LABELS:
        mask = lab_plot == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=LABEL_COLORS[lab], marker=LABEL_MARKERS[lab],
            alpha=0.55, s=18, linewidths=0, label=lab.replace("_", " "),
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8, markerscale=1.2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_sim_distributions(embs: torch.Tensor, labels: list[str], title: str, save_path: Path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })

    embs_n = Fn.normalize(embs.float(), dim=1).numpy()
    label_arr = np.array(labels)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for a, b, color, ls in [
        ("biased",     "biased",     "#c0392b", "-"),
        ("anti_biased","anti_biased","#2980b9", "-"),
        ("biased",     "anti_biased","#8e44ad", "--"),
    ]:
        pts_a = embs_n[label_arr == a][:300]
        pts_b = embs_n[label_arr == b][:300]
        if len(pts_a) < 2 or len(pts_b) < 2:
            continue
        if a == b:
            sims = pts_a @ pts_a.T
            np.fill_diagonal(sims, np.nan)
        else:
            sims = pts_a @ pts_b.T
        vals = sims.ravel()
        vals = vals[~np.isnan(vals)]
        label_str = f"intra-{a.replace('_biased','').replace('anti_','anti-')}" if a == b else "biased vs anti"
        ax.hist(vals, bins=40, density=True, alpha=0.45, color=color,
                linestyle=ls, label=label_str, histtype="stepfilled")

    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(results_df: pd.DataFrame, category: str):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })

    pivot_acc = results_df.pivot(index="template", columns="category", values="accuracy")
    pivot_auc = results_df.pivot(index="template", columns="category", values="roc_auc")
    pivot_cd  = results_df.pivot(index="template", columns="category", values="cohens_d_pc1")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, data, title, fmt in zip(
        axes,
        [pivot_acc, pivot_auc, pivot_cd.abs()],
        ["Accuracy", "AUC (OvR)", "|Cohen's d| PC1"],
        [".2f", ".2f", ".2f"],
    ):
        im = ax.imshow(data.values.astype(float), aspect="auto",
                       cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center",
                            fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"BBQ {category}, embedding geometry", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / f"{category}_summary_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  summary heatmap → {out.name}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_category(category: str, templates: list[str], max_examples: int | None,
                 no_plots: bool, embedder: Embedder, binary: bool = False,
                 normalize: bool = False):
    flags = []
    if binary:    flags.append("binary")
    if normalize: flags.append("normalized")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    print(f"\n{'='*60}")
    print(f"  Category: {category}{flag_str}")
    print(f"{'='*60}")

    rows = load_bbq_rows(category, max_examples, normalize=normalize)
    if binary:
        rows = [r for r in rows if r["label"] != "unknown"]
    n_entries = len(rows) // (2 if binary else 3)
    print(f"  {n_entries} entries → {len(rows)} rows")

    if len(rows) == 0:
        print("  [error] No rows loaded, check BBQ file and category name.")
        return pd.DataFrame()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for tmpl in templates:
        print(f"\n  Template: {tmpl}")
        embs, labels = embed_rows(rows, tmpl, embedder, category, normalize=normalize)
        print(f"    embeddings: {tuple(embs.shape)}")

        cluster = compute_cluster_stats(embs, labels)
        clf_res  = classify(embs, labels)

        row = {"category": category, "template": tmpl, **cluster, **clf_res}
        all_results.append(row)

        # Print quick summary
        print(f"    accuracy={clf_res.get('accuracy', '?'):.3f}  "
              f"balanced={clf_res.get('balanced_accuracy', '?'):.3f}  "
              f"AUC={clf_res.get('roc_auc', '?'):.3f}  "
              f"Cohen's d={cluster.get('cohens_d_pc1', float('nan')):.3f}  "
              f"centroid(b,ab)={cluster.get('centroid_sim_bia_ant', float('nan')):.4f}")

        if not no_plots:
            suffix = ("_binary" if binary else "") + ("_norm" if normalize else "")
            stem = f"{category}_{tmpl}{suffix}"
            plot_pca(embs, labels,
                     f"{category} / {tmpl}, PCA",
                     OUT_DIR / f"{stem}_pca.png")
            plot_tsne(embs, labels,
                      f"{category} / {tmpl}, t-SNE",
                      OUT_DIR / f"{stem}_tsne.png")
            plot_sim_distributions(embs, labels,
                                   f"{category} / {tmpl}, cosine sim",
                                   OUT_DIR / f"{stem}_sim.png")

    df = pd.DataFrame(all_results)
    csv_path = OUT_DIR / f"{category}_geometry.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path.name}")

    if not no_plots and len(all_results) > 1:
        try:
            plot_summary_heatmap(df, category)
        except Exception as ex:
            print(f"  [warn] summary heatmap failed: {ex}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BBQ embedding geometry analysis")
    p.add_argument("--category", default="Gender_identity",
                   help="BBQ category filename without .jsonl")
    p.add_argument("--templates", nargs="+", default=TEMPLATES,
                   choices=TEMPLATES, help="Which templates to test")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Limit number of BBQ entries (triples = 3× this)")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--binary", action="store_true",
                   help="Only biased vs anti_biased (drop unknown rows)")
    p.add_argument("--normalize", action="store_true",
                   help="Replace raw answer text with canonical role phrase "
                        "(e.g. 'The man', 'The woman') so all same-role answers embed identically")
    p.add_argument("--force", action="store_true",
                   help="Recompute embeddings even if cached")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading embedder: {args.model_name}")
    embedder = Embedder(args.model_name)

    df = run_category(
        category=args.category,
        templates=args.templates,
        max_examples=args.max_examples,
        no_plots=args.no_plots,
        embedder=embedder,
        binary=args.binary,
        normalize=args.normalize,
    )

    print("\n\n--- Results table ---")
    cols = ["template", "accuracy", "balanced_accuracy", "roc_auc",
            "cohens_d_pc1", "centroid_sim_bia_ant"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False, float_format="{:.4f}".format))
    print("\nDone.")


if __name__ == "__main__":
    main()
