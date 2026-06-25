"""
SAIR Rigorous Experimental Controls
====================================
Implements five critical checks for experimental validity:

A. Grouped CV           -- hold out all examples sharing equation1 or equation2
B. Cross-template transfer -- train on one template, test on another
C. Same-instance delta  -- how does embedding geometry change across templates?
D. Null baselines       -- surface features that can embarrass the embedder
E. Encoder robustness   -- replicate across three encoders

Usage:
    python sair_rigorous.py --checks all --split normal
    python sair_rigorous.py --checks grouped transfer delta --split normal
    python sair_rigorous.py --checks baselines --split normal
    python sair_rigorous.py --checks encoders --split normal
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data" / "SAIR-competition"
CACHE_DIR   = PROJECT_DIR / "cache" / "sair"
OUTPUT_DIR  = PROJECT_DIR / "results" / "sair"

TEMPLATES = {
    "raw":       lambda e1, e2: f"{e1} implies {e2}",
    "natural":   lambda e1, e2: f"If {e1}, then {e2}.",
    "conjoined": lambda e1, e2: f"{e1} and {e2}",
    "eq1_only":  lambda e1, e2: e1,
    "eq2_only":  lambda e1, e2: e2,
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(split: str):
    rows = []
    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows

def load_embeddings(split: str, template: str, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    safe = model_name.replace("/", "_").replace("\\", "_")
    main_pt = CACHE_DIR / f"{split}_{template}_{safe}_main.pt"
    lbl_pt  = CACHE_DIR / f"{split}_{template}_{safe}_labels.pt"
    if not main_pt.exists():
        raise FileNotFoundError(
            f"No cache for {split}/{template}/{model_name}. "
            f"Run: python sair_experiment.py --data {split} --template {template} --model {model_name} --steps embed"
        )
    X = torch.load(main_pt).cpu().float().numpy()
    y = torch.load(lbl_pt).cpu().numpy()
    return X, y

def pipe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    )

# ---------------------------------------------------------------------------
# A. Grouped CV
# ---------------------------------------------------------------------------

def run_grouped_cv(split: str, model_name: str, group_by: str = "eq1"):
    """
    GroupKFold where each fold holds out all rows sharing the same equation.

    group_by:
        'eq1'    -- group by equation1
        'eq2'    -- group by equation2
        'either' -- group by (eq1, eq2) unordered union
    """
    print(f"\n{'='*70}")
    print(f"A. GROUPED CV | split={split} | group_by={group_by} | model={model_name}")
    print(f"{'='*70}")

    rows = load_split(split)
    results = []

    for template in TEMPLATES:
        try:
            X, y = load_embeddings(split, template, model_name)
        except FileNotFoundError as e:
            print(f"  Skipping {template}: {e}")
            continue

        # Build groups
        if group_by == "eq1":
            group_keys = [r["equation1"] for r in rows]
        elif group_by == "eq2":
            group_keys = [r["equation2"] for r in rows]
        elif group_by == "either":
            group_keys = [tuple(sorted([r["equation1"], r["equation2"]])) for r in rows]
        else:
            raise ValueError(f"Unknown group_by: {group_by}")

        # Map keys to integer group IDs
        key_to_id = {k: i for i, k in enumerate(dict.fromkeys(group_keys))}
        groups = np.array([key_to_id[k] for k in group_keys])

        n_groups = len(set(groups))
        n_splits = min(5, n_groups)

        if n_splits < 2:
            print(f"  {template}: only {n_groups} groups, cannot CV — skipping")
            continue

        gkf = GroupKFold(n_splits=n_splits)
        fold_scores = []
        fold_bacc   = []
        fold_auc    = []

        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            p = pipe()
            p.fit(X_tr, y_tr)

            from sklearn.metrics import balanced_accuracy_score, roc_auc_score
            y_pred = p.predict(X_te)
            y_prob = p.predict_proba(X_te)[:, 1]

            fold_scores.append((y_pred == y_te).mean())
            fold_bacc.append(balanced_accuracy_score(y_te, y_pred))
            try:
                fold_auc.append(roc_auc_score(y_te, y_prob))
            except Exception:
                fold_auc.append(float("nan"))

        res = {
            "split":    split,
            "template": template,
            "group_by": group_by,
            "n_groups": n_groups,
            "acc":      round(np.nanmean(fold_scores), 4),
            "bacc":     round(np.nanmean(fold_bacc), 4),
            "auc":      round(np.nanmean(fold_auc), 4),
        }
        print(f"  {template:12} groups={n_groups:4}  acc={res['acc']:.4f}  bacc={res['bacc']:.4f}  auc={res['auc']:.4f}")
        results.append(res)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# B. Cross-template transfer
# ---------------------------------------------------------------------------

def run_cross_template_transfer(split: str, model_name: str):
    """
    Train on one template's embeddings, test on another.
    """
    print(f"\n{'='*70}")
    print(f"B. CROSS-TEMPLATE TRANSFER | split={split} | model={model_name}")
    print(f"{'='*70}")

    available = {}
    for t in TEMPLATES:
        try:
            X, y = load_embeddings(split, t, model_name)
            available[t] = (X, y)
        except FileNotFoundError:
            pass

    print(f"\nAvailable templates: {list(available.keys())}")
    print(f"\nTransfer matrix (train -> test accuracy):\n")

    templates = list(available.keys())
    col_header = "train / test"
    header = f"{col_header:15}" + "".join(f"{t:12}" for t in templates)
    print(header)
    print("-" * len(header))

    results = []
    matrix  = {}

    for train_t in templates:
        X_train, y_train = available[train_t]
        row_str = f"{train_t:15}"

        for test_t in templates:
            X_test, y_test = available[test_t]

            p = pipe()
            p.fit(X_train, y_train)

            from sklearn.metrics import balanced_accuracy_score
            y_pred = p.predict(X_test)
            acc = (y_pred == y_test).mean()
            bacc = balanced_accuracy_score(y_test, y_pred)

            row_str += f"{acc:.3f} ({bacc:.3f})"[:12].ljust(12)

            results.append({
                "train_template": train_t,
                "test_template":  test_t,
                "same_template":  train_t == test_t,
                "acc":            round(float(acc), 4),
                "bacc":           round(float(bacc), 4),
            })
            matrix[(train_t, test_t)] = acc

        print(row_str)

    df = pd.DataFrame(results)

    # Summarize transfer vs. within-template
    within = df[df["same_template"]]["acc"].mean()
    cross  = df[~df["same_template"]]["acc"].mean()
    print(f"\nWithin-template (same):  {within:.4f}")
    print(f"Cross-template (different): {cross:.4f}")
    print(f"Transfer gap: {within - cross:.4f}")

    if within - cross < 0.05:
        print("[GOOD] Strong transfer — signal is mostly template-invariant")
    else:
        print("[NOTE] Weak transfer — prompt wrapper affects usable geometry")

    return df


# ---------------------------------------------------------------------------
# C. Same-instance delta analysis
# ---------------------------------------------------------------------------

def run_delta_analysis(split: str, model_name: str, template_a: str = "natural", template_b: str = "raw"):
    """
    For each instance i, compute delta_i = embed(template_a) - embed(template_b).
    Ask:
    - Are deltas nearly constant across instances? (wrapper effect)
    - Do deltas differ by label?   (label-geometry interaction)
    - What fraction of variance in embeddings is in the delta?
    """
    print(f"\n{'='*70}")
    print(f"C. DELTA ANALYSIS | split={split} | {template_a} - {template_b}")
    print(f"{'='*70}")

    try:
        X_a, y = load_embeddings(split, template_a, model_name)
        X_b, _ = load_embeddings(split, template_b, model_name)
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        return None

    delta = X_a - X_b  # (N, D)

    # 1. Is the delta nearly constant? (measure std across instances)
    delta_std_per_dim = delta.std(axis=0)       # std per embedding dimension
    mean_delta_std    = delta_std_per_dim.mean()
    max_delta_std     = delta_std_per_dim.max()

    # 2. Do deltas differ by label?
    delta_true  = delta[y == 1]
    delta_false = delta[y == 0]
    mean_delta_true  = delta_true.mean(axis=0)
    mean_delta_false = delta_false.mean(axis=0)

    # Cohen's d on delta magnitude
    delta_mag_true  = np.linalg.norm(delta_true,  axis=1)
    delta_mag_false = np.linalg.norm(delta_false, axis=1)
    pooled_std = np.sqrt((delta_mag_true.std()**2 + delta_mag_false.std()**2) / 2)
    cohens_d   = (delta_mag_true.mean() - delta_mag_false.mean()) / (pooled_std + 1e-9)

    # 3. Variance explained by delta vs. original
    var_original = X_a.var(axis=0).sum()
    var_delta    = delta.var(axis=0).sum()

    print(f"\nDelta statistics ({template_a} - {template_b}):")
    print(f"  Mean delta std per dimension: {mean_delta_std:.6f}")
    print(f"  Max delta std per dimension:  {max_delta_std:.6f}")
    print(f"  Delta variance / original variance: {var_delta / var_original:.4f}")

    print(f"\nLabel interaction:")
    print(f"  Mean delta magnitude (TRUE):  {delta_mag_true.mean():.4f} +/- {delta_mag_true.std():.4f}")
    print(f"  Mean delta magnitude (FALSE): {delta_mag_false.mean():.4f} +/- {delta_mag_false.std():.4f}")
    print(f"  Cohen's d on delta magnitude: {cohens_d:.4f}")

    # 4. Can a classifier distinguish labels from deltas alone?
    scores = cross_val_score(
        pipe(), delta, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy"
    )
    print(f"\n  Classify from delta alone: {scores.mean():.4f} +/- {scores.std():.4f}")

    if scores.mean() > 0.60:
        print(f"  [NOTE] Delta contains label signal — prompt wrapper interacts with label")
    else:
        print(f"  [GOOD] Delta carries no label signal — prompt is additive constant")

    if mean_delta_std < 0.01:
        print(f"\n  [GOOD] Deltas are nearly constant — prompt adds a consistent wrapper")
    else:
        print(f"\n  [NOTE] Deltas vary across instances — prompt interacts with content")

    return {
        "template_a": template_a,
        "template_b": template_b,
        "mean_delta_std": mean_delta_std,
        "var_fraction": var_delta / var_original,
        "cohens_d_delta": cohens_d,
        "delta_classify_acc": scores.mean(),
    }


# ---------------------------------------------------------------------------
# D. Null and artifact baselines
# ---------------------------------------------------------------------------

def run_baselines(split: str):
    """
    Cheap surface baselines that can embarrass the sentence encoder.
    """
    print(f"\n{'='*70}")
    print(f"D. NULL AND ARTIFACT BASELINES | split={split}")
    print(f"{'='*70}")

    rows = load_split(split)
    y = np.array([1 if r["answer"] else 0 for r in rows])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    def eval_baseline(name, X, y):
        scores = cross_validate(pipe(), X, y, cv=cv,
                                scoring=["accuracy", "balanced_accuracy", "roc_auc"],
                                return_train_score=False)
        acc  = scores["test_accuracy"].mean()
        bacc = scores["test_balanced_accuracy"].mean()
        auc  = scores["test_roc_auc"].mean()
        print(f"  {name:40} acc={acc:.4f}  bacc={bacc:.4f}  auc={auc:.4f}")
        return {"baseline": name, "acc": round(acc, 4), "bacc": round(bacc, 4), "auc": round(auc, 4)}

    # 1. Random labels (lower bound)
    np.random.seed(42)
    y_random = np.random.permutation(y)
    results.append(eval_baseline("random labels", np.zeros((len(y), 1)), y_random))

    # 2. Majority class
    maj_pred = np.ones((len(y), 1)) * y.mean()
    results.append(eval_baseline("majority class constant", maj_pred, y))

    # 3. Equation lengths only
    X_len = np.array([[len(r["equation1"]), len(r["equation2"]),
                       len(r["equation1"]) - len(r["equation2"])] for r in rows])
    results.append(eval_baseline("equation lengths only", X_len, y))

    # 4. Operator/symbol counts
    def count_features(eq):
        return [
            eq.count("*"),
            eq.count("+"),
            eq.count("-"),
            eq.count("("),
            eq.count("="),
            len(set(c for c in eq if c.isalpha())),  # unique variables
            len(eq),  # total length
        ]

    X_ops = np.array([[*count_features(r["equation1"]), *count_features(r["equation2"])] for r in rows])
    results.append(eval_baseline("operator/symbol counts", X_ops, y))

    # 5. Character n-gram TF-IDF
    texts = [r["equation1"] + " IMPLIES " + r["equation2"] for r in rows]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=500)
    X_tfidf = tfidf.fit_transform(texts).toarray()
    results.append(eval_baseline("char TF-IDF (2-4 gram)", X_tfidf, y))

    # 6. Shuffle eq2 across rows (breaks pairing, keeps individual equations)
    np.random.seed(42)
    shuffled_rows = [{"equation1": rows[i]["equation1"],
                      "equation2": rows[np.random.randint(len(rows))]["equation2"]}
                     for i in range(len(rows))]
    texts_shuffled = [r["equation1"] + " IMPLIES " + r["equation2"] for r in shuffled_rows]
    X_shuf = tfidf.transform(texts_shuffled).toarray()
    results.append(eval_baseline("shuffled eq2 (broken pairing)", X_shuf, y))

    # 7. Variable-renamed TF-IDF (canonical variable names)
    import re
    def canonicalize(eq):
        vars_found = list(dict.fromkeys(re.findall(r'\b[a-z]\b', eq)))
        for i, v in enumerate(vars_found):
            eq = re.sub(rf'\b{v}\b', f"VAR{i}", eq)
        return eq

    texts_canon = [canonicalize(r["equation1"]) + " IMPLIES " + canonicalize(r["equation2"]) for r in rows]
    X_canon = tfidf.fit_transform(texts_canon).toarray()
    results.append(eval_baseline("canonical variable names (TF-IDF)", X_canon, y))

    # 8. Swap equation order (eq2 first)
    texts_swap = [r["equation2"] + " IMPLIES " + r["equation1"] for r in rows]
    X_swap = tfidf.fit_transform(texts_swap).toarray()
    results.append(eval_baseline("swapped equation order", X_swap, y))

    # 9. Bag of parentheses/operators only (no variables)
    def paren_profile(eq):
        return "".join(c for c in eq if c in "()+-*/=")
    texts_paren = [paren_profile(r["equation1"]) + "|" + paren_profile(r["equation2"]) for r in rows]
    tfidf2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=200)
    X_paren = tfidf2.fit_transform(texts_paren).toarray()
    results.append(eval_baseline("parenthesis/operator profile only", X_paren, y))

    df = pd.DataFrame(results)
    print(f"\nSummary:")
    print(f"  Best baseline:  {df.loc[df['auc'].idxmax(), 'baseline']} (AUC {df['auc'].max():.4f})")
    print(f"  MPNET target:   natural template (AUC ~0.9496 on normal split)")

    return df


# ---------------------------------------------------------------------------
# E. Encoder robustness
# ---------------------------------------------------------------------------

def run_encoder_comparison(split: str, template: str = "natural"):
    """
    Replicate classification across multiple encoders.
    Requires embeddings to be pre-computed for each.
    """
    print(f"\n{'='*70}")
    print(f"E. ENCODER ROBUSTNESS | split={split} | template={template}")
    print(f"{'='*70}")

    encoders = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ]

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name in encoders:
        try:
            X, y = load_embeddings(split, template, model_name)
        except FileNotFoundError:
            print(f"  {model_name}: NOT CACHED — run sair_experiment.py --model {model_name} --steps embed first")
            continue

        scores = cross_validate(pipe(), X, y, cv=cv,
                                scoring=["accuracy", "balanced_accuracy", "roc_auc"],
                                return_train_score=False)

        res = {
            "model":  model_name,
            "acc":    round(scores["test_accuracy"].mean(), 4),
            "bacc":   round(scores["test_balanced_accuracy"].mean(), 4),
            "auc":    round(scores["test_roc_auc"].mean(), 4),
            "acc_std": round(scores["test_accuracy"].std(), 4),
        }
        print(f"  {model_name:35} acc={res['acc']:.4f}  bacc={res['bacc']:.4f}  auc={res['auc']:.4f}")
        results.append(res)

    if len(results) > 1:
        accs = [r["acc"] for r in results]
        print(f"\n  Spread across encoders: {max(accs) - min(accs):.4f}")
        if max(accs) - min(accs) < 0.05:
            print(f"  [GOOD] Pattern is robust — not encoder-specific geometry")
        else:
            print(f"  [NOTE] Large spread — results may be encoder-specific")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Rigorous experimental controls for SAIR")
    parser.add_argument("--split",   default="normal", choices=["normal", "hard", "hard1", "hard2", "all"])
    parser.add_argument("--model",   default="all-mpnet-base-v2")
    parser.add_argument("--checks",  nargs="+", default=["all"],
                        choices=["all", "grouped", "transfer", "delta", "baselines", "encoders"])
    parser.add_argument("--group-by", default="eq1", choices=["eq1", "eq2", "either"])
    return parser.parse_args()


def main():
    args   = parse_args()
    checks = set(args.checks)
    if "all" in checks:
        checks = {"grouped", "transfer", "delta", "baselines", "encoders"}

    splits = ["normal", "hard", "hard1", "hard2"] if args.split == "all" else [args.split]
    all_results = {}

    for split in splits:
        print(f"\n\n{'#'*70}")
        print(f"# SPLIT: {split}")
        print(f"{'#'*70}")

        if "grouped" in checks:
            df = run_grouped_cv(split, args.model, group_by=args.group_by)
            all_results[f"grouped_{split}"] = df

        if "transfer" in checks:
            df = run_cross_template_transfer(split, args.model)
            all_results[f"transfer_{split}"] = df

        if "delta" in checks:
            pairs = [("natural", "raw"), ("natural", "conjoined"), ("natural", "eq1_only")]
            for ta, tb in pairs:
                run_delta_analysis(split, args.model, ta, tb)

        if "baselines" in checks:
            df = run_baselines(split)
            all_results[f"baselines_{split}"] = df

        if "encoders" in checks:
            df = run_encoder_comparison(split, template="natural")
            all_results[f"encoders_{split}"] = df

    # Save all results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in all_results.items():
        if df is not None and len(df) > 0:
            path = OUTPUT_DIR / f"rigorous_{name}.csv"
            df.to_csv(path, index=False)
            print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
