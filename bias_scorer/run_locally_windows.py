"""
BBQ Bias Measurement, local runner (sentence-transformers embedder).
"""

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent
BBQ_DATA_DIR = PROJECT_DIR / "BBQ_Data"
BBQ_FILE = BBQ_DATA_DIR / "Gender_identity.jsonl"
MAX_EXAMPLES = None  # Set to None to load all examples
OUTPUT_DIR = PROJECT_DIR / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding model - using sentence-transformers (Windows compatible)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight and fast

print(f"Device: {DEVICE}")
print(f"Project directory: {PROJECT_DIR}")
print(f"BBQ file: {BBQ_FILE}")
print(f"File exists: {BBQ_FILE.exists()}")
print(f"Embedding model: {EMBEDDING_MODEL}")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# BBQ LOADER
# ============================================================================

def load_bbq(filepath, max_examples=None):
    """Load BBQ JSONL, return list of parsed entries."""
    entries = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            entry = json.loads(line.strip())
            entries.append(entry)
    return entries


def identify_answer_roles(entry):
    """
    Determine which answer is male/female/unknown from answer_info.
    Returns dict with male/female/unknown answer mappings.
    """
    info = entry["answer_info"]
    roles = {}

    for key in ["ans0", "ans1", "ans2"]:
        group_label = info[key][1].lower()
        # Male categories: man, boy, male, trans male
        if group_label in ["man", "m", "boy", "male", "trans_m"]:
            roles["male_ans"] = entry[key]
            roles["male_key"] = key
        # Female categories: woman, girl, female, trans female
        elif group_label in ["woman", "f", "girl", "female", "trans_f"]:
            roles["female_ans"] = entry[key]
            roles["female_key"] = key
        # Unknown/uncertain categories
        elif group_label == "unknown":
            roles["unknown_ans"] = entry[key]
            roles["unknown_key"] = key

    # Get stereotyped group from metadata
    stereo = entry.get("additional_metadata", {}).get("stereotyped_groups", [])
    roles["stereotyped_group"] = stereo[0] if stereo else "unknown"

    return roles


# ============================================================================
# EMBEDDING HELPER
# ============================================================================

def embed_texts(model, texts, batch_size=32):
    """Embed a list of strings using sentence-transformers, return tensor."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


# ============================================================================
# BIAS SCORING
# ============================================================================

def compute_bias_scores(model, entries):
    """
    For each BBQ entry, compute:
    1. Contextualized bias: cos_sim(cq, cq+male) - cos_sim(cq, cq+female)
    2. Predicted answer (argmax similarity)
    3. Accuracy against ground truth
    """
    results = []

    # Pre-build all text variants
    cq_texts = []
    cq_male_texts = []
    cq_female_texts = []
    cq_unknown_texts = []
    valid_entries = []

    for entry in entries:
        roles = identify_answer_roles(entry)

        # Skip entries where we can't identify all three roles
        if not all(k in roles for k in ["male_ans", "female_ans", "unknown_ans"]):
            continue

        cq = entry["context"] + " " + entry["question"]
        cq_texts.append(cq)
        cq_male_texts.append(cq + " " + roles["male_ans"])
        cq_female_texts.append(cq + " " + roles["female_ans"])
        cq_unknown_texts.append(cq + " " + roles["unknown_ans"])
        valid_entries.append((entry, roles))

    print(f"Embedding {len(valid_entries)} valid entries...")
    print(f"  (Skipped {len(entries) - len(valid_entries)} entries with missing roles)")

    # Batch embed all variants
    e_cq = embed_texts(model, cq_texts)
    e_male = embed_texts(model, cq_male_texts)
    e_female = embed_texts(model, cq_female_texts)
    e_unknown = embed_texts(model, cq_unknown_texts)

    print("Computing bias scores...")

    for i, (entry, roles) in enumerate(valid_entries):
        # Cosine similarities
        sim_male = F.cosine_similarity(e_cq[i : i + 1], e_male[i : i + 1]).item()
        sim_female = F.cosine_similarity(e_cq[i : i + 1], e_female[i : i + 1]).item()
        sim_unknown = F.cosine_similarity(e_cq[i : i + 1], e_unknown[i : i + 1]).item()

        # Core bias score: positive = pulled toward male, negative = toward female
        bias_score = sim_male - sim_female

        # Predicted answer (argmax similarity)
        sims = {"male": sim_male, "female": sim_female, "unknown": sim_unknown}
        predicted = max(sims, key=sims.get)

        # Determine predicted key
        if predicted == "unknown":
            predicted_key = roles["unknown_key"]
        elif predicted == "male":
            predicted_key = roles["male_key"]
        else:
            predicted_key = roles["female_key"]

        # Ground truth
        label_idx = entry["label"]
        label_key = f"ans{label_idx}"
        is_correct = predicted_key == label_key

        results.append(
            {
                "example_id": entry["example_id"],
                "context_condition": entry["context_condition"],
                "question_polarity": entry["question_polarity"],
                "stereotyped_group": roles["stereotyped_group"],
                "context": entry["context"],
                "question": entry["question"],
                "ans_male": roles["male_ans"],
                "ans_female": roles["female_ans"],
                "ans_unknown": roles["unknown_ans"],
                "sim_male": sim_male,
                "sim_female": sim_female,
                "sim_unknown": sim_unknown,
                "bias_score": bias_score,
                "abs_bias": abs(bias_score),
                "predicted": predicted,
                "is_correct": is_correct,
                "is_ambiguous": entry["context_condition"] == "ambig",
                "label": label_idx,
            }
        )

    return pd.DataFrame(results)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(df, save_dir=OUTPUT_DIR):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sentence-BERT Embedding Bias on BBQ Gender Identity", fontsize=14, fontweight="bold")

    # 1. Bias score histogram, ambiguous only
    ax = axes[0, 0]
    ambig = df[df["is_ambiguous"]]
    if len(ambig) > 0:
        ax.hist(ambig["bias_score"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Unbiased (0)")
        ax.axvline(
            ambig["bias_score"].mean(),
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Mean = {ambig['bias_score'].mean():.4f}",
        )
        ax.set_xlabel("Bias Score (+ = male, − = female)")
        ax.set_ylabel("Count")
        ax.set_title("Bias Score Distribution (Ambiguous)")
        ax.legend()

    # 2. Predicted answer distribution, ambiguous only
    ax = axes[0, 1]
    if len(ambig) > 0:
        pred_counts = ambig["predicted"].value_counts()
        colors = {"male": "#4a90d9", "female": "#d94a7a", "unknown": "#7ac36a"}
        bars = ax.bar(pred_counts.index, pred_counts.values, color=[colors.get(x, "gray") for x in pred_counts.index])
        ax.set_title("Predicted Answers (Ambiguous)")
        ax.set_ylabel("Count")
        # Add percentage labels
        total = pred_counts.sum()
        for bar, count in zip(bars, pred_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{count/total*100:.1f}%",
                ha="center",
                fontsize=11,
            )

    # 3. Similarity distributions
    ax = axes[1, 0]
    if len(ambig) > 0:
        ax.hist(ambig["sim_male"], bins=30, alpha=0.5, label="Male", color="#4a90d9")
        ax.hist(ambig["sim_female"], bins=30, alpha=0.5, label="Female", color="#d94a7a")
        ax.hist(ambig["sim_unknown"], bins=30, alpha=0.5, label="Unknown", color="#7ac36a")
        ax.set_xlabel("Cosine Similarity to context+question")
        ax.set_ylabel("Count")
        ax.set_title("Similarity Distributions (Ambiguous)")
        ax.legend()

    # 4. Bias by stereotyped group - Boxplots
    ax = axes[1, 1]
    if "stereotyped_group" in ambig.columns and len(ambig) > 0:
        # Filter groups with enough data
        group_counts = ambig["stereotyped_group"].value_counts()
        valid_groups = group_counts[group_counts >= 5].index.tolist()

        if len(valid_groups) > 0:
            filtered_data = ambig[ambig["stereotyped_group"].isin(valid_groups)]

            # Create boxplot with all individual values
            bp = ax.boxplot(
                [filtered_data[filtered_data["stereotyped_group"] == group]["bias_score"].values
                 for group in valid_groups],
                labels=valid_groups,
                patch_artist=True,
                showfliers=True,
                widths=0.6
            )

            # Style the boxplots
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.6)

            for whisker in bp['whiskers']:
                whisker.set(color='#555555', linewidth=1.5)

            for cap in bp['caps']:
                cap.set(color='#555555', linewidth=1.5)

            for median in bp['medians']:
                median.set(color='red', linewidth=2)

            ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Unbiased (0)")
            ax.set_title("Bias Score Distribution by Stereotyped Group")
            ax.set_ylabel("Bias Score (+ = male, − = female)")
            ax.set_xlabel("Stereotyped Group")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No stereotype info", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    save_path = save_dir / "bias_results_windows.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()


def print_summary(df):
    """Print key summary statistics."""
    ambig = df[df["is_ambiguous"]]
    disambig = df[~df["is_ambiguous"]]

    print("=" * 60)
    print("BIAS MEASUREMENT SUMMARY")
    print("=" * 60)
    print(f"\nTotal entries:        {len(df)}")
    print(f"  Ambiguous:          {len(ambig)}")
    print(f"  Disambiguated:      {len(disambig)}")

    if len(ambig) > 0:
        print(f"\n--- Ambiguous Context (ground truth = 'unknown') ---")
        print(f"  Mean bias score:    {ambig['bias_score'].mean():.4f}")
        print(f"  Std bias score:     {ambig['bias_score'].std():.4f}")
        print(f"  Median bias score:  {ambig['bias_score'].median():.4f}")
        print(f"  |bias| > 0.01:     {(ambig['abs_bias'] > 0.01).sum()}/{len(ambig)} ({(ambig['abs_bias'] > 0.01).mean()*100:.1f}%)")
        print(f"\n  Predicted answers:")
        for ans, count in ambig["predicted"].value_counts().items():
            print(f"    {ans:>10}: {count:>4} ({count/len(ambig)*100:.1f}%)")
        print(f"  Accuracy (unknown): {ambig['is_correct'].mean()*100:.1f}%")

    if len(disambig) > 0:
        print(f"\n--- Disambiguated Context ---")
        print(f"  Accuracy:           {disambig['is_correct'].mean()*100:.1f}%")
        print(f"  Mean bias score:    {disambig['bias_score'].mean():.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading BBQ data...")
    entries = load_bbq(BBQ_FILE, max_examples=MAX_EXAMPLES)
    print(f"Loaded {len(entries)} entries")

    ambig_count = sum(1 for e in entries if e["context_condition"] == "ambig")
    disambig_count = len(entries) - ambig_count
    print(f"  Ambiguous: {ambig_count}, Disambiguated: {disambig_count}")

    print("\nInitializing Sentence-BERT embedder...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        print("Embedder ready.")
    except ImportError as e:
        print(f"ERROR: Could not import sentence-transformers: {e}")
        print("Please install with: pip install sentence-transformers")
        return None

    print("\nComputing bias scores...")
    df = compute_bias_scores(model, entries)

    print_summary(df)
    plot_results(df, OUTPUT_DIR)

    # Save CSV
    csv_path = OUTPUT_DIR / "bias_scores_windows.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    df = main()
