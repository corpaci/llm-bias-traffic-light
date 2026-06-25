"""
Visualize SAIR experiment results across splits and templates.
Creates heatmaps, line plots, and scatter plots from cluster_results.csv and classify_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup
PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results" / "sair"
OUTPUT_DIR = RESULTS_DIR

# Load CSVs
cluster_df = pd.read_csv(RESULTS_DIR / "cluster_results.csv")
classify_df = pd.read_csv(RESULTS_DIR / "classify_results.csv")

# Merge for easier analysis
df = cluster_df.merge(classify_df, on=["split", "template"])

print(f"Loaded {len(df)} results")
print(f"Splits: {df['split'].unique()}")
print(f"Templates: {df['template'].unique()}\n")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# ===== 1. Accuracy Heatmap (split × template) =====
ax1 = plt.subplot(3, 3, 1)
acc_pivot = df.pivot_table(values="acc_mean", index="template", columns="split", aggfunc="first")
sns.heatmap(acc_pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5, ax=ax1, cbar_kws={"label": "Accuracy"}, vmin=0.5, vmax=1.0)
ax1.set_title("Accuracy Across Splits & Templates", fontweight="bold", fontsize=11)
ax1.set_xlabel("Data Split")
ax1.set_ylabel("Text Template")

# ===== 2. AUC Heatmap (split × template) =====
ax2 = plt.subplot(3, 3, 2)
auc_pivot = df.pivot_table(values="auc_mean", index="template", columns="split", aggfunc="first")
sns.heatmap(auc_pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5, ax=ax2, cbar_kws={"label": "AUC"}, vmin=0.5, vmax=1.0)
ax2.set_title("AUC Across Splits & Templates", fontweight="bold", fontsize=11)
ax2.set_xlabel("Data Split")
ax2.set_ylabel("Text Template")

# ===== 3. Cohen's d Heatmap =====
ax3 = plt.subplot(3, 3, 3)
cohens_pivot = df.pivot_table(values="cohens_d_pc1", index="template", columns="split", aggfunc="first")
sns.heatmap(cohens_pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax3, cbar_kws={"label": "Cohen's d"})
ax3.set_title("Cohen's d (Effect Size) Across Splits", fontweight="bold", fontsize=11)
ax3.set_xlabel("Data Split")
ax3.set_ylabel("Text Template")

# ===== 4. Accuracy by Template (across all splits) =====
ax4 = plt.subplot(3, 3, 4)
template_acc = df.groupby("template")["acc_mean"].agg(["mean", "std"]).sort_values("mean", ascending=False)
colors = ["#2ecc71" if x > 0.85 else "#f39c12" if x > 0.75 else "#e74c3c" for x in template_acc["mean"]]
ax4.bar(range(len(template_acc)), template_acc["mean"], yerr=template_acc["std"], capsize=5, color=colors, alpha=0.7, edgecolor="black")
ax4.set_xticks(range(len(template_acc)))
ax4.set_xticklabels(template_acc.index, rotation=45, ha="right")
ax4.set_ylabel("Mean Accuracy")
ax4.set_title("Template Performance (Average Across Splits)", fontweight="bold", fontsize=11)
ax4.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
ax4.set_ylim([0.5, 1.0])
ax4.legend()
ax4.grid(axis="y", alpha=0.3)

# ===== 5. Accuracy by Split (across all templates) =====
ax5 = plt.subplot(3, 3, 5)
split_acc = df.groupby("split")["acc_mean"].agg(["mean", "std"]).reindex(["normal", "hard", "hard1", "hard2"])
colors_split = ["#2ecc71" if x > 0.80 else "#f39c12" if x > 0.70 else "#e74c3c" for x in split_acc["mean"]]
ax5.bar(range(len(split_acc)), split_acc["mean"], yerr=split_acc["std"], capsize=5, color=colors_split, alpha=0.7, edgecolor="black")
ax5.set_xticks(range(len(split_acc)))
ax5.set_xticklabels(split_acc.index, rotation=45, ha="right")
ax5.set_ylabel("Mean Accuracy")
ax5.set_title("Split Difficulty (Average Across Templates)", fontweight="bold", fontsize=11)
ax5.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
ax5.set_ylim([0.5, 1.0])
ax5.legend()
ax5.grid(axis="y", alpha=0.3)

# ===== 6. Centroid Similarity by Template =====
ax6 = plt.subplot(3, 3, 6)
template_centroid = df.groupby("template")["centroid_cosine_sim"].mean().sort_values()
ax6.barh(range(len(template_centroid)), template_centroid.values, color="#3498db", alpha=0.7, edgecolor="black")
ax6.set_yticks(range(len(template_centroid)))
ax6.set_yticklabels(template_centroid.index)
ax6.set_xlabel("Centroid Cosine Similarity")
ax6.set_title("Class Centroid Separation by Template", fontweight="bold", fontsize=11)
ax6.axvline(0.99, color="red", linestyle="--", alpha=0.5, label="Very close (0.99)")
ax6.set_xlim([0.97, 0.995])
ax6.legend(fontsize=9)
ax6.grid(axis="x", alpha=0.3)

# ===== 7. Cohen's d by Template =====
ax7 = plt.subplot(3, 3, 7)
template_cohens = df.groupby("template")["cohens_d_pc1"].mean().sort_values()
colors_cohens = ["#e74c3c" if x < -0.3 else "#f39c12" if x < 0 else "#2ecc71" for x in template_cohens.values]
ax7.barh(range(len(template_cohens)), np.abs(template_cohens.values), color=colors_cohens, alpha=0.7, edgecolor="black")
ax7.set_yticks(range(len(template_cohens)))
ax7.set_yticklabels(template_cohens.index)
ax7.set_xlabel("|Cohen's d|")
ax7.set_title("Effect Size (PC1 Separation) by Template", fontweight="bold", fontsize=11)
ax7.grid(axis="x", alpha=0.3)

# ===== 8. Cohen's d vs AUC Scatter =====
ax8 = plt.subplot(3, 3, 8)
for template in df["template"].unique():
    mask = df["template"] == template
    ax8.scatter(np.abs(df.loc[mask, "cohens_d_pc1"]), df.loc[mask, "auc_mean"],
               label=template, s=80, alpha=0.6, edgecolors="black", linewidth=0.5)
ax8.set_xlabel("|Cohen's d| (Effect Size)")
ax8.set_ylabel("AUC")
ax8.set_title("Clustering Strength vs Classification", fontweight="bold", fontsize=11)
ax8.legend(fontsize=8, loc="lower right")
ax8.grid(True, alpha=0.3)

# ===== 9. Performance Stability (Std Dev) =====
ax9 = plt.subplot(3, 3, 9)
stability = df.groupby("template")[["acc_std", "auc_std"]].mean()
x = np.arange(len(stability))
width = 0.35
ax9.bar(x - width/2, stability["acc_std"], width, label="Accuracy Std", alpha=0.7, color="#3498db", edgecolor="black")
ax9.bar(x + width/2, stability["auc_std"], width, label="AUC Std", alpha=0.7, color="#e74c3c", edgecolor="black")
ax9.set_xticks(x)
ax9.set_xticklabels(stability.index, rotation=45, ha="right")
ax9.set_ylabel("Standard Deviation")
ax9.set_title("Model Stability Across Folds", fontweight="bold", fontsize=11)
ax9.legend()
ax9.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "results_comprehensive.png", dpi=150, bbox_inches="tight")
print(f"\nSaved comprehensive visualization to: {OUTPUT_DIR / 'results_comprehensive.png'}")

# ===== Additional: Line plot showing performance degradation =====
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy across splits for each template
splits_order = ["normal", "hard", "hard1", "hard2"]
for template in sorted(df["template"].unique()):
    mask = df["template"] == template
    split_data = df.loc[mask].set_index("split").loc[splits_order]
    ax_a.plot(range(len(splits_order)), split_data["acc_mean"], marker="o", label=template, linewidth=2, markersize=8)

ax_a.set_xticks(range(len(splits_order)))
ax_a.set_xticklabels(splits_order)
ax_a.set_ylabel("Accuracy")
ax_a.set_xlabel("Data Split (→ harder)")
ax_a.set_title("Accuracy Degradation Across Splits", fontweight="bold", fontsize=12)
ax_a.legend(title="Template", fontsize=9)
ax_a.grid(True, alpha=0.3)
ax_a.set_ylim([0.5, 1.0])

# AUC across splits for each template
for template in sorted(df["template"].unique()):
    mask = df["template"] == template
    split_data = df.loc[mask].set_index("split").loc[splits_order]
    ax_b.plot(range(len(splits_order)), split_data["auc_mean"], marker="s", label=template, linewidth=2, markersize=8)

ax_b.set_xticks(range(len(splits_order)))
ax_b.set_xticklabels(splits_order)
ax_b.set_ylabel("AUC")
ax_b.set_xlabel("Data Split (→ harder)")
ax_b.set_title("AUC Across Splits", fontweight="bold", fontsize=12)
ax_b.legend(title="Template", fontsize=9)
ax_b.grid(True, alpha=0.3)
ax_b.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "performance_across_splits.png", dpi=150, bbox_inches="tight")
print(f"Saved split performance plot to: {OUTPUT_DIR / 'performance_across_splits.png'}")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print("\nBest performing template:")
best = df.loc[df["auc_mean"].idxmax()]
print(f"  {best['template']:15} split={best['split']:8} acc={best['acc_mean']:.4f} auc={best['auc_mean']:.4f} cohen_d={best['cohens_d_pc1']:.3f}")

print("\nWorst performing template:")
worst = df.loc[df["auc_mean"].idxmin()]
print(f"  {worst['template']:15} split={worst['split']:8} acc={worst['acc_mean']:.4f} auc={worst['auc_mean']:.4f} cohen_d={worst['cohens_d_pc1']:.3f}")

print("\nTemplate rankings (by avg AUC across all splits):")
rankings = df.groupby("template")["auc_mean"].mean().sort_values(ascending=False)
for i, (template, auc) in enumerate(rankings.items(), 1):
    print(f"  {i}. {template:15} avg_auc={auc:.4f}")

print("\nSplit difficulty (by avg accuracy across all templates):")
split_ranks = df.groupby("split")["acc_mean"].mean().sort_values(ascending=False)
for i, (split, acc) in enumerate(split_ranks.items(), 1):
    print(f"  {i}. {split:10} avg_acc={acc:.4f}")

print("\n" + "="*70)
