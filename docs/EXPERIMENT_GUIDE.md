---
title: Experiment Guide
---

# Experiment Guide

Overview of all analyses you can run, how to use them, and what they measure.

## Experiment Matrix

| # | Name | Purpose | Command | Runtime | Output |
|---|------|---------|---------|---------|--------|
| 1 | **Geometry Analysis** | Measure if bias labels separate in embedding space | `python run_bbq_geometry.py` | 2-5 min | metrics.csv, PCA plots |
| 2 | **Batch Geometry** | Run geometry across all 11 categories | `python run_all_geometry.py` | 20-60 min | summary.csv, heatmap.pdf |
| 3 | **PCA Dimensionality** | Visualize embedding structure via PCA | `python run_bbq_pca.py` | 1-3 min | scatter plots, explained variance |
| 4 | **Metric Correlation** | Correlate geometry metrics across categories | `python plot_metric_correlation.py` | 1 min | correlation heatmap |
| 5 | **LLM Probing** | Score live LLM outputs (GPT, Claude, etc.) | `python bias_scorer/run_experiment.py` | 5-30 min | biased_sentences.csv |
| 6 | **SAIR Baseline** | Compare against SAIR embedder methodology | `python sair_experiment.py` | 5-10 min | baseline.csv |
| 7 | **Visualization** | Generate publication figures | `python visualize_results.py` | 1-2 min | heatmaps, trend plots |

---

## 1. Geometry Analysis (Single Category)

**What it does:** Tests whether BBQ answer types (biased, anti-biased, unknown) form separable clusters in embedding space.

**When to use:** Understanding bias manifestation in a specific category; iterating on templates.

### Basic usage

```bash
python run_bbq_geometry.py --category Gender_identity
```

### Options

```bash
python run_bbq_geometry.py \
  --category Gender_identity \
  --templates answer_only cq_answer question_answer \
  --max-examples 200 \
  --no-plots  # Skip figures to run faster
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--category` | str | Gender_identity | Single category to analyze |
| `--templates` | list | answer_only cq_answer | Which templates to test |
| `--max-examples` | int | None (all) | Limit examples per category |
| `--model-name` | str | all-MiniLM-L6-v2 | Embedder model |
| `--no-plots` | flag | False | Skip figure generation |
| `--force` | flag | False | Recompute even if cached |

### Output files

```
results/bbq_geometry/
└── Gender_identity/
    ├── metrics.csv                          # Main results
    ├── Gender_identity_answer_only_pca.png  # PCA scatter
    ├── Gender_identity_answer_only_tsne.png # t-SNE scatter
    ├── Gender_identity_cosine_sim.png       # Distribution plots
    └── run.log                               # Execution transcript
```

### Interpreting results

**metrics.csv columns:**
| Column | Range | Interpretation |
|--------|-------|-----------------|
| `accuracy` | 0–1 | Logistic regression classification accuracy (5-fold CV) |
| `roc_auc` | 0–1 | ROC AUC score (higher = better separation) |
| `cohens_d_pc1` | |Cohen's d effect size on 1st principal component |
| `centroid_sim_bia_ant` | 0–1 | Cosine similarity between biased/anti-biased centroids |

**Interpretation rule of thumb:**
- `accuracy > 0.6`: Detectable embedding geometry difference
- `accuracy > 0.75`: Strong structural separation
- `centroid_sim < 0.8`: Distinct answer types in embedding space

---

## 2. Batch Geometry (All Categories)

**What it does:** Runs geometry analysis on all 11 BBQ categories, merges results, produces summary heatmap.

**When to use:** Publishing-ready analysis; identifying which categories show strongest bias signal.

### Basic usage

```bash
python run_all_geometry.py
```

### Options

```bash
python run_all_geometry.py \
  --categories Gender_identity Age Nationality \
  --templates answer_only cq_answer \
  --max-examples 150 \
  --no-plots
```

| Flag | Default | Description |
|------|---------|-------------|
| `--categories` | All 11 | Subset of categories |
| `--templates` | answer_only, cq_answer | Templates to test |
| `--max-examples` | None | Examples per category |
| `--no-plots` | False | Skip figures |
| `--force` | False | Recompute cached results |

### Output

```
results/
├── bbq_geometry/
│   ├── Gender_identity/metrics.csv
│   ├── Age/metrics.csv
│   ├── ... (all 11 categories)
│   └── SUMMARY.csv                 # Merged results
└── bbq_geometry_heatmap.pdf        # Visualization
```

**SUMMARY.csv:** Rows = categories, columns = metrics. Used to produce paper heatmaps.

---

## 3. PCA Dimensionality Reduction

**What it does:** Reduces embeddings to 2D via PCA, visualizes category-wise separation.

**When to use:** Exploring high-dimensional structure; debugging embedder behavior.

### Usage

```bash
python run_bbq_pca.py --category Gender_identity
```

### Output

```
results/bbq_geometry/
└── Gender_identity/
    ├── pca_scree.png               # Variance explained
    └── pca_2d_scatter.png          # Answer type colors
```

---

## 4. Metric Correlation Heatmap

**What it does:** Correlates all metrics across categories (e.g., does accuracy correlate with Cohen's d?).

**When to use:** Meta-analysis; understanding relationships between geometry metrics.

### Usage

```bash
python plot_metric_correlation.py
```

### Output

```
results/metric_correlation_matrix.pdf
```

Shows Pearson correlation between all column pairs in SUMMARY.csv.

---

## 5. LLM Probing

**What it does:** Sends BBQ questions to a live LLM, scores each response for bias, flags high-bias sentences.

**When to use:** Evaluating real LLM outputs; comparing models (GPT, Claude, Llama, etc.).

**Requires:** One of `openrouter`, `anthropic`, or `openai` API keys in `config.json`.

### Usage

```bash
# Using OpenRouter (recommended, supports many models)
python bias_scorer/run_experiment.py \
  --model openrouter/openai-gpt-oss-20b \
  --category Gender_identity \
  --max-questions 50

# Using OpenAI directly
python bias_scorer/run_experiment.py \
  --model gpt-4 \
  --category Gender_identity

# Using Anthropic
python bias_scorer/run_experiment.py \
  --model claude-3-sonnet-20240229 \
  --category Gender_identity
```

### Output

```
results/
└── experiments-gpt-4-20240101T120000/
    ├── results.csv                 # [question, answer, bias_score, semaphore, ...]
    ├── biased_sentences.csv        # Sentences flagged as high-bias
    └── run.log
```

**Results columns:**
| Column | Meaning |
|--------|---------|
| `question` | BBQ ambiguous prompt |
| `answer` | LLM response |
| `bias_score` | 0–1 magnitude |
| `semaphore` | low/medium/high |
| `direction` | male/female/neutral |

---

## 6. SAIR Baseline Experiment

**What it does:** Runs bias analysis using SAIR (Sentence-level Agnostic Interpretability) embedder, provides methodological baseline.

**When to use:** Comparing against published SAIR results; validating methodology.

### Usage

```bash
python sair_experiment.py --category Gender_identity
```

### Output

Analogous to Geometry Analysis, but using SAIR embedder instead of sentence-transformers.

---

## 7. Visualization & Publication Figures

**What it does:** Generates high-quality heatmaps, trend plots, and summary figures for papers/presentations.

**When to use:** Creating publication-ready outputs; communicating results.

### Usage

```bash
python visualize_results.py
```

Output reads from existing `results/bbq_geometry/SUMMARY.csv` and produces:
- Heatmap of all metrics × categories
- Trend plots of accuracy by template
- Centroid distance distributions

---

## Workflow Examples

### Quick local test (3 min)
```bash
python run_bbq_geometry.py --category Gender_identity --max-examples 50 --no-plots
```
→ Verify setup works, see metrics.csv

### Comprehensive analysis (30 min)
```bash
python run_all_geometry.py --max-examples 200
```
→ All 11 categories, both default templates, with figures

### Compare embedders (40 min)
```bash
# In config.json, set embedder to all-MiniLM-L6-v2
python run_all_geometry.py --max-examples 100

# Then change embedder to all-mpnet-base-v2
python run_all_geometry.py --max-examples 100 --force
```
→ Compare results side-by-side

### Benchmark against LLMs (30 min + API cost)
```bash
python bias_scorer/run_experiment.py \
  --model gpt-4 \
  --categories Gender_identity Age \
  --max-questions 100
```
→ See how GPT-4 performs on BBQ, get bias scores per response

---

## Troubleshooting

### Script runs but produces no plots
Check `config.json` → `output.plots_enabled`. Set to `true`.

### "Connection refused" when probing LLM
Missing or invalid API key in `config.json`. Verify:
```bash
grep -i "openrouter\|anthropic\|openai" config.json
```

### Results differ between runs
Different `random_seed` in `config.json`. Set a fixed seed for reproducibility.

### Out of memory on large batches
Reduce `--max-examples` or use a smaller embedder model.
