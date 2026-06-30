# LLM Bias Traffic Light

Measurement and visualization framework for LLM bias. Analyze how language models exhibit stereotypical associations using the BBQ (Bias Benchmark for QA) dataset.

## Quick Start (5 minutes)

→ **[Getting Started Guide](docs/GETTING_STARTED.md)** ←

```bash
pip install -r requirements_experiments.txt
cp config.template.json config.json
python run_bbq_geometry.py --category Gender_identity
```

See metrics in `results/bbq_geometry/Gender_identity/metrics.csv` and plots in `*.png`.

## What This Does

**Geometry Analysis:** Embeds BBQ questions and answers, tests whether biased/anti-biased/unknown answer types separate in embedding space.

**Metrics:** Classification accuracy, ROC AUC, Cohen's d on principal components, centroid distances.

**Experiments:** 7 different analysis types (see [Experiment Guide](docs/EXPERIMENT_GUIDE.md)).

## Repository Layout

### Core Analysis Tools
- **`run_bbq_geometry.py`**, Geometry analysis for a single category
- **`run_all_geometry.py`**, Batch analysis across all 11 categories + summary heatmap
- **`run_bbq_pca.py`**, PCA visualization
- **`plot_metric_correlation.py`**, Correlation matrix heatmap
- **`visualize_results.py`**, Publication figures

### Datasets & Loaders
- **`BBQ_Data/`**, 11 bias benchmark categories (Nationality, Religion, Gender_identity, etc.)
- **`DataLoader/`**, BBQ JSONL loader and utilities
  - `bbq_loader.py`, Load and filter examples
  - `example_usage.py`, Usage examples

### Core Library
- **`bias_scorer/`**, Bias measurement module (can be imported as a package)
  - `embedder.py`, Sentence-transformer wrapper
  - `scorer.py`, Bias scoring logic
  - `anchors.py`, Gender-based anchor embeddings (for scaling)
  - `run_experiment.py`, CLI for probing live LLMs
  - `INTEGRATION.md`, API reference for plugin backend

### Plugin (Browser Extension)
- **`LLM_Bias_Traffic_Light_PlugIn/`**, Browser extension + FastAPI backend
  - Integrates scoring into Chrome/Firefox workflows

### Documentation
- **`docs/GETTING_STARTED.md`**, 5-minute onboarding
- **`docs/CONFIG_GUIDE.md`**, Config option reference
- **`docs/EXPERIMENT_GUIDE.md`**, All 7 analysis types + examples
- **`docs/EXTENDING.md`**, How to add custom metrics
- **`config.template.json`**, Copy this to `config.json` and fill in API keys

## All Experiments at a Glance

| Name | Command | Purpose | Runtime |
|------|---------|---------|---------|
| **Geometry (single)** | `python run_bbq_geometry.py --category Gender_identity` | Measure embedding separation | 2–5 min |
| **Geometry (batch)** | `python run_all_geometry.py` | All categories + summary heatmap | 20–60 min |
| **PCA** | `python run_bbq_pca.py` | Dimensionality reduction | 1–3 min |
| **Correlation** | `python plot_metric_correlation.py` | Metric correlations | 1 min |
| **LLM Probing** | `python bias_scorer/run_experiment.py --model gpt-4` | Score live LLM outputs | 5–30 min |
| **SAIR Baseline** | `python sair_experiment.py` | Alternative embedder methodology | 5–10 min |
| **Visualization** | `python visualize_results.py` | Publication figures | 1–2 min |

→ See **[Experiment Guide](docs/EXPERIMENT_GUIDE.md)** for detailed options, outputs, and interpretation.

## Setup

### Requirements

- Python 3.8+
- ~2GB disk (for embedder models)
- Optional: API key for LLM probing (OpenRouter, Anthropic, or OpenAI)

### Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_experiments.txt
```

### Configuration

```bash
cp config.template.json config.json
# Edit config.json to add API keys (optional; geometry + viz work offline)
```

→ [Configuration Guide](docs/CONFIG_GUIDE.md)

## Common Commands

```bash
# Test one category (2 min)
python run_bbq_geometry.py --category Gender_identity --max-examples 100

# Full analysis, all categories (30 min)
python run_all_geometry.py

# Just metrics, no plots (faster)
python run_all_geometry.py --no-plots

# Different embedder
python run_bbq_geometry.py --model all-mpnet-base-v2 --category Gender_identity

# Try multiple templates
python run_bbq_geometry.py --templates answer_only cq_answer question_answer --category Age

# Score live LLM
python bias_scorer/run_experiment.py --model gpt-4 --category Gender_identity
```

## Interpreting Results

### metrics.csv columns

| Column | Meaning | Range |
|--------|---------|-------|
| `accuracy` | Logistic regression classification accuracy (5-fold CV) | 0–1 |
| `roc_auc` | ROC AUC for label separation | 0–1 |
| `cohens_d_pc1` | Effect size on 1st PC | |
| `centroid_sim_bia_ant` | Cosine sim (biased / anti-biased centroids) | 0–1 |

**Interpretation:**
- `accuracy > 0.6` → Detectable bias signal in embeddings
- `accuracy > 0.75` → Strong structural separation
- `centroid_sim < 0.8` → Distinct answer types

### Figures

- **PCA scatter**: Do answer types cluster? (color = answer type)
- **t-SNE scatter**: 2D overview of similarity structure
- **Cosine distributions**: How far apart are centroids?

→ [Experiment Guide – Interpreting Results](docs/EXPERIMENT_GUIDE.md#interpreting-results)

## DataLoader API

Load BBQ data programmatically:

```python
from bias_scorer.DataLoader.bbq_loader import BBQDataLoader, load_bbq_directory

# Single file
loader = BBQDataLoader("BBQ_Data/Gender_identity.jsonl")
all_examples = loader.load_all()

# Filtered subset
subset = loader.load_subset(
    context_condition="ambig",
    question_polarity="neg",
    max_examples=100
)

# All files
all_bbq = load_bbq_directory("BBQ_Data")
```

## bias_scorer API

Use the scoring library in your own code:

```python
from bias_scorer import analyze

result = analyze("The CEO was a woman.")

print(result.bias_score)        # 0–1 magnitude
print(result.direction)         # "male", "female", or "neutral"
print(result.semaphore)         # "low", "medium", "high"
print(result.explanation)       # Human-readable
print(result.biased_sentences)  # Flagged sentences (deep mode)
```

See [bias_scorer INTEGRATION.md](bias_scorer/INTEGRATION.md) for full API reference.

## For Developers

→ **[EXTENDING.md](docs/EXTENDING.md)**, Add custom metrics, create new experiments, debug.

### Adding a Metric

1. Edit `run_bbq_geometry.py` → locate `run_category()` function
2. Compute your metric (e.g., silhouette score)
3. Add to `result_row` dict
4. Re-run with `--force` to recompute

### Creating a Custom Experiment

Start from the [template in EXTENDING.md](docs/EXTENDING.md#template-run-analysis-on-a-custom-dataset).

## Tests

```bash
pytest tests/
```

Includes:
- `test_bias_scorer.py`, Embedder and scorer logic
- `test_integration.py`, End-to-end pipelines

## Troubleshooting

**Q: "No module named 'torch'"**
A: `pip install torch sentence-transformers`

**Q: "BBQ_Data not found"**
A: Make sure you're in the `llm-bias-traffic-light/` directory.

**Q: Results differ each run**
A: Set a fixed `random_seed` in `config.json`.

**Q: Out of memory**
A: Reduce `--max-examples` or use a smaller embedder model.

**Q: Can't probe LLMs (API errors)**
A: Check API key in `config.json` and verify account has quota.

See [Experiment Guide – Troubleshooting](docs/EXPERIMENT_GUIDE.md#troubleshooting) for more.

## Contact

Questions or contributions? Open an issue or reach out.
