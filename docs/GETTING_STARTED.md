---
title: Getting Started
---

# Getting Started with LLM Bias Traffic Light

Get up and running in 5 minutes.

## Prerequisites

- Python 3.8+
- Git

## Installation (2 minutes)

### 1. Clone and navigate

```bash
cd llm-bias-traffic-light
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements_experiments.txt
```

This installs: PyTorch, sentence-transformers, scikit-learn, pandas, matplotlib, and more.

**Note:** First run will auto-download the embedder model (~50MB) and cache it locally. This happens silently on first import.

### 4. Configure

```bash
cp config.template.json config.json
# Edit config.json and add your API keys if needed (optional for basic experiments)
```

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for detailed options.

## Run Your First Experiment (2 minutes)

### Analyze one bias category

```bash
python run_bbq_geometry.py --category Gender_identity --max-examples 100
```

**What happens:**
1. Loads 100 examples from Gender_identity.jsonl
2. Embeds each with three answer types: biased, anti-biased, unknown
3. Tests if these separate in embedding space (clustering, classification, PCA)
4. Saves metrics to `results/bbq_geometry/Gender_identity/` (CSV + plots)

**Output files:**
- `metrics.csv` — clustering accuracy, logistic regression AUC, Cohen's d, etc.
- `*.png` — PCA scatter, t-SNE, cosine similarity distributions
- `run.log` — script output (timestamps, model info, warnings)

### Analyze all categories

```bash
python run_all_geometry.py --max-examples 200 --no-plots
```

This batches over all 11 BBQ categories, runs geometry analysis on each, and merges results into a summary CSV.

## Next Steps

- **See all experiments:** [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) — matrix of what's available
- **Customize config:** [CONFIG_GUIDE.md](CONFIG_GUIDE.md) — deep dive on each setting
- **Add your own analysis:** [EXTENDING.md](EXTENDING.md) — template for custom metrics
- **Use the library:** `from bias_scorer import analyze` — single-call API for bias scoring

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch sentence-transformers
```

### "No such file or directory: BBQ_Data/"
Make sure you're in the `llm-bias-traffic-light/` directory:
```bash
cd bias-semaphor-project/llm-bias-traffic-light
python run_bbq_geometry.py --help
```

### Out of memory on large batches
Reduce `--max-examples`:
```bash
python run_all_geometry.py --max-examples 50
```

### Want to use a better embedder?
Edit `config.json` and set `models.embedder` to `all-mpnet-base-v2` (slower, higher quality). See [CONFIG_GUIDE.md](CONFIG_GUIDE.md#embedder-models).

## Quick Command Reference

| Goal | Command |
|------|---------|
| Test one category | `python run_bbq_geometry.py --category Gender_identity` |
| Test all categories | `python run_all_geometry.py` |
| Test custom subset | `python run_bbq_geometry.py --categories Gender_identity Age Religion` |
| Skip figure generation | `python run_bbq_geometry.py --no-plots` |
| Use more examples | `python run_bbq_geometry.py --max-examples 500` |
| Try different templates | `python run_bbq_geometry.py --templates answer_only cq_answer question_answer` |

## Got questions?

- **How do I interpret results?** → [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md#interpreting-results)
- **How do I probe live LLMs?** → [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md#5-llm-probing)
- **How do I add a custom metric?** → [EXTENDING.md](EXTENDING.md)
