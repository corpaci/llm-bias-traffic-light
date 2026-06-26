# Configuration Guide

Detailed reference for every config option.

## Overview

Config lives in `config.json` (copy from `config.template.json`). All fields are optional; defaults are sensible for local experiments.

## API Keys

### openrouter
- **Used by:** `bias_scorer/run_experiment.py` (LLM probing experiments)
- **Get key:** https://openrouter.ai
- **Example:** `"sk-or-v1-48d578529f87a02bb838bf6806dfcb98..."`
- **Leave blank if:** You only run geometry/visualization experiments 


### anthropic
- **Used by:** `bias_scorer/run_experiment.py` (optional alternative to OpenRouter)
- **Get key:** https://console.anthropic.com
- **Leave blank if:** You don't plan to use Claude for LLM probing

### openai
- **Used by:** `bias_scorer/run_experiment.py` (optional alternative)
- **Get key:** https://platform.openai.com/api-keys
- **Leave blank if:** You don't plan to use GPT for LLM probing

### gitlab_token
- **Used by:** CI/CD pipelines (internal)
- **Leave blank:** Unless you're running this in a GitLab CI environment

---

## Models

### embedder
| Option | Speed | Quality | Memory | Notes |
|--------|-------|---------|--------|-------|
| `all-MiniLM-L6-v2` (default) | ⚡⚡⚡ | ⭐⭐⭐ | ~50MB | Recommended for most analyses |
| `all-mpnet-base-v2` | ⚡⚡ | ⭐⭐⭐⭐ | ~400MB | Higher quality, slower (~2-3x) |
| `sentence-transformers/all-distilroberta-v1` | ⚡⚡ | ⭐⭐⭐ | ~300MB | Good middle ground |

**AutoDownload:** First run will fetch the model from Hugging Face (~50-400MB depending on choice) and cache it locally. This is silent and happens only once.

---

## Experiment Defaults

### categories
- **Type:** `null` or array of strings
- **Default:** `null` (use all 11 BBQ categories)
- **Examples:**
  ```json
  "categories": null,  // All 11
  "categories": ["Gender_identity", "Age"],  // Just these 2
  ```
- **Available:** `Gender_identity`, `Age`, `Nationality`, `Religion`, `SES`, `Sexual_orientation`, `Physical_appearance`, `Race_ethnicity`, `Disability_status`, `Race_x_gender`, `Race_x_SES`
- **Override at CLI:** `python run_bbq_geometry.py --categories Gender_identity Age`

### templates
- **Type:** array of template names
- **Default:** `["answer_only", "cq_answer"]`
- **Available:**
  - `answer_only` — just the answer text
  - `cq_answer` — context + question + answer  (SAIR baseline)
  - `question_answer` — question + answer, no context
  - `context_only` — control: same context for all labels
  - `separate` — concatenation of different embeddings (advanced)

### max_examples
- **Type:** `null` or integer
- **Default:** `null` (use all examples in each category)
- **Use case:** Quick testing or memory-constrained machines
- **Example:**
  ```json
  "max_examples": 100  // Limit to 100 examples per category
  ```

### random_seed
- **Type:** integer
- **Default:** `42`
- **Purpose:** Reproducibility across runs
- **Change if:** You want different random subsets

---

## Output

### results_dir
- **Type:** string
- **Default:** `results`
- **creates:** `results/bbq_geometry/`, `results/experiments-*/`, etc.

### figure_format
- **Type:** `"pdf"` or `"png"`
- **Default:** `"pdf"`
- **PDF:** Lightweight, publication-ready
- **PNG:** Raster, better for quick viewing

### plots_enabled
- **Type:** boolean
- **Default:** `true`
- **Set to `false`:** To skip figure generation and speed up runs (useful for CI/CD)

### log_level
- **Type:** `"DEBUG"` | `"INFO"` | `"WARNING"` | `"ERROR"`
- **Default:** `"INFO"`
- **DEBUG:** Verbose (skip embedder progress bars)
- **INFO:** Standard
- **ERROR:** Only critical issues

---

## Common Setups

### Minimal (local laptop, quick test)
```json
{
  "api_keys": {},
  "models": { "embedder": "all-MiniLM-L6-v2" },
  "experiment_defaults": { "max_examples": 100 },
  "output": { "plots_enabled": false }
}
```

### Full analysis (all categories, all templates, high quality)
```json
{
  "models": { "embedder": "all-mpnet-base-v2" },
  "experiment_defaults": {
    "templates": ["answer_only", "cq_answer", "question_answer", "separate"]
  },
  "output": { "figure_format": "pdf" }
}
```

### LLM probing (need API key)
```json
{
  "api_keys": { "openrouter": "sk-or-v1-..." },
  "models": { "embedder": "all-MiniLM-L6-v2" }
}
```
