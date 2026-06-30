# Experiment Map

Reference for running, extending, and interpreting all experiments in this repo.
Designed so anyone can pick up the work without asking for clarification.

---

## Mental model

Every experiment varies along one or more of these axes:

| Axis | Options |
|---|---|
| **LLM under test** | claude, gpt-4o, llama-3.2-1b, gpt-oss-20b/120b, openrouter (any model) |
| **Bias category** | 11 BBQ categories (Gender, Age, Religion, Nationality, …) |
| **Embedding model** | all-MiniLM-L6-v2 (default), all-mpnet-base-v2 |
| **Metric** | cosine anchor, whitened-cosine (Mahalanobis), BBQ behavioral score |
| **Template** | how text is assembled before embedding (see §Templates) |
| **Mode** | normal (full text), deep (per-sentence), binary (drop unknown) |

---

## Experiment 1, LLM behavioral probing

**Script:** `bias_scorer/run_experiment.py`
**What it does:** Sends BBQ ambiguous multiple-choice questions to a live LLM,
records which answer it picks, then scores each answer with the embedding metric.
Produces two scores per response: embedding direction bias + BBQ behavioral score.

### Run

```bash
# Any model via OpenRouter (free tier, add --request-delay 3)
python -m bias_scorer.run_experiment \
    --llm openrouter \
    --openrouter-model meta-llama/llama-3.2-1b-instruct \
    --data Gender_identity \
    --output results/experiments-llama-3.2-1b \
    --request-delay 3

# Paid API, no delay needed
python -m bias_scorer.run_experiment \
    --llm openrouter \
    --openrouter-model openai/gpt-oss-120b \
    --output results/experiments-gpt-oss-120b \
    --request-delay 0

# Native Anthropic key
python -m bias_scorer.run_experiment --llm claude --data Gender_identity

# Limit to 100 examples for a quick test
python -m bias_scorer.run_experiment --llm openrouter \
    --openrouter-model openai/gpt-4o-mini --max-examples 100
```

### Plugin models (SB, use these slugs on OpenRouter)

| Plugin label | OpenRouter slug | Notes |
|---|---|---|
| DeepSeek-V3 | `deepseek/deepseek-chat` | free tier available |
| Gemini 3 Flash | `google/gemini-flash-1.5` | check current slug |
| Claude Sonnet | `anthropic/claude-sonnet-4-6` | or use `--llm claude` directly |
| GPT-5.3 free | `openai/gpt-4o-mini` | closest free-tier equivalent |

### Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--llm` | `claude` | which LLM driver (`claude`, `claude-opus`, `gpt-4o`, `gpt-4o-mini`, `openrouter`) |
| `--openrouter-model` | `openai/gpt-4o-mini` | model slug when using `--llm openrouter` |
| `--data` | `Gender_identity` | BBQ category filename without `.jsonl` |
| `--max-examples` | all | limit number of BBQ entries (saves cost + time) |
| `--request-delay` | `3.0` | seconds between calls, use `0` for paid APIs |
| `--output` | `results/experiments` | output directory |
| `--input` |, | skip LLM calls; score a pre-collected CSV instead |

### Output files

```
results/experiments-{model}/
  {llm}_{data}_responses.csv   ← raw LLM answers (prompt, llm_answer, stereotyped_group, …)
  {llm}_{data}_bias.csv        ← same + bias_score_norm, bias_corrected, direction,
                                  semaphore, mahal_bias_score, bbq_label
```

### Expanding to other bias categories

Only `Gender_identity` has pre-built anchors for the embedding metric.
For other categories, the behavioral score (`bbq_label`) still works.

```bash
# Run behavioral score only on Nationality
python -m bias_scorer.run_experiment \
    --llm openrouter --openrouter-model deepseek/deepseek-chat \
    --data Nationality --request-delay 3
```

To get the embedding metric on other categories, the anchors must be built first.
See §Building anchors for a new category below.

---

## Experiment 2, Embedding geometry (SAIR-style)

**Script:** `run_bbq_geometry.py`
**What it does:** Tests whether the embedding space separates biased from
anti-biased BBQ answers. Runs clustering stats (centroid similarity, Cohen's d)
and logistic regression classification (5-fold CV, accuracy, AUC).

### Key finding to expect

With `--normalize`: biased and anti-biased overlap almost completely (AUC ≈ 0.5,
centroid sim ≈ 1.0). This is **expected and meaningful**, it shows the embedder
encodes *which group* the answer mentions, not *whether that choice was stereotyped*.
This motivates the anchor direction approach used by `bias_scorer`.

### Run

```bash
# Single category, binary (biased vs anti-biased only), normalized answers
python run_bbq_geometry.py --category Gender_identity --binary --normalize

# Single category, all labels (biased / anti-biased / unknown)
python run_bbq_geometry.py --category Age

# Change embedder
python run_bbq_geometry.py --category Religion --model-name all-mpnet-base-v2

# Quick test with 100 examples
python run_bbq_geometry.py --category Nationality --max-examples 100 --no-plots
```

### Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--category` | `Gender_identity` | BBQ category to analyse |
| `--templates` | all 5 | subset: `answer_only cq_answer question_answer context_only separate` |
| `--binary` | off | drop unknown rows; binary biased/anti-biased classification |
| `--normalize` | off | replace raw answer text with canonical role phrase ("The man", "The woman") |
| `--model-name` | `all-MiniLM-L6-v2` | sentence-transformer model |
| `--max-examples` | all | limit BBQ entries (3× rows) |
| `--no-plots` | off | skip figure generation |
| `--force` | off | ignore embedding cache, recompute |

### Templates

| Template | What is embedded | Signal present? |
|---|---|---|
| `answer_only` | just the answer text | gender/group of answer only |
| `cq_answer` | context + question + answer | full BBQ entry |
| `question_answer` | question + answer | no context |
| `context_only` | context only (same for all 3 labels) | **control**, no answer signal |
| `separate` | concat of [E(cq), E(ans), E(cq)−E(ans), E(cq)⊙E(ans)] | feature-engineered |

`context_only` is the control, classifier accuracy ≈ chance there confirms no data leakage.

### Output

```
results/bbq_geometry/
  {category}_geometry.csv       ← one row per (category, template): accuracy, AUC, Cohen's d, …
  {category}_{template}_pca.png
  {category}_{template}_tsne.png
  {category}_{template}_sim.png
  {category}_summary_heatmap.pdf
```

---

## Experiment 3, Batch geometry (all 11 categories)

**Script:** `run_all_geometry.py`
**What it does:** Runs Experiment 2 for every BBQ category and produces the
summary table used in the paper.

### Run

```bash
# Full run, all 11 categories, raw + normalized, answer_only + cq_answer
python run_all_geometry.py

# Subset of categories
python run_all_geometry.py --categories Gender_identity Age Religion Nationality

# No plots (faster)
python run_all_geometry.py --no-plots

# Different embedder
python run_all_geometry.py --model-name all-mpnet-base-v2
```

### Output

```
results/bbq_geometry/all_categories_geometry.csv
results/figures/geometry_all_answer_only.pdf
results/figures/geometry_all_cq_answer.pdf
```

---

## Experiment 4, PCA / t-SNE embedding space visualisation

**Script:** `run_bbq_pca.py`
**What it does:** Projects BBQ embeddings into 2D and visualises the structure.
Three modes produce very different figures.

### Run

```bash
# Mode A: by gender ROLE (male/female/unknown), shows real separation
# Use this for the paper figure showing the embedder can distinguish genders
python run_bbq_pca.py --category Gender_identity --by-role

# Mode B: binary biased/anti-biased, shows overlap (negative result)
# Use this to demonstrate why naive classification fails
python run_bbq_pca.py --category Gender_identity --binary

# Mode C: all three labels
python run_bbq_pca.py --category Gender_identity

# Non-gender category (disable anchor overlay)
python run_bbq_pca.py --category Age --no-anchor

# Overlay LLM answers on the PCA plot
python run_bbq_pca.py --category Gender_identity --by-role \
    --llm-csv results/experiments-gpt-oss-120b/openrouter_Gender_identity_bias.csv
```

### Mode comparison

| Mode | Colors | Expected result | Use for paper |
|---|---|---|---|
| `--by-role` | male=blue, female=red, unknown=gray | **clear separation** (embedder sees gender) | positive result figure |
| `--binary` | biased=red, anti_biased=blue | **complete overlap** (label is context-dependent) | negative result / motivation |
| default | all three | partial overlap | context |

### Output (per run)

```
results/figures/
  {category}_embedding_space_pca{suffix}.pdf/.png
  {category}_embedding_space_tsne{suffix}.png
  {category}_pc1_histogram{suffix}.png
  {category}_bias_axis_1d{suffix}.png   ← 1D projection onto gender anchor axis
```

---

## Experiment 5, Publication figures

**Script:** `visualize_bias.py`
**What it does:** Reads LLM experiment CSVs and produces the three paper figures.
Requires at least two model result directories.

### Run

```bash
python visualize_bias.py
```

Edit `RESULTS` dict at the top of the file to point to your model directories.

### Output

```
results/figures/
  fig1_semaphore_distribution.pdf/.png  ← stacked bar: low/medium/high per model
  fig2_score_distribution.pdf/.png      ← box plot: normalised bias score per model
  fig3_answer_type.pdf/.png             ← gendered vs neutral answers per model
```

---

## Experiment 6, Metric correlation

**Script:** `plot_metric_correlation.py`
**What it does:** Compares embedding bias score (continuous) against BBQ behavioral
score (did the model choose the stereotyped answer?).
Requires CSVs that have a `bbq_label` column, re-run experiments after May 2026.

### Run

```bash
python plot_metric_correlation.py

# Custom result directories
python plot_metric_correlation.py \
    --results-dir results/experiments-llama \
                  results/experiments-gpt-oss-120b
```

### Output

```
results/figures/
  correlation_violin.pdf/.png    ← embedding score distribution per bbq_label
  correlation_model_scatter.png  ← per-model: mean embedding vs BBQ behavioral score
  correlation_roc.png            ← ROC: does embedding score predict biased choice?
```

---

## Extension recipes

### A. Add a new LLM (OpenRouter)

1. Find the model slug at https://openrouter.ai/models
2. Run:
   ```bash
   python -m bias_scorer.run_experiment \
       --llm openrouter \
       --openrouter-model {slug} \
       --output results/experiments-{short-name} \
       --request-delay 3   # 0 if paid tier
   ```
3. Add the model to `visualize_bias.py` RESULTS dict
4. Re-run `python plot_metric_correlation.py`

### B. Run on all bias categories

```bash
# Behavioral probing on all 11 categories (slow, ~15 min each at free tier)
for cat in Age Disability_status Gender_identity Nationality \
           Physical_appearance Race_ethnicity Religion SES Sexual_orientation; do
    python -m bias_scorer.run_experiment \
        --llm openrouter --openrouter-model deepseek/deepseek-chat \
        --data $cat --request-delay 3 \
        --output results/experiments-deepseek
done
```

Or on Windows PowerShell:
```powershell
$cats = "Age","Disability_status","Gender_identity","Nationality","Religion","SES","Sexual_orientation"
foreach ($cat in $cats) {
    python -m bias_scorer.run_experiment --llm openrouter `
        --openrouter-model deepseek/deepseek-chat `
        --data $cat --request-delay 3 `
        --output "results/experiments-deepseek"
}
```

### C. Build anchors for a new bias category

Currently only `Gender_identity` has pre-built anchors. To add another:

1. Check which answer_info labels are used in the category:
   ```python
   import json
   from pathlib import Path
   for line in open("BBQ_Data/Religion.jsonl"):
       e = json.loads(line)
       if e["context_condition"] == "ambig":
           print({k: e["answer_info"][k][1] for k in ("ans0","ans1","ans2")})
           break
   ```

2. Add a new anchor-building call in `bias_scorer/anchors.py`:
   ```python
   # At the bottom of anchors.py, or in a new file:
   def compute_religion_anchors(embedder, bbq_file=..., cache_file=...):
       # Same as compute_anchors but with a mapping for
       # "muslim", "christian", "jewish", etc. vs a reference group
       ...
   ```

3. For the geometry analysis, this isn't needed, `run_bbq_geometry.py` already
   handles all categories via `_canonicalize`.

### D. Change the embedding model

```bash
# Single experiment
python run_bbq_geometry.py --category Gender_identity --model-name all-mpnet-base-v2

# Full batch
python run_all_geometry.py --model-name all-mpnet-base-v2

# PCA visualisation
python run_bbq_pca.py --category Gender_identity --by-role --model-name all-mpnet-base-v2
```

For the anchor-based scoring with a different model, delete the cache to force rebuild:
```bash
del bias_scorer\cache\gender_anchors.pt   # Windows
```
Then re-run, the cache will rebuild automatically on next `analyze()` call.

### E. Score pre-collected answers (no API needed)

If you have a CSV with columns `prompt` and `llm_answer`:
```bash
python -m bias_scorer.run_experiment --input my_responses.csv --output results/my_exp
```

This skips all LLM calls and just scores the existing answers.

---

## Results file reference

| File | Produced by | Key columns |
|---|---|---|
| `{llm}_{data}_responses.csv` | `run_experiment.py` | context, question, llm_answer, stereotyped_group, stereo_ans, anti_stereo_ans |
| `{llm}_{data}_bias.csv` | `run_experiment.py` | + bias_score_norm, bias_corrected, direction, semaphore, mahal_bias_score, bbq_label |
| `{category}_geometry.csv` | `run_bbq_geometry.py` | category, template, accuracy, roc_auc, cohens_d_pc1, centroid_sim_bia_ant |
| `all_categories_geometry.csv` | `run_all_geometry.py` | same + variant (raw/normalized) |

---

## Shared flags across all scripts

| Flag | Scripts | Meaning |
|---|---|---|
| `--model-name` | geometry, pca | sentence-transformer model name |
| `--max-examples` | experiment, geometry, pca | cap BBQ entries (for quick tests) |
| `--no-plots` | geometry, all_geometry | skip figure generation |
| `--force` | geometry, pca | ignore embedding cache, recompute |
| `--binary` | geometry, pca | only biased vs anti_biased (drop unknown) |
| `--normalize` | geometry | replace raw answer text with canonical role phrase |
| `--by-role` | pca | colour by gender role instead of biased/anti_biased |
| `--category` | geometry, pca | BBQ category (must match filename in `BBQ_Data/`) |

---

## Timing reference (all-MiniLM-L6-v2, CPU)

| Task | Time |
|---|---|
| First anchor build (cache miss) | ~30s |
| Subsequent anchor load (cache hit) | <1s |
| Embed 2400 BBQ examples (all Gender_identity ambiguous) | ~5s |
| Geometry analysis, one category, one template | ~10s |
| Full `run_all_geometry.py` (11 categories × 2 templates, cached) | ~3 min |
| LLM probing, 100 examples at 3s delay | ~5 min |
| PCA + t-SNE on 2400 embeddings | ~20s |
