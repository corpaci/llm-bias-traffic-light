# Extending the Framework

Guide for adding custom analyses, metrics, and experiments.

## Adding a New Metric to Geometry Analysis

Suppose you want to add **silhouette score** alongside Cohen's d.

### Step 1: Locate the metrics computation

File: `run_bbq_geometry.py`, search for `def run_category()` around line ~200.

This function computes:
- `accuracy` (logistic regression)
- `roc_auc`
- `cohens_d_pc1`
- `centroid_sim_bia_ant`

### Step 2: Add your metric

Example: Add silhouette coefficient for clustering quality.

```python
from sklearn.metrics import silhouette_score

def run_category(category, templates, max_examples=None, embedder=None):
    # ... existing code ...

    for template in templates:
        # ... compute embeddings, labels ...

        result_row = {
            "category": category,
            "template": template,
            "accuracy": acc,
            "roc_auc": auc,
            "cohens_d_pc1": cohens_d,
            "centroid_sim_bia_ant": centroid_sim,
        }

        # NEW: Add silhouette score
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(embeddings, labels)
            result_row["silhouette_score"] = sil_score

        results.append(result_row)

    return results
```

### Step 3: Verify in SUMMARY.csv

After running:
```bash
python run_all_geometry.py --force
```

Check `results/bbq_geometry/SUMMARY.csv`, new column `silhouette_score` will appear.

---

## Creating a Custom Experiment Script

### Template: Run analysis on a custom dataset

```python
"""
analyze_custom_bias.py, Custom bias analysis on external data.

Usage:
    python analyze_custom_bias.py --dataset my_data.csv --embedder all-MiniLM-L6-v2
"""

import argparse
import pandas as pd
from pathlib import Path

# Import from bias_scorer library
from bias_scorer.embedder import Embedder
from bias_scorer.scorer import BiasScorer

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "custom_analyses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_custom_dataset(csv_path, embedder_name="all-MiniLM-L6-v2"):
    """
    Load CSV with columns: [text, label]
    Score each text for bias, save results.
    """
    df = pd.read_csv(csv_path)
    embedder = Embedder(model_name=embedder_name)

    scorer = BiasScorer(embedder)

    results = []
    for idx, row in df.iterrows():
        result = scorer.analyze(row["text"])
        results.append({
            "text": row["text"],
            "label": row["label"],
            "bias_score": result.bias_score,
            "direction": result.direction,
            "semaphore": result.semaphore,
        })

    results_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "custom_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--embedder", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    analyze_custom_dataset(args.dataset, args.embedder)
```

### Usage
```bash
python analyze_custom_bias.py --dataset my_data.csv
```

---

## Hooking into the Scorer API

The core API is simple:

```python
from bias_scorer import analyze

# Single call: takes text, returns bias metrics
result = analyze("The CEO was a man", mode_depth="normal")

print(result.bias_score)        # 0–1
print(result.direction)         # "male", "female", or "neutral"
print(result.semaphore)         # "low", "medium", "high"
print(result.explanation)       # Human-readable summary
```

### For batch processing

```python
from bias_scorer import analyze

texts = ["The engineer was a woman.", "The nurse was a man."]
results = [analyze(text) for text in texts]

# Build DataFrame
import pandas as pd
df = pd.DataFrame([
    {"text": t, "bias_score": r.bias_score, "direction": r.direction}
    for t, r in zip(texts, results)
])
```

---

## Contributing Back

Found a useful metric or analysis? Consider:

1. **Add docs**, Explain what it measures and why
2. **Add tests**, Verify it works on known examples (see `tests/`)
3. **Submit PR**, Include example usage in commit message

See `.git` history for examples of previous contributions.

---

## Common Patterns

### Pattern 1: Run all categories with custom metric

```python
from pathlib import Path
import pandas as pd
from run_bbq_geometry import BBQ_DIR, run_category

categories = [p.stem for p in BBQ_DIR.glob("*.jsonl")]

all_results = []
for cat in categories:
    print(f"Processing {cat}...")
    results = run_category(cat, templates=["answer_only", "cq_answer"])
    all_results.extend(results)

df = pd.DataFrame(all_results)
df.to_csv("results/my_analysis.csv", index=False)
```

### Pattern 2: Compare two embedders

```python
from bias_scorer.embedder import Embedder

embedder_1 = Embedder(model_name="all-MiniLM-L6-v2")
embedder_2 = Embedder(model_name="all-mpnet-base-v2")

text = "The doctor was a woman."
emb_1 = embedder_1.encode(text)
emb_2 = embedder_2.encode(text)

print(f"Embedder 1 shape: {emb_1.shape}")
print(f"Embedder 2 shape: {emb_2.shape}")
# Compare embeddings, distributions, etc.
```

### Pattern 3: Filter BBQ data before analysis

```python
from bias_scorer.DataLoader.bbq_loader import BBQDataLoader

loader = BBQDataLoader("BBQ_Data/Gender_identity.jsonl")

# Load only ambiguous questions
subset = loader.load_subset(
    context_condition="ambig",
    question_polarity="neg",
    max_examples=100
)

# Use subset in your analysis
for example in subset:
    print(example["question"])
```

---

## Debugging

### Print embedder dimensions
```python
from bias_scorer.embedder import Embedder
emb = Embedder()
text = "The manager was a woman."
vec = emb.encode(text)
print(vec.shape)  # Should be (384,) for all-MiniLM-L6-v2
```

### Inspect a single BBQ example
```python
from bias_scorer.DataLoader.bbq_loader import load_bbq_data
data = load_bbq_data("BBQ_Data/Gender_identity.jsonl")
print(data[0])  # Inspect structure: keys, answer_info, context, etc.
```

### Mock Scorer without API keys
```python
from bias_scorer.scorer import BiasScorer
from bias_scorer.embedder import Embedder

scorer = BiasScorer(Embedder())
# Works offline (no API needed for geometry/visualization, only for LLM probing)
result = scorer.analyze("Text here")
```
