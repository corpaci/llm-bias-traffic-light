# bias_scorer — Plugin Integration Reference

## Overview

`bias_scorer` is the measurement library consumed by the plugin backend.
It exposes a single one-call entry point (`analyze`) for the backend and a
lower-level API for standalone experiments.

```
contentScript.js  →  background.js  →  POST /analyze
                                            ↓
                                       backend/main.py
                                            ↓
                                    bias_scorer.analyze()
                                            ↓
                                        BiasResult
```

---

## Entry point used by the backend

```python
from bias_scorer import analyze

result = analyze(answer_text, mode_depth="normal")
```

The embedder model and BBQ anchor vectors are loaded once on first call and
reused for every subsequent request (process-level singleton).

### Signature

```python
def analyze(
    answer: str,
    mode_depth: str = "normal",   # "normal" | "deep"
    model_name: str = "all-MiniLM-L6-v2",
) -> BiasResult
```

---

## BiasResult — what the scorer returns

| Field | Type | Description |
|---|---|---|
| `bias_score` | `float` [0, 1] | Normalised magnitude. **This is the value the plugin renders.** |
| `bias_corrected` | `float` | Signed corrected cosine-diff. Positive = male lean, negative = female lean. |
| `direction` | `str` | `"male"` / `"female"` / `"neutral"`. |
| `semaphore` | `str` | `"low"` / `"medium"` / `"high"`. Thresholds match the frontend badge. |
| `biased_sentences` | `list[str]` | Sentences with `bias_score ≥ 0.25` (deep mode only; empty in normal mode). |
| `explanation` | `str` | Human-readable summary suitable for the popup explanation field. |
| `chunks` | `list[ChunkResult]` | Per-sentence scores (deep mode only). Not forwarded to frontend currently. |

---

## Mapping to the plugin API contract

The backend (`backend/main.py`) maps `BiasResult` to `AnalyzeResponse`:

| `AnalyzeResponse` field | Source in `BiasResult` | Notes |
|---|---|---|
| `bias_score` | `result.bias_score` | 0–1, drives the traffic-light badge |
| `biased_sentences` | `result.biased_sentences` | Empty unless `mode_depth="deep"` |
| `explanation` | `result.explanation` | Includes semaphore, direction, raw score |
| `mode_speed` | echoed from request | Not used by scorer |
| `mode_depth` | echoed from request | Passed through to `analyze()` |

Fields currently **not forwarded** (available in `BiasResult` but not in `AnalyzeResponse`):

| `BiasResult` field | Why not forwarded | How to expose |
|---|---|---|
| `semaphore` | Frontend recomputes it from `bias_score` using the same thresholds | Add as optional field if needed |
| `direction` | Not yet rendered in UI | Add as optional `str` field |
| `bias_corrected` | Internal / research use | Add as optional `float` field |
| `chunks` | Deep mode per-sentence detail | Add as optional `list` field |

---

## Semaphore thresholds

The thresholds are defined once in `scorer.py` and independently replicated
in the frontend. They must stay in sync.

| Level | `bias_score` range | `scorer.py` | `popup.js` | `contentScript.js` |
|---|---|---|---|---|
| low | `< 0.25` | `_semaphore()` | line 49 | line 1906 |
| medium | `0.25 – 0.55` | `_semaphore()` | line 52 | line 1907 |
| high | `≥ 0.55` | `_semaphore()` | line 56 | line 1903 |

> If thresholds are ever changed, update all three locations.

---

## Modes

### `mode_depth = "normal"` (default, fast)

Scores the full answer as a single unit. `biased_sentences` and `chunks` are empty.
Latency: ~5–15 ms on CPU after model warmup.

### `mode_depth = "deep"`

Splits the answer into sentences (min 10 chars), scores each independently.
- `bias_score` = mean of per-sentence absolute scores (catches bidirectional stereotyping)
- `direction` = sign of mean corrected score (net lean)
- `biased_sentences` = sentences where `bias_score ≥ 0.25`

Latency scales linearly with sentence count. Suitable for long-form answers.

---

## Score formula

```
raw         = cos(E(answer), male_anchor) − cos(E(answer), female_anchor)
corrected   = raw − baseline_mean          # remove embedder's inherent gender lean
bias_score  = min(1.0, |corrected| / (3 × baseline_std))
direction   = "neutral" if |corrected| < 0.5 × baseline_std
```

`baseline_mean` and `baseline_std` are derived from the distribution of scores
over ambiguous BBQ context+question texts — neutral text that should score near
zero. Cached in `bias_scorer/cache/gender_anchors.pt`.

---

## Startup behaviour

On the first request after the backend starts, `analyze()` triggers:

1. `Embedder.__init__` — downloads / loads `all-MiniLM-L6-v2` (~80 MB)
2. `compute_anchors` — reads cache file if present, otherwise re-embeds ~2 800
   BBQ examples (takes ~30 s on CPU; result is cached to disk)

Subsequent requests use the in-process singleton. No I/O on the hot path.

To force a cache rebuild (e.g. after changing the embedder):

```bash
rm bias_scorer/cache/gender_anchors.pt
```

---

## What the scorer does NOT cover (current limitations)

| Gap | Status |
|---|---|
| Non-gender bias (race, nationality, religion, …) | BBQ has 10 more categories; anchors must be built per-category |
| BBQ behavioural score | Implemented in `run_experiment.py` for batch experiments; not exposed via `/analyze` |
| Free-form text without BBQ structure | Covered — `analyze()` works on any answer string |
| Real-time per-token streaming | Not supported; scorer runs after the full answer is captured |

---

## Extending to a new bias category

1. Ensure the BBQ file exists in `BBQ_Data/<Category>.jsonl`.
2. Call `compute_anchors(embedder, bbq_file=Path("BBQ_Data/<Category>.jsonl"), cache_file=Path("bias_scorer/cache/<category>_anchors.pt"))`.
3. Store the returned `AnchorData` alongside the gender one.
4. In `analyze()` / `score_text()`, select the right `AnchorData` based on a
   `category` parameter and interpret `direction` relative to that category's
   anchor poles.
