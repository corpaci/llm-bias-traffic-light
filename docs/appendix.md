---
title: Appendix
---

[← Home](index.html) · [Paper (PDF)](paper.html) · [User Manual](user-manual.html)

# Appendix

Supplementary material for *LLM Bias Traffic Light*. Sections marked _(draft)_ are pending
final content before submission.

## A. Plugin UI description + screenshots _(draft)_
A walkthrough of the extension UI: the scan controls, the Settings panel (depth, bias
categories, on-page overlays), the result card (traffic-light verdict, score, explanation,
spider chart, flagged sentences), and the floating on-page overlay boxes.

> _TODO: add annotated screenshots of the popup and the on-page overlays._

## B. Example bias results from live conversations _(draft)_
End-to-end examples on ChatGPT / Claude / Gemini / DeepSeek showing a biased reply and the
tool's verdict.

> _TODO: add annotated conversation screenshots (green / yellow / red cases)._

## C. Text extraction in the plugin _(draft)_
How the content script extracts the prompt / context / answer from each chat platform's DOM,
and how it segments text for per-sentence (Deep mode) analysis.

> _TODO (SB): describe the per-platform extraction + segmentation logic._

## D. Timing analysis _(preliminary)_
End-to-end `/analyze` latency on a local dev machine (single-process FastAPI, CPU,
`all-MiniLM-L6-v2`). Numbers are indicative, not benchmarked across hardware.

| Phase | Latency | Notes |
|---|---|---|
| First request (cold) | ~20–25 s | one-time model load + anchor (cache) load |
| Warm — Normal mode | sub-second | whole-answer single embedding pass |
| Warm — Deep mode (≈3 sentences) | ~0.9–2.5 s | per-sentence embedding + per-category projection |
| DOM text extraction (long pages) | can exceed 5 s | dominated by page DOM size, not scoring |

> _TODO (LC): replace with a measured table (mean ± std over N runs, per mode and text length)._
> Per the timing note, extended timing is reported here as an external appendix page.

## E. Extended evaluation results

Complete sweep over **4 models × 11 BBQ categories** (44/44 cells), 10% stratified subsample,
10 repeats/prompt, temperature 0.7 (run `all_run`: 186,296 OpenRouter calls, ≈ \$67).
Failure rate 0.02% (empty/refused responses), no scoring errors.

### Per-model summary (mean across all 11 categories)

| Model | Embedding bias | Mahalanobis | BBQ ambig. bias | Stereotype rate | Abstention (unknown) | Disambig. accuracy |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V3 | 0.430 | 0.513 | **0.154** | 0.326 | 0.728 | **0.904** |
| GPT-5.3 | 0.392 | 0.478 | 0.051 | 0.256 | 0.905 | 0.888 |
| Claude Sonnet 4.6 | 0.375 | 0.460 | 0.034 | 0.238 | 0.944 | 0.874 |
| Gemini 3 Flash | 0.342 | 0.432 | 0.025 | 0.191 | **0.985** | 0.766 |

<small>BBQ ambiguous bias is the (1−accuracy)-scaled conditional stereotype lean
(Parrish et al.); 5 full-abstention cells yield an undefined conditional score and are
excluded from that column.</small>

### Headline: does the embedding score track behavioural bias?
Across the 39 cells with a defined behavioural score, the **local embedding bias correlates
with the behavioural BBQ bias at _r_ ≈ 0.61** (Mahalanobis variant _r_ ≈ 0.55). This is the
core evidence that the zero-token, embedding-based traffic light is a usable proxy/oracle for
behavioural social bias — strong and positive, though not a perfect substitute.

**Behavioural reading:** models that abstain more in ambiguous contexts (Gemini 98.5%,
Claude 94.4%) show the least stereotype lean; DeepSeek commits far more often (abstains
72.8%) and shows the highest behavioural bias. The abstention↔bias relationship is consistent
across categories.

**Per-category note:** bias is not uniform within a model — e.g., GPT-5's behavioural bias is
concentrated in **Age** (ambig. bias 0.308; it abstains only 60.5% there) and SES, while it is
near-zero on the other categories.

> Full per-(model × category) numbers: `results/sweep/all_run/SUMMARY.csv`.

## F. BBQ category distribution
The tool's anchors and baseline are derived from all 11 BBQ demographic subsets.

| Demographic domain | No. examples |
|---|---:|
| Age | 3,680 |
| Disability status | 1,556 |
| Gender identity | 5,672 |
| Nationality | 3,080 |
| Physical appearance | 1,576 |
| Race / ethnicity | 6,880 |
| Religion | 1,200 |
| Socioeconomic status | 6,864 |
| Sexual orientation | 864 |
| Race × Gender | 15,960 |
| Race × SES | 11,160 |
| **Total** | **58,492** |
