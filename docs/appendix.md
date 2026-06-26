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

## E. Extended evaluation results _(pending sweep)_
Full per-(model × category) results from the OpenRouter sweep — embedding bias, Mahalanobis
bias, and the BBQ behavioural metrics (ambiguous/disambiguated), plus the run's cost and
failure breakdown.

> _Pending the full model sweep (4 models × 11 categories × 10% × 10 repeats). Table to be
> generated from `results/sweep/<run>/SUMMARY.csv`._

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
