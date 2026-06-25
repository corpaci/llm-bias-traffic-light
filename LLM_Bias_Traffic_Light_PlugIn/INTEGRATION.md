# Plugin Integration Guide

## Purpose

This document explains how the browser extension frontend connects to the backend, including the endpoint used and the data format expected in requests and responses.

## Backend endpoint

The extension sends analysis requests to a single backend endpoint:

- `POST http://localhost:8000/analyze`

This endpoint is called from `frontend/background.js`.

Note: current dev default port is **8000** (see `frontend/background.js`).

## When the endpoint is used

The endpoint is used in two main scenarios:

1. Manual scan from the popup UI
   - `popup.js` collects page text and sends `ANALYZE_TEXT` to `background.js`
   - `background.js` forwards the payload to `/analyze`

2. Automatic LLM-turn capture
   - `contentScript.js` tries to extract the latest prompt/answer pair from chat pages
   - it sends `NEW_LLM_TURN` to `background.js`
   - `background.js` forwards the same payload to `/analyze`

## Request payload shape

The backend expects JSON with the following fields:

- `text` (optional string)
  - the raw page text or visible text captured from the page
- `question` (optional string)
  - explicit user question text when available
- `prompt` (optional string)
  - prompt text or user input text when available
- `context` (optional string)
  - any supporting context the page or UI provides
- `answer` (optional string)
  - the model/assistant response text when available
- `mode_speed` (string)
  - expected values: `fast` or `slow`
- `mode_depth` (string)
  - expected values: `normal` or `deep`

Example request body:

```json
{
  "text": "Visible page text goes here...",
  "prompt": "What is the safety issue?",
  "context": "Page is about traffic policy.",
  "answer": "The response mentions equity concerns.",
  "mode_speed": "fast",
  "mode_depth": "normal"
}
```

## Response payload shape

The backend returns JSON with these core fields:

- `bias_score` (float)
  - a numeric bias/risk score
- `biased_sentences` (array of strings)
  - text fragments identified as biased or problematic
- `explanation` (string)
  - a human-readable explanation of the result

Optional fields that the extension can support:

- `section_scores` (object)
  - map of section name to numeric score
- `similarity_prompt_answer` (float)
- `similarity_context_answer` (float)
- `similarity_prompt_context` (float)
- `similarity_question_answer` (float)
- `similarity_question_context` (float)
- `mode_speed` (string)
- `mode_depth` (string)

Example response body:

```json
{
  "bias_score": 0.38,
  "biased_sentences": [
    "This neighborhood is dangerous.",
    "Only people from X are affected."
  ],
  "explanation": "Bias detected due to broad generalizations and negative framing.",
  "section_scores": {
    "safety": 0.5,
    "equity": 0.27
  },
  "similarity_prompt_answer": 0.12,
  "mode_speed": "fast",
  "mode_depth": "normal"
}
```

## Frontend usage

- `popup.js` renders:
  - numeric bias score
  - qualitative badge (Low / Medium / High)
  - explanation text
  - list of biased sentences
- `contentScript.js` highlights sentences returned in `biased_sentences`
- `background.js` stores the latest LLM turn and broadcasts results for page overlays

## Integration notes

- The backend now bridges to the external `bias_scorer` package from the sibling project
  `llm-bias-traffic-light-luiza-wip` without modifying that codebase.
- The frontend expects the API contract to remain stable.
- For full integration, the backend should implement actual bias detection logic and return the response fields shown above.

## Recommended contract for metrics integration

Use the above request/response shapes as the contract between the plugin and the backend.

If new metrics are added later, they should be included as optional response fields so the frontend can continue working with minimal changes.
