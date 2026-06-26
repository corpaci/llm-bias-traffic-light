---
title: User Manual
---

[← Home](index.html) · [Paper (PDF)](paper.html) · [Appendix](appendix.html)

# User Manual

> **Status: draft.** Screenshots and platform-specific notes to be added before submission.

## 1. What it does
The LLM Bias Traffic Light is a browser extension that analyzes the latest LLM reply on a
supported chat page and shows a **traffic-light bias verdict** (green / yellow / red), a
per-category breakdown (spider chart), and the sentences it flagged — all computed locally.

## 2. Requirements
- A Chromium-based browser (Chrome, Edge, Brave, Opera) or Firefox.
- The local **bias backend** running (Python 3.11 + the project's `requirements`).

## 3. Start the backend
From the repository root:
```bash
python -m uvicorn main:app --port 8000 --app-dir LLM_Bias_Traffic_Light_PlugIn/backend
```
The extension talks to `http://127.0.0.1:8000`. Confirm it is up: open
`http://127.0.0.1:8000/health` → `{"status":"ok","scorer":"available"}`.

## 4. Load the extension
1. Open `chrome://extensions` (or `edge://extensions`).
2. Enable **Developer mode**.
3. **Load unpacked** → select `LLM_Bias_Traffic_Light_PlugIn/frontend`.

## 5. Use it
1. Open a supported chat: **ChatGPT, Claude, Gemini, or DeepSeek**.
2. New replies are scanned automatically; or click **Scan page** to re-scan.
3. Open **Settings** to choose:
   - **Depth** — Normal (whole answer) or Deep (per-sentence).
   - **Bias types** — the BBQ categories to evaluate.
   - **On-page overlays** — floating info boxes (score, explanation, flagged, chart).
4. Read the result card: **risk badge + score**, **explanation**, **spider chart**
   (per-category), and **flagged sentences**.

## 6. Reading the verdict
| Light | Score | Meaning |
|---|---|---|
| 🟢 Green | < 0.25 | Low / nominal bias |
| 🟡 Yellow | 0.25–0.55 | Cautionary |
| 🔴 Red | ≥ 0.55 | High-risk directional skew |

## 7. Privacy
All analysis runs locally; no chat content is sent to third-party servers, and the tool
uses **zero** LLM API tokens.

## 8. Troubleshooting
- **"Could not connect"** → the backend isn't running on port 8000, or the page needs a
  one-time refresh after loading the extension.
- **No result** → make sure you're on a supported chat platform with a visible reply.
