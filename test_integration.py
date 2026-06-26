"""
Integration test for the plugin backend <-> bias_scorer pipeline.

Requires the backend to be running:
    cd LLM_Bias_Traffic_Light_PlugIn/backend
    python -m uvicorn main:app --port 8000

Run:
    python test_integration.py
    python test_integration.py --url http://127.0.0.1:8000   # custom URL
"""

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "http://127.0.0.1:8000"
for arg in sys.argv[1:]:
    if arg.startswith("--url="):
        BASE_URL = arg.split("=", 1)[1]
    elif arg == "--url":
        idx = sys.argv.index("--url")
        BASE_URL = sys.argv[idx + 1]

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, timeout: int = 60) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE_URL + path, timeout=10) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    ok: bool
    note: str = ""


_results: list[Result] = []


def check(name: str, condition: bool, note: str = ""):
    r = Result(name, condition, note)
    _results.append(r)
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({note})" if note else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health():
    print("\n=== Health ===")
    resp = _get("/health")
    check("status ok", resp.get("status") == "ok", str(resp))
    check("scorer field present", "scorer" in resp, str(resp))
    check("scorer available", resp.get("scorer") == "available",
          f"scorer={resp.get('scorer')} — install sentence-transformers + torch in venv")


def test_no_text():
    print("\n=== No text ===")
    resp = _post("/analyze", {"mode_speed": "fast", "mode_depth": "normal"})
    check("bias_score is 0", resp["bias_score"] == 0.0, str(resp["bias_score"]))
    check("biased_sentences empty", resp["biased_sentences"] == [], str(resp["biased_sentences"]))
    check("explanation present", bool(resp.get("explanation")))
    check("mode echoed", resp.get("mode_speed") == "fast" and resp.get("mode_depth") == "normal")


def test_neutral_text():
    print("\n=== Neutral text ===")
    resp = _post("/analyze", {
        "answer": "The weather today is sunny with a light breeze.",
        "mode_speed": "fast",
        "mode_depth": "normal",
    })
    score = resp["bias_score"]
    check("bias_score is float", isinstance(score, float), str(score))
    check("bias_score in [0,1]", 0.0 <= score <= 1.0, str(score))
    check("neutral scores low (<0.25)", score < 0.25,
          f"score={score:.4f} — a neutral sentence should be low")
    check("biased_sentences empty in normal mode", resp["biased_sentences"] == [])
    check("explanation present", bool(resp.get("explanation")))


def test_biased_text_answer_field():
    print("\n=== Biased text (answer field) ===")
    resp = _post("/analyze", {
        "answer": "Women are naturally better suited to nursing and caregiving roles than men.",
        "mode_speed": "fast",
        "mode_depth": "normal",
    })
    score = resp["bias_score"]
    check("bias_score > 0", score > 0.0, str(score))
    check("bias_score >= medium threshold (0.25)", score >= 0.25,
          f"score={score:.4f}")
    check("explanation mentions direction or semaphore",
          any(w in resp.get("explanation", "").lower()
              for w in ("female", "male", "high", "medium", "low")),
          repr(resp.get("explanation")))


def test_biased_text_text_field():
    print("\n=== Biased text (text field fallback) ===")
    resp = _post("/analyze", {
        "text": "Men are typically more aggressive and dominant in the workplace.",
        "mode_speed": "fast",
        "mode_depth": "normal",
    })
    score = resp["bias_score"]
    check("text field used as fallback", score > 0.0,
          f"score={score:.4f} — text field should be scored when answer is absent")


def test_answer_takes_precedence():
    print("\n=== answer overrides text ===")
    resp = _post("/analyze", {
        "text": "Women are naturally better suited to nursing roles.",
        "answer": "The weather today is sunny with a light breeze.",
        "mode_speed": "fast",
        "mode_depth": "normal",
    })
    score = resp["bias_score"]
    check("answer field takes precedence over text (score should be low)",
          score < 0.25, f"score={score:.4f}")


def test_deep_mode():
    print("\n=== Deep mode ===")
    resp = _post("/analyze", {
        "answer": (
            "Women are naturally better at caregiving. "
            "The engineer solved the problem efficiently. "
            "Men are more suited for leadership roles."
        ),
        "mode_speed": "fast",
        "mode_depth": "deep",
    })
    score = resp["bias_score"]
    sentences = resp["biased_sentences"]
    check("bias_score > 0 in deep mode", score > 0.0, str(score))
    check("biased_sentences non-empty in deep mode", len(sentences) > 0,
          f"got {len(sentences)} sentences")
    check("biased_sentences are strings", all(isinstance(s, str) for s in sentences))
    check("mode_depth echoed as deep", resp.get("mode_depth") == "deep")


def test_mode_echo():
    print("\n=== Mode echo ===")
    resp = _post("/analyze", {
        "answer": "He is a great leader.",
        "mode_speed": "slow",
        "mode_depth": "deep",
    })
    check("mode_speed echoed", resp.get("mode_speed") == "slow", str(resp.get("mode_speed")))
    check("mode_depth echoed", resp.get("mode_depth") == "deep", str(resp.get("mode_depth")))


def test_response_shape():
    print("\n=== Response shape ===")
    resp = _post("/analyze", {"answer": "She leads the team.", "mode_depth": "normal"})
    required = ["bias_score", "biased_sentences", "explanation", "mode_speed", "mode_depth"]
    for field in required:
        check(f"field '{field}' present", field in resp)
    check("bias_score is float", isinstance(resp["bias_score"], float))
    check("biased_sentences is list", isinstance(resp["biased_sentences"], list))
    check("explanation is str", isinstance(resp["explanation"], str))


def test_sentence_scores():
    print("\n=== Sentence-level scores (deep) ===")
    resp = _post("/analyze", {
        "answer": "Women are naturally better at caregiving. "
                  "The engineer solved the problem efficiently.",
        "mode_depth": "deep",
        "bias_types": ["gender"],
    })
    ss = resp.get("sentence_scores")
    check("sentence_scores present in deep mode", isinstance(ss, list) and len(ss) > 0, str(ss))
    if ss:
        s0 = ss[0]
        for f in ("text", "score", "level", "direction", "category"):
            check(f"sentence_scores[0] has '{f}'", f in s0, str(s0))
        check("score in [0,1]", 0.0 <= s0.get("score", -1) <= 1.0, str(s0.get("score")))
        check("level is low/medium/high", s0.get("level") in ("low", "medium", "high"),
              str(s0.get("level")))
        check("sorted descending by score",
              all(ss[i]["score"] >= ss[i + 1]["score"] for i in range(len(ss) - 1)))


def test_sentence_scores_normal_mode():
    print("\n=== Sentence scores absent in normal mode ===")
    resp = _post("/analyze", {
        "answer": "Women are naturally better at caregiving.",
        "mode_depth": "normal", "bias_types": ["gender"],
    })
    check("sentence_scores empty/none in normal mode", not resp.get("sentence_scores"),
          str(resp.get("sentence_scores")))


def test_hierarchy():
    print("\n=== Hierarchical roll-up (deep) ===")
    answer = ("Women are naturally better at caregiving. Men should lead.\n\n"
              "The weather is pleasant today.")
    resp = _post("/analyze", {"answer": answer, "mode_depth": "deep", "bias_types": ["gender"]})
    h = resp.get("hierarchy")
    check("hierarchy present in deep mode", isinstance(h, dict), str(h))
    if isinstance(h, dict):
        for f in ("score", "level", "weighting", "paragraphs"):
            check(f"hierarchy has '{f}'", f in h, str(list(h.keys())))
        check("default weighting is length", h.get("weighting") == "length", str(h.get("weighting")))
        check("response score in [0,1]", 0.0 <= h.get("score", -1) <= 1.0, str(h.get("score")))
        paras = h.get("paragraphs", [])
        check("hierarchy has >= 2 paragraphs", len(paras) >= 2, f"got {len(paras)}")
        if paras:
            check("paragraph has sentences",
                  isinstance(paras[0].get("sentences"), list) and len(paras[0]["sentences"]) > 0)


def test_hierarchy_weighting_max():
    print("\n=== Hierarchy weighting=max echoed ===")
    resp = _post("/analyze", {
        "answer": "Women are naturally better at caregiving. The sky is blue.",
        "mode_depth": "deep", "bias_types": ["gender"], "hierarchy_weighting": "max",
    })
    h = resp.get("hierarchy")
    check("weighting echoed as max", isinstance(h, dict) and h.get("weighting") == "max",
          str(h.get("weighting") if isinstance(h, dict) else h))


def test_embedder_comparison():
    print("\n=== Cross-embedder comparison (downloads 2nd model; slow) ===")
    resp = _post("/analyze", {
        "answer": "Women are naturally better at caregiving.",
        "mode_depth": "normal", "bias_types": ["gender"],
        "compare_embedders": True,
    }, timeout=900)
    cmp = resp.get("embedder_comparison")
    check("embedder_comparison present", isinstance(cmp, dict), str(cmp))
    if isinstance(cmp, dict):
        check(">= 2 models compared", len(cmp.get("models", [])) >= 2, str(cmp.get("models")))
        check("overall_scores keyed by model",
              set(cmp.get("overall_scores", {})) == set(cmp.get("models", [])))
        check("has divergence/level_agreement/mean_divergence",
              all(k in cmp for k in ("divergence", "level_agreement", "mean_divergence")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Testing backend at {BASE_URL}")
    print("Waiting for server...", end=" ", flush=True)
    for _ in range(10):
        try:
            _get("/health")
            print("up.")
            break
        except Exception:
            time.sleep(1)
    else:
        print("unreachable. Is the backend running?")
        sys.exit(1)

    test_health()
    test_no_text()
    test_neutral_text()
    test_biased_text_answer_field()
    test_biased_text_text_field()
    test_answer_takes_precedence()
    test_deep_mode()
    test_mode_echo()
    test_response_shape()
    test_sentence_scores()
    test_sentence_scores_normal_mode()
    test_hierarchy()
    test_hierarchy_weighting_max()
    if "--compare" in sys.argv:
        test_embedder_comparison()
    else:
        print("\n(skipping cross-embedder comparison; pass --compare to run it — downloads a 2nd model)")

    total = len(_results)
    passed = sum(1 for r in _results if r.ok)
    failed = [r for r in _results if not r.ok]

    print(f"\n{'='*40}")
    print(f"  {passed}/{total} passed")
    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  - {r.name}: {r.note}")
    else:
        print("  All checks passed.")
    print('='*40)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
