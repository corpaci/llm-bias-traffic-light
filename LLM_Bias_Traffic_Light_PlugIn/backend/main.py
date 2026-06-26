from pathlib import Path
from typing import List, Dict, Optional
import os
import re
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    # Optional free-form text (e.g. whole page)
    text: Optional[str] = None
    # Optional explicit question / context / answer triplet from the UI
    question: Optional[str] = None
    prompt: Optional[str] = None
    context: Optional[str] = None
    answer: Optional[str] = None
    # Mode knobs controlled by the extension UI
    mode_speed: str = "fast"   # "fast" | "slow"
    mode_depth: str = "normal"  # "normal" | "deep"
    # Bias categories to focus on (BBQ taxonomy names)
    bias_types: Optional[List[str]] = None
    # How sentence scores roll up to paragraph/response in the hierarchy
    # (deep mode only): "length" (char-length weighted), "mean", or "max".
    hierarchy_weighting: str = "length"
    # Score with several sentence-transformers embedders and compare them
    # (robustness signal). Defaults below if enabled without explicit models.
    compare_embedders: bool = False
    embedder_models: Optional[List[str]] = None


class SentenceScore(BaseModel):
    text: str
    score: float        # 0–1 normalized bias level for this sentence
    level: str          # "low" | "medium" | "high"
    direction: str      # lean, e.g. "male"/"female"/"stereotyped"/"neutral"
    category: str       # BBQ category responsible for this sentence's score


class HierarchyParagraph(BaseModel):
    text: str
    score: float                    # rolled up from its sentences
    level: str
    sentences: List[SentenceScore]


class BiasHierarchy(BaseModel):
    """Response -> paragraphs -> sentences roll-up (deep mode)."""
    score: float                    # rolled up from paragraphs
    level: str
    weighting: str                  # "length" | "mean" | "max"
    paragraphs: List[HierarchyParagraph]


class EmbedderComparison(BaseModel):
    """Same text scored by several embedders, for a robustness/agreement read."""
    models: List[str]
    overall_scores: Dict[str, float]               # model -> overall bias score
    category_scores: Dict[str, Dict[str, float]]   # model -> {category -> score}
    divergence: Dict[str, float]                   # category -> (max - min) across models
    level_agreement: Dict[str, bool]               # category -> do all models share a level?
    mean_divergence: float


class AnalyzeResponse(BaseModel):
    bias_score: float
    biased_sentences: List[str]
    explanation: str
    # Per-sentence bias breakdown (populated in deep / full-analysis mode)
    sentence_scores: Optional[List[SentenceScore]] = None
    # Hierarchical roll-up: response -> paragraphs -> sentences (deep mode)
    hierarchy: Optional[BiasHierarchy] = None
    # Cross-embedder comparison (when compare_embedders=true)
    embedder_comparison: Optional[EmbedderComparison] = None
    # Optional richer metadata for slow / deep modes
    section_scores: Optional[Dict[str, float]] = None
    similarity_prompt_answer: Optional[float] = None
    similarity_context_answer: Optional[float] = None
    similarity_prompt_context: Optional[float] = None
    similarity_question_answer: Optional[float] = None
    similarity_question_context: Optional[float] = None
    mode_speed: Optional[str] = None
    mode_depth: Optional[str] = None
    bias_types: Optional[List[str]] = None
    # Per-bias breakdown for spider/radar plot in the extension
    bias_type_scores: Optional[Dict[str, float]] = None
    # Per-bias lean (e.g. "male"/"female", "stereotyped"/"anti-stereotype", "neutral")
    bias_type_directions: Optional[Dict[str, str]] = None
    # Per-bias whitened-cosine (Mahalanobis) magnitude, 0–1; 0.0 when unavailable
    mahal_type_scores: Optional[Dict[str, float]] = None


app = FastAPI(title="Extension Capture API")


_BIAS_TYPE_TO_CATEGORY = {
    "gender": "Gender_identity",
    "nationality": "Nationality",
    "religion": "Religion",
    "age": "Age",
    "disability": "Disability_status",
    "sexual_orientation": "Sexual_orientation",
    "race": "Race_ethnicity",
    "race_ethnicity": "Race_ethnicity",
    "race_x_gender": "Race_x_gender",
    "race_x_ses": "Race_x_SES",
    "ses": "SES",
    "physical_appearance": "Physical_appearance",
}

# Polished bias_scorer tree (single source of truth). It contains both the
# bias_scorer package and the BBQ_Data the anchors are built from.
#   backend/main.py -> backend -> LLM_Bias_Traffic_Light_PlugIn -> llm-bias-traffic-light
BIAS_SCORER_ROOT = Path(__file__).resolve().parents[2]
BBQ_DATA_DIR = BIAS_SCORER_ROOT / "BBQ_Data"
CACHE_DIR = BIAS_SCORER_ROOT / "bias_scorer" / "cache"
if str(BIAS_SCORER_ROOT) not in sys.path:
    sys.path.insert(0, str(BIAS_SCORER_ROOT))

try:
    from bias_scorer import analyze as analyze_bias_text
except Exception as exc:
    analyze_bias_text = None
    _BIAS_IMPORT_ERROR = str(exc)
else:
    _BIAS_IMPORT_ERROR = None

def _normalize_bias_types(bias_types: Optional[List[str]]) -> List[str]:
    cleaned: list[str] = []
    for bias_type in bias_types or []:
        if not isinstance(bias_type, str):
            continue
        value = bias_type.strip().lower()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned or ["gender"]


def _extract_answer_text(request: AnalyzeRequest) -> str:
    text = (
        request.answer
        or request.text
        or request.context
        or request.prompt
        or request.question
        or ""
    )
    return text.strip()


def _build_bias_type_scores(scores_by_type: Dict[str, float], bias_types: Optional[List[str]]) -> Dict[str, float]:
    selected_bias_types = _normalize_bias_types(bias_types)
    return {
        bias_type: float(scores_by_type.get(_BIAS_TYPE_TO_CATEGORY.get(bias_type, bias_type), 0.0))
        for bias_type in selected_bias_types
    }


def _build_bias_type_directions(directions_by_category: Dict[str, str], bias_types: Optional[List[str]]) -> Dict[str, str]:
    selected_bias_types = _normalize_bias_types(bias_types)
    return {
        bias_type: str(directions_by_category.get(_BIAS_TYPE_TO_CATEGORY.get(bias_type, bias_type), "neutral"))
        for bias_type in selected_bias_types
    }


# ---------------------------------------------------------------------------
# Sentence-transformers path — delegates entirely to bias_scorer.analyze().
# ---------------------------------------------------------------------------

def _score_requested_categories(
    answer_text: str, bias_types: Optional[List[str]], mode_depth: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[float, List[str], str, Dict[str, float], Dict[str, str], Dict[str, float], List[SentenceScore]]:
    if analyze_bias_text is None:
        raise RuntimeError(
            "bias_scorer unavailable. "
            f"Import error: {_BIAS_IMPORT_ERROR}"
        )

    category_scores: Dict[str, float] = {}
    category_directions: Dict[str, str] = {}
    category_mahal: Dict[str, float] = {}
    explanations: list[str] = []
    all_sentences: list[str] = []
    # Per-sentence breakdown, keyed by sentence text. When several categories
    # are requested, each sentence keeps the highest-scoring category's verdict.
    sentence_agg: dict[str, SentenceScore] = {}

    for bias_type in _normalize_bias_types(bias_types):
        category = _BIAS_TYPE_TO_CATEGORY.get(bias_type, bias_type)
        if category in category_scores:
            continue
        result = analyze_bias_text(answer_text, mode_depth=mode_depth, category=category, model_name=model_name)
        category_scores[category] = float(result.bias_score)
        category_directions[category] = result.direction
        category_mahal[category] = float(result.mahal_bias_score)
        explanations.append(f"{bias_type}: {result.bias_score:.3f} ({result.direction})")
        all_sentences.extend(result.biased_sentences)
        # chunks are only populated in deep mode; in normal mode this is empty.
        for chunk in result.chunks:
            score = float(chunk.bias_score)
            existing = sentence_agg.get(chunk.text)
            if existing is None or score > existing.score:
                sentence_agg[chunk.text] = SentenceScore(
                    text=chunk.text,
                    score=round(score, 4),
                    level=_semaphore(score),
                    direction=chunk.direction,
                    category=category,
                )

    overall_score = max(category_scores.values()) if category_scores else 0.0
    sentence_scores = sorted(
        sentence_agg.values(), key=lambda s: s.score, reverse=True
    )
    explanation = (
        "Category scores -> " + ", ".join(explanations)
        if explanations else "No supported bias categories selected."
    )
    return (overall_score, all_sentences, explanation, category_scores,
            category_directions, category_mahal, sentence_scores)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _semaphore(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _aggregate(scores: List[float], weights: List[float], weighting: str) -> float:
    """Roll a set of child scores up to a parent score."""
    if not scores:
        return 0.0
    if weighting == "max":
        return max(scores)
    if weighting == "mean":
        return sum(scores) / len(scores)
    # "length": weight by char length; fall back to uniform mean if weights are 0
    total_w = sum(weights)
    if total_w <= 0:
        return sum(scores) / len(scores)
    return sum(s * w for s, w in zip(scores, weights)) / total_w


def _build_hierarchy(
    answer_text: str,
    sentence_scores: List[SentenceScore],
    weighting: str = "length",
) -> Optional[BiasHierarchy]:
    """Build a response -> paragraphs -> sentences roll-up from per-sentence
    scores. Sentences never cross a blank-line break, so each maps to exactly
    one paragraph (matched by substring)."""
    if not sentence_scores:
        return None
    weighting = weighting if weighting in {"length", "mean", "max"} else "length"

    paragraphs_text = [p.strip() for p in re.split(r"\n\s*\n", answer_text) if p.strip()]
    if not paragraphs_text:
        paragraphs_text = [answer_text.strip()]

    remaining = list(sentence_scores)
    paragraphs: List[HierarchyParagraph] = []

    def _make_paragraph(text: str, members: List[SentenceScore]) -> HierarchyParagraph:
        p_score = round(
            _aggregate([s.score for s in members],
                       [float(len(s.text)) for s in members], weighting),
            4,
        )
        return HierarchyParagraph(
            text=text, score=p_score, level=_semaphore(p_score), sentences=members
        )

    for ptext in paragraphs_text:
        members = [s for s in remaining if s.text and s.text in ptext]
        for s in members:
            remaining.remove(s)  # assign once, avoid double counting duplicates
        if members:
            paragraphs.append(_make_paragraph(ptext, members))

    # Any unmatched sentences (whitespace/normalization drift) -> trailing group
    if remaining:
        paragraphs.append(_make_paragraph(" ".join(s.text for s in remaining), list(remaining)))

    if not paragraphs:
        return None

    resp_score = round(
        _aggregate([p.score for p in paragraphs],
                   [float(len(p.text)) for p in paragraphs], weighting),
        4,
    )
    return BiasHierarchy(
        score=resp_score, level=_semaphore(resp_score),
        weighting=weighting, paragraphs=paragraphs,
    )


def _compare_embedders(
    answer_text: str, bias_types: Optional[List[str]], mode_depth: str, models: List[str]
) -> EmbedderComparison:
    """Score the same text with each embedder and report per-category divergence.

    First use of a non-default model builds (and caches) its anchors, so the
    initial call is slower; subsequent calls reuse the cache.
    """
    overall_scores: Dict[str, float] = {}
    category_scores: Dict[str, Dict[str, float]] = {}
    for model_name in models:
        score, _sent, _expl, cat_scores, _dirs, _mahal, _ss = _score_requested_categories(
            answer_text, bias_types, mode_depth, model_name=model_name
        )
        overall_scores[model_name] = round(float(score), 4)
        category_scores[model_name] = {c: round(float(v), 4) for c, v in cat_scores.items()}

    categories = sorted({c for scores in category_scores.values() for c in scores})
    divergence: Dict[str, float] = {}
    level_agreement: Dict[str, bool] = {}
    for category in categories:
        vals = [category_scores[m].get(category, 0.0) for m in models]
        divergence[category] = round(max(vals) - min(vals), 4)
        level_agreement[category] = len({_semaphore(v) for v in vals}) == 1

    mean_divergence = (
        round(sum(divergence.values()) / len(divergence), 4) if divergence else 0.0
    )
    return EmbedderComparison(
        models=list(models),
        overall_scores=overall_scores,
        category_scores=category_scores,
        divergence=divergence,
        level_agreement=level_agreement,
        mean_divergence=mean_divergence,
    )


# Allow requests from the extension (running from file:// or localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health() -> Dict[str, str]:
    if analyze_bias_text is None:
        return {
            "status": "degraded",
            "scorer": "unavailable",
            "reason": _BIAS_IMPORT_ERROR or "bias_scorer import failed",
        }
    return {"status": "ok", "scorer": "available"}


@app.get("/debug")
async def debug() -> Dict[str, object]:
    return {
        "file": __file__,
        "cwd": os.getcwd(),
        "bias_scorer_root": str(BIAS_SCORER_ROOT),
        "bbq_data_dir": str(BBQ_DATA_DIR),
        "bias_scorer_loaded": analyze_bias_text is not None,
        "bias_scorer_import_error": _BIAS_IMPORT_ERROR,
        "analyze_request_fields": list(AnalyzeRequest.model_fields.keys()),
        "analyze_response_fields": list(AnalyzeResponse.model_fields.keys()),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    answer_text = _extract_answer_text(request)
    if not answer_text:
        return AnalyzeResponse(
            bias_score=0.0,
            biased_sentences=[],
            explanation="No text provided for analysis.",
            mode_speed=request.mode_speed,
            mode_depth=request.mode_depth,
            bias_types=request.bias_types,
            bias_type_scores=_build_bias_type_scores({}, request.bias_types),
        )

    mode_depth = request.mode_depth or "normal"

    sentence_scores: List[SentenceScore] = []
    hierarchy: Optional[BiasHierarchy] = None
    embedder_comparison: Optional[EmbedderComparison] = None
    try:
        (score, sentences, explanation, category_scores,
         category_directions, category_mahal,
         sentence_scores) = _score_requested_categories(
            answer_text,
            request.bias_types,
            mode_depth=mode_depth,
        )
        explanation = f"[embedder=sentence_transformers] {explanation}"
        hierarchy = _build_hierarchy(answer_text, sentence_scores, request.hierarchy_weighting)
        if request.compare_embedders:
            models = request.embedder_models or ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
            embedder_comparison = _compare_embedders(
                answer_text, request.bias_types, mode_depth, models
            )
    except Exception as exc:
        return AnalyzeResponse(
            bias_score=0.0,
            biased_sentences=[],
            explanation=f"Bias analysis failed: {exc}",
            mode_speed=request.mode_speed,
            mode_depth=request.mode_depth,
            bias_types=request.bias_types,
            bias_type_scores=_build_bias_type_scores({}, request.bias_types),
        )

    return AnalyzeResponse(
        bias_score=score,
        biased_sentences=sentences,
        explanation=explanation,
        sentence_scores=sentence_scores,
        hierarchy=hierarchy,
        embedder_comparison=embedder_comparison,
        mode_speed=request.mode_speed,
        mode_depth=request.mode_depth,
        bias_types=request.bias_types,
        bias_type_scores=_build_bias_type_scores(category_scores, request.bias_types),
        bias_type_directions=_build_bias_type_directions(category_directions, request.bias_types),
        mahal_type_scores=_build_bias_type_scores(category_mahal, request.bias_types),
    )
