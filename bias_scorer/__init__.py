"""
bias_scorer — gender bias measurement via BBQ anchor embeddings.

Quickstart (plugin backend):
    from bias_scorer import analyze
    result = analyze(answer_text, mode_depth="normal")
    print(result.bias_score, result.semaphore)

Quickstart (standalone experiment):
    from bias_scorer import Embedder, compute_anchors, score_text
    emb = Embedder()
    anchors = compute_anchors(emb)
    result = score_text(my_text, emb, anchors, mode_depth="deep")
"""

import re

from .embedder import Embedder
from .anchors import compute_anchors, AnchorData, _default_paths_for_category
from .scorer import score_text, BiasResult, ChunkResult

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

__all__ = [
    "Embedder", "compute_anchors", "AnchorData",
    "score_text", "BiasResult", "ChunkResult",
    "analyze",
]

# ---------------------------------------------------------------------------
# Lazy singleton for the plugin backend (avoids re-loading the model per request)
# ---------------------------------------------------------------------------

_embedders: dict[str, Embedder] = {}
_anchors: dict[str, AnchorData] = {}


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def _ensure_loaded(model_name: str = _DEFAULT_MODEL, category: str = "Gender_identity") -> tuple[Embedder, AnchorData]:
    """Ensure a model and anchors for the requested BBQ category are loaded.

    Embedders are cached per model_name, and anchors per (model_name, category),
    so comparing several embedders never reuses one model's weights or anchors
    for another. The default model keeps the original cache filenames; any other
    model gets a model-suffixed cache file (embedding dims differ between models,
    so they must not share a cache).
    """
    global _embedders, _anchors
    if model_name not in _embedders:
        _embedders[model_name] = Embedder(model_name)
    embedder = _embedders[model_name]

    key = f"{model_name}::{category}"
    if key not in _anchors:
        if model_name == _DEFAULT_MODEL:
            _anchors[key] = compute_anchors(embedder, category=category)
        else:
            bbq_file, default_cache = _default_paths_for_category(category)
            cache_file = default_cache.with_name(
                f"{default_cache.stem}__{_model_slug(model_name)}.pt"
            )
            _anchors[key] = compute_anchors(
                embedder, bbq_file=bbq_file, cache_file=cache_file, category=category
            )
    return embedder, _anchors[key]


def analyze(
    answer: str,
    mode_depth: str = "normal",
    model_name: str = "all-MiniLM-L6-v2",
    category: str = "Gender_identity",
) -> BiasResult:
    """One-call entry point: load model + anchors once, then score.

    `category` should be a BBQ dataset name (e.g. "Gender_identity").
    """
    embedder, anchors = _ensure_loaded(model_name, category=category)
    return score_text(answer, embedder, anchors, mode_depth=mode_depth)
