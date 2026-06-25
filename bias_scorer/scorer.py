import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .anchors import AnchorData
from .embedder import Embedder


@dataclass
class ChunkResult:
    text: str
    bias_score: float   # normalized 0–1
    bias_raw: float     # corrected cosine-diff (signed)
    direction: str      # "male" | "female" | "neutral"


@dataclass
class BiasResult:
    bias_score: float           # normalized 0–1 (what the plugin renders)
    bias_corrected: float       # signed corrected cosine-diff (for direction)
    direction: str
    semaphore: str              # "low" | "medium" | "high"
    biased_sentences: list[str]
    explanation: str
    chunks: list[ChunkResult] = field(default_factory=list)
    # Whitened-cosine (Mahalanobis) metric — 0.0/"neutral" when anchors lack W
    mahal_bias_score: float = 0.0
    mahal_corrected:  float = 0.0
    mahal_direction:  str   = "neutral"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(abs_corrected: float, std: float) -> float:
    """Maps |corrected| to [0, 1] using 3σ as the ceiling."""
    return min(1.0, abs_corrected / (3.0 * std + 1e-9))


def _semaphore(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _direction(
    signed_corrected: float,
    std: float,
    pos_label: str = "male",
    neg_label: str = "female",
) -> str:
    if abs(signed_corrected) < 0.5 * std:
        return "neutral"
    return pos_label if signed_corrected > 0 else neg_label


def _raw_scores(texts: list[str], embedder: Embedder, anchors: AnchorData) -> list[float]:
    """Corrected cosine bias scores (signed) for a batch of texts."""
    emb = F.normalize(embedder.encode(texts).float(), dim=1)
    raw = (emb @ anchors.male_anchor.float()) - (emb @ anchors.female_anchor.float())
    return (raw - anchors.baseline_mean).tolist()


def _mahal_raw_scores(
    texts: list[str], embedder: Embedder, anchors: AnchorData
) -> list[float]:
    """
    Whitened-cosine (Mahalanobis) bias scores (signed, baseline-corrected).

    Applies the whitening matrix W = Σ^{-1/2} to decorrelate embedding
    dimensions, then computes the same cosine difference as the standard
    metric. Bounded in [-2, 2]; reduces to cosine when Σ = I.
    """
    emb   = embedder.encode(texts).float()
    e_w   = F.normalize(emb @ anchors.mahal_W.float().T, dim=1)
    raw   = (e_w @ anchors.mahal_anchor_m.float()) - (e_w @ anchors.mahal_anchor_f.float())
    return (raw - anchors.mahal_baseline_mean).tolist()


def _split_sentences(text: str, min_chars: int = 10) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= min_chars]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_text(
    answer: str,
    embedder: Embedder,
    anchors: AnchorData,
    mode_depth: str = "normal",
) -> BiasResult:
    if not answer or not answer.strip():
        return BiasResult(0.0, 0.0, "neutral", "low", [], "No answer text provided.")

    std = anchors.baseline_std
    pos_label, neg_label = anchors.pos_label, anchors.neg_label

    if mode_depth == "deep":
        sentences = _split_sentences(answer) or [answer]
        corrected = _raw_scores(sentences, embedder, anchors)

        chunks = [
            ChunkResult(
                text=s,
                bias_score=_normalize(abs(c), std),
                bias_raw=c,
                direction=_direction(c, std, pos_label, neg_label),
            )
            for s, c in zip(sentences, corrected)
        ]

        abs_mean    = sum(abs(c) for c in corrected) / len(corrected)
        signed_mean = sum(corrected) / len(corrected)
        bias_norm   = _normalize(abs_mean, std)
        biased      = [c for c in chunks if c.bias_score >= 0.25]
    else:
        corrected   = _raw_scores([answer], embedder, anchors)
        signed_mean = corrected[0]
        bias_norm   = _normalize(abs(signed_mean), std)
        chunks      = []
        biased      = []

    direction = _direction(signed_mean, std, pos_label, neg_label)
    sem       = _semaphore(bias_norm)

    # --- Mahalanobis metric (only when anchors carry the axis) ---
    mahal_bias_score = 0.0
    mahal_corrected  = 0.0
    mahal_direction  = "neutral"
    if anchors.has_mahal:
        m_std = anchors.mahal_baseline_std
        if mode_depth == "deep":
            m_corrected = _mahal_raw_scores(sentences, embedder, anchors)
            m_abs_mean    = sum(abs(c) for c in m_corrected) / len(m_corrected)
            m_signed_mean = sum(m_corrected) / len(m_corrected)
        else:
            m_corrected   = _mahal_raw_scores([answer], embedder, anchors)
            m_abs_mean    = abs(m_corrected[0])
            m_signed_mean = m_corrected[0]
        mahal_bias_score = _normalize(m_abs_mean, m_std)
        mahal_corrected  = round(m_signed_mean, 6)
        mahal_direction  = _direction(m_signed_mean, m_std, pos_label, neg_label)

    risk_label = "Gender bias" if anchors.scheme == "gender" else "Stereotype bias"
    explanation = (
        f"{risk_label} risk: {sem}. "
        f"Direction: {direction}. "
        f"Score: {bias_norm:.3f} "
        f"(cosine corrected: {signed_mean:+.5f}; "
        f"mahal score: {mahal_bias_score:.3f}, baseline std: {std:.5f})."
    )

    return BiasResult(
        bias_score=round(bias_norm, 4),
        bias_corrected=round(signed_mean, 6),
        direction=direction,
        semaphore=sem,
        biased_sentences=[c.text for c in biased],
        explanation=explanation,
        chunks=chunks,
        mahal_bias_score=round(mahal_bias_score, 4),
        mahal_corrected=mahal_corrected,
        mahal_direction=mahal_direction,
    )
