"""
Pre-computes and caches BBQ-derived gender anchor embeddings and baseline stats.

Cosine metric:
  male_anchor / female_anchor = L2-normalised centroids of E(CQ + answer) over
  ambiguous BBQ examples.  Score = cos(e, m) - cos(e, f) - baseline_mean.

Mahalanobis (whitened cosine) metric:
  Estimate covariance Σ over all BBQ embeddings (CQ + completions).
  Shrink it toward a scaled identity, Σ̃ = (1-α)Σ + α·(trΣ/d)·I, with α chosen
  to bound the condition number (the raw Σ is near-singular in 384-D, so an
  uncorrected Σ^{-1/2} amplifies noise directions and saturates the score).
  Compute whitening matrix W = Σ̃^{-1/2} via eigendecomposition.
  Whiten and re-normalise the anchors: m_w = L2(W·m), f_w = L2(W·f).
  Score = cos(W·e, m_w) - cos(W·e, f_w) - mahal_baseline_mean.

  Properties:
  - Bounded in [-2, 2]; same 3σ normalisation as cosine.
  - Reduces to cosine when Σ = I (all dimensions equally informative).
  - Corrects for correlations between embedding dimensions introduced by
    the transformer encoder.
  - Runs as a single matmul per call after W is cached.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .embedder import Embedder

_DEFAULT_BBQ_DIR = Path(__file__).parent.parent / "BBQ_Data"
_DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"


def _default_paths_for_category(category: str) -> tuple[Path, Path]:
    bbq_file = _DEFAULT_BBQ_DIR / f"{category}.jsonl"
    cache_file = _DEFAULT_CACHE_DIR / f"{category.lower()}_anchors.pt"
    return bbq_file, cache_file

_CACHE_REQUIRED_KEYS = {
    "male_anchor", "female_anchor",
    "baseline_mean", "baseline_std",
    "mahal_W", "mahal_anchor_m", "mahal_anchor_f",
    "mahal_baseline_mean", "mahal_baseline_std",
}


@dataclass
class AnchorData:
    # The two poles of the bias axis. For the "gender" scheme these are the
    # male/female centroids; for the "stereo" scheme they hold the stereotyped
    # (positive pole) and anti-stereotype (negative pole) centroids. The field
    # names are kept as male_anchor/female_anchor for backward compatibility
    # with the experiment scripts that read them directly.
    male_anchor:   torch.Tensor   # (dim,) unit vector, original space — positive pole
    female_anchor: torch.Tensor   # (dim,) unit vector, original space — negative pole
    baseline_mean: float
    baseline_std:  float
    # Mahalanobis (whitened cosine) fields — None on old cache
    mahal_W:              torch.Tensor | None = field(default=None)  # (dim, dim) whitening
    mahal_anchor_m:       torch.Tensor | None = field(default=None)  # (dim,) whitened+unit
    mahal_anchor_f:       torch.Tensor | None = field(default=None)  # (dim,) whitened+unit
    mahal_baseline_mean:  float = 0.0
    mahal_baseline_std:   float = 1.0
    # Axis semantics — drives the direction labels surfaced by the scorer.
    scheme:    str = "gender"   # "gender" | "stereo"
    pos_label: str = "male"     # label for male_anchor (positive corrected score)
    neg_label: str = "female"   # label for female_anchor (negative corrected score)

    @property
    def has_mahal(self) -> bool:
        return self.mahal_W is not None


def _answer_roles(entry: dict) -> dict | None:
    info = entry["answer_info"]
    roles: dict[str, str] = {}
    for key in ("ans0", "ans1", "ans2"):
        label = info[key][1].lower()
        if label in ("man", "m", "boy", "male", "trans_m"):
            roles["male"] = entry[key]
        elif label in ("woman", "f", "girl", "female", "trans_f"):
            roles["female"] = entry[key]
        elif label == "unknown":
            roles["unknown"] = entry[key]
    return roles if len(roles) == 3 else None


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _stereo_pair_for_entry(entry: dict) -> tuple[str, str] | None:
    """Return (stereotyped_answer, anti_stereotype_answer) for a non-gender
    BBQ entry, using the dataset's `stereotyped_groups` metadata. Returns None
    when the entry has no usable stereotype/anti-stereotype contrast."""
    info = entry.get("answer_info") or {}
    sg_list = entry.get("additional_metadata", {}).get("stereotyped_groups", [])
    sg_tokens = [_normalize_token(str(sg)) for sg in sg_list if str(sg).strip()]
    if not sg_tokens:
        return None

    stereo_text = None
    anti_text = None
    for key in ("ans0", "ans1", "ans2"):
        label = (info.get(key) or ["", ""])[1]
        text = entry.get(key, "")
        label_n = _normalize_token(str(label))
        text_n = _normalize_token(str(text))

        if label_n == "unknown" or text_n in {"cantanswer", "cannotbedetermined"}:
            continue

        matches_group = any(sg and (sg in label_n or sg in text_n) for sg in sg_tokens)
        is_non_stereotype = (
            label_n.startswith("non") or text_n.startswith("non")
            or label_n.startswith("not") or text_n.startswith("not")
        )

        if matches_group and not is_non_stereotype:
            stereo_text = text
        elif anti_text is None:
            anti_text = text

    if stereo_text and anti_text:
        return stereo_text, anti_text
    return None


def _collect_pairs(
    ambig: list[dict], scheme: str
) -> tuple[list[str], list[str], list[str]]:
    """Build (cq_texts, pos_texts, neg_texts) for the chosen anchor scheme.

    pos = male / stereotyped pole, neg = female / anti-stereotype pole.
    """
    cq_texts, pos_texts, neg_texts = [], [], []
    for e in ambig:
        if scheme == "gender":
            roles = _answer_roles(e)
            if roles is None:
                continue
            pos, neg = roles["male"], roles["female"]
        else:
            pair = _stereo_pair_for_entry(e)
            if pair is None:
                continue
            pos, neg = pair
        cq = e["context"] + " " + e["question"]
        cq_texts.append(cq)
        pos_texts.append(cq + " " + pos)
        neg_texts.append(cq + " " + neg)
    return cq_texts, pos_texts, neg_texts


def _shrink_alpha(eigenvalues: torch.Tensor, target_cond: float, fallback: float) -> float:
    """Smallest shrinkage α whose regularised covariance has condition number
    ≤ target_cond. Shrinkage keeps eigenvectors and maps each eigenvalue
    λ → (1-α)λ + α·μ (μ = mean eigenvalue = trΣ/d), so the spectrum — and thus
    the condition number — can be evaluated without re-decomposing."""
    ev = eigenvalues.clamp(min=0.0)
    mu = ev.mean()
    for a in (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9):
        ev_s = (1.0 - a) * ev + a * mu
        cond = (ev_s.max() / ev_s.clamp(min=1e-12).min()).item()
        if cond <= target_cond:
            return a
    return fallback


def _compute_whitening(
    e_cq:   torch.Tensor,
    e_male: torch.Tensor,
    e_fem:  torch.Tensor,
    male_anchor:   torch.Tensor,
    female_anchor: torch.Tensor,
    target_cond: float = 50.0,
    alpha_fallback: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Compute a shrinkage-regularised whitening matrix W = Σ̃^{-1/2}, whiten the
    anchors, and derive the Mahalanobis baseline over BBQ CQ texts.

    The 384-dim covariance estimated from a few thousand BBQ embeddings is
    near-singular: its smallest eigenvalues are ~0, and an uncorrected
    Σ^{-1/2} amplifies those noise directions by orders of magnitude, so the
    whitened score saturates on essentially any input. We therefore shrink the
    covariance toward a scaled identity, Σ̃ = (1-α)Σ + α·(trΣ/d)·I, choosing the
    smallest α that bounds the condition number (default ≤ 50). This bounds the
    eigenvalue amplification and restores a discriminative metric.

    Returns (W, mahal_anchor_m, mahal_anchor_f, baseline_mean, baseline_std).
    """
    all_embs = torch.cat([e_cq, e_male, e_fem], dim=0).double()
    mean_emb = all_embs.mean(dim=0)
    centered = all_embs - mean_emb
    cov = (centered.T @ centered) / (len(all_embs) - 1)           # (dim, dim)

    # Shrinkage-regularised W = Σ̃^{-1/2}. Shrinkage preserves eigenvectors and
    # maps λ → (1-α)λ + α·μ, so we decompose once and shift the eigenvalues.
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)            # ascending order
    mu = eigenvalues.clamp(min=0.0).mean()
    alpha = _shrink_alpha(eigenvalues, target_cond, alpha_fallback)
    ev_reg = ((1.0 - alpha) * eigenvalues + alpha * mu).clamp(min=1e-8)
    cond_after = (ev_reg.max() / ev_reg.min()).item()
    print(f"[anchors] whitening shrinkage alpha={alpha:.3f} -> condition number {cond_after:.1f}")
    W = (eigenvectors * ev_reg.pow(-0.5).unsqueeze(0)) @ eigenvectors.T
    W = W.float()

    # Whiten and renormalise anchors
    m_w = F.normalize((W @ male_anchor.float()).unsqueeze(0), dim=1).squeeze(0)
    f_w = F.normalize((W @ female_anchor.float()).unsqueeze(0), dim=1).squeeze(0)

    # Baseline mean from CQ-only (embedder inherent lean on neutral text).
    # Std from all BBQ completions (CQ + male + female) — CQ alone clusters
    # too tightly after whitening because the bias axis aligns with high-variance
    # directions that whitening compresses. Floor at cosine_std to prevent
    # the normalization from saturating on all inputs.
    all_w   = torch.cat([e_cq, e_male, e_fem], dim=0).float()
    all_w_n = F.normalize(all_w @ W.T, dim=1)
    e_cq_w  = F.normalize(e_cq.float() @ W.T, dim=1)
    all_scores  = (all_w_n @ m_w) - (all_w_n @ f_w)
    cq_scores   = (e_cq_w  @ m_w) - (e_cq_w  @ f_w)
    mean = cq_scores.mean().item()
    std  = max(all_scores.std().item(), 1e-4)   # floor prevents ceiling saturation
    return W, m_w, f_w, mean, std


def compute_anchors(
    embedder:   Embedder,
    bbq_file:   Path | None = None,
    cache_file: Path | None = None,
    category:   str | None = None,
    force:      bool = False,
) -> AnchorData:
    # Derive default paths from category when explicit files are not provided.
    if category is not None and (bbq_file is None or cache_file is None):
        d_bbq, d_cache = _default_paths_for_category(category)
        bbq_file = bbq_file or d_bbq
        cache_file = cache_file or d_cache

    if bbq_file is None:
        bbq_file = Path(__file__).parent.parent / "BBQ_Data" / "Gender_identity.jsonl"
    if cache_file is None:
        cache_file = Path(__file__).parent / "cache" / "gender_anchors.pt"

    # Gender uses male/female poles; every other category uses the generic
    # stereotyped/anti-stereotype contrast derived from BBQ metadata.
    scheme = "gender" if category in (None, "Gender_identity") else "stereo"
    pos_label, neg_label = (
        ("male", "female") if scheme == "gender"
        else ("stereotyped", "anti-stereotype")
    )

    if not force and cache_file.exists():
        saved = torch.load(cache_file, weights_only=False)
        if _CACHE_REQUIRED_KEYS.issubset(saved.keys()):
            return AnchorData(
                male_anchor=saved["male_anchor"],
                female_anchor=saved["female_anchor"],
                baseline_mean=float(saved["baseline_mean"]),
                baseline_std=float(saved["baseline_std"]),
                mahal_W=saved["mahal_W"],
                mahal_anchor_m=saved["mahal_anchor_m"],
                mahal_anchor_f=saved["mahal_anchor_f"],
                mahal_baseline_mean=float(saved["mahal_baseline_mean"]),
                mahal_baseline_std=float(saved["mahal_baseline_std"]),
                scheme=saved.get("scheme", scheme),
                pos_label=saved.get("pos_label", pos_label),
                neg_label=saved.get("neg_label", neg_label),
            )
        print("[anchors] Cache outdated (missing Mahalanobis fields) — recomputing.")

    with open(bbq_file, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    ambig = [e for e in entries if e["context_condition"] == "ambig"]

    cq_texts, male_texts, female_texts = _collect_pairs(ambig, scheme)
    if not cq_texts:
        raise RuntimeError(
            f"No usable ambiguous BBQ pairs found (scheme={scheme}, file={bbq_file})."
        )

    print(f"[anchors] Embedding {len(cq_texts)} ambiguous BBQ examples (scheme={scheme})...")
    e_cq   = embedder.encode(cq_texts)
    e_male = embedder.encode(male_texts)
    e_fem  = embedder.encode(female_texts)

    # --- Cosine anchors and baseline (unchanged) ---
    male_anchor   = F.normalize(e_male.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    female_anchor = F.normalize(e_fem.mean(dim=0, keepdim=True), dim=1).squeeze(0)

    e_cq_n      = F.normalize(e_cq.float(), dim=1)
    base_scores = (e_cq_n @ male_anchor.float()) - (e_cq_n @ female_anchor.float())
    baseline_mean = base_scores.mean().item()
    baseline_std  = base_scores.std().item()

    # --- Whitened cosine (Mahalanobis) metric ---
    print("[anchors] Computing whitening matrix Sigma^{-1/2} (one-time, ~5s)...")
    W, m_w, f_w, mahal_mean, mahal_std = _compute_whitening(
        e_cq, e_male, e_fem, male_anchor, female_anchor
    )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            male_anchor=male_anchor,
            female_anchor=female_anchor,
            baseline_mean=torch.tensor(baseline_mean),
            baseline_std=torch.tensor(baseline_std),
            mahal_W=W,
            mahal_anchor_m=m_w,
            mahal_anchor_f=f_w,
            mahal_baseline_mean=torch.tensor(mahal_mean),
            mahal_baseline_std=torch.tensor(mahal_std),
            scheme=scheme,
            pos_label=pos_label,
            neg_label=neg_label,
        ),
        cache_file,
    )
    print(
        f"[anchors] Cached to {cache_file}\n"
        f"          cosine      baseline: mean={baseline_mean:.6f}  std={baseline_std:.6f}\n"
        f"          mahal (wcs) baseline: mean={mahal_mean:.6f}  std={mahal_std:.6f}"
    )
    return AnchorData(
        male_anchor, female_anchor, baseline_mean, baseline_std,
        W, m_w, f_w, mahal_mean, mahal_std,
        scheme=scheme, pos_label=pos_label, neg_label=neg_label,
    )
