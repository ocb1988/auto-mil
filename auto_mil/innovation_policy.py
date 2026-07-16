from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


METHOD_TOKEN_RULES: tuple[tuple[str, str], ...] = (
    ("FOCAL", "class-balanced focal objective"),
    ("PROTO", "prototype auxiliary head or regularizer"),
    ("COORD", "coordinate-aware pathology MIL aggregation"),
    ("SPATIAL", "spatial pathology inductive bias"),
    ("GRAPH", "graph-based bag interaction module"),
    ("HYBRID", "hybrid bag-instance MIL aggregation"),
    ("CLASSWISE", "class-wise evidence attention aggregation"),
    ("TOPK", "top-k instance evidence aggregation"),
    ("ROUTE", "prototype-routed bag aggregation"),
    ("UNCERT", "uncertainty-aware attention modulation"),
    ("CONTRAST", "contrastive representation objective"),
    ("CALIB", "calibration-aware training objective"),
)

SUPPORT_TOKEN_RULES: tuple[tuple[str, str], ...] = (
    ("ENSEMBLE", "ensemble"),
    ("TRI_ENSEMBLE", "ensemble"),
    ("LR", "learning-rate tuning"),
    ("DO", "dropout tuning"),
    ("NOBAL", "sampler tuning"),
    ("BAL", "sampler tuning"),
    ("FG", "focal-gamma tuning"),
    ("PL", "prototype-loss weight tuning"),
    ("PT", "prototype-temperature tuning"),
    ("TK", "top-k instance count tuning"),
    ("IA", "instance-branch weight tuning"),
    ("NR", "route-count tuning"),
    ("RT", "route-temperature tuning"),
    ("UA", "uncertainty-weight tuning"),
    ("SEED", "seed search"),
    ("EPOCH", "training-length tuning"),
)


@dataclass(frozen=True)
class InnovationPolicy:
    track: str
    core_modules: list[str] = field(default_factory=list)
    support_tags: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_variant(variant: str) -> InnovationPolicy:
    """Classify a named custom innovation variant into method/support/baseline."""
    upper = variant.upper()
    core_modules = _dedupe([label for token, label in METHOD_TOKEN_RULES if token in upper])
    support_tags = _dedupe(
        [
            label
            for token, label in SUPPORT_TOKEN_RULES
            if _token_present(upper, token)
        ]
    )
    if "ENSEMBLE" in upper:
        return InnovationPolicy(
            track="support",
            core_modules=core_modules,
            support_tags=_dedupe(["ensemble", *support_tags]),
            rationale="Ensembling is tracked as auxiliary/support evidence, not a single-model method claim.",
        )
    if core_modules:
        return InnovationPolicy(
            track="method",
            core_modules=core_modules,
            support_tags=support_tags,
            rationale="Variant contains at least one ablatable method component.",
        )
    if support_tags:
        return InnovationPolicy(
            track="support",
            support_tags=support_tags,
            rationale="Variant changes only engineering, runtime, sampling, seed, or hyperparameter settings.",
        )
    return InnovationPolicy(
        track="baseline",
        rationale="No explicit method or support token was detected.",
    )


def classify_recipe_stage(stage: str, notes: str = "", overrides: dict[str, Any] | None = None) -> InnovationPolicy:
    text = " ".join([stage, notes, " ".join((overrides or {}).keys())]).upper()
    if "BASELINE" in text or "SCREEN" in text or "FAMILY" in text:
        return InnovationPolicy(track="baseline", rationale="Screening or family-comparison recipe.")
    if "EXPLOIT" in text or "TUNE" in text or "FOCUSED" in text:
        return InnovationPolicy(
            track="support",
            support_tags=["local tuning"],
            rationale="Focused/exploit recipes are treated as support unless a method module is declared.",
        )
    return classify_variant(text)


def render_policy_note(policy: InnovationPolicy) -> str:
    modules = ", ".join(policy.core_modules) if policy.core_modules else "none"
    support = ", ".join(policy.support_tags) if policy.support_tags else "none"
    return f"track={policy.track}; core_modules={modules}; support_tags={support}; {policy.rationale}"


def _token_present(text: str, token: str) -> bool:
    if token in {"LR", "DO", "FG", "PL", "PT", "TK", "IA", "NR", "RT", "UA"}:
        return re.search(rf"{token}[0-9]", text) is not None
    if token in {"BAL", "NOBAL"}:
        return re.search(rf"(^|_){token}($|_)", text) is not None
    return token in text


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
