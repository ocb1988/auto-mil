from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .state import json_ready, now_iso


DEFAULT_PROFILE = "generic-pathology-ai"


@dataclass(frozen=True)
class ManuscriptProfile:
    profile_id: str
    display_name: str
    scope: str
    sections: list[str]
    checklist: list[str]


@dataclass(frozen=True)
class ManuscriptPackage:
    generated_at: str
    root: str
    profile: ManuscriptProfile
    target_journal: str | None
    source_draft: str | None
    source_evidence: str | None
    outputs: dict[str, str]
    warnings: list[str] = field(default_factory=list)


PROFILES: dict[str, ManuscriptProfile] = {
    "generic-pathology-ai": ManuscriptProfile(
        profile_id="generic-pathology-ai",
        display_name="Generic pathology AI manuscript",
        scope="Evidence-bounded Methods and Results for a pathology MIL study.",
        sections=[
            "Dataset and cohort",
            "Task definition",
            "Feature extraction and bag construction",
            "Split design and leakage prevention",
            "Baseline and proposed methods",
            "Training and evaluation protocol",
            "Main comparison",
            "Statistical analysis",
            "Ablation study",
            "Error analysis and limitations",
        ],
        checklist=[
            "Verify patient/case counts, slide counts, labels, and exclusion criteria.",
            "Verify the split policy and ensure no patient appears in more than one split.",
            "Confirm all baseline and proposed-method rows are completed, not dry-run records.",
            "Confirm confidence intervals or paired tests are present for repeated folds/seeds.",
            "Confirm case-level prediction figures match the primary analysis unit.",
            "Keep missing evidence as TODO items instead of filling from memory.",
        ],
    ),
    "structured-medical-ai": ManuscriptProfile(
        profile_id="structured-medical-ai",
        display_name="Structured medical AI report",
        scope="Concise medical AI experimental section with separated evidence, claims, and limitations.",
        sections=[
            "Study design",
            "Participants and data sources",
            "Model development",
            "Validation strategy",
            "Performance metrics",
            "Model comparison",
            "Robustness and ablation",
            "Limitations",
        ],
        checklist=[
            "Separate model-development data from held-out or external validation data.",
            "Report denominators for each metric and each cohort.",
            "Avoid clinical utility claims unless calibration and external validation are present.",
            "State whether any hyperparameter tuning used validation or test outcomes.",
            "Retain implementation details needed to reproduce the experiment.",
        ],
    ),
    "methods-results-only": ManuscriptProfile(
        profile_id="methods-results-only",
        display_name="Methods and Results only",
        scope="Minimal package for drafting only the experimental Methods and Results.",
        sections=[
            "Methods",
            "Results",
            "Evidence gaps",
        ],
        checklist=[
            "Preserve numeric values exactly from the evidence artifacts.",
            "Preserve model and split names exactly as recorded.",
            "Mark every missing result as TODO.",
            "Do not add discussion claims.",
        ],
    ),
}


def available_manuscript_profiles() -> list[str]:
    return sorted(PROFILES)


def package_manuscript(
    root: str | Path,
    *,
    draft_path: str | Path | None = None,
    evidence_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    target_journal: str | None = None,
) -> tuple[ManuscriptPackage, Path]:
    root = Path(root)
    profile_spec = PROFILES.get(profile)
    if profile_spec is None:
        raise ValueError(f"unknown manuscript profile: {profile}")

    output_dir = Path(output_dir) if output_dir else root / "manuscript" / "package"
    output_dir.mkdir(parents=True, exist_ok=True)

    draft_file = Path(draft_path) if draft_path else root / "manuscript" / "manuscript_draft.md"
    evidence_file = Path(evidence_path) if evidence_path else root / "manuscript" / "manuscript_evidence.json"
    warnings: list[str] = []

    draft_text = _read_text(draft_file)
    if draft_text is None:
        warnings.append(f"Draft file was not found: {draft_file}")
        draft_text = "# Manuscript Draft\n\nTODO: Run write-manuscript before package-manuscript."

    evidence = _read_json(evidence_file)
    if evidence is None:
        warnings.append(f"Evidence JSON was not found or invalid: {evidence_file}")
        evidence = {}

    package_md = output_dir / "methods_results_package.md"
    prompt_md = output_dir / "llm_polish_prompt.md"
    checklist_md = output_dir / "submission_checklist.md"
    manifest_json = output_dir / "manuscript_package.json"

    package_md.write_text(
        _render_package_markdown(
            profile_spec=profile_spec,
            target_journal=target_journal,
            root=root,
            draft_file=draft_file if draft_file.exists() else None,
            evidence_file=evidence_file if evidence_file.exists() else None,
            evidence=evidence,
            draft_text=draft_text,
            warnings=warnings,
        ),
        encoding="utf-8",
    )
    prompt_md.write_text(
        _render_polish_prompt(
            profile_spec=profile_spec,
            target_journal=target_journal,
            evidence=evidence,
            draft_text=draft_text,
        ),
        encoding="utf-8",
    )
    checklist_md.write_text(
        _render_checklist(profile_spec=profile_spec, target_journal=target_journal, warnings=warnings),
        encoding="utf-8",
    )

    package = ManuscriptPackage(
        generated_at=now_iso(),
        root=str(root),
        profile=profile_spec,
        target_journal=target_journal,
        source_draft=str(draft_file) if draft_file.exists() else None,
        source_evidence=str(evidence_file) if evidence_file.exists() else None,
        outputs={
            "methods_results_package": str(package_md),
            "llm_polish_prompt": str(prompt_md),
            "submission_checklist": str(checklist_md),
            "manifest": str(manifest_json),
        },
        warnings=warnings,
    )
    manifest_json.write_text(json.dumps(json_ready(asdict(package)), indent=2, ensure_ascii=True), encoding="utf-8")
    return package, manifest_json


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _evidence_lines(evidence: dict[str, Any]) -> list[str]:
    if not evidence:
        return ["- Evidence manifest: not available"]
    lines = [
        f"- Generated at: `{evidence.get('generated_at', 'NA')}`",
        f"- Title: `{evidence.get('title', 'NA')}`",
        f"- Primary metric: `{evidence.get('primary_metric', 'NA')}`",
    ]
    evidence_index = evidence.get("evidence")
    if isinstance(evidence_index, dict):
        for key in sorted(evidence_index):
            lines.append(f"- {key}: `{evidence_index.get(key)}`")
    warnings = evidence.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append("- Evidence warnings:")
        lines.extend([f"  - {warning}" for warning in warnings])
    return lines


def _render_package_markdown(
    *,
    profile_spec: ManuscriptProfile,
    target_journal: str | None,
    root: Path,
    draft_file: Path | None,
    evidence_file: Path | None,
    evidence: dict[str, Any],
    draft_text: str,
    warnings: list[str],
) -> str:
    target = target_journal or "not specified"
    lines = [
        f"# Manuscript Package: {profile_spec.display_name}",
        "",
        "## Target",
        "",
        f"- Profile: `{profile_spec.profile_id}`",
        f"- Target journal: `{target}`",
        f"- Run root: `{root}`",
        f"- Source draft: `{draft_file or 'not available'}`",
        f"- Source evidence: `{evidence_file or 'not available'}`",
        "",
        "If a specific journal is named, verify the current author instructions before final submission.",
        "",
        "## Intended Sections",
        "",
        *[f"- {section}" for section in profile_spec.sections],
        "",
        "## Evidence Ledger",
        "",
        *_evidence_lines(evidence),
        "",
        "## Package Warnings",
        "",
        *([f"- {warning}" for warning in warnings] if warnings else ["- None"]),
        "",
        "## Evidence-Bounded Draft",
        "",
        draft_text.strip(),
        "",
    ]
    return "\n".join(lines)


def _render_polish_prompt(
    *,
    profile_spec: ManuscriptProfile,
    target_journal: str | None,
    evidence: dict[str, Any],
    draft_text: str,
) -> str:
    target = target_journal or "the selected pathology AI journal"
    evidence_json = json.dumps(json_ready(evidence), indent=2, ensure_ascii=True)
    sections = "\n".join(f"- {section}" for section in profile_spec.sections)
    return "\n".join(
        [
            "# LLM Polish Prompt",
            "",
            "You are editing the experimental Methods and Results for a pathology AI manuscript.",
            f"Target journal or style: {target}. Verify current author instructions separately; do not invent journal rules.",
            "",
            "Use only the evidence and draft text below. Follow these rules:",
            "",
            "- Preserve all metric values, model names, split names, counts, and file references exactly.",
            "- Do not add state-of-the-art, clinical utility, robustness, or external validation claims unless explicitly supported.",
            "- Convert missing information into clear TODO markers.",
            "- Keep Methods factual and reproducible.",
            "- Keep Results ordered around the primary metric, statistical evidence, figures, ablation, and errors.",
            "- Return polished Markdown with Methods, Results, Limitations, and Evidence Gaps sections.",
            "",
            "Expected section plan:",
            "",
            sections,
            "",
            "## Evidence JSON",
            "",
            "```json",
            evidence_json,
            "```",
            "",
            "## Draft Text",
            "",
            "```markdown",
            draft_text.strip(),
            "```",
            "",
        ]
    )


def _render_checklist(
    *,
    profile_spec: ManuscriptProfile,
    target_journal: str | None,
    warnings: list[str],
) -> str:
    target = target_journal or "not specified"
    lines = [
        f"# Submission Checklist: {profile_spec.display_name}",
        "",
        f"- Target journal: `{target}`",
        "- Verify current journal author instructions before final formatting.",
        "- Confirm generated text is consistent with source evidence files.",
        "",
        "## Evidence Checks",
        "",
        *[f"- [ ] {item}" for item in profile_spec.checklist],
        "",
        "## Current Warnings",
        "",
        *([f"- [ ] Resolve: {warning}" for warning in warnings] if warnings else ["- [x] No package-level warnings"]),
        "",
        "## Finalization Checks",
        "",
        "- [ ] Add exact feature extractor and preprocessing details.",
        "- [ ] Add hardware, CUDA, PyTorch, and package versions.",
        "- [ ] Add seed, epoch, optimizer, learning-rate, and batch-size details.",
        "- [ ] Confirm all figures and tables cited in text exist on disk.",
        "- [ ] Confirm git commit hash and experiment artifact paths are recorded.",
        "",
    ]
    return "\n".join(lines)
