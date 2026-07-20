from __future__ import annotations

import csv
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import AutoMilConfig
from .state import json_ready, now_iso


@dataclass(frozen=True)
class LiteraturePaper:
    title: str
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    abstract: str | None = None
    source: str = "user"


@dataclass(frozen=True)
class LiteratureProposal:
    name: str
    motivation: str
    core_modules: list[str]
    mechanism: str
    expected_contribution: str
    ablation_plan: list[str]
    evidence_titles: list[str] = field(default_factory=list)
    model_name: str = "AB_MIL"
    support_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LiteratureSearchSummary:
    generated_at: str
    query: str
    mode: str
    min_year: int
    max_papers: int
    papers: list[LiteraturePaper]
    proposals: list[LiteratureProposal]
    warnings: list[str] = field(default_factory=list)


def run_literature_search(
    cfg: AutoMilConfig,
    *,
    output_dir: str | Path | None = None,
    query: str | None = None,
    max_papers: int | None = None,
    min_year: int | None = None,
    user_sources: str | Path | None = None,
    local_review: str | Path | None = None,
    offline: bool = False,
) -> tuple[Path, Path]:
    settings = cfg.raw.get("innovation", {}).get("literature_search", {})
    output_dir = Path(output_dir) if output_dir else cfg.output_dir / "literature_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    query = query or str(settings.get("query") or _default_query(cfg))
    max_papers = int(max_papers if max_papers is not None else settings.get("max_papers", 8))
    min_year = int(min_year if min_year is not None else settings.get("min_year", 2023))

    warnings: list[str] = []
    local_papers = _load_local_review(cfg, local_review=local_review, warnings=warnings)
    user_papers = _load_user_papers(cfg, user_sources=user_sources, warnings=warnings)
    papers = [*local_papers, *user_papers]
    mode = "local_review" if local_papers else "user_provided"
    if local_papers and user_papers:
        mode = "local_review_plus_user"
    if not papers and not offline:
        mode = "auto_search"
        papers = _auto_search(query=query, max_papers=max_papers, min_year=min_year, warnings=warnings)
    if not papers:
        mode = "offline_empty" if offline else mode
        warnings.append("No literature papers were available; proposal seeds use conservative pathology MIL defaults.")

    papers = _dedupe_papers(papers)[:max_papers]
    proposals = _build_proposals(papers)
    summary = LiteratureSearchSummary(
        generated_at=now_iso(),
        query=query,
        mode=mode,
        min_year=min_year,
        max_papers=max_papers,
        papers=papers,
        proposals=proposals,
        warnings=warnings,
    )
    json_path = output_dir / "literature_proposals.json"
    md_path = output_dir / "literature_report.md"
    json_path.write_text(json.dumps(json_ready(asdict(summary)), indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    return md_path, json_path


def _default_query(cfg: AutoMilConfig) -> str:
    task = cfg.task_spec
    dataset = cfg.dataset_spec
    label = task.label_column or task.outcome_column or "biomarker"
    return f"pathology multiple instance learning {label} whole slide image 2025"


def _load_user_papers(cfg: AutoMilConfig, *, user_sources: str | Path | None, warnings: list[str]) -> list[LiteraturePaper]:
    papers: list[LiteraturePaper] = []
    innovation = cfg.raw.get("innovation", {})
    inline = innovation.get("literature_sources", [])
    if isinstance(inline, list):
        papers.extend(_papers_from_records(inline, source="user_config"))
    settings = innovation.get("literature_search", {})
    source_path = user_sources or settings.get("user_sources_file")
    if source_path:
        papers.extend(_load_papers_file(Path(source_path), warnings=warnings))
    return papers


def _load_local_review(
    cfg: AutoMilConfig,
    *,
    local_review: str | Path | None,
    warnings: list[str],
) -> list[LiteraturePaper]:
    settings = cfg.raw.get("innovation", {}).get("literature_search", {})
    review_path = local_review or settings.get("local_review_path")
    if not review_path:
        return []

    path = Path(review_path)
    if not path.exists():
        warnings.append(f"Local literature review not found: {path}")
        return []
    overview_path = path / "MIL_methods_overview.md" if path.is_dir() else path
    if not overview_path.is_file():
        warnings.append(f"Local literature overview is not a readable file: {overview_path}")
        return []
    try:
        overview = overview_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warnings.append(f"Could not read local literature overview {overview_path}: {exc}")
        return []

    summary_dir = overview_path.parent / "method_summaries"
    summary_count = len([item for item in summary_dir.glob("*.md") if item.name.lower() != "readme.md"]) if summary_dir.is_dir() else 0
    if not summary_count:
        warnings.append(f"Local review has no method_summaries directory next to {overview_path}; using overview only.")
    return [
        LiteraturePaper(
            title=f"Local pathology MIL literature review ({summary_count} method summaries)",
            venue="local literature corpus",
            url=str(overview_path),
            abstract=overview,
            source="local_review",
        )
    ]


def _load_papers_file(path: Path, warnings: list[str]) -> list[LiteraturePaper]:
    if not path.exists():
        warnings.append(f"User literature file not found: {path}")
        return []
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("papers", payload) if isinstance(payload, dict) else payload
            return _papers_from_records(records if isinstance(records, list) else [], source="user_file")
        if path.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                return _papers_from_records(list(csv.DictReader(f, delimiter=delimiter)), source="user_file")
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        warnings.append(f"Could not read user literature file {path}: {exc}")
    warnings.append(f"Unsupported user literature format: {path}")
    return []


def _papers_from_records(records: list[Any], *, source: str) -> list[LiteraturePaper]:
    papers = []
    for item in records:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("paper_title") or "").strip()
        if not title:
            continue
        papers.append(
            LiteraturePaper(
                title=title,
                year=_to_int(item.get("year")),
                venue=_clean(item.get("venue") or item.get("journal")),
                url=_clean(item.get("url") or item.get("link")),
                abstract=_clean(item.get("abstract") or item.get("summary") or item.get("methods")),
                source=source,
            )
        )
    return papers


def _auto_search(query: str, max_papers: int, min_year: int, warnings: list[str]) -> list[LiteraturePaper]:
    papers: list[LiteraturePaper] = []
    papers.extend(_search_semantic_scholar(query, max_papers=max_papers, min_year=min_year, warnings=warnings))
    if len(papers) < max_papers:
        papers.extend(_search_arxiv(query, max_papers=max_papers - len(papers), min_year=min_year, warnings=warnings))
    return _dedupe_papers(papers)


def _search_semantic_scholar(query: str, max_papers: int, min_year: int, warnings: list[str]) -> list[LiteraturePaper]:
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": max(1, min(max_papers, 20)),
            "fields": "title,year,abstract,url,venue",
        }
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    try:
        payload = _fetch_json(url)
    except Exception as exc:  # noqa: BLE001 - network failures should not break local experiments.
        warnings.append(f"Semantic Scholar search failed: {exc}")
        return []
    out = []
    for item in payload.get("data", []):
        year = _to_int(item.get("year"))
        if year is not None and year < min_year:
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        out.append(
            LiteraturePaper(
                title=title,
                year=year,
                venue=_clean(item.get("venue")),
                url=_clean(item.get("url")),
                abstract=_clean(item.get("abstract")),
                source="semantic_scholar",
            )
        )
    return out


def _search_arxiv(query: str, max_papers: int, min_year: int, warnings: list[str]) -> list[LiteraturePaper]:
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(max_papers, 20)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    try:
        xml_text = _fetch_text(url)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"arXiv search failed: {exc}")
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    out = []
    for entry in root.findall("atom:entry", ns):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
        abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        year = _to_int(published[:4])
        if year is not None and year < min_year:
            continue
        link = entry.findtext("atom:id", default="", namespaces=ns) or None
        if title:
            out.append(LiteraturePaper(title=title, year=year, venue="arXiv", url=link, abstract=abstract, source="arxiv"))
    return out


def _fetch_json(url: str) -> dict[str, Any]:
    return json.loads(_fetch_text(url))


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "auto-mil-literature-search/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def _build_proposals(papers: list[LiteraturePaper]) -> list[LiteratureProposal]:
    corpus = " ".join([paper.title + " " + (paper.abstract or "") for paper in papers]).lower()
    titles = [paper.title for paper in papers[:4]]
    proposals: list[LiteratureProposal] = []
    if _contains_any(corpus, ["spatial", "coordinate", "graph", "topolog", "neighbor", "neighbour"]):
        proposals.append(
            LiteratureProposal(
                name="coordinate_context_mil",
                motivation="WSI bags lose tissue layout when patch features are treated as an unordered set.",
                core_modules=["coordinate-aware neighborhood attention", "slide-to-case context pooling"],
                mechanism="Use patch coordinates to bias attention toward local neighborhoods before case-level MIL aggregation.",
                expected_contribution="Improve pathology-aware aggregation without relying on slide-level post-hoc averaging.",
                ablation_plan=["remove coordinate bias", "replace neighborhood attention with standard AB_MIL pooling"],
                evidence_titles=titles,
            )
        )
    if _contains_any(corpus, ["prototype", "cluster", "contrast", "hard instance", "instance selection"]):
        proposals.append(
            LiteratureProposal(
                name="prototype_guided_mil",
                motivation="Biomarker prediction may be driven by sparse discriminative regions inside large bags.",
                core_modules=["class prototype memory", "hard-instance auxiliary objective"],
                mechanism="Maintain class prototypes from high-attention instances and regularize bag embeddings toward class-consistent prototypes.",
                expected_contribution="Make the attention model less sensitive to noisy or weakly informative patches.",
                ablation_plan=["remove prototype memory", "remove hard-instance auxiliary objective"],
                evidence_titles=titles,
            )
        )
    if _contains_any(corpus, ["uncertain", "stable", "robust", "noise", "label noise", "confidence"]):
        proposals.append(
            LiteratureProposal(
                name="uncertainty_weighted_mil",
                motivation="Weak slide/case labels and heterogeneous tumor regions create noisy bag supervision.",
                core_modules=["uncertainty-weighted attention", "confidence-tempered loss"],
                mechanism="Down-weight uncertain instances and temper the bag loss when attention entropy indicates ambiguous evidence.",
                expected_contribution="Improve robustness under weak labels while preserving single-model inference.",
                ablation_plan=["remove uncertainty weights", "replace confidence-tempered loss with cross entropy"],
                evidence_titles=titles,
            )
        )
    if _contains_any(corpus, ["domain", "stain", "center", "centre", "scanner", "generalization"]):
        proposals.append(
            LiteratureProposal(
                name="domain_stable_mil",
                motivation="Pathology cohorts often vary by center, scanner, and staining protocol.",
                core_modules=["domain-stable feature normalization", "center-invariant bag regularizer"],
                mechanism="Normalize bag statistics and penalize center-predictive bag representations when center labels are available.",
                expected_contribution="Improve center robustness and external-test transfer.",
                ablation_plan=["remove feature normalization", "remove center-invariant regularizer"],
                evidence_titles=titles,
            )
        )
    if not proposals:
        proposals.append(
            LiteratureProposal(
                name="focal_proto_abmil",
                motivation="A compact first method proposal for imbalanced pathology MIL endpoints.",
                core_modules=["class-balanced focal objective", "prototype auxiliary head"],
                mechanism="Combine focal reweighting with a prototype regularizer on bag embeddings to emphasize sparse positive evidence.",
                expected_contribution="Improve single-model classification while keeping the method easy to ablate.",
                ablation_plan=["remove focal objective", "remove prototype auxiliary head"],
                evidence_titles=titles,
            )
        )
    return proposals[:4]


def _render_markdown(summary: LiteratureSearchSummary) -> str:
    lines = [
        "# Literature and Proposal Search",
        "",
        f"- Generated: `{summary.generated_at}`",
        f"- Mode: `{summary.mode}`",
        f"- Query: `{summary.query}`",
        f"- Min year: `{summary.min_year}`",
        "",
        "## Papers",
        "",
        "| # | Source | Year | Title | Venue | URL |",
        "|---:|---|---:|---|---|---|",
    ]
    for idx, paper in enumerate(summary.papers, start=1):
        lines.append(
            f"| {idx} | {paper.source} | {paper.year or ''} | {paper.title.replace('|', '/')} | "
            f"{(paper.venue or '').replace('|', '/')} | {paper.url or ''} |"
        )
    lines.extend(["", "## Proposal Seeds", ""])
    for proposal in summary.proposals:
        lines.extend(
            [
                f"### {proposal.name}",
                "",
                f"- Motivation: {proposal.motivation}",
                f"- Core modules: {', '.join(proposal.core_modules)}",
                f"- Mechanism: {proposal.mechanism}",
                f"- Expected contribution: {proposal.expected_contribution}",
                f"- Ablation plan: {', '.join(proposal.ablation_plan)}",
                f"- Evidence titles: {', '.join(proposal.evidence_titles) if proposal.evidence_titles else 'none'}",
                "",
            ]
        )
    lines.extend(["## Config Snippet", "", "```yaml", "innovation:", "  method_proposals:"])
    for proposal in summary.proposals:
        lines.extend(
            [
                f"    - name: {proposal.name}",
                f"      model_name: {proposal.model_name}",
                "      core_modules:",
                *[f"        - {module}" for module in proposal.core_modules],
                "      ablation_plan:",
                *[f"        - {item}" for item in proposal.ablation_plan],
                "      support_tags: []",
                "      config_overrides: {}",
            ]
        )
    lines.append("```")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in summary.warnings])
    return "\n".join(lines)


def _dedupe_papers(papers: list[LiteraturePaper]) -> list[LiteraturePaper]:
    out = []
    seen = set()
    for paper in papers:
        key = re.sub(r"\W+", " ", paper.title).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(paper)
    return out


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_int(value: Any) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None
