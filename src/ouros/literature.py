"""Literature search clients and Milestone 1 literature review agent."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from ouros.blackboard import Blackboard
from ouros.schemas import LitReview, Paper

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org"
ARXIV_BASE_URL = "https://export.arxiv.org"
REQUEST_TIMEOUT_SEC = 15
MAX_RESULTS = 5


class SupportsGet(Protocol):
    """Minimal requests-compatible protocol used for tests."""

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: int,
    ) -> requests.Response: ...


def assert_allowed_url(url: str, allowed_hosts: set[str]) -> None:
    """Allow only fixed academic API hosts for outbound literature calls."""

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError(f"Blocked literature request URL: {url}")


class SemanticScholarClient:
    """Small Semantic Scholar paper search client."""

    def __init__(self, session: SupportsGet | None = None) -> None:
        self.session = session or requests.Session()

    def search(self, query: str, limit: int = MAX_RESULTS) -> list[Paper]:
        """Search Semantic Scholar and normalize results to `Paper`."""

        bounded_limit = max(1, min(limit, MAX_RESULTS))
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/graph/v1/paper/search"
        assert_allowed_url(url, {"api.semanticscholar.org"})

        response = self.session.get(
            url,
            params={
                "query": query,
                "limit": bounded_limit,
                "fields": "title,abstract,url,year",
            },
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()

        payload = response.json()
        papers: list[Paper] = []
        for item in payload.get("data", []):
            title = item.get("title")
            if not title:
                continue

            papers.append(
                Paper(
                    title=str(title),
                    summary=str(item.get("abstract") or "No abstract available."),
                    url=str(item.get("url") or ""),
                    year=item.get("year"),
                )
            )

        return papers


class ArxivClient:
    """Small arXiv Atom API search client."""

    def __init__(self, session: SupportsGet | None = None) -> None:
        self.session = session or requests.Session()

    def search(self, query: str, limit: int = MAX_RESULTS) -> list[Paper]:
        """Search arXiv and normalize results to `Paper`."""

        bounded_limit = max(1, min(limit, MAX_RESULTS))
        url = f"{ARXIV_BASE_URL}/api/query"
        assert_allowed_url(url, {"export.arxiv.org"})

        response = self.session.get(
            url,
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": bounded_limit,
            },
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()

        return _parse_arxiv_feed(response.text)


class LiteratureReviewAgent:
    """Build a lightweight `LitReview` and write it to the blackboard."""

    agent_name = "literature_review"
    output_key = "lit_review"

    def __init__(
        self,
        semantic_scholar: SemanticScholarClient | None = None,
        arxiv: ArxivClient | None = None,
        result_limit: int = MAX_RESULTS,
    ) -> None:
        self.semantic_scholar = semantic_scholar or SemanticScholarClient()
        self.arxiv = arxiv or ArxivClient()
        self.result_limit = max(1, min(result_limit, MAX_RESULTS))

    def run(
        self,
        *,
        blackboard: Blackboard,
        run_id: str,
        problem: str,
        domain_tags: list[str],
    ) -> LitReview:
        """Run literature search and persist the typed result."""

        papers = self._collect_papers(problem, domain_tags)
        review = LitReview(
            papers=papers,
            gaps=_infer_gaps(problem, papers),
            methodologies=_infer_methodologies(papers),
            baselines=_infer_baselines(papers),
            confidence=_estimate_confidence(papers),
        )

        blackboard.write_entry(
            run_id=run_id,
            agent_name=self.agent_name,
            key=self.output_key,
            value=asdict(review),
        )
        return review

    def _collect_papers(self, problem: str, domain_tags: list[str]) -> list[Paper]:
        query = " ".join([problem, *domain_tags]).strip()
        collected: list[Paper] = []

        for search in (self.semantic_scholar.search, self.arxiv.search):
            try:
                collected.extend(search(query, self.result_limit))
            except requests.RequestException:
                continue

        return _dedupe_papers(collected)[: self.result_limit * 2]


def _parse_arxiv_feed(xml_text: str) -> list[Paper]:
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []

    for entry in root.findall("atom:entry", namespace):
        title = _entry_text(entry, "atom:title", namespace)
        if not title:
            continue

        published = _entry_text(entry, "atom:published", namespace)
        papers.append(
            Paper(
                title=" ".join(title.split()),
                summary=" ".join(_entry_text(entry, "atom:summary", namespace).split()),
                url=_entry_text(entry, "atom:id", namespace),
                year=int(published[:4]) if published[:4].isdigit() else None,
            )
        )

    return papers


def _entry_text(entry: ET.Element, selector: str, namespace: dict[str, str]) -> str:
    element = entry.find(selector, namespace)
    return "" if element is None or element.text is None else element.text.strip()


def _dedupe_papers(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        identity = (paper.url or paper.title).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(paper)
    return unique


def _infer_gaps(problem: str, papers: list[Paper]) -> list[str]:
    if not papers:
        return [f"No literature results were available for: {problem}"]
    return ["Milestone 1 gap extraction is heuristic; review sources before relying on claims."]


def _infer_methodologies(papers: list[Paper]) -> list[str]:
    text = " ".join(f"{paper.title} {paper.summary}" for paper in papers).casefold()
    methods = {
        "classification": ["classification", "classifier"],
        "benchmarking": ["benchmark", "baseline"],
        "simulation": ["simulation", "simulator"],
        "survey": ["survey", "review"],
    }
    return [name for name, terms in methods.items() if any(term in text for term in terms)]


def _infer_baselines(papers: list[Paper]) -> list[str]:
    text = " ".join(f"{paper.title} {paper.summary}" for paper in papers).casefold()
    baselines = []
    for candidate in ("random forest", "logistic regression", "transformer", "svm"):
        if candidate in text:
            baselines.append(candidate.replace(" ", "_"))
    return baselines


def _estimate_confidence(papers: list[Paper]) -> float:
    if not papers:
        return 0.0
    return round(min(1.0, len(papers) / (MAX_RESULTS * 2)), 2)
