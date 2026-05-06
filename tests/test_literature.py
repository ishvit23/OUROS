from pathlib import Path
from typing import Any

import requests

from ouros.blackboard import Blackboard
from ouros.literature import ArxivClient, LiteratureReviewAgent, SemanticScholarClient
from ouros.schemas import Paper


class FakeResponse:
    def __init__(self, *, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.response


class FailingSearch:
    def search(self, query: str, limit: int) -> list[Paper]:
        raise requests.Timeout("network unavailable")


class StaticSearch:
    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers

    def search(self, query: str, limit: int) -> list[Paper]:
        return self.papers[:limit]


def test_semantic_scholar_client_normalizes_results() -> None:
    session = FakeSession(
        FakeResponse(
            payload={
                "data": [
                    {
                        "title": "A baseline study",
                        "abstract": "Random forest classification benchmark.",
                        "url": "https://example.org/a",
                        "year": 2025,
                    }
                ]
            }
        )
    )

    papers = SemanticScholarClient(session=session).search("classification", limit=20)

    assert len(papers) == 1
    assert papers[0].title == "A baseline study"
    assert papers[0].year == 2025
    assert session.calls[0]["params"]["limit"] == 5


def test_arxiv_client_normalizes_atom_feed() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/1234.5678</id>
        <title> A transformer baseline </title>
        <summary> Classification with a transformer. </summary>
        <published>2024-01-01T00:00:00Z</published>
      </entry>
    </feed>
    """
    session = FakeSession(FakeResponse(text=feed))

    papers = ArxivClient(session=session).search("transformer", limit=2)

    assert len(papers) == 1
    assert papers[0].title == "A transformer baseline"
    assert papers[0].url == "https://arxiv.org/abs/1234.5678"
    assert papers[0].year == 2024


def test_literature_review_agent_writes_review(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")
    run = blackboard.create_run("Improve classification baselines", ["ml"])
    paper = Paper(
        title="A baseline study",
        summary="Random forest classification benchmark.",
        url="https://example.org/a",
        year=2025,
    )
    agent = LiteratureReviewAgent(
        semantic_scholar=StaticSearch([paper]),  # type: ignore[arg-type]
        arxiv=FailingSearch(),  # type: ignore[arg-type]
    )

    review = agent.run(
        blackboard=blackboard,
        run_id=run.run_id,
        problem=run.problem,
        domain_tags=run.domain_tags,
    )
    entry = blackboard.read_latest(run.run_id, "lit_review")

    assert review.papers[0].title == "A baseline study"
    assert review.methodologies == ["classification", "benchmarking"]
    assert review.baselines == ["random_forest"]
    assert entry is not None
    assert entry.value["papers"][0]["title"] == "A baseline study"


def test_literature_review_agent_handles_network_failure(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")
    run = blackboard.create_run("Sparse topic", [])
    agent = LiteratureReviewAgent(
        semantic_scholar=FailingSearch(),  # type: ignore[arg-type]
        arxiv=FailingSearch(),  # type: ignore[arg-type]
    )

    review = agent.run(
        blackboard=blackboard,
        run_id=run.run_id,
        problem=run.problem,
        domain_tags=[],
    )

    assert review.papers == []
    assert review.confidence == 0.0
    assert review.gaps == ["No literature results were available for: Sparse topic"]
