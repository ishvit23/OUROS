"""Hypothesis generation agent for the Milestone 1 walking skeleton."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

import litellm

from ouros.blackboard import Blackboard
from ouros.schemas import Hypothesis, LitReview, Paper


class ModelClient(Protocol):
    """Minimal chat-completion protocol for model-backed agents."""

    def complete(self, *, model: str, messages: list[dict[str, str]]) -> str: ...


class LiteLLMModelClient:
    """LiteLLM-backed model client."""

    def complete(self, *, model: str, messages: list[dict[str, str]]) -> str:
        """Call the configured model and return text content."""

        response = litellm.completion(model=model, messages=messages)
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model returned non-text content")
        return content


class HypothesisGenerationAgent:
    """Generate ranked hypotheses from a literature review."""

    agent_name = "hypothesis_generation"
    output_key = "hypotheses"

    def __init__(
        self,
        *,
        model: str,
        model_client: ModelClient | None = None,
        max_hypotheses: int = 5,
    ) -> None:
        self.model = model
        self.model_client = model_client or LiteLLMModelClient()
        self.max_hypotheses = max(1, min(max_hypotheses, 12))

    def run(
        self,
        *,
        blackboard: Blackboard,
        run_id: str,
        problem: str,
        lit_review: LitReview,
    ) -> list[Hypothesis]:
        """Generate, validate, rank, and persist hypotheses."""

        content = self.model_client.complete(
            model=self.model,
            messages=_build_messages(problem, lit_review, self.max_hypotheses),
        )
        hypotheses = parse_hypotheses(content)
        ranked = rank_hypotheses(hypotheses)[: self.max_hypotheses]

        blackboard.write_entry(
            run_id=run_id,
            agent_name=self.agent_name,
            key=self.output_key,
            value={"hypotheses": [asdict(hypothesis) for hypothesis in ranked]},
        )
        return ranked


def parse_hypotheses(content: str) -> list[Hypothesis]:
    """Parse and validate JSON hypotheses from a model response."""

    payload = json.loads(_strip_json_fence(content))
    items = payload.get("hypotheses") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("Model output must contain a non-empty hypotheses list")

    hypotheses = [_hypothesis_from_mapping(item, index) for index, item in enumerate(items, 1)]
    if not hypotheses:
        raise ValueError("No valid hypotheses found")
    return hypotheses


def rank_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Rank hypotheses by average novelty and feasibility."""

    return sorted(
        hypotheses,
        key=lambda item: (item.novelty_score + item.feasibility_score) / 2,
        reverse=True,
    )


def lit_review_from_dict(value: dict[str, Any]) -> LitReview:
    """Rehydrate a `LitReview` from a blackboard JSON dictionary."""

    return LitReview(
        papers=[
            Paper(
                title=str(paper.get("title", "")),
                summary=str(paper.get("summary", "")),
                url=str(paper.get("url", "")),
                year=paper.get("year"),
                domain_tags=list(paper.get("domain_tags", [])),
            )
            for paper in value.get("papers", [])
            if isinstance(paper, dict)
        ],
        gaps=[str(item) for item in value.get("gaps", [])],
        methodologies=[str(item) for item in value.get("methodologies", [])],
        baselines=[str(item) for item in value.get("baselines", [])],
        confidence=float(value.get("confidence", 0.0)),
    )


def _build_messages(
    problem: str,
    lit_review: LitReview,
    max_hypotheses: int,
) -> list[dict[str, str]]:
    paper_summaries = [
        {
            "title": paper.title,
            "summary": paper.summary,
            "url": paper.url,
            "year": paper.year,
        }
        for paper in lit_review.papers[:10]
    ]
    user_payload = {
        "problem": problem,
        "papers": paper_summaries,
        "gaps": lit_review.gaps,
        "methodologies": lit_review.methodologies,
        "baselines": lit_review.baselines,
        "max_hypotheses": max_hypotheses,
    }

    return [
        {
            "role": "system",
            "content": (
                "You generate ranked, testable research hypotheses. "
                "Return only JSON with a top-level 'hypotheses' array. "
                "Each item must include text, novelty_score, feasibility_score, "
                "rationale, and source_papers. Scores must be numbers from 0 to 1."
            ),
        },
        {"role": "user", "content": json.dumps(user_payload)},
    ]


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _hypothesis_from_mapping(item: Any, index: int) -> Hypothesis:
    if not isinstance(item, dict):
        raise ValueError("Hypothesis item must be a JSON object")

    text = str(item.get("text", "")).strip()
    rationale = str(item.get("rationale", "")).strip()
    novelty = _score(item.get("novelty_score"), "novelty_score")
    feasibility = _score(item.get("feasibility_score"), "feasibility_score")

    if not text:
        raise ValueError("Hypothesis text is required")
    if not rationale:
        raise ValueError("Hypothesis rationale is required")

    source_papers = item.get("source_papers", [])
    if not isinstance(source_papers, list):
        raise ValueError("source_papers must be a list")

    return Hypothesis(
        hypothesis_id=str(item.get("hypothesis_id") or f"H{index:03}"),
        text=text,
        novelty_score=novelty,
        feasibility_score=feasibility,
        rationale=rationale,
        source_papers=[str(source) for source in source_papers],
    )


def _score(value: Any, field_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a number") from error

    if not 0 <= score <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return score
