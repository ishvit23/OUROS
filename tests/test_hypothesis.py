from pathlib import Path

import pytest

from ouros.blackboard import Blackboard
from ouros.hypothesis import HypothesisGenerationAgent, parse_hypotheses, rank_hypotheses
from ouros.schemas import LitReview, Paper


class StaticModelClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def complete(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append({"model": model, "messages": messages})
        return self.content


def sample_review() -> LitReview:
    return LitReview(
        papers=[
            Paper(
                title="A baseline study",
                summary="Random forest classification benchmark.",
                url="https://example.org/a",
                year=2025,
            )
        ],
        gaps=["Need small-data validation"],
        methodologies=["classification"],
        baselines=["random_forest"],
        confidence=0.5,
    )


def test_parse_hypotheses_accepts_json_fence() -> None:
    hypotheses = parse_hypotheses(
        """```json
        {
          "hypotheses": [
            {
              "text": "Small-data validation will expose baseline fragility.",
              "novelty_score": 0.7,
              "feasibility_score": 0.9,
              "rationale": "The literature focuses on large benchmark datasets.",
              "source_papers": ["https://example.org/a"]
            }
          ]
        }
        ```"""
    )

    assert hypotheses[0].hypothesis_id == "H001"
    assert hypotheses[0].novelty_score == 0.7


def test_parse_hypotheses_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match="novelty_score"):
        parse_hypotheses(
            """
            {
              "hypotheses": [
                {
                  "text": "Invalid",
                  "novelty_score": 2,
                  "feasibility_score": 0.5,
                  "rationale": "Out of range.",
                  "source_papers": []
                }
              ]
            }
            """
        )


def test_rank_hypotheses_sorts_by_average_score() -> None:
    hypotheses = parse_hypotheses(
        """
        {
          "hypotheses": [
            {
              "text": "Lower ranked",
              "novelty_score": 0.4,
              "feasibility_score": 0.4,
              "rationale": "Lower average.",
              "source_papers": []
            },
            {
              "text": "Higher ranked",
              "novelty_score": 0.8,
              "feasibility_score": 0.9,
              "rationale": "Higher average.",
              "source_papers": []
            }
          ]
        }
        """
    )

    ranked = rank_hypotheses(hypotheses)

    assert ranked[0].text == "Higher ranked"


def test_hypothesis_agent_writes_ranked_hypotheses(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")
    run = blackboard.create_run("Improve classification baselines", ["ml"])
    model_client = StaticModelClient(
        """
        {
          "hypotheses": [
            {
              "hypothesis_id": "candidate-low",
              "text": "Lower ranked hypothesis.",
              "novelty_score": 0.3,
              "feasibility_score": 0.5,
              "rationale": "Lower average.",
              "source_papers": []
            },
            {
              "hypothesis_id": "candidate-high",
              "text": "Higher ranked hypothesis.",
              "novelty_score": 0.8,
              "feasibility_score": 0.9,
              "rationale": "Higher average.",
              "source_papers": ["https://example.org/a"]
            }
          ]
        }
        """
    )
    agent = HypothesisGenerationAgent(model="fake/model", model_client=model_client)

    hypotheses = agent.run(
        blackboard=blackboard,
        run_id=run.run_id,
        problem=run.problem,
        lit_review=sample_review(),
    )
    entry = blackboard.read_latest(run.run_id, "hypotheses")

    assert hypotheses[0].hypothesis_id == "candidate-high"
    assert entry is not None
    assert entry.value["hypotheses"][0]["hypothesis_id"] == "candidate-high"
    assert model_client.calls[0]["model"] == "fake/model"
