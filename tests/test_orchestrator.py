from pathlib import Path

from ouros.blackboard import Blackboard
from ouros.hypothesis import HypothesisGenerationAgent
from ouros.literature import LiteratureReviewAgent
from ouros.orchestrator import Milestone1Orchestrator
from ouros.schemas import Paper


class StaticSearch:
    def search(self, query: str, limit: int) -> list[Paper]:
        return [
            Paper(
                title="A baseline study",
                summary="Random forest classification benchmark.",
                url="https://example.org/a",
                year=2025,
            )
        ]


class StaticModelClient:
    def complete(self, *, model: str, messages: list[dict[str, str]]) -> str:
        return """
        {
          "hypotheses": [
            {
              "text": "Small-data validation will expose baseline fragility.",
              "novelty_score": 0.7,
              "feasibility_score": 0.9,
              "rationale": "The reviewed papers focus on benchmark performance.",
              "source_papers": ["https://example.org/a"]
            }
          ]
        }
        """


def test_milestone1_orchestrator_completes_walking_skeleton(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")
    orchestrator = Milestone1Orchestrator(
        blackboard=blackboard,
        literature_agent=LiteratureReviewAgent(
            semantic_scholar=StaticSearch(),  # type: ignore[arg-type]
            arxiv=StaticSearch(),  # type: ignore[arg-type]
        ),
        hypothesis_agent=HypothesisGenerationAgent(
            model="fake/model",
            model_client=StaticModelClient(),
        ),
    )

    result = orchestrator.run("Improve classification baselines", ["ml"])

    assert result.run.status == "complete"
    assert result.run.current_stage == "complete"
    assert result.lit_review is not None
    assert result.hypotheses[0].text == "Small-data validation will expose baseline fragility."
    assert "Ouros Milestone 1 Report" in result.report
    assert blackboard.read_latest(result.run.run_id, "lit_review") is not None
    assert blackboard.read_latest(result.run.run_id, "hypotheses") is not None
    assert blackboard.read_latest(result.run.run_id, "console_report") is not None
