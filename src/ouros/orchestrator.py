"""LangGraph orchestrator for the Milestone 1 walking skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ouros.blackboard import Blackboard
from ouros.config import load_models_config, load_system_config
from ouros.hypothesis import HypothesisGenerationAgent, lit_review_from_dict
from ouros.literature import LiteratureReviewAgent
from ouros.schemas import Hypothesis, LitReview, RunState


class Milestone1State(TypedDict, total=False):
    """State carried through the Milestone 1 DAG."""

    problem: str
    domain_tags: list[str]
    run_id: str
    lit_review: LitReview
    hypotheses: list[Hypothesis]
    report: str
    error: str


@dataclass(slots=True)
class Milestone1Result:
    """Result returned by a completed walking-skeleton run."""

    run: RunState
    lit_review: LitReview | None
    hypotheses: list[Hypothesis]
    report: str


class Milestone1Orchestrator:
    """Run the Milestone 1 DAG against a blackboard."""

    def __init__(
        self,
        *,
        blackboard: Blackboard,
        literature_agent: LiteratureReviewAgent,
        hypothesis_agent: HypothesisGenerationAgent,
        strategy_id: str = "S001",
    ) -> None:
        self.blackboard = blackboard
        self.literature_agent = literature_agent
        self.hypothesis_agent = hypothesis_agent
        self.strategy_id = strategy_id
        self.graph = self._build_graph()

    def run(self, problem: str, domain_tags: list[str] | None = None) -> Milestone1Result:
        """Execute one walking-skeleton run."""

        initial_state: Milestone1State = {
            "problem": problem,
            "domain_tags": domain_tags or [],
        }
        final_state = self.graph.invoke(initial_state)

        run = self.blackboard.get_run(final_state["run_id"])
        if run is None:
            raise RuntimeError("Run was not persisted")

        return Milestone1Result(
            run=run,
            lit_review=final_state.get("lit_review"),
            hypotheses=final_state.get("hypotheses", []),
            report=final_state.get("report", ""),
        )

    def _build_graph(self) -> Any:
        graph = StateGraph(Milestone1State)
        graph.add_node("initialize", self._initialize)
        graph.add_node("literature_review", self._literature_review)
        graph.add_node("hypothesis_generation", self._hypothesis_generation)
        graph.add_node("report", self._report)
        graph.add_node("fail", self._fail)

        graph.set_entry_point("initialize")
        graph.add_conditional_edges(
            "initialize",
            _route_on_error,
            {"ok": "literature_review", "error": "fail"},
        )
        graph.add_conditional_edges(
            "literature_review",
            _route_on_error,
            {"ok": "hypothesis_generation", "error": "fail"},
        )
        graph.add_conditional_edges(
            "hypothesis_generation",
            _route_on_error,
            {"ok": "report", "error": "fail"},
        )
        graph.add_edge("report", END)
        graph.add_edge("fail", END)
        return graph.compile()

    def _initialize(self, state: Milestone1State) -> Milestone1State:
        try:
            run = self.blackboard.create_run(
                problem=state["problem"],
                domain_tags=state.get("domain_tags", []),
                strategy_id=self.strategy_id,
            )
            self.blackboard.update_run_status(run.run_id, "running", "literature_review")
            return {"run_id": run.run_id}
        except Exception as error:
            return {"error": _safe_error(error)}

    def _literature_review(self, state: Milestone1State) -> Milestone1State:
        try:
            self.blackboard.update_run_status(state["run_id"], "running", "literature_review")
            lit_review = self.literature_agent.run(
                blackboard=self.blackboard,
                run_id=state["run_id"],
                problem=state["problem"],
                domain_tags=state.get("domain_tags", []),
            )
            return {"lit_review": lit_review}
        except Exception as error:
            return {"error": _safe_error(error)}

    def _hypothesis_generation(self, state: Milestone1State) -> Milestone1State:
        try:
            self.blackboard.update_run_status(
                state["run_id"],
                "running",
                "hypothesis_generation",
            )
            lit_review = state.get("lit_review")
            if lit_review is None:
                entry = self.blackboard.read_latest(state["run_id"], "lit_review")
                if entry is None:
                    raise RuntimeError("Missing literature review")
                lit_review = lit_review_from_dict(entry.value)

            hypotheses = self.hypothesis_agent.run(
                blackboard=self.blackboard,
                run_id=state["run_id"],
                problem=state["problem"],
                lit_review=lit_review,
            )
            return {"hypotheses": hypotheses}
        except Exception as error:
            return {"error": _safe_error(error)}

    def _report(self, state: Milestone1State) -> Milestone1State:
        report = format_console_report(
            problem=state["problem"],
            lit_review=state.get("lit_review"),
            hypotheses=state.get("hypotheses", []),
        )
        self.blackboard.write_entry(
            run_id=state["run_id"],
            agent_name="reporter",
            key="console_report",
            value={"text": report},
        )
        self.blackboard.update_run_status(state["run_id"], "complete", "complete")
        return {"report": report}

    def _fail(self, state: Milestone1State) -> Milestone1State:
        run_id = state.get("run_id")
        if run_id:
            self.blackboard.update_run_status(run_id, "failed", "failed")
            self.blackboard.write_entry(
                run_id=run_id,
                agent_name="orchestrator",
                key="error",
                value={"message": state.get("error", "Unknown failure")},
            )
        return {"report": f"Run failed: {state.get('error', 'Unknown failure')}"}


def create_milestone1_orchestrator(
    *,
    system_config_path: str | Path | None = None,
    models_config_path: str | Path | None = None,
    blackboard: Blackboard | None = None,
    literature_agent: LiteratureReviewAgent | None = None,
    hypothesis_agent: HypothesisGenerationAgent | None = None,
) -> Milestone1Orchestrator:
    """Create a Milestone 1 orchestrator from config files."""

    system_config = (
        load_system_config(system_config_path) if system_config_path else load_system_config()
    )
    models = load_models_config(models_config_path) if models_config_path else load_models_config()
    db_path = system_config.get("blackboard", {}).get("db_path", "./data/blackboard.db")

    return Milestone1Orchestrator(
        blackboard=blackboard or Blackboard(db_path),
        literature_agent=literature_agent or LiteratureReviewAgent(),
        hypothesis_agent=hypothesis_agent
        or HypothesisGenerationAgent(model=models.get("hypothesis_gen", "ollama/mistral:7b")),
    )


def format_console_report(
    *,
    problem: str,
    lit_review: LitReview | None,
    hypotheses: list[Hypothesis],
) -> str:
    """Render the walking-skeleton console report."""

    lines = [
        "Ouros Milestone 1 Report",
        f"Problem: {problem}",
        "",
        "Literature Review",
        f"- Papers found: {len(lit_review.papers) if lit_review else 0}",
        f"- Confidence: {lit_review.confidence if lit_review else 0.0}",
    ]

    if lit_review:
        lines.extend(f"- Gap: {gap}" for gap in lit_review.gaps)

    lines.extend(["", "Ranked Hypotheses"])
    for index, hypothesis in enumerate(hypotheses, 1):
        score = (hypothesis.novelty_score + hypothesis.feasibility_score) / 2
        lines.extend(
            [
                f"{index}. {hypothesis.text}",
                f"   score={score:.2f} novelty={hypothesis.novelty_score:.2f} "
                f"feasibility={hypothesis.feasibility_score:.2f}",
                f"   rationale={hypothesis.rationale}",
            ]
        )

    if not hypotheses:
        lines.append("No hypotheses generated.")

    return "\n".join(lines)


def _route_on_error(state: Milestone1State) -> str:
    return "error" if state.get("error") else "ok"


def _safe_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
