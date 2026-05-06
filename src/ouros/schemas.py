"""Typed contracts shared by Ouros agents and the blackboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RunStatus = Literal["pending", "running", "awaiting_human", "complete", "failed"]
ExperimentStatus = Literal["success", "partial", "failed", "timeout"]
AnalysisVerdict = Literal["supported", "rejected", "inconclusive"]
RunOutcome = Literal["success", "partial", "failed"]


@dataclass(slots=True)
class RunState:
    """Top-level state for one research run."""

    run_id: str
    problem: str
    domain_tags: list[str]
    status: RunStatus
    current_stage: str
    strategy_id: str
    created_at: float
    updated_at: float
    human_notes: list[str] = field(default_factory=list)
    human_score: float | None = None


@dataclass(slots=True)
class BlackboardEntry:
    """Versioned handoff record written by an agent."""

    run_id: str
    agent_name: str
    key: str
    value: dict[str, Any]
    timestamp: float
    version: int


@dataclass(slots=True)
class Paper:
    """Literature source used by the research pipeline."""

    title: str
    summary: str
    url: str
    year: int | None
    domain_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LitReview:
    """Output contract for the literature review agent."""

    papers: list[Paper]
    gaps: list[str]
    methodologies: list[str]
    baselines: list[str]
    confidence: float


@dataclass(slots=True)
class Hypothesis:
    """Ranked, testable hypothesis proposed by the hypothesis agent."""

    hypothesis_id: str
    text: str
    novelty_score: float
    feasibility_score: float
    rationale: str
    source_papers: list[str]


@dataclass(slots=True)
class ExperimentSpec:
    """Locked experiment design for an approved hypothesis."""

    dataset: str
    algorithm: str
    baselines: list[str]
    metrics: list[str]
    hyperparams: dict[str, Any]
    success_criteria: str
    estimated_compute: str


@dataclass(slots=True)
class RunResult:
    """Captured output from the experiment execution agent."""

    metrics: dict[str, Any]
    logs: list[str]
    artifacts: list[str]
    status: ExperimentStatus
    errors: list[dict[str, Any]]
    retries: int


@dataclass(slots=True)
class AnalysisReport:
    """Interpretation of experiment results against the hypothesis."""

    verdict: AnalysisVerdict
    stat_summary: dict[str, Any]
    anomalies: list[str]
    follow_ups: list[str]
    confidence: float


@dataclass(slots=True)
class ReflectionOutput:
    """Reflection agent output and reward signal."""

    critique: str
    reward_signal: float
    strategy_updates: list[str]
    next_run_hints: list[str]
    exclude_from_training: bool


@dataclass(slots=True)
class EpisodicRecord:
    """Searchable summary of a completed run."""

    run_id: str
    problem: str
    domain_tags: list[str]
    outcome: RunOutcome
    reward_signal: float
    summary: str
    strategy_id: str
    timestamp: float


@dataclass(slots=True)
class ProceduralStrategy:
    """Strategy selected by the Phase 1 contextual bandit."""

    strategy_id: str
    name: str
    description: str
    domain_tags: list[str]
    agent_configs: dict[str, Any]
    performance_history: list[float]
    weight: float
    times_used: int


def to_json_dict(value: Any) -> dict[str, Any]:
    """Serialize a dataclass contract into a JSON-ready dictionary."""

    return asdict(value)
