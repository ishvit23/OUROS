from ouros.schemas import Hypothesis, LitReview, Paper, RunState, to_json_dict


def test_run_state_defaults_are_isolated() -> None:
    first = RunState(
        run_id="run-1",
        problem="Test problem",
        domain_tags=["ml"],
        status="pending",
        current_stage="created",
        strategy_id="S001",
        created_at=1.0,
        updated_at=1.0,
    )
    second = RunState(
        run_id="run-2",
        problem="Another problem",
        domain_tags=[],
        status="pending",
        current_stage="created",
        strategy_id="S001",
        created_at=1.0,
        updated_at=1.0,
    )

    first.human_notes.append("Needs a tighter scope")

    assert first.human_notes == ["Needs a tighter scope"]
    assert second.human_notes == []


def test_contracts_serialize_to_json_ready_dicts() -> None:
    review = LitReview(
        papers=[
            Paper(
                title="Example Paper",
                summary="A concise summary.",
                url="https://example.org/paper",
                year=2026,
                domain_tags=["ml"],
            )
        ],
        gaps=["No small-data baseline"],
        methodologies=["classification"],
        baselines=["logistic_regression"],
        confidence=0.75,
    )

    serialized = to_json_dict(review)

    assert serialized["papers"][0]["title"] == "Example Paper"
    assert serialized["confidence"] == 0.75


def test_hypothesis_contract_keeps_scores_explicit() -> None:
    hypothesis = Hypothesis(
        hypothesis_id="H001",
        text="A simpler baseline will improve reliability.",
        novelty_score=0.4,
        feasibility_score=0.9,
        rationale="The dataset is small and easy to evaluate locally.",
        source_papers=["https://example.org/paper"],
    )

    assert hypothesis.novelty_score == 0.4
    assert hypothesis.feasibility_score == 0.9
