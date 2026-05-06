from pathlib import Path

import pytest

from ouros.blackboard import Blackboard


def test_create_and_get_run(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")

    created = blackboard.create_run(
        problem="Can small models produce useful literature summaries?",
        domain_tags=["ml", "literature-review"],
        strategy_id="S001",
    )
    stored = blackboard.get_run(created.run_id)

    assert stored is not None
    assert stored.run_id == created.run_id
    assert stored.problem == "Can small models produce useful literature summaries?"
    assert stored.domain_tags == ["ml", "literature-review"]
    assert stored.status == "pending"
    assert stored.current_stage == "created"
    assert stored.strategy_id == "S001"


def test_write_and_read_latest_entry(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")
    run = blackboard.create_run("Find robust baselines", ["ml"])

    first = blackboard.write_entry(
        run_id=run.run_id,
        agent_name="literature_review",
        key="lit_review",
        value={"papers": [{"title": "Paper A"}], "confidence": 0.5},
    )
    second = blackboard.write_entry(
        run_id=run.run_id,
        agent_name="literature_review",
        key="lit_review",
        value={"papers": [{"title": "Paper B"}], "confidence": 0.8},
    )

    latest = blackboard.read_latest(run.run_id, "lit_review")

    assert first.version == 1
    assert second.version == 2
    assert latest is not None
    assert latest.version == 2
    assert latest.value["papers"][0]["title"] == "Paper B"


def test_list_entries_returns_all_versions_in_order(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")
    run = blackboard.create_run("Rank candidate hypotheses", [])

    blackboard.write_entry(run.run_id, "hypothesis_generation", "hypotheses", {"items": [1]})
    blackboard.write_entry(run.run_id, "hypothesis_generation", "hypotheses", {"items": [1, 2]})
    blackboard.write_entry(run.run_id, "reporter", "console_report", {"text": "done"})

    entries = blackboard.list_entries(run.run_id)

    assert [entry.key for entry in entries] == ["hypotheses", "hypotheses", "console_report"]
    assert [entry.version for entry in entries] == [1, 2, 1]


def test_write_entry_rejects_unknown_run(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")

    with pytest.raises(KeyError, match="Run not found"):
        blackboard.write_entry(
            run_id="missing",
            agent_name="literature_review",
            key="lit_review",
            value={},
        )


def test_update_run_status(tmp_path: Path) -> None:
    blackboard = Blackboard(tmp_path / "blackboard.db")
    run = blackboard.create_run("Track run progress", [])

    updated = blackboard.update_run_status(
        run_id=run.run_id,
        status="running",
        current_stage="literature_review",
    )

    assert updated.status == "running"
    assert updated.current_stage == "literature_review"
    assert updated.updated_at >= run.updated_at
