"""SQLite-backed blackboard for typed agent handoffs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, Session, SQLModel, create_engine, select

from ouros.schemas import BlackboardEntry, RunState, RunStatus


class RunStateRecord(SQLModel, table=True):
    """SQLite representation of a research run."""

    __tablename__ = "run_states"

    run_id: str = Field(primary_key=True)
    problem: str
    domain_tags: list[str] = Field(sa_column=Column(JSON, nullable=False))
    status: str
    current_stage: str
    strategy_id: str
    created_at: float
    updated_at: float
    human_notes: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    human_score: float | None = None


class BlackboardEntryRecord(SQLModel, table=True):
    """SQLite representation of a versioned blackboard entry."""

    __tablename__ = "blackboard_entries"
    __table_args__ = (
        UniqueConstraint("run_id", "key", "version", name="uq_blackboard_entry_version"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    agent_name: str
    key: str = Field(index=True)
    value: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    timestamp: float
    version: int


class Blackboard:
    """Repository-style API for run state and agent handoff storage."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self.engine)

    def create_run(
        self,
        problem: str,
        domain_tags: list[str] | None = None,
        strategy_id: str = "S001",
    ) -> RunState:
        """Create and persist a new pending run."""

        now = time.time()
        record = RunStateRecord(
            run_id=str(uuid4()),
            problem=problem,
            domain_tags=domain_tags or [],
            status="pending",
            current_stage="created",
            strategy_id=strategy_id,
            created_at=now,
            updated_at=now,
        )

        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)

        return _run_state_from_record(record)

    def get_run(self, run_id: str) -> RunState | None:
        """Read a run by ID."""

        with Session(self.engine) as session:
            record = session.get(RunStateRecord, run_id)
            if record is None:
                return None
            return _run_state_from_record(record)

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        current_stage: str,
    ) -> RunState:
        """Update run progress for the orchestrator."""

        with Session(self.engine) as session:
            record = session.get(RunStateRecord, run_id)
            if record is None:
                raise KeyError(f"Run not found: {run_id}")

            record.status = status
            record.current_stage = current_stage
            record.updated_at = time.time()
            session.add(record)
            session.commit()
            session.refresh(record)

        return _run_state_from_record(record)

    def write_entry(
        self,
        run_id: str,
        agent_name: str,
        key: str,
        value: dict[str, Any],
    ) -> BlackboardEntry:
        """Write a versioned handoff entry for an agent."""

        with Session(self.engine) as session:
            run = session.get(RunStateRecord, run_id)
            if run is None:
                raise KeyError(f"Run not found: {run_id}")

            latest = session.exec(
                select(BlackboardEntryRecord)
                .where(
                    BlackboardEntryRecord.run_id == run_id,
                    BlackboardEntryRecord.key == key,
                )
                .order_by(BlackboardEntryRecord.version.desc())
            ).first()
            version = 1 if latest is None else latest.version + 1

            record = BlackboardEntryRecord(
                run_id=run_id,
                agent_name=agent_name,
                key=key,
                value=value,
                timestamp=time.time(),
                version=version,
            )
            session.add(record)
            session.commit()
            session.refresh(record)

        return _entry_from_record(record)

    def read_latest(self, run_id: str, key: str) -> BlackboardEntry | None:
        """Read the latest version for a run/key pair."""

        with Session(self.engine) as session:
            record = session.exec(
                select(BlackboardEntryRecord)
                .where(
                    BlackboardEntryRecord.run_id == run_id,
                    BlackboardEntryRecord.key == key,
                )
                .order_by(BlackboardEntryRecord.version.desc())
            ).first()
            if record is None:
                return None
            return _entry_from_record(record)

    def list_entries(self, run_id: str) -> list[BlackboardEntry]:
        """List all entries for a run in write order."""

        with Session(self.engine) as session:
            records = session.exec(
                select(BlackboardEntryRecord)
                .where(BlackboardEntryRecord.run_id == run_id)
                .order_by(BlackboardEntryRecord.timestamp, BlackboardEntryRecord.version)
            ).all()
            return [_entry_from_record(record) for record in records]


def _run_state_from_record(record: RunStateRecord) -> RunState:
    return RunState(
        run_id=record.run_id,
        problem=record.problem,
        domain_tags=list(record.domain_tags),
        status=record.status,  # type: ignore[arg-type]
        current_stage=record.current_stage,
        strategy_id=record.strategy_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        human_notes=list(record.human_notes),
        human_score=record.human_score,
    )


def _entry_from_record(record: BlackboardEntryRecord) -> BlackboardEntry:
    return BlackboardEntry(
        run_id=record.run_id,
        agent_name=record.agent_name,
        key=record.key,
        value=dict(record.value),
        timestamp=record.timestamp,
        version=record.version,
    )
