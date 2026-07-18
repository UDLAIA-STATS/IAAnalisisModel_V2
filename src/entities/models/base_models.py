from datetime import datetime, timezone
from sqlmodel import func
from uuid import uuid4

from sqlmodel import Field, SQLModel


class NumericIdModel(SQLModel):
    id: int = Field(
        primary_key=True,
        index=True,
        default=None,
        sa_column_kwargs={"autoincrement": True},
    )


class UUIDModel(SQLModel):
    id: str = Field(primary_key=True, index=True, default_factory=lambda: str(uuid4()))


class AuditTable(SQLModel):
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": func.now()},
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": func.now()},
    )


class AuditTableCompletedTable(AuditTable):
    completed_at: datetime | None = Field(default=None, nullable=True)
