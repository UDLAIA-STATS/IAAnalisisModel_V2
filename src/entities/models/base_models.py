from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Field, SQLModel


class NumericIdModel(SQLModel):
    id: int = Field(primary_key=True, index=True, default=None, sa_column_kwargs={"autoincrement": True})


class UUIDModel(SQLModel):
    id: str = Field(primary_key=True, index=True, default_factory=lambda: str(uuid4()))


class AuditTable(SQLModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": datetime.now(timezone.utc)},
    )


class AuditTableCompletedTable(AuditTable, table=True):
    completed_at: datetime | None = Field(default=None, nullable=True)
