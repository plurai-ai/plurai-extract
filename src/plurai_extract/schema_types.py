"""Pure schema data types shared by the SQL source connectors.

``ColumnInfo`` lives in its own module so the SQL extraction path can build per-column
metadata without importing any LLM-driven code.
"""

from __future__ import annotations

import typing as t

from pydantic import BaseModel


class ColumnInfo(BaseModel):
    name: str
    type: str
    is_pk: bool = False
    is_fk: bool = False
    fk_ref: t.Optional[str] = None  # e.g. "customers.id"
