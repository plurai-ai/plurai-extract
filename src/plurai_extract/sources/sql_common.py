"""Shared helpers for SQL data source adapters.

All SQL sources (SQLite, BigQuery, Redshift, etc.) produce a normalized schema
dict and call ``schema_to_documents`` to convert it into Documents with
per-column ColumnInfo metadata.
"""

from __future__ import annotations

import logging
import typing as t

from plurai_extract.sources.base import Document, InitialRelationship

logger = logging.getLogger(__name__)


def find_col_idx(column_names: list[list], table_idx: int, col_name: str) -> int:
    """Find the global index of a column in the column_names list."""
    for i, (t_idx, name) in enumerate(column_names):
        if t_idx == table_idx and name == col_name:
            return i
    return -1


def schema_to_documents(
    db_id: str,
    schema: dict,
    fetch_sample_fn: t.Callable[[str, int], list[dict]] | None = None,
) -> tuple[list[Document], list[InitialRelationship]]:
    """Convert a normalized schema dict into Documents and FK relationships.

    Parameters
    ----------
    db_id
        Database identifier (used as ``Document.name``).
    schema
        Normalized schema dict with keys: ``table_names``, ``column_names``,
        ``column_types``, ``primary_keys``, ``foreign_keys``, ``sample_rows``.
    fetch_sample_fn
        Optional callback ``(table_name, max_rows) -> list[dict]`` to lazily
        fetch sample rows.  Used when ``schema["sample_rows"]`` is empty.
    """
    from plurai_extract.schema_types import ColumnInfo

    table_names = schema["table_names"]
    documents: list[Document] = []
    relationships: list[InitialRelationship] = []

    for t_idx, table_name in enumerate(table_names):
        sample_rows = schema.get("sample_rows", {}).get(table_name, [])
        if not sample_rows and fetch_sample_fn is not None:
            sample_rows = fetch_sample_fn(table_name, 3)

        # Build schema text block
        col_lines = []
        for global_idx, (ti, col_name) in enumerate(schema["column_names"]):
            if ti != t_idx:
                continue
            col_type = schema["column_types"][global_idx] if global_idx < len(schema["column_types"]) else "text"
            tags = []
            if global_idx in schema.get("primary_keys", []):
                tags.append("PK")
            if global_idx in {fk[0] for fk in schema.get("foreign_keys", [])}:
                tags.append("FK")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            col_lines.append(f"  - {col_name} ({col_type}){tag_str}")

        schema_block = "\n".join(col_lines)
        sample_block = ""
        if sample_rows:
            sample_block = "\n\nSample rows:\n" + "\n".join(f"  {r}" for r in sample_rows[:3])

        text = f"[Database: {db_id} | Table: {table_name} | Table {t_idx + 1} of {len(table_names)}]\n\nSchema:\n{schema_block}{sample_block}"

        # Build per-column ColumnInfo objects
        columns = []
        fk_map: dict[int, int] = {fk[0]: fk[1] for fk in schema.get("foreign_keys", [])}
        for global_idx, (ti, col_name) in enumerate(schema["column_names"]):
            if ti != t_idx:
                continue
            is_pk = global_idx in schema.get("primary_keys", [])
            is_fk = global_idx in fk_map
            fk_ref = None
            if is_fk:
                ref_idx = fk_map[global_idx]
                ref_t_idx, ref_col = schema["column_names"][ref_idx]
                if 0 <= ref_t_idx < len(table_names):
                    fk_ref = f"{table_names[ref_t_idx]}.{ref_col}"
            columns.append(
                ColumnInfo(
                    name=col_name,
                    type=(schema["column_types"][global_idx] if global_idx < len(schema["column_types"]) else "text"),
                    is_pk=is_pk,
                    is_fk=is_fk,
                    fk_ref=fk_ref,
                )
            )

        # FK relationship descriptions
        fk_descs = []
        for global_idx, (ti, col_name) in enumerate(schema["column_names"]):
            if ti != t_idx or global_idx not in fk_map:
                continue
            ref_idx = fk_map[global_idx]
            ref_t_idx, ref_col = schema["column_names"][ref_idx]
            if 0 <= ref_t_idx < len(table_names):
                fk_descs.append(f"{table_name}.{col_name} → {table_names[ref_t_idx]}.{ref_col}")

        raw_schema = {
            "db_id": db_id,
            "table_name": table_name,
            "columns": columns,
            "sample_rows": sample_rows,
            "db_table_names": table_names,
            "fk_relationships": fk_descs,
            "_table_idx": t_idx,
            "_total_tables": len(table_names),
        }

        documents.append(
            Document(
                name=db_id,
                text=text,
                chunk_idx=t_idx,
                metadata={"table_name": table_name, "raw_schema": raw_schema},
            )
        )

    # FK relationships as InitialRelationship objects
    col_names = schema["column_names"]
    for fk_col_idx, ref_col_idx in schema.get("foreign_keys", []):
        if fk_col_idx >= len(col_names) or ref_col_idx >= len(col_names):
            continue
        fk_t_idx = col_names[fk_col_idx][0]
        ref_t_idx = col_names[ref_col_idx][0]
        if fk_t_idx < 0 or ref_t_idx < 0:
            continue
        relationships.append(
            InitialRelationship(
                source_doc_name=db_id,
                source_chunk_idx=fk_t_idx,
                target_doc_name=db_id,
                target_chunk_idx=ref_t_idx,
                type="foreign_key",
                properties={"source": "schema"},
            )
        )

    return documents, relationships
