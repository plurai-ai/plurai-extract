"""SQLiteSource — extracts schemas from SQLite databases as Documents."""

from __future__ import annotations

import logging
import sqlite3
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from plurai_extract.sources.base import Document, InitialRelationship
from plurai_extract.sources.sql_common import find_col_idx, schema_to_documents

logger = logging.getLogger(__name__)


def _fetch_sample_rows(db_path: Path, table_name: str, max_rows: int = 3) -> list[dict]:
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        # table_name comes from sqlite_master introspection, but escape to be safe
        safe_name = table_name.replace('"', '""')
        cursor = conn.execute(f'SELECT * FROM "{safe_name}" LIMIT {max_rows}')
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        logger.warning("Failed to read sample rows from %s.%s: %s", db_path.stem, table_name, exc)
        return []


def _introspect_sqlite(db_path: Path) -> dict:
    """Introspect a SQLite database and return a normalized schema dict."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

    # Build column info
    column_names: list[list] = [[-1, "*"]]
    column_types: list[str] = ["text"]
    primary_keys: list[int] = []
    global_idx = 1

    for t_idx, table_name in enumerate(table_names):
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        for col in cursor.fetchall():
            # col: (cid, name, type, notnull, dflt_value, pk)
            column_names.append([t_idx, col[1]])
            column_types.append(col[2].lower() if col[2] else "text")
            if col[5]:  # pk flag
                primary_keys.append(global_idx)
            global_idx += 1

    # Get foreign keys
    foreign_keys: list[list[int]] = []
    for t_idx, table_name in enumerate(table_names):
        cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
        for fk in cursor.fetchall():
            # fk: (id, seq, table, from, to, ...)
            ref_table = fk[2]
            from_col = fk[3]
            to_col = fk[4]
            # Find global indices
            from_gidx = find_col_idx(column_names, t_idx, from_col)
            ref_t_idx = table_names.index(ref_table) if ref_table in table_names else -1
            to_gidx = find_col_idx(column_names, ref_t_idx, to_col) if ref_t_idx >= 0 else -1
            if from_gidx > 0 and to_gidx > 0:
                foreign_keys.append([from_gidx, to_gidx])

    conn.close()

    return {
        "table_names": table_names,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "db_path": db_path,
        "sample_rows": {t: _fetch_sample_rows(db_path, t) for t in table_names},
    }


@dataclass
class SQLiteSource:
    """Extracts schemas from SQLite databases as Documents.

    Each database table becomes one Document. Foreign key constraints
    become InitialRelationship objects.

    Parameters
    ----------
    db_paths : list of Path, optional
        Explicit list of SQLite database files.
    db_dir : str or Path, optional
        Directory to scan for ``*.sqlite`` files.
    sample_rows : int
        Number of sample rows to fetch per table (default 3).

    Example
    -------
    ::

        source = SQLiteSource(db_dir="/data/databases/")
        docs = source.load()
    """

    db_paths: list[Path] = field(default_factory=list)
    db_dir: str | Path | None = None
    sample_rows: int = 3
    _relationships: list[InitialRelationship] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def initial_relationships(self) -> list[InitialRelationship]:
        """FK relationships discovered during load(). Available after calling load()."""
        return self._relationships

    def load(self, **kwargs: t.Any) -> list[Document]:
        paths = list(self.db_paths)
        if self.db_dir:
            db_dir = Path(self.db_dir)
            paths.extend(sorted(db_dir.glob("**/*.sqlite")))

        if not paths:
            raise FileNotFoundError("No SQLite databases found. Provide db_paths or db_dir.")

        all_docs: list[Document] = []
        all_rels: list[InitialRelationship] = []

        for db_path in paths:
            db_id = db_path.stem
            schema = _introspect_sqlite(db_path)
            docs, rels = schema_to_documents(db_id, schema)
            all_docs.extend(docs)
            all_rels.extend(rels)

        self._relationships = all_rels
        logger.info("Loaded %d table documents from %d databases (%d FK relationships)", len(all_docs), len(paths), len(all_rels))
        return all_docs


# Private aliases of the shared SQL helpers, for callers that import them from this module.
_find_col_idx = find_col_idx
_schema_to_documents = schema_to_documents
