"""AzureSQLSource — introspects Azure SQL / SQL Server databases as Documents."""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from plurai_extract.sources.base import Document, InitialRelationship
from plurai_extract.sources.sql_common import schema_to_documents

logger = logging.getLogger(__name__)


@dataclass
class AzureSQLSource:
    """Fetch table schemas and sample data from Azure SQL Database.

    Uses ``pyodbc`` for connectivity.  Requires the ODBC Driver for SQL
    Server to be installed on the host.

    Usage::

        source = AzureSQLSource(
            connection_string=(
                "Driver={ODBC Driver 18 for SQL Server};"
                "Server=myserver.database.windows.net;"
                "Database=mydb;UID=user;PWD=pass;Encrypt=yes"
            ),
        )
        documents, relationships = load_only(source)
    """

    connection_string: str
    schema_name: str = "dbo"
    sample_rows: int = 3
    _relationships: list[InitialRelationship] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def initial_relationships(self) -> list[InitialRelationship]:
        return self._relationships

    def load(self, **kwargs: t.Any) -> list[Document]:
        import pyodbc

        try:
            conn = pyodbc.connect(self.connection_string)
        except Exception as exc:
            logger.error("Failed to connect to Azure SQL: %s", exc)
            raise RuntimeError(f"Failed to connect to Azure SQL: {exc}") from exc

        try:
            schema = self._introspect(conn)
        except Exception as exc:
            conn.close()
            logger.error("Failed to introspect Azure SQL: %s", exc)
            raise RuntimeError(f"Failed to introspect Azure SQL: {exc}") from exc

        db_id = self._extract_db_name()
        docs, rels = schema_to_documents(
            db_id,
            schema,
            fetch_sample_fn=lambda tbl, n: self._fetch_samples(conn, tbl, n),
        )
        conn.close()

        self._relationships = rels
        logger.info(
            "Loaded %d table documents from Azure SQL (%d FK relationships)",
            len(docs),
            len(rels),
        )
        return docs

    def _extract_db_name(self) -> str:
        for part in self.connection_string.split(";"):
            if part.strip().lower().startswith("database="):
                return part.split("=", 1)[1].strip()
        return "azure_sql"

    def _introspect(self, conn: t.Any) -> dict:
        cursor = conn.cursor()

        # Tables
        cursor.execute(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME",
            self.schema_name,
        )
        table_names = [row[0] for row in cursor.fetchall()]

        column_names: list[list] = [[-1, "*"]]
        column_types: list[str] = ["text"]
        primary_keys: list[int] = []
        foreign_keys: list[list[int]] = []
        sample_rows_map: dict[str, list[dict]] = {}
        global_idx = 1

        # Columns
        cursor.execute(
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME, ORDINAL_POSITION",
            self.schema_name,
        )
        for table_name, col_name, col_type in cursor.fetchall():
            if table_name not in table_names:
                continue
            t_idx = table_names.index(table_name)
            column_names.append([t_idx, col_name])
            column_types.append(col_type.lower())
            global_idx += 1

        # Primary keys
        cursor.execute(
            "SELECT kcu.TABLE_NAME, kcu.COLUMN_NAME "
            "FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc "
            "JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu "
            "  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME "
            "WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY' AND tc.TABLE_SCHEMA = ?",
            self.schema_name,
        )
        for table_name, col_name in cursor.fetchall():
            if table_name in table_names:
                t_idx = table_names.index(table_name)
                for i, (ti, cn) in enumerate(column_names):
                    if ti == t_idx and cn == col_name:
                        primary_keys.append(i)
                        break

        # Foreign keys
        cursor.execute(
            "SELECT "
            "  kcu.TABLE_NAME, kcu.COLUMN_NAME, "
            "  kcu2.TABLE_NAME, kcu2.COLUMN_NAME "
            "FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc "
            "JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu "
            "  ON rc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME "
            "JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu2 "
            "  ON rc.UNIQUE_CONSTRAINT_NAME = kcu2.CONSTRAINT_NAME "
            "WHERE kcu.TABLE_SCHEMA = ?",
            self.schema_name,
        )
        for from_table, from_col, to_table, to_col in cursor.fetchall():
            from_t = table_names.index(from_table) if from_table in table_names else -1
            to_t = table_names.index(to_table) if to_table in table_names else -1
            if from_t >= 0 and to_t >= 0:
                from_gidx = next((i for i, (ti, cn) in enumerate(column_names) if ti == from_t and cn == from_col), -1)
                to_gidx = next((i for i, (ti, cn) in enumerate(column_names) if ti == to_t and cn == to_col), -1)
                if from_gidx > 0 and to_gidx > 0:
                    foreign_keys.append([from_gidx, to_gidx])

        # Sample rows
        for tbl in table_names:
            try:
                safe = tbl.replace("'", "''")
                cursor.execute(f"SELECT TOP {self.sample_rows} * FROM [{self.schema_name}].[{safe}]")
                cols = [desc[0] for desc in cursor.description]
                sample_rows_map[tbl] = [dict(zip(cols, row)) for row in cursor.fetchall()]
            except Exception:
                sample_rows_map[tbl] = []

        return {
            "table_names": table_names,
            "column_names": column_names,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "db_path": None,
            "sample_rows": sample_rows_map,
        }

    def _fetch_samples(self, conn: t.Any, table_name: str, max_rows: int) -> list[dict]:
        try:
            cursor = conn.cursor()
            safe = table_name.replace("'", "''")
            cursor.execute(f"SELECT TOP {max_rows} * FROM [{self.schema_name}].[{safe}]")
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception:
            return []
