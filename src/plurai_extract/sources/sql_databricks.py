"""DatabricksSource — introspects Databricks Unity Catalog schemas as Documents."""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from plurai_extract.sources.base import Document, InitialRelationship
from plurai_extract.sources.sql_common import schema_to_documents

logger = logging.getLogger(__name__)


@dataclass
class DatabricksSource:
    """Fetch table schemas and sample data from Databricks.

    Uses the ``databricks-sql-connector`` package.  Requires a SQL warehouse
    or cluster HTTP path.

    Usage::

        source = DatabricksSource(
            server_hostname="adb-123.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/abc123",
            access_token="dapi...",
            catalog="main",
            schema_name="default",
        )
        documents, relationships = load_only(source)
    """

    server_hostname: str
    http_path: str
    access_token: str = ""
    catalog: str = "main"
    schema_name: str = "default"
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
        from databricks import sql as dbsql

        try:
            conn = dbsql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token,
            )
        except Exception as exc:
            logger.error("Failed to connect to Databricks: %s", exc)
            raise RuntimeError(f"Failed to connect to Databricks: {exc}") from exc

        try:
            schema = self._introspect(conn)
        except Exception as exc:
            conn.close()
            logger.error("Failed to introspect Databricks: %s", exc)
            raise RuntimeError(f"Failed to introspect Databricks: {exc}") from exc

        db_id = f"{self.catalog}.{self.schema_name}"
        docs, rels = schema_to_documents(
            db_id,
            schema,
            fetch_sample_fn=lambda tbl, n: self._fetch_samples(conn, tbl, n),
        )
        conn.close()

        self._relationships = rels
        logger.info(
            "Loaded %d table documents from Databricks (%d FK relationships)",
            len(docs),
            len(rels),
        )
        return docs

    def _introspect(self, conn: t.Any) -> dict:
        cursor = conn.cursor()

        # List tables
        cursor.execute(
            f"SELECT table_name FROM {self.catalog}.information_schema.tables "
            f"WHERE table_schema = '{self.schema_name}' AND table_type = 'MANAGED' "
            f"ORDER BY table_name"
        )
        table_names = [row[0] for row in cursor.fetchall()]

        # Also include EXTERNAL tables
        cursor.execute(
            f"SELECT table_name FROM {self.catalog}.information_schema.tables "
            f"WHERE table_schema = '{self.schema_name}' AND table_type = 'EXTERNAL' "
            f"ORDER BY table_name"
        )
        table_names.extend(row[0] for row in cursor.fetchall())
        table_names.sort()

        column_names: list[list] = [[-1, "*"]]
        column_types: list[str] = ["text"]
        primary_keys: list[int] = []
        foreign_keys: list[list[int]] = []
        sample_rows_map: dict[str, list[dict]] = {}
        global_idx = 1

        # Columns
        cursor.execute(
            f"SELECT table_name, column_name, data_type "
            f"FROM {self.catalog}.information_schema.columns "
            f"WHERE table_schema = '{self.schema_name}' "
            f"ORDER BY table_name, ordinal_position"
        )
        for table_name, col_name, col_type in cursor.fetchall():
            if table_name not in table_names:
                continue
            t_idx = table_names.index(table_name)
            column_names.append([t_idx, col_name])
            column_types.append(col_type.lower())
            global_idx += 1

        # Primary keys (Unity Catalog)
        try:
            cursor.execute(
                f"SELECT kcu.table_name, kcu.column_name "
                f"FROM {self.catalog}.information_schema.table_constraints tc "
                f"JOIN {self.catalog}.information_schema.key_column_usage kcu "
                f"  ON tc.constraint_name = kcu.constraint_name "
                f"WHERE tc.constraint_type = 'PRIMARY KEY' "
                f"  AND tc.table_schema = '{self.schema_name}'"
            )
            for table_name, col_name in cursor.fetchall():
                if table_name in table_names:
                    t_idx = table_names.index(table_name)
                    for i, (ti, cn) in enumerate(column_names):
                        if ti == t_idx and cn == col_name:
                            primary_keys.append(i)
                            break
        except Exception:
            pass  # Constraints may not be available

        # Foreign keys (Unity Catalog)
        try:
            cursor.execute(
                f"SELECT "
                f"  kcu.table_name, kcu.column_name, "
                f"  ccu.table_name, ccu.column_name "
                f"FROM {self.catalog}.information_schema.referential_constraints rc "
                f"JOIN {self.catalog}.information_schema.key_column_usage kcu "
                f"  ON rc.constraint_name = kcu.constraint_name "
                f"JOIN {self.catalog}.information_schema.constraint_column_usage ccu "
                f"  ON rc.unique_constraint_name = ccu.constraint_name "
                f"WHERE kcu.table_schema = '{self.schema_name}'"
            )
            for from_table, from_col, to_table, to_col in cursor.fetchall():
                from_t = table_names.index(from_table) if from_table in table_names else -1
                to_t = table_names.index(to_table) if to_table in table_names else -1
                if from_t >= 0 and to_t >= 0:
                    from_gidx = next((i for i, (ti, cn) in enumerate(column_names) if ti == from_t and cn == from_col), -1)
                    to_gidx = next((i for i, (ti, cn) in enumerate(column_names) if ti == to_t and cn == to_col), -1)
                    if from_gidx > 0 and to_gidx > 0:
                        foreign_keys.append([from_gidx, to_gidx])
        except Exception:
            pass

        # Sample rows
        fqn = f"`{self.catalog}`.`{self.schema_name}`"
        for tbl in table_names:
            try:
                cursor.execute(f"SELECT * FROM {fqn}.`{tbl}` LIMIT {self.sample_rows}")
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
            fqn = f"`{self.catalog}`.`{self.schema_name}`.`{table_name}`"
            cursor.execute(f"SELECT * FROM {fqn} LIMIT {max_rows}")
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception:
            return []
