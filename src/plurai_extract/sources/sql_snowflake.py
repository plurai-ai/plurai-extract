"""SnowflakeSource — introspects Snowflake schemas as Documents."""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from plurai_extract.sources.base import Document, InitialRelationship
from plurai_extract.sources.sql_common import schema_to_documents

logger = logging.getLogger(__name__)


@dataclass
class SnowflakeSource:
    """Fetch table schemas and sample data from Snowflake.

    Uses ``snowflake-connector-python`` for connectivity.

    Usage::

        source = SnowflakeSource(
            account="xy12345.us-east-1",
            user="myuser",
            password="...",
            warehouse="COMPUTE_WH",
            database="MYDB",
        )
        documents, relationships = load_only(source)
    """

    account: str
    user: str
    password: str = ""
    authenticator: str = ""  # "externalbrowser" for SSO
    warehouse: str = ""
    database: str = ""
    schema_name: str = "PUBLIC"
    role: str = ""
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
        import snowflake.connector

        connect_kwargs: dict[str, t.Any] = {
            "account": self.account,
            "user": self.user,
        }
        if self.password:
            connect_kwargs["password"] = self.password
        if self.authenticator:
            connect_kwargs["authenticator"] = self.authenticator
        if self.warehouse:
            connect_kwargs["warehouse"] = self.warehouse
        if self.database:
            connect_kwargs["database"] = self.database
        if self.role:
            connect_kwargs["role"] = self.role

        try:
            conn = snowflake.connector.connect(**connect_kwargs)
        except Exception as exc:
            logger.error("Failed to connect to Snowflake: %s", exc)
            raise RuntimeError(f"Failed to connect to Snowflake: {exc}") from exc

        try:
            schema = self._introspect(conn)
        except Exception as exc:
            conn.close()
            logger.error("Failed to introspect Snowflake: %s", exc)
            raise RuntimeError(f"Failed to introspect Snowflake: {exc}") from exc

        db_id = self.database or "snowflake"
        docs, rels = schema_to_documents(
            db_id,
            schema,
            fetch_sample_fn=lambda tbl, n: self._fetch_samples(conn, tbl, n),
        )
        conn.close()

        self._relationships = rels
        logger.info(
            "Loaded %d table documents from Snowflake (%d FK relationships)",
            len(docs),
            len(rels),
        )
        return docs

    def _introspect(self, conn: t.Any) -> dict:
        cursor = conn.cursor()

        # Tables
        cursor.execute(
            f"SELECT TABLE_NAME FROM {self.database}.INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{self.schema_name}' AND TABLE_TYPE = 'BASE TABLE' "
            f"ORDER BY TABLE_NAME"
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
            f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
            f"FROM {self.database}.INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{self.schema_name}' "
            f"ORDER BY TABLE_NAME, ORDINAL_POSITION"
        )
        for table_name, col_name, col_type in cursor.fetchall():
            if table_name not in table_names:
                continue
            t_idx = table_names.index(table_name)
            column_names.append([t_idx, col_name])
            column_types.append(col_type.lower())
            global_idx += 1

        # Primary keys
        try:
            cursor.execute(
                f"SELECT kcu.TABLE_NAME, kcu.COLUMN_NAME "
                f"FROM {self.database}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc "
                f"JOIN {self.database}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu "
                f"  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME "
                f"WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY' "
                f"  AND tc.TABLE_SCHEMA = '{self.schema_name}'"
            )
            for table_name, col_name in cursor.fetchall():
                if table_name in table_names:
                    t_idx = table_names.index(table_name)
                    for i, (ti, cn) in enumerate(column_names):
                        if ti == t_idx and cn == col_name:
                            primary_keys.append(i)
                            break
        except Exception:
            pass

        # Foreign keys
        try:
            cursor.execute(
                f"SELECT "
                f"  fk.FK_TABLE_NAME, fk.FK_COLUMN_NAME, "
                f"  fk.PK_TABLE_NAME, fk.PK_COLUMN_NAME "
                f"FROM {self.database}.INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc "
                f"JOIN TABLE(RESULT_SCAN(LAST_QUERY_ID())) fk "
                f"  ON TRUE "
                f"WHERE rc.CONSTRAINT_SCHEMA = '{self.schema_name}'"
            )
        except Exception:
            # Fallback: use SHOW IMPORTED KEYS
            try:
                for tbl in table_names:
                    cursor.execute(f"SHOW IMPORTED KEYS IN TABLE {self.database}.{self.schema_name}.{tbl}")
                    for row in cursor.fetchall():
                        # row format varies; typically: pk_db, pk_schema, pk_table, pk_col, fk_db, fk_schema, fk_table, fk_col, ...
                        if len(row) >= 8:
                            pk_table, pk_col = row[2], row[3]
                            fk_table, fk_col = row[6], row[7]
                            from_t = table_names.index(fk_table) if fk_table in table_names else -1
                            to_t = table_names.index(pk_table) if pk_table in table_names else -1
                            if from_t >= 0 and to_t >= 0:
                                from_gidx = next((i for i, (ti, cn) in enumerate(column_names) if ti == from_t and cn == fk_col), -1)
                                to_gidx = next((i for i, (ti, cn) in enumerate(column_names) if ti == to_t and cn == pk_col), -1)
                                if from_gidx > 0 and to_gidx > 0:
                                    foreign_keys.append([from_gidx, to_gidx])
            except Exception:
                pass

        # Sample rows
        for tbl in table_names:
            try:
                cursor.execute(f'SELECT * FROM "{self.database}"."{self.schema_name}"."{tbl}" LIMIT {self.sample_rows}')
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
            cursor.execute(f'SELECT * FROM "{self.database}"."{self.schema_name}"."{table_name}" LIMIT {max_rows}')
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception:
            return []
