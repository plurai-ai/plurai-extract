"""RedshiftSource — introspects Amazon Redshift schemas as Documents."""

from __future__ import annotations

import logging
import time
import typing as t
from dataclasses import dataclass, field

from plurai_extract.sources.base import Document, InitialRelationship
from plurai_extract.sources.sql_common import schema_to_documents

logger = logging.getLogger(__name__)


@dataclass
class RedshiftSource:
    """Fetch table schemas and sample data from Amazon Redshift.

    Uses the Redshift Data API (``boto3`` ``redshift-data`` client) so no
    direct database connection or VPC access is required.

    Usage::

        source = RedshiftSource(
            cluster_identifier="my-cluster",
            database="mydb",
            db_user="admin",
        )
        documents, relationships = load_only(source)
    """

    cluster_identifier: str = ""
    workgroup_name: str = ""  # For Redshift Serverless
    database: str = "dev"
    db_user: str = ""
    schema_name: str = "public"
    region_name: str = "us-east-1"
    sample_rows: int = 3
    _poll_interval: float = 0.5
    _relationships: list[InitialRelationship] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def initial_relationships(self) -> list[InitialRelationship]:
        return self._relationships

    def load(self, **kwargs: t.Any) -> list[Document]:
        import boto3

        client = boto3.client("redshift-data", region_name=self.region_name)

        try:
            schema = self._introspect(client)
        except Exception as exc:
            logger.error("Failed to introspect Redshift: %s", exc)
            raise RuntimeError(f"Failed to introspect Redshift: {exc}") from exc

        db_id = self.cluster_identifier or self.workgroup_name or "redshift"
        docs, rels = schema_to_documents(
            db_id,
            schema,
            fetch_sample_fn=lambda tbl, n: self._fetch_samples(client, tbl, n),
        )
        self._relationships = rels
        logger.info(
            "Loaded %d table documents from Redshift (%d FK relationships)",
            len(docs),
            len(rels),
        )
        return docs

    # ── Helpers ──────────────────────────────────────────────────────────

    def _exec_params(self) -> dict[str, t.Any]:
        """Build common parameters for execute_statement / describe_table."""
        params: dict[str, t.Any] = {"Database": self.database}
        if self.workgroup_name:
            params["WorkgroupName"] = self.workgroup_name
        else:
            params["ClusterIdentifier"] = self.cluster_identifier
            if self.db_user:
                params["DbUser"] = self.db_user
        return params

    def _run_sql(self, client: t.Any, sql: str) -> list[list]:
        """Execute SQL via Data API, poll for completion, return rows."""
        resp = client.execute_statement(Sql=sql, **self._exec_params())
        stmt_id = resp["Id"]

        while True:
            status = client.describe_statement(Id=stmt_id)
            if status["Status"] in ("FINISHED", "FAILED", "ABORTED"):
                break
            time.sleep(self._poll_interval)

        if status["Status"] != "FINISHED":
            raise RuntimeError(f"Redshift query failed: {status.get('Error', 'unknown')}")

        result = client.get_statement_result(Id=stmt_id)
        rows = []
        for record in result.get("Records", []):
            rows.append([list(field.values())[0] if field else None for field in record])
        return rows

    def _introspect(self, client: t.Any) -> dict:
        # List tables
        table_rows = self._run_sql(
            client,
            f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = '{self.schema_name}' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """,
        )
        table_names = [r[0] for r in table_rows]

        column_names: list[list] = [[-1, "*"]]
        column_types: list[str] = ["text"]
        primary_keys: list[int] = []
        foreign_keys: list[list[int]] = []
        sample_rows_map: dict[str, list[dict]] = {}
        global_idx = 1

        # Columns per table
        col_rows = self._run_sql(
            client,
            f"""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{self.schema_name}'
            ORDER BY table_name, ordinal_position
        """,
        )
        for table_name, col_name, col_type in col_rows:
            if table_name not in table_names:
                continue
            t_idx = table_names.index(table_name)
            column_names.append([t_idx, col_name])
            column_types.append(col_type.lower())
            global_idx += 1

        # Primary keys
        pk_rows = self._run_sql(
            client,
            f"""
            SELECT kcu.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = '{self.schema_name}'
        """,
        )
        for table_name, col_name in pk_rows:
            if table_name in table_names:
                t_idx = table_names.index(table_name)
                for i, (ti, cn) in enumerate(column_names):
                    if ti == t_idx and cn == col_name:
                        primary_keys.append(i)
                        break

        # Foreign keys
        fk_rows = self._run_sql(
            client,
            f"""
            SELECT
                kcu.table_name AS from_table, kcu.column_name AS from_col,
                ccu.table_name AS to_table, ccu.column_name AS to_col
            FROM information_schema.referential_constraints rc
            JOIN information_schema.key_column_usage kcu
                ON rc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON rc.unique_constraint_name = ccu.constraint_name
            WHERE kcu.table_schema = '{self.schema_name}'
        """,
        )
        for from_table, from_col, to_table, to_col in fk_rows:
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
                rows = self._run_sql(client, f'SELECT * FROM "{self.schema_name}"."{tbl}" LIMIT {self.sample_rows}')
                # Get column names for this table
                cols = [cn for ti, cn in column_names if ti == table_names.index(tbl)]
                sample_rows_map[tbl] = [dict(zip(cols, r)) for r in rows]
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

    def _fetch_samples(self, client: t.Any, table_name: str, max_rows: int) -> list[dict]:
        try:
            rows = self._run_sql(client, f'SELECT * FROM "{self.schema_name}"."{table_name}" LIMIT {max_rows}')
            table_names = [
                r[0]
                for r in self._run_sql(
                    client,
                    f"""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = '{self.schema_name}' AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """,
                )
            ]
            return [dict(zip(table_names, r)) for r in rows]
        except Exception:
            return []
