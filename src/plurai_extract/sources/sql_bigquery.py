"""BigQuerySource — introspects Google BigQuery datasets as Documents."""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from plurai_extract.sources.base import Document, InitialRelationship
from plurai_extract.sources.sql_common import schema_to_documents

logger = logging.getLogger(__name__)


@dataclass
class BigQuerySource:
    """Fetch table schemas and sample data from Google BigQuery.

    Each table becomes one Document with per-column ColumnInfo
    metadata.  Foreign key constraints (if declared) become
    InitialRelationship objects.

    Usage::

        source = BigQuerySource(project="my-gcp-project", dataset="my_dataset")
        documents, relationships = load_only(source)
    """

    project: str
    dataset: str | None = None
    credentials: t.Any = None  # google.oauth2.credentials.Credentials
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
        from google.cloud import bigquery

        client = bigquery.Client(
            project=self.project,
            credentials=self.credentials,
        )

        datasets = [self.dataset] if self.dataset else [ds.dataset_id for ds in client.list_datasets()]

        all_docs: list[Document] = []
        all_rels: list[InitialRelationship] = []

        for ds_id in datasets:
            try:
                schema = self._introspect_dataset(client, ds_id)
            except Exception as exc:
                logger.error("Failed to introspect BigQuery dataset %s: %s", ds_id, exc)
                raise RuntimeError(f"Failed to introspect BigQuery dataset {ds_id!r}: {exc}") from exc

            db_id = f"{self.project}.{ds_id}"
            docs, rels = schema_to_documents(
                db_id,
                schema,
                fetch_sample_fn=lambda tbl, n: self._fetch_samples(client, ds_id, tbl, n),
            )
            all_docs.extend(docs)
            all_rels.extend(rels)

        self._relationships = all_rels
        logger.info(
            "Loaded %d table documents from BigQuery (%d FK relationships)",
            len(all_docs),
            len(all_rels),
        )
        return all_docs

    def _introspect_dataset(self, client: t.Any, dataset_id: str) -> dict:
        dataset_ref = f"{self.project}.{dataset_id}"
        tables = list(client.list_tables(dataset_ref))

        table_names: list[str] = []
        column_names: list[list] = [[-1, "*"]]
        column_types: list[str] = ["text"]
        primary_keys: list[int] = []
        foreign_keys: list[list[int]] = []
        sample_rows_map: dict[str, list[dict]] = {}
        global_idx = 1

        for t_idx, table_item in enumerate(tables):
            table = client.get_table(table_item.reference)
            table_names.append(table.table_id)

            for sf in table.schema:
                column_names.append([t_idx, sf.name])
                column_types.append(sf.field_type.lower())
                global_idx += 1

            # Sample rows
            try:
                rows_iter = client.list_rows(table, max_results=self.sample_rows)
                sample_rows_map[table.table_id] = [dict(row) for row in rows_iter]
            except Exception:
                sample_rows_map[table.table_id] = []

        # BigQuery FK constraints (if declared)
        try:
            fk_query = f"""
                SELECT
                    tc.table_name AS from_table,
                    kcu.column_name AS from_col,
                    ccu.table_name AS to_table,
                    ccu.column_name AS to_col
                FROM `{dataset_ref}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` tc
                JOIN `{dataset_ref}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN `{dataset_ref}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE` ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
            """
            for row in client.query(fk_query).result():
                from_t = table_names.index(row.from_table) if row.from_table in table_names else -1
                to_t = table_names.index(row.to_table) if row.to_table in table_names else -1
                if from_t >= 0 and to_t >= 0:
                    from_gidx = self._find_col(column_names, from_t, row.from_col)
                    to_gidx = self._find_col(column_names, to_t, row.to_col)
                    if from_gidx > 0 and to_gidx > 0:
                        foreign_keys.append([from_gidx, to_gidx])
        except Exception:
            pass  # FKs are optional in BigQuery

        return {
            "table_names": table_names,
            "column_names": column_names,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "db_path": None,
            "sample_rows": sample_rows_map,
        }

    @staticmethod
    def _find_col(column_names: list[list], table_idx: int, col_name: str) -> int:
        for i, (t_idx, name) in enumerate(column_names):
            if t_idx == table_idx and name == col_name:
                return i
        return -1

    @staticmethod
    def _fetch_samples(client: t.Any, dataset_id: str, table_name: str, max_rows: int) -> list[dict]:
        try:
            table_ref = f"{client.project}.{dataset_id}.{table_name}"
            rows = client.list_rows(table_ref, max_results=max_rows)
            return [dict(r) for r in rows]
        except Exception:
            return []
