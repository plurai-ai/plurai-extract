"""Data source adapters for plurai-extract.

This package ``__init__`` EAGERLY imports every connector, so each per-source
DRIVER (google-cloud-bigquery, snowflake-connector, docling, …) MUST be imported
lazily INSIDE the connector's ``__init__``/``load()`` — never at module scope.
Otherwise importing this package (or any single connector, e.g. via
``importlib.import_module``) would fail for a user who
installed only one driver extra. Keep driver imports lazy.
"""

from plurai_extract.sources.base import (
    DataSource,
    Document,
    InitialRelationship,
    SeedQA,
    chunk_tokens,
    fingerprint_match_seed_qa,
)
from plurai_extract.sources.cloud import (
    AzureAISearchSource,
    BedrockKnowledgeBaseSource,
    LangChainVectorStoreSource,
    LlamaIndexVectorStoreSource,
    VertexAISearchSource,
)
from plurai_extract.sources.json_records import JsonRecordsSource
from plurai_extract.sources.sql import SQLiteSource
from plurai_extract.sources.sql_azure import AzureSQLSource
from plurai_extract.sources.sql_bigquery import BigQuerySource
from plurai_extract.sources.sql_databricks import DatabricksSource
from plurai_extract.sources.sql_redshift import RedshiftSource
from plurai_extract.sources.sql_snowflake import SnowflakeSource

__all__ = [
    "AzureAISearchSource",
    "AzureSQLSource",
    "BedrockKnowledgeBaseSource",
    "BigQuerySource",
    "DatabricksSource",
    "DataSource",
    "Document",
    "InitialRelationship",
    "JsonRecordsSource",
    "LangChainVectorStoreSource",
    "LlamaIndexVectorStoreSource",
    "RedshiftSource",
    "SQLiteSource",
    "SeedQA",
    "SnowflakeSource",
    "VertexAISearchSource",
    "chunk_tokens",
    "fingerprint_match_seed_qa",
]
