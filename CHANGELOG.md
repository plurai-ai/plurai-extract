# Changelog

All notable changes to this project are documented in this file.

## 1.0.2

- **License.** The package metadata and `LICENSE` declare the package as proprietary software under the terms in `LICENSE`. Version 1.0.1 declares no license.

## 1.0.1

First public release.

- `plurai-extract` extracts documents and records from data sources for Plurai simulation, with the caller's own
  credentials on the caller's machine. It involves no LLM and no embeddings: it returns deterministic chunks plus
  structural relationships as plain dataclasses that serialize with `to_dict()`.
- Install with `pip install plurai-extract`. The import package is `plurai_extract`:
  `plurai_extract.loader.load_only` pulls documents and relationships from a source, `plurai_extract.sources.*` holds the
  source adapters, and `plurai_extract.schema_types.ColumnInfo` describes SQL columns.
- Sources: local directories (PDF, Markdown, plain text), JSON and JSON Lines records, cloud search services, SQL
  databases, and LangChain and LlamaIndex vector stores and document loaders.
- The core install depends only on `pydantic` and `tiktoken`. Each source driver is an opt-in extra: `pdf`, `http`,
  `vertex`, `bigquery`, `redshift`, `azure-sql`, `databricks`, `snowflake`, `bedrock`, and `all` for every driver at once.
- Python 3.11, 3.12 and 3.13 are supported.
- Extracted PDF pages are cached under `$XDG_CACHE_HOME/plurai-extract` (`~/.cache/plurai-extract` when it is unset);
  `PLURAI_EXTRACT_CACHE_DIR` overrides the location.
