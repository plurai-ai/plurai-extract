# plurai-extract

Extract documents and records from data sources for Plurai simulation. The package reads
documents, schemas and records from a data source with the caller's own credentials, on the
caller's machine, and emits a deterministic set of chunks plus structural relationships. It
involves no LLM and no embeddings; the output is plain dataclasses that serialize with
`to_dict()`.

## Install

```bash
pip install plurai-extract
```

The core install depends only on `pydantic` and `tiktoken`. Each data-source driver is an
opt-in extra, so you install only what your source needs:

```bash
pip install "plurai-extract[bigquery]"
pip install "plurai-extract[pdf,snowflake]"
```

| Extra | Source | Driver |
|---|---|---|
| `pdf` | PDF directories | docling, pypdfium2 |
| `http` | Azure AI Search | aiohttp |
| `vertex` | Vertex AI Search | google-cloud-discoveryengine |
| `bigquery` | BigQuery | google-cloud-bigquery |
| `redshift` | Redshift | boto3 |
| `azure-sql` | Azure SQL | pyodbc |
| `databricks` | Databricks | databricks-sql-connector |
| `snowflake` | Snowflake | snowflake-connector-python |
| `bedrock` | Bedrock Knowledge Base | langchain-aws |
| `all` | every extra above | |

SQLite, JSON and JSON Lines records, Markdown and text directories, and the LangChain and
LlamaIndex vector-store sources need no extra.

Python 3.11, 3.12 and 3.13 are supported.

## Usage

```python
from plurai_extract import load_only
from plurai_extract.sources.directory import DirectorySource

documents, relationships = load_only(DirectorySource(path="./docs"))
payload = [d.to_dict() for d in documents]
```

`load_only` takes a source instance, runs the extraction, and returns
`(documents, relationships)`. Source classes are importable from `plurai_extract.sources`
(or from their own module, as above).

## Issues and release notes

See [CONTRIBUTING.md](https://github.com/plurai-ai/plurai-extract/blob/main/CONTRIBUTING.md) for how to report issues, and
[CHANGELOG.md](https://github.com/plurai-ai/plurai-extract/blob/main/CHANGELOG.md) for release notes.
