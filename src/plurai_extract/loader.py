"""Source loading: the deterministic extraction seam (no LLM, no indexing).

``load_only`` is the only step that needs the source's data, credentials and network, so
it runs on the machine that owns them. It returns plain ``(documents, relationships)``
dataclasses (each with ``to_dict()``) that can be serialized and handed to the Plurai
simulation backend, which runs on a different machine.

This module imports only ``plurai_extract.sources``, so the distribution stays free of
LLM dependencies.
"""

from __future__ import annotations

import logging

from plurai_extract.sources.base import DataSource, Document, InitialRelationship

logger = logging.getLogger(__name__)


def load_only(
    source: DataSource,
    initial_relationships: list[InitialRelationship] | None = None,
    **source_kwargs,
) -> tuple[list[Document], list[InitialRelationship]]:
    """Pull documents (+ structural relationships) from a source, with no
    indexing, embeddings, partitioning, or LLM.

    This is the extraction seam: the only step that needs access to the source's
    data, credentials and network. The returned ``(documents, relationships)`` are
    plain dataclasses with ``to_dict()`` and can be serialized and handed to another
    machine.
    """
    logger.info("Loading data from source")
    try:
        documents = source.load(**source_kwargs)
    except ImportError:
        # A missing per-source DRIVER (drivers are imported lazily inside
        # load()) must stay an ImportError so the caller can turn it into an
        # actionable "install the driver extra" message — wrapping it in
        # RuntimeError here would hide both the type and the missing module name.
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load data from source: {exc}") from exc

    rels = list(initial_relationships or [])
    # SQL sources populate ``initial_relationships`` (FK edges) during load().
    if hasattr(source, "initial_relationships"):
        rels.extend(source.initial_relationships)

    logger.info("Loaded %d documents", len(documents))
    return documents, rels
