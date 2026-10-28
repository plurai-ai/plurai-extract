"""Record-per-chunk ingestion for JSON and JSON Lines catalogs.

A JSON array of records — products, tickets, accounts, listings — is the most common
shape operational data arrives in.
:class:`~plurai_extract.sources.directory.DirectorySource` skips ``.json`` by extension,
and the SQL sources want a database. A hand-written JSON-to-markdown renderer would decide
chunk boundaries itself, and a boundary in the wrong place yields a corpus that cannot
answer a question about one record.

This adapter fixes the boundary by construction: **one record, one chunk**. A record is
already the semantic unit a question is asked about ("what is the price of SKU X"), so
splitting on it means every answer lives whole inside exactly one chunk, and the
``next_section`` relationships come out of the array's own order rather than a regex over
generated headings. Consumers need at least two linked records to form a path, so a record
with no relationships is unreachable.

Each record renders as a markdown section — a heading naming the record, then one bullet
per field. The heading is load-bearing beyond looking tidy: a consumer that anchors on the
first non-empty line of a chunk finds the identifier there, so a summary can cite the
record's id without paraphrasing it.
"""

from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from plurai_extract.sources.base import Document, InitialRelationship

logger = logging.getLogger(__name__)

REL_NEXT_RECORD = "next_section"
"""Reading order between consecutive records.

Deliberately the same relationship type the directory source emits between markdown sections
(``next_section``), not a json-specific spelling: consumers follow relationships by type,
so a new name here would make records invisible to every consumer that already follows
``next_section`` — the relation genuinely is "the next section of this document".
"""

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_LINE_DELIMITED_SUFFIXES = frozenset({".jsonl", ".ndjson"})

# Nested containers are rendered inline rather than recursed into: a record's own fields
# are what a question is asked about, and a deep tree flattened into bullets reads as
# noise that dilutes the fields that matter.
_MAX_INLINE_ITEMS = 20


def _is_scalar(value: t.Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _render_value(value: t.Any) -> str:
    """One field's value as a flat, human-readable string.

    Scalars pass through verbatim — an identifier or price MUST survive byte-identical or
    a consumer matching it verbatim against the source finds no match.
    """
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "true" if value else "false"
    if value is None:
        return ""
    if _is_scalar(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        rendered = [r for r in (_render_value(v) for v in list(value)[:_MAX_INLINE_ITEMS]) if r != ""]
        dropped = len(value) - _MAX_INLINE_ITEMS
        # Say so when items are cut. A silently short list reads to a summarizer exactly
        # like a complete one, so it will answer "no" about an item that is simply not
        # shown — the same invented-absence failure an omitted field would cause.
        return ", ".join(rendered) + (f", … ({dropped} more)" if dropped > 0 else "")
    if isinstance(value, dict):
        # Drop empty-rendered entries, exactly as the list branch above does and for the same
        # reason: `quantity: ; detail: ` reads to a summarizer as "this record HAS a quantity
        # and it is blank", which is the invented-absence failure _render_record's own contract
        # forbids for top-level fields. An all-empty dict renders "" and the caller omits it.
        rendered = [f"{k}: {r}" for k, v in list(value.items())[:_MAX_INLINE_ITEMS] if (r := _render_value(v)) != ""]
        return "; ".join(rendered)
    return str(value)


_STRUCTURAL_METADATA_KEYS = frozenset({"heading_path", "record_index", "doc_name", "chunk_idx", "page_content", "type"})

_MISSING = object()
"""Sentinel for "this field_map key resolved to nothing" — distinct from ``None``, which
is a real JSON value a record can legitimately hold."""


def _resolve_field(record: dict, key: str) -> t.Any:
    """A field_map key's value in this record, honoring dotted paths into nested dicts.

    A literal top-level key always wins, so an undotted key, or a record that genuinely has
    a dot in a key name, resolves as a plain lookup. Otherwise a dotted key walks nested
    DICTS only: list indices are not supported, and a path that hits a non-dict mid-way
    or a missing key yields :data:`_MISSING`, never a raise.
    """
    if key in record:
        return record[key]
    if "." not in key:
        return _MISSING
    node: t.Any = record
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _scalar_fields(record: dict, field_map: dict[str, str], dropped_nested: set[str] | None = None) -> dict[str, t.Any]:
    """The record's scalar fields, keyed for use as chunk metadata.

    Scalars, plus the scalars one level inside a nested OBJECT, which are flattened to
    ``<key>_<subkey>`` automatically (``availability: {status: in_stock}`` →
    ``availability_status``). A LIST has no single value a structured consumer could
    treat as a field, so lists are left to the rendered text and reported via
    ``dropped_nested``. A DOTTED ``field_map`` key (``availability.status``) projects a
    nested scalar under its mapped label and takes precedence over the synthesized name.
    Keys that would collide with the structural metadata (``chunk_idx``, ``heading_path``) are dropped rather than allowed to
    shadow it — a record field called ``chunk_idx`` must not become the chunk index.

    ``dropped_nested`` (when passed) collects the names of keys skipped for resolving to a
    nested container, so the caller can report them ONCE per source. The filter is deliberate,
    but its effect is not visible anywhere: an entity seeded from these properties is simply
    missing the field, indistinguishable from a record that never had it. For example, when
    every product's ``availability`` is a nested object, it never reaches chunk metadata while
    the rendered text still carries it, so the entity has no stock field at all.
    """
    selected = list(field_map) if field_map else list(record)
    out: dict[str, t.Any] = {}
    # TWO PASSES, because a collision can only be judged against a COMPLETE set of claimed names.
    # Pass 1 places every scalar; pass 2 flattens objects into whatever names are left. In a
    # single pass the outcome would depend on key ORDER: a dotted label resolving to
    # `availability_status` would overwrite a value the flatten had already put there, leaving
    # metadata holding one field's value under another field's name with nothing reported, and
    # a wrong value is worse than a missing one.
    containers: list[tuple[str, str, t.Any]] = []
    for key in selected:
        value = _resolve_field(record, key)
        if value is _MISSING:
            continue
        # A dotted path's resolved scalar rides under its mapped LABEL — the dotted path
        # itself is not a usable property name, and the label is what the operator chose
        # the field to be called. Literal keys keep their own name.
        out_key = field_map[key] if key not in record and "." in key else key
        if out_key in _STRUCTURAL_METADATA_KEYS:
            continue
        if _is_scalar(value) and value is not None and value != "":
            out[out_key] = value
        else:
            containers.append((key, out_key, value))
    for key, out_key, value in containers:
        if isinstance(value, dict) and value:
            # Auto-flatten one level. Dropping nested objects would be a policy no consumer
            # imposes: the backend keeps chunk metadata unfiltered and seeds entity fields by
            # name, so nested values must arrive as flat properties. The scalars inside a
            # one-level object are projected under ``<key>_<subkey>``, the same shape a dotted
            # field_map produces, without requiring the operator to edit their export first.
            #
            # One level only, and scalars only: deeper nesting has no obvious flat name and a
            # list of objects has no single value, so those fall through to the warning.
            flattened_any = False
            reported_any = False
            for sub_key, sub_value in value.items():
                flat_key = f"{out_key}_{sub_key}"
                # A sub-field can fail to land three ways, and every one of them is reported by
                # sub-path. Reporting only the all-or-nothing case would let a partially
                # flattened object look like a complete success while the colliding half
                # vanished silently.
                if not _is_scalar(sub_value) or sub_value is None or sub_value == "":
                    # Deeper nesting, or an empty value: same rule as the top level.
                    if dropped_nested is not None and isinstance(sub_value, (dict, list, tuple)) and sub_value:
                        dropped_nested.add(f"{key}.{sub_key}")
                        reported_any = True
                    continue
                # A name already CLAIMED wins over a synthesized one, and structural keys stay
                # untouchable — but a name we decline to take is a field the consumer will not
                # see, so it is named rather than skipped in silence. Judged against ``out``,
                # which pass 1 has fully populated: a record key that field_map did not select
                # is not in the metadata and must not block a name nothing is using.
                if flat_key in _STRUCTURAL_METADATA_KEYS or flat_key in out:
                    if dropped_nested is not None:
                        dropped_nested.add(f"{key}.{sub_key}")
                        reported_any = True
                    continue
                out[flat_key] = sub_value
                flattened_any = True
            # The bare key covers only the case nothing above could name — every sub-value
            # empty or blank; when sub-paths were reported, naming the parent too would list
            # the same export decision twice.
            if not flattened_any and not reported_any and dropped_nested is not None:
                dropped_nested.add(key)
        elif dropped_nested is not None and isinstance(value, (list, tuple)) and value:
            # Truthy only. An empty container carries nothing anywhere — the renderer omits it
            # too — so naming it would tell the operator a field was "text-only" when it holds no
            # value at all, and the repair the warning suggests (flatten it) is inapplicable.
            dropped_nested.add(key)
    return out


def _coerce_records(raw: t.Any, records_path: str = "") -> list[dict]:
    """The record list inside a parsed JSON payload.

    Accepts the array itself, or an envelope holding it (``{"products": [...]}``) —
    ``records_path`` names the key, dotted for nesting. Without a path, a single-key
    envelope wrapping a list is unwrapped, because that shape is unambiguous; a
    multi-key object is not guessed at.
    """
    node = raw
    if records_path:
        for part in records_path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"records_path {records_path!r} does not resolve: no key {part!r} in {type(node).__name__}.")
            node = node[part]
        if isinstance(node, dict):
            node = [node]  # an explicitly named path may point at a single record
    elif isinstance(node, dict):
        # A single key holding a list OF RECORDS is an unambiguous envelope. The
        # element check is what keeps a lone record from being mistaken for one: an
        # order with an `items: [...]` line-item array would otherwise unwrap to its
        # line items and lose the order itself — silently, which is the wrong-boundary
        # failure this module exists to prevent. Anything else (several arrays, or an
        # array of scalars) is genuinely ambiguous and falls through to the error below,
        # which names records_path.
        record_lists = [k for k, v in node.items() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        if len(record_lists) == 1 and len(node) == 1:
            node = node[record_lists[0]]
        elif not record_lists:
            node = [node]  # a lone record; a list of scalars is one of its FIELDS

    if not isinstance(node, list):
        # ValueError, not the TypeError ruff suggests: every shape complaint this module
        # raises is about the caller's file, not a programming error, and callers catch
        # one class for all of them.
        raise ValueError(
            f"Could not find the record array in a {type(node).__name__} with keys "
            f"{sorted(node)[:8] if isinstance(node, dict) else '-'}. "
            f"Pass records_path=... to name the key holding it."
        )
    records = [r for r in node if isinstance(r, dict)]
    if not records and node:
        raise ValueError("No object records found — a JSON records source expects objects (one per record), not bare scalars.")
    return records


@dataclass
class JsonRecordsSource:
    """Loads a JSON / JSON Lines file (or a directory of them), one chunk per record.

    ``field_map`` selects and labels the fields to render, in its own order
    (``{"sku": "SKU", "price": "Price"}``); omitted, every field is rendered
    under its own key. A DOTTED key (``{"availability.status": "availability_status"}``)
    reaches a scalar inside nested dicts (dicts only, no list indices) and carries it —
    under the label — into both the rendered text and the chunk metadata. ``title_field``
    and ``id_field`` compose the record's heading — an id in the heading is what makes
    the identifier citable (see the module docstring).
    """

    path: str | Path
    glob: str = "**/*"
    records_path: str = ""
    title_field: str = ""
    id_field: str = ""
    field_map: dict[str, str] = field(default_factory=dict)
    _relationships: list[InitialRelationship] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @property
    def initial_relationships(self) -> list[InitialRelationship]:
        """Structural edges discovered during load(). Available after calling load()."""
        return self._relationships

    def load(self, **kwargs: t.Any) -> list[Document]:
        root = Path(self.path)
        if not root.exists():
            raise FileNotFoundError(f"JSON source not found: {root}")

        records_path = kwargs.get("records_path", self.records_path)
        title_field = kwargs.get("title_field", self.title_field)
        id_field = kwargs.get("id_field", self.id_field)
        field_map = kwargs.get("field_map", self.field_map) or {}

        if root.is_file():
            files = [root]
        else:
            files = sorted(f for f in root.glob(self.glob) if f.is_file() and f.suffix.lower() in _JSON_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No .json/.jsonl files under {root} matching {self.glob!r}.")

        documents: list[Document] = []
        relationships: list[InitialRelationship] = []

        for fp in files:
            records = self._read_records(fp, records_path)
            name = fp.name
            label = fp.stem.replace("_", " ")

            dropped_nested: set[str] = set()
            for idx, record in enumerate(records):
                heading = self._heading(record, title_field, id_field, idx)
                body = self._render_record(record, field_map)
                documents.append(
                    Document(
                        name=name,
                        # Same envelope the directory source uses, so a corpus mixing
                        # markdown files and JSON records reads as one document set.
                        text=f"[Document: {label} | Part {idx + 1}]\n\n## {heading}\n\n{body}",
                        chunk_idx=idx,
                        # The record's own scalar fields ride in metadata, not just in the
                        # rendered text. Consumers read a chunk's metadata as its
                        # properties, and a consumer seeding entities projects an entity's
                        # fields from exactly those properties — so a record whose values exist only in its
                        # chunk prose yields an entity with no fields at all. Rendering it
                        # twice is the point: the text is what retrieval reads, the metadata is
                        # what structured consumers read.
                        metadata={
                            "heading_path": [heading],
                            "record_index": idx,
                            **_scalar_fields(record, field_map, dropped_nested),
                        },
                    )
                )

            if dropped_nested:
                # Once per source, not per record. Silent here means a structured consumer
                # (entity seeding projects fields from exactly these properties)
                # produces entities missing the field, with nothing anywhere saying why.
                logger.warning(
                    "json_records %s: field(s) %s are carried in the chunk TEXT only, not in chunk metadata. A "
                    "consumer that projects entities from metadata (e.g. entity seeding) will not see them. "
                    "One level of nested OBJECT is flattened automatically (availability.status -> "
                    "availability_status), so what remains here is a list, a value nested deeper than one level, or "
                    "a sub-field whose flattened name was already taken (reported as parent.child). Flatten it in "
                    "the export, or map it with a dotted field_map key, if an entity needs it as a field.",
                    label,
                    sorted(dropped_nested),
                )

            for idx in range(len(records) - 1):
                relationships.append(InitialRelationship(name, idx, name, idx + 1, REL_NEXT_RECORD))

        self._relationships = relationships
        return documents

    def _read_records(self, fp: Path, records_path: str) -> list[dict]:
        raw_text = fp.read_text(encoding="utf-8", errors="replace")
        if fp.suffix.lower() in _LINE_DELIMITED_SUFFIXES:
            records: list[dict] = []
            skipped: list[int] = []
            for lineno, line in enumerate(raw_text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{fp.name}:{lineno} is not valid JSON: {exc}") from exc
                if isinstance(parsed, dict):
                    records.append(parsed)
                else:
                    skipped.append(lineno)
            if skipped:
                # As strict as the array path, which rejects the same content outright. A
                # dropped line is a record missing from the corpus, and a short corpus
                # fails much later as an unanswerable question, where the cause is gone.
                raise ValueError(
                    f"{fp.name}: line(s) {skipped[:10]} are not JSON objects. A records source expects one object per line; remove them or fix the export."
                )
            return records
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            # Named explicitly: a mis-parse here yields an empty corpus much later, where
            # the cause is out of sight.
            raise ValueError(f"{fp.name} is not valid JSON: {exc}") from exc
        return _coerce_records(parsed, records_path)

    def _heading(self, record: dict, title_field: str, id_field: str, idx: int) -> str:
        """The record's naming line: its title, with its id alongside when distinct."""
        title = ""
        if title_field:
            title = _render_value(record.get(title_field, "")).strip()
        if not title:
            title = next(
                (v.strip() for k, v in record.items() if isinstance(v, str) and v.strip() and k != id_field),
                "",
            )
        identifier = _render_value(record.get(id_field, "")).strip() if id_field else ""
        if title and identifier:
            return f"{title} ({identifier})"
        return title or identifier or f"Record {idx + 1}"

    def _render_record(self, record: dict, field_map: dict[str, str]) -> str:
        """One bullet per field, in ``field_map`` order when one is given.

        A field named in ``field_map`` but absent from this record — including a dotted
        path that does not resolve — is skipped rather than rendered empty: "Price:" with
        nothing after it is exactly the shape a summarizer reads as "this record has no
        price".
        """
        pairs = (
            [(label, value) for key, label in field_map.items() if (value := _resolve_field(record, key)) is not _MISSING]
            if field_map
            else list(record.items())
        )
        lines = []
        for label, value in pairs:
            rendered = _render_value(value)
            if rendered:
                lines.append(f"- {label}: {rendered}")
        return "\n".join(lines)
