"""DirectorySource — loads and chunks files from a local directory.

Supports PDF (via Docling), plain text, and Markdown.

Markdown is chunked BY SECTION, not by a flat token window, and the resulting
chunks carry structural relationships (``next_section`` / ``parent_section`` /
``links_to``). That shape is load-bearing for consumers: a path needs at least two linked
chunks, so one chunk per file with no relationships is unreachable as a path. One chunk per
heading section plus the relationships gives the same doc folder chunks that link to each other.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import time
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from plurai_extract.sources.base import Document, InitialRelationship, chunk_tokens

logger = logging.getLogger(__name__)

#: Moves plurai-extract's cache alone. XDG_CACHE_HOME also moves the PDF reader's model cache, so it is
#: never the remedy offered: on an offline machine that would break PDF reading.
CACHE_DIR_ENV = "PLURAI_EXTRACT_CACHE_DIR"
#: Page-cache dirs already reported unusable, so a corpus of many PDFs logs one line, not one per file.
_UNUSABLE_CACHE_DIRS: set[str] = set()
#: A cache temp file whose age exceeds this was left by a writer that died mid-write; the next write removes it.
_ORPHAN_AGE_S = 3600


class UnreadablePDFError(RuntimeError):
    """The PDF reader's verdict on ONE file: it cannot be converted. Any other failure is not per-file."""


@dataclass
class PdfRead:
    """What reading one PDF gave: the (page_no, text) of every page with text, and each page without text,
    typed as ``blank_pages`` (it draws nothing, so nothing was lost) or ``unread_pages`` (it draws something
    the reader took no text from, a scan say)."""

    pages: list[tuple[int, str]]
    page_count: int
    unread_pages: list[int]
    blank_pages: list[int]


def pdf_to_pages(pdf_path: Path) -> PdfRead:
    """Read a PDF's pages using Docling. Results cached in the user's cache dir."""
    import hashlib
    import importlib.metadata

    def _version(dist: str) -> str:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return "none"

    # Cached per user, never beside the documents. Keyed on the file's BYTES and the reader's version, so
    # a PDF edited in place or a reader upgrade reads the pages again; the stem only names the file.
    readers = f"docling={_version('docling')};plurai-extract={_version('plurai-extract')}"
    pdf_hash = hashlib.sha256(pdf_path.read_bytes() + b"\0" + readers.encode()).hexdigest()[:16]
    cache_dir = _page_cache_dir()
    cache_path = cache_dir / f"{pdf_path.stem}_{pdf_hash}.pages.json" if cache_dir else None
    if cache_path is not None:
        try:  # every key is required: an entry that does not say which pages it lacks is a miss
            entry = json.loads(cache_path.read_text())
            return PdfRead(
                pages=[(e["page_no"], e["text"]) for e in entry["pages"]],
                page_count=entry["page_count"],
                unread_pages=entry["unread_pages"],
                blank_pages=entry["blank_pages"],
            )
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:  # a corrupt entry is a miss, replaced below
            logger.warning("Ignoring an unreadable PDF page cache entry %s (%s); reading the PDF again", cache_path, exc)

    try:
        from docling.datamodel.base_models import ConversionStatus
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.exceptions import ConversionError
    except ImportError:
        raise ImportError("docling is required for PDF processing. Install it with: pip install 'plurai-extract[pdf]'")

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})
    try:
        conv_res = converter.convert(str(pdf_path))
    except ConversionError as exc:
        raise UnreadablePDFError(f"{pdf_path.name}: {exc}") from exc
    doc = conv_res.document

    pages: list[tuple[int, str]] = []
    textless: list[int] = []
    for page_no in sorted(doc.pages.keys()):
        texts = []
        for item, _ in doc.iterate_items():
            provenance = getattr(item, "prov", [])
            if any(getattr(p, "page_no", None) == page_no for p in provenance):
                if hasattr(item, "text") and item.text:
                    texts.append(item.text)
        text = "\n\n".join(texts)
        if text.strip():
            pages.append((page_no, text))
        else:
            textless.append(page_no)

    # A page Docling drops from `doc.pages` entirely — a transient per-page failure does exactly this on some
    # Docling versions, rather than keeping the page, empty, in a partial success — is invisible to the loop
    # above: it is in neither `pages` nor `textless`, and `len(doc.pages)` never counted it either. pypdfium2's
    # own page count is the authority that catches it, for every Docling version; such a page is always reported
    # unread, never guessed blank, since Docling gave no verdict on it at all to render-check.
    true_page_count, blank, decided = _pdfium_read(pdf_path, textless)
    missing = [p for p in range(1, true_page_count + 1) if p not in doc.pages] if true_page_count is not None else []

    read = PdfRead(
        pages,
        true_page_count if true_page_count is not None else len(doc.pages),
        sorted([p for p in textless if p not in blank] + missing),
        blank,
    )
    # A page pdfium could not render, one Docling failed on (kept, empty, in a partial success), or one Docling
    # dropped entirely, is decided again next run, never cached.
    if cache_path is not None and decided and not missing and conv_res.status == ConversionStatus.SUCCESS:
        _write_cache_entry(cache_path, read)
    return read


def _pdfium_read(pdf_path: Path, page_nos: list[int]) -> tuple[int | None, list[int], bool]:
    """One pypdfium2 open of ``pdf_path`` for both facts Docling's assembled result cannot always be trusted
    for (see ``pdf_to_pages``): the file's own page count, and which of ``page_nos`` (1-based, the pages
    Docling gave no text for) draw nothing at all — rendered, their bitmap is one uniform value — versus which
    are unread: a page that cannot be rendered is not blank, so it is reported as unread, and the third value
    is False: the verdict on every page was not reached.

    The page count is ``None`` when the file cannot even be opened; the caller then falls back to Docling's
    own count."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError("pypdfium2 is required for PDF processing. Install it with: pip install 'plurai-extract[pdf]'")

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except pdfium.PdfiumError as exc:
        # Nothing needed a page-by-page verdict, so a page count that Docling's own count can stand in for is
        # the only fact lost: not a reason to skip caching, same as when this file has no textless pages at all.
        if page_nos:
            logger.warning(
                "Could not open %s to count its pages or tell blank ones from unread ones (%s); its pages without text are reported unread", pdf_path.name, exc
            )
        else:
            logger.warning("Could not open %s to count its pages (%s); Docling's own count is used instead", pdf_path.name, exc)
        return None, [], not page_nos
    blank: list[int] = []
    decided = True
    try:
        page_count = len(doc)
        for page_no in page_nos:
            try:
                bitmap = doc[page_no - 1].render(scale=20 / 72, grayscale=True)
            except (pdfium.PdfiumError, IndexError) as exc:
                logger.warning("Could not render page %d of %s (%s); it is reported unread", page_no, pdf_path.name, exc)
                decided = False
                continue
            row = bitmap.width * bitmap.n_channels
            buffer = bytes(bitmap.buffer)
            pixels = b"".join(buffer[r * bitmap.stride : r * bitmap.stride + row] for r in range(bitmap.height))
            if pixels and min(pixels) == max(pixels):
                blank.append(page_no)
    finally:
        doc.close()
    return page_count, blank, decided


def _page_cache_dir() -> Path | None:
    """The per-user PDF page cache dir (owner-only) under ``$PLURAI_EXTRACT_CACHE_DIR``, else
    ``$XDG_CACHE_HOME/plurai-extract`` (default ``~/.cache``), or ``None`` when it cannot be created: the cache is an
    optimisation, so an unusable location costs the cache and never the ingest."""
    base = os.environ.get(CACHE_DIR_ENV) or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "plurai-extract"
    cache_dir = Path(base) / "pdf_pages"
    try:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        if str(cache_dir) not in _UNUSABLE_CACHE_DIRS:
            _UNUSABLE_CACHE_DIRS.add(str(cache_dir))
            logger.warning("PDF page cache disabled (%s); set %s to a writable directory to enable it", exc, CACHE_DIR_ENV)
        return None
    return cache_dir


def _write_cache_entry(cache_path: Path, read: PdfRead) -> None:
    """Write aside and rename into place, so a write cut off mid-way never leaves an entry to trust. A failed
    write only loses the cache. Temp files a killed writer left behind are removed once their age passes the sweep threshold."""
    partial = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    entry = {
        "page_count": read.page_count,
        "pages": [{"page_no": p, "text": t} for p, t in read.pages],
        "unread_pages": read.unread_pages,
        "blank_pages": read.blank_pages,
    }
    try:
        partial.write_text(json.dumps(entry))
        os.replace(partial, cache_path)
    except OSError as exc:
        if str(cache_path.parent) not in _UNUSABLE_CACHE_DIRS:
            _UNUSABLE_CACHE_DIRS.add(str(cache_path.parent))
            logger.warning("PDF page cache entries not written (%s); set %s to a writable directory", exc, CACHE_DIR_ENV)
        try:
            partial.unlink()
        except OSError:
            pass
        return
    for orphan in cache_path.parent.glob("*.tmp"):
        try:
            if time.time() - orphan.stat().st_mtime > _ORPHAN_AGE_S:
                orphan.unlink()
        except OSError:
            pass


def chunk_pdf(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> tuple[list[tuple[str, int, int]], PdfRead]:
    """Chunk a PDF into (text, page_start, page_end) tuples, returned with the read they came from."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    read = pdf_to_pages(pdf_path)
    pages = read.pages
    step = max(1, chunk_size - chunk_overlap)

    results: list[tuple[str, int, int]] = []
    buf: list[int] = []
    buf_start = pages[0][0] if pages else 1

    for page_num, page_text in pages:
        if not buf:
            buf_start = page_num
        else:
            buf += enc.encode("\n\n")
        buf += enc.encode(page_text)

        if len(buf) >= chunk_size:
            for i in range(0, len(buf), step):
                chunk = enc.decode(buf[i : i + chunk_size])
                if chunk.strip():
                    results.append((chunk, buf_start, page_num))
                if i + chunk_size >= len(buf):
                    break
            buf = buf[-chunk_overlap:]
            buf_start = page_num

    if buf:
        chunk = enc.decode(buf)
        if chunk.strip():
            results.append((chunk, buf_start, pages[-1][0] if pages else 1))

    return results, read


_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
#: The suffixes split at their headings in the default section mode. Every other text suffix is read in
#: token windows, one chain per file (N chunks, N-1 ``next_section`` edges).
SECTION_CHUNKED_SUFFIXES = _MARKDOWN_EXTENSIONS

# Chunking modes for text files. "section" is the default; "tokens" is a flat sliding
# window (for prose with no heading structure).
CHUNK_MODE_SECTION = "section"
CHUNK_MODE_TOKENS = "tokens"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
# [text](target) / [text](target "title") — bare, non-image markdown links.
_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")


@dataclass
class Section:
    """One heading-delimited slice of a markdown document.

    ``heading_path`` is the breadcrumb of enclosing headings (``()`` for a
    pre-first-heading preamble). ``owned_paths`` is every heading whose text ended
    up inside this section — normally just ``heading_path``, but a heading with no
    body of its own is merged into a neighbour and its path travels with it, so
    ``parent_section`` edges can still find the chunk that carries the parent
    heading.
    """

    heading_path: tuple[str, ...]
    owned_paths: tuple[tuple[str, ...], ...]
    text: str


def _heading_lines(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line_index, level, title) for every ATX heading OUTSIDE a fenced code block.

    Fence tracking matters: a ``# comment`` inside a shell/python block is not a
    heading, and treating it as one shreds a code example across chunks.
    """
    heads: list[tuple[int, int, str]] = []
    fence: str | None = None
    for i, line in enumerate(lines):
        if fence is not None:
            if line.strip().startswith(fence):
                fence = None
            continue
        opening = _FENCE_RE.match(line)
        if opening:
            fence = opening.group(1)
            continue
        m = _HEADING_RE.match(line.strip())
        if m:
            heads.append((i, len(m.group(1)), m.group(2).strip()))
    return heads


def _body_is_empty(section: Section) -> bool:
    """True when the section contributes nothing but its own heading line."""
    body = section.text.splitlines()
    if section.heading_path:
        body = body[1:]
    return not any(line.strip() for line in body)


def _merge_headings_without_body(sections: list[Section]) -> list[Section]:
    """Fold a heading that carries no prose of its own into its neighbour.

    A lone ``# Title`` above the first real section would otherwise become a chunk
    with nothing to retrieve. Merging (rather than dropping) keeps the title text in
    the corpus and keeps its ``owned_paths`` addressable as a parent.
    """
    out: list[Section] = []
    pending: list[Section] = []
    for sec in sections:
        if _body_is_empty(sec):
            pending.append(sec)
            continue
        if pending:
            sec = Section(
                sec.heading_path,
                tuple(p for held in pending for p in held.owned_paths) + sec.owned_paths,
                "".join(held.text for held in pending) + sec.text,
            )
            pending = []
        out.append(sec)

    if pending:  # trailing headings with no body (or a doc that is only headings)
        merged_paths = tuple(p for held in pending for p in held.owned_paths)
        merged_text = "".join(held.text for held in pending)
        if out:
            last = out[-1]
            out[-1] = Section(last.heading_path, last.owned_paths + merged_paths, last.text + merged_text)
        else:
            out.append(Section(pending[0].heading_path, merged_paths, merged_text))
    return out


def split_markdown_sections(raw: str) -> list[Section]:
    """Split markdown into one :class:`Section` per ATX heading.

    Text before the first heading is its own section (nothing is silently dropped);
    a document with no headings at all is one section. Sub-headings nest, so a
    section's ``heading_path`` is the full breadcrumb from the top-level heading down.
    """
    if not raw.strip():
        return []

    lines = raw.splitlines(keepends=True)
    heads = _heading_lines(lines)
    if not heads:
        return [Section((), (), raw)]

    sections: list[Section] = []
    preamble = "".join(lines[: heads[0][0]])
    if preamble.strip():
        sections.append(Section((), (), preamble))

    path: list[str] = []
    for i, (start, level, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(lines)
        del path[level - 1 :]
        path.append(title)
        breadcrumb = tuple(path)
        sections.append(Section(breadcrumb, (breadcrumb,), "".join(lines[start:end])))

    return _merge_headings_without_body(sections)


def _split_oversize(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """The section itself, or token windows over it when it exceeds ``chunk_size``.

    The character-count guard is exact, not a heuristic: a tiktoken token is at least
    one character, so text of at most ``chunk_size`` characters can never exceed
    ``chunk_size`` tokens — and skipping the encode keeps the common (small doc)
    path free of tiktoken entirely.
    """
    if len(text) <= chunk_size:
        return [text]
    return chunk_tokens(text, chunk_size, chunk_overlap) or [text]


REL_NEXT_SECTION = "next_section"
REL_PARENT_SECTION = "parent_section"
REL_LINKS_TO = "links_to"


def _derive_relationships(
    per_file: list[tuple[Path, str, list[tuple[tuple[str, ...], ...]], list[str]]],
    by_path: dict[Path, str],
) -> list[InitialRelationship]:
    """Structural edges over the loaded chunks.

    * ``next_section`` — reading order within a document (also holds for the token
      windows of a .txt or a PDF, where "next chunk" is the same relation).
    * ``parent_section`` — a subsection points at the chunk carrying its parent heading.
    * ``links_to`` — a markdown link that resolves to another ingested file, pointing at
      that file's first chunk.

    Types are plain strings so the backend can carry them through untouched.
    """
    rels: list[InitialRelationship] = []

    for fp, name, owned, texts in per_file:
        for idx in range(len(texts) - 1):
            rels.append(InitialRelationship(name, idx, name, idx + 1, REL_NEXT_SECTION))

        # First chunk owning a heading wins; a heading merged into a neighbour is
        # owned by that neighbour, which is exactly the chunk a child should point at.
        owner: dict[tuple[str, ...], int] = {}
        for idx, paths in enumerate(owned):
            for path in paths:
                owner.setdefault(path, idx)

        for idx, paths in enumerate(owned):
            for path in paths:
                parent_idx = owner.get(path[:-1]) if len(path) > 1 else None
                if parent_idx is not None and parent_idx != idx:
                    rels.append(
                        InitialRelationship(
                            name,
                            idx,
                            name,
                            parent_idx,
                            REL_PARENT_SECTION,
                            {"heading_path": list(path)},
                        )
                    )

        for idx, text in enumerate(texts):
            for target in _resolve_links(fp, text, by_path):
                if target != name:
                    rels.append(
                        InitialRelationship(
                            name,
                            idx,
                            target,
                            0,
                            REL_LINKS_TO,
                            {"target_file": target},
                        )
                    )

    return rels


def _resolve_links(fp: Path, text: str, by_path: dict[Path, str]) -> list[str]:
    """Doc names this chunk's markdown links point at, de-duplicated, in order.

    Only intra-corpus links become edges: URLs, mailto:, and bare anchors have no
    ingested target, and a relative path that resolves outside the ingested set is
    a dangling reference, not a relationship.
    """
    seen: dict[str, None] = {}
    for raw_target in _LINK_RE.findall(text):
        target = raw_target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        try:
            resolved = (fp.parent / target).resolve()
        except (OSError, ValueError):
            continue
        name = by_path.get(resolved)
        if name is not None:
            seen.setdefault(name, None)
    return list(seen)


def _is_pdf(fp: Path) -> bool:
    """Detect PDF by extension or magic bytes (%PDF header)."""
    if fp.suffix.lower() == ".pdf":
        return True
    try:
        with open(fp, "rb") as f:
            return f.read(5) == b"%PDF-"
    except (OSError, PermissionError):
        return False


@dataclass
class DirectorySource:
    """Loads and chunks files from a local directory.

    Supports PDF (via Docling) and the suffixes in ``text_suffixes`` (default .txt, .md, .markdown),
    each read verbatim as UTF-8 text. For other formats use LangChainSource.
    Every file under the glob that yields no chunk (an unsupported format, an empty file, a file
    that is not UTF-8 text, a PDF that cannot be read or has no text) is named in :attr:`skipped_files` (``{"path", "reason"}``, the path relative to
    ``path``) after load(), and logged at WARNING. A PDF that gave chunks from only some of its pages is named in
    :attr:`partial_files` (``{"path", "reason", "pages", "of"}``): ``"no text extracted"`` for pages that draw
    something with no text, ``"empty"`` for blank pages.

    Markdown is split at ATX headings (one chunk per section, ``chunk_mode="section"``)
    and the loaded chunks carry structural :class:`InitialRelationship` edges, exposed
    on :attr:`initial_relationships` for ``load_only`` to pick up. Set
    ``chunk_mode="tokens"`` to fall back to the flat token window for every file.
    """

    path: str | Path
    glob: str = "**/*"
    chunk_size: int = 1024
    chunk_overlap: int = 50
    chunk_mode: str = CHUNK_MODE_SECTION
    #: Suffixes read as text, declared by the caller. Markdown keeps section chunking; any other
    #: declared suffix is read verbatim in token windows.
    text_suffixes: t.Collection[str] = _TEXT_EXTENSIONS
    skipped_files: list[dict[str, str]] = field(default_factory=list, init=False)
    partial_files: list[dict[str, t.Any]] = field(default_factory=list, init=False)
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
            raise FileNotFoundError(f"Directory not found: {root}")

        cs = kwargs.get("chunk_size", self.chunk_size)
        co = kwargs.get("chunk_overlap", self.chunk_overlap)
        mode = kwargs.get("chunk_mode", self.chunk_mode)
        if mode not in (CHUNK_MODE_SECTION, CHUNK_MODE_TOKENS):
            raise ValueError(f"Unknown chunk_mode {mode!r}; expected {CHUNK_MODE_SECTION!r} or {CHUNK_MODE_TOKENS!r}.")

        documents: list[Document] = []
        self.skipped_files = []
        self.partial_files = []
        files = sorted(f for f in root.glob(self.glob) if f.is_file())
        # A chunk's identity is its source file: a file name two files share is named by its
        # path under the root instead, so each keeps its own chunks and links.
        shared = {n for n, c in collections.Counter(f.name for f in files).items() if c > 1}
        # Per file: (path, doc_name, [heading breadcrumbs each chunk carries], [chunk text]).
        # Consumed after the loop to derive the structural relationships.
        per_file: list[tuple[Path, str, list[tuple[tuple[str, ...], ...]], list[str]]] = []

        for fp in files:
            ext = fp.suffix.lower()
            name = fp.relative_to(root).as_posix() if fp.name in shared else fp.name
            label = fp.stem.replace("_", " ")
            owned: list[tuple[tuple[str, ...], ...]] = []
            texts: list[str] = []

            is_pdf = _is_pdf(fp)
            if is_pdf:
                if fp.stat().st_size == 0:
                    self._skip(fp, root, "empty")
                    continue
                try:
                    pdf_parts, read = chunk_pdf(fp, cs, co)
                except UnreadablePDFError as exc:  # only the reader's verdict on this file; anything else fails the load
                    logger.warning("Could not read %s as a PDF (%s: %s)", fp.name, type(exc).__name__, exc)
                    self._skip(fp, root, "unreadable PDF")
                    continue
                if pdf_parts:  # a PDF that gave no chunk at all is named once, as a whole file, below
                    for reason, page_nos in (("no text extracted", read.unread_pages), ("empty", read.blank_pages)):
                        if page_nos:
                            self._partial(fp, root, reason, page_nos, read.page_count)
                for idx, (text, pg_start, pg_end) in enumerate(pdf_parts):
                    documents.append(
                        Document(
                            name=name,
                            text=f"[Document: {label} | Part {idx + 1}]\n\n{text}",
                            chunk_idx=idx,
                            metadata={"page_start": pg_start, "page_end": pg_end},
                        )
                    )
                    owned.append(())
                    texts.append(text)
            elif ext in self.text_suffixes:
                raw: str | None
                try:  # strict: undecodable bytes are named, never ingested as U+FFFD; a BOM is not text
                    raw = fp.read_text(encoding="utf-8-sig")
                except UnicodeDecodeError:
                    raw = None
                if raw is None or "\x00" in raw:  # NUL also marks UTF-16 without a BOM, which decodes as UTF-8
                    self._skip(fp, root, "not UTF-8 text")
                    continue
                if mode == CHUNK_MODE_SECTION and ext in _MARKDOWN_EXTENSIONS:
                    parts = self._section_chunks(raw, cs, co)
                else:
                    parts = [(text, (), ()) for text in chunk_tokens(raw, cs, co)]
                for idx, (text, heading_path, owned_paths) in enumerate(parts):
                    metadata = {"heading_path": list(heading_path)} if heading_path else {}
                    documents.append(
                        Document(
                            name=name,
                            text=f"[Document: {label} | Part {idx + 1}]\n\n{text}",
                            chunk_idx=idx,
                            metadata=metadata,
                        )
                    )
                    owned.append(owned_paths)
                    texts.append(text)
            else:
                self._skip(fp, root, "unsupported format")
                continue
            if not texts:
                # A PDF that gave text from none of its pages: if any of them was unread (a scan Docling
                # took no text from), that is real loss and still needs the OCR remedy. A PDF whose pages
                # were all blank lost nothing, so it is named "empty" like any other empty source — the
                # per-page block above never ran, since it only fires when at least one chunk landed.
                self._skip(fp, root, ("no text extracted" if read.unread_pages else "empty") if is_pdf else "empty")

            per_file.append((fp, name, owned, texts))

        # resolved path → doc name, built from the files that actually produced chunks: a
        # link to an image, a skipped format, or an empty file has no node to point at.
        by_path = {fp.resolve(): name for fp, name, _, texts in per_file if texts}
        self._relationships = _derive_relationships(per_file, by_path)

        logger.info(
            "Loaded %d chunks + %d structural relationships from %d files in %s",
            len(documents),
            len(self._relationships),
            len(by_path),
            root,
        )
        return documents

    def _skip(self, fp: Path, root: Path, reason: str) -> None:
        rel = fp.relative_to(root).as_posix()
        self.skipped_files.append({"path": rel, "reason": reason})
        logger.warning("Not ingested: %s (%s)", rel, reason)

    def _partial(self, fp: Path, root: Path, reason: str, pages: list[int], of: int) -> None:
        rel = fp.relative_to(root).as_posix()
        self.partial_files.append({"path": rel, "reason": reason, "pages": pages, "of": of})
        logger.warning("Partly ingested: %s, pages %s of %d (%s)", rel, ", ".join(map(str, pages)), of, reason)

    @staticmethod
    def _section_chunks(
        raw: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]]]:
        """(text, heading_path, owned_paths) per chunk for one markdown document.

        A section that is still too large for ``chunk_size`` falls back to token
        windows; only its FIRST window owns the heading, so the section resolves to a
        single parent for its children instead of an ambiguous set.
        """
        chunks: list[tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]]] = []
        for section in split_markdown_sections(raw):
            windows = _split_oversize(section.text.strip(), chunk_size, chunk_overlap)
            for i, window in enumerate(windows):
                chunks.append((window, section.heading_path, section.owned_paths if i == 0 else ()))
        return chunks
