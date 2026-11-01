from __future__ import annotations

import re
from pathlib import Path

from .config import Settings
from .logger import get_logger
from .models import Chunk
from .sample_data import BUILTIN_DOCUMENTS


logger = get_logger(__name__)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "document"


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"[ \t]+", " ", text.strip())
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", normalized)
        if paragraph.strip()
    ]

    units: list[str] = []
    for paragraph in paragraphs:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
            if sentence.strip()
        ]
        for sentence in sentences:
            words = sentence.split()
            if len(words) <= chunk_size:
                units.append(sentence)
                continue
            for start in range(0, len(words), chunk_size):
                units.append(" ".join(words[start : start + chunk_size]))

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for unit in units:
        unit_size = len(unit.split())
        if current and current_size + unit_size > chunk_size:
            chunks.append(" ".join(current))

            overlap_units: list[str] = []
            overlap_size = 0
            for previous in reversed(current):
                previous_size = len(previous.split())
                if overlap_size + previous_size > overlap:
                    break
                overlap_units.insert(0, previous)
                overlap_size += previous_size

            current = overlap_units
            current_size = overlap_size

        if current and current_size + unit_size > chunk_size:
            current = []
            current_size = 0

        current.append(unit)
        current_size += unit_size

    if current:
        chunks.append(" ".join(current))
    return chunks


def _make_chunks(
    source: str,
    text: str,
    page: int,
    settings: Settings,
    start_index: int,
) -> list[Chunk]:
    pieces = _split_text(
        text,
        chunk_size=settings.chunk_size_words,
        overlap=settings.chunk_overlap_words,
    )
    slug = _slug(source)
    return [
        Chunk(
            id=f"{slug}::page-{page:03d}::chunk-{start_index + offset:03d}",
            text=piece,
            source=source,
            page=page,
        )
        for offset, piece in enumerate(pieces)
    ]


def load_builtin_documents(settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in BUILTIN_DOCUMENTS:
        chunks.extend(
            _make_chunks(
                source=source,
                text=text,
                page=1,
                settings=settings,
                start_index=len(chunks),
            )
        )
    return chunks


def load_local_documents(
    paths: list[Path],
    settings: Settings,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for path in paths:
        try:
            suffix = path.suffix.lower()
            if suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8")
                chunks.extend(
                    _make_chunks(
                        source=path.name,
                        text=text,
                        page=1,
                        settings=settings,
                        start_index=len(chunks),
                    )
                )
                continue

            if suffix == ".pdf":
                from PyPDF2 import PdfReader

                reader = PdfReader(str(path))
                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    chunks.extend(
                        _make_chunks(
                            source=path.name,
                            text=text,
                            page=page_number,
                            settings=settings,
                            start_index=len(chunks),
                        )
                    )
        except Exception as exc:
            logger.error("Could not load document %s: %s", path, exc)
            raise

    return chunks
