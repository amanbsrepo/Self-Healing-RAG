from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _read_path(root: Path, name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default)))
    return value if value.is_absolute() else root / value


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    index_path: Path = PROJECT_ROOT / "data" / "faiss_index.bin"
    chunks_path: Path = PROJECT_ROOT / "data" / "chunks.json"
    evaluation_results_path: Path = PROJECT_ROOT / "evaluation" / "results.md"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "auto"
    top_k: int = 5
    max_top_k: int = 10
    max_attempts: int = 3

    chunk_size_words: int = 100
    chunk_overlap_words: int = 20

    llm_provider: str = "stub"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"
    openai_api_key: str | None = field(default=None, repr=False)
    anthropic_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or PROJECT_ROOT).resolve()
        load_dotenv(root / ".env")

        top_k = max(1, _read_int("TOP_K", 5))
        max_top_k = max(2 * top_k, _read_int("MAX_TOP_K", 2 * top_k))
        max_attempts = max(1, _read_int("MAX_ATTEMPTS", 3))

        chunk_size = max(20, _read_int("CHUNK_SIZE_WORDS", 100))
        overlap = max(0, _read_int("CHUNK_OVERLAP_WORDS", 20))
        if overlap >= chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP_WORDS must be smaller than CHUNK_SIZE_WORDS"
            )

        provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        if provider not in {"auto", "stub", "openai", "anthropic"}:
            raise ValueError(
                "LLM_PROVIDER must be auto, stub, openai, or anthropic"
            )

        embedding_backend = (
            os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
        )
        if embedding_backend not in {
            "auto",
            "hashing",
            "sentence_transformers",
        }:
            raise ValueError(
                "EMBEDDING_BACKEND must be auto, hashing, or sentence_transformers"
            )

        openai_key = os.getenv("OPENAI_API_KEY") or None
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") or None

        if provider == "auto":
            if openai_key:
                provider = "openai"
            elif anthropic_key:
                provider = "anthropic"
            else:
                provider = "stub"

        return cls(
            project_root=root,
            data_dir=_read_path(root, "DATA_DIR", root / "data"),
            documents_dir=_read_path(
                root,
                "DOCUMENTS_DIR",
                root / "data" / "documents",
            ),
            index_path=_read_path(
                root,
                "FAISS_INDEX_PATH",
                root / "data" / "faiss_index.bin",
            ),
            chunks_path=_read_path(
                root,
                "CHUNKS_METADATA_PATH",
                root / "data" / "chunks.json",
            ),
            evaluation_results_path=root / "evaluation" / "results.md",
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            embedding_backend=embedding_backend,
            top_k=top_k,
            max_top_k=max_top_k,
            max_attempts=max_attempts,
            chunk_size_words=chunk_size,
            chunk_overlap_words=overlap,
            llm_provider=provider,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_model=os.getenv(
                "ANTHROPIC_MODEL",
                "claude-3-5-haiku-latest",
            ),
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
        )
