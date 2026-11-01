from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .critic import Critic
from .document_loader import load_builtin_documents, load_local_documents
from .llm import LLMBackend, create_backend
from .pipeline import RAGPipeline
from .retriever import (
    FAISSRetriever,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)


@dataclass
class Runtime:
    settings: Settings
    retriever: FAISSRetriever
    backend: LLMBackend
    critic: Critic
    pipeline: RAGPipeline


def _select_embedder(settings: Settings):
    backend = settings.embedding_backend
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if backend == "auto":
        if settings.llm_provider == "stub":
            return HashingEmbedder()
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise RuntimeError(f"Unknown embedding backend: {backend}")


def build_runtime(
    settings: Settings | None = None,
    force_rebuild: bool = False,
) -> Runtime:
    active_settings = settings or Settings.from_env()

    local_paths: list[Path] = []
    if active_settings.documents_dir.exists():
        local_paths = sorted(
            path
            for path in active_settings.documents_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".txt", ".md", ".pdf"}
        )

    chunks = load_builtin_documents(active_settings)
    chunks.extend(load_local_documents(local_paths, active_settings))

    embedder = _select_embedder(active_settings)
    retriever = FAISSRetriever(
        index_path=active_settings.index_path,
        chunks_path=active_settings.chunks_path,
        embedding_model_name=(
            getattr(embedder, "model_name", None)
            or f"hashing-{getattr(embedder, 'dimension', 'unknown')}"
        ),
        embedder=embedder,
    )
    if force_rebuild:
        retriever.build(chunks)
    else:
        retriever.build_or_load(chunks)

    backend = create_backend(active_settings)
    critic = Critic(backend=backend, backend_retries=1)
    pipeline = RAGPipeline(
        settings=active_settings,
        retriever=retriever,
        backend=backend,
        critic=critic,
    )
    return Runtime(
        settings=active_settings,
        retriever=retriever,
        backend=backend,
        critic=critic,
        pipeline=pipeline,
    )
