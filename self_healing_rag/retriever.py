from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Protocol, Sequence

import faiss
import numpy as np

from .logger import get_logger
from .models import Chunk, RetrievedChunk


logger = get_logger(__name__)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into a two-dimensional NumPy array."""


class HashingEmbedder:
    """Deterministic lexical embedder used by fully offline stub mode."""

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
            if len(token) >= 2
        ]

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = self._tokens(text)
            features = list(tokens)
            features.extend(
                f"{left}::{right}"
                for left, right in zip(tokens, tokens[1:])
            )
            for feature in features:
                digest = hashlib.blake2b(
                    feature.encode("utf-8"),
                    digest_size=16,
                ).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, index] += sign
        return vectors


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FAISSRetriever:
    def __init__(
        self,
        index_path: Path,
        chunks_path: Path,
        embedding_model_name: str,
        embedder: Embedder | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.embedding_model_name = embedding_model_name
        self.embedder = embedder or SentenceTransformerEmbedder(
            embedding_model_name
        )
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    @staticmethod
    def _fingerprint(chunks: Sequence[Chunk]) -> str:
        payload = json.dumps(
            [chunk.to_dict() for chunk in chunks],
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load(self, expected_chunks: Sequence[Chunk]) -> bool:
        if not self.index_path.exists() or not self.chunks_path.exists():
            return False
        try:
            metadata = json.loads(
                self.chunks_path.read_text(encoding="utf-8")
            )
            if metadata.get("embedding_model") != self.embedding_model_name:
                logger.info("Stored index uses another embedding model; rebuilding")
                return False
            if metadata.get("fingerprint") != self._fingerprint(expected_chunks):
                logger.info("Stored chunk metadata changed; rebuilding")
                return False

            index = faiss.read_index(str(self.index_path))
            chunks = [
                Chunk.from_dict(item)
                for item in metadata["chunks"]
            ]
            if index.ntotal != len(chunks):
                logger.info("Stored FAISS index is inconsistent; rebuilding")
                return False

            self._index = index
            self._chunks = chunks
            return True
        except Exception as exc:
            logger.warning("Could not load stored FAISS index: %s", exc)
            return False

    def build(self, chunks: Sequence[Chunk]) -> None:
        chunk_list = list(chunks)
        if len({chunk.id for chunk in chunk_list}) != len(chunk_list):
            raise ValueError("Chunk IDs must be unique")

        if chunk_list:
            vectors = _normalize(
                self.embedder.encode([chunk.text for chunk in chunk_list])
            )
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        else:
            index = faiss.IndexFlatIP(1)

        self._index = index
        self._chunks = chunk_list

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_index = self.index_path.with_suffix(
            self.index_path.suffix + ".tmp"
        )
        faiss.write_index(self._index, str(temporary_index))
        temporary_index.replace(self.index_path)

        metadata = {
            "version": 1,
            "embedding_model": self.embedding_model_name,
            "fingerprint": self._fingerprint(chunk_list),
            "chunks": [chunk.to_dict() for chunk in chunk_list],
        }
        temporary_metadata = self.chunks_path.with_suffix(
            self.chunks_path.suffix + ".tmp"
        )
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_metadata.replace(self.chunks_path)
        logger.info("Built FAISS index with %s chunks", len(chunk_list))

    def build_or_load(self, chunks: Sequence[Chunk]) -> None:
        if self._load(chunks):
            logger.info("Loaded existing FAISS index")
            return
        self.build(chunks)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if self._index is None and not self._load(self._chunks):
            return []
        if self._index is None or self._index.ntotal == 0:
            return []

        vector = _normalize(self.embedder.encode([query]))
        scores, indices = self._index.search(vector, max(1, int(k)))
        results: list[RetrievedChunk] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self._chunks):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[int(index)],
                    score=float(score),
                )
            )
        return results
