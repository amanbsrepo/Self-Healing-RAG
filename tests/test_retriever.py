import json

import numpy as np

from self_healing_rag.models import Chunk
from self_healing_rag.retriever import FAISSRetriever


class FakeEmbedder:
    def encode(self, texts):
        vectors = {
            "Alpha": [1.0, 0.0],
            "Beta": [0.0, 1.0],
        }
        return np.asarray(
            [vectors[text] for text in texts],
            dtype=np.float32,
        )


def test_faiss_index_build_search_and_json_metadata(tmp_path):
    retriever = FAISSRetriever(
        index_path=tmp_path / "custom" / "index.bin",
        chunks_path=tmp_path / "custom" / "chunks.json",
        embedding_model_name="fake-model",
        embedder=FakeEmbedder(),
    )
    chunks = [
        Chunk("alpha-id", "Alpha", "alpha.md"),
        Chunk("beta-id", "Beta", "beta.md"),
    ]

    retriever.build(chunks)
    results = retriever.search("Alpha", k=1)

    assert (tmp_path / "custom" / "index.bin").exists()
    assert (tmp_path / "custom" / "chunks.json").exists()
    assert not list(tmp_path.rglob("*.pkl"))
    assert len(results) == 1
    assert results[0].chunk.id == "alpha-id"
    assert results[0].score > 0.99

    metadata = json.loads(
        (tmp_path / "custom" / "chunks.json").read_text(encoding="utf-8")
    )
    assert metadata["chunks"][0]["source"] == "alpha.md"
    assert "fingerprint" in metadata
