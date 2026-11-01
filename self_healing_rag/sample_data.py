"""Built-in documents so the application works without any upload."""

_RAW_DOCUMENTS: list[tuple[str, str]] = [
    (
        "rag_architecture.md",
        """
The self-healing RAG pipeline answers a question in several stages and checks
its own answer before returning it. The LangGraph StateGraph framework controls
the RAG workflow. The StateGraph includes the nodes retrieve, generate,
critique, final, and fallback. The typed RAG state stores the original question,
current search query, retrieved chunks, the generated answer, and the critique
result.

The critique node checks the factual claims against the retrieved chunks. A
conditional edge creates a cycle by sending an unsupported answer back to
retrieval. After an answer has an unsupported claim, the workflow goes back to
retrieval with a changed query. The maximum number of total generation attempts
is three. The safe fallback returns the exact sentence: I don't have enough
information.

The self-healing RAG workflow decides to retry when the critique finds an
unsupported claim and attempts remain. The retry strategy uses the original
query first, then the critic's reformulated query, and doubles the retrieval
depth on the third attempt.
""",
    ),
    (
        "vector_store.md",
        """
FAISS is the vector database used by the project. The FAISS index type created
is IndexFlatIP, an exact inner-product index. The default embedding model used
is sentence-transformers/all-MiniLM-L6-v2. The embeddings are normalized so that
the inner product equals cosine similarity.

The FAISS index is persisted with faiss.write_index to data/faiss_index.bin.
Chunk metadata is persisted as JSON in data/chunks.json. The project does not
use pickle for chunk metadata. Every chunk includes a unique ID, source name,
page number, and text.
""",
    ),
    (
        "quality_and_evaluation.md",
        """
Each citation includes the chunk ID, source filename, page, text, and
similarity score. Per-claim grounding checks every claim and marks it as
supported or unsupported with the supporting chunks.

The test suite is run with pytest. The evaluation script calculates the
claim-support rate, key-fact hit rate, expected-source hit rate, out-of-scope
refusal rate, fallback rate, and average attempts.
""",
    ),
    (
        "operations_guide.md",
        """
The file self_healing_rag/config.py controls the runtime configuration. The stub
provider requires no API key. The application does not require the user to
upload any documents. The command-line application is started with python
main.py. The Streamlit application is started with streamlit run app.py.
""",
    ),
]


def _flow(text: str) -> str:
    """Join hard-wrapped lines so that every paragraph is a single line."""
    return "\n\n".join(" ".join(part.split()) for part in text.strip().split("\n\n"))


BUILTIN_DOCUMENTS: list[tuple[str, str]] = [
    (source, _flow(text)) for source, text in _RAW_DOCUMENTS
]
