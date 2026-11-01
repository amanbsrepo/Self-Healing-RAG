from self_healing_rag.config import Settings
from self_healing_rag.document_loader import (
    load_builtin_documents,
    load_local_documents,
)


def make_settings(tmp_path):
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        documents_dir=tmp_path / "documents",
        index_path=tmp_path / "index.bin",
        chunks_path=tmp_path / "chunks.json",
        chunk_size_words=30,
        chunk_overlap_words=5,
    )


def test_builtin_documents_preserve_source_and_page(tmp_path):
    chunks = load_builtin_documents(make_settings(tmp_path))

    assert chunks
    assert all(chunk.source for chunk in chunks)
    assert all(chunk.page == 1 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_local_text_document_is_chunked_with_metadata(tmp_path):
    document = tmp_path / "local.md"
    document.write_text(
        "First paragraph. Second sentence.\n\n"
        "A separate paragraph with metadata.",
        encoding="utf-8",
    )

    chunks = load_local_documents([document], make_settings(tmp_path))

    assert chunks
    assert all(chunk.source == "local.md" for chunk in chunks)
    assert all(chunk.page == 1 for chunk in chunks)
