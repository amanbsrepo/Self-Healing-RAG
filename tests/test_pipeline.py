import json

from self_healing_rag.config import Settings
from self_healing_rag.critic import Critic
from self_healing_rag.llm import StubBackend
from self_healing_rag.models import Chunk, FALLBACK_ANSWER, RetrievedChunk
from self_healing_rag.pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, chunks=None):
        self.calls = []
        self.chunks = chunks

    def search(self, query, k):
        self.calls.append((query, k))
        if self.chunks is None:
            return [
                RetrievedChunk(
                    chunk=Chunk(
                        "c1",
                        "LangGraph is a stateful graph framework.",
                        "test.md",
                    ),
                    score=0.9,
                )
            ]
        return self.chunks


class AlwaysUnsupportedBackend:
    def __init__(self):
        self.generations = 0

    def generate_answer(self, query, chunks):
        self.generations += 1
        return "This answer is unsupported."

    def critique_answer(self, query, answer, chunks):
        return json.dumps(
            {
                "verdict": "ungrounded",
                "claims": [
                    {
                        "claim": "This answer is unsupported.",
                        "supported": False,
                        "evidence_chunk_ids": [],
                    }
                ],
                "reformulated_query": "broader evidence",
            }
        )


class GroundedOnSecondBackend:
    def __init__(self):
        self.generations = 0

    def generate_answer(self, query, chunks):
        self.generations += 1
        if self.generations == 1:
            return "Unsupported first answer."
        return "LangGraph is a stateful graph framework. [c1]"

    def critique_answer(self, query, answer, chunks):
        if answer.startswith("Unsupported"):
            return json.dumps(
                {
                    "verdict": "ungrounded",
                    "claims": [
                        {
                            "claim": answer,
                            "supported": False,
                            "evidence_chunk_ids": [],
                        }
                    ],
                    "reformulated_query": "LangGraph stateful framework",
                }
            )
        return json.dumps(
            {
                "verdict": "grounded",
                "claims": [
                    {
                        "claim": (
                            "LangGraph is a stateful graph framework."
                        ),
                        "supported": True,
                        "evidence_chunk_ids": ["c1"],
                    }
                ],
                "reformulated_query": "unused query",
            }
        )


class BrokenCriticBackend:
    def generate_answer(self, query, chunks):
        return "An answer that cannot be checked."

    def critique_answer(self, query, answer, chunks):
        raise RuntimeError("critic unavailable")


class MisleadingFallbackBackend:
    def generate_answer(self, query, chunks):
        return (
            "I don't have enough information, but the answer is invented."
        )

    def critique_answer(self, query, answer, chunks):
        return json.dumps(
            {
                "verdict": "ungrounded",
                "claims": [
                    {
                        "claim": answer,
                        "supported": False,
                        "evidence_chunk_ids": [],
                    }
                ],
                "reformulated_query": "invented claim evidence",
            }
        )


def make_settings(tmp_path):
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        documents_dir=tmp_path / "documents",
        index_path=tmp_path / "index.bin",
        chunks_path=tmp_path / "chunks.json",
        top_k=2,
        max_top_k=4,
        max_attempts=3,
        llm_provider="stub",
    )


def test_hallucination_falls_back_after_exactly_three_generations(tmp_path):
    retriever = FakeRetriever()
    backend = AlwaysUnsupportedBackend()
    pipeline = RAGPipeline(
        settings=make_settings(tmp_path),
        retriever=retriever,
        backend=backend,
        critic=Critic(backend),
    )

    result = pipeline.run("What is unsupported?")

    assert result.fallback is True
    assert result.answer == FALLBACK_ANSWER
    assert result.attempts == 3
    assert backend.generations == 3
    assert len(result.sources) == 0
    assert [call[1] for call in retriever.calls] == [2, 2, 4]
    assert retriever.calls[1][0] == "broader evidence"
    assert retriever.calls[2][0] == "broader evidence"


def test_grounded_answer_on_attempt_two_has_sources(tmp_path):
    retriever = FakeRetriever()
    backend = GroundedOnSecondBackend()
    pipeline = RAGPipeline(
        settings=make_settings(tmp_path),
        retriever=retriever,
        backend=backend,
        critic=Critic(backend),
    )

    result = pipeline.run("What is LangGraph?")

    assert result.grounded is True
    assert result.fallback is False
    assert result.status == "grounded"
    assert result.attempts == 2
    assert result.sources[0].chunk_id == "c1"
    assert "final" in [event["node"] for event in result.trace]


def test_critic_error_is_fail_closed(tmp_path):
    retriever = FakeRetriever()
    backend = BrokenCriticBackend()
    pipeline = RAGPipeline(
        settings=make_settings(tmp_path),
        retriever=retriever,
        backend=backend,
        critic=Critic(backend),
    )

    result = pipeline.run("Check this answer.")

    assert result.fallback is True
    assert result.answer == FALLBACK_ANSWER
    assert result.critique_status == "critic_unavailable"
    assert result.sources == []


def test_misleading_fallback_sentence_is_not_accepted(tmp_path):
    retriever = FakeRetriever()
    backend = MisleadingFallbackBackend()
    pipeline = RAGPipeline(
        settings=make_settings(tmp_path),
        retriever=retriever,
        backend=backend,
        critic=Critic(backend),
    )

    result = pipeline.run("Give me a misleading answer.")

    assert result.attempts == 3
    assert result.fallback is True
    assert result.answer == FALLBACK_ANSWER


def test_empty_index_returns_safe_fallback(tmp_path):
    retriever = FakeRetriever(chunks=[])
    backend = StubBackend()
    pipeline = RAGPipeline(
        settings=make_settings(tmp_path),
        retriever=retriever,
        backend=backend,
        critic=Critic(backend),
    )

    result = pipeline.run("Anything?")

    assert result.fallback is True
    assert result.answer == FALLBACK_ANSWER
    assert result.sources == []


def test_empty_query_is_rejected(tmp_path):
    pipeline = RAGPipeline(
        settings=make_settings(tmp_path),
        retriever=FakeRetriever(),
        backend=StubBackend(),
        critic=Critic(StubBackend()),
    )

    try:
        pipeline.run("   ")
    except ValueError as exc:
        assert "Query must not be empty" in str(exc)
    else:
        raise AssertionError("Empty query should raise ValueError")
