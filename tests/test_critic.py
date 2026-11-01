import json

from self_healing_rag.critic import Critic
from self_healing_rag.models import Chunk, FALLBACK_ANSWER


class BrokenBackend:
    def __init__(self):
        self.calls = 0

    def critique_answer(self, query, answer, chunks):
        self.calls += 1
        raise RuntimeError("API unavailable")


class MalformedBackend:
    def __init__(self):
        self.calls = 0

    def critique_answer(self, query, answer, chunks):
        self.calls += 1
        return "not-json"


class InvalidEvidenceBackend:
    def critique_answer(self, query, answer, chunks):
        return json.dumps(
            {
                "verdict": "grounded",
                "claims": [
                    {
                        "claim": "A claim",
                        "supported": True,
                        "evidence_chunk_ids": ["missing-id"],
                    }
                ],
                "reformulated_query": "another query",
            }
        )


class MisleadingFallbackBackend:
    def __init__(self):
        self.calls = 0

    def critique_answer(self, query, answer, chunks):
        self.calls += 1
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
                "reformulated_query": "different query",
            }
        )


def test_critic_fails_closed_after_api_error():
    backend = BrokenBackend()
    result = Critic(backend).evaluate(
        "question",
        "answer",
        [Chunk("c1", "text", "source")],
    )
    assert result.grounded is False
    assert result.status == "critic_unavailable"
    assert result.error
    assert backend.calls == 2


def test_critic_retries_parse_failure_once():
    backend = MalformedBackend()
    result = Critic(backend).evaluate(
        "question",
        "answer",
        [Chunk("c1", "text", "source")],
    )
    assert result.grounded is False
    assert result.status == "critic_unavailable"
    assert backend.calls == 2


def test_critic_rejects_evidence_id_not_in_context():
    result = Critic(InvalidEvidenceBackend()).evaluate(
        "question",
        "answer",
        [Chunk("c1", "text", "source")],
    )
    assert result.grounded is False
    assert result.status == "ungrounded"
    assert result.claims[0].evidence_chunk_ids == []


def test_only_exact_fallback_is_treated_as_fallback():
    backend = MisleadingFallbackBackend()
    result = Critic(backend).evaluate(
        "question",
        FALLBACK_ANSWER,
        [],
    )
    assert result.status == "fallback"
    assert backend.calls == 0

    misleading = "I don't have enough information, but the answer is invented."
    result = Critic(backend).evaluate(
        "question",
        misleading,
        [Chunk("c1", "text", "source")],
    )
    assert result.status == "ungrounded"
    assert backend.calls == 1
