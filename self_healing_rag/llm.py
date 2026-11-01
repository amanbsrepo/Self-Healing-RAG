from __future__ import annotations

import json
import re
from typing import Any, Protocol, Sequence

from .config import Settings
from .logger import get_logger
from .models import Chunk, FALLBACK_ANSWER


logger = get_logger(__name__)

# The stub answers only when its best sentence covers at least this share of
# the meaningful words in the question.
MIN_QUERY_COVERAGE = 0.4

_STOPWORDS = {
    "the", "and", "for", "with", "what", "which", "how", "does", "did", "are",
    "was", "were", "who", "whom", "why", "when", "where", "this", "that",
    "these", "those", "from", "into", "onto", "about", "have", "has", "had",
    "its", "can", "will", "would", "should", "could", "not", "any", "all",
    "each", "every", "per", "use", "used", "uses", "using", "there", "their",
    "them", "they", "then", "than", "also", "one", "two", "you", "your", "our",
    "out", "many", "much",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _raw_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
    ]


def _tokens(text: str) -> set[str]:
    return {_stem(token) for token in _raw_tokens(text) if len(token) >= 2}


def _meaningful_tokens(text: str) -> set[str]:
    return {
        _stem(token)
        for token in _raw_tokens(text)
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


class LLMBackend(Protocol):
    def generate_answer(self, query: str, chunks: Sequence[Chunk]) -> str:
        """Return an answer that cites chunk IDs in square brackets."""

    def critique_answer(
        self,
        query: str,
        answer: str,
        chunks: Sequence[Chunk],
    ) -> str:
        """Return a JSON string with verdict, claims and reformulated_query."""


class StubBackend:
    """Deterministic offline backend with a heuristic critic."""

    def generate_answer(self, query: str, chunks: Sequence[Chunk]) -> str:
        if not chunks:
            return FALLBACK_ANSWER

        query_tokens = _meaningful_tokens(query)
        if not query_tokens:
            return FALLBACK_ANSWER

        scored: list[tuple[int, Chunk, str]] = []
        for chunk in chunks:
            for sentence in _split_sentences(chunk.text):
                overlap = len(query_tokens & _tokens(sentence))
                scored.append((overlap, chunk, sentence))
        if not scored:
            return FALLBACK_ANSWER

        best_score, best_chunk, _ = max(scored, key=lambda item: item[0])
        if best_score / len(query_tokens) < MIN_QUERY_COVERAGE:
            return FALLBACK_ANSWER

        ranked_sentences = sorted(
            _split_sentences(best_chunk.text),
            key=lambda sentence: len(query_tokens & _tokens(sentence)),
            reverse=True,
        )
        selected = [
            sentence
            for sentence in ranked_sentences
            if query_tokens & _tokens(sentence)
        ][:3]
        return " ".join(f"{sentence} [{best_chunk.id}]" for sentence in selected)

    def critique_answer(
        self,
        query: str,
        answer: str,
        chunks: Sequence[Chunk],
    ) -> str:
        claims: list[dict[str, Any]] = []
        corpus_tokens: set[str] = set()
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        for chunk in chunks:
            corpus_tokens |= _tokens(chunk.text)

        for sentence in _split_sentences(answer):
            cited_ids = re.findall(r"\[([^\]]+)\]", sentence)
            clean = re.sub(r"\[[^\]]+\]", "", sentence).strip()
            if not clean:
                continue

            claim_tokens = _meaningful_tokens(clean)
            candidates = [
                chunk_by_id[chunk_id]
                for chunk_id in cited_ids
                if chunk_id in chunk_by_id
            ]
            if not candidates:
                candidates = sorted(
                    chunks,
                    key=lambda chunk: len(claim_tokens & _tokens(chunk.text)),
                    reverse=True,
                )[:1]

            evidence = [
                candidate.id
                for candidate in candidates
                if len(claim_tokens & _tokens(candidate.text))
                / max(1, len(claim_tokens))
                >= 0.5
            ]
            claims.append(
                {
                    "claim": clean,
                    "supported": bool(evidence),
                    "evidence_chunk_ids": evidence,
                }
            )

        unsupported = [claim for claim in claims if not claim["supported"]]
        unmatched: set[str] = set()
        for claim in unsupported:
            unmatched |= _meaningful_tokens(claim["claim"]) - corpus_tokens
        unmatched -= _meaningful_tokens(query)

        if unmatched:
            reformulated = f"{query} {' '.join(sorted(unmatched)[:10])}"
        else:
            reformulated = f"{query} source evidence"

        return json.dumps(
            {
                "verdict": (
                    "grounded" if claims and not unsupported else "ungrounded"
                ),
                "claims": claims,
                "reformulated_query": reformulated,
            }
        )


_GENERATE_SYSTEM = (
    "You answer questions using ONLY the context chunks provided by the user. "
    "Cite the chunk IDs you used in square brackets, for example [chunk-id]. "
    "If the context does not contain the answer, reply with exactly: "
    f"{FALLBACK_ANSWER}"
)

_CRITIQUE_SYSTEM = (
    "You are a strict fact checker. Split the ANSWER into atomic factual "
    "claims. For each claim decide whether the CONTEXT chunks fully support "
    "it. Respond with ONLY a JSON object of this shape: "
    '{"verdict": "grounded" or "ungrounded", '
    '"claims": [{"claim": "...", "supported": true or false, '
    '"evidence_chunk_ids": ["..."]}], '
    '"reformulated_query": "a better search query for the missing evidence"}. '
    "Use verdict grounded only if every claim is supported. "
    "evidence_chunk_ids must be chunk IDs that appear in the CONTEXT."
)


def _context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in chunks)


def _generate_prompt(query: str, chunks: Sequence[Chunk]) -> str:
    return f"CONTEXT:\n{_context(chunks)}\n\nQUESTION: {query}"


def _critique_prompt(
    query: str,
    answer: str,
    chunks: Sequence[Chunk],
) -> str:
    return (
        f"CONTEXT:\n{_context(chunks)}\n\n"
        f"QUESTION: {query}\n\nANSWER:\n{answer}"
    )


class OpenAIBackend:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def _complete(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()

    def generate_answer(self, query: str, chunks: Sequence[Chunk]) -> str:
        if not chunks:
            return FALLBACK_ANSWER
        return self._complete(_GENERATE_SYSTEM, _generate_prompt(query, chunks))

    def critique_answer(
        self,
        query: str,
        answer: str,
        chunks: Sequence[Chunk],
    ) -> str:
        return self._complete(
            _CRITIQUE_SYSTEM,
            _critique_prompt(query, answer, chunks),
            json_mode=True,
        )


class AnthropicBackend:
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()

    def generate_answer(self, query: str, chunks: Sequence[Chunk]) -> str:
        if not chunks:
            return FALLBACK_ANSWER
        return self._complete(_GENERATE_SYSTEM, _generate_prompt(query, chunks))

    def critique_answer(
        self,
        query: str,
        answer: str,
        chunks: Sequence[Chunk],
    ) -> str:
        return self._complete(
            _CRITIQUE_SYSTEM,
            _critique_prompt(query, answer, chunks),
        )


def create_backend(settings: Settings) -> LLMBackend:
    provider = settings.llm_provider
    if provider == "stub":
        return StubBackend()
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        return OpenAIBackend(settings.openai_api_key, settings.openai_model)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic"
            )
        return AnthropicBackend(
            settings.anthropic_api_key,
            settings.anthropic_model,
        )
    raise RuntimeError(f"Unknown LLM provider: {provider}")
