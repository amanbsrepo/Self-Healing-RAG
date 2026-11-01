from __future__ import annotations

import re
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .critic import Critic
from .llm import LLMBackend
from .logger import get_logger
from .models import (
    Citation,
    Claim,
    FALLBACK_ANSWER,
    PipelineResult,
    RetrievedChunk,
    Source,
)
from .retriever import FAISSRetriever


logger = get_logger(__name__)


class RAGState(TypedDict, total=False):
    run_id: str
    original_query: str
    query: str
    pending_query: str
    reformulated_query: str
    attempt: int
    k: int
    chunks: list[RetrievedChunk]
    answer: str
    claims: list[Claim]
    unsupported_claims: list[Claim]
    grounded: bool
    fallback: bool
    status: str
    critique_status: str
    citations: list[Citation]
    sources: list[Source]
    trace: list[dict[str, Any]]
    attempt_trace: list[dict[str, Any]]
    retrieval_error: str
    generation_error: str
    critic_error: str
    error: str


def _normalized_words(value: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)
        if len(word) >= 3
    }


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: FAISSRetriever,
        backend: LLMBackend,
        critic: Critic,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.backend = backend
        self.critic = critic
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("critique", self._critique)
        workflow.add_node("final", self._final)
        workflow.add_node("fallback", self._fallback)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "critique")
        workflow.add_conditional_edges(
            "critique",
            self._route,
            {
                "retrieve": "retrieve",
                "final": "final",
                "fallback": "fallback",
            },
        )
        workflow.add_edge("final", END)
        workflow.add_edge("fallback", END)
        return workflow.compile()

    def _trace(
        self,
        state: RAGState,
        node: str,
        started: float,
        **details: Any,
    ) -> list[dict[str, Any]]:
        event = {
            "node": node,
            "attempt": state.get("attempt", 0),
            "query": state.get(
                "query",
                state.get("original_query", ""),
            ),
            "top_k": state.get("k", self.settings.top_k),
            "grounding_verdict": state.get("status", "pending"),
            "critique": state.get("critique_status", "pending"),
            "latency_ms": round(
                (time.perf_counter() - started) * 1000,
                2,
            ),
            **details,
        }
        return state.get("trace", []) + [event]

    def _update_attempt_trace(
        self,
        state: RAGState,
        attempt: int,
        **values: Any,
    ) -> list[dict[str, Any]]:
        records = [dict(record) for record in state.get("attempt_trace", [])]
        for record in records:
            if record.get("attempt") == attempt:
                record.update(values)
                break
        else:
            record = {
                "attempt": attempt,
                "query": state.get("query", ""),
                "top_k": state.get("k", self.settings.top_k),
                "retrieved_chunks": [],
                "answer": "",
                "grounding_verdict": "pending",
                "critique": "pending",
                "unsupported_claims": [],
                "strategy": "initial",
            }
            record.update(values)
            records.append(record)
        return records

    def _retrieve(self, state: RAGState) -> dict[str, Any]:
        started = time.perf_counter()
        attempt = state.get("attempt", 0) + 1
        current = (
            state.get("pending_query")
            or state.get("reformulated_query")
            or state.get("query")
            or state.get("original_query", "")
        ).strip()

        if attempt == 1:
            strategy = "initial"
            k = self.settings.top_k
        elif attempt == 2:
            strategy = "critic_reformulated_query"
            k = self.settings.top_k
        else:
            strategy = "critic_reformulated_query_and_double_k"
            k = self.settings.top_k * 2

        logger.info(
            "Retrieval strategy run_id=%s attempt=%s strategy=%s "
            "query=%r top_k=%s",
            state.get("run_id", ""),
            attempt,
            strategy,
            current,
            k,
        )

        try:
            chunks = self.retriever.search(current, k)
            retrieval_error = ""
        except Exception as exc:
            logger.exception("Retrieval failed")
            chunks = []
            retrieval_error = f"Retrieval error: {exc}"

        retrieved_sources = [
            {
                "chunk_id": item.chunk.id,
                "source": item.chunk.source,
                "snippet": item.chunk.text[:400],
                "score": item.score,
            }
            for item in chunks
        ]

        updated: dict[str, Any] = {
            "query": current,
            "pending_query": "",
            "k": k,
            "chunks": chunks,
            "retrieval_error": retrieval_error,
            "error": retrieval_error or state.get("error", ""),
        }
        updated["attempt_trace"] = self._update_attempt_trace(
            state,
            attempt,
            query=current,
            top_k=k,
            strategy=strategy,
            retrieved_chunks=retrieved_sources,
        )
        updated["trace"] = self._trace(
            state,
            "retrieve",
            started,
            attempt=attempt,
            query=current,
            top_k=k,
            strategy=strategy,
            retrieved_chunks=len(chunks),
        )
        return updated

    def _generate(self, state: RAGState) -> dict[str, Any]:
        started = time.perf_counter()
        attempt = state.get("attempt", 0) + 1
        chunks = [item.chunk for item in state.get("chunks", [])]
        query = state.get("query", state.get("original_query", ""))

        try:
            answer = self.backend.generate_answer(query, chunks).strip()
            generation_error = ""
        except Exception as exc:
            logger.exception("Generation failed")
            answer = ""
            generation_error = f"Generation error: {exc}"

        updated: dict[str, Any] = {
            "attempt": attempt,
            "answer": answer,
            "generation_error": generation_error,
            "error": generation_error or state.get("error", ""),
        }
        updated["attempt_trace"] = self._update_attempt_trace(
            state,
            attempt,
            answer=answer,
        )
        updated["trace"] = self._trace(
            state,
            "generate",
            started,
            attempt=attempt,
            answer_length=len(answer),
        )
        return updated

    def _reformulated_query(
        self,
        state: RAGState,
        result,
        current_query: str,
    ) -> str:
        candidate = result.reformulated_query.strip()
        current_words = _normalized_words(current_query)
        original = state.get("original_query", current_query)

        if candidate and _normalized_words(candidate) != current_words:
            return candidate

        unsupported_text = " ".join(
            claim.text for claim in result.unsupported_claims
        )
        terms = sorted(
            _normalized_words(unsupported_text) - current_words
        )
        if terms:
            return f"{original} {' '.join(terms[:10])}"
        return f"{original} specific source evidence"

    def _critique(self, state: RAGState) -> dict[str, Any]:
        started = time.perf_counter()
        attempt = state.get("attempt", 0)
        query = state.get("query", state.get("original_query", ""))
        answer = state.get("answer", "")
        chunks = [item.chunk for item in state.get("chunks", [])]

        try:
            result = self.critic.evaluate(query, answer, chunks)
        except Exception as exc:
            logger.exception("Unexpected critic failure")
            result = type(
                "FailedCriticResult",
                (),
                {
                    "grounded": False,
                    "claims": [],
                    "unsupported_claims": [],
                    "reformulated_query": "",
                    "error": f"Unexpected critic error: {exc}",
                    "status": "critic_unavailable",
                },
            )()

        fallback = result.status == "fallback"
        pending_query = ""
        reformulated_query = state.get("reformulated_query", "")

        if result.status == "ungrounded":
            if attempt == 1 or not reformulated_query:
                reformulated_query = self._reformulated_query(
                    state,
                    result,
                    query,
                )
            pending_query = reformulated_query

        retrieved_by_id = {
            item.chunk.id: item for item in state.get("chunks", [])
        }
        evidence_ids: list[str] = []
        for claim in result.claims:
            if claim.supported:
                evidence_ids.extend(claim.evidence_chunk_ids)

        citations: list[Citation] = []
        for chunk_id in dict.fromkeys(evidence_ids):
            retrieved = retrieved_by_id.get(chunk_id)
            if retrieved is None:
                continue
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    source=retrieved.chunk.source,
                    page=retrieved.chunk.page,
                    text=retrieved.chunk.text,
                    score=retrieved.score,
                )
            )

        if result.error:
            critique_summary = f"critic_unavailable: {result.error}"
        elif result.status == "fallback":
            critique_summary = "exact fallback sentence"
        elif result.grounded:
            critique_summary = f"{len(result.claims)} claims supported"
        else:
            critique_summary = (
                f"{len(result.unsupported_claims)} unsupported claims"
            )

        critic_error = result.error or ""
        updated: dict[str, Any] = {
            "grounded": result.grounded,
            "fallback": fallback,
            "status": result.status,
            "critique_status": result.status,
            "claims": result.claims,
            "unsupported_claims": result.unsupported_claims,
            "citations": citations,
            "pending_query": pending_query,
            "reformulated_query": reformulated_query,
            "critic_error": critic_error,
            "error": critic_error or state.get("error", ""),
        }
        updated["attempt_trace"] = self._update_attempt_trace(
            state,
            attempt,
            grounding_verdict=result.status,
            critique=critique_summary,
            unsupported_claims=[
                claim.text for claim in result.unsupported_claims
            ],
        )
        updated["trace"] = self._trace(
            state,
            "critique",
            started,
            attempt=attempt,
            grounding_verdict=result.status,
            critique=critique_summary,
            unsupported_claims=len(result.unsupported_claims),
            next_query=pending_query or None,
        )
        return updated

    def _route(self, state: RAGState) -> str:
        if state.get("fallback"):
            return "fallback"
        if state.get("grounded"):
            return "final"
        if (
            state.get("retrieval_error")
            or state.get("generation_error")
            or state.get("critic_error")
        ):
            return "fallback"
        if state.get("attempt", 0) < self.settings.max_attempts:
            return "retrieve"
        return "fallback"

    def _final(self, state: RAGState) -> dict[str, Any]:
        started = time.perf_counter()
        sources: list[Source] = []
        if state.get("citations"):
            for citation in state["citations"]:
                sources.append(
                    Source(
                        chunk_id=citation.chunk_id,
                        source=citation.source,
                        snippet=citation.text,
                    )
                )
        else:
            # Evidence is unavailable: use the retrieved chunks as a fallback
            # source list for a grounded final answer.
            for item in state.get("chunks", []):
                sources.append(
                    Source(
                        chunk_id=item.chunk.id,
                        source=item.chunk.source,
                        snippet=item.chunk.text,
                    )
                )

        updated: dict[str, Any] = {
            "status": "grounded",
            "fallback": False,
            "sources": sources,
        }
        updated["trace"] = self._trace(
            state,
            "final",
            started,
            grounding_verdict="grounded",
            critique=state.get("critique_status", "grounded"),
            source_count=len(sources),
        )
        return updated

    def _fallback(self, state: RAGState) -> dict[str, Any]:
        started = time.perf_counter()
        updated: dict[str, Any] = {
            "answer": FALLBACK_ANSWER,
            "grounded": False,
            "fallback": True,
            "status": "fallback",
            "citations": [],
            "sources": [],
        }
        updated["trace"] = self._trace(
            state,
            "fallback",
            started,
            grounding_verdict="fallback",
            critique=state.get("critique_status", "unavailable"),
            reason=(
                state.get("critic_error")
                or state.get("generation_error")
                or state.get("retrieval_error")
                or "attempt cap"
            ),
        )
        return updated

    def run(self, query: str) -> PipelineResult:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")

        run_id = uuid.uuid4().hex
        initial_state: RAGState = {
            "run_id": run_id,
            "original_query": query,
            "query": query,
            "pending_query": "",
            "reformulated_query": "",
            "attempt": 0,
            "chunks": [],
            "answer": "",
            "claims": [],
            "unsupported_claims": [],
            "grounded": False,
            "fallback": False,
            "status": "pending",
            "critique_status": "pending",
            "citations": [],
            "sources": [],
            "trace": [],
            "attempt_trace": [],
            "retrieval_error": "",
            "generation_error": "",
            "critic_error": "",
            "error": "",
        }

        final_state = self.graph.invoke(
            initial_state,
            config={
                "recursion_limit": max(
                    20,
                    self.settings.max_attempts * 5,
                )
            },
        )

        return PipelineResult(
            query=query,
            answer=final_state.get("answer", ""),
            grounded=bool(final_state.get("grounded", False)),
            fallback=bool(final_state.get("fallback", False)),
            attempts=int(final_state.get("attempt", 0)),
            citations=final_state.get("citations", []),
            claims=final_state.get("claims", []),
            retrieved_chunks=final_state.get("chunks", []),
            trace=final_state.get("trace", []),
            error=final_state.get("error") or None,
            sources=(
                []
                if final_state.get("fallback")
                else final_state.get("sources", [])
            ),
            status=final_state.get("status", "unknown"),
            critique_status=final_state.get(
                "critique_status",
                "unknown",
            ),
            run_id=run_id,
            attempt_trace=final_state.get("attempt_trace", []),
        )
