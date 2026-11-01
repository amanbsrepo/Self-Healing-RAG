from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


FALLBACK_ANSWER = "I don't have enough information."


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source=str(data["source"]),
            page=int(data.get("page", 1)),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float

    @property
    def id(self) -> str:
        return self.chunk.id


@dataclass
class Claim:
    text: str
    supported: bool
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class CriticResult:
    grounded: bool
    claims: list[Claim] = field(default_factory=list)
    reformulated_query: str = ""
    error: str | None = None
    status: str = "ungrounded"

    @property
    def unsupported_claims(self) -> list[Claim]:
        return [claim for claim in self.claims if not claim.supported]


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class Source:
    chunk_id: str
    source: str
    snippet: str


@dataclass
class PipelineResult:
    query: str
    answer: str
    grounded: bool
    fallback: bool
    attempts: int
    citations: list[Citation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    sources: list[Source] = field(default_factory=list)
    status: str = "unknown"
    critique_status: str = "unknown"
    run_id: str = ""
    attempt_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
