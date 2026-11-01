from __future__ import annotations

import json
from typing import Any, Sequence

from .llm import LLMBackend
from .logger import get_logger
from .models import Chunk, Claim, CriticResult, FALLBACK_ANSWER


logger = get_logger(__name__)


def _extract_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        payload = raw
    else:
        text = raw.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```")
            text = text.removesuffix("```").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Critic response did not contain a JSON object")
        payload = json.loads(text[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Critic response must be a JSON object")
    if not isinstance(payload.get("claims"), list):
        raise ValueError("Critic response must contain a claims list")
    if not payload["claims"]:
        raise ValueError("Critic response contains no claims")

    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"grounded", "ungrounded"}:
        raise ValueError("Critic verdict must be grounded or ungrounded")
    return payload


class Critic:
    def __init__(self, backend: LLMBackend, backend_retries: int = 1) -> None:
        self.backend = backend
        self.backend_retries = max(0, backend_retries)

    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: Sequence[Chunk],
    ) -> CriticResult:
        # Only the exact shared sentence is the safe fallback.
        if answer == FALLBACK_ANSWER:
            return CriticResult(
                grounded=False,
                claims=[],
                status="fallback",
            )

        if not answer.strip():
            return CriticResult(
                grounded=False,
                claims=[],
                error="Generation returned an empty answer",
                status="ungrounded",
            )

        valid_ids = {chunk.id for chunk in chunks}
        last_error: str | None = None

        for attempt in range(self.backend_retries + 1):
            try:
                raw = self.backend.critique_answer(query, answer, chunks)
                payload = _extract_json(raw)
            except Exception as exc:
                last_error = f"Critic call/parse error: {exc}"
                logger.warning(
                    "Critic attempt %s failed: %s",
                    attempt + 1,
                    exc,
                )
                continue

            claims: list[Claim] = []
            for item in payload["claims"]:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("claim", "")).strip()
                if not text:
                    continue

                supported = item.get("supported") is True
                raw_ids = item.get("evidence_chunk_ids", [])
                if not isinstance(raw_ids, list):
                    raw_ids = []
                evidence_ids = [
                    str(chunk_id)
                    for chunk_id in raw_ids
                    if str(chunk_id) in valid_ids
                ]

                if supported and not evidence_ids:
                    supported = False
                if not supported:
                    evidence_ids = []

                claims.append(
                    Claim(
                        text=text,
                        supported=supported,
                        evidence_chunk_ids=list(dict.fromkeys(evidence_ids)),
                    )
                )

            if not claims:
                last_error = "Critic returned no usable claims"
                continue

            grounded = (
                str(payload["verdict"]).lower() == "grounded"
                and all(claim.supported for claim in claims)
            )
            reformulated = payload.get("reformulated_query", "")
            if not isinstance(reformulated, str):
                reformulated = ""

            return CriticResult(
                grounded=grounded,
                claims=claims,
                reformulated_query=reformulated.strip(),
                error=None,
                status="grounded" if grounded else "ungrounded",
            )

        return CriticResult(
            grounded=False,
            claims=[],
            reformulated_query="",
            error=last_error or "Critic failed after retries",
            status="critic_unavailable",
        )
