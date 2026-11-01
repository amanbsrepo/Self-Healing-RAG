from __future__ import annotations

from typing import Any


def _percentage(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def calculate_metrics(
    records: list[dict[str, Any]],
    mode: str,
) -> dict[str, float]:
    if mode not in {"single_pass", "self_healing"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")

    total_claims = 0
    supported_claims = 0
    key_fact_hits = 0
    key_fact_total = 0
    expected_source_hits = 0
    expected_source_total = 0
    fallback_count = 0
    out_of_scope_count = 0
    out_of_scope_refusals = 0
    attempt_total = 0

    for record in records:
        result = record[mode]
        claims = result.get("claims", [])
        total_claims += len(claims)
        supported_claims += sum(
            1 for claim in claims if claim.get("supported")
        )
        attempt_total += int(result.get("attempts", 0))

        is_fallback = result.get("status") == "fallback"
        fallback_count += int(is_fallback)

        if record.get("scope") == "out_of_scope":
            out_of_scope_count += 1
            out_of_scope_refusals += int(is_fallback)
        else:
            answer = str(result.get("answer", "")).lower()
            for fact in record.get("key_facts", []):
                key_fact_total += 1
                key_fact_hits += int(str(fact).lower() in answer)

            source_names = {
                str(source.get("source", ""))
                for source in result.get("sources", [])
            }
            for expected_source in record.get("expected_sources", []):
                expected_source_total += 1
                expected_source_hits += int(
                    expected_source in source_names
                )

    return {
        "claim_support_rate": _percentage(
            supported_claims,
            total_claims,
        ),
        "key_fact_hit_rate": _percentage(
            key_fact_hits,
            key_fact_total,
        ),
        "out_of_scope_refusal_rate": _percentage(
            out_of_scope_refusals,
            out_of_scope_count,
        ),
        "fallback_rate": _percentage(
            fallback_count,
            len(records),
        ),
        "average_attempts": round(
            attempt_total / len(records),
            2,
        ) if records else 0.0,
        "expected_source_hit_rate": _percentage(
            expected_source_hits,
            expected_source_total,
        ),
    }
