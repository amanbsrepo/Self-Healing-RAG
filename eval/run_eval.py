from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from self_healing_rag.config import Settings
from self_healing_rag.evaluation import calculate_metrics
from self_healing_rag.models import FALLBACK_ANSWER
from self_healing_rag.runtime import build_runtime


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _source_dicts(chunks) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": item.chunk.id,
            "source": item.chunk.source,
            "snippet": item.chunk.text,
        }
        for item in chunks
    ]


def _single_pass(runtime, question: str) -> dict:
    retrieved = runtime.retriever.search(
        question,
        runtime.settings.top_k,
    )
    plain_chunks = [item.chunk for item in retrieved]
    answer = runtime.backend.generate_answer(question, plain_chunks)
    critique = runtime.critic.evaluate(question, answer, plain_chunks)

    if answer == FALLBACK_ANSWER:
        status = "fallback"
    elif critique.error:
        status = "critic_unavailable"
    elif critique.grounded:
        status = "grounded"
    else:
        status = "ungrounded"

    evidence_ids = {
        chunk_id
        for claim in critique.claims
        if claim.supported
        for chunk_id in claim.evidence_chunk_ids
    }
    if status == "fallback":
        sources = []
    elif evidence_ids:
        sources = _source_dicts(
            [
                item
                for item in retrieved
                if item.chunk.id in evidence_ids
            ]
        )
    else:
        sources = _source_dicts(retrieved)

    return {
        "answer": answer,
        "status": status,
        "attempts": 1,
        "sources": sources,
        "claims": [asdict(claim) for claim in critique.claims],
        "error": critique.error,
    }


def _self_healing(runtime, question: str) -> dict:
    result = runtime.pipeline.run(question)
    return {
        "answer": result.answer,
        "status": result.status,
        "attempts": result.attempts,
        "sources": [asdict(source) for source in result.sources],
        "claims": [asdict(claim) for claim in result.claims],
        "error": result.error,
        "run_id": result.run_id,
    }


def _markdown(
    mode: str,
    mode_label: str,
    metrics: dict,
    records: list[dict],
) -> str:
    lines = [
        "# Self-Healing RAG Evaluation",
        "",
        f"**Mode: {mode_label}**",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        "| Metric | Single-pass RAG | Self-healing RAG |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("Claim-support rate", "claim_support_rate"),
        ("Key-fact hit rate", "key_fact_hit_rate"),
        ("Expected-source hit rate", "expected_source_hit_rate"),
        ("Out-of-scope refusal rate", "out_of_scope_refusal_rate"),
        ("Fallback rate", "fallback_rate"),
    ]:
        lines.append(
            f"| {label} | {metrics['single_pass'][key]:.2f}% | "
            f"{metrics['self_healing'][key]:.2f}% |"
        )
    lines.append(
        "| Average attempts | 1.00 | "
        f"{metrics['self_healing']['average_attempts']:.2f} |"
    )

    lines.extend(
        [
            "",
            "## Detailed results",
            "",
            "| Scope | Question | Single status | Healing status | "
            "Attempts | Healing sources |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for record in records:
        source_names = ", ".join(
            sorted(
                {
                    source["source"]
                    for source in record["self_healing"]["sources"]
                }
            )
        )
        lines.append(
            f"| {record['scope']} | "
            f"{_escape(record['question'])} | "
            f"{record['single_pass']['status']} | "
            f"{record['self_healing']['status']} | "
            f"{record['self_healing']['attempts']} | "
            f"{_escape(source_names)} |"
        )

    lines.extend(
        [
            "",
            f"Exact safe fallback: `{FALLBACK_ANSWER}`",
            "",
            "Stub-mode results use a deterministic retrieval/generation backend "
            "and a heuristic critic. They are not LLM evaluation results.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate single-pass and self-healing RAG."
    )
    parser.add_argument(
        "--mode",
        choices=["stub", "llm"],
        default="stub",
    )
    parser.add_argument(
        "--questions",
        default=str(ROOT / "eval" / "questions.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "eval" / "results"),
    )
    args = parser.parse_args(argv)

    base_settings = Settings.from_env()
    with tempfile.TemporaryDirectory(prefix="self-healing-rag-eval-") as temp:
        temp_path = Path(temp)
        settings = replace(
            base_settings,
            index_path=temp_path / "faiss_index.bin",
            chunks_path=temp_path / "chunks.json",
        )

        if args.mode == "stub":
            settings = replace(
                settings,
                llm_provider="stub",
                embedding_backend="hashing",
            )
            mode_label = "stub mode, heuristic critic"
        else:
            if settings.llm_provider == "stub":
                raise SystemExit(
                    "LLM mode requires LLM_PROVIDER=openai or anthropic "
                    "and the corresponding API key."
                )
            if (
                settings.llm_provider == "openai"
                and not settings.openai_api_key
            ):
                raise SystemExit("OPENAI_API_KEY is required for LLM mode.")
            if (
                settings.llm_provider == "anthropic"
                and not settings.anthropic_api_key
            ):
                raise SystemExit(
                    "ANTHROPIC_API_KEY is required for LLM mode."
                )
            mode_label = "LLM mode"

        runtime = build_runtime(settings, force_rebuild=True)

        question_document = json.loads(
            Path(args.questions).read_text(encoding="utf-8")
        )
        questions = question_document["questions"]
        records: list[dict] = []

        for question_data in questions:
            question = question_data["question"]
            single = _single_pass(runtime, question)
            healing = _self_healing(runtime, question)
            records.append(
                {
                    "id": question_data["id"],
                    "scope": question_data["scope"],
                    "question": question,
                    "expected_sources": question_data.get(
                        "expected_sources",
                        [],
                    ),
                    "key_facts": question_data.get("key_facts", []),
                    "single_pass": single,
                    "self_healing": healing,
                }
            )

    metrics = {
        mode: calculate_metrics(records, mode)
        for mode in ("single_pass", "self_healing")
    }
    report = _markdown(args.mode, mode_label, metrics, records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.mode}_results.json"
    markdown_path = output_dir / f"{args.mode}_results.md"

    json_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "mode_label": mode_label,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"JSON results: {json_path}")
    print(f"Markdown results: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
