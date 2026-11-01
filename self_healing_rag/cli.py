from __future__ import annotations

import argparse
import json

from .logger import configure_logging, get_logger
from .runtime import build_runtime


logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the self-healing RAG pipeline."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask. If omitted, the question is read from stdin.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the per-run trace.",
    )
    args = parser.parse_args(argv)
    configure_logging()

    query = " ".join(args.query).strip()
    if not query:
        query = input("Question: ").strip()
    if not query:
        logger.error("A non-empty question is required.")
        return 2

    try:
        runtime = build_runtime()
        result = runtime.pipeline.run(query)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    logger.info("\nAnswer:\n%s", result.answer)
    logger.info("\nStatus: %s", result.status)
    logger.info("Attempts: %s", result.attempts)
    logger.info("Grounded: %s", result.grounded)
    logger.info("Fallback: %s", result.fallback)
    logger.info("Run ID: %s", result.run_id)

    if result.sources:
        logger.info("\nSources:")
        for source in result.sources:
            logger.info(
                "- [%s] %s: %s",
                source.chunk_id,
                source.source,
                source.snippet,
            )

    if args.verbose:
        logger.info("\nTrace:")
        logger.info(json.dumps(result.trace, indent=2, default=str))
        logger.info("\nAttempt trace:")
        logger.info(
            json.dumps(
                result.attempt_trace,
                indent=2,
                default=str,
            )
        )
    return 0
