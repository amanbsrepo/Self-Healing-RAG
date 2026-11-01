import json
from pathlib import Path

from self_healing_rag.evaluation import calculate_metrics


def test_metric_calculations():
    records = [
        {
            "scope": "in_scope",
            "key_facts": ["alpha"],
            "expected_sources": ["doc.md"],
            "single_pass": {
                "answer": "alpha",
                "status": "grounded",
                "attempts": 1,
                "sources": [
                    {
                        "chunk_id": "c1",
                        "source": "doc.md",
                        "snippet": "alpha",
                    }
                ],
                "claims": [
                    {"text": "alpha", "supported": True}
                ],
            },
            "self_healing": {
                "answer": "alpha",
                "status": "grounded",
                "attempts": 2,
                "sources": [
                    {
                        "chunk_id": "c1",
                        "source": "doc.md",
                        "snippet": "alpha",
                    }
                ],
                "claims": [
                    {"text": "alpha", "supported": True}
                ],
            },
        },
        {
            "scope": "out_of_scope",
            "key_facts": [],
            "expected_sources": [],
            "single_pass": {
                "answer": "invented",
                "status": "ungrounded",
                "attempts": 1,
                "sources": [],
                "claims": [
                    {"text": "invented", "supported": False}
                ],
            },
            "self_healing": {
                "answer": "I don't have enough information.",
                "status": "fallback",
                "attempts": 3,
                "sources": [],
                "claims": [],
            },
        },
    ]

    single = calculate_metrics(records, "single_pass")
    healing = calculate_metrics(records, "self_healing")

    assert single["claim_support_rate"] == 50.0
    assert healing["claim_support_rate"] == 100.0
    assert single["key_fact_hit_rate"] == 100.0
    assert single["out_of_scope_refusal_rate"] == 0.0
    assert healing["out_of_scope_refusal_rate"] == 100.0
    assert single["fallback_rate"] == 0.0
    assert healing["fallback_rate"] == 50.0
    assert healing["average_attempts"] == 2.5


def test_evaluation_question_metadata_has_25_and_8_questions():
    path = Path(__file__).parents[1] / "eval" / "questions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload["questions"]

    in_scope = [
        question
        for question in questions
        if question["scope"] == "in_scope"
    ]
    out_of_scope = [
        question
        for question in questions
        if question["scope"] == "out_of_scope"
    ]

    assert len(in_scope) == 25
    assert len(out_of_scope) == 8
    assert all(
        question["expected_sources"]
        for question in in_scope
    )
    assert all(question["key_facts"] for question in in_scope)
