#!/usr/bin/env python3
"""Build eval/questions_100.json: 25 original + 45 paraphrased in-scope + 30 out-of-scope."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = json.loads((root / "eval" / "questions.json").read_text(encoding="utf-8"))["questions"]
orig_in = [q for q in base if q["scope"] == "in_scope"]
orig_out = [q for q in base if q["scope"] == "out_of_scope"]
by_n = {int(q["id"].split("_")[-1]): q for q in orig_in}

# (original in_scope number, rephrased question)
PARAPHRASES = [
    (1, "What framework orchestrates the RAG workflow?"),
    (1, "Which library is used to control the workflow?"),
    (2, "Which nodes does the StateGraph include?"),
    (2, "What are the nodes in the RAG workflow graph?"),
    (3, "How does the conditional edge make the workflow cycle?"),
    (3, "Why does the graph loop back to retrieval?"),
    (4, "What does the typed RAG state store?"),
    (4, "Which information does the RAG state keep?"),
    (5, "What does the critique node verify?"),
    (5, "Which claims does the critique node check?"),
    (6, "What does the workflow do after an unsupported claim?"),
    (6, "What happens when an answer has an unsupported claim?"),
    (7, "What is the maximum number of generation attempts?"),
    (7, "How many generation attempts are allowed in total?"),
    (8, "Which exact sentence is the safe fallback?"),
    (8, "What sentence does the safe fallback return?"),
    (9, "Which vector database is used by the project?"),
    (9, "What is the vector database used by the project?"),
    (10, "What FAISS index type is created?"),
    (10, "Which type of FAISS index does the project create?"),
    (11, "What is the default embedding model?"),
    (11, "Which embedding model is used by default?"),
    (12, "Why do the embeddings get normalized?"),
    (12, "Why are embeddings normalized?"),
    (13, "How does the project persist the FAISS index?"),
    (13, "Where is the FAISS index persisted?"),
    (14, "How does the project persist chunk metadata?"),
    (14, "Where is chunk metadata persisted?"),
    (15, "Is pickle used for chunk metadata?"),
    (15, "Does the project use pickle?"),
    (16, "What does every chunk include?"),
    (16, "Which information is included with each chunk?"),
    (17, "What does each citation include?"),
    (17, "Which information is included in every citation?"),
    (18, "How does per-claim grounding work?"),
    (18, "How is per-claim grounding done?"),
    (19, "Which file controls the runtime configuration?"),
    (20, "Does the stub provider require an API key?"),
    (21, "Must users upload documents for the application?"),
    (22, "How do I start the command-line application?"),
    (23, "How do I start the Streamlit application?"),
    (24, "How do I run the test suite?"),
    (24, "Which tool runs the test suite?"),
    (25, "What metrics does the evaluation script calculate?"),
    (25, "Which metrics does the evaluation script report?"),
]
assert len(PARAPHRASES) == 45, len(PARAPHRASES)

NEW_OUT = [
    "What is the capital city of France?",
    "How many players are on a cricket team?",
    "What is the boiling point of water at sea level?",
    "Who wrote the play Hamlet?",
    "What is the population of Tokyo?",
    "Which planet is closest to the sun?",
    "How do I bake a chocolate cake?",
    "What is the price of a plane ticket to London?",
    "Who is the current president of Brazil?",
    "What is the exchange rate between dollars and euros today?",
    "How tall is Mount Everest?",
    "What are the symptoms of the common flu?",
    "Which team won the last cricket world cup?",
    "What is my bank account balance?",
    "Tell me the phone number of the nearest hospital.",
    "What movie won the best picture award last year?",
    "How do I change a flat tire on a bicycle?",
    "What is the speed of light in a vacuum?",
    "Who painted the Mona Lisa?",
    "What is the salary of the chief executive officer?",
    "How many calories are in a banana?",
    "What is the next public holiday in India?",
]
assert len(NEW_OUT) == 22

questions = []
for i, q in enumerate(orig_in, 1):
    questions.append({**q, "id": f"in_scope_{i:03d}", "variant": "original"})
for j, (n, text) in enumerate(PARAPHRASES, 26):
    src = by_n[n]
    questions.append({
        "id": f"in_scope_{j:03d}",
        "scope": "in_scope",
        "question": text,
        "expected_sources": src["expected_sources"],
        "key_facts": src["key_facts"],
        "variant": f"paraphrase_of_{n:02d}",
    })
for k, q in enumerate(orig_out, 1):
    questions.append({**q, "id": f"out_of_scope_{k:03d}", "variant": "original"})
for k, text in enumerate(NEW_OUT, len(orig_out) + 1):
    questions.append({"id": f"out_of_scope_{k:03d}", "scope": "out_of_scope",
                      "question": text, "expected_sources": [], "key_facts": [], "variant": "new"})

assert len(questions) == 100, len(questions)
assert len({q["question"] for q in questions}) == 100, "duplicate question text"
out = root / "eval" / "questions_100.json"
out.write_text(json.dumps({"questions": questions}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("written", out, "-", sum(q["scope"] == "in_scope" for q in questions), "in-scope,",
      sum(q["scope"] == "out_of_scope" for q in questions), "out-of-scope")
