# Self-Healing RAG Evaluation

**Mode: stub mode, heuristic critic**

Generated: 2026-10-30T07:23:34.402783+00:00

## Summary

| Metric | Single-pass RAG | Self-healing RAG |
|---|---:|---:|
| Claim-support rate | 100.00% | 100.00% |
| Key-fact hit rate | 100.00% | 100.00% |
| Expected-source hit rate | 100.00% | 100.00% |
| Out-of-scope refusal rate | 100.00% | 100.00% |
| Fallback rate | 24.24% | 24.24% |
| Average attempts | 1.00 | 1.00 |

## Detailed results

| Scope | Question | Single status | Healing status | Attempts | Healing sources |
|---|---|---|---:|---:|---|
| in_scope | Which framework controls the RAG workflow? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Which nodes are included in the StateGraph? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | How does the conditional edge create a cycle? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What information is stored in the typed RAG state? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What does the critique node check? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What happens after an answer has an unsupported claim? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | How many total generation attempts are allowed? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What exact sentence is returned as the safe fallback? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Which vector database does the project use? | grounded | grounded | 1 | vector_store.md |
| in_scope | Which FAISS index type is created? | grounded | grounded | 1 | vector_store.md |
| in_scope | Which default embedding model is used? | grounded | grounded | 1 | vector_store.md |
| in_scope | Why are the embeddings normalized? | grounded | grounded | 1 | vector_store.md |
| in_scope | How is the FAISS index persisted? | grounded | grounded | 1 | vector_store.md |
| in_scope | How is chunk metadata persisted? | grounded | grounded | 1 | vector_store.md |
| in_scope | Does the project use pickle for chunk metadata? | grounded | grounded | 1 | vector_store.md |
| in_scope | What information is included with every chunk? | grounded | grounded | 1 | vector_store.md |
| in_scope | What information is included in each citation? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | How is per-claim grounding performed? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | Which file controls runtime configuration? | grounded | grounded | 1 | operations_guide.md |
| in_scope | What does the stub provider require? | grounded | grounded | 1 | operations_guide.md |
| in_scope | Does the application require users to upload documents? | grounded | grounded | 1 | operations_guide.md |
| in_scope | How is the command-line application started? | grounded | grounded | 1 | operations_guide.md |
| in_scope | How is the Streamlit application started? | grounded | grounded | 1 | operations_guide.md |
| in_scope | How is the test suite run? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | Which metrics are calculated by the evaluation script? | grounded | grounded | 1 | quality_and_evaluation.md |
| out_of_scope | What is the company's paid vacation policy? | fallback | fallback | 1 |  |
| out_of_scope | Who won the 2032 FIFA World Cup? | fallback | fallback | 1 |  |
| out_of_scope | What is the capital city of Mars? | fallback | fallback | 1 |  |
| out_of_scope | What medication was prescribed to patient Robert? | fallback | fallback | 1 |  |
| out_of_scope | What is the current stock price of Acme Corporation? | fallback | fallback | 1 |  |
| out_of_scope | What will the weather be in Paris tomorrow? | fallback | fallback | 1 |  |
| out_of_scope | What is the administrator password? | fallback | fallback | 1 |  |
| out_of_scope | Give me a recipe for lasagna. | fallback | fallback | 1 |  |

Exact safe fallback: `I don't have enough information.`

Stub-mode results use a deterministic retrieval/generation backend and a heuristic critic. They are not LLM evaluation results.
