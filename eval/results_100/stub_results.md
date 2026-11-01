# Self-Healing RAG Evaluation

**Mode: stub mode, heuristic critic**

Generated: 2026-10-30T08:11:12.729559+00:00

## Summary

| Metric | Single-pass RAG | Self-healing RAG |
|---|---:|---:|
| Claim-support rate | 100.00% | 100.00% |
| Key-fact hit rate | 100.00% | 100.00% |
| Expected-source hit rate | 100.00% | 100.00% |
| Out-of-scope refusal rate | 100.00% | 100.00% |
| Fallback rate | 30.00% | 30.00% |
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
| in_scope | What framework orchestrates the RAG workflow? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Which library is used to control the workflow? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Which nodes does the StateGraph include? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What are the nodes in the RAG workflow graph? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | How does the conditional edge make the workflow cycle? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Why does the graph loop back to retrieval? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What does the typed RAG state store? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Which information does the RAG state keep? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What does the critique node verify? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Which claims does the critique node check? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What does the workflow do after an unsupported claim? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What happens when an answer has an unsupported claim? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What is the maximum number of generation attempts? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | How many generation attempts are allowed in total? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Which exact sentence is the safe fallback? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | What sentence does the safe fallback return? | grounded | grounded | 1 | rag_architecture.md |
| in_scope | Which vector database is used by the project? | grounded | grounded | 1 | vector_store.md |
| in_scope | What is the vector database used by the project? | grounded | grounded | 1 | vector_store.md |
| in_scope | What FAISS index type is created? | grounded | grounded | 1 | vector_store.md |
| in_scope | Which type of FAISS index does the project create? | grounded | grounded | 1 | vector_store.md |
| in_scope | What is the default embedding model? | grounded | grounded | 1 | vector_store.md |
| in_scope | Which embedding model is used by default? | grounded | grounded | 1 | vector_store.md |
| in_scope | Why do the embeddings get normalized? | grounded | grounded | 1 | vector_store.md |
| in_scope | Why are embeddings normalized? | grounded | grounded | 1 | vector_store.md |
| in_scope | How does the project persist the FAISS index? | grounded | grounded | 1 | vector_store.md |
| in_scope | Where is the FAISS index persisted? | grounded | grounded | 1 | vector_store.md |
| in_scope | How does the project persist chunk metadata? | grounded | grounded | 1 | vector_store.md |
| in_scope | Where is chunk metadata persisted? | grounded | grounded | 1 | vector_store.md |
| in_scope | Is pickle used for chunk metadata? | grounded | grounded | 1 | vector_store.md |
| in_scope | Does the project use pickle? | grounded | grounded | 1 | vector_store.md |
| in_scope | What does every chunk include? | grounded | grounded | 1 | vector_store.md |
| in_scope | Which information is included with each chunk? | grounded | grounded | 1 | vector_store.md |
| in_scope | What does each citation include? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | Which information is included in every citation? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | How does per-claim grounding work? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | How is per-claim grounding done? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | Which file controls the runtime configuration? | grounded | grounded | 1 | operations_guide.md |
| in_scope | Does the stub provider require an API key? | grounded | grounded | 1 | operations_guide.md |
| in_scope | Must users upload documents for the application? | grounded | grounded | 1 | operations_guide.md |
| in_scope | How do I start the command-line application? | grounded | grounded | 1 | operations_guide.md |
| in_scope | How do I start the Streamlit application? | grounded | grounded | 1 | operations_guide.md |
| in_scope | How do I run the test suite? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | Which tool runs the test suite? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | What metrics does the evaluation script calculate? | grounded | grounded | 1 | quality_and_evaluation.md |
| in_scope | Which metrics does the evaluation script report? | grounded | grounded | 1 | quality_and_evaluation.md |
| out_of_scope | What is the company's paid vacation policy? | fallback | fallback | 1 |  |
| out_of_scope | Who won the 2032 FIFA World Cup? | fallback | fallback | 1 |  |
| out_of_scope | What is the capital city of Mars? | fallback | fallback | 1 |  |
| out_of_scope | What medication was prescribed to patient Robert? | fallback | fallback | 1 |  |
| out_of_scope | What is the current stock price of Acme Corporation? | fallback | fallback | 1 |  |
| out_of_scope | What will the weather be in Paris tomorrow? | fallback | fallback | 1 |  |
| out_of_scope | What is the administrator password? | fallback | fallback | 1 |  |
| out_of_scope | Give me a recipe for lasagna. | fallback | fallback | 1 |  |
| out_of_scope | What is the capital city of France? | fallback | fallback | 1 |  |
| out_of_scope | How many players are on a cricket team? | fallback | fallback | 1 |  |
| out_of_scope | What is the boiling point of water at sea level? | fallback | fallback | 1 |  |
| out_of_scope | Who wrote the play Hamlet? | fallback | fallback | 1 |  |
| out_of_scope | What is the population of Tokyo? | fallback | fallback | 1 |  |
| out_of_scope | Which planet is closest to the sun? | fallback | fallback | 1 |  |
| out_of_scope | How do I bake a chocolate cake? | fallback | fallback | 1 |  |
| out_of_scope | What is the price of a plane ticket to London? | fallback | fallback | 1 |  |
| out_of_scope | Who is the current president of Brazil? | fallback | fallback | 1 |  |
| out_of_scope | What is the exchange rate between dollars and euros today? | fallback | fallback | 1 |  |
| out_of_scope | How tall is Mount Everest? | fallback | fallback | 1 |  |
| out_of_scope | What are the symptoms of the common flu? | fallback | fallback | 1 |  |
| out_of_scope | Which team won the last cricket world cup? | fallback | fallback | 1 |  |
| out_of_scope | What is my bank account balance? | fallback | fallback | 1 |  |
| out_of_scope | Tell me the phone number of the nearest hospital. | fallback | fallback | 1 |  |
| out_of_scope | What movie won the best picture award last year? | fallback | fallback | 1 |  |
| out_of_scope | How do I change a flat tire on a bicycle? | fallback | fallback | 1 |  |
| out_of_scope | What is the speed of light in a vacuum? | fallback | fallback | 1 |  |
| out_of_scope | Who painted the Mona Lisa? | fallback | fallback | 1 |  |
| out_of_scope | What is the salary of the chief executive officer? | fallback | fallback | 1 |  |
| out_of_scope | How many calories are in a banana? | fallback | fallback | 1 |  |
| out_of_scope | What is the next public holiday in India? | fallback | fallback | 1 |  |

Exact safe fallback: `I don't have enough information.`

Stub-mode results use a deterministic retrieval/generation backend and a heuristic critic. They are not LLM evaluation results.
