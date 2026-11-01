from pathlib import Path

import streamlit as st

from self_healing_rag.runtime import build_runtime


@st.cache_resource
def get_runtime(force_rebuild: bool = False):
    return build_runtime(force_rebuild=force_rebuild)


st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🔎",
    layout="wide",
)
st.title("Self-Healing RAG Pipeline")
st.caption(
    "LangGraph • FAISS • Per-claim grounding • Citation-aware answers"
)

uploaded_files = st.file_uploader(
    "Optional documents (.txt, .md, .pdf)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Rebuild index with uploads", type="primary"):
    settings = build_runtime.__globals__["build_runtime"].__module__
    from self_healing_rag.config import Settings

    active_settings = Settings.from_env()
    active_settings.documents_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        safe_name = Path(uploaded.name).name
        (active_settings.documents_dir / safe_name).write_bytes(
            uploaded.getvalue()
        )
    get_runtime.clear()
    st.session_state["runtime"] = get_runtime(force_rebuild=True)
    st.success("The FAISS index was rebuilt with the uploaded documents.")

query = st.text_input(
    "Ask a question",
    value="How does the self-healing RAG workflow decide to retry?",
)

if st.button("Run pipeline", type="primary"):
    if not query.strip():
        st.warning("Enter a non-empty question.")
    else:
        with st.spinner(
            "Retrieving, generating, and checking claims..."
        ):
            runtime = st.session_state.get("runtime")
            if runtime is None:
                runtime = get_runtime()
                st.session_state["runtime"] = runtime
            result = runtime.pipeline.run(query)

        st.subheader("Answer")
        st.write(result.answer)

        columns = st.columns(3)
        columns[0].metric("Status", result.status)
        columns[1].metric("Attempts", result.attempts)
        columns[2].metric("Grounded", result.grounded)

        st.caption(f"Run ID: {result.run_id}")

        if result.error:
            st.warning(result.error)

        st.subheader("Sources")
        if result.sources:
            for source in result.sources:
                with st.container(border=True):
                    st.markdown(
                        f"**[{source.chunk_id}] {source.source}**"
                    )
                    st.text(source.snippet)
        else:
            st.info("No sources are available for this result.")

        st.subheader("Attempt trace")
        if result.attempt_trace:
            for attempt in result.attempt_trace:
                number = attempt.get("attempt", "?")
                with st.expander(f"Attempt {number}"):
                    st.write("Query:", attempt.get("query", ""))
                    st.write("Top-k:", attempt.get("top_k", ""))
                    st.write(
                        "Strategy:",
                        attempt.get("strategy", ""),
                    )
                    st.write(
                        "Grounding verdict:",
                        attempt.get("grounding_verdict", ""),
                    )
                    st.write(
                        "Critique:",
                        attempt.get("critique", ""),
                    )
                    unsupported = attempt.get(
                        "unsupported_claims",
                        [],
                    )
                    st.write("Unsupported claims:", unsupported)
                    st.json(attempt.get("retrieved_chunks", []))
        else:
            st.info("No attempt trace is available.")
