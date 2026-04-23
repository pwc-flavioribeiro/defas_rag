import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "3.Query"))

import streamlit as st
from query import QueryPipeline, build_chroma_collection
from config import CONFIG

st.set_page_config(
    page_title="DESFA H&S Law Assistant",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_pipeline():
    from utils.azure_openai_models_utils import AsyncAzureOpenAIModels
    azure_model = AsyncAzureOpenAIModels().initialize()
    collection = build_chroma_collection(CONFIG["folder"]["chroma_db"])
    return QueryPipeline(azure_model=azure_model, chroma_collection=collection)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚕️ DESFA H&S")
    st.caption("Health & Safety Law Assistant")
    st.divider()

    st.subheader("Filters")
    law_group_option = st.selectbox(
        "Law topic",
        ["All topics", "Vibration (Group 1)", "Noise (Group 2)"],
    )
    law_group_map = {
        "All topics": None,
        "Vibration (Group 1)": "1",
        "Noise (Group 2)": "2",
    }
    selected_law_group = law_group_map[law_group_option]

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "This assistant searches across all available versions of each directive. "
        "When multiple versions are found, it will automatically compare them."
    )

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("Health & Safety Law Assistant")
st.caption("Ask questions about EU H&S directives. The assistant will highlight changes between law versions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} chunks)"):
                for i, chunk in enumerate(msg["sources"], 1):
                    meta = chunk["metadata"]
                    st.markdown(
                        f"**{i}. {meta.get('topic', 'Unknown').capitalize()} — "
                        f"Version {meta.get('year', '?')} — "
                        f"Page {meta.get('page', '?')}** "
                        f"*(score: {chunk['score']:.2f})*"
                    )
                    st.markdown(chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""))
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about H&S laws — e.g. 'What changed in the noise directive?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching directives..."):
            try:
                pipeline = get_pipeline()
                # Build history from all previous turns (exclude the current user message just appended)
                history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                    if msg.get("content")
                ]
                answer, sources = pipeline.run_with_sources(
                    prompt, law_group=selected_law_group, history=history or None
                )
            except Exception as e:
                answer  = None
                sources = []
                st.error(f"**Error:** {e}")

        if answer:
            st.markdown(answer)
        elif answer is not None:
            st.warning("The model returned an empty response. Try rephrasing your question.")

        if sources:
            with st.expander(f"Sources ({len(sources)} chunks)"):
                for i, chunk in enumerate(sources, 1):
                    meta = chunk["metadata"]
                    st.markdown(
                        f"**{i}. {meta.get('topic', 'Unknown').capitalize()} — "
                        f"Version {meta.get('year', '?')} — "
                        f"Page {meta.get('page', '?')}** "
                        f"*(score: {chunk['score']:.2f})*"
                    )
                    st.markdown(chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""))
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
