"""
RAG Knowledge Base Chatbot
===========================
Upload company PDFs and ask questions — the AI answers exclusively
from your documents, cites the source page, and never hallucinates.

Requirements: pip install streamlit langchain langchain-openai
              langchain-community faiss-cpu pypdf tiktoken openai
Usage: streamlit run rag_chatbot.py
"""

import os
import tempfile
from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Base Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.user-msg   {background:#EFF6FF;border-left:4px solid #3B82F6;padding:1rem;border-radius:8px;margin:.5rem 0}
.bot-msg    {background:#F0FDF4;border-left:4px solid #22C55E;padding:1rem;border-radius:8px;margin:.5rem 0}
.source-tag {background:#FEF9C3;font-size:.8rem;padding:.25rem .6rem;border-radius:4px;margin-top:.4rem;display:inline-block}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────

def init():
    for key, val in {
        "history":  [],
        "store":    None,
        "chain":    None,
        "loaded":   False,
        "n_chunks": 0,
        "n_docs":   0,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

init()


# ── PDF processing ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_vectorstore(paths: tuple, company: str):
    """Load PDFs, split into chunks, embed and index with FAISS."""
    documents = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["company"] = company
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(documents)
    store    = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return store, len(chunks)


def build_chain(store, company: str, persona: str):
    """Create a conversational retrieval chain with a strict prompt."""
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=f"""You are a helpful assistant for {company}.

{persona}

Use ONLY the context below to answer. If the answer is not in the context,
say: "I don't have that information — please contact the team directly."

Context: {{context}}
Chat history: {{chat_history}}
Question: {{question}}

Answer:""",
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    mem = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(search_kwargs={"k": 4}),
        memory=mem,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def ask(question: str) -> dict:
    """Run a question through the chain and return answer + sources."""
    result  = st.session_state.chain({"question": question})
    sources = list({
        f"{doc.metadata.get('source','Document')} — p.{doc.metadata.get('page',0)+1}"
        for doc in result.get("source_documents", [])
    })
    return {"answer": result["answer"], "sources": sources}


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    company = st.text_input("Company name", value="Acme Corp")
    persona = st.text_area(
        "Assistant persona",
        value="You are a friendly HR assistant. Help employees find information about policies and procedures.",
        height=90,
    )

    st.divider()
    st.subheader("📄 Upload Documents")
    uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)

    if uploads and st.button("Build Knowledge Base", type="primary"):
        if not api_key:
            st.error("Enter your OpenAI API key first.")
        else:
            with st.spinner("Processing documents…"):
                tmp_paths = []
                for f in uploads:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.read())
                    tmp_paths.append(tmp.name)
                    tmp.close()

                try:
                    store, n = build_vectorstore(tuple(tmp_paths), company)
                    st.session_state.store    = store
                    st.session_state.chain    = build_chain(store, company, persona)
                    st.session_state.loaded   = True
                    st.session_state.n_chunks = n
                    st.session_state.n_docs   = len(uploads)
                    st.session_state.history  = []
                    st.success(f"✓ {len(uploads)} document(s) indexed ({n} chunks)")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    for p in tmp_paths:
                        os.unlink(p)

    if st.session_state.loaded:
        st.divider()
        st.metric("Documents", st.session_state.n_docs)
        st.metric("Chunks indexed", st.session_state.n_chunks)
        if st.button("Reset"):
            for k in ["history", "store", "chain", "loaded"]:
                st.session_state[k] = [] if k == "history" else None
            st.session_state.loaded = False
            st.rerun()


# ── Main chat area ─────────────────────────────────────────────────

st.title("🤖 Knowledge Base Chatbot")
st.caption("Ask anything — answers come only from your uploaded documents.")

if not st.session_state.loaded:
    st.info("👈 Upload your company documents in the sidebar to get started.")
    st.markdown("""
    **Example use cases**
    - HR policy Q&A for employees
    - Customer support from product manuals
    - Onboarding guide for new staff
    - Contract and compliance Q&A
    """)
else:
    # Display history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            src = " &nbsp;|&nbsp; ".join(msg.get("sources", []))
            st.markdown(
                f'<div class="bot-msg">🤖 {msg["content"]}'
                f'{"<br><span class=source-tag>📄 " + src + "</span>" if src else ""}'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Input
    with st.form("chat", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        q = c1.text_input("Ask a question…", label_visibility="collapsed")
        sent = c2.form_submit_button("Send", type="primary")

    if sent and q.strip():
        st.session_state.history.append({"role": "user", "content": q})
        with st.spinner("Thinking…"):
            try:
                result = ask(q)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.error(str(e))
        st.rerun()
