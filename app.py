import streamlit as st
import os
from rag_engine import RAGEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark theme override */
.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2330;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #a78bfa !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #161b27 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 12px !important;
    margin-bottom: 10px;
}

/* Input box */
[data-testid="stChatInputTextArea"] {
    background: #161b27 !important;
    color: #e2e8f0 !important;
    border: 1px solid #7c3aed !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Text input */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #161b27 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
}

/* Expander (sources) */
.streamlit-expanderHeader {
    background: #161b27 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 8px !important;
    color: #a78bfa !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
.streamlit-expanderContent {
    background: #0d1117 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 0 0 8px 8px !important;
}

/* Source chips */
.source-chip {
    display: inline-block;
    background: #1e2330;
    border: 1px solid #7c3aed44;
    color: #a78bfa;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    margin: 3px;
}

/* Status badges */
.badge-success {
    background: #064e3b;
    color: #34d399;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
}
.badge-info {
    background: #1e3a5f;
    color: #60a5fa;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #161b27 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

/* Select box */
[data-testid="stSelectbox"] > div > div {
    background: #161b27 !important;
    border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #7c3aed55; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }

/* Divider */
hr { border-color: #1e2330 !important; }

/* Info/warning boxes */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources_loaded" not in st.session_state:
        st.session_state.sources_loaded = []
    if "groq_key_set" not in st.session_state:
        st.session_state.groq_key_set = False

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🤖 RAG Chatbot")
    st.markdown("*Retrieval-Augmented Generation*")
    st.divider()

    # API Key
    st.markdown("### 🔑 Groq API Key")
    groq_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        if not st.session_state.groq_key_set:
            st.session_state.groq_key_set = True
            st.session_state.rag = RAGEngine(groq_api_key=groq_key)
        st.markdown('<span class="badge-success">✓ Key loaded</span>', unsafe_allow_html=True)
    else:
        st.info("⚠️ Add your Groq API key to start")

    st.divider()

    # Model selector
    st.markdown("### ⚙️ Model")
    model = st.selectbox(
        "Groq LLM",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="Llama 3.3 70B is most capable",
    )
    if st.session_state.rag:
        st.session_state.rag.model = model

    st.divider()

    # ── Data ingestion ────────────────────────────────────────────────────────
    st.markdown("### 📥 Add Knowledge Sources")

    tab_url, tab_text = st.tabs(["🌐 Web URL", "📝 Raw Text"])

    with tab_url:
        url_input = st.text_input("Website URL", placeholder="https://example.com")
        crawl_depth = st.slider("Crawl depth", 1, 3, 1, help="Pages deep to follow links")
        if st.button("🕷️ Scrape & Index", use_container_width=True):
            if not st.session_state.rag:
                st.error("Add Groq API key first!")
            elif not url_input:
                st.error("Enter a URL!")
            else:
                with st.spinner(f"Scraping {url_input}..."):
                    try:
                        result = st.session_state.rag.add_url(url_input, depth=crawl_depth)
                        st.session_state.sources_loaded.append({"type": "url", "src": url_input})
                        st.success(f"✓ Indexed {result['chunks']} chunks from {result['pages']} page(s)")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_text:
        raw_text = st.text_area("Paste text / docs", height=150, placeholder="Paste any text, docs, notes...")
        text_label = st.text_input("Label (optional)", placeholder="My Document")
        if st.button("📄 Index Text", use_container_width=True):
            if not st.session_state.rag:
                st.error("Add Groq API key first!")
            elif not raw_text.strip():
                st.error("Enter some text!")
            else:
                with st.spinner("Indexing..."):
                    try:
                        result = st.session_state.rag.add_text(raw_text, label=text_label or "Manual Text")
                        st.session_state.sources_loaded.append({"type": "text", "src": text_label or "Manual Text"})
                        st.success(f"✓ Indexed {result['chunks']} chunks")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Sources list
    if st.session_state.sources_loaded:
        st.divider()
        st.markdown("### 📚 Loaded Sources")
        for s in st.session_state.sources_loaded:
            icon = "🌐" if s["type"] == "url" else "📝"
            label = s["src"][:35] + "…" if len(s["src"]) > 35 else s["src"]
            st.markdown(f'<span class="source-chip">{icon} {label}</span>', unsafe_allow_html=True)

    st.divider()

    # RAG settings
    st.markdown("### 🎛️ RAG Settings")
    top_k = st.slider("Top-K chunks", 1, 10, 4, help="How many chunks to retrieve")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    if st.session_state.rag:
        st.session_state.rag.top_k = top_k
        st.session_state.rag.temperature = temperature

    # Stats
    if st.session_state.rag and st.session_state.rag.vector_store:
        st.divider()
        st.markdown("### 📊 Index Stats")
        stats = st.session_state.rag.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Chunks", stats["chunks"])
        col2.metric("Sources", stats["sources"])

    # Clear
    st.divider()
    if st.button("🗑️ Clear Everything", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_loaded = []
        if st.session_state.rag:
            st.session_state.rag.clear()
        st.success("Cleared!")
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 💬 RAG Chatbot")
st.markdown("Ask anything about your loaded documents or URLs.")

# Show welcome if no messages
if not st.session_state.messages:
    st.markdown("""
    <div style="background:#161b27;border:1px solid #1e2330;border-radius:12px;padding:24px;margin:20px 0;">
        <h3 style="color:#a78bfa;font-family:'Space Mono',monospace;margin-top:0;">🚀 Getting Started</h3>
        <ol style="color:#94a3b8;line-height:2;">
            <li>Add your <b style="color:#e2e8f0;">Groq API key</b> in the sidebar</li>
            <li>Paste a <b style="color:#e2e8f0;">URL</b> to scrape or add <b style="color:#e2e8f0;">text</b> to build your knowledge base</li>
            <li>Start <b style="color:#e2e8f0;">chatting</b> — the bot will answer using your documents</li>
        </ol>
        <p style="color:#64748b;font-size:13px;margin-bottom:0;">
            Powered by Groq LLMs + FAISS vector search + BeautifulSoup scraping
        </p>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask something about your documents…"):
    if not st.session_state.rag:
        st.error("Please add your Groq API key in the sidebar first!")
        st.stop()

    if not st.session_state.sources_loaded:
        st.warning("No knowledge sources loaded yet. Add a URL or text in the sidebar first!")

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.rag.query(prompt)
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})"):
                        for src in sources:
                            st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
