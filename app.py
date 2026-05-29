import streamlit as st
import os
from rag_engine import RAGEngine

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0f14; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #a78bfa !important; }
[data-testid="stChatMessage"] {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 12px !important; margin-bottom: 10px;
}
[data-testid="stChatInputTextArea"] {
    background: #161b27 !important; color: #e2e8f0 !important;
    border: 1px solid #7c3aed !important; border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 13px !important;
    padding: 8px 16px !important; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #161b27 !important; color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important; border-radius: 8px !important;
}
.streamlit-expanderHeader {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 8px !important; color: #a78bfa !important;
    font-family: 'Space Mono', monospace !important; font-size: 12px !important;
}
.streamlit-expanderContent {
    background: #0d1117 !important; border: 1px solid #1e2330 !important;
    border-radius: 0 0 8px 8px !important;
}
.source-chip {
    display: inline-block; background: #1e2330; border: 1px solid #7c3aed44;
    color: #a78bfa; padding: 4px 10px; border-radius: 20px;
    font-size: 11px; font-family: 'Space Mono', monospace; margin: 3px;
}
.badge-success { background:#064e3b;color:#34d399;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-web    { background:#1e3050;color:#60a5fa;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-warn   { background:#3b2000;color:#fbbf24;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
[data-testid="metric-container"] {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 10px !important; padding: 12px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #161b27 !important; border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #7c3aed55; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
hr { border-color: #1e2330 !important; }
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
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

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.markdown("### 🔑 API Keys")

    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    tavily_key = st.text_input(
        "Tavily API Key (Live Web Search)",
        type="password",
        placeholder="tvly-...",
        help="Free at app.tavily.com — enables real-time web search when no docs match",
    )

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        if not st.session_state.groq_key_set:
            st.session_state.groq_key_set = True
            st.session_state.rag = RAGEngine(
                groq_api_key=groq_key,
                tavily_api_key=tavily_key,
            )
        # Update tavily key if user adds it later
        if tavily_key and st.session_state.rag:
            st.session_state.rag.set_tavily_key(tavily_key)

        col1, col2 = st.columns(2)
        col1.markdown('<span class="badge-success">✓ Groq</span>', unsafe_allow_html=True)
        if tavily_key:
            col2.markdown('<span class="badge-web">✓ Web Search</span>', unsafe_allow_html=True)
        else:
            col2.markdown('<span class="badge-warn">⚠ No Web Search</span>', unsafe_allow_html=True)
    else:
        st.info("⚠️ Add your Groq API key to start")

    st.markdown(
        "<small style='color:#64748b'>Tavily free tier: 1,000 searches/month. "
        "Get key at <b>app.tavily.com</b></small>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Model")
    model = st.selectbox(
        "Groq LLM",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    )
    if st.session_state.rag:
        st.session_state.rag.model = model

    st.divider()

    # ── Knowledge sources ─────────────────────────────────────────────────────
    st.markdown("### 📥 Add Knowledge Sources")
    tab_url, tab_text = st.tabs(["🌐 Web URL", "📝 Raw Text"])

    with tab_url:
        url_input = st.text_input("Website URL", placeholder="https://example.com")
        crawl_depth = st.slider("Crawl depth", 1, 3, 1)
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
        raw_text = st.text_area("Paste text / docs", height=150, placeholder="Paste any text...")
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

    if st.session_state.sources_loaded:
        st.divider()
        st.markdown("### 📚 Loaded Sources")
        for s in st.session_state.sources_loaded:
            icon = "🌐" if s["type"] == "url" else "📝"
            label = s["src"][:35] + "…" if len(s["src"]) > 35 else s["src"]
            st.markdown(f'<span class="source-chip">{icon} {label}</span>', unsafe_allow_html=True)

    st.divider()

    # ── RAG settings ──────────────────────────────────────────────────────────
    st.markdown("### 🎛️ RAG Settings")
    top_k = st.slider("Top-K chunks", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    if st.session_state.rag:
        st.session_state.rag.top_k = top_k
        st.session_state.rag.temperature = temperature

    if st.session_state.rag and st.session_state.rag.vector_store:
        st.divider()
        st.markdown("### 📊 Index Stats")
        stats = st.session_state.rag.get_stats()
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats["chunks"])
        c2.metric("Sources", stats["sources"])

    st.divider()
    if st.button("🗑️ Clear Everything", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_loaded = []
        if st.session_state.rag:
            st.session_state.rag.clear()
        st.success("Cleared!")
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("# 💬 RAG Chatbot")

has_web = st.session_state.rag and st.session_state.rag.live_search
mode_badge = (
    '<span class="badge-web">🌐 Live Web Search ON</span>'
    if has_web else
    '<span class="badge-warn"></span>'
)
st.markdown(mode_badge, unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div style="background:#161b27;border:1px solid #1e2330;border-radius:12px;padding:24px;margin:20px 0;">
        <h3 style="color:#a78bfa;font-family:'Space Mono',monospace;margin-top:0;"></h3>
        <p style="color:#64748b;font-size:13px;margin-bottom:0;">
        </p>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("used_web_search"):
            st.markdown('<span class="badge-web">🌐 Answered via live web search</span>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask anything — current events, your docs, or general questions…"):
    if not st.session_state.rag:
        st.error("Please add your Groq API key in the sidebar first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.rag.query(prompt)
                answer = result["answer"]
                sources = result.get("sources", [])
                used_web = result.get("used_web_search", False)

                st.markdown(answer)

                if used_web:
                    st.markdown('<span class="badge-web">🌐 Answered via live web search</span>', unsafe_allow_html=True)

                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})"):
                        for src in sources:
                            st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "used_web_search": used_web,
                })
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
