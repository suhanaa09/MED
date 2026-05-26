import os
import re
import streamlit as st
from time import sleep
from typing import Annotated, List
from typing_extensions import TypedDict

st.set_page_config(
    page_title="🏥 MediBot — Medical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner="🌐 Scraping knowledge base from Mayo Clinic · Healthline · WebMD (first run ~3 min)…")
def initialise(gemini_api_key: str):

    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    import requests
    from bs4 import BeautifulSoup
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.tools import create_retriever_tool, tool
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    MEDICAL_URLS = [
        # Mayo Clinic — Diseases
        ('Diabetes mellitus',   'https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444'),
        ('Hypertension',        'https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410'),
        ('Asthma',              'https://www.mayoclinic.org/diseases-conditions/asthma/symptoms-causes/syc-20369653'),
        ('Cancer',              'https://www.mayoclinic.org/diseases-conditions/cancer/symptoms-causes/syc-20370588'),
        ('Heart disease',       'https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118'),
        ('Stroke',              'https://www.mayoclinic.org/diseases-conditions/stroke/symptoms-causes/syc-20350113'),
        ('Pneumonia',           'https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204'),
        ('Arthritis',           'https://www.mayoclinic.org/diseases-conditions/arthritis/symptoms-causes/syc-20350772'),
        ('Migraine',            'https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201'),
        ('Epilepsy',            'https://www.mayoclinic.org/diseases-conditions/epilepsy/symptoms-causes/syc-20350093'),
        ("Alzheimer's disease", 'https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447'),
        ("Parkinson's disease", 'https://www.mayoclinic.org/diseases-conditions/parkinsons-disease/symptoms-causes/syc-20376055'),
        ('Obesity',             'https://www.mayoclinic.org/diseases-conditions/obesity/symptoms-causes/syc-20375742'),
        ('Anemia',              'https://www.mayoclinic.org/diseases-conditions/anemia/symptoms-causes/syc-20352360'),
        ('Kidney failure',      'https://www.mayoclinic.org/diseases-conditions/kidney-failure/symptoms-causes/syc-20369048'),
        ('Liver disease',       'https://www.mayoclinic.org/diseases-conditions/liver-problems/symptoms-causes/syc-20374502'),
        # Healthline — Symptoms
        ('Fever',               'https://www.healthline.com/health/fever'),
        ('Headache',            'https://www.healthline.com/health/headache'),
        ('Fatigue',             'https://www.healthline.com/health/fatigue'),
        ('Chest pain',          'https://www.healthline.com/health/chest-pain'),
        ('Abdominal pain',      'https://www.healthline.com/health/abdominal-pain'),
        ('Diarrhea',            'https://www.healthline.com/health/diarrhea'),
        ('Cough',               'https://www.healthline.com/health/cough'),
        ('Shortness of breath', 'https://www.healthline.com/health/dyspnea'),
        ('Weight loss',         'https://www.healthline.com/health/unexplained-weight-loss'),
        # WebMD — Medicines & Treatments
        ('Ibuprofen',           'https://www.webmd.com/drugs/2/drug-5166/ibuprofen-oral/details'),
        ('Paracetamol',         'https://www.webmd.com/drugs/2/drug-362/acetaminophen-oral/details'),
        ('Antibiotic',          'https://www.webmd.com/a-to-z-guides/what-are-antibiotics'),
        ('Vaccination',         'https://www.webmd.com/vaccines/default.htm'),
        ('Insulin',             'https://www.webmd.com/diabetes/insulin-therapy'),
        ('Chemotherapy',        'https://www.webmd.com/cancer/chemotherapy-what-to-expect'),
        ('Pain management',     'https://www.webmd.com/pain-management/default.htm'),
        # Healthline — Health Topics
        ('Immune system',       'https://www.healthline.com/health/immune-system-disorders'),
        ('Mental health',       'https://www.healthline.com/health/mental-health'),
        ('Nutrition',           'https://www.healthline.com/nutrition'),
        ('Exercise',            'https://www.healthline.com/health/exercise-fitness'),
        ('Vitamin',             'https://www.healthline.com/health/vitamins-supplements'),
    ]

    def scrape_page(title, url):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                return None
            soup = BeautifulSoup(resp.text, 'lxml')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'noscript', 'iframe']):
                tag.decompose()
            content = ''
            for selector in ['article', 'main', '[class*="article"]', '[class*="content"]', '[class*="body"]']:
                block = soup.select_one(selector)
                if block:
                    content = block.get_text(separator=' ', strip=True)
                    break
            if not content:
                content = soup.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content).strip()
            if len(content) < 100:
                return None
            return Document(page_content=content[:5000], metadata={'title': title, 'source': url})
        except Exception:
            return None

    # Scrape KB with progress bar
    all_docs = []
    progress = st.progress(0, text="🌐 Scraping medical knowledge base…")
    for idx, (topic, url) in enumerate(MEDICAL_URLS):
        doc = scrape_page(topic, url)
        if doc:
            all_docs.append(doc)
        progress.progress((idx + 1) / len(MEDICAL_URLS),
                          text=f"🌐 Scraping: {topic} ({idx+1}/{len(MEDICAL_URLS)})")
        sleep(0.8)
    progress.empty()

    if not all_docs:
        raise RuntimeError("No pages scraped. Check internet connection.")

    # Build FAISS
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    # ── Tools ────────────────────────────────────────────────────
    rag_tool = create_retriever_tool(
        retriever,
        name='medical_rag_retriever',
        description=(
            'Search the local medical knowledge base scraped from Mayo Clinic, '
            'Healthline, and WebMD. Use this FIRST for any medical question about '
            'diseases, symptoms, treatments, medications, anatomy, or health conditions.'
        ),
    )

    SCRAPE_HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    @tool
    def web_scraper_tool(query: str) -> str:
        """Scrape live medical information from Mayo Clinic and Healthline for a given query."""
        import urllib.parse

        sources = [
            ('Mayo Clinic', f'https://www.mayoclinic.org/search/search-results?q={urllib.parse.quote(query)}'),
            ('Healthline',  f'https://www.healthline.com/search?q1={urllib.parse.quote(query)}'),
        ]

        for source_name, search_url in sources:
            try:
                resp = requests.get(search_url, headers=SCRAPE_HEADERS, timeout=8)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, 'lxml')
                link = None
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if source_name == 'Mayo Clinic' and '/diseases-conditions/' in href:
                        link = href if href.startswith('http') else 'https://www.mayoclinic.org' + href
                        break
                    elif source_name == 'Healthline' and '/health/' in href and href.startswith('/health/'):
                        link = 'https://www.healthline.com' + href
                        break
                if not link:
                    continue
                page_resp = requests.get(link, headers=SCRAPE_HEADERS, timeout=8)
                page_soup = BeautifulSoup(page_resp.text, 'lxml')
                for tag in page_soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                content = ''
                for selector in ['article', 'main', '[class*="article"]', '[class*="content"]']:
                    block = page_soup.select_one(selector)
                    if block:
                        content = block.get_text(separator=' ', strip=True)
                        break
                if not content:
                    content = page_soup.get_text(separator=' ', strip=True)
                content = re.sub(r'\s+', ' ', content).strip()
                if len(content) > 200:
                    return f"[Source: {source_name} — {link}]\n\n{content[:2000]}"
            except Exception as e:
                continue

        return f"Could not scrape results for '{query}'. Try rephrasing or consult a medical professional."

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Given a comma-separated list of symptoms, return possible medical conditions."""
        symptom_map = {
            'fever':               ['Influenza', 'COVID-19', 'Malaria', 'Typhoid'],
            'cough':               ['Asthma', 'COVID-19', 'Tuberculosis', 'Bronchitis'],
            'chest pain':          ['Heart disease', 'Angina', 'Pneumonia', 'GERD'],
            'headache':            ['Migraine', 'Hypertension', 'Tension headache', 'Meningitis'],
            'fatigue':             ['Anemia', 'Diabetes', 'Hypothyroidism', 'Depression'],
            'shortness of breath': ['Asthma', 'Heart failure', 'COVID-19', 'Pulmonary embolism'],
            'frequent urination':  ['Diabetes mellitus', 'UTI', 'Diabetes insipidus'],
            'weight loss':         ['Diabetes', 'Cancer', 'Tuberculosis', 'Hyperthyroidism'],
            'joint pain':          ['Arthritis', 'Gout', 'Lupus', 'Fibromyalgia'],
            'rash':                ['Eczema', 'Psoriasis', 'Allergic reaction', 'Lupus'],
            'nausea':              ['Gastritis', 'Food poisoning', 'Pregnancy', 'Migraine'],
            'dizziness':           ['Hypertension', 'Anemia', 'Inner ear disorder', 'Dehydration'],
            'vomiting':            ['Gastritis', 'Food poisoning', 'Appendicitis', 'Migraine'],
            'abdominal pain':      ['Gastritis', 'Appendicitis', 'IBS', 'Kidney stones'],
            'back pain':           ['Muscle strain', 'Herniated disc', 'Kidney stones', 'Sciatica'],
        }
        entered = [s.strip().lower() for s in symptoms.split(',')]
        result = {}
        for sym in entered:
            for key, conditions in symptom_map.items():
                if key in sym:
                    result[key] = conditions
        if not result:
            return 'No specific matches found. Please describe symptoms more clearly or consult a healthcare professional.'
        lines = ['Possible conditions (educational only):\n']
        for sym, conds in result.items():
            lines.append(f'  • {sym.capitalize()}: {", ".join(conds)}')
        lines.append('\nAlways consult a licensed physician for proper diagnosis.')
        return '\n'.join(lines)

    tools = [rag_tool, web_scraper_tool, symptom_checker]

    # ── LangGraph Agent ──────────────────────────────────────────
    SYSTEM_PROMPT = """You are MediBot, an expert Medical AI Assistant.

Your tools:
1. medical_rag_retriever — search local FAISS knowledge base (Mayo Clinic, Healthline, WebMD). Use FIRST.
2. web_scraper_tool — scrape live from Mayo Clinic / Healthline. Use as fallback or for extra detail.
3. symptom_checker — map symptoms to conditions. Use whenever symptoms are mentioned.

Rules:
- ALWAYS use medical_rag_retriever first.
- Use symptom_checker immediately if symptoms are described.
- Use web_scraper_tool for live/extra details or rare topics.
- Give structured, detailed answers with headers and bullet points.

Always end with:
⚠️ For informational purposes only. Always consult a healthcare professional."""

    class AgentState(TypedDict):
        messages:   Annotated[list, add_messages]
        tools_used: List[str]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key,
        temperature=0,
        convert_system_message_to_human=True,
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: AgentState):
        msgs = list(state['messages'])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        return {'messages': [llm_with_tools.invoke(msgs)]}

    def run_tools(state: AgentState):
        last = state['messages'][-1]
        used = list(state.get('tools_used', []))
        if hasattr(last, 'tool_calls') and last.tool_calls:
            for tc in last.tool_calls:
                name = tc.get('name', 'unknown')
                if name not in used:
                    used.append(name)
        result = tool_node.invoke(state)
        result['tools_used'] = used
        return result

    def should_continue(state: AgentState):
        last = state['messages'][-1]
        return 'tools' if (hasattr(last, 'tool_calls') and last.tool_calls) else END

    graph = StateGraph(AgentState)
    graph.add_node('agent', call_model)
    graph.add_node('tools', run_tools)
    graph.set_entry_point('agent')
    graph.add_conditional_edges('agent', should_continue, {'tools': 'tools', END: END})
    graph.add_edge('tools', 'agent')
    agent = graph.compile()

    return agent, HumanMessage, AIMessage, SystemMessage


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/hospital-emoji.png", width=72)
    st.title("MediBot")
    st.caption("Medical RAG Chatbot · Gemini · Web Scraper")
    st.divider()

    gemini_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    st.divider()
    st.markdown("**🔧 Stack**")
    st.markdown("""
- 🤖 LLM: `Gemini 2.0 Flash` (Google)
- 🧠 Embeddings: `all-MiniLM-L6-v2`
- 📚 Vector DB: FAISS
- 🌐 Sources: Mayo Clinic · Healthline · WebMD
- 🔗 Orchestration: LangGraph ReAct
    """)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**💡 Try asking:**")
    st.markdown("""
- What is diabetes mellitus?
- I have fever, cough and fatigue
- Side effects of ibuprofen?
- Tell me about the immune system
- What is Kawasaki disease?
    """)

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("## 🏥 MediBot — Medical Chatbot")
st.caption("FAISS RAG · Web Scraper (Mayo Clinic · Healthline · WebMD) · Gemini 2.0 Flash")
st.warning("⚠️ For educational purposes only. Always consult a qualified healthcare professional.", icon="⚕️")

if not gemini_key:
    st.info("👈 Enter your Google Gemini API key in the sidebar to get started.", icon="🔑")
    st.markdown("Get a **free** key → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) · 1,500 req/day · No credit card")
    st.stop()

try:
    agent, HumanMessage, AIMessage, SystemMessage = initialise(gemini_key)
except Exception as e:
    st.error(f"Initialisation failed: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🏥"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a medical question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    lc_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))
    lc_history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("🤔 Thinking…"):
            try:
                result = agent.invoke(
                    {"messages": lc_history, "tools_used": []},
                    config={"recursion_limit": 20},
                )
                ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
                reply = ai_msgs[-1].content if ai_msgs else "⚠️ No response received."
            except Exception as e:
                reply = f"⚠️ Agent error: {e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
