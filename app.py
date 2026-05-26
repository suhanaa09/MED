import os
import streamlit as st
from time import sleep
from typing import Annotated, List
from typing_extensions import TypedDict
from collections import defaultdict

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import ConfigDict
import wikipediaapi

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot — Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar: API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 MediBot")
    st.caption("Medical AI Assistant · Hybrid RAG · LangGraph")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.divider()
    st.markdown("**Tools available**")
    st.markdown(
        "- 🔀 Hybrid RAG (FAISS + BM25)\n"
        "- 🔍 FAISS Semantic RAG\n"
        "- 🔑 BM25 Keyword RAG\n"
        "- 🌐 Live Wikipedia\n"
        "- 🩺 Symptom Checker\n"
        "- 💊 Drug Information"
    )
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
    st.caption("⚠️ For educational purposes only. Always consult a licensed physician.")

# ── Medical Knowledge Base ────────────────────────────────────────────────────
MEDICAL_TOPICS = [
    "Diabetes mellitus", "Hypertension", "COVID-19", "Cancer", "Asthma",
    "Tuberculosis", "Pneumonia", "Influenza", "Dengue fever", "Malaria",
    "Heart disease", "Stroke", "Kidney failure", "Liver disease", "Obesity",
    "Anemia", "Arthritis", "Migraine", "Epilepsy",
    "Parkinson's disease", "Alzheimer's disease",
    "Human body", "Brain", "Heart", "Lung", "Kidney", "Liver",
    "Digestive system", "Nervous system", "Immune system",
    "Respiratory system", "Circulatory system", "Endocrine system",
    "Fever", "Cough", "Chest pain", "Headache", "Fatigue",
    "Shortness of breath", "Abdominal pain", "Diarrhea", "Weight loss",
    "Antibiotic", "Vaccination", "Insulin", "Chemotherapy",
    "Paracetamol", "Ibuprofen", "Pain management", "Metformin",
    "Aspirin", "Amoxicillin",
    "Nutrition", "Vitamin", "Mental health", "Exercise", "Public health",
]

# ── RRF + Ensemble Retriever ──────────────────────────────────────────────────
def reciprocal_rank_fusion(results: list, k: int = 60) -> list:
    fused_scores = defaultdict(float)
    unique_docs = {}
    for result_list in results:
        for rank, doc in enumerate(result_list):
            doc_id = (doc.page_content, doc.metadata.get("source"))
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank)
    sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
    return [unique_docs[d] for d in sorted_ids]


class CustomEnsembleRetriever(BaseRetriever):
    retrievers: list
    weights: list = None
    k: int = 60
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = [r.invoke(query) for r in self.retrievers]
        return reciprocal_rank_fusion(results, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError


# ── Cached KB build (runs once per session) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_knowledge_base():
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="MediBot/2.0")
    docs = []
    for topic in MEDICAL_TOPICS:
        try:
            page = wiki.page(topic)
            if page.exists():
                docs.append(Document(
                    page_content=page.text[:5000],
                    metadata={"title": page.title, "source": page.fullurl},
                ))
        except Exception:
            pass
        sleep(0.2)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    faiss_ret = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = 4

    ensemble_ret = CustomEnsembleRetriever(
        retrievers=[faiss_ret, bm25_ret], weights=[0.6, 0.4]
    )
    return faiss_ret, bm25_ret, ensemble_ret


# ── Tools ─────────────────────────────────────────────────────────────────────
def make_tools(faiss_ret, bm25_ret, ensemble_ret):
    @tool
    def faiss_rag_retriever(query: str) -> str:
        """Semantic/dense vector search in local FAISS medical knowledge base. Best for conceptual medical questions about diseases, treatments, and anatomy."""
        return faiss_ret.invoke(query)

    @tool
    def bm25_rag_retriever(query: str) -> str:
        """Keyword BM25 search in local medical knowledge base. Best when exact medical terms, drug names, or specific condition names are mentioned."""
        return bm25_ret.invoke(query)

    @tool
    def hybrid_rag_retriever(query: str) -> str:
        """Hybrid RAG: FAISS semantic + BM25 keyword via Reciprocal Rank Fusion. Use this FIRST as the PRIMARY retriever for all medical questions."""
        return ensemble_ret.invoke(query)

    @tool
    def wikipedia_live_tool(query: str) -> str:
        """Search Wikipedia live for medical information not in the local knowledge base. Use as fallback for rare/recent conditions."""
        wiki_client = wikipediaapi.Wikipedia(language="en", user_agent="MediBot/2.0")
        try:
            page = wiki_client.page(query)
            if page.exists():
                return f"[Source: {page.fullurl}]\n\n{page.text[:3000]}"
            return f"No Wikipedia page found for '{query}'."
        except Exception as e:
            return f"Wikipedia error: {e}"

    @tool
    def symptom_checker(symptoms: str) -> str:
        """Analyze comma-separated symptoms, return possible conditions with urgency levels. Always use when user mentions symptoms."""
        symptom_map = {
            "fever":               {"conditions": ["Influenza", "COVID-19", "Malaria", "Typhoid", "Dengue"], "urgency": "Medium"},
            "cough":               {"conditions": ["Asthma", "COVID-19", "Tuberculosis", "Bronchitis"], "urgency": "Low-Medium"},
            "chest pain":          {"conditions": ["Heart disease", "Angina", "Pneumonia", "GERD"], "urgency": "🚨 HIGH — Seek immediate care"},
            "headache":            {"conditions": ["Migraine", "Hypertension", "Tension headache", "Meningitis"], "urgency": "Low-Medium"},
            "fatigue":             {"conditions": ["Anemia", "Diabetes", "Hypothyroidism", "Depression"], "urgency": "Low"},
            "shortness of breath": {"conditions": ["Asthma", "Heart failure", "COVID-19", "Pulmonary embolism"], "urgency": "🚨 HIGH — Seek immediate care"},
            "frequent urination":  {"conditions": ["Diabetes mellitus", "UTI", "Prostate issues"], "urgency": "Medium"},
            "weight loss":         {"conditions": ["Diabetes", "Cancer", "Tuberculosis", "Hyperthyroidism"], "urgency": "Medium-High"},
            "joint pain":          {"conditions": ["Arthritis", "Gout", "Lupus", "Fibromyalgia"], "urgency": "Low-Medium"},
            "rash":                {"conditions": ["Eczema", "Psoriasis", "Allergic reaction", "Lupus"], "urgency": "Low"},
            "nausea":              {"conditions": ["Gastritis", "Food poisoning", "Pregnancy", "Migraine"], "urgency": "Low-Medium"},
            "dizziness":           {"conditions": ["Hypertension", "Anemia", "Inner ear disorder", "Stroke"], "urgency": "Medium"},
            "vomiting":            {"conditions": ["Gastritis", "Food poisoning", "Appendicitis", "Migraine"], "urgency": "Medium"},
            "abdominal pain":      {"conditions": ["Gastritis", "Appendicitis", "IBS", "Kidney stones"], "urgency": "Medium-High"},
            "back pain":           {"conditions": ["Muscle strain", "Herniated disc", "Kidney stones", "Sciatica"], "urgency": "Low-Medium"},
            "swelling":            {"conditions": ["Heart failure", "Kidney disease", "DVT", "Lymphedema"], "urgency": "Medium"},
            "blurred vision":      {"conditions": ["Diabetes", "Hypertension", "Glaucoma", "Stroke"], "urgency": "🚨 HIGH — Seek immediate care"},
            "confusion":           {"conditions": ["Stroke", "Hypoglycemia", "Dementia", "Encephalitis"], "urgency": "🚨 HIGH — Seek immediate care"},
        }
        entered = [s.strip().lower() for s in symptoms.split(",")]
        result = {}
        for sym in entered:
            for key, data in symptom_map.items():
                if key in sym:
                    result[key] = data
        if not result:
            return "No matches found. Describe symptoms more clearly or consult a healthcare professional."
        lines = ["Symptom Analysis (Educational Only):\n"]
        for sym, data in result.items():
            lines.append(f"  • {sym.capitalize()}")
            lines.append(f"    Possible: {', '.join(data['conditions'])}")
            lines.append(f"    Urgency: {data['urgency']}")
        lines.append("\nEducational only. Always consult a licensed physician.")
        return "\n".join(lines)

    @tool
    def drug_information(drug_name: str) -> str:
        """Get drug details: class, uses, dosage, side effects, contraindications, interactions. Use when user asks about a specific medication."""
        drug_db = {
            "paracetamol": {"class": "Analgesic/Antipyretic", "uses": "Fever, mild-moderate pain", "dosage": "500-1000mg every 4-6h, max 4g/day", "side_effects": "Rare at normal doses; overdose causes liver damage", "contraindications": "Severe liver disease", "interactions": "Warfarin (high doses)"},
            "ibuprofen":   {"class": "NSAID", "uses": "Pain, fever, inflammation", "dosage": "200-400mg every 4-6h, max 1200mg/day OTC", "side_effects": "GI upset, ulcers, increased BP, kidney issues", "contraindications": "Peptic ulcer, kidney disease, pregnancy (3rd trimester)", "interactions": "Aspirin, warfarin, ACE inhibitors"},
            "amoxicillin": {"class": "Penicillin Antibiotic", "uses": "Bacterial infections: respiratory, UTI, ear, skin", "dosage": "250-500mg every 8h or 875mg every 12h", "side_effects": "Diarrhea, nausea, rash, yeast overgrowth", "contraindications": "Penicillin allergy", "interactions": "Warfarin, oral contraceptives"},
            "metformin":   {"class": "Biguanide Antidiabetic", "uses": "Type 2 diabetes (first-line)", "dosage": "Start 500mg twice daily, max 2550mg/day", "side_effects": "GI upset (temporary); rare: lactic acidosis", "contraindications": "Kidney failure (eGFR<30), liver disease", "interactions": "Alcohol, iodinated contrast, cimetidine"},
            "aspirin":     {"class": "Salicylate NSAID / Antiplatelet", "uses": "Pain, fever; low-dose cardiovascular protection", "dosage": "Pain: 325-650mg. Cardio: 75-100mg/day", "side_effects": "GI irritation, tinnitus (high dose), bleeding", "contraindications": "Children <16 (Reye syndrome), peptic ulcer", "interactions": "Warfarin, ibuprofen, SSRIs"},
            "insulin":     {"class": "Hormone / Injectable Antidiabetic", "uses": "Type 1 DM (essential), Type 2 DM (refractory)", "dosage": "Individualized — medical supervision required", "side_effects": "Hypoglycemia, weight gain, injection site reactions", "contraindications": "Hypoglycemia", "interactions": "Beta-blockers, corticosteroids, alcohol"},
        }
        key = drug_name.lower().strip()
        matched = next((k for k in drug_db if k in key or key in k), None)
        if not matched:
            return f"No local data for '{drug_name}'. Try hybrid_rag_retriever or wikipedia_live_tool."
        i = drug_db[matched]
        return (
            f"{matched.capitalize()}\n"
            f"  Class: {i['class']}\n"
            f"  Uses: {i['uses']}\n"
            f"  Dosage: {i['dosage']}\n"
            f"  Side Effects: {i['side_effects']}\n"
            f"  Contraindications: {i['contraindications']}\n"
            f"  Key Interactions: {i['interactions']}\n\n"
            f"Follow your doctor's/pharmacist's instructions."
        )

    return [hybrid_rag_retriever, faiss_rag_retriever, bm25_rag_retriever,
            wikipedia_live_tool, symptom_checker, drug_information]


# ── Agent builder ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are MediBot, an expert Medical AI Assistant with a Hybrid RAG system.

## Tools Available:
1. hybrid_rag_retriever — PRIMARY: FAISS semantic + BM25 keyword. Use FIRST for all medical questions.
2. faiss_rag_retriever — Dense semantic search. Use for conceptual questions.
3. bm25_rag_retriever — Keyword search. Use for exact medical terms/drug names.
4. wikipedia_live_tool — Live Wikipedia. Use as fallback for rare/recent conditions.
5. symptom_checker — ALWAYS call when the user mentions any symptoms.
6. drug_information — ALWAYS call when a specific drug/medication is named.

## Rules:
- Start every medical question with hybrid_rag_retriever.
- Call symptom_checker immediately if symptoms are described.
- Call drug_information when a drug is named.
- Use wikipedia_live_tool for extra depth or unknown topics.
- Structure answers with headers and bullet points.
- Always end with a Sources & Tools Used section listing each tool invoked.
- Finish with the medical disclaimer.

*Medical Disclaimer: For educational purposes only. Always consult a qualified healthcare professional.*
"""

TOOL_LABELS = {
    "hybrid_rag_retriever": "🔀 Hybrid RAG (FAISS + BM25)",
    "faiss_rag_retriever":  "🔍 FAISS Semantic RAG",
    "bm25_rag_retriever":   "🔑 BM25 Keyword RAG",
    "wikipedia_live_tool":  "🌐 Live Wikipedia",
    "symptom_checker":      "🩺 Symptom Checker",
    "drug_information":     "💊 Drug Information DB",
}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tools_used: List[str]


def build_agent(tools):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.2,
        max_tokens=1024,
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: AgentState):
        msgs = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    def run_tools(state: AgentState):
        last = state["messages"][-1]
        used = list(state.get("tools_used", []))
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                name = tc.get("name", "unknown")
                if name not in used:
                    used.append(name)
        result = tool_node.invoke(state)
        result["tools_used"] = used
        return result

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🏥 MediBot — Medical AI Assistant")
st.caption("Hybrid RAG (FAISS + BM25) · LangGraph ReAct Agent · Groq Llama-3.3-70b")

# Build KB on first load
if not st.session_state.kb_ready:
    with st.status("🔨 Building medical knowledge base from Wikipedia...", expanded=True) as status:
        st.write("Loading medical articles (this takes ~60s on first load)...")
        faiss_ret, bm25_ret, ensemble_ret = build_knowledge_base()
        tools = make_tools(faiss_ret, bm25_ret, ensemble_ret)
        st.session_state.faiss_ret = faiss_ret
        st.session_state.bm25_ret = bm25_ret
        st.session_state.ensemble_ret = ensemble_ret
        st.session_state.tools = tools
        st.session_state.kb_ready = True
        status.update(label="✅ Knowledge base ready!", state="complete")
else:
    tools = st.session_state.tools

# Example prompts
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    examples = [
        "What is diabetes mellitus and how is it treated?",
        "I have fever, cough and fatigue — what could it be?",
        "What are the side effects of ibuprofen?",
        "Tell me about Kawasaki disease in children.",
        "How does the immune system work?",
        "What is the dosage of metformin?",
    ]
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}"):
            st.session_state.pending_question = ex
            st.rerun()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            with st.expander("🔧 Tools used"):
                for t in msg["tools_used"]:
                    st.markdown(f"- {TOOL_LABELS.get(t, t)}")

# Handle pending example question
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.conversation_history.append(HumanMessage(content=question))

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("MediBot is thinking..."):
                try:
                    agent = build_agent(tools)
                    result = agent.invoke(
                        {"messages": st.session_state.conversation_history, "tools_used": []},
                        config={"recursion_limit": 25},
                    )
                    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
                    reply = ai_msgs[-1].content if ai_msgs else "No response."
                    tools_used = result.get("tools_used", [])
                    st.session_state.conversation_history.append(ai_msgs[-1])
                    st.markdown(reply)
                    if tools_used:
                        with st.expander("🔧 Tools used"):
                            for t in tools_used:
                                st.markdown(f"- {TOOL_LABELS.get(t, t)}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply, "tools_used": tools_used}
                    )
                except Exception as e:
                    st.error(f"Agent error: {e}")

# Chat input
if question := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.conversation_history.append(HumanMessage(content=question))

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("MediBot is thinking..."):
                try:
                    agent = build_agent(tools)
                    result = agent.invoke(
                        {"messages": st.session_state.conversation_history, "tools_used": []},
                        config={"recursion_limit": 25},
                    )
                    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
                    reply = ai_msgs[-1].content if ai_msgs else "No response."
                    tools_used = result.get("tools_used", [])
                    st.session_state.conversation_history.append(ai_msgs[-1])
                    st.markdown(reply)
                    if tools_used:
                        with st.expander("🔧 Tools used"):
                            for t in tools_used:
                                st.markdown(f"- {TOOL_LABELS.get(t, t)}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply, "tools_used": tools_used}
                    )
                except Exception as e:
                    st.error(f"Agent error: {e}")
