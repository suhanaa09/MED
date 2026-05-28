# 🤖 RAG Chatbot — Groq + FAISS + Streamlit

A production-ready **Retrieval-Augmented Generation** chatbot that:
- 🕷️ **Scrapes any website** (with configurable crawl depth)
- 📝 **Indexes custom text/documents**
- 🔍 **Retrieves relevant chunks** using FAISS vector search
- 💬 **Answers questions** with Groq's ultra-fast LLMs
- 🚀 **Deploys in one click** to Streamlit Cloud

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────┐     ┌───────────────────────────────┐
│  Sentence-BERT  │────▶│  FAISS Vector Index           │
│  Embedder       │     │  (chunked docs + web pages)   │
│ (all-MiniLM-L6) │◀────│                               │
└─────────────────┘     └───────────────────────────────┘
      │ Top-K chunks
      ▼
┌─────────────────┐
│   Groq LLM      │  ← llama-3.3-70b / mixtral / gemma
│  (RAG prompt)   │
└─────────────────┘
      │
      ▼
   Answer + Sources
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone / download this folder
cd rag_chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this folder to a **GitHub repo**
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
3. Click **New app** → select your repo → set `app.py` as the entry point
4. *(Optional)* Add your Groq key in **Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy** — done! 🎉

---

## 🔑 Get a Free Groq API Key

1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up for free
3. Create an API key under **API Keys**
4. Paste it in the sidebar or Streamlit secrets

---

## 📦 Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Scraping | requests + BeautifulSoup4 |
| Embeddings | sentence-transformers (local, free) |
| Vector DB | FAISS (in-memory) |
| LLM | Groq API (llama / mixtral / gemma) |

---

## 🎛️ Features

- **Multi-source RAG** — combine URLs + manual text in one index
- **Configurable crawl depth** — scrape just one page or follow internal links
- **Source citations** — every answer shows which chunks were used
- **Model selector** — switch between Groq models at runtime
- **Top-K & temperature** sliders for fine-tuning retrieval & generation
- **Dark theme** with clean UI

---

## 📁 File Structure

```
rag_chatbot/
├── app.py              # Streamlit UI
├── rag_engine.py       # RAG core (scraper + embedder + FAISS + Groq)
├── requirements.txt    # Python dependencies
├── .gitignore
└── .streamlit/
    ├── config.toml     # Dark theme config
    └── secrets.toml    # API key (gitignored)
```
