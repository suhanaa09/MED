"""
rag_engine.py  –  Core RAG logic
  • Web scraping    : requests + BeautifulSoup
  • Embeddings      : sentence-transformers (all-MiniLM-L6-v2, runs locally, free)
  • Vector store    : FAISS (in-memory)
  • LLM             : Groq API (llama / mixtral / gemma)
  • Live web search : Tavily Search API (FREE tier = 1000 searches/month)
                      → fallback when FAISS index has no relevant docs
"""

from __future__ import annotations
import re
import time
import textwrap
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Live Web Search  (Tavily — free tier)
# ─────────────────────────────────────────────────────────────────────────────

class LiveSearch:
    """Wraps Tavily Search API for real-time web results."""

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Return list of {title, url, content} dicts."""
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
        }
        try:
            resp = requests.post(self.ENDPOINT, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            # Tavily returns a top-level "answer" and "results" list
            if data.get("answer"):
                results.append({
                    "title": "Tavily Direct Answer",
                    "url": "tavily://answer",
                    "content": data["answer"],
                })
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                })
            return results
        except Exception as e:
            return [{"title": "Search Error", "url": "", "content": str(e)}]


# ─────────────────────────────────────────────────────────────────────────────
# Web Scraper
# ─────────────────────────────────────────────────────────────────────────────

class WebScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    def scrape_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        visited: set[str] = set()
        all_pages: List[Dict] = []
        self._crawl(url, url, depth, visited, all_pages)
        return {"pages": all_pages, "total": len(all_pages)}

    def _crawl(self, base: str, url: str, depth: int, visited: set, pages: list):
        if url in visited or depth < 0:
            return
        visited.add(url)
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception:
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        body = soup.get_text(separator=" ")
        text = clean_text(body)

        if len(text) > 100:
            pages.append({"url": url, "title": title, "text": text})

        if depth > 1:
            for a in soup.find_all("a", href=True):
                href = urljoin(base, a["href"])
                if urlparse(href).netloc == urlparse(base).netloc:
                    self._crawl(base, href, depth - 1, visited, pages)
                    time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[str], sources: List[str]):
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)
        self.metadata.extend([{"source": s} for s in sources])

    def search(self, query_embedding: np.ndarray, top_k: int = 4):
        if self.index.ntotal == 0:
            return [], []
        q = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results, srcs = [], []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                # Only include chunks with reasonable similarity (L2 distance < 2.0)
                if distances[0][i] < 2.0:
                    results.append(self.chunks[idx])
                    srcs.append(self.metadata[idx]["source"])
        return results, srcs

    def clear(self):
        self.index.reset()
        self.chunks.clear()
        self.metadata.clear()

    @property
    def total(self):
        return self.index.ntotal


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(
        self,
        groq_api_key: str,
        tavily_api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        top_k: int = 4,
        temperature: float = 0.3,
    ):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.tavily_api_key = tavily_api_key
        self.live_search: Optional[LiveSearch] = (
            LiveSearch(tavily_api_key) if tavily_api_key else None
        )

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = VectorStore(dim=384)
        self.scraper = WebScraper()
        self._source_set: set[str] = set()

    def set_tavily_key(self, key: str):
        self.tavily_api_key = key
        self.live_search = LiveSearch(key)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        result = self.scraper.scrape_url(url, depth=depth)
        pages = result["pages"]
        if not pages:
            raise ValueError(f"No content extracted from {url}")

        total_chunks = 0
        for page in pages:
            chunks = chunk_text(page["text"])
            if not chunks:
                continue
            embeddings = self.embedder.encode(chunks, show_progress_bar=False)
            sources = [page["url"]] * len(chunks)
            self.vector_store.add(np.array(embeddings), chunks, sources)
            self._source_set.add(page["url"])
            total_chunks += len(chunks)

        return {"chunks": total_chunks, "pages": len(pages)}

    def add_text(self, text: str, label: str = "manual") -> Dict[str, Any]:
        text = clean_text(text)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No usable text found.")
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        sources = [label] * len(chunks)
        self.vector_store.add(np.array(embeddings), chunks, sources)
        self._source_set.add(label)
        return {"chunks": len(chunks)}

    # ── Query pipeline ────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        # 1. Try FAISS retrieval first
        q_emb = self.embedder.encode([question], show_progress_bar=False)[0]
        chunks, sources = self.vector_store.search(q_emb, top_k=self.top_k)

        used_web_search = False

        if chunks:
            # ── FAISS path ────────────────────────────────────────────────────
            context = "\n\n---\n\n".join(
                f"[Source: {s}]\n{c}" for c, s in zip(chunks, sources)
            )
            system_prompt = textwrap.dedent(f"""
                You are a helpful RAG assistant with access to indexed documents.
                Answer the user's question using the context below.
                Be concise, accurate, and cite sources when relevant.
                Today's date context: you have real-time document access.

                CONTEXT:
                {context}
            """).strip()

        elif self.live_search:
            # ── Live web search fallback ──────────────────────────────────────
            used_web_search = True
            web_results = self.live_search.search(question, max_results=5)
            context_parts = []
            for r in web_results:
                if r["content"]:
                    context_parts.append(
                        f"[Source: {r['url']}]\nTitle: {r['title']}\n{r['content']}"
                    )
                    sources.append(r["url"])
            context = "\n\n---\n\n".join(context_parts) if context_parts else "No results found."

            system_prompt = textwrap.dedent(f"""
                You are a helpful assistant with access to LIVE web search results.
                Use the web search results below to answer the question accurately.
                You have up-to-date information — do NOT say your knowledge is limited.
                Be concise, accurate, and cite the sources provided.

                WEB SEARCH RESULTS (fetched live):
                {context}
            """).strip()

        else:
            # ── No docs, no web search — LLM general knowledge ────────────────
            system_prompt = textwrap.dedent("""
                You are a helpful, knowledgeable assistant.
                Answer using your training knowledge. Be honest about uncertainty
                but do NOT claim you have a "knowledge cutoff" — just answer.
                If you're unsure, say so naturally without blaming limitations.

                TIP for user: Add a Tavily API key in the sidebar to enable
                live web search for real-time questions!
            """).strip()

        # 2. Call Groq LLM
        chat = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )

        answer = chat.choices[0].message.content
        unique_sources = [s for s in list(dict.fromkeys(sources)) if s and s != "tavily://answer"]

        return {
            "answer": answer,
            "sources": unique_sources,
            "chunks_used": len(chunks),
            "used_web_search": used_web_search,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return {
            "chunks": self.vector_store.total,
            "sources": len(self._source_set),
        }

    def clear(self):
        self.vector_store.clear()
        self._source_set.clear()
