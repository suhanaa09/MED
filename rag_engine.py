"""
rag_engine.py  –  Core RAG logic
  • Web scraping   : requests + BeautifulSoup
  • Embeddings     : sentence-transformers (all-MiniLM-L6-v2, runs locally, free)
  • Vector store   : FAISS (in-memory)
  • LLM            : Groq API (llama / mixtral / gemma)
"""

from __future__ import annotations
import re
import time
import textwrap
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and remove junk."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── Web scraper ───────────────────────────────────────────────────────────────

class WebScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    def scrape_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        """Scrape a URL and optionally follow internal links (depth > 1)."""
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

        # Remove noise tags
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
                # Stay on same domain
                if urlparse(href).netloc == urlparse(base).netloc:
                    self._crawl(base, href, depth - 1, visited, pages)
                    time.sleep(0.3)  # polite crawl delay


# ── Vector store wrapper ──────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []  # {"source": str}

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
        for idx in indices[0]:
            if idx != -1:
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


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama-3.3-70b-versatile",
        top_k: int = 4,
        temperature: float = 0.3,
    ):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = model
        self.top_k = top_k
        self.temperature = temperature

        # Local embedding model (no API key needed)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = VectorStore(dim=384)
        self.scraper = WebScraper()
        self._source_set: set[str] = set()

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

    # ── Retrieval + generation ────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        # 1. Embed the question
        q_emb = self.embedder.encode([question], show_progress_bar=False)[0]

        # 2. Retrieve relevant chunks
        chunks, sources = self.vector_store.search(q_emb, top_k=self.top_k)

        # 3. Build context
        if chunks:
            context = "\n\n---\n\n".join(
                f"[Source: {s}]\n{c}" for c, s in zip(chunks, sources)
            )
            system_prompt = textwrap.dedent(f"""
                You are a helpful RAG assistant. Answer the user's question using ONLY 
                the context below. If the answer isn't in the context, say so clearly.
                Be concise, accurate, and cite sources when possible.

                CONTEXT:
                {context}
            """).strip()
        else:
            system_prompt = (
                "You are a helpful assistant. No documents have been indexed yet. "
                "Answer the question using your general knowledge and mention that "
                "no specific documents were found."
            )

        # 4. Call Groq LLM
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

        # 5. Deduplicate sources for display
        unique_sources = list(dict.fromkeys(sources))

        return {"answer": answer, "sources": unique_sources, "chunks_used": len(chunks)}

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return {
            "chunks": self.vector_store.total,
            "sources": len(self._source_set),
        }

    def clear(self):
        self.vector_store.clear()
        self._source_set.clear()
