"""RAG engine with TF-IDF fallback.

Uses ChromaDB + Sentence Transformers if available, else falls back to
scikit-learn TF-IDF cosine similarity over the user's documents.
"""
import os
import re
from typing import List, Dict, Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += s + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


class SimpleRAG:
    """TF-IDF fallback RAG. Fast, no external deps."""

    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.chunks = []
        self.metadata = []

    def index(self, documents: List[Dict[str, Any]]):
        """documents: [{id, filename, content}]"""
        self.chunks = []
        self.metadata = []
        for doc in documents:
            for chunk in chunk_text(doc["content"]):
                self.chunks.append(chunk)
                self.metadata.append({"filename": doc["filename"], "doc_id": doc["id"]})

        if not self.chunks:
            return

        if _HAS_SK:
            self.vectorizer = TfidfVectorizer(max_features=2048, stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.chunks)

    def query(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []

        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        idxs = sims.argsort()[::-1][:top_k]

        results = []
        for i in idxs:
            if sims[i] > 0:
                results.append({
                    "chunk": self.chunks[i],
                    "score": float(sims[i]),
                    **self.metadata[i],
                })
        return results


def simple_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate an answer from context (mock LLM)."""
    if not context_chunks:
        return "I could not find relevant information in your uploaded documents."

    # Try real LLM
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.2)
            ctx = "\n\n".join(c["chunk"] for c in context_chunks)
            msgs = [
                SystemMessage(content="Answer using only the provided context. If not in context, say so."),
                HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {question}"),
            ]
            return llm.invoke(msgs).content
    except Exception:
        pass

    # Mock fallback
    top = context_chunks[0]
    return (
        f"Based on your documents, here is what I found:\n\n"
        f"{top['chunk'][:400]}\n\n"
        f"(Source: {top['filename']}, relevance: {top['score']:.2f})"
    )


# ChromaDB-based RAG (used when available)
def build_chroma_rag(user_id: int, documents: List[Dict[str, Any]]):
    """Returns a chroma-backed retriever or None if unavailable."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=f"user_{user_id}")

    ids, docs, metas = [], [], []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["content"])):
            ids.append(f"{doc['id']}_{i}")
            docs.append(chunk)
            metas.append({"filename": doc["filename"], "doc_id": doc["id"]})

    if docs:
        embeddings = model.encode(docs).tolist()
        collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    return {"model": model, "collection": collection}
