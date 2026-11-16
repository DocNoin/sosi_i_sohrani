import hashlib
import requests
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import json
import sys

# Lazy FAISS import - avoid hanging on Python 3.14
def _try_import_faiss():
    """Try to import faiss, return module or None. Uses timeout to avoid hangs."""
    # On Python 3.14, faiss import can hang indefinitely. Skip it.
    if sys.version_info >= (3, 14):
        return None
    try:
        import faiss
        return faiss
    except Exception:
        return None

faiss = _try_import_faiss()
FAISS_AVAILABLE = faiss is not None

from rank_bm25 import BM25Okapi

from rag.env import Config
from rag.knowledge import DocumentChunk


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    source: str


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def build(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\w+", text.lower())
    
    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        if not self.bm25 or not self.chunks:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    source="bm25"
                ))
        return results


class VectorIndex:
    def __init__(self, config: Config):
        self.config = config
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = None
        self.index = None
        self.dimension = None
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        url = f"{self.config.llm_base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json"
        }
        
        all_embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {
                "model": self.config.emb_model,
                "input": batch
            }

            response = requests.post(
                url, headers=headers, json=payload, timeout=self.config.request_timeout
            )

            if response.status_code != 200:
                print(f"Embeddings error {response.status_code}: {response.text[:500]}", file=sys.stderr, flush=True)

            response.raise_for_status()
            data = response.json()

            if "data" not in data:
                raise RuntimeError(f"Embeddings API returned unexpected format: {data}")

            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def build(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]

        cache_dir = self.config.data_dir / ".rag_cache"
        cache_key = f"{self.config.emb_model}|{len(texts)}|{len(texts[0]) if texts else 0}|{len(texts[-1]) if texts else 0}"
        cache_name = f"emb_{hashlib.md5(cache_key.encode('utf-8')).hexdigest()}.npz"
        cache_path = cache_dir / cache_name

        embeddings = None
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    cached = data["embeddings"]
                if len(cached) == len(texts):
                    embeddings = cached.astype(np.float32)
                    print(f"Загружен кэш эмбеддингов: {cache_path}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Не удалось прочитать кэш эмбеддингов: {e}", file=sys.stderr, flush=True)

        if embeddings is None:
            embeddings = self._embed_texts(texts)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, embeddings=embeddings)
                print(f"Сохранён кэш эмбеддингов: {cache_path}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Не удалось сохранить кэш эмбеддингов: {e}", file=sys.stderr, flush=True)

        self.dimension = embeddings.shape[1]
        self.embeddings = embeddings.copy()

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = embeddings / norms
    
    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        if not self.chunks:
            return []
        
        query_embedding = self._embed_texts([query])[0]
        
        if FAISS_AVAILABLE and self.index:
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            scores = self.embeddings @ query_norm
            top_indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[top_indices]
            indices = top_indices
        
        results = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.chunks):
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source="vector"
                ))
        return results


def reciprocal_rank_fusion(
    bm25_results: List[RetrievalResult],
    vector_results: List[RetrievalResult],
    top_k: int,
    k: int = 60
) -> List[RetrievalResult]:
    chunk_scores: Dict[str, float] = {}
    chunk_map: Dict[str, DocumentChunk] = {}
    
    for rank, result in enumerate(bm25_results):
        chunk_id = result.chunk.id
        chunk_map[chunk_id] = result.chunk
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
    
    for rank, result in enumerate(vector_results):
        chunk_id = result.chunk.id
        chunk_map[chunk_id] = result.chunk
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
    
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for chunk_id, score in sorted_chunks:
        results.append(RetrievalResult(
            chunk=chunk_map[chunk_id],
            score=score,
            source="rrf"
        ))
    
    return results


class HybridRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.bm25_index = BM25Index()
        self.vector_index = VectorIndex(config)
        self.chunks: List[DocumentChunk] = []
    
    def build(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        self.bm25_index.build(chunks)
        self.vector_index.build(chunks)
    
    def search(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        if top_k is None:
            top_k = self.config.top_k
        
        bm25_results = self.bm25_index.search(query, top_k)
        vector_results = self.vector_index.search(query, top_k)
        
        return reciprocal_rank_fusion(bm25_results, vector_results, top_k)