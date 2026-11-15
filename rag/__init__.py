# Lazy imports to avoid loading heavy dependencies (faiss, numpy) at package import time
# Import modules directly when needed:
#   from rag.knowledge import DocumentChunk, load_knowledge_base, chunk_text
#   from rag.retrieval import HybridRetriever, RetrievalResult
#   from rag.rerank import Reranker, RerankResult
#   from rag.generation import Generator, GenerationResult

__all__ = [
    "DocumentChunk",
    "load_knowledge_base",
    "chunk_text",
    "HybridRetriever",
    "RetrievalResult",
    "Reranker",
    "RerankResult",
    "Generator",
    "GenerationResult",
]