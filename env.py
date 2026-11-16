import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    data_dir: Path
    temperature: float
    max_tokens: int
    max_workers: int
    top_k: int
    chunker_chunk_size: int
    chunker_overlap: int
    emb_model: str
    reranker_model: str
    reranker_top_n: int
    use_reranker: bool
    request_timeout: int
    max_retries: int


def load_config() -> Config:
    data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
    
    llm_base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not llm_base_url:
        raise RuntimeError("LLM_BASE_URL или OPENAI_BASE_URL не заданы")
    
    llm_base_url = llm_base_url.strip().rstrip("/")
    
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        raise RuntimeError("LLM_API_KEY или OPENAI_API_KEY не заданы")
    
    llm_model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("GEN_MODEL")
    if not llm_model:
        raise RuntimeError("LLM_MODEL/OPENAI_MODEL/GEN_MODEL не заданы")
    
    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("MAX_TOKENS", "512"))
    max_workers = int(os.getenv("MAX_WORKERS", "1"))
    top_k = int(os.getenv("TOP_K", "10"))
    
    chunker_chunk_size = int(os.getenv("CHUNKER_CHUNK_SIZE", "512"))
    chunker_overlap = int(os.getenv("CHUNKER_OVERLAP", "100"))
    
    emb_model = os.getenv("EMB_MODEL", "BAAI/bge-m3")
    reranker_model = os.getenv("RERANKER_MODEL", "Qwen3-Reranker-8B")
    reranker_top_n = int(os.getenv("RERANKER_TOP_N", "3"))
    use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
    
    request_timeout = int(os.getenv("REQUEST_TIMEOUT", "180"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    
    return Config(
        llm_base_url=llm_base_url.rstrip("/"),
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        data_dir=data_dir,
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=max_workers,
        top_k=top_k,
        chunker_chunk_size=chunker_chunk_size,
        chunker_overlap=chunker_overlap,
        emb_model=emb_model,
        reranker_model=reranker_model,
        reranker_top_n=reranker_top_n,
        use_reranker=use_reranker,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )