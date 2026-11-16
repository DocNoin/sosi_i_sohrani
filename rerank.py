import requests
import time
from typing import List
from dataclasses import dataclass

from env import Config
from retrieval import RetrievalResult
from knowledge import DocumentChunk


@dataclass
class RerankResult:
    chunk: DocumentChunk
    score: float


class Reranker:
    def __init__(self, config: Config):
        self.config = config
        self.url = f"{config.llm_base_url}/rerank"
        self.headers = {
            "Authorization": f"Bearer {config.llm_api_key}",
            "Content-Type": "application/json"
        }
    
    def rerank(self, query: str, results: List[RetrievalResult], top_n: int = None) -> List[RerankResult]:
        if not results:
            return []
        
        if top_n is None:
            top_n = self.config.reranker_top_n
        
        if len(results) <= top_n:
            return [RerankResult(chunk=r.chunk, score=r.score) for r in results]
        
        payload = {
            "model": self.config.reranker_model,
            "query": query,
            "documents": [r.chunk.text for r in results],
            "top_n": top_n,
            "return_documents": False
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.url, headers=self.headers, json=payload, timeout=self.config.request_timeout
                )
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                data = response.json()
                
                reranked = []
                for item in data.get("results", []):
                    idx = item["index"]
                    if 0 <= idx < len(results):
                        reranked.append(RerankResult(
                            chunk=results[idx].chunk,
                            score=item.get("relevance_score", results[idx].score)
                        ))
                
                if reranked:
                    return reranked
                else:
                    break
                    
            except requests.exceptions.Timeout:
                if attempt == self.config.max_retries - 1:
                    break
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    break
                time.sleep(2 ** attempt)
        
        return [RerankResult(chunk=r.chunk, score=r.score) for r in results[:top_n]]