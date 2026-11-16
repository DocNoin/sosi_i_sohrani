import requests
import time
from typing import List, Tuple
from dataclasses import dataclass

from rag.env import Config
from rag.knowledge import DocumentChunk
from rag.rerank import RerankResult


@dataclass
class GenerationResult:
    answer: str
    usage: dict = None


class Generator:
    def __init__(self, config: Config):
        self.config = config
        self.url = f"{config.llm_base_url}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.llm_api_key}",
            "Content-Type": "application/json"
        }
    
    def _build_prompt(
        self,
        question: str,
        history: List[Tuple[str, str]],
        context_chunks: List[RerankResult]
    ) -> List[dict]:
        context_parts = []
        for i, chunk_result in enumerate(context_chunks):
            source = chunk_result.chunk.metadata.get("source", "неизвестно")
            context_parts.append(f"[Источник {i+1}: {source}]\n{chunk_result.chunk.text}")
        
        context = "\n\n".join(context_parts) if context_parts else "Контекст не найден."
        
        # Simplified prompt for local testing (qwen3-8b)
        # Full MTS prompt is too complex for 8B model; use simplified version locally
        system_prompt = (
            "Ты — консультант МТС EnergyTool. Отвечай на русском языке, кратко и по существу. "
            "Используй ТОЛЬКО предоставленный контекст. Если информации недостаточно, отвечай: "
            "\"В предоставленных материалах этого нет / для ответа нужен специалист.\" "
            "Формат ответа: прямой ответ (Да/Нет/Может иметь смысл/Конечно) → пояснение 1-3 предложения. "
            "Не используй markdown, списки, не обещай звонков/встреч."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            history_text = "История диалога:\n"
            for role, content in history[-6:]:
                role_label = "Пользователь" if role == "user" else "Ассистент"
                history_text += f"{role_label}: {content}\n"
            messages.append({"role": "user", "content": history_text})
        
        user_prompt = f"Контекст из базы знаний:\n{context}\n\nВопрос: {question}\n\nОтвет:"
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def generate(
        self,
        question: str,
        history: List[Tuple[str, str]],
        context_chunks: List[RerankResult]
    ) -> GenerationResult:
        messages = self._build_prompt(question, history, context_chunks)
        
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        last_error = None
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
                
                answer = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                
                return GenerationResult(answer=answer, usage=usage)
                
            except requests.exceptions.Timeout:
                last_error = "Timeout"
                if attempt == self.config.max_retries - 1:
                    break
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt == self.config.max_retries - 1:
                    break
                time.sleep(2 ** attempt)
        
        return GenerationResult(
            answer="В предоставленных материалах этого нет / для ответа нужен специалист.",
            usage={}
        )


def format_history_for_prompt(history: List[Tuple[str, str]]) -> str:
    if not history:
        return ""
    lines = []
    for role, content in history:
        label = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)