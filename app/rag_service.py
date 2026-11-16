import time
import uuid
import requests
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from rag.env import load_config
from utils.csv_io import parse_conversation
from rag.knowledge import load_knowledge_base
from rag.retrieval import HybridRetriever
from rag.rerank import Reranker
from rag.generation import Generator


@dataclass
class RAGResult:
    answer: str
    debug_info: Dict[str, Any]
    requires_clarification: bool = False
    requires_specialist: bool = False
    lead_summary: Optional[str] = None
    should_collect_lead: bool = False


class RAGService:
    def __init__(self):
        self.config = None
        self.retriever = None
        self.reranker = None
        self.generator = None
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        self.config = load_config()
        
        print("Loading knowledge base...", flush=True)
        kb_chunks = load_knowledge_base(
            self.config.data_dir,
            self.config.chunker_chunk_size,
            self.config.chunker_overlap
        )
        print(f"Loaded {len(kb_chunks)} chunks", flush=True)
        
        print("Building indexes...", flush=True)
        self.retriever = HybridRetriever(self.config)
        self.retriever.build(kb_chunks)
        
        self.reranker = Reranker(self.config)
        self.generator = Generator(self.config)
        
        self._initialized = True
        print("RAG service initialized", flush=True)
    
    def _build_consultant_prompt(
        self,
        question: str,
        history: List[Tuple[str, str]],
        context_chunks: List,
        is_clarification: bool = False,
        awaiting_lead: bool = False,
        lead_info: dict = None
    ) -> List[dict]:
        context_parts = []
        for i, chunk_result in enumerate(context_chunks):
            source = chunk_result.chunk.metadata.get("source", "неизвестно")
            context_parts.append(f"[Источник {i+1}: {source}]\n{chunk_result.chunk.text}")
        
        context = "\n\n".join(context_parts) if context_parts else "Контекст не найден."
        
        system_prompt = (
            "Ты — консультант МТС EnergyTool. Твоя задача: помочь клиенту подобрать подходящее решение "
            "из продуктов EnergyTool, задать уточняющие вопросы при необходимости, и в подходящий момент "
            "предложить оставить заявку для менеджера.\n\n"
            "ПРАВИЛА:\n"
            "1. Отвечай на русском языке, кратко и по существу (обычно до 150 слов).\n"
            "2. Используй ТОЛЬКО предоставленный контекст из базы знаний. Не придумывай факты.\n"
            "3. Если информации в контексте недостаточно — честно скажи об этом и предложи уточнить детали или связаться со специалистом.\n"
            "4. Веди диалог как консультант: выясняй задачу клиента, подбирай продукты, объясняй простым языком.\n"
            "5. Если вопрос вне компетенции (не про продукты EnergyTool/энергетику) — скажи об этом и предложи специалиста.\n"
            "6. Когда у тебя достаточно информации для подбора решения, предложи оставить заявку (ФИО, телефон, компания).\n"
            "7. Формируй краткое резюме диалога для менеджера при сборе заявки.\n"
            "8. Не используй markdown, списки, не обещай звонков/встреч.\n"
        )
        
        if is_clarification:
            system_prompt += "\nСейчас тебе нужно задать УТОЧНЯЮЩИЙ вопрос клиенту, чтобы лучше понять его задачу. Задай один конкретный вопрос."
        
        if awaiting_lead and not lead_info:
            system_prompt += "\nКлиент заинтересован. Предложи оставить контакты (ФИО, телефон, компания) для связи менеджера."
        
        if lead_info:
            system_prompt += f"\nКлиент оставил контакты: {lead_info}. Сформируй краткое резюме диалога для менеджера (3-5 предложений: задача клиента, обсуждаемые продукты, ключевые детали)."
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            history_text = "История диалога:\n"
            for role, content in history[-8:]:
                role_label = "Пользователь" if role == "user" else "Ассистент"
                history_text += f"{role_label}: {content}\n"
            messages.append({"role": "user", "content": history_text})
        
        user_prompt = f"Контекст из базы знаний:\n{context}\n\nВопрос клиента: {question}\n\nТвой ответ:"
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def _detect_intent(self, question: str, history: List[Tuple[str, str]], context_chunks: List) -> Dict[str, Any]:
        """Простая эвристика для определения намерения клиента"""
        question_lower = question.lower()
        
        # Ключевые слова для определения готовности оставить заявку
        lead_keywords = ["заявк", "связат", "менеджер", "звон", "контакт", "телефон", "компани", "фио", "оставить"]
        clarification_keywords = ["как", "что", "какой", "какая", "сколько", "где", "когда", "почему", "можно ли", "возможно ли"]
        specialist_keywords = ["документ", "настройк", "монтаж", "установк", "техническ", "инженер", "специалист", "проект", "смет"]
        
        has_context = len(context_chunks) > 0 and any(c.chunk.text.strip() for c in context_chunks)
        
        needs_clarification = (
            not has_context or
            any(kw in question_lower for kw in clarification_keywords) or
            len(question.split()) < 5
        )
        
        ready_for_lead = any(kw in question_lower for kw in lead_keywords)
        needs_specialist = any(kw in question_lower for kw in specialist_keywords) or not has_context
        
        return {
            "needs_clarification": needs_clarification and not ready_for_lead,
            "ready_for_lead": ready_for_lead and has_context,
            "needs_specialist": needs_specialist,
            "has_context": has_context,
        }
    
    def process_message(
        self,
        question: str,
        history: List[Tuple[str, str]],
        session_data: dict = None
    ) -> RAGResult:
        if not self._initialized:
            self.initialize()
        
        session_data = session_data or {}
        awaiting_lead = session_data.get("awaiting_lead", False)
        lead_info = session_data.get("lead_info")
        
        debug = {
            "retrieve_time": 0.0,
            "rerank_time": 0.0,
            "generate_time": 0.0,
            "total_time": 0.0,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        
        t0 = time.perf_counter()
        
        # Поиск
        t1 = time.perf_counter()
        candidates = self.retriever.search(question, self.config.top_k)
        t2 = time.perf_counter()
        debug["retrieve_time"] = t2 - t1
        
        debug["retrieved_chunks"] = [
            {
                "source": c.chunk.metadata.get("source", "неизвестно"),
                "score": c.score,
                "text_preview": c.chunk.text[:200] + "..." if len(c.chunk.text) > 200 else c.chunk.text
            }
            for c in candidates[:5]
        ]
        
        # Реранкинг
        if self.config.use_reranker and candidates:
            reranked = self.reranker.rerank(question, candidates, self.config.reranker_top_n)
        else:
            from rag.rerank import RerankResult
            reranked = [RerankResult(chunk=r.chunk, score=r.score) for r in candidates[:self.config.reranker_top_n]]
        t3 = time.perf_counter()
        debug["rerank_time"] = t3 - t2
        
        debug["reranked_chunks"] = [
            {
                "source": c.chunk.metadata.get("source", "неизвестно"),
                "score": c.score,
                "text_preview": c.chunk.text[:200] + "..." if len(c.chunk.text) > 200 else c.chunk.text
            }
            for c in reranked
        ]
        
        # Определение намерения
        intent = self._detect_intent(question, history, reranked)
        
        # Генерация
        is_clarification = intent["needs_clarification"] and not awaiting_lead
        needs_specialist = intent["needs_specialist"]
        
        t3_gen = time.perf_counter()
        messages = self._build_consultant_prompt(
            question, history, reranked,
            is_clarification=is_clarification,
            awaiting_lead=awaiting_lead,
            lead_info=lead_info
        )
        
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        answer = "В предоставленных материалах этого нет / для ответа нужен специалист."
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.generator.url, headers=self.generator.headers, json=payload, timeout=self.config.request_timeout
                )
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                debug["prompt_tokens"] = usage.get("prompt_tokens", 0)
                debug["completion_tokens"] = usage.get("completion_tokens", 0)
                break
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    answer = f"Ошибка генерации: {str(e)}"
                time.sleep(2 ** attempt)
        
        t4 = time.perf_counter()
        debug["generate_time"] = t4 - t3_gen
        debug["total_time"] = t4 - t0
        
        # Постобработка ответа
        requires_clarification = is_clarification
        requires_specialist = needs_specialist or "специалист" in answer.lower() or "недостаточно" in answer.lower()
        should_collect_lead = intent["ready_for_lead"] and not awaiting_lead and not lead_info
        lead_summary = None
        
        if lead_info:
            # Формируем резюме для менеджера
            lead_summary = self._generate_lead_summary(question, history, reranked, lead_info)
        
        return RAGResult(
            answer=answer,
            debug_info=debug,
            requires_clarification=requires_clarification,
            requires_specialist=requires_specialist,
            lead_summary=lead_summary,
            should_collect_lead=should_collect_lead,
        )
    
    def _generate_lead_summary(
        self,
        question: str,
        history: List[Tuple[str, str]],
        context_chunks: List,
        lead_info: dict
    ) -> str:
        """Генерирует краткое резюме для менеджера"""
        sources = list(set(c.chunk.metadata.get("source", "") for c in context_chunks if c.chunk.metadata.get("source")))
        products_mentioned = []
        for src in sources:
            if "solar" in src.lower(): products_mentioned.append("Солнечные станции")
            elif "meter" in src.lower(): products_mentioned.append("Счетчики/Учет")
            elif "tariff" in src.lower() or "price" in src.lower(): products_mentioned.append("Тарифы/Ценовые категории")
            elif "loss" in src.lower(): products_mentioned.append("Поиск потерь/Хищения")
            elif "market" in src.lower(): products_mentioned.append("Рынок/Оптовые закупки")
        
        products = ", ".join(set(products_mentioned)) if products_mentioned else "EnergyTool продукты"
        
        # Собираем контекст диалога
        dialog_context = " ".join([f"{r}: {c}" for r, c in history[-4:]]) if history else question
        
        summary = (
            f"Клиент: {lead_info.get('full_name', 'Не указано')}, "
            f"телефон: {lead_info.get('phone', 'Не указан')}, "
            f"компания: {lead_info.get('company', 'Не указана')}. "
            f"Задача: {dialog_context[:300]}. "
            f"Обсуждаемые продукты: {products}. "
            f"Рекомендуется связаться для уточнения деталей и коммерческого предложения."
        )
        return summary


rag_service = RAGService()