from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    debug_info: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    debug_mode: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    history: List[ChatMessage]
    debug_info: Optional[Dict[str, Any]] = None
    requires_lead: bool = False
    lead_summary: Optional[str] = None


class LeadInfo(BaseModel):
    full_name: str
    phone: str
    company: str


class LeadSubmitRequest(BaseModel):
    session_id: str
    lead: LeadInfo


class LeadSubmitResponse(BaseModel):
    success: bool
    summary: str
    message: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    message_count: int
    has_lead: bool
    lead_info: Optional[LeadInfo] = None
    lead_summary: Optional[str] = None


class DebugInfo(BaseModel):
    retrieve_time: float = 0.0
    rerank_time: float = 0.0
    generate_time: float = 0.0
    total_time: float = 0.0
    retrieved_chunks: List[Dict[str, Any]] = []
    reranked_chunks: List[Dict[str, Any]] = []
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ResetResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None