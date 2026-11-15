import uuid
import threading
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from app.schemas import ChatMessage, MessageRole, LeadInfo, SessionInfo


@dataclass
class Session:
    session_id: str
    created_at: datetime
    history: List[ChatMessage] = field(default_factory=list)
    lead_info: Optional[LeadInfo] = None
    lead_summary: Optional[str] = None
    lead_submitted: bool = False
    awaiting_lead: bool = False
    last_topic: Optional[str] = None
    
    def add_message(self, role: MessageRole, content: str, debug_info: dict = None) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, debug_info=debug_info)
        self.history.append(msg)
        return msg
    
    def get_history(self, limit: int = 20) -> List[ChatMessage]:
        return self.history[-limit:]
    
    def reset(self) -> None:
        self.history = []
        self.lead_info = None
        self.lead_summary = None
        self.lead_submitted = False
        self.awaiting_lead = False
        self.last_topic = None
    
    def to_info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.session_id,
            created_at=self.created_at,
            message_count=len(self.history),
            has_lead=self.lead_submitted,
            lead_info=self.lead_info,
            lead_summary=self.lead_summary,
        )


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
    
    def create_session(self) -> Session:
        session_id = str(uuid.uuid4())[:8]
        session = Session(session_id=session_id, created_at=datetime.now())
        with self._lock:
            self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)
    
    def get_or_create(self, session_id: Optional[str]) -> Session:
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()
    
    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False
    
    def list_sessions(self) -> List[SessionInfo]:
        with self._lock:
            return [s.to_info() for s in self._sessions.values()]


session_manager = SessionManager()