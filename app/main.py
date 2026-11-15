import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import (
    ChatRequest, ChatResponse, LeadSubmitRequest, LeadSubmitResponse,
    SessionInfo, ResetResponse, DebugInfo
)
from app.session import session_manager, Session
from app.rag_service import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    rag_service.initialize()
    yield
    # Shutdown (if needed)


app = FastAPI(
    title="МТС EnergyTool — ИИ-консультант",
    description="Веб-интерфейс чата с режимом отладки для Хакатона Hack&Change 2026",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session = session_manager.get_or_create(request.session_id)
    
    # Подготавливаем историю для RAG
    history = [(msg.role.value, msg.content) for msg in session.get_history(limit=10)]
    
    # Данные сессии для RAG
    session_data = {
        "awaiting_lead": session.awaiting_lead,
        "lead_info": session.lead_info.model_dump() if session.lead_info else None,
    }
    
    # Обрабатываем сообщение через RAG
    result = rag_service.process_message(request.message, history, session_data)
    
    # Добавляем сообщение пользователя в историю
    session.add_message(role="user", content=request.message)
    
    # Определяем, нужно ли переключить состояние ожидания лида
    if result.should_collect_lead and not session.awaiting_lead:
        session.awaiting_lead = True
    
    # Добавляем ответ ассистента
    debug_info = result.debug_info if request.debug_mode else None
    session.add_message(role="assistant", content=result.answer, debug_info=debug_info)
    
    # Если есть резюме лида — сохраняем
    if result.lead_summary:
        session.lead_summary = result.lead_summary
        session.lead_submitted = True
        session.awaiting_lead = False
    
    return ChatResponse(
        response=result.answer,
        session_id=session.session_id,
        history=session.get_history(),
        debug_info=debug_info,
        requires_lead=session.awaiting_lead,
        lead_summary=session.lead_summary,
    )


@app.post("/api/lead", response_model=LeadSubmitResponse)
async def submit_lead(request: LeadSubmitRequest):
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    session.lead_info = request.lead
    session.awaiting_lead = False
    session.lead_submitted = True
    
    # Генерируем резюме через RAG
    history = [(msg.role.value, msg.content) for msg in session.get_history(limit=10)]
    # Последнее сообщение пользователя
    last_user_msg = next((msg.content for msg in reversed(session.history) if msg.role == "user"), "")
    
    result = rag_service.process_message(
        last_user_msg,
        history[:-1] if history else [],
        {"awaiting_lead": False, "lead_info": request.lead.model_dump()}
    )
    
    session.lead_summary = result.lead_summary or "Резюме будет сформировано менеджером."
    
    return LeadSubmitResponse(
        success=True,
        summary=session.lead_summary,
        message="Заявка принята! Менеджер свяжется с вами в ближайшее время."
    )


@app.post("/api/reset", response_model=ResetResponse)
async def reset_session(session_id: Optional[str] = None):
    if session_id:
        session = session_manager.get_session(session_id)
        if session:
            session.reset()
            return ResetResponse(success=True, message="Контекст диалога сброшен", session_id=session_id)
        return ResetResponse(success=False, message="Сессия не найдена")
    else:
        # Создаем новую сессию
        new_session = session_manager.create_session()
        return ResetResponse(success=True, message="Начат новый диалог", session_id=new_session.session_id)


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions():
    return session_manager.list_sessions()


@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return session.to_info()


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_manager.delete_session(session_id):
        return {"success": True, "message": "Сессия удалена"}
    raise HTTPException(status_code=404, detail="Сессия не найдена")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "energytool-chat"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)