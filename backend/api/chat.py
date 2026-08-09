"""
Chat API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.ai_context_state import context_manager
from services.assistant_service import assistant_service

router = APIRouter()

if assistant_service._context_manager is None:
    assistant_service._context_manager = context_manager


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    success: bool = True
    source: Optional[str] = None        
    similarity: Optional[float] = None
    cache_hit: bool = False
    error: Optional[str] = None


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the AI assistant."""
    try:
        response = assistant_service.chat(request.message)
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def chat_status():
    """Check the assistant status."""
    available = assistant_service.is_available()
    return {
        "available": available,
        "message": "Assistant ready" if available else "Assistant unavailable",
    }


@router.get("/debug-context")
async def debug_context():
    """Diagnostic endpoint - returns the current structured context."""
    return context_manager.get_full_context()