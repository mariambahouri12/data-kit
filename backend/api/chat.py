"""
Chat API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services.assistant_service import AssistantService

router = APIRouter()
assistant_service = AssistantService()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    documents: List[dict] = []
    selected_files: List[str] = []
    success: bool = True


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Envoyer un message à l'assistant IA."""
    try:
        response = assistant_service.chat(request.message)
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def chat_status():
    """Vérifier le statut de l'assistant."""
    return {
        "available": assistant_service.is_available(),
        "message": "Assistant prêt" if assistant_service.is_available() else "Assistant non disponible"
    }