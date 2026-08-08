"""
Chat API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from services.ai_context_state import context_manager
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
    """Send a message to the AI assistant."""
    try:
        response = assistant_service.chat(request.message)
        return ChatResponse(**response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/status")
async def chat_status():
    """Check the assistant status."""

    available = assistant_service.is_available()

    return {
        "available": available,
        "message": (
            "Assistant ready"
            if available
            else "Assistant unavailable"
        )
    }


@router.get("/debug-context")
async def debug_context():
    """
    Diagnostic endpoint.

    Returns the currently available structured context
    for the assistant without calling the LLM.
    """

    return context_manager.get_full_context()