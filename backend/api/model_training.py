
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any


from services.model_service import ModelService


router = APIRouter()
model_service = ModelService()


class TrainRequest(BaseModel):
    model_name: str
    task: str
    params: Dict[str, Any]
    test_size: float = 0.2
    random_state: int = 42


@router.get("/available")
async def get_available_models():
    """List available models."""
    return {"models": model_service.get_available_models()}


@router.get("/params/{model_name}")
async def get_model_params(model_name: str):
    """Get the parameter schema of a model."""
    try:
        return model_service.get_parameter_schema(model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/train")
async def train_model(request: TrainRequest):
    """Train a model."""
    try:
        result = model_service.train(
            model_name=request.model_name,
            task=request.task,
            params=request.params,
            test_size=request.test_size,
            random_state=request.random_state
        )
        return {
            "success": True,
            "model_name": request.model_name,
            "task": request.task,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
