from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from services.preprocessing_service import PreprocessingService

router = APIRouter()
preprocessing_service = PreprocessingService()


class PreprocessRequest(BaseModel):
    config: Dict[str, Any]


@router.post("/")
async def apply_preprocessing(request: PreprocessRequest):
    """Appliquer le preprocessing"""
    try:
        result = preprocessing_service.process(request.config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processed")
async def get_processed_data():
    """Récupérer les données traitées"""
    try:
        result = preprocessing_service.get_processed()
        if result is None:
            raise HTTPException(status_code=404, detail="No processed data")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))