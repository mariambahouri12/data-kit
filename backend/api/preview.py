
from fastapi import APIRouter, HTTPException
from services.upload_service import UploadService


router = APIRouter()
upload_service = UploadService()


@router.get("/")
async def get_preview(limit: int = 100):
    """Preview loaded data."""
    try:
        result = upload_service.get_preview(limit)
        if result is None:
            raise HTTPException(status_code=404, detail="No data loaded")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

