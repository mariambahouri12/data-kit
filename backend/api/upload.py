from fastapi import APIRouter, UploadFile, File, HTTPException
from services.upload_service import UploadService

router = APIRouter()
upload_service = UploadService()


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    """Uploader un fichier de données"""
    try:
        result = upload_service.upload(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))