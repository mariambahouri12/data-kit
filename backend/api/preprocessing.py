
"""
Preprocessing API routes.
"""

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from services.preprocessing_service import PreprocessingService

router = APIRouter()

preprocessing_service = PreprocessingService()


class PreprocessRequest(BaseModel):
    """Request body for preprocessing."""

    config: Dict[str, Any] = {}
    preset: Optional[str] = None


@router.post("/")
async def apply_preprocessing(
    request: PreprocessRequest,
):
    """Apply preprocessing."""

    try:
        result = await run_in_threadpool(
            preprocessing_service.process,
            request.config,
            request.preset,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/processed")
async def get_processed_data():
    """Return processed data."""

    try:
        dataframe = await run_in_threadpool(
            preprocessing_service.get_processed_dataframe,
        )

        if dataframe is None:
            raise HTTPException(
                status_code=404,
                detail="No processed data.",
            )

        preview_json = dataframe.head(100).to_json(orient="records")
        preview = json.loads(preview_json)

        return {
            "success": True,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "columns_list": dataframe.columns.tolist(),
            "preview": preview,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/detect")
async def detect_issues():
    """Diagnostic pipeline (missing values, outliers, correlation,
    cardinality, duplicates)."""

    try:
        report = await run_in_threadpool(
            preprocessing_service.detect_issues
        )
        return {"success": True, "report": report}

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/balance-suggestion")
async def balance_suggestion():
    """Suggestion for a rebalancing method (ImbalanceAnalyzer)."""

    try:
        suggestion = await run_in_threadpool(
            preprocessing_service.suggest_balancing
        )
        return {"success": True, **suggestion}

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/presets")
async def list_presets():
    """List available complete preprocessing presets (PreprocessingPresets)."""

    presets = await run_in_threadpool(
        preprocessing_service.list_presets
    )
    return {"success": True, "presets": presets}


@router.get("/presets/{name}")
async def get_preset(name: str):
    """Get details of a preprocessing preset."""

    try:
        preset = await run_in_threadpool(
            preprocessing_service.get_preset,
            name
        )
        return {"success": True, "preset": preset}

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.get("/preprocessors")
async def list_preprocessors():
    """List available individual components (PreprocessingFactory) —
    distinct from presets: these are reusable individual building blocks."""

    preprocessors = await run_in_threadpool(
        preprocessing_service.list_preprocessors
    )
    return {"success": True, "preprocessors": preprocessors}


@router.get("/preprocessors/{name}")
async def get_preprocessor_info(name: str):
    """Get details of an individual preprocessing component."""

    try:
        info = await run_in_threadpool(
            preprocessing_service.get_preprocessor_info,
            name
        )
        return {"success": True, "preprocessor": info}

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

