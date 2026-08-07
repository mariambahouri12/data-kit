from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importer les routes
from api import chat_router, model_router, preprocessing_router, preview_router, upload_router

app = FastAPI(
    title="AI Experimentation Platform API",
    version="1.0.0",
    description="Backend API pour DataKit",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(upload_router, prefix="/api/upload", tags=["Upload"])
app.include_router(preview_router, prefix="/api/preview", tags=["Preview"])
app.include_router(preprocessing_router, prefix="/api/preprocess", tags=["Preprocess"])
app.include_router(model_router, prefix="/api/models", tags=["Models"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "AI Experimentation Platform API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}