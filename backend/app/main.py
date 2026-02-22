from fastapi import FastAPI
from app.api.symmetric_routes import router as symmetric_router

app = FastAPI(title="CipherVerse API", version="1.0.0")

app.include_router(symmetric_router)