from fastapi import FastAPI
from app.api import classical_routes

app = FastAPI(
    title="CipherVerse API",
    version="1.0.0"
)

app.include_router(classical_routes.router)