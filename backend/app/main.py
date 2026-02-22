from fastapi import FastAPI
from app.api import classical_routes
from app.api.symmetric_routes import router as symmetric_router

app = FastAPI(
    title="CipherVerse API",
    version="1.0.0"
)

app.include_router(classical_routes.router)
app.include_router(symmetric_router)

@app.get("/")
def root():
    return {"message": "CipherVerse backend running"}
