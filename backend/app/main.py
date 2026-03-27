from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import (
    classical_routes,
    symmetric_routes,
    hashing_routes,
    encoding_routes,
    asymmetric_routes,
    certificates_routes,
    file_tools_routes,
    analysis_routes,
    blockchain_routes,
    steganography_routes,
    utilities_routes,
    historic_routes
)

app = FastAPI(
    title="CipherVerse API",
    version="1.0.0",
    description="A production-grade cryptographic and forensics API"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all origins; adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(classical_routes.router)
app.include_router(symmetric_routes.router)
app.include_router(hashing_routes.router)
app.include_router(encoding_routes.router)
app.include_router(asymmetric_routes.router)
app.include_router(certificates_routes.router)
app.include_router(file_tools_routes.router)
app.include_router(analysis_routes.router)
app.include_router(blockchain_routes.router)
app.include_router(steganography_routes.router)
app.include_router(utilities_routes.router)
app.include_router(historic_routes.router)

@app.get("/")
def root():
    return {
        "message": "CipherVerse backend running",
        "documentation": "/docs",
        "available_endpoints": [
            "/classical", "/symmetric", "/hashing", "/encoding", 
            "/asymmetric", "/certificates", "/file-tools", "/analysis",
            "/blockchain", "/steganography", "/utilities", "/historic"
        ]
    }
