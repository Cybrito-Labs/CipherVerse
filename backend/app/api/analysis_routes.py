from fastapi import APIRouter
from app.core import analysis
from app.models.analysis_models import (
    HashAnalysisRequest,
    PEAnalysisResponse
)

router = APIRouter(
    prefix="/analysis",
    tags=["Malware & Hash Analysis"]
)

@router.post("/hash", response_model=dict)
def hash_analysis_route(request: HashAnalysisRequest):
    return analysis.analyze_hash(request.hash_value)

@router.post("/tlsh/compare", response_model=dict)
def tlsh_compare_route(h1: str, h2: str):
    return {"score": analysis.tlsh_compare(h1, h2)}

@router.post("/pe/analyze", response_model=PEAnalysisResponse)
def pe_analyze_route(filepath: str):
    return analysis.pe_hash_analyzer(filepath)

