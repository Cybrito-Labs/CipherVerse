from fastapi import APIRouter
from app.core import file_tools
from app.models.file_tools_models import (
    FileHashRequest,
    FileHashResponse,
    FileMultiHashResponse,
    EntropyResponse,
    RandomnessResponse
)

router = APIRouter(
    prefix="/file-tools",
    tags=["File Tools & Forensics"]
)

@router.post("/hash", response_model=FileHashResponse)
def file_hash_route(request: FileHashRequest):
    return {"hash": file_tools.file_hash(request.filepath, request.algorithm)}

@router.post("/multi-hash", response_model=FileMultiHashResponse)
def file_multi_hash_route(request: FileHashRequest):
    return {"hashes": file_tools.file_multi_hash(request.filepath)}

@router.post("/entropy", response_model=EntropyResponse)
def entropy_route(request: FileHashRequest):
    entropy = file_tools.calculate_entropy(request.filepath)
    return {"entropy": entropy, "interpretation": file_tools.interpret_entropy(entropy)}

@router.post("/randomness-test", response_model=RandomnessResponse)
def randomness_test_route(request: FileHashRequest):
    return file_tools.randomness_test_suite(request.filepath)

