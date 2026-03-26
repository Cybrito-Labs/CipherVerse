from pydantic import BaseModel, Field
from typing import List, Dict

class FileHashRequest(BaseModel):
    filepath: str = Field(..., example="C:/test/file.txt")
    algorithm: str = "sha256"

class FileHashResponse(BaseModel):
    hash: str

class FileMultiHashResponse(BaseModel):
    hashes: Dict[str, str]

class EntropyResponse(BaseModel):
    entropy: float
    interpretation: str

class RandomnessResponse(BaseModel):
    entropy: float
    bit_balance: Dict[str, float]
    runs: int
    chi_square: float
