from pydantic import BaseModel, Field
from typing import List, Dict

class HashAnalysisRequest(BaseModel):
    hash_value: str = Field(..., example="5d41402abc4b2a76b9719d911017c592")

class SectionInfo(BaseModel):
    Name: str
    Entropy: float
    RawSize: int

class PEAnalysisResponse(BaseModel):
    MD5: str
    Imphash: str
    Sections: List[SectionInfo]
