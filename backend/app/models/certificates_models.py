from pydantic import BaseModel, Field
from typing import List, Optional

class X509Response(BaseModel):
    Subject: str
    Issuer: str
    Serial_Number: int = Field(alias="Serial Number")
    Not_Before: str = Field(alias="Not Before")
    Not_After: str = Field(alias="Not After")
    Fingerprint_SHA256: str = Field(alias="Fingerprint SHA256")
    Version: str
    Extensions: List[str]

class TLSRequest(BaseModel):
    hostname: str = Field(..., example="google.com")
    port: int = Field(443, example=443)

class FingerprintRequest(BaseModel):
    data: str
    algorithm: str = "sha256"
