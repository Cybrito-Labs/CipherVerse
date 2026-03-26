from pydantic import BaseModel, Field
from typing import Optional

class HashRequest(BaseModel):
    text: str = Field(..., example="hello")
    algorithm: str = Field("sha256", example="sha256")

class HMACRequest(BaseModel):
    message: str = Field(..., example="secret message")
    key: str = Field(..., example="supersecret")
    algorithm: str = Field("sha256", example="sha256")

class PBKDF2Request(BaseModel):
    password: str = Field(..., example="mypassword")
    iterations: int = Field(100000, example=100000)
    dklen: int = Field(32, example=32)
    hash_name: str = Field("sha256", example="sha256")

class ScryptRequest(BaseModel):
    password: str = Field(..., example="mypassword")
    n: int = Field(16384, example=16384)
    r: int = Field(8, example=8)
    p: int = Field(1, example=1)
    dklen: int = Field(64, example=64)

class HashResponse(BaseModel):
    result: str

class KDFResponse(BaseModel):
    salt: str
    derived_key: Optional[str] = None
    hash: Optional[str] = None
    iterations: Optional[int] = None
    n: Optional[int] = None
    r: Optional[int] = None
    p: Optional[int] = None
