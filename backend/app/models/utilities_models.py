from pydantic import BaseModel, Field
from typing import List, Optional

class PasswordStrengthRequest(BaseModel):
    password: str = Field(..., example="P@ssw0rd123!")

class PasswordStrengthResponse(BaseModel):
    entropy: float
    strength: str

class SaltRequest(BaseModel):
    length: int = 16

class JWTRequest(BaseModel):
    payload: dict
    secret: str
    algo: str = "HS256"
    exp_seconds: int = 3600

class ChecksumRequest(BaseModel):
    data: str
