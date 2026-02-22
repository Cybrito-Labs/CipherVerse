from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


# Generic encryption request
class TextPasswordModeRequest(BaseModel):
    text: str
    password: str
    mode: Optional[str] = "CBC"


class CiphertextPasswordModeRequest(BaseModel):
    ciphertext: str
    password: str
    mode: Optional[str] = "CBC"


class AESRequest(BaseModel):
    text: str
    password: str
    mode: str = "CBC"
    key_size: int = 32


class AESDecryptRequest(BaseModel):
    ciphertext: str
    password: str
    mode: str = "CBC"
    key_size: int = 32


class XORRequest(BaseModel):
    text: str
    key: str


class XORDecryptRequest(BaseModel):
    hex_data: str
    key: str


class XORBruteforceRequest(BaseModel):
    hex_data: str


class RC4DropRequest(BaseModel):
    text: str
    key: str
    drop_n: int = 768


class SymmetricResponse(BaseModel):
    result: str


class XORBruteforceResponse(BaseModel):
    results: List[Tuple[int, str]]