from pydantic import BaseModel, Field
from typing import Optional, List

class RSAKeyResponse(BaseModel):
    public_key: str
    private_key: str

class RSAEncryptRequest(BaseModel):
    text: str
    public_key: str

class RSADecryptRequest(BaseModel):
    encrypted_base64: str
    private_key: str

class RSASignRequest(BaseModel):
    message: str
    private_key: str

class RSAVerifyRequest(BaseModel):
    message: str
    signature: str
    public_key: str

class AsymmetricResponse(BaseModel):
    result: str
    valid: Optional[bool] = None

class DHResponse(BaseModel):
    shared_secret_hex: str
