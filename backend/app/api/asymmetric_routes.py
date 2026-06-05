from fastapi import APIRouter
from app.core import asymmetric
from app.models.asymmetric_models import (
    RSAKeyResponse,
    RSAEncryptRequest,
    RSADecryptRequest,
    RSASignRequest,
    RSAVerifyRequest,
    AsymmetricResponse,
    DHResponse
)

router = APIRouter(
    prefix="/asymmetric",
    tags=["Asymmetric Cryptography"]
)

@router.post("/rsa/generate-keys", response_model=RSAKeyResponse)
def rsa_generate_keys_route():
    pub, priv = asymmetric.rsa_generate_keys()
    return {"public_key": pub, "private_key": priv}

@router.post("/rsa/encrypt", response_model=AsymmetricResponse)
def rsa_encrypt_route(request: RSAEncryptRequest):
    result = asymmetric.rsa_encrypt(request.text, request.public_key)
    return {"result": result}

@router.post("/rsa/decrypt", response_model=AsymmetricResponse)
def rsa_decrypt_route(request: RSADecryptRequest):
    result = asymmetric.rsa_decrypt(request.encrypted_base64, request.private_key)
    return {"result": result}

@router.post("/rsa/sign", response_model=AsymmetricResponse)
def rsa_sign_route(request: RSASignRequest):
    result = asymmetric.rsa_sign(request.message, request.private_key)
    return {"result": result}

@router.post("/rsa/verify", response_model=AsymmetricResponse)
def rsa_verify_route(request: RSAVerifyRequest):
    valid = asymmetric.rsa_verify(request.message, request.signature, request.public_key)
    return {"result": "Valid" if valid else "Invalid", "valid": valid}

@router.post("/dsa/generate-keys", response_model=RSAKeyResponse)
def dsa_generate_keys_route():
    pub, priv = asymmetric.dsa_generate_keys()
    return {"public_key": pub, "private_key": priv}

@router.post("/dsa/sign", response_model=AsymmetricResponse)
def dsa_sign_route(request: RSASignRequest):
    result = asymmetric.dsa_sign(request.message, request.private_key)
    return {"result": result}

@router.post("/dsa/verify", response_model=AsymmetricResponse)
def dsa_verify_route(request: RSAVerifyRequest):
    valid = asymmetric.dsa_verify(request.message, request.signature, request.public_key)
    return {"result": "Valid" if valid else "Invalid", "valid": valid}
