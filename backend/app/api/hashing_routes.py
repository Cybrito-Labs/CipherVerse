from fastapi import APIRouter
from app.core import hashing
from app.models.hashing_models import (
    HashRequest,
    HMACRequest,
    PBKDF2Request,
    ScryptRequest,
    HashResponse,
    KDFResponse
)

router = APIRouter(
    prefix="/hashing",
    tags=["Hashing & KDFs"]
)

@router.post("/hash", response_model=HashResponse)
def hash_route(request: HashRequest):
    if request.algorithm.lower() == "md2":
        result = hashing.md2_hash(request.text)
    elif request.algorithm.lower() == "md4":
        result = hashing.md4_hash(request.text)
    elif request.algorithm.lower() == "md5":
        result = hashing.md5_hash(request.text)
    elif request.algorithm.lower().startswith("sha"):
        result = hashing.sha_family_hash(request.text, request.algorithm)
    else:
        # Generic handler for other hashes
        try:
            import hashlib
            h = hashlib.new(request.algorithm.lower())
            h.update(request.text.encode())
            result = h.hexdigest()
        except Exception:
            result = "Unsupported algorithm"
    return {"result": result}

@router.post("/hmac", response_model=HashResponse)
def hmac_route(request: HMACRequest):
    result = hashing.hmac_generate(request.message, request.key, request.algorithm)
    return {"result": result}

@router.post("/pbkdf2", response_model=KDFResponse)
def pbkdf2_route(request: PBKDF2Request):
    result = hashing.pbkdf2_derive_key(request.password, request.iterations, request.dklen, request.hash_name)
    return result

@router.post("/scrypt", response_model=KDFResponse)
def scrypt_route(request: ScryptRequest):
    result = hashing.scrypt_hash(request.password, request.n, request.r, request.p, request.dklen)
    return result

@router.post("/bcrypt/hash", response_model=HashResponse)
def bcrypt_hash_route(request: HashRequest):
    result = hashing.bcrypt_hash(request.text)
    return {"result": result}
