from fastapi import APIRouter
from app.core import utilities
from app.models.utilities_models import (
    PasswordStrengthRequest,
    PasswordStrengthResponse,
    SaltRequest,
    JWTRequest,
    ChecksumRequest
)

router = APIRouter(
    prefix="/utilities",
    tags=["Utility Tools"]
)

@router.post("/password/strength", response_model=PasswordStrengthResponse)
def password_strength_route(request: PasswordStrengthRequest):
    entropy = utilities.password_entropy(request.password)
    strength = utilities.interpret_password_strength(entropy)
    return {"entropy": entropy, "strength": strength}

@router.post("/salt/generate", response_model=dict)
def salt_generate_route(request: SaltRequest):
    return {"salt": utilities.generate_salt(request.length)}

@router.post("/checksum/fletcher16", response_model=dict)
def fletcher16_route(request: ChecksumRequest):
    return {"checksum": utilities.fletcher16(request.data.encode())}

@router.post("/jwt/sign", response_model=dict)
def jwt_sign_route(request: JWTRequest):
    token = utilities.jwt_sign(request.payload, request.secret, request.algo, request.exp_seconds)
    return {"token": token}
