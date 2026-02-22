from fastapi import APIRouter
from app.core.classical import (
    caeser,
    vigenere_encrypter,
    vigenere_decrypter,
    atbash_cipher,
    becon_encoder,
    becon_decoder,
    bifid_encrypt,
    bifid_decrypt,
    affine_encrypt,
    affine_decrypt,
    A1Z26_encoder,
    A1Z26_decoder,
    rail_fence_encrypt,
    rail_fence_decrypt,
    substitution_encrypt,
    substitution_decrypt,
)
from app.models.classical_models import (
    CaesarRequest,
    VigenereRequest,
    AtbashRequest,
    ClassicalResponse,
    BaconRequest,
    BifidRequest,
    AffineRequest,
    A1Z26Request,
    RailFenceRequest,
    SubstituteRequest,
)

router = APIRouter(
    prefix="/classical",
    tags=["Classical Ciphers"]
)


@router.post("/caesar", response_model=ClassicalResponse)
def caesar_route(request: CaesarRequest):
    result = caeser(request.text, request.shift)
    return {"result": result}


@router.post("/vigenere/encrypt", response_model=ClassicalResponse)
def vigenere_encrypt_route(request: VigenereRequest):
    result = vigenere_encrypter(request.keyword, request.text)
    return {"result": result}


@router.post("/vigenere/decrypt", response_model=ClassicalResponse)
def vigenere_decrypt_route(request: VigenereRequest):
    result = vigenere_decrypter(request.text, request.keyword)
    return {"result": result}


@router.post("/atbash", response_model=ClassicalResponse)
def atbash_route(request: AtbashRequest):
    result = atbash_cipher(request.text)
    return {"result": result}

@router.post("/bacon/encrypt", response_model=ClassicalResponse)
def bacon_encrypt_route(request: BaconRequest):
    result = becon_encoder(request.text)
    return {"result": result}

@router.post("/bacon/decrypt", response_model=ClassicalResponse)
def bacon_decrypt_route(request: BaconRequest):
    result = becon_decoder(request.text)
    return {"result": result}

@router.post("/bifid/encrypt", response_model=ClassicalResponse)
def bifid_encrypt_route(request: BifidRequest):
    result = bifid_encrypt(request.text, request.keyword)
    return {"result": result}

@router.post("/bifid/decrypt", response_model=ClassicalResponse)
def bifid_decrypt_route(request: BifidRequest):
    result = bifid_decrypt(request.text, request.keyword)
    return {"result": result}

@router.post("/affine/encrypt", response_model=ClassicalResponse)
def affine_encrypt_route(request: AffineRequest):
    result = affine_encrypt(request.text, request.a, request.b)
    return {"result": result}

@router.post("/affine/decrypt", response_model=ClassicalResponse)
def affine_decrypt_route(request: AffineRequest):
    result = affine_decrypt(request.text, request.a, request.b)
    return {"result": result}

@router.post("/a1z26/encrypt", response_model=ClassicalResponse)
def a1z26_encrypt_route(request: A1Z26Request):
    result = A1Z26_encoder(request.text)
    return {"result": result}

@router.post("/a1z26/decrypt", response_model=ClassicalResponse)
def a1z26_decrypt_route(request: A1Z26Request):
    result = A1Z26_decoder(request.text)
    return {"result": result}

@router.post("/rail_fence/encrypt", response_model=ClassicalResponse)
def rail_fence_encrypt_route(request: RailFenceRequest):
    result = rail_fence_encrypt(request.text, request.rails)
    return {"result": result}

@router.post("/rail_fence/decrypt", response_model=ClassicalResponse)
def rail_fence_decrypt_route(request: RailFenceRequest):
    result = rail_fence_decrypt(request.text, request.rails)
    return {"result": result}

@router.post("/substitute/encrypt", response_model=ClassicalResponse)
def substitute_encrypt_route(request: SubstituteRequest):
    result = substitution_encrypt(request.text, request.keyword)
    return {"result": result}

@router.post("/substitute/decrypt", response_model=ClassicalResponse)
def substitute_decrypt_route(request: SubstituteRequest):
    result = substitution_decrypt(request.text, request.keyword)
    return {"result": result}
