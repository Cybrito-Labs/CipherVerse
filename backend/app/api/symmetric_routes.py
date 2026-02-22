from fastapi import APIRouter, HTTPException
from app.models.symmetric_models import *
from app.core.symmetric import *

router = APIRouter(
    prefix="/symmetric",
    tags=["Symmetric Ciphers"]
)

# ===================== XOR =====================

@router.post("/xor/encrypt", response_model=SymmetricResponse)
def xor_encrypt_route(request: XORRequest):
    return {"result": xor_cipher(request.text, request.key)}


@router.post("/xor/decrypt", response_model=SymmetricResponse)
def xor_decrypt_route(request: XORDecryptRequest):
    return {"result": xor_decipher(request.hex_data, request.key)}


@router.post("/xor/bruteforce", response_model=XORBruteforceResponse)
def xor_bruteforce_route(request: XORBruteforceRequest):
    return {"results": xor_bruteforce(request.hex_data)}


# ===================== RC2 =====================

@router.post("/rc2/encrypt", response_model=SymmetricResponse)
def rc2_encrypt_route(request: TextPasswordModeRequest):
    return {"result": rc2_encrypt(request.text, request.password)}


@router.post("/rc2/decrypt", response_model=SymmetricResponse)
def rc2_decrypt_route(request: CiphertextPasswordModeRequest):
    return {"result": rc2_decrypt(request.ciphertext, request.password)}


# ===================== RC4 =====================

@router.post("/rc4/encrypt", response_model=SymmetricResponse)
def rc4_encrypt_route(request: XORRequest):
    return {"result": rc4_encrypt(request.text, request.key)}


@router.post("/rc4/decrypt", response_model=SymmetricResponse)
def rc4_decrypt_route(request: XORDecryptRequest):
    return {"result": rc4_decrypt(request.hex_data, request.key)}


@router.post("/rc4/drop/encrypt", response_model=SymmetricResponse)
def rc4_drop_encrypt_route(request: RC4DropRequest):
    return {"result": rc4_drop_encrypt(request.text, request.key, request.drop_n)}


@router.post("/rc4/drop/decrypt", response_model=SymmetricResponse)
def rc4_drop_decrypt_route(request: RC4DropRequest):
    return {"result": rc4_drop_decrypt(request.text, request.key, request.drop_n)}


# ===================== CipherSaber2 =====================

@router.post("/ciphersaber2/encrypt", response_model=SymmetricResponse)
def cs2_encrypt_route(request: TextPasswordModeRequest):
    return {"result": ciphersaber2_encrypt(request.text, request.password)}


@router.post("/ciphersaber2/decrypt", response_model=SymmetricResponse)
def cs2_decrypt_route(request: CiphertextPasswordModeRequest):
    return {"result": ciphersaber2_decrypt(request.ciphertext, request.password)}


# ===================== AES =====================

@router.post("/aes/encrypt", response_model=SymmetricResponse)
def aes_encrypt_route(request: AESRequest):
    return {
        "result": aes_encrypt(
            request.text,
            request.password,
            request.mode,
            request.key_size
        )
    }


@router.post("/aes/decrypt", response_model=SymmetricResponse)
def aes_decrypt_route(request: AESDecryptRequest):
    return {
        "result": aes_decrypt(
            request.ciphertext,
            request.password,
            request.mode,
            request.key_size
        )
    }


# ===================== DES =====================

@router.post("/des/encrypt", response_model=SymmetricResponse)
def des_encrypt_route(request: TextPasswordModeRequest):
    return {"result": des_encrypt(request.text, request.password, request.mode)}


@router.post("/des/decrypt", response_model=SymmetricResponse)
def des_decrypt_route(request: CiphertextPasswordModeRequest):
    return {"result": des_decrypt(request.ciphertext, request.password, request.mode)}


# ===================== Triple DES =====================

@router.post("/3des/encrypt", response_model=SymmetricResponse)
def tdes_encrypt_route(request: TextPasswordModeRequest):
    return {"result": tdes_encrypt(request.text, request.password, request.mode)}


@router.post("/3des/decrypt", response_model=SymmetricResponse)
def tdes_decrypt_route(request: CiphertextPasswordModeRequest):
    return {"result": tdes_decrypt(request.ciphertext, request.password, request.mode)}


# ===================== Blowfish =====================

@router.post("/blowfish/encrypt", response_model=SymmetricResponse)
def blowfish_encrypt_route(request: TextPasswordModeRequest):
    return {"result": blowfish_encrypt(request.text, request.password, request.mode)}


@router.post("/blowfish/decrypt", response_model=SymmetricResponse)
def blowfish_decrypt_route(request: CiphertextPasswordModeRequest):
    return {"result": blowfish_decrypt(request.ciphertext, request.password, request.mode)}


# ===================== SM4 =====================

@router.post("/sm4/encrypt", response_model=SymmetricResponse)
def sm4_encrypt_route(request: TextPasswordModeRequest):
    return {"result": sm4_encrypt(request.text, request.password, request.mode)}


@router.post("/sm4/decrypt", response_model=SymmetricResponse)
def sm4_decrypt_route(request: CiphertextPasswordModeRequest):
    return {"result": sm4_decrypt(request.ciphertext, request.password, request.mode)}