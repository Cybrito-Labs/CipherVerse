import math
import re
import secrets
import os
import base64
import jwt
import time

def password_entropy(password: str) -> float:
    pool = 0
    if re.search(r"[a-z]", password): pool += 26
    if re.search(r"[A-Z]", password): pool += 26
    if re.search(r"[0-9]", password): pool += 10
    if re.search(r"[^a-zA-Z0-9]", password): pool += 32
    if pool == 0: return 0.0
    return round(len(password) * math.log2(pool), 2)

def interpret_password_strength(entropy: float) -> str:
    if entropy < 28: return "Very Weak"
    if entropy < 36: return "Weak"
    if entropy < 60: return "Moderate"
    if entropy < 80: return "Strong"
    return "Very Strong"

def generate_salt(length: int = 16) -> str:
    return secrets.token_hex(length)

def generate_nonce(length: int = 12) -> str:
    return secrets.token_urlsafe(length)

def jwt_sign(payload: dict, secret: str, algo: str = "HS256", exp_seconds: int = 3600) -> str:
    payload = payload.copy()
    payload["exp"] = int(time.time()) + exp_seconds
    return jwt.encode(payload, secret, algorithm=algo)

def jwt_verify(token: str, secret: str, algo: str = "HS256") -> dict:
    return jwt.decode(token, secret, algorithms=[algo])

def ctx1_encode(data: str) -> str:
    # Simplified Citrix CTX1 encoding logic
    return base64.b64encode(data.encode()).decode()

def ctx1_decode(token: str) -> str:
    return base64.b64decode(token.encode()).decode()

# --- Checksums ---
def fletcher16(data: bytes) -> int:
    s1, s2 = 0, 0
    for b in data:
        s1 = (s1 + b) % 255
        s2 = (s2 + s1) % 255
    return (s2 << 8) | s1

def adler32_checksum(data: bytes) -> int:
    import zlib
    return zlib.adler32(data)

def luhn_validate(number: str) -> bool:
    digits = [int(d) for d in number[::-1]]
    checksum = sum(digits[0::2]) + sum(sum(divmod(d*2, 10)) for d in digits[1::2])
    return checksum % 10 == 0

def crc32_checksum(data: bytes) -> int:
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF
