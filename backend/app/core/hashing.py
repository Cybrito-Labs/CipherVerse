import hashlib
import hmac as hmac_lib
import bcrypt
import base64
from Crypto.Protocol.KDF import PBKDF2, scrypt
from Crypto.Hash import MD2, MD4, SHA3_224, SHA3_256, SHA3_384, SHA3_512, SHAKE128, SHAKE256, RIPEMD160, BLAKE2b, BLAKE2s
from Crypto.Util.Padding import pad, unpad

def md2_hash(text: str) -> str:
    h = MD2.new()
    h.update(text.encode())
    return h.hexdigest()

def md4_hash(text: str) -> str:
    h = MD4.new()
    h.update(text.encode())
    return h.hexdigest()

def md5_hash(text: str) -> str:
    h = hashlib.md5()
    h.update(text.encode())
    return h.hexdigest()

def sha_family_hash(text: str, algo: str = "sha256") -> str:
    normal_algo = algo.lower().replace("-", "").replace("_", "")
    data = text.encode()
    
    # Try standard hashlib first (most robust)
    try:
        if normal_algo == "sha1": h = hashlib.sha1()
        elif normal_algo == "sha224": h = hashlib.sha224()
        elif normal_algo == "sha256": h = hashlib.sha256()
        elif normal_algo == "sha384": h = hashlib.sha384()
        elif normal_algo == "sha512": h = hashlib.sha512()
        else: h = hashlib.new(normal_algo)
        
        h.update(data)
        return h.hexdigest()
    except Exception:
        pass

    # Fallback to PyCryptodome for specialized hashes
    if normal_algo == "sha3224": h = SHA3_224.new()
    elif normal_algo == "sha3256": h = SHA3_256.new()
    elif normal_algo == "sha3384": h = SHA3_384.new()
    elif normal_algo == "sha3512": h = SHA3_512.new()
    elif normal_algo == "shake128":
        h = SHAKE128.new()
        h.update(data)
        return h.read(32).hex()
    elif normal_algo == "shake256":
        h = SHAKE256.new()
        h.update(data)
        return h.read(64).hex()
    elif normal_algo == "ripemd160": h = RIPEMD160.new()
    else: return "Unsupported algorithm"
    
    h.update(data)
    return h.hexdigest()

def sm3_hash(text: str) -> str:
    from gmssl import sm3
    return sm3.sm3_hash(list(text.encode()))

def keccak_hash(text: str, bits: int = 256) -> str:
    from Crypto.Hash import keccak
    k = keccak.new(digest_bits=bits)
    k.update(text.encode())
    return k.hexdigest()

def shake_hash(text: str, bits: int = 256) -> str:
    if bits == 128:
        h = SHAKE128.new()
        h.update(text.encode())
        return h.read(32).hex()
    if bits == 256:
        h = SHAKE256.new()
        h.update(text.encode())
        return h.read(64).hex()
    return "Unsupported bits"

def ripemd160_hash(text: str) -> str:
    return sha_family_hash(text, "ripemd160")

def whirlpool_hash(text: str) -> str:
    try:
        h = hashlib.new('whirlpool')
        h.update(text.encode())
        return h.hexdigest()
    except Exception:
        return "Whirlpool not available in this environment"

def blake2b_hash(text: str, size: int = 64) -> str:
    h = hashlib.blake2b(digest_size=size)
    h.update(text.encode())
    return h.hexdigest()

def blake2s_hash(text: str, size: int = 32) -> str:
    h = hashlib.blake2s(digest_size=size)
    h.update(text.encode())
    return h.hexdigest()

def hmac_generate(message: str, key: str, algo: str = "sha256") -> str:
    name = algo.lower().replace("-", "").replace("_", "")
    kb, mb = key.encode(), message.encode()
    
    # Use standard hmac with hashlib
    try:
        if name == "sha1": digestmod = hashlib.sha1
        elif name == "sha224": digestmod = hashlib.sha224
        elif name == "sha256": digestmod = hashlib.sha256
        elif name == "sha384": digestmod = hashlib.sha384
        elif name == "sha512": digestmod = hashlib.sha512
        else: digestmod = name
        
        return hmac_lib.new(kb, mb, digestmod).hexdigest()
    except Exception:
        return "Unsupported hash algorithm"

def hmac_verify(message: str, key: str, algo: str, given_mac: str) -> bool:
    calculated = hmac_generate(message, key, algo)
    return hmac_lib.compare_digest(calculated, given_mac)

def bcrypt_hash(password: str, rounds: int = 12) -> str:
    salt = bcrypt.gensalt(rounds)
    return bcrypt.hashpw(password.encode(), salt).decode()

def bcrypt_compare(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except:
        return False

def pbkdf2_derive_key(password: str, iterations: int = 100000, dklen: int = 32, hash_name: str = "sha256") -> dict:
    import os
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac(hash_name.lower().replace("-", ""), password.encode(), salt, iterations, dklen)
    return {
        "salt": base64.b64encode(salt).decode(),
        "derived_key": base64.b64encode(key).decode(),
        "iterations": iterations,
        "hash": hash_name
    }

def pbkdf2_verify(password: str, stored: dict) -> bool:
    salt = base64.b64decode(stored["salt"])
    derived_key = base64.b64decode(stored["derived_key"])
    iterations = stored["iterations"]
    hash_name = stored["hash"]
    
    new_key = hashlib.pbkdf2_hmac(hash_name.lower().replace("-", ""), password.encode(), salt, iterations, len(derived_key))
    return hmac_lib.compare_digest(new_key, derived_key)

def scrypt_hash(password: str, n: int = 16384, r: int = 8, p: int = 1, dklen: int = 64) -> dict:
    import os
    salt = os.urandom(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=dklen)
    return {
        "salt": base64.b64encode(salt).decode(),
        "hash": base64.b64encode(key).decode(),
        "n": n,
        "r": r,
        "p": p
    }

def scrypt_compare(password: str, stored: dict) -> bool:
    salt = base64.b64decode(stored["salt"])
    key = base64.b64decode(stored["hash"])
    n, r, p = stored["n"], stored["r"], stored["p"]
    new_key = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=len(key))
    return hmac_lib.compare_digest(new_key, key)
