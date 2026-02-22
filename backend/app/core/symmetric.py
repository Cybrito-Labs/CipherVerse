import hashlib
def xor_cipher(data: str, key: str) -> str:
    result = []
    key_len = len(key)

    for i, char in enumerate(data):
        xor_val = ord(char) ^ ord(key[i % key_len])
        result.append(format(xor_val, '02x')) 

    return ' '.join(result)
def xor_decipher(hex_data: str, key: str) -> str:
    result = []
    bytes_data = hex_data.split()
    key_len = len(key)

    for i, byte in enumerate(bytes_data):
        xor_val = int(byte, 16) ^ ord(key[i % key_len])
        result.append(chr(xor_val))

    return ''.join(result)
def xor_bruteforce(hex_data: str):
    import string
    results = []

    bytes_data = bytes.fromhex(hex_data.replace(" ", ""))

    for key in range(256):
        decoded = ''.join(chr(b ^ key) for b in bytes_data)

        if all(c in string.printable for c in decoded):
            results.append((key, decoded))

    return results
def ciphersaber2_encrypt(text: str, key: str, rounds: int = 20) -> str:
    import os, base64
    iv = os.urandom(10)
    return base64.b64encode(iv + rc4_crypt(text.encode(), key.encode() + iv)).decode()
def ciphersaber2_decrypt(data: str, key: str, rounds: int = 20) -> str:
    import base64
    raw = base64.b64decode(data)
    iv, cipher = raw[:10], raw[10:]
    return rc4_crypt(cipher, key.encode() + iv).decode(errors="ignore")
def rc2_encrypt(plaintext: str, key: str) -> str:
    from Crypto.Cipher import ARC2
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    import base64
    key_bytes = key.encode()
    iv = get_random_bytes(8)  

    cipher = ARC2.new(key_bytes, ARC2.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), 8))
    return base64.b64encode(iv + ciphertext).decode()
def rc2_decrypt(ciphertext_b64: str, key: str) -> str:
    from Crypto.Cipher import ARC2
    from Crypto.Util.Padding import unpad
    import base64
    data = base64.b64decode(ciphertext_b64)
    iv = data[:8]
    ciphertext = data[8:]

    cipher = ARC2.new(key.encode(), ARC2.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), 8)

    return plaintext.decode()
def rc4_init(key: bytes):
    S = list(range(256))
    j = 0

    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) % 256
        S[i], S[j] = S[j], S[i]

    return S


def rc4_generate(S, data_len):
    i = j = 0
    keystream = []

    for _ in range(data_len):
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        K = S[(S[i] + S[j]) % 256]
        keystream.append(K)

    return keystream


def rc4_crypt(data: bytes, key: bytes) -> bytes:
    S = rc4_init(key)
    keystream = rc4_generate(S, len(data))
    return bytes([d ^ k for d, k in zip(data, keystream)])


def rc4_encrypt(text: str, key: str) -> str:
    return rc4_crypt(text.encode(), key.encode()).hex()


def rc4_decrypt(hexdata: str, key: str) -> str:
    return rc4_crypt(bytes.fromhex(hexdata), key.encode()).decode()

def rc4_drop_crypt(data: bytes, key: bytes, drop_n: int = 768) -> bytes:
    S = rc4_init(key)
    _ = rc4_generate(S, drop_n)
    keystream = rc4_generate(S, len(data))
    return bytes([d ^ k for d, k in zip(data, keystream)])


def rc4_drop_encrypt(text: str, key: str, drop_n: int = 768) -> str:
    return rc4_drop_crypt(text.encode(), key.encode(), drop_n).hex()


def rc4_drop_decrypt(hexdata: str, key: str, drop_n: int = 768) -> str:
    return rc4_drop_crypt(bytes.fromhex(hexdata), key.encode(), drop_n).decode()
def derive_aes_key(password: str, key_size: int = 32) -> bytes:
    import hashlib
    """
    key_size:
      16 → AES-128
      24 → AES-192
      32 → AES-256
    """
    return hashlib.sha256(password.encode()).digest()[:key_size]
def aes_encrypt(plaintext: str,password: str,mode: str = "CBC",key_size: int = 32) -> str:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    import base64

    key = derive_aes_key(password, key_size)
    mode = mode.upper()

    if mode == "ECB":
        cipher = AES.new(key, AES.MODE_ECB)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), 16))
        return base64.b64encode(ciphertext).decode()

    elif mode == "CBC":
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), 16))
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "CFB":
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CFB, iv)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "OFB":
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_OFB, iv)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "CTR":
        cipher = AES.new(key, AES.MODE_CTR)
        ciphertext = cipher.encrypt(plaintext.encode())
        nonce = cipher.nonce
        return base64.b64encode(len(nonce).to_bytes(1, "big") + nonce + ciphertext).decode()

    elif mode == "GCM":
        nonce = get_random_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
        return base64.b64encode(nonce + tag + ciphertext).decode()

    elif mode == "CCM":
        nonce = get_random_bytes(11)
        cipher = AES.new(key, AES.MODE_CCM, nonce=nonce)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(nonce + cipher.digest() + ciphertext).decode()

    elif mode == "OCB":
        nonce = get_random_bytes(15)
        cipher = AES.new(key, AES.MODE_OCB, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
        return base64.b64encode(nonce + tag + ciphertext).decode()

    elif mode == "XTS":
        cipher = AES.new(key + key, AES.MODE_XTS)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(ciphertext).decode()

    elif mode == "SIV":
        cipher = AES.new(key, AES.MODE_SIV)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
        return base64.b64encode(tag + ciphertext).decode()

    else:
        raise ValueError("Unsupported AES mode")
def aes_decrypt(ciphertext_b64: str,password: str,mode: str = "CBC",key_size: int = 32) -> str:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    import base64

    key = derive_aes_key(password, key_size)
    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        cipher = AES.new(key, AES.MODE_ECB)
        return unpad(cipher.decrypt(data), 16).decode()

    elif mode == "CBC":
        iv, ct = data[:16], data[16:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), 16).decode()

    elif mode == "CFB":
        iv, ct = data[:16], data[16:]
        cipher = AES.new(key, AES.MODE_CFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "OFB":
        iv, ct = data[:16], data[16:]
        cipher = AES.new(key, AES.MODE_OFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "CTR":
        nonce_len = data[0]
        nonce = data[1:1+nonce_len]
        ct = data[1+nonce_len:]
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "GCM":
        nonce, tag, ct = data[:12], data[12:28], data[28:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode()

    elif mode == "CCM":
        nonce, tag, ct = data[:11], data[11:27], data[27:]
        cipher = AES.new(key, AES.MODE_CCM, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode()

    elif mode == "OCB":
        nonce, tag, ct = data[:15], data[15:31], data[31:]
        cipher = AES.new(key, AES.MODE_OCB, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode()

    elif mode == "XTS":
        cipher = AES.new(key + key, AES.MODE_XTS)
        return cipher.decrypt(data).decode(errors="ignore")

    elif mode == "SIV":
        tag, ct = data[:16], data[16:]
        cipher = AES.new(key, AES.MODE_SIV)
        return cipher.decrypt_and_verify(ct, tag).decode()

    else:
        raise ValueError("Unsupported AES mode")
def des_encrypt(plaintext: str, password: str, mode: str = "CBC") -> str:
    from Crypto.Cipher import DES
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    from Crypto.Util import Counter
    import base64
    import hashlib

    key = hashlib.md5(password.encode()).digest()[:8]
    mode = mode.upper()

    if mode == "ECB":
        cipher = DES.new(key, DES.MODE_ECB)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), 8))
        return base64.b64encode(ciphertext).decode()

    elif mode == "CBC":
        iv = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), 8))
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "CFB":
        iv = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_CFB, iv)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "OFB":
        iv = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_OFB, iv)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "CTR":
        ctr = Counter.new(64)
        cipher = DES.new(key, DES.MODE_CTR, counter=ctr)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(ciphertext).decode()

    else:
        raise ValueError("Unsupported DES mode")
def des_decrypt(ciphertext_b64: str, password: str, mode: str = "CBC") -> str:
    from Crypto.Cipher import DES
    from Crypto.Util.Padding import unpad
    from Crypto.Util import Counter
    import base64
    import hashlib

    key = hashlib.md5(password.encode()).digest()[:8]
    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        cipher = DES.new(key, DES.MODE_ECB)
        return unpad(cipher.decrypt(data), 8).decode()

    elif mode == "CBC":
        iv, ct = data[:8], data[8:]
        cipher = DES.new(key, DES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), 8).decode()

    elif mode == "CFB":
        iv, ct = data[:8], data[8:]
        cipher = DES.new(key, DES.MODE_CFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "OFB":
        iv, ct = data[:8], data[8:]
        cipher = DES.new(key, DES.MODE_OFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "CTR":
        ctr = Counter.new(64)
        cipher = DES.new(key, DES.MODE_CTR, counter=ctr)
        return cipher.decrypt(data).decode(errors="ignore")

    else:
        raise ValueError("Unsupported DES mode")
def tdes_encrypt(plaintext: str, password: str, mode: str = "CBC") -> str:
    from Crypto.Cipher import DES3
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    from Crypto.Util import Counter
    import base64
    import hashlib

    key = hashlib.sha256(password.encode()).digest()[:24]
    key = DES3.adjust_key_parity(key)

    mode = mode.upper()
    data = plaintext.encode()

    if mode == "ECB":
        cipher = DES3.new(key, DES3.MODE_ECB)
        ct = cipher.encrypt(pad(data, 8))
        return base64.b64encode(ct).decode()

    elif mode == "CBC":
        iv = get_random_bytes(8)
        cipher = DES3.new(key, DES3.MODE_CBC, iv)
        ct = cipher.encrypt(pad(data, 8))
        return base64.b64encode(iv + ct).decode()

    elif mode == "CFB":
        iv = get_random_bytes(8)
        cipher = DES3.new(key, DES3.MODE_CFB, iv)
        ct = cipher.encrypt(data)
        return base64.b64encode(iv + ct).decode()

    elif mode == "OFB":
        iv = get_random_bytes(8)
        cipher = DES3.new(key, DES3.MODE_OFB, iv)
        ct = cipher.encrypt(data)
        return base64.b64encode(iv + ct).decode()

    elif mode == "CTR":
        ctr = Counter.new(64)
        cipher = DES3.new(key, DES3.MODE_CTR, counter=ctr)
        ct = cipher.encrypt(data)
        return base64.b64encode(ct).decode()

    elif mode == "EAX":
        cipher = DES3.new(key, DES3.MODE_EAX)
        ct, tag = cipher.encrypt_and_digest(data)
        return base64.b64encode(cipher.nonce + tag + ct).decode()

    else:
        raise ValueError("Unsupported Triple DES mode")
def tdes_decrypt(ciphertext_b64: str, password: str, mode: str = "CBC") -> str:
    from Crypto.Cipher import DES3
    from Crypto.Util.Padding import unpad
    from Crypto.Util import Counter
    import base64
    import hashlib

    key = hashlib.sha256(password.encode()).digest()[:24]
    key = DES3.adjust_key_parity(key)

    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        cipher = DES3.new(key, DES3.MODE_ECB)
        return unpad(cipher.decrypt(data), 8).decode()

    elif mode == "CBC":
        iv, ct = data[:8], data[8:]
        cipher = DES3.new(key, DES3.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), 8).decode()

    elif mode == "CFB":
        iv, ct = data[:8], data[8:]
        cipher = DES3.new(key, DES3.MODE_CFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "OFB":
        iv, ct = data[:8], data[8:]
        cipher = DES3.new(key, DES3.MODE_OFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "CTR":
        ctr = Counter.new(64)
        cipher = DES3.new(key, DES3.MODE_CTR, counter=ctr)
        return cipher.decrypt(data).decode(errors="ignore")

    elif mode == "EAX":
        nonce, tag, ct = data[:16], data[16:32], data[32:]
        cipher = DES3.new(key, DES3.MODE_EAX, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode()

    else:
        raise ValueError("Unsupported Triple DES mode")
def derive_blowfish_key(password: str, max_len: int = 56) -> bytes:
    return hashlib.sha256(password.encode()).digest()[:max_len]
def blowfish_encrypt(plaintext: str,password: str,mode: str = "CBC") -> str:
    from Crypto.Cipher import Blowfish
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    import base64

    key = derive_blowfish_key(password)
    mode = mode.upper()

    if mode == "ECB":
        cipher = Blowfish.new(key, Blowfish.MODE_ECB)
        ct = cipher.encrypt(pad(plaintext.encode(), 8))
        return base64.b64encode(ct).decode()

    elif mode == "CBC":
        iv = get_random_bytes(8)
        cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
        ct = cipher.encrypt(pad(plaintext.encode(), 8))
        return base64.b64encode(iv + ct).decode()

    elif mode == "CFB":
        iv = get_random_bytes(8)
        cipher = Blowfish.new(key, Blowfish.MODE_CFB, iv)
        ct = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ct).decode()

    elif mode == "OFB":
        iv = get_random_bytes(8)
        cipher = Blowfish.new(key, Blowfish.MODE_OFB, iv)
        ct = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ct).decode()

    elif mode == "CTR":
        cipher = Blowfish.new(key, Blowfish.MODE_CTR)
        ct = cipher.encrypt(plaintext.encode())
        return base64.b64encode(cipher.nonce + ct).decode()

    else:
        raise ValueError("Unsupported Blowfish mode")
def blowfish_decrypt(ciphertext_b64: str,password: str,mode: str = "CBC") -> str:
    from Crypto.Cipher import Blowfish
    from Crypto.Util.Padding import unpad
    import base64

    key = derive_blowfish_key(password)
    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        cipher = Blowfish.new(key, Blowfish.MODE_ECB)
        return unpad(cipher.decrypt(data), 8).decode()

    elif mode == "CBC":
        iv, ct = data[:8], data[8:]
        cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), 8).decode()

    elif mode == "CFB":
        iv, ct = data[:8], data[8:]
        cipher = Blowfish.new(key, Blowfish.MODE_CFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "OFB":
        iv, ct = data[:8], data[8:]
        cipher = Blowfish.new(key, Blowfish.MODE_OFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "CTR":
        nonce, ct = data[:8], data[8:]
        cipher = Blowfish.new(key, Blowfish.MODE_CTR, nonce=nonce)
        return cipher.decrypt(ct).decode(errors="ignore")

    else:
        raise ValueError("Unsupported Blowfish mode")
def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)
def pkcs7_unpad(data: bytes) -> bytes:
    pad_len = data[-1]
    return data[:-pad_len]
def sm4_encrypt(plaintext: str, password: str, mode: str = "ECB") -> str:
    try:
        from gmssl.sm4 import CryptSM4, SM4_ENCRYPT
    except ImportError:
        raise RuntimeError("SM4 requires gmssl library")
    import hashlib, base64, os

    key = hashlib.sha256(password.encode()).digest()[:16]
    crypt_sm4 = CryptSM4()
    crypt_sm4.set_key(key, SM4_ENCRYPT)

    data = pkcs7_pad(plaintext.encode())
    mode = mode.upper()

    if mode == "ECB":
        ct = crypt_sm4.crypt_ecb(data)
        return base64.b64encode(ct).decode()

    elif mode == "CBC":
        iv = os.urandom(16)
        ct = crypt_sm4.crypt_cbc(iv, data)
        return base64.b64encode(iv + ct).decode()

    else:
        raise ValueError("gmssl supports ONLY ECB and CBC modes")
def sm4_decrypt(ciphertext_b64: str, password: str, mode: str = "ECB") -> str:
    try:
        from gmssl.sm4 import CryptSM4, SM4_DECRYPT
    except ImportError:
        raise RuntimeError("SM4 requires gmssl library")
    import hashlib, base64

    key = hashlib.sha256(password.encode()).digest()[:16]
    crypt_sm4 = CryptSM4()
    crypt_sm4.set_key(key, SM4_DECRYPT)

    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        pt = crypt_sm4.crypt_ecb(data)
        return pkcs7_unpad(pt).decode()

    elif mode == "CBC":
        iv, ct = data[:16], data[16:]
        pt = crypt_sm4.crypt_cbc(iv, ct)
        return pkcs7_unpad(pt).decode()

    else:
        raise ValueError("gmssl supports ONLY ECB and CBC modes")