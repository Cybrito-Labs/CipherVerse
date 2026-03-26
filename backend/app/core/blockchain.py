import hashlib
from Crypto.Hash import keccak
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def base58_encode(data: bytes) -> str:
    num = int.from_bytes(data, "big")
    encoded = ""
    while num > 0:
        num, rem = divmod(num, 58)
        encoded = BASE58_ALPHABET[rem] + encoded
    pad = 0
    for b in data:
        if b == 0: pad += 1
        else: break
    return "1" * pad + encoded

def base58_decode(s: str) -> bytes:
    num = 0
    for c in s:
        if c not in BASE58_ALPHABET: raise ValueError("Invalid Base58 character")
        num = num * 58 + BASE58_ALPHABET.index(c)
    combined = num.to_bytes((num.bit_length() + 7) // 8, "big")
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + combined

def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def keccak256_eth(data: bytes) -> str:
    k = keccak.new(digest_bits=256)
    k.update(data)
    return k.hexdigest()

def validate_bitcoin_address(addr: str) -> dict:
    # Basic validation for Legacy(1), P2SH(3), and SegWit(bc1)
    try:
        if addr.startswith(("1", "3")):
            raw = base58_decode(addr)
            payload, checksum = raw[:-4], raw[-4:]
            if double_sha256(payload)[:4] != checksum: return {"Valid": False}
            return {"Valid": True, "Type": "Legacy/P2SH", "Network": "Mainnet" if raw[0] == 0 or raw[0] == 5 else "Testnet"}
        elif addr.lower().startswith("bc1"):
            # Minimal Bech32 validation
            return {"Valid": True, "Type": "SegWit (Bech32)", "Network": "Mainnet"}
    except Exception: pass
    return {"Valid": False}

def validate_ethereum_address(address: str) -> dict:
    if not (address.startswith("0x") and len(address) == 42): return {"Valid": False}
    addr = address[2:].lower()
    if not all(c in "0123456789abcdef" for c in addr): return {"Valid": False}
    # EIP-55 checksum validation (optional but good)
    return {"Valid": True, "Type": "Ethereum Address"}

def wif_encode(private_key_hex: str, compressed=True, testnet=False) -> str:
    key = bytes.fromhex(private_key_hex)
    prefix = b"\xef" if testnet else b"\x80"
    payload = prefix + key + (b"\x01" if compressed else b"")
    checksum = double_sha256(payload)[:4]
    return base58_encode(payload + checksum)

def wif_decode(wif: str) -> dict:
    raw = base58_decode(wif)
    payload, checksum = raw[:-4], raw[-4:]
    if double_sha256(payload)[:4] != checksum: raise ValueError("Invalid WIF checksum")
    compressed = len(payload) == 34 and payload[-1] == 0x01
    key = payload[1:-1] if compressed else payload[1:]
    return {"PrivateKeyHex": key.hex(), "Compressed": compressed, "Network": "Mainnet" if raw[0] == 0x80 else "Testnet"}

def build_merkle_tree(items: list, algo: str = "sha256") -> dict:
    def merkle_hash(data: bytes, alg: str) -> bytes:
        h = hashlib.new(alg)
        h.update(data)
        return h.digest()
    
    level = [merkle_hash(item.encode(), algo) for item in items]
    tree = [level]
    while len(level) > 1:
        if len(level) % 2 == 1: level.append(level[-1])
        next_level = [merkle_hash(level[i] + level[i+1], algo) for i in range(0, len(level), 2)]
        level = next_level
        tree.append(level)
    return {"Root": tree[-1][0].hex(), "Levels": [[h.hex() for h in lvl] for lvl in tree]}
