import os
import hashlib
import math
import json
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

def file_hash(filepath: str, algorithm: str = "SHA256") -> str:
    h = hashlib.new(algorithm.lower())
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def file_multi_hash(filepath: str) -> dict:
    algorithms = ["md5", "sha1", "sha256", "sha512"]
    hashers = {alg: hashlib.new(alg) for alg in algorithms}
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            for h in hashers.values():
                h.update(chunk)
    return {name: h.hexdigest() for name, h in hashers.items()}

def directory_hash(directory: str, algorithm: str = "SHA256", ignore_hidden: bool = True) -> str:
    dir_hasher = hashlib.new(algorithm.lower())
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        files.sort()
        for filename in files:
            if ignore_hidden and filename.startswith("."):
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, directory)
            dir_hasher.update(rel_path.encode())
            with open(filepath, "rb") as f:
                while chunk := f.read(8192):
                    dir_hasher.update(chunk)
    return dir_hasher.hexdigest()

def compare_file_hashes(file1: str, file2: str, algorithm: str = "SHA256") -> dict:
    h1 = file_hash(file1, algorithm)
    h2 = file_hash(file2, algorithm)
    return {"match": h1 == h2, "file1": h1, "file2": h2}

def file_encrypt_aes(input_file: str, output_file: str, key: bytes):
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    with open(input_file, "rb") as f:
        data = f.read()
    ct = cipher.encrypt(pad(data, AES.block_size))
    with open(output_file, "wb") as f:
        f.write(iv + ct)

def file_decrypt_aes(input_file: str, output_file: str, key: bytes):
    with open(input_file, "rb") as f:
        iv = f.read(16)
        ct = f.read()
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    with open(output_file, "wb") as f:
        f.write(pt)

def calculate_entropy(filepath: str) -> float:
    with open(filepath, "rb") as f:
        data = f.read()
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    entropy = 0.0
    for count in freq:
        if count > 0:
            p = count / len(data)
            entropy -= p * math.log2(p)
    return round(entropy, 4)

def interpret_entropy(entropy: float) -> str:
    if entropy < 3: return "Plain text / Structured"
    if entropy < 6: return "Moderate / Binary"
    if entropy < 7.5: return "High / Compressed"
    return "Very High / Encrypted"

def bit_balance_test(data: bytes) -> dict:
    ones = sum(bin(b).count('1') for b in data)
    zeros = len(data) * 8 - ones
    ratio = ones / (ones + zeros) if (ones + zeros) > 0 else 0
    return {"ones": ones, "zeros": zeros, "ratio": round(ratio, 4)}

def runs_test(data: bytes) -> int:
    bits = "".join(f"{b:08b}" for b in data)
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i-1]:
            runs += 1
    return runs

def chi_square_test(data: bytes) -> float:
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    expected = len(data) / 256
    chi = sum((count - expected)**2 / expected for count in freq) if expected > 0 else 0
    return round(chi, 4)

def randomness_test_suite(filepath: str) -> dict:
    with open(filepath, "rb") as f:
        data = f.read()
    return {
        "entropy": calculate_entropy(filepath),
        "bit_balance": bit_balance_test(data),
        "runs": runs_test(data),
        "chi_square": chi_square_test(data)
    }
