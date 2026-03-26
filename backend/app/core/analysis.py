import pefile
try:
    import tlsh
except ImportError:
    tlsh = None

import hashlib
import os

def analyze_hash(h: str) -> dict:
    length = len(h)
    format_type = "Base64" if length % 4 == 0 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in h) else "Hex"
    
    analysis = []
    if length == 32: analysis.append({"algorithm": "MD5", "security": "Insecure", "crack_feasibility": "Very High"})
    elif length == 40: analysis.append({"algorithm": "SHA1", "security": "Weak", "crack_feasibility": "High"})
    elif length == 64: analysis.append({"algorithm": "SHA256", "security": "Secure", "crack_feasibility": "Low"})
    elif length == 128: analysis.append({"algorithm": "SHA512", "security": "Secure", "crack_feasibility": "Very Low"})
    
    return {"format": format_type, "length": length, "analysis": analysis}

def tlsh_hash_bytes(data: bytes) -> str:
    if tlsh is None: raise RuntimeError("TLSH library not available")
    h = tlsh.hash(data)
    if not h or h == "TNULL":
        raise ValueError("Data too small or low entropy for TLSH")
    return h

def tlsh_compare(hash1: str, hash2: str) -> int:
    if tlsh is None: raise RuntimeError("TLSH library not available")
    return tlsh.diff(hash1, hash2)


def pe_hash_analyzer(pe_path: str) -> dict:
    pe = pefile.PE(pe_path)
    results = {
        "MD5": hashlib.md5(pe.get_memory_mapped_image()).hexdigest(),
        "Imphash": pe.get_imphash(),
        "Sections": []
    }
    for sec in pe.sections:
        results["Sections"].append({
            "Name": sec.Name.decode().strip('\x00'),
            "Entropy": round(sec.get_entropy(), 2),
            "RawSize": sec.SizeOfRawData
        })
    return results

def generate_all_hashes(data: str) -> dict:
    b_data = data.encode()
    return {
        "MD5": hashlib.md5(b_data).hexdigest(),
        "SHA1": hashlib.sha1(b_data).hexdigest(),
        "SHA256": hashlib.sha256(b_data).hexdigest(),
        "SHA512": hashlib.sha512(b_data).hexdigest(),
    }
