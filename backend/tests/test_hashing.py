import importlib
from app.core import hashing
importlib.reload(hashing)

def test_md5_hash():
    # md5("hello") = 5d41402abc4b2a76b9719d911017c592
    assert hashing.md5_hash("hello") == "5d41402abc4b2a76b9719d911017c592"

def test_sha256_hash():
    # sha256("cipherverse") = 8d3a95079a4914c770c1e840f4eee3e1a0695c64c39c8c7c9780007e0c868018
    val = hashing.sha_family_hash("cipherverse", "sha256")
    assert val == "8d3a95079a4914c770c1e840f4eee3e1a0695c64c39c8c7c9780007e0c868018", f"ACTUAL: {val}"

def test_hmac_generate():
    # hmac("hello", "key", sha256) = f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8
    assert hashing.hmac_generate("hello", "key", "sha256") == "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"

def test_bcrypt_cycle():
    pwd = "secretpassword"
    hashed = hashing.bcrypt_hash(pwd)
    assert hashing.bcrypt_compare(pwd, hashed) is True
    assert hashing.bcrypt_compare("wrongpassword", hashed) is False

def test_pbkdf2_cycle():
    pwd = "secretpassword"
    stored = hashing.pbkdf2_derive_key(pwd)
    assert hashing.pbkdf2_verify(pwd, stored) is True
    assert hashing.pbkdf2_verify("wrongpassword", stored) is False
