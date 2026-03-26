import pytest

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "CipherVerse" in response.json()["message"]

def test_classical_caesar(client):
    response = client.post("/classical/caesar", json={"text": "HELLO", "shift": 3})
    assert response.status_code == 200
    assert response.json()["result"] == "KHOOR"

def test_hashing_md5(client):
    response = client.post("/hashing/hash", json={"text": "hello", "algorithm": "md5"})
    assert response.status_code == 200
    assert response.json()["result"] == "5d41402abc4b2a76b9719d911017c592"

def test_encoding_base64_encode(client):
    response = client.post("/encoding/base64/encode", json={"data": "hello"})
    assert response.status_code == 200
    assert response.json()["result"] == "aGVsbG8="

def test_asymmetric_rsa_flow(client):
    # 1. Generate keys
    gen_resp = client.post("/asymmetric/rsa/generate-keys")
    assert gen_resp.status_code == 200
    pub = gen_resp.json()["public_key"]
    priv = gen_resp.json()["private_key"]
    
    # 2. Encrypt
    enc_resp = client.post("/asymmetric/rsa/encrypt", json={"text": "secret", "public_key": pub})
    assert enc_resp.status_code == 200
    encrypted = enc_resp.json()["result"]
    
    # 3. Decrypt
    dec_resp = client.post("/asymmetric/rsa/decrypt", json={"encrypted_base64": encrypted, "private_key": priv})
    assert dec_resp.status_code == 200
    assert dec_resp.json()["result"] == "secret"

def test_utilities_password_strength(client):
    response = client.post("/utilities/password/strength", json={"password": "Password123!"})
    assert response.status_code == 200
    assert response.json()["strength"] in ["Moderate", "Strong", "Very Strong"]

