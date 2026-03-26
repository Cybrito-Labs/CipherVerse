from app.core import symmetric

def test_xor_cipher_cycle():
    data = "hello"
    key = "secret"
    encrypted = symmetric.xor_cipher(data, key)
    decrypted = symmetric.xor_decipher(encrypted, key)
    assert decrypted == data

def test_rc4_cycle():
    data = "hello world"
    key = "secretkey"
    encrypted = symmetric.rc4_encrypt(data, key)
    decrypted = symmetric.rc4_decrypt(encrypted, key)
    assert decrypted == data

def test_aes_cbc_cycle():
    data = "secret message that needs padding"
    password = "strongpassword"
    encrypted = symmetric.aes_encrypt(data, password, mode="CBC")
    decrypted = symmetric.aes_decrypt(encrypted, password, mode="CBC")
    assert decrypted == data

def test_aes_gcm_cycle():
    data = "authenticated data"
    password = "strongpassword"
    encrypted = symmetric.aes_encrypt(data, password, mode="GCM")
    decrypted = symmetric.aes_decrypt(encrypted, password, mode="GCM")
    assert decrypted == data

def test_blowfish_cycle():
    data = "blowfish test"
    password = "password"
    encrypted = symmetric.blowfish_encrypt(data, password, mode="CBC")
    decrypted = symmetric.blowfish_decrypt(encrypted, password, mode="CBC")
    assert decrypted == data
