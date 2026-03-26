from app.core import asymmetric
import base64

def test_rsa_key_gen():
    pub, priv = asymmetric.rsa_generate_keys(1024)
    assert "BEGIN PUBLIC KEY" in pub
    assert "BEGIN PRIVATE KEY" in priv

def test_rsa_encrypt_decrypt():
    pub, priv = asymmetric.rsa_generate_keys(1024)
    text = "secret message"
    encrypted = asymmetric.rsa_encrypt(text, pub)
    decrypted = asymmetric.rsa_decrypt(encrypted, priv)
    assert decrypted == text

def test_rsa_sign_verify():
    pub, priv = asymmetric.rsa_generate_keys(1024)
    message = "message to sign"
    signature = asymmetric.rsa_sign(message, priv)
    assert asymmetric.rsa_verify(message, signature, pub) is True
    assert asymmetric.rsa_verify("other message", signature, pub) is False

def test_dsa_sign_verify():
    pub, priv = asymmetric.dsa_generate_keys(1024)
    message = "message to sign"
    signature = asymmetric.dsa_sign(message, priv)
    assert asymmetric.dsa_verify(message, signature, pub) is True

