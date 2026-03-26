import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, dh, ec, dsa, ed25519, x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

# --- RSA ---
def rsa_generate_keys(key_size: int = 2048):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size, backend=default_backend())
    public_key = private_key.public_key()
    
    private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()).decode()
    public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    
    return public_pem, private_pem

def rsa_encrypt(text: str, public_key_pem: str) -> str:
    public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())
    encrypted = public_key.encrypt(text.encode(), padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return base64.b64encode(encrypted).decode()

def rsa_decrypt(encrypted_base64: str, private_key_pem: str) -> str:
    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None, backend=default_backend())
    decrypted = private_key.decrypt(base64.b64decode(encrypted_base64), padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return decrypted.decode()

def rsa_sign(message: str, private_key_pem: str) -> str:
    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None, backend=default_backend())
    signature = private_key.sign(message.encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    return base64.b64encode(signature).decode()

def rsa_verify(message: str, signature_base64: str, public_key_pem: str) -> bool:
    public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())
    try:
        public_key.verify(base64.b64decode(signature_base64), message.encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
        return True
    except Exception:
        return False

# --- Diffie-Hellman ---
def dh_generate_parameters():
    parameters = dh.generate_parameters(generator=2, key_size=2048, backend=default_backend())
    return parameters

def dh_generate_keypair(parameters):
    private_key = parameters.generate_private_key()
    public_key = private_key.public_key()
    return private_key, public_key

def dh_compute_shared_secret(private_key, peer_public_key):
    shared_key = private_key.exchange(peer_public_key)
    # Perform key derivation
    derived_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'handshake data', backend=default_backend()).derive(shared_key)
    return derived_key

# --- ECDH ---
def ecdh_generate_keypair(curve=ec.SECP256R1()):
    private_key = ec.generate_private_key(curve, backend=default_backend())
    public_key = private_key.public_key()
    return private_key, public_key

def ecdh_shared_secret(private_key, peer_public_key):
    shared_key = private_key.exchange(ec.ECDH(), peer_public_key)
    derived_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'ecdh data', backend=default_backend()).derive(shared_key)
    return derived_key

# --- DSA ---
def dsa_generate_keys(key_size: int = 2048):
    private_key = dsa.generate_private_key(key_size=key_size, backend=default_backend())
    public_key = private_key.public_key()
    
    private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()).decode()
    public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    
    return public_pem, private_pem

def dsa_sign(message: str, private_key_pem: str) -> str:
    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None, backend=default_backend())
    signature = private_key.sign(message.encode(), hashes.SHA256())
    return base64.b64encode(signature).decode()

def dsa_verify(message: str, signature_base64: str, public_key_pem: str) -> bool:
    public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())
    try:
        public_key.verify(base64.b64decode(signature_base64), message.encode(), hashes.SHA256())
        return True
    except Exception:
        return False

# --- ECDSA ---
def ecdsa_generate_keys(curve=ec.SECP256R1()):
    private_key = ec.generate_private_key(curve, backend=default_backend())
    public_key = private_key.public_key()
    
    private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()).decode()
    public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    
    return public_pem, private_pem

def ecdsa_sign(message: str, private_key_pem: str) -> str:
    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None, backend=default_backend())
    signature = private_key.sign(message.encode(), ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(signature).decode()

def ecdsa_verify(message: str, signature_base64: str, public_key_pem: str) -> bool:
    public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())
    try:
        public_key.verify(base64.b64decode(signature_base64), message.encode(), ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False

# --- Ed25519 ---
def ed25519_generate_keys():
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()).decode()
    public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    
    return public_pem, private_pem

def ed25519_sign(message: str, private_key_pem: str) -> str:
    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None, backend=default_backend())
    signature = private_key.sign(message.encode())
    return base64.b64encode(signature).decode()

def ed25519_verify(message: str, signature_base64: str, public_key_pem: str) -> bool:
    public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend())
    try:
        public_key.verify(base64.b64decode(signature_base64), message.encode())
        return True
    except Exception:
        return False

# --- X25519 ---
def x25519_generate_keypair():
    private_key = x25519.X25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key

def x25519_shared_secret(private_key, peer_public_key):
    shared_key = private_key.exchange(peer_public_key)
    derived_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'x25519 data', backend=default_backend()).derive(shared_key)
    return derived_key
