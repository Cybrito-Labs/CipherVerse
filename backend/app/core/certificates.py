import ssl
import socket
from datetime import datetime
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes

def parse_x509_certificate(cert_data: bytes) -> dict:
    try:
        cert = x509.load_pem_x509_certificate(cert_data, default_backend())
    except ValueError:
        cert = x509.load_der_x509_certificate(cert_data, default_backend())
    
    return {
        "Subject": str(cert.subject),
        "Issuer": str(cert.issuer),
        "Serial Number": cert.serial_number,
        "Not Before": cert.not_valid_before_utc.isoformat(),
        "Not After": cert.not_valid_after_utc.isoformat(),
        "Fingerprint SHA256": cert.fingerprint(hashes.SHA256()).hex(),
        "Version": cert.version.name,
        "Extensions": [str(ext.value) for ext in cert.extensions]
    }

def analyze_tls_certificate(hostname: str, port: int = 443) -> dict:
    context = ssl.create_default_context()
    with socket.create_connection((hostname, port)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert_der = ssock.getpeercert(binary_form=True)
            cert = x509.load_der_x509_certificate(cert_der, default_backend())
            
            return parse_x509_certificate(cert_der)

def pem_to_der(pem_path: str, der_path: str):
    with open(pem_path, "rb") as f:
        pem_data = f.read()
    
    # Check if it's a certificate or key
    try:
        obj = x509.load_pem_x509_certificate(pem_data, default_backend())
        der_data = obj.public_bytes(serialization.Encoding.DER)
    except ValueError:
        # Try loading as a private key
        try:
            obj = serialization.load_pem_private_key(pem_data, password=None, backend=default_backend())
            der_data = obj.private_bytes(encoding=serialization.Encoding.DER, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
        except ValueError:
            # Try loading as a public key
            obj = serialization.load_pem_public_key(pem_data, backend=default_backend())
            der_data = obj.public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
            
    with open(der_path, "wb") as f:
        f.write(der_data)

def der_to_pem(der_path: str, pem_path: str):
    with open(der_path, "rb") as f:
        der_data = f.read()
    
    try:
        obj = x509.load_der_x509_certificate(der_data, default_backend())
        pem_data = obj.public_bytes(serialization.Encoding.PEM)
    except ValueError:
        try:
            obj = serialization.load_der_private_key(der_data, password=None, backend=default_backend())
            pem_data = obj.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
        except ValueError:
            obj = serialization.load_der_public_key(der_data, backend=default_backend())
            pem_data = obj.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)

    with open(pem_path, "wb") as f:
        f.write(pem_data)

def extract_public_key_from_cert(cert_path: str) -> str:
    with open(cert_path, "rb") as f:
        data = f.read()
    try:
        cert = x509.load_pem_x509_certificate(data, default_backend())
    except ValueError:
        cert = x509.load_der_x509_certificate(data, default_backend())
    
    public_key = cert.public_key()
    return public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()

def generate_fingerprint(data: bytes, algo: str = "sha256") -> str:
    h = hashes.Hash(getattr(hashes, algo.upper())(), backend=default_backend())
    h.update(data)
    return h.finalize().hex()
