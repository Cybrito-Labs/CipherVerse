from fastapi import APIRouter
from app.core import certificates
from app.models.certificates_models import (
    X509Response,
    TLSRequest,
    FingerprintRequest,
    X509ParseRequest
)

router = APIRouter(
    prefix="/certificates",
    tags=["Certificates & TLS"]
)

@router.post("/x509/parse", response_model=X509Response)
def x509_parse_route(request: X509ParseRequest):
    return certificates.parse_x509_certificate(request.cert_data.encode())

@router.post("/tls/analyze", response_model=X509Response)
def tls_analyze_route(request: TLSRequest):
    return certificates.analyze_tls_certificate(request.hostname, request.port)

@router.post("/x509/fingerprint", response_model=dict)
def x509_fingerprint_route(request: FingerprintRequest):
    result = certificates.generate_fingerprint(request.data.encode(), request.algorithm)
    return {"result": result}
