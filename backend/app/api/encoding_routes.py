from fastapi import APIRouter
from app.core import encoding
from app.models.encoding_models import (
    EncodingRequest,
    EncodingResponse,
    MorseRequest,
    BinaryRequest
)

router = APIRouter(
    prefix="/encoding",
    tags=["Encoding & Decoding"]
)

@router.post("/base64/encode", response_model=EncodingResponse)
def base64_encode_route(request: EncodingRequest):
    return {"result": encoding.base64_encoder(request.data)}

@router.post("/base64/decode", response_model=EncodingResponse)
def base64_decode_route(request: EncodingRequest):
    return {"result": encoding.base64_decoder(request.data)}

@router.post("/hex/encode", response_model=EncodingResponse)
def hex_encode_route(request: EncodingRequest):
    return {"result": encoding.hex_encoder(request.data)}

@router.post("/hex/decode", response_model=EncodingResponse)
def hex_decode_route(request: EncodingRequest):
    return {"result": encoding.hex_decoder(request.data)}

@router.post("/url/encode", response_model=EncodingResponse)
def url_encode_route(request: EncodingRequest):
    return {"result": encoding.url_encoder(request.data)}

@router.post("/url/decode", response_model=EncodingResponse)
def url_decode_route(request: EncodingRequest):
    return {"result": encoding.url_decoder(request.data)}

@router.post("/binary/encode", response_model=EncodingResponse)
def binary_encode_route(request: EncodingRequest):
    return {"result": encoding.binary_encoder(request.data)}

@router.post("/binary/decode", response_model=EncodingResponse)
def binary_decode_route(request: EncodingRequest):
    return {"result": encoding.binary_decoder(request.data)}

@router.post("/morse/encode", response_model=EncodingResponse)
def morse_encode_route(request: MorseRequest):
    return {"result": encoding.morse_encoder(request.data)}

@router.post("/morse/decode", response_model=EncodingResponse)
def morse_decode_route(request: MorseRequest):
    return {"result": encoding.morse_decoder(request.data)}
