from fastapi import APIRouter
from app.core import steganography
from app.models.steganography_models import (
    StegoTextRequest,
    StegoImageRequest,
    StegoAudioRequest,
    StegoResponse
)

router = APIRouter(
    prefix="/steganography",
    tags=["Steganography"]
)

@router.post("/text/encode", response_model=StegoResponse)
def steg_text_encode_route(request: StegoTextRequest):
    return {"result": steganography.steg_text_encode(request.cover_text, request.secret)}

@router.post("/text/decode", response_model=StegoResponse)
def steg_text_decode_route(text: str):
    return {"result": steganography.steg_text_decode(text)}

@router.post("/image/encode", response_model=StegoResponse)
def steg_image_encode_route(request: StegoImageRequest):
    steganography.image_lsb_encode(request.input_img, request.output_img, request.secret)
    return {"result": "Success"}

@router.post("/audio/encode", response_model=StegoResponse)
def steg_audio_encode_route(request: StegoAudioRequest):
    steganography.audio_lsb_encode(request.input_wav, request.output_wav, request.secret)
    return {"result": "Success"}
