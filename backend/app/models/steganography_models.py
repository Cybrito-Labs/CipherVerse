from pydantic import BaseModel, Field

class StegoTextRequest(BaseModel):
    cover_text: str = Field(..., example="This is a normal message.")
    secret: str = Field(..., example="top secret")

class StegoImageRequest(BaseModel):
    input_img: str = Field(..., example="C:/test/input.png")
    output_img: str = Field(..., example="C:/test/stego.png")
    secret: str = Field(..., example="top secret")

class StegoAudioRequest(BaseModel):
    input_wav: str = Field(..., example="C:/test/input.wav")
    output_wav: str = Field(..., example="C:/test/stego.wav")
    secret: str = Field(..., example="top secret")

class StegoResponse(BaseModel):
    result: str
