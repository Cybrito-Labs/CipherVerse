from pydantic import BaseModel, Field

class EncodingRequest(BaseModel):
    data: str = Field(..., example="hello")

class EncodingResponse(BaseModel):
    result: str

class MorseRequest(BaseModel):
    data: str = Field(..., example="HELLO")

class BinaryRequest(BaseModel):
    data: str = Field(..., example="hello")
