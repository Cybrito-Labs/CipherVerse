from pydantic import BaseModel, Field


class CaesarRequest(BaseModel):
    text: str = Field(..., example="HELLO")
    shift: int = Field(..., example=3)


class VigenereRequest(BaseModel):
    text: str = Field(..., example="HELLO")
    keyword: str = Field(..., example="KEY")


class AtbashRequest(BaseModel):
    text: str = Field(..., example="HELLO")


class ClassicalResponse(BaseModel):
    result: str

class BaconRequest(BaseModel):
    text: str = Field(..., example="HELLO")

class BifidRequest(BaseModel):
    text: str = Field(..., example="HELLO")
    keyword: str = Field(..., example="KEY")

class AffineRequest(BaseModel):
    text: str = Field(..., example="HELLO")
    a: int = Field(..., example=3)
    b: int = Field(..., example=5)

class A1Z26Request(BaseModel):
    text: str = Field(..., example="HELLO")

class RailFenceRequest(BaseModel):
    text: str = Field(..., example="HELLO")
    rails: int = Field(..., example=3)

class SubstituteRequest(BaseModel):
    text: str = Field(..., example="HELLO")
    keyword: str = Field(..., example="QWERTYUIOPASDFGHJKLZXCVBNM")
