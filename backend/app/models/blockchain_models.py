from pydantic import BaseModel, Field
from typing import List, Optional

class AddressRequest(BaseModel):
    address: str = Field(..., example="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")

class AddressResponse(BaseModel):
    Valid: bool
    Type: Optional[str] = None
    Network: Optional[str] = None

class MerkleRequest(BaseModel):
    items: List[str]
    algorithm: str = "sha256"

class MerkleResponse(BaseModel):
    Root: str
    Levels: List[List[str]]

class WIFRequest(BaseModel):
    private_key_hex: str
    compressed: bool = True
    testnet: bool = False
