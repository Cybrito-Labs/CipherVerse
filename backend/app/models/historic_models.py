from pydantic import BaseModel, Field
from typing import List, Tuple

class EnigmaRequest(BaseModel):
    text: str
    rotor_order: Tuple[str, str, str] = ("I", "II", "III")
    rotor_positions: Tuple[int, int, int] = (0, 0, 0)

class BombeRequest(BaseModel):
    ciphertext: str
    crib: str
    rotor_order: Tuple[str, str, str] = ("I", "II", "III")

class TypexRequest(BaseModel):
    text: str
    rotors: int = 5
    positions: List[int] = [0, 0, 0, 0, 0]

class HistoricResponse(BaseModel):
    result: str
    matches: List[str] = []
