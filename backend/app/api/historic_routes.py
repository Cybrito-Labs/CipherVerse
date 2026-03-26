from fastapi import APIRouter
from app.core import historic
from app.models.historic_models import (
    EnigmaRequest,
    BombeRequest,
    TypexRequest,
    HistoricResponse
)

router = APIRouter(
    prefix="/historic",
    tags=["Historical Machines"]
)

@router.post("/enigma", response_model=HistoricResponse)
def enigma_route(request: EnigmaRequest):
    result = historic.enigma_cipher(request.text, request.rotor_order, request.rotor_positions)
    return {"result": result}

@router.post("/bombe", response_model=HistoricResponse)
def bombe_route(request: BombeRequest):
    matches = historic.bombe_find_settings(request.ciphertext, request.crib, request.rotor_order)
    return {"result": "Search complete", "matches": matches}

@router.post("/typex", response_model=HistoricResponse)
def typex_route(request: TypexRequest):
    result = historic.typex_encrypt(request.text, request.rotors, tuple(request.positions))
    return {"result": result}
