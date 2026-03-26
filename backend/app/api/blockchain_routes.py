from fastapi import APIRouter
from app.core import blockchain
from app.models.blockchain_models import (
    AddressRequest,
    AddressResponse,
    MerkleRequest,
    MerkleResponse,
    WIFRequest
)

router = APIRouter(
    prefix="/blockchain",
    tags=["Blockchain Tools"]
)

@router.post("/bitcoin/validate", response_model=AddressResponse)
def btc_validate_route(request: AddressRequest):
    return blockchain.validate_bitcoin_address(request.address)

@router.post("/ethereum/validate", response_model=AddressResponse)
def eth_validate_route(request: AddressRequest):
    return blockchain.validate_ethereum_address(request.address)

@router.post("/merkle", response_model=MerkleResponse)
def merkle_route(request: MerkleRequest):
    return blockchain.build_merkle_tree(request.items, request.algorithm)

@router.post("/wif/encode", response_model=dict)
def wif_encode_route(request: WIFRequest):
    return {"wif": blockchain.wif_encode(request.private_key_hex, request.compressed, request.testnet)}
