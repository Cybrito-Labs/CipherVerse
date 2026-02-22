from fastapi import FastAPI
from app.core.classical import caesar_cipher
app = FastAPI()

@app.get("/")
def root():
    return {"message": "CipherVerse backend running"}



print(caesar_cipher("HELLO", 3))