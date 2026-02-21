# CipherVerse - Cryptography Toolkit

![Status](https://img.shields.io/badge/Status-Backend%20Completed-success)
![Python](https://img.shields.io/badge/Python-3.x-blue)

**CipherVerse** is a comprehensive cryptography toolkit designed for educational purposes, CTF challenges, and cryptographic experimentation. It lists a wide array of classical and modern ciphers, encoding schemes, historical machine emulators, and crypto-analysis tools.

> **Note:** The backend logic is currently fully implemented in `CipherVerse_backend.py`.

## Features

The toolkit includes **91+** cryptographic functions across various categories:

### 🏛️ Classical Ciphers
- **Caesar Cipher** (Encode/Decode)
- **Vigenère Cipher** (Encode/Decode)
- **Atbash Cipher**
- **Bacon Cipher** (Encode/Decode)
- **Bifid Cipher** (Encode/Decode)
- **Affine Cipher** (Encode/Decode)
- **A1Z26 Cipher** (Encode/Decode)
- **Rail Fence Cipher** (Encode/Decode)
- **Substitution Cipher**

### 🔐 Block & Stream Ciphers
- **XOR** (Cipher / Brute Force)
- **ROT13 / ROT47**
- **CipherSaber2** (Encrypt/Decrypt)
- **RC2, RC4** (Encrypt/Decrypt)
- **AES, DES, Triple DES** (Encrypt/Decrypt)
- **Blowfish, SM4** (Encrypt/Decrypt)

### 📜 Historic Machines
- **Enigma**
- **Bombe**
- **Multiple Bombe**
- **Typex**
- **Lorenz**
- **Colossus**
- **SIGABA**

### 🔄 Encoding / Decoding
- **Base64, Base32**
- **Hexadecimal**
- **URL Encoding**
- **Binary, ASCII**
- **Morse Code**

### 🔑 Public Key Cryptography
- **RSA** (Encrypt/Decrypt/Sign/Verify)
- **Diffie-Hellman (DH)** & **ECDH**
- **DSA** & **ECDSA**
- **Ed25519 / EdDSA**
- **X25519 Key Exchange**

### 🛡️ Hashing & Analysis
- **MD2, MD4, MD5, MD6**
- **SHA0, SHA1, SHA2, SHA3**
- **SM3, Keccak, Shake**
- **RIPEMD, Whirlpool**
- **BLAKE2b, BLAKE2s**
- **HMAC, Bcrypt**

### 🛠️ Utilities & Forensics
- **File Hashing & Integrity Checkers**
- **Password Strength Estimator**
- **JWT Sign/Verify**
- **Certificate Parsing (X.509, TLS)**
- **Blockchain Address Validators (Bitcoin, Ethereum)**
- **Steganography (Text, Image, Audio)**

## Installation

Ensure you have Python installed. You will need to install a few dependencies for full functionality:

```bash
pip install pycryptodome cryptography pysha3 pillow pefile py-tlsh ssdeep bcrypt
```

*Note: Some modules like `ssdeep` or `py-tlsh` may require system-level libraries.*

## Usage

Run the backend script directly to access the interactive menu:

```bash
python CipherVerse/CipherVerse_backend.py
```

Follow the on-screen prompts to select a category and a specific tool.

## Project Structure

- `CipherVerse_backend.py`: Main script containing all cryptographic implementations and the interactive menu interface.

---
*Disclaimer: This tool is for educational and testing purposes only. Do not use for illegal activities.*
