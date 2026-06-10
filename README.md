# CipherVerse - The Modern Cryptography Toolkit

![Status](https://img.shields.io/badge/Status-Completed-success)
![Frontend](https://img.shields.io/badge/Frontend-React%20%7C%20Vite%20%7C%20Tailwind-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Python-green)

**CipherVerse** is a beautiful, comprehensive, and interactive cryptography toolkit. Designed for educational purposes, CTF challenges, and cryptographic experimentation, it provides a massive array of classical and modern ciphers, encoding schemes, historical machine emulators, and crypto-analysis tools—all wrapped in a sleek, dark-themed cybersecurity aesthetic.

## 🚀 The Architecture

CipherVerse has been entirely rebuilt into a modern web application:

* **Frontend**: Built with React, Vite, Tailwind CSS, and Framer Motion. It features a responsive, unified 45/55 split-pane layout for all tools, interactive visualizations (like Merkle Trees and Rotor Machines), and a premium "hacker" aesthetic.
* **Backend**: Powered by FastAPI (Python), providing a robust, stateless REST API for over 90+ cryptographic functions.

## 🛠️ Features (91+ Cryptographic Tools)

### 🏛️ Classical Ciphers
- **Caesar**, **Vigenère**, **Atbash**, **Bacon**, **Bifid**, **Affine**, **A1Z26**, **Rail Fence**, **Substitution**

### 🔐 Block & Stream Ciphers
- **AES, DES, Triple DES** (Encrypt/Decrypt)
- **Blowfish, SM4** (Encrypt/Decrypt)
- **RC2, RC4, RC4-Drop**
- **CipherSaber2** 
- **XOR** (Cipher / Brute Force), **ROT13 / ROT47**

### 📜 Historic Machines
- **Enigma Machine** (Interactive rotor simulation)
- **Bombe & Multiple Bombe**
- **Typex, Lorenz, Colossus, SIGABA**

### 🔄 Encoding & Decoding
- **Base64, Base32**, **Hexadecimal**, **URL Encoding**, **Binary, ASCII**, **Morse Code**

### 🔑 Public Key Cryptography
- **RSA** (Encrypt/Decrypt/Sign/Verify)
- **DSA & ECDSA**
- **Diffie-Hellman (DH) & ECDH**
- **Ed25519 / EdDSA**, **X25519 Key Exchange**

### 🛡️ Hashing & Analysis
- **MD2, MD4, MD5, MD6**
- **SHA0, SHA1, SHA2 (SHA-256/512), SHA3**
- **SM3, Keccak, Shake**, **RIPEMD, Whirlpool**, **BLAKE2b, BLAKE2s**
- **HMAC, Bcrypt, PBKDF2, Scrypt**

### 🧩 Blockchain & Steganography
- **Bitcoin & Ethereum Address Validation**
- **Merkle Tree Visualization**
- **WIF Checksum Generation**
- **Text, Image, and Audio Steganography**

### 🔍 Forensics & Utilities
- **File Hashing & Integrity Checkers** (MD5/SHA1/SHA256)
- **PE Header Malware Analysis**
- **TLSH Fuzzy Hash Comparison**
- **Password Strength Estimator**
- **JWT Sign/Verify**
- **Certificate Parsing** (X.509, TLS)

## 🌐 Deployment

CipherVerse is fully containerized and deployable to modern cloud platforms:

* **Frontend**: Easily deployed on **Vercel** with automatic SPA routing.
* **Backend**: Deployable on **Render** (or any Docker/FastAPI hosting provider).

## 💻 Local Development

### 1. Start the Backend (FastAPI)

Navigate to the root directory and set up your Python environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server (starts on http://localhost:8000)
uvicorn app.main:app --reload
```

### 2. Start the Frontend (Vite/React)

Navigate to the `frontend` directory:

```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server (starts on http://localhost:5173)
npm run dev
```

Your frontend will automatically communicate with the local backend!

---
*Disclaimer: This toolkit is built exclusively for educational, ethical testing, and CTF purposes. Do not use these tools for illegal activities.*