export interface PageSEO {
  title: string;
  description: string;
  keywords: string[];
  category?: string;
  ogImage?: string;
  faqs?: { question: string; answer: string }[];
}

export const SITE_NAME = 'CipherVerse';
export const SITE_URL = 'https://cipherverse.cybrito.tech';
export const DEFAULT_OG_IMAGE = 'https://cipherverse.cybrito.tech/og-image.png';
export const DEFAULT_KEYWORDS = [
  'cybersecurity platform',
  'cryptography tools',
  'encryption',
  'decryption',
  'classical ciphers',
  'symmetric encryption',
  'asymmetric cryptography',
  'hashing algorithms',
  'steganography',
  'malware analysis',
  'file forensics',
  'blockchain validator',
  'historical cipher machines',
  'online cipher solver',
];

export const seoConfigMap: Record<string, PageSEO> = {
  "/": {
    title: "CipherVerse — Modern Cryptography & Security Suite",
    description: "Explore 40+ interactive online cryptography, malware analysis, file forensics, steganography, and historical cipher tools in one high-performance platform.",
    keywords: [
      "cybersecurity platform",
      "online cryptography",
      "cipher tools",
      "encryption tools",
      "malware analysis online",
      "steganography online"
    ],
    category: "Overview",
  },
  "/classical": {
    title: "Classical Ciphers Hub — Ciphers & Solvers | CipherVerse",
    description: "Interactive online suite for classical ciphers including Caesar, Vigenère, Atbash, Bacon, Bifid, Affine, A1Z26, Rail Fence, and Substitution ciphers.",
    keywords: [
      "classical ciphers",
      "historical ciphers",
      "substitution cipher solver",
      "transposition ciphers",
      "cryptanalysis tools"
    ],
    category: "Classical Ciphers",
  },
  "/classical/caesar": {
    title: "Caesar Cipher Decoder, Encoder & ROT13 Solver | CipherVerse",
    description: "Free online Caesar Cipher encoder, decoder, and brute-force solver. Shift text by any key instantly with detailed letter frequency analysis.",
    keywords: [
      "caesar cipher",
      "shift cipher",
      "rot13 solver",
      "caesar cipher decoder",
      "caesar cipher encoder",
      "caesar brute force"
    ],
    category: "Classical Ciphers",
  },
  "/classical/vigenere": {
    title: "Vigenère Cipher Encoder, Decoder & Key Solver | CipherVerse",
    description: "Encrypt and decrypt messages using the polyalphabetic Vigenère Cipher. Includes automatic keyword analysis and tabular visualization.",
    keywords: [
      "vigenere cipher",
      "polyalphabetic cipher",
      "vigenere decoder",
      "vigenere key solver",
      "vigenere square"
    ],
    category: "Classical Ciphers",
  },
  "/classical/atbash": {
    title: "Atbash Cipher Tool — Reverse Alphabet Solver | CipherVerse",
    description: "Fast online Atbash Cipher tool to substitute alphabet letters in reverse order (A to Z, B to Y) with instant reciprocal encryption and decryption.",
    keywords: [
      "atbash cipher",
      "atbash decoder",
      "atbash encoder",
      "reverse alphabet cipher",
      "hebrew cipher"
    ],
    category: "Classical Ciphers",
  },
  "/classical/bacon": {
    title: "Baconian Cipher Encoder & Steganographic Decoder | CipherVerse",
    description: "Encode and decode hidden messages using Francis Bacon's 5-letter binary steganographic cipher system with 24-letter and 26-letter alphabet options.",
    keywords: [
      "bacon cipher",
      "baconian cipher",
      "binary steganography",
      "bacon decoder",
      "francis bacon cipher"
    ],
    category: "Classical Ciphers",
  },
  "/classical/bifid": {
    title: "Bifid Cipher Tool — Polybius Square Fractionation | CipherVerse",
    description: "Encrypt and decrypt using the Bifid Cipher, combining Polybius square substitution with vertical transposition fractionation.",
    keywords: [
      "bifid cipher",
      "fractionation cipher",
      "polybius square",
      "bifid decoder",
      "bifid solver"
    ],
    category: "Classical Ciphers",
  },
  "/classical/affine": {
    title: "Affine Cipher Solver & Mathematical Decoder | CipherVerse",
    description: "Online Affine Cipher tool to perform mathematical substitution encryption and cryptanalysis using linear modular arithmetic formulas E(x) = (ax + b) mod 26.",
    keywords: [
      "affine cipher",
      "modular arithmetic cipher",
      "affine decoder",
      "coprime key cipher"
    ],
    category: "Classical Ciphers",
  },
  "/classical/a1z26": {
    title: "A1Z26 Cipher & Number Substitution Tool | CipherVerse",
    description: "Convert text to numbers and numbers back to text instantly with the A1Z26 letter-number substitution converter, custom separators, and error handling.",
    keywords: [
      "a1z26 cipher",
      "letter to number cipher",
      "a1z26 decoder",
      "number substitution"
    ],
    category: "Classical Ciphers",
  },
  "/classical/rail-fence": {
    title: "Rail Fence Cipher — Zigzag Transposition | CipherVerse",
    description: "Online Rail Fence Cipher calculator. Encrypt and decrypt messages using multi-rail zigzag transposition paths with interactive rail offset controls.",
    keywords: [
      "rail fence cipher",
      "zigzag cipher",
      "transposition cipher",
      "rail fence decoder"
    ],
    category: "Classical Ciphers",
  },
  "/classical/substitution": {
    title: "Monoalphabetic Substitution Cipher Solver | CipherVerse",
    description: "Solve monoalphabetic substitution ciphers with custom alphabet key mappings, real-time letter frequency analysis, and automated pattern substitution.",
    keywords: [
      "substitution cipher",
      "monoalphabetic cipher",
      "cipher key mapping",
      "letter frequency analysis"
    ],
    category: "Classical Ciphers",
  },
  "/encoding": {
    title: "Online Encoding & Decoding Tools Hub | Base64, Hex, URL, Binary",
    description: "Comprehensive suite of developer encoding and decoding utilities including Base64, Hexadecimal, URL, Binary, and Morse code.",
    keywords: [
      "encoding tools",
      "base64 encoder",
      "hex decoder",
      "url encode decode",
      "binary converter"
    ],
    category: "Encoding & Decoding",
  },
  "/encoding/base64": {
    title: "Base64 Encoder & Decoder Online — UTF-8 & URL-Safe | CipherVerse",
    description: "Instant online Base64 text and binary data encoder and decoder with UTF-8, URL-safe Base64URL, and hex output support for secure data transmission.",
    keywords: [
      "base64 encode",
      "base64 decode",
      "base64 converter",
      "url safe base64"
    ],
    category: "Encoding & Decoding",
  },
  "/encoding/hex": {
    title: "Hexadecimal (Hex) Encoder & Decoder Online | CipherVerse",
    description: "Convert plain text to Hex bytes and decode Hex strings to readable ASCII text instantly with custom byte delimiters, prefix formatting, and validation.",
    keywords: [
      "hex encoder",
      "hex decoder",
      "hex to string",
      "string to hex",
      "hexadecimal converter"
    ],
    category: "Encoding & Decoding",
  },
  "/encoding/url": {
    title: "URL Percent Encoder & Decoder Online | CipherVerse",
    description: "Encode special characters into percent-encoded URI strings and decode encoded query parameters safely with full UTF-8 and component encoding support.",
    keywords: [
      "url encode",
      "url decode",
      "percent encoding",
      "uri component encoder"
    ],
    category: "Encoding & Decoding",
  },
  "/encoding/binary": {
    title: "Text to Binary & Binary to Text Converter | CipherVerse",
    description: "Convert ASCII and UTF-8 text into 8-bit binary numbers (0s and 1s) and decode binary byte streams instantly with customizable spacing and bit formats.",
    keywords: [
      "text to binary",
      "binary decoder",
      "binary to text",
      "8 bit binary converter"
    ],
    category: "Encoding & Decoding",
  },
  "/encoding/morse": {
    title: "Morse Code Translator — Audio & Visual Signals | CipherVerse",
    description: "Translate text into International Morse Code dots and dashes with real-time audio playback, custom WPM speeds, visual flashing, and audio download options.",
    keywords: [
      "morse code translator",
      "morse code decoder",
      "text to morse",
      "morse code audio"
    ],
    category: "Encoding & Decoding",
  },
  "/symmetric": {
    title: "Symmetric Cryptography Tools — AES, DES & Stream | CipherVerse",
    description: "Interactive modern symmetric block and stream cipher toolset featuring AES-GCM/CBC, Triple DES, Blowfish, RC4, SM4, and multi-byte XOR bruteforce analysis.",
    keywords: [
      "symmetric encryption",
      "aes online",
      "des cipher",
      "blowfish encryption",
      "stream ciphers",
      "block ciphers"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/aes": {
    title: "AES Encryption & Decryption Online (128/256-bit) | CipherVerse",
    description: "Secure online Advanced Encryption Standard (AES) calculator supporting CBC, GCM, CTR modes with cryptographic key generation and authentication tags.",
    keywords: [
      "aes encryption online",
      "aes-256 calculator",
      "aes cbc mode",
      "aes gcm online",
      "aes decrypt"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/des": {
    title: "DES (Data Encryption Standard) Online Tool | CipherVerse",
    description: "Explore the Data Encryption Standard (DES) with interactive encryption, decryption, CBC/ECB mode selection, and 16-round subkey schedule visualization.",
    keywords: [
      "des cipher",
      "data encryption standard",
      "des online",
      "des decrypt"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/3des": {
    title: "Triple DES (3DES / TDEA) Encryption Tool | CipherVerse",
    description: "Online Triple DES (3DES/TDEA) calculator supporting 2-key and 3-key EDE modes with custom IVs and padding configurations for legacy cryptographic inspection.",
    keywords: [
      "triple des",
      "3des encryption",
      "tdea cipher",
      "3des online"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/rc2": {
    title: "RC2 Block Cipher Encryption & Decryption | CipherVerse",
    description: "Encrypt and decrypt data using Ron Rivest's RC2 variable key-size block cipher with customizable effective key bit lengths and CBC/ECB mode options.",
    keywords: [
      "rc2 cipher",
      "rc2 encryption",
      "ron rivest cipher",
      "rc2 online"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/rc4": {
    title: "RC4 Stream Cipher Generator & Decrypter | CipherVerse",
    description: "Generate RC4 keystream sequences and encrypt or decrypt arbitrary data online with interactive Key Scheduling (KSA) and PRGA state array inspections.",
    keywords: [
      "rc4 cipher",
      "rc4 online",
      "stream cipher rc4",
      "rc4 keystream"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/rc4-drop": {
    title: "RC4-Drop Stream Cipher & Keystream Tool | CipherVerse",
    description: "Enhanced RC4-Drop implementation discarding initial N keystream bytes to eliminate Fluhrer-Mantin-Shamir (FMS) weak key vulnerabilities and state biases.",
    keywords: [
      "rc4 drop",
      "rc4 drop initial bytes",
      "strengthened rc4"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/blowfish": {
    title: "Blowfish Encryption & Decryption Tool | CipherVerse",
    description: "Encrypt and decrypt data using Bruce Schneier's 64-bit Blowfish block cipher with variable key lengths up to 448 bits, custom IVs, and mode selections.",
    keywords: [
      "blowfish cipher",
      "blowfish encryption online",
      "bruce schneier cipher"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/sm4": {
    title: "SM4 National Standard Block Cipher Tool | CipherVerse",
    description: "Encrypt and decrypt data using the Chinese National Standard SM4 128-bit block cipher (GB/T 32907-2016) with CBC and ECB mode configurations.",
    keywords: [
      "sm4 cipher",
      "sm4 encryption",
      "chinese national standard cipher",
      "sm4 block cipher"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/ciphersaber2": {
    title: "CipherSaber-2 Key Stream Encryption Tool | CipherVerse",
    description: "Encrypt and decrypt using the CipherSaber-2 cryptographic protocol, featuring 10-byte initialization vectors and configurable RC4 state array mixing rounds.",
    keywords: [
      "ciphersaber",
      "ciphersaber-2",
      "rc4 IV mixing"
    ],
    category: "Symmetric Crypto",
  },
  "/symmetric/xor": {
    title: "XOR Cipher & Multi-Byte Key Bruteforce Tool | CipherVerse",
    description: "Encrypt, decrypt, and crack single-byte and multi-byte repeating XOR ciphers with automated Hamming distance key-size detection and English frequency analysis.",
    keywords: [
      "xor cipher",
      "xor solver",
      "xor bruteforce",
      "xor key breaker"
    ],
    category: "Symmetric Crypto",
  },
  "/asymmetric": {
    title: "Asymmetric Cryptography — RSA & DSA Tools | CipherVerse",
    description: "Generate RSA and DSA keypairs, perform public key encryption, compute digital signatures, and verify authentic cryptographic certificates with ease.",
    keywords: [
      "asymmetric cryptography",
      "public key encryption",
      "rsa key generator",
      "dsa signature",
      "digital signatures"
    ],
    category: "Asymmetric Crypto",
  },
  "/asymmetric/rsa": {
    title: "RSA Key Generator, Encryption & Signatures | CipherVerse",
    description: "Online RSA tool to generate PEM keypairs (1024, 2048, 4096 bit), encrypt messages, decrypt ciphertexts, sign, and verify signatures.",
    keywords: [
      "rsa online tool",
      "rsa keypair generator",
      "rsa encrypt decrypt",
      "rsa digital signature"
    ],
    category: "Asymmetric Crypto",
  },
  "/asymmetric/dsa": {
    title: "DSA Key Generator & Digital Signature Algorithm | CipherVerse",
    description: "Generate FIPS 186 Digital Signature Algorithm (DSA) key parameters, sign cryptographic digests, and verify digital signatures against public verification keys.",
    keywords: [
      "dsa signature",
      "digital signature algorithm",
      "dsa key generation",
      "dsa verify"
    ],
    category: "Asymmetric Crypto",
  },
  "/hashing": {
    title: "SHA-256 & Cryptographic Hash Generator | CipherVerse",
    description: "Calculate cryptographic hashes (SHA-256, SHA-512, MD5, SHA-3), HMAC signatures, and run PBKDF2, Scrypt, and bcrypt key derivation functions instantly.",
    keywords: [
      "hash generator online",
      "sha256 calculator",
      "hmac generator",
      "pbkdf2 calculator",
      "scrypt tool"
    ],
    category: "Hashing & KDFs",
  },
  "/certificates": {
    title: "X.509 Certificate Inspector & TLS Analyzer | CipherVerse",
    description: "Parse X.509 PEM/DER certificates, inspect SAN extensions, verify issuer trust chains, test TLS connections, and compute SSL fingerprints in your browser.",
    keywords: [
      "x509 parser",
      "ssl certificate inspector",
      "tls handshake analyzer",
      "certificate fingerprint"
    ],
    category: "Certificates & TLS",
  },
  "/certificates/x509": {
    title: "X.509 SSL/TLS Certificate Parser Online | CipherVerse",
    description: "Parse X.509 PEM and DER certificates to inspect subject, issuer, serial numbers, validity windows, SAN extensions, and extract public key parameters instantly.",
    keywords: [
      "x509 certificate parser",
      "ssl cert viewer",
      "pem parser online",
      "san inspector"
    ],
    category: "Certificates & TLS",
  },
  "/certificates/tls": {
    title: "TLS Protocol & Cipher Suite Inspection Tool | CipherVerse",
    description: "Analyze SSL/TLS handshake sequences, cipher suite negotiations, protocol versions (TLS 1.2/1.3), and forward secrecy parameters for secure web connections.",
    keywords: [
      "tls inspector",
      "ssl handshake tool",
      "cipher suite checker"
    ],
    category: "Certificates & TLS",
  },
  "/certificates/fingerprint": {
    title: "SSL Certificate Fingerprint Calculator | CipherVerse",
    description: "Compute SHA-1, SHA-256, and MD5 thumbprints for X.509 SSL/TLS certificates instantly to verify host identity and validate pin headers in your application.",
    keywords: [
      "certificate fingerprint",
      "sha256 cert fingerprint",
      "ssl thumbprint calculator"
    ],
    category: "Certificates & TLS",
  },
  "/blockchain": {
    title: "Blockchain Suite — Bitcoin, ETH & Merkle | CipherVerse",
    description: "Validate Bitcoin and Ethereum public addresses, construct cryptographic Merkle tree proofs, and encode private keys into Wallet Import Format (WIF).",
    keywords: [
      "blockchain tools",
      "bitcoin address validator",
      "ethereum address checker",
      "merkle tree generator",
      "wif encoder"
    ],
    category: "Blockchain",
  },
  "/blockchain/bitcoin": {
    title: "Bitcoin Address Validator & Script Inspector | CipherVerse",
    description: "Validate Bitcoin legacy (P2PKH), SegWit (P2SH), and native Bech32/Taproot addresses with Base58Check and BCH error-detecting checksum verifications online.",
    keywords: [
      "bitcoin address validator",
      "bech32 validator",
      "segwit address checker",
      "btc checksum"
    ],
    category: "Blockchain",
  },
  "/blockchain/ethereum": {
    title: "Ethereum Address Validator & EIP-55 Tool | CipherVerse",
    description: "Verify Ethereum 0x hex wallet addresses and test EIP-55 mixed-case Keccak-256 checksum validity online to prevent costly mistyped transaction errors.",
    keywords: [
      "ethereum address validator",
      "eip55 checker",
      "eth 0x address checksum"
    ],
    category: "Blockchain",
  },
  "/blockchain/merkle": {
    title: "Merkle Tree Construction & Visual Proof Generator | CipherVerse",
    description: "Build interactive Merkle Trees from transaction hashes, calculate the cryptographic root hash, and verify audit path inclusion proofs with visual graphs.",
    keywords: [
      "merkle tree generator",
      "merkle root calculator",
      "merkle proof verifier",
      "blockchain merkle tree"
    ],
    category: "Blockchain",
  },
  "/blockchain/wif": {
    title: "Wallet Import Format (WIF) Private Key Encoder | CipherVerse",
    description: "Encode and decode Bitcoin 256-bit ECDSA private keys into compressed and uncompressed Wallet Import Format (WIF) with Base58Check checksum validation.",
    keywords: [
      "wif encoder",
      "wallet import format",
      "bitcoin private key wif",
      "wif decoder"
    ],
    category: "Blockchain",
  },
  "/steganography": {
    title: "Online Steganography — Text, Image & Audio | CipherVerse",
    description: "Conceal hidden payload messages inside digital images (LSB), audio WAV waveforms, and zero-width unicode text with browser-based cryptographic privacy.",
    keywords: [
      "steganography online",
      "image steganography",
      "hide text in image",
      "audio steganography",
      "zero width steganography"
    ],
    category: "Steganography",
  },
  "/steganography/text": {
    title: "Zero-Width Text Steganography Encoder & Decoder | CipherVerse",
    description: "Conceal hidden secret messages inside ordinary plain text using zero-width invisible unicode characters (ZWSP, ZWNJ) undetectable to standard text viewers.",
    keywords: [
      "text steganography",
      "zero width space hide text",
      "invisible text encoder",
      "stego text"
    ],
    category: "Steganography",
  },
  "/steganography/image": {
    title: "Image Steganography Tool — LSB Secret Text Hiding | CipherVerse",
    description: "Embed and extract hidden encrypted messages inside PNG and JPEG image pixels using Least Significant Bit (LSB) encoding with AES encryption and passphrases.",
    keywords: [
      "image steganography online",
      "lsb steganography",
      "hide secret text in photo",
      "steganography decoder online"
    ],
    category: "Steganography",
  },
  "/steganography/audio": {
    title: "Audio Steganography Tool — LSB Waveform Hiding | CipherVerse",
    description: "Embed hidden secret messages into uncompressed WAV audio sample data using Least Significant Bit (LSB) encoding while preserving acoustic fidelity.",
    keywords: [
      "audio steganography",
      "wav steganography",
      "hide message in sound file"
    ],
    category: "Steganography",
  },
  "/malware-analysis": {
    title: "Malware Analysis Toolkit — Hashes & PE | CipherVerse",
    description: "Online malware triage platform featuring multi-hash signature lookup, Trend Micro Locality Sensitive Hashing (TLSH), and Windows PE binary header parsing.",
    keywords: [
      "malware analysis online",
      "tlsh similarity",
      "pe header parser",
      "file hash lookup",
      "malware triage"
    ],
    category: "Malware Analysis",
  },
  "/malware-analysis/hash": {
    title: "Malware File Hash Lookup & Threat Triage | CipherVerse",
    description: "Generate and query MD5, SHA-1, and SHA-256 file hashes against threat intelligence databases to quickly triage malware samples and detect malicious binaries.",
    keywords: [
      "malware hash lookup",
      "sha256 threat intelligence",
      "virustotal hash lookup"
    ],
    category: "Malware Analysis",
  },
  "/malware-analysis/tlsh": {
    title: "TLSH Fuzzy Hash & Malware Comparator | CipherVerse",
    description: "Calculate Trend Micro Locality Sensitive Hashes (TLSH) and compute distance scores to detect polymorphic malware family variants and binary similarities.",
    keywords: [
      "tlsh comparison",
      "locality sensitive hashing",
      "fuzzy hashing malware",
      "tlsh distance score"
    ],
    category: "Malware Analysis",
  },
  "/malware-analysis/pe": {
    title: "PE Header Parser & EXE Section Inspector | CipherVerse",
    description: "Parse Windows PE32/PE32+ executable headers, section tables (.text, .data, .rsrc), import/export directory tables, and compile timestamps in your browser.",
    keywords: [
      "pe header parser online",
      "exe section analyzer",
      "windows pe inspection",
      "import table parser"
    ],
    category: "Malware Analysis",
  },
  "/file-forensics": {
    title: "File Forensics Suite — Entropy & Hashes | CipherVerse",
    description: "Perform digital forensic file analysis: calculate Shannon entropy curves, compute simultaneous file hashes, and evaluate PRNG randomness without server uploads.",
    keywords: [
      "file forensics online",
      "shannon entropy calculator",
      "file randomness test",
      "digital forensics tools"
    ],
    category: "File Forensics",
  },
  "/file-forensics/hash": {
    title: "Single File Hash Generator & Verifier | CipherVerse",
    description: "Calculate cryptographic digests (MD5, SHA-1, SHA-256) for local files directly in your browser using Web Crypto API without uploading any bytes to a server.",
    keywords: [
      "file hash generator",
      "sha256 file hash",
      "md5 file check"
    ],
    category: "File Forensics",
  },
  "/file-forensics/multi-hash": {
    title: "Multi-Algorithm Concurrent File Hasher | CipherVerse",
    description: "Compute MD5, SHA-1, SHA-256, SHA-512, and RIPEMD-160 checksums simultaneously in a single stream pass directly in your browser for rapid digital forensics.",
    keywords: [
      "multi file hash",
      "simultaneous hashing",
      "file integrity check"
    ],
    category: "File Forensics",
  },
  "/file-forensics/entropy": {
    title: "Shannon File Entropy Analyzer & Visualization | CipherVerse",
    description: "Calculate Shannon entropy distributions across binary files to detect packed executables, compressed sections, or hidden encrypted payloads with visual graphs.",
    keywords: [
      "file entropy calculator",
      "shannon entropy",
      "detect encrypted file",
      "packed exe detection"
    ],
    category: "File Forensics",
  },
  "/file-forensics/randomness": {
    title: "Cryptographic Randomness Test Suite | CipherVerse",
    description: "Evaluate random number generator outputs with NIST-inspired Chi-square tests, byte distribution histograms, and serial correlation tests for high entropy.",
    keywords: [
      "randomness test online",
      "chi square test",
      "prng quality test",
      "entropy evaluation"
    ],
    category: "File Forensics",
  },
  "/utilities": {
    title: "Cybersecurity Utilities — Password, JWT | CipherVerse",
    description: "Developer security utilities including password strength analyzer, JWT signature generator, cryptographically secure salt generator, and checksum tools.",
    keywords: [
      "security utilities",
      "password strength tester",
      "jwt sign tool",
      "crypto salt generator"
    ],
    category: "Utilities",
  },
  "/utilities/password": {
    title: "Password Strength & Entropy Calculator | CipherVerse",
    description: "Evaluate password entropy (bits), estimate brute-force cracking time, and detect common dictionary vulnerability patterns.",
    keywords: [
      "password strength analyzer",
      "password entropy calculator",
      "time to crack password"
    ],
    category: "Utilities",
  },
  "/utilities/jwt": {
    title: "JWT (JSON Web Token) HMAC Signer & Verifier | CipherVerse",
    description: "Sign and verify JSON Web Tokens (JWT) online using HS256, HS384, or HS512 HMAC secret keys with real-time decoded header and payload JSON inspections.",
    keywords: [
      "jwt signer online",
      "jwt hmac sign",
      "jwt secret tester",
      "jwt generator"
    ],
    category: "Utilities",
  },
  "/utilities/salt": {
    title: "Cryptographic Salt & Token Generator | CipherVerse",
    description: "Generate high-entropy cryptographically secure random salts, IVs, and session tokens in Hex, Base64, and C-array formats using browser Web Crypto CSPRNG.",
    keywords: [
      "crypto salt generator",
      "random salt generator",
      "secure random byte token"
    ],
    category: "Utilities",
  },
  "/utilities/fletcher16": {
    title: "Fletcher-16 Checksum & Integrity Tool | CipherVerse",
    description: "Calculate 16-bit Fletcher checksums online with modular block arithmetic to verify data integrity and detect transmission errors in digital data.",
    keywords: [
      "fletcher 16 checksum",
      "fletcher checksum calculator",
      "data integrity checksum"
    ],
    category: "Utilities",
  },
  "/historical": {
    title: "Historical Cipher Machines — Enigma & Bombe | CipherVerse",
    description: "Experience WWII cryptographic history with accurate software simulators of the German Enigma machine, Turing Bombe, and British Typex cipher machine.",
    keywords: [
      "historical cipher machines",
      "enigma machine simulator",
      "turing bombe simulator",
      "typex cipher machine"
    ],
    category: "Historical Machines",
  },
  "/historical/enigma": {
    title: "Enigma Machine Simulator (I, M3, M4) Online | CipherVerse",
    description: "Authentic 3-rotor and 4-rotor Enigma machine simulator. Configure rotors, ring settings, reflector, and plugboard (Steckerbrett).",
    keywords: [
      "enigma machine simulator",
      "online enigma machine",
      "ww2 enigma cipher",
      "steckerbrett simulator"
    ],
    category: "Historical Machines",
  },
  "/historical/bombe": {
    title: "Alan Turing Bombe Crib Analysis Simulator | CipherVerse",
    description: "Interactive simulation of Alan Turing's Bombe electromechanical machine used at Bletchley Park to break Enigma rotor keys.",
    keywords: [
      "turing bombe simulator",
      "bletchley park bombe",
      "enigma crib solver"
    ],
    category: "Historical Machines",
  },
  "/historical/typex": {
    title: "British Typex Cipher Machine Simulator | CipherVerse",
    description: "Simulate the British WWII Typex cipher machine with authentic multi-rotor stepping mechanisms, plugboard options, and statutory transposition decoding.",
    keywords: [
      "typex cipher machine",
      "british enigma",
      "typex simulator"
    ],
    category: "Historical Machines",
  },
  "/api-explorer": {
    title: "API Explorer — Cryptographic REST Endpoints | CipherVerse",
    description: "Explore and test CipherVerse REST API endpoints with interactive JSON request builders, authentication headers, and automated cryptographic response payloads.",
    keywords: [
      "cryptography api",
      "cipher rest api",
      "api explorer",
      "security api endpoints"
    ],
    category: "Developer",
  },
  "/settings": {
    title: "Platform Settings & Preferences | CipherVerse",
    description: "Customize CipherVerse theme appearance, default cryptographic algorithm preferences, and local browser workspace configurations securely.",
    keywords: [
      "cipherverse settings",
      "preferences"
    ],
    category: "System",
  },
  "/404": {
    title: "404 - Page Not Found | CipherVerse",
    description: "The requested cryptographic or cybersecurity tool page could not be found on CipherVerse. Search our directory of 40+ free security tools.",
    keywords: [
      "404",
      "page not found"
    ],
    category: "System",
  },
};

/**
 * Retrieves the SEO configuration for a given path.
 * If exact match fails, fallback logic auto-generates structured title and meta description.
 */
export function getSEOConfig(pathname: string): PageSEO {
  // Normalize path (remove trailing slash except for root)
  const cleanPath = pathname.length > 1 && pathname.endsWith('/') ? pathname.slice(0, -1) : pathname;

  if (seoConfigMap[cleanPath]) {
    return seoConfigMap[cleanPath];
  }

  // Handle dynamic /encoding/:tool route
  if (cleanPath.startsWith('/encoding/')) {
    const toolName = cleanPath.replace('/encoding/', '');
    const capitalized = toolName.charAt(0).toUpperCase() + toolName.slice(1);
    return {
      title: `${capitalized} Encoder & Decoder Online | CipherVerse`,
      description: `Free online ${capitalized} encoding and decoding tool. Convert data quickly with real-time output and formatting options.`,
      keywords: [`${toolName} encoder`, `${toolName} decoder`, `${toolName} converter`, 'online encoding'],
      category: 'Encoding & Decoding',
    };
  }

  // Fallback default SEO config for unknown paths
  return {
    title: 'CipherVerse — Modern Cryptography & Security Suite',
    description: 'Interactive online suite for ciphers, hashing, malware analysis, file forensics, steganography, and historical cryptographic machines.',
    keywords: DEFAULT_KEYWORDS,
    category: 'Cybersecurity',
  };
}
