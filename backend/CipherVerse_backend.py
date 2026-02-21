# CIPHERVERSE - Crypto Toolkit
import hashlib , base64
def menu():
    print("="*50)
    print("     CIPHERVERSE - Crypto Toolkit")
    print("="*50)

    # Classical Ciphers
    print("\n[CLASSICAL CIPHERS]")
    print("1. Caesar Cipher (Encode/Decode)") #DONE
    print("2. Vigenère Cipher (Encode/Decode)") #DONE
    print("3. Atbash Cipher") #DONE
    print("4. Bacon Cipher (Encode/Decode)") #DONE
    print("5. Bifid Cipher (Encode/Decode)") #DONE
    print("6. Affine Cipher (Encode/Decode)") #DONE
    print("7. A1Z26 Cipher (Encode/Decode)") #DONE
    print("8. Rail Fence Cipher (Encode/Decode)") #DONE
    print("9. Substitute Cipher") #DONE

    # Modern and Block Ciphers
    print("\n[BLOCK & STREAM CIPHERS]")
    print("10. XOR / XOR Brute Force") #DONE
    print("11. ROT13 / ROT47") #DONE
    print("12. CipherSaber2 (Encrypt/Decrypt)") #DONE
    print("13. RC2 (Encrypt/Decrypt)") #pip install pycryptodome #DONE
    print("14. RC4 / RC4 Drop")  #DONE
    print("15. AES (Encrypt/Decrypt)") #DONE
    print("16. DES (Encrypt/Decrypt)") #DONE
    print("17. Triple DES (Encrypt/Decrypt)") #DONE
    print("18. Blowfish (Encrypt/Decrypt)") #DONE
    print("19. SM4 (Encrypt/Decrypt)") #DONE

    # Emulated / Historic Machines
    print("\n[HISTORIC MACHINES]") 
    print("20. Enigma") #DONE
    print("21. Bombe") #DONE
    print("22. Multiple Bombe") #DONE
    print("23. Typex") #DONE
    print("24. Lorenz") #DONE
    print("25. Colossus") 
    print("26. SIGABA")

    # Encoding/Decoding
    print("\n[ENCODING/DECODING]")
    print("27. Base64 (Encode/Decode)")
    print("28. Hexadecimal (Encode/Decode)")
    print("29. URL (Encode/Decode)")
    print("30. Binary (Encode/Decode)")
    print("31. ASCII (Encode/Decode)")
    print("32. Base32 (Encode/Decode)")
    print("33. Morse Code (To/From)")

    # Hashing & Analysis
    print("\n[HASHING & ANALYSIS]")
    print("34. Analyze Hash")
    print("35. Generate All Hashes")
    print("36. MD2 / MD4 / MD5 / MD6") #MD4 MD5 DONE
    print("37. SHA0 / SHA1 / SHA2 / SHA3")
    print("38. SM3 / Keccak / Shake")
    print("39. RIPEMD / HAS-160 / Whirlpool / Snefru")
    print("40. BLAKE2b / BLAKE2s") 
    print("41. GOST / Streebog") #fail------------------------pip install -U pycryptodome
    print("42. SSDEEP / CTPH / Compare SSDEEP or CTPH hashes") #fail--------pip install ssdeep

    # Passwords and HMACs
    print("\n[PASSWORDS & MACS]")
    print("43. HMAC")
    print("44. Bcrypt (Hash/Compare/Parse)")  #fail-----------pip install bcrypt

    # Key Derivation & Random
    print("\n[KEYS & RANDOMNESS]")
    print("46. Derive PBKDF2 key")
    print("47. Derive EVP key")
    print("48. Pseudo-Random Number Generator")

    # JWT and Citrix
    print("\n[JWT & CITRIX]")
    print("49. JWT Sign / Verify / Decode")
    print("50. Citrix CTX1 (Encode/Decode)")

    # Checksums and Validators
    print("\n[CHECKSUMS & VALIDATORS]")
    print("51. Fletcher-8/16/32/64 Checksum")
    print("52. Adler-32 Checksum")
    print("53. Luhn Checksum")
    print("54. CRC-8/16/32 Checksum")
    print("55. TCP/IP Checksum")

        # Public Key Cryptography
    print("\n[PUBLIC KEY CRYPTOGRAPHY]")#DONE
    print("56. RSA (Encrypt/Decrypt)")#DONE
    print("57. RSA (Sign/Verify)") #pip install pycryptodome  #DONE
    print("58. Diffie-Hellman (DH)") #pip install pycryptodome   #DONE
    print("59. Elliptic Curve Diffie-Hellman (ECDH)") #pip install pycryptodome  #DONE
    print("60. DSA (Sign/Verify)") #pip install pycryptodome  #DONE
    print("61. ECDSA (Sign/Verify)")   #DONE #pip install pycryptodome
    print("62. Ed25519 / EdDSA")   #DONE #pip install pycryptodome
    print("63. X25519 Key Exchange")   #DONE #pip install pycryptodome

    # Certificates & TLS
    print("\n[CERTIFICATES & TLS]")
    print("64. X.509 Certificate Parser")#pip install cryptography  #DONE
    print("65. TLS Certificate Analyzer")#pip install cryptography  #DONE
    print("66. PEM <-> DER Converter")#pip install cryptography  #DONE
    print("67. Public Key Extractor")
    print("68. Fingerprint Generator")

    # File & Forensics Tools
    print("\n[FILE & FORENSICS]")
    print("69. File Hashing") #DONE
    print("70. Directory Hashing") #DONE
    print("71. Compare File Hashes") #DONE
    print("72. File Encryption / Decryption") #DONE
    print("73. File Integrity Checker") #DONE
    print("74. Entropy Analyzer")
    print("75. Randomness Test Suite")
    print("76. Key Strength Analyzer")


    # Malware & Fuzzy Analysis
    print("\n[MALWARE & FUZZY ANALYSIS]")
    print("77. TLSH Fuzzy Hashing")#pip install py-tlsh,pip install tlsh
    print("78. Imphash Generator")
    print("79. PE Hash Analyzer")#pip install pefile

    # Blockchain & Cryptocurrency
    print("\n[BLOCKCHAIN & CRYPTOCURRENCY]")
    print("80. Bitcoin Address Validator")
    print("81. Ethereum Address Validator")
    print("82. Keccak-256 (Ethereum)")#pip install pysha3
    print("83. Merkle Tree Generator")
    print("84. Wallet Import Format (WIF)")

    # Steganography
    print("\n[STEGANOGRAPHY]")
    print("85. Text Steganography")
    print("86. Image LSB Steganography")#pip install pillow
    print("87. Audio Steganography")

    # Utilities
    print("\n[UTILITIES]")
    print("88. Password Strength Estimator")
    print("89. Salt Generator")
    print("90. Nonce Generator")
    print("91. Diceware Password Generator")

    print("\n[UTILITIES]")
    print("99. Process ALL")
    print("999. Return to Main Menu")
    print("0. Exit")

    print("="*50)



# base64 decoder
def base64_decoder(data):
    import base64
    return base64.b64decode(data).decode('utf-8')
#base64 encoder
def base64_encoder(data):
    import base64
    return base64.b64encode(data.encode('utf-8')).decode('utf-8')
#HEX encoder
def hex_encoder(data):
    return data.encode().hex()
#HEX decoder
def hex_decoder(data):
    return bytes.fromhex(data).decode()
#URL encoder
def url_encoder(data):
    import urllib.parse
    return urllib.parse.quote(data)
#URL decoder
def url_decoder(data):
    import urllib.parse
    return urllib.parse.unquote(data)
#caesar/ROT13 encrypter and decrypter
def CAESAR_encrypt_and_decrypt(data,shift):
    result=''
    for char in data:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base + shift) % 26 + base)
        else:
            result += char
    return result
#vigenere encoder
def vigenere_encoder(keyword, data):
    result = []
    keyword = keyword.lower()
    k_len = len(keyword)
    for i, char in enumerate(data):
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shift = ord(keyword[i % k_len]) - ord('a')
            result.append(chr((ord(char) - base + shift) % 26 + base))
        else:
            result.append(char)
    return ''.join(result)
#vigenere decoder
def vigenere_decoder(data, keyword):
    result = []
    keyword = keyword.lower()
    k_len = len(keyword)
    for i, char in enumerate(data):
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shift = ord(keyword[i % k_len]) - ord('a')
            result.append(chr((ord(char) - base - shift) % 26 + base))
        else:
            result.append(char)
    return ''.join(result)
#Atbash cipher
def atbash_cipher(data):
    result = ""
    for char in data:
        if char.isupper():
            result += chr(ord('Z') - (ord(char) - ord('A')))
        elif char.islower():
            result += chr(ord('z') - (ord(char) - ord('a')))
        else:
            result += char
    return result
#MD5 encoder
def MD5_encoder(data):
    import hashlib
    return hashlib.md5(data.encode()).hexdigest()
    #MD5 decoder
def MD5_decoder(data):
    import hashlib
    # MD5 is a one-way hash function, so it cannot be decoded back to the original text.
    # However, you can check if a given MD5 hash matches a known value.
#SHA1 encoder
def SHA1_encoder(data):
    import hashlib
    return hashlib.sha1(data.encode()).hexdigest()
#SHA1 decoder
def SHA1_decoder():
    import hashlib
    # SHA1 is also a one-way hash function, so it cannot be decoded back to the original text.
    # However, you can check if a given SHA1 hash matches a known value.
def SHA256_encoder(data):
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()
def binary_encoder(data):
    return ' '.join(format(ord(char), '08b') for char in data)
def binary_decoder(data):
    return ''.join(chr(int(byte, 2)) for byte in data.split())
def ascii_encoder(data):
    return ' '.join(str(ord(char)) for char in data)
def ascii_decoder(data):
    return ''.join(chr(int(num)) for num in data.split())
def base32_encoder(data):
    import base64
    return base64.b32encode(data.encode()).decode()
def base32_decoder(data):
    import base64
    return base64.b32decode(data).decode()  
def morse_encoder(data):
    MORSE_CODE_DICT = { 'A':'.-', 'B':'-...', 'C':'-.-.', 'D':'-..', 'E':'.', 'F':'..-.', 'G':'--.', 'H':'....', 'I':'..', 'J':'.---', 'K':'-.-', 'L':'.-..', 'M':'--', 'N':'-.', 'O':'---', 'P':'.--.', 'Q':'--.-', 'R':'.-.', 'S':'...', 'T':'-', 'U':'..-', 'V':'...-', 'W':'.--', 'X':'-..-', 'Y':'-.--', 'Z':'--..', '1':'.----', '2':'..---', '3':'...--', '4':'....-', '5':'.....', '6':'-....', '7':'--...', '8':'---..', '9':'----.', '0':'-----' }
    REVERSE_MORSE_CODE_DICT = {v:k for k,v in MORSE_CODE_DICT.items()}
    return ' '.join(MORSE_CODE_DICT.get(char.upper(), '') for char in data)
def morse_decoder(data):
    MORSE_CODE_DICT = { 'A':'.-', 'B':'-...', 'C':'-.-.', 'D':'-..', 'E':'.', 'F':'..-.', 'G':'--.', 'H':'....', 'I':'..', 'J':'.---', 'K':'-.-', 'L':'.-..', 'M':'--', 'N':'-.', 'O':'---', 'P':'.--.', 'Q':'--.-', 'R':'.-.', 'S':'...', 'T':'-', 'U':'..-', 'V':'...-', 'W':'.--', 'X':'-..-', 'Y':'-.--', 'Z':'--..', '1':'.----', '2':'..---', '3':'...--', '4':'....-', '5':'.....', '6':'-....', '7':'--...', '8':'---..', '9':'----.', '0':'-----' }
    REVERSE_MORSE_CODE_DICT = {v:k for k,v in MORSE_CODE_DICT.items()}
    return ''.join(REVERSE_MORSE_CODE_DICT.get(code, '') for code in data.split())
#which encoder or decoder options
def becon_encoder(data):
    BACON_TABLE = {
        'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA',
        'F': 'AABAB', 'G': 'AABBA', 'H': 'AABBB', 'I': 'ABAAA', 'J': 'ABAAB',
        'K': 'ABABA', 'L': 'ABABB', 'M': 'ABBAA', 'N': 'ABBAB', 'O': 'ABBBA',
        'P': 'ABBBB', 'Q': 'BAAAA', 'R': 'BAAAB', 'S': 'BAABA', 'T': 'BAABB',
        'U': 'BABAA', 'V': 'BABAB', 'W': 'BABBA', 'X': 'BABBB', 'Y': 'BBAAA',
        'Z': 'BBAAB'
    }

    # Reverse table for decoding
    REVERSE_BACON_TABLE = {v: k for k, v in BACON_TABLE.items()}
    result = []
    for char in data.upper():
        if char.isalpha():
            result.append(BACON_TABLE[char])
    return " ".join(result)
def becon_decoder(data):
    BACON_TABLE = {
        'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA',
        'F': 'AABAB', 'G': 'AABBA', 'H': 'AABBB', 'I': 'ABAAA', 'J': 'ABAAB',
        'K': 'ABABA', 'L': 'ABABB', 'M': 'ABBAA', 'N': 'ABBAB', 'O': 'ABBBA',
        'P': 'ABBBB', 'Q': 'BAAAA', 'R': 'BAAAB', 'S': 'BAABA', 'T': 'BAABB',
        'U': 'BABAA', 'V': 'BABAB', 'W': 'BABBA', 'X': 'BABBB', 'Y': 'BBAAA',
        'Z': 'BBAAB'
    }

    # Reverse table for decoding
    REVERSE_BACON_TABLE = {v: k for k, v in BACON_TABLE.items()}
    # Remove spaces and split every 5 chars
    data = data.replace(" ", "").upper()
    result = []
    for i in range(0, len(data), 5):
        chunk = data[i:i+5]
        if len(chunk) == 5 and chunk in REVERSE_BACON_TABLE:
            result.append(REVERSE_BACON_TABLE[chunk])
        else:
            result.append('?')  # unknown/invalid pattern
    return "".join(result)
def generate_polybius_square(key=""):
    # Bifid Cipher (Classic, 5x5, I/J combined)

    import string

    ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # J removed

    key = key.upper().replace("J", "I")
    seen = set()
    square = []

    for c in key + ALPHABET:
        if c.isalpha() and c not in seen:
            seen.add(c)
            square.append(c)

    # Build lookup tables
    char_to_pos = {}
    pos_to_char = {}

    idx = 0
    for r in range(1, 6):
        for c in range(1, 6):
            char_to_pos[square[idx]] = (r, c)
            pos_to_char[(r, c)] = square[idx]
            idx += 1

    return char_to_pos, pos_to_char
def bifid_encrypt(plaintext: str, key="") -> str:

    char_to_pos, pos_to_char = generate_polybius_square(key)

    plaintext = plaintext.upper().replace("J", "I")
    plaintext = "".join(c for c in plaintext if c.isalpha())

    rows = []
    cols = []

    # Step 1: get coordinates
    for ch in plaintext:
        r, c = char_to_pos[ch]
        rows.append(r)
        cols.append(c)

    # Step 2: concatenate rows + cols
    merged = rows + cols

    # Step 3: re-pair
    ciphertext = []
    for i in range(0, len(merged), 2):
        r = merged[i]
        c = merged[i + 1]
        ciphertext.append(pos_to_char[(r, c)])

    return "".join(ciphertext)
def bifid_decrypt(ciphertext: str, key="") -> str:

    char_to_pos, pos_to_char = generate_polybius_square(key)

    ciphertext = ciphertext.upper()
    ciphertext = "".join(c for c in ciphertext if c.isalpha())

    coords = []

    # Step 1: get all coordinates
    for ch in ciphertext:
        r, c = char_to_pos[ch]
        coords.append(r)
        coords.append(c)

    half = len(coords) // 2
    rows = coords[:half]
    cols = coords[half:]

    # Step 2: re-pair
    plaintext = []
    for r, c in zip(rows, cols):
        plaintext.append(pos_to_char[(r, c)])

    return "".join(plaintext)
def mod_inverse(a, m=26):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None
def affine_encrypt(text, a, b):
    if mod_inverse(a) is None:
        raise ValueError("Invalid key 'a'. It must be coprime with 26.")

    result = ""
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            x = ord(char) - base
            encrypted = (a * x + b) % 26
            result += chr(encrypted + base)
        else:
            result += char
    return result
def affine_decrypt(ciphertext, a, b):
    a_inv = mod_inverse(a)
    if a_inv is None:
        raise ValueError("Invalid key 'a'. It must be coprime with 26.")

    result = ""
    for char in ciphertext:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            y = ord(char) - base
            decrypted = (a_inv * (y - b)) % 26
            result += chr(decrypted + base)
        else:
            result += char
    return result
def A1Z26_encoder(data: str) -> str:
    result = []
    for char in data.upper():
        if char.isalpha():
            result.append(str(ord(char) - ord('A') + 1))
        elif char == ' ':
            result.append('/')  # word separator (optional)
    return ' '.join(result)
def A1Z26_decoder(data: str) -> str:
    result = []
    for token in data.split():
        if token == '/':
            result.append(' ')
        elif token.isdigit():
            num = int(token)
            if 1 <= num <= 26:
                result.append(chr(num + ord('A') - 1))
            else:
                result.append('?')
        else:
            result.append('?')
    return ''.join(result)
def rail_fence_encrypt(text, rails):
    if rails <= 1:
        return text

    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1

    for char in text:
        fence[rail].append(char)
        rail += direction

        if rail == 0 or rail == rails - 1:
            direction *= -1

    return ''.join(''.join(row) for row in fence)
def rail_fence_decrypt(ciphertext, rails):
    if rails <= 1:
        return ciphertext

    # Step 1: mark zigzag pattern
    pattern = [[None] * len(ciphertext) for _ in range(rails)]

    rail = 0
    direction = 1
    for i in range(len(ciphertext)):
        pattern[rail][i] = '*'
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction *= -1

    # Step 2: fill pattern with ciphertext
    index = 0
    for r in range(rails):
        for c in range(len(ciphertext)):
            if pattern[r][c] == '*' and index < len(ciphertext):
                pattern[r][c] = ciphertext[index]
                index += 1

    # Step 3: read zigzag
    result = []
    rail = 0
    direction = 1
    for i in range(len(ciphertext)):
        result.append(pattern[rail][i])
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction *= -1

    return ''.join(result)
def validate_substitution_key(key):
    key = key.upper()
    if len(key) != 26:
        raise ValueError("Key must be exactly 26 characters.")
    if not key.isalpha():
        raise ValueError("Key must contain only letters.")
    if len(set(key)) != 26:
        raise ValueError("Key must contain unique letters.")
    return key
def substitution_encrypt(text, key):
    key = validate_substitution_key(key)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    mapping = {alphabet[i]: key[i] for i in range(26)}

    result = ""
    for char in text:
        if char.isalpha():
            base = char.upper()
            enc = mapping[base]
            result += enc if char.isupper() else enc.lower()
        else:
            result += char
    return result
def substitution_decrypt(ciphertext, key):
    key = validate_substitution_key(key)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    reverse_mapping = {key[i]: alphabet[i] for i in range(26)}

    result = ""
    for char in ciphertext:
        if char.isalpha():
            base = char.upper()
            dec = reverse_mapping[base]
            result += dec if char.isupper() else dec.lower()
        else:
            result += char
    return result
def xor_cipher(data: str, key: str) -> str:
    result = []
    key_len = len(key)

    for i, char in enumerate(data):
        xor_val = ord(char) ^ ord(key[i % key_len])
        result.append(format(xor_val, '02x'))  # hex output

    return ' '.join(result)
def xor_decipher(hex_data: str, key: str) -> str:
    result = []
    bytes_data = hex_data.split()
    key_len = len(key)

    for i, byte in enumerate(bytes_data):
        xor_val = int(byte, 16) ^ ord(key[i % key_len])
        result.append(chr(xor_val))

    return ''.join(result)
def xor_bruteforce(hex_data: str):
    import string
    results = []

    bytes_data = bytes.fromhex(hex_data.replace(" ", ""))

    for key in range(256):
        decoded = ''.join(chr(b ^ key) for b in bytes_data)

        # Heuristic: mostly printable characters
        if all(c in string.printable for c in decoded):
            results.append((key, decoded))

    return results
def rot47_encoder_decoder(text):
    result = ""
    for char in text:
        o = ord(char)
        if 33 <= o <= 126:
            result += chr(33 + ((o - 33 + 47) % 94))
        else:
            result += char
    return result
def rc4(data: bytes, key: bytes, rounds: int = 1) -> bytes:
    S = list(range(256))
    j = 0

    for _ in range(rounds):
        for i in range(256):
            j = (j + S[i] + key[i % len(key)]) % 256
            S[i], S[j] = S[j], S[i]

    i = j = 0
    out = bytearray()
    for byte in data:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        out.append(byte ^ S[(S[i] + S[j]) % 256])

    return bytes(out)
def rc4_encrypt(text: str, key: str) -> str:
    return rc4(text.encode(), key.encode()).hex()
def rc4_decrypt(hexdata: str, key: str) -> str:
    return rc4(bytes.fromhex(hexdata), key.encode()).decode(errors="ignore")
def ciphersaber2_encrypt(text: str, key: str, rounds: int = 20) -> str:
    import os, base64
    iv = os.urandom(10)
    return base64.b64encode(iv + rc4(text.encode(), key.encode() + iv, rounds)).decode()
def ciphersaber2_decrypt(data: str, key: str, rounds: int = 20) -> str:
    import os, base64
    raw = base64.b64decode(data)
    iv, cipher = raw[:10], raw[10:]
    return rc4(cipher, key.encode() + iv, rounds).decode(errors="ignore")
def rc2_encrypt(plaintext: str, key: str) -> str:
    from Crypto.Cipher import ARC2
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    import base64
    key_bytes = key.encode()
    iv = get_random_bytes(8)  # RC2 block size = 8 bytes

    cipher = ARC2.new(key_bytes, ARC2.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), 8))

    # Return IV + ciphertext (Base64)
    return base64.b64encode(iv + ciphertext).decode()
def rc2_decrypt(ciphertext_b64: str, key: str) -> str:
    from Crypto.Cipher import ARC2
    from Crypto.Util.Padding import unpad
    import base64
    data = base64.b64decode(ciphertext_b64)
    iv = data[:8]
    ciphertext = data[8:]

    cipher = ARC2.new(key.encode(), ARC2.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), 8)

    return plaintext.decode()
def rc4_init(key: bytes):
    S = list(range(256))
    j = 0

    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) % 256
        S[i], S[j] = S[j], S[i]

    return S


def rc4_generate(S, data_len):
    i = j = 0
    keystream = []

    for _ in range(data_len):
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        K = S[(S[i] + S[j]) % 256]
        keystream.append(K)

    return keystream
def rc4_crypt(data: bytes, key: bytes) -> bytes:
    S = rc4_init(key)
    keystream = rc4_generate(S, len(data))
    return bytes([d ^ k for d, k in zip(data, keystream)])
def rc4_drop_crypt(data: bytes, key: bytes, drop_n=768) -> bytes:
    S = rc4_init(key)
    _ = rc4_generate(S, drop_n)   # discard biased output
    keystream = rc4_generate(S, len(data))
    return bytes([d ^ k for d, k in zip(data, keystream)])

def derive_aes_key(password: str, key_size: int = 32) -> bytes:
    import hashlib
    """
    key_size:
      16 → AES-128
      24 → AES-192
      32 → AES-256
    """
    return hashlib.sha256(password.encode()).digest()[:key_size]
def aes_encrypt(plaintext: str,password: str,mode: str = "CBC",key_size: int = 32) -> str:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    import base64

    key = derive_aes_key(password, key_size)
    mode = mode.upper()

    if mode == "ECB":
        cipher = AES.new(key, AES.MODE_ECB)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), 16))
        return base64.b64encode(ciphertext).decode()

    elif mode == "CBC":
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), 16))
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "CFB":
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CFB, iv)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "OFB":
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_OFB, iv)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "CTR":
        cipher = AES.new(key, AES.MODE_CTR)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(cipher.nonce + ciphertext).decode()

    elif mode == "GCM":
        nonce = get_random_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
        return base64.b64encode(nonce + tag + ciphertext).decode()

    elif mode == "CCM":
        nonce = get_random_bytes(11)
        cipher = AES.new(key, AES.MODE_CCM, nonce=nonce)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(nonce + cipher.digest() + ciphertext).decode()

    elif mode == "OCB":
        nonce = get_random_bytes(15)
        cipher = AES.new(key, AES.MODE_OCB, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
        return base64.b64encode(nonce + tag + ciphertext).decode()

    elif mode == "XTS":
        cipher = AES.new(key + key, AES.MODE_XTS)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(ciphertext).decode()

    elif mode == "SIV":
        cipher = AES.new(key, AES.MODE_SIV)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
        return base64.b64encode(tag + ciphertext).decode()

    else:
        raise ValueError("Unsupported AES mode")
def aes_decrypt(ciphertext_b64: str,password: str,mode: str = "CBC",key_size: int = 32) -> str:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    import base64

    key = derive_aes_key(password, key_size)
    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        cipher = AES.new(key, AES.MODE_ECB)
        return unpad(cipher.decrypt(data), 16).decode()

    elif mode == "CBC":
        iv, ct = data[:16], data[16:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), 16).decode()

    elif mode == "CFB":
        iv, ct = data[:16], data[16:]
        cipher = AES.new(key, AES.MODE_CFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "OFB":
        iv, ct = data[:16], data[16:]
        cipher = AES.new(key, AES.MODE_OFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "CTR":
        nonce, ct = data[:8], data[8:]
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "GCM":
        nonce, tag, ct = data[:12], data[12:28], data[28:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode()

    elif mode == "CCM":
        nonce, tag, ct = data[:11], data[11:27], data[27:]
        cipher = AES.new(key, AES.MODE_CCM, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode()

    elif mode == "OCB":
        nonce, tag, ct = data[:15], data[15:31], data[31:]
        cipher = AES.new(key, AES.MODE_OCB, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode()

    elif mode == "XTS":
        cipher = AES.new(key + key, AES.MODE_XTS)
        return cipher.decrypt(data).decode(errors="ignore")

    elif mode == "SIV":
        tag, ct = data[:16], data[16:]
        cipher = AES.new(key, AES.MODE_SIV)
        return cipher.decrypt_and_verify(ct, tag).decode()

    else:
        raise ValueError("Unsupported AES mode")
def des_encrypt(plaintext: str, password: str, mode: str = "CBC") -> str:
    from Crypto.Cipher import DES
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    from Crypto.Util import Counter
    import base64
    import hashlib

    # DES key must be exactly 8 bytes
    key = hashlib.md5(password.encode()).digest()[:8]
    mode = mode.upper()

    if mode == "ECB":
        cipher = DES.new(key, DES.MODE_ECB)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), 8))
        return base64.b64encode(ciphertext).decode()

    elif mode == "CBC":
        iv = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), 8))
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "CFB":
        iv = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_CFB, iv)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "OFB":
        iv = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_OFB, iv)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ciphertext).decode()

    elif mode == "CTR":
        ctr = Counter.new(64)
        cipher = DES.new(key, DES.MODE_CTR, counter=ctr)
        ciphertext = cipher.encrypt(plaintext.encode())
        return base64.b64encode(ciphertext).decode()

    else:
        raise ValueError("Unsupported DES mode")
def des_decrypt(ciphertext_b64: str, password: str, mode: str = "CBC") -> str:
    from Crypto.Cipher import DES
    from Crypto.Util.Padding import unpad
    from Crypto.Util import Counter
    import base64
    import hashlib

    key = hashlib.md5(password.encode()).digest()[:8]
    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        cipher = DES.new(key, DES.MODE_ECB)
        return unpad(cipher.decrypt(data), 8).decode()

    elif mode == "CBC":
        iv, ct = data[:8], data[8:]
        cipher = DES.new(key, DES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), 8).decode()

    elif mode == "CFB":
        iv, ct = data[:8], data[8:]
        cipher = DES.new(key, DES.MODE_CFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "OFB":
        iv, ct = data[:8], data[8:]
        cipher = DES.new(key, DES.MODE_OFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "CTR":
        ctr = Counter.new(64)
        cipher = DES.new(key, DES.MODE_CTR, counter=ctr)
        return cipher.decrypt(data).decode(errors="ignore")

    else:
        raise ValueError("Unsupported DES mode")

def tdes_encrypt(plaintext: str, password: str, mode: str = "CBC") -> str:
    from Crypto.Cipher import DES3
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    from Crypto.Util import Counter
    import base64
    import hashlib

    # Derive 24-byte (3-key) DES3 key
    key = hashlib.sha256(password.encode()).digest()[:24]
    key = DES3.adjust_key_parity(key)

    mode = mode.upper()
    data = plaintext.encode()

    if mode == "ECB":
        cipher = DES3.new(key, DES3.MODE_ECB)
        ct = cipher.encrypt(pad(data, 8))
        return base64.b64encode(ct).decode()

    elif mode == "CBC":
        iv = get_random_bytes(8)
        cipher = DES3.new(key, DES3.MODE_CBC, iv)
        ct = cipher.encrypt(pad(data, 8))
        return base64.b64encode(iv + ct).decode()

    elif mode == "CFB":
        iv = get_random_bytes(8)
        cipher = DES3.new(key, DES3.MODE_CFB, iv)
        ct = cipher.encrypt(data)
        return base64.b64encode(iv + ct).decode()

    elif mode == "OFB":
        iv = get_random_bytes(8)
        cipher = DES3.new(key, DES3.MODE_OFB, iv)
        ct = cipher.encrypt(data)
        return base64.b64encode(iv + ct).decode()

    elif mode == "CTR":
        ctr = Counter.new(64)
        cipher = DES3.new(key, DES3.MODE_CTR, counter=ctr)
        ct = cipher.encrypt(data)
        return base64.b64encode(ct).decode()

    elif mode == "EAX":
        cipher = DES3.new(key, DES3.MODE_EAX)
        ct, tag = cipher.encrypt_and_digest(data)
        return base64.b64encode(cipher.nonce + tag + ct).decode()

    else:
        raise ValueError("Unsupported Triple DES mode")
def tdes_decrypt(ciphertext_b64: str, password: str, mode: str = "CBC") -> str:
    from Crypto.Cipher import DES3
    from Crypto.Util.Padding import unpad
    from Crypto.Util import Counter
    import base64
    import hashlib

    key = hashlib.sha256(password.encode()).digest()[:24]
    key = DES3.adjust_key_parity(key)

    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        cipher = DES3.new(key, DES3.MODE_ECB)
        return unpad(cipher.decrypt(data), 8).decode()

    elif mode == "CBC":
        iv, ct = data[:8], data[8:]
        cipher = DES3.new(key, DES3.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), 8).decode()

    elif mode == "CFB":
        iv, ct = data[:8], data[8:]
        cipher = DES3.new(key, DES3.MODE_CFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "OFB":
        iv, ct = data[:8], data[8:]
        cipher = DES3.new(key, DES3.MODE_OFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "CTR":
        ctr = Counter.new(64)
        cipher = DES3.new(key, DES3.MODE_CTR, counter=ctr)
        return cipher.decrypt(data).decode(errors="ignore")

    elif mode == "EAX":
        nonce, tag, ct = data[:16], data[16:32], data[32:]
        cipher = DES3.new(key, DES3.MODE_EAX, nonce=nonce)
        return cipher.decrypt_and_verify(ct, tag).decode()

    else:
        raise ValueError("Unsupported Triple DES mode")
def derive_blowfish_key(password: str, max_len: int = 56) -> bytes:
    # Blowfish supports up to 56 bytes
    return hashlib.sha256(password.encode()).digest()[:max_len]
def blowfish_encrypt(plaintext: str,password: str,mode: str = "CBC") -> str:
    from Crypto.Cipher import Blowfish
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad
    import base64

    key = derive_blowfish_key(password)
    mode = mode.upper()

    if mode == "ECB":
        cipher = Blowfish.new(key, Blowfish.MODE_ECB)
        ct = cipher.encrypt(pad(plaintext.encode(), 8))
        return base64.b64encode(ct).decode()

    elif mode == "CBC":
        iv = get_random_bytes(8)
        cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
        ct = cipher.encrypt(pad(plaintext.encode(), 8))
        return base64.b64encode(iv + ct).decode()

    elif mode == "CFB":
        iv = get_random_bytes(8)
        cipher = Blowfish.new(key, Blowfish.MODE_CFB, iv)
        ct = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ct).decode()

    elif mode == "OFB":
        iv = get_random_bytes(8)
        cipher = Blowfish.new(key, Blowfish.MODE_OFB, iv)
        ct = cipher.encrypt(plaintext.encode())
        return base64.b64encode(iv + ct).decode()

    elif mode == "CTR":
        cipher = Blowfish.new(key, Blowfish.MODE_CTR)
        ct = cipher.encrypt(plaintext.encode())
        return base64.b64encode(cipher.nonce + ct).decode()

    else:
        raise ValueError("Unsupported Blowfish mode")
def blowfish_decrypt(ciphertext_b64: str,password: str,mode: str = "CBC") -> str:
    from Crypto.Cipher import Blowfish
    from Crypto.Util.Padding import unpad
    import base64

    key = derive_blowfish_key(password)
    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        cipher = Blowfish.new(key, Blowfish.MODE_ECB)
        return unpad(cipher.decrypt(data), 8).decode()

    elif mode == "CBC":
        iv, ct = data[:8], data[8:]
        cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), 8).decode()

    elif mode == "CFB":
        iv, ct = data[:8], data[8:]
        cipher = Blowfish.new(key, Blowfish.MODE_CFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "OFB":
        iv, ct = data[:8], data[8:]
        cipher = Blowfish.new(key, Blowfish.MODE_OFB, iv)
        return cipher.decrypt(ct).decode(errors="ignore")

    elif mode == "CTR":
        nonce, ct = data[:8], data[8:]
        cipher = Blowfish.new(key, Blowfish.MODE_CTR, nonce=nonce)
        return cipher.decrypt(ct).decode(errors="ignore")

    else:
        raise ValueError("Unsupported Blowfish mode")
def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)
def pkcs7_unpad(data: bytes) -> bytes:
    pad_len = data[-1]
    return data[:-pad_len]
def sm4_encrypt(plaintext: str, password: str, mode: str = "ECB") -> str:
    from gmssl.sm4 import CryptSM4, SM4_ENCRYPT
    import hashlib, base64, os

    key = hashlib.sha256(password.encode()).digest()[:16]
    crypt_sm4 = CryptSM4()
    crypt_sm4.set_key(key, SM4_ENCRYPT)

    data = pkcs7_pad(plaintext.encode())
    mode = mode.upper()

    if mode == "ECB":
        ct = crypt_sm4.crypt_ecb(data)
        return base64.b64encode(ct).decode()

    elif mode == "CBC":
        iv = os.urandom(16)
        ct = crypt_sm4.crypt_cbc(iv, data)
        return base64.b64encode(iv + ct).decode()

    else:
        raise ValueError("gmssl supports ONLY ECB and CBC modes")
def sm4_decrypt(ciphertext_b64: str, password: str, mode: str = "ECB") -> str:
    from gmssl.sm4 import CryptSM4, SM4_DECRYPT
    import hashlib, base64

    key = hashlib.sha256(password.encode()).digest()[:16]
    crypt_sm4 = CryptSM4()
    crypt_sm4.set_key(key, SM4_DECRYPT)

    data = base64.b64decode(ciphertext_b64)
    mode = mode.upper()

    if mode == "ECB":
        pt = crypt_sm4.crypt_ecb(data)
        return pkcs7_unpad(pt).decode()

    elif mode == "CBC":
        iv, ct = data[:16], data[16:]
        pt = crypt_sm4.crypt_cbc(iv, ct)
        return pkcs7_unpad(pt).decode()

    else:
        raise ValueError("gmssl supports ONLY ECB and CBC modes")
def enigma_cipher(text: str,rotor_order=("I", "II", "III"),rotor_positions=(0, 0, 0)) -> str:

    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Rotor definitions (Classic Enigma I)
    ROTORS = {
        "I":     "EKMFLGDQVZNTOWYHXUSPAIBRCJ",
        "II":    "AJDKSIRUXBLHWTMCQGZNPYFVOE",
        "III":   "BDFHJLCPRTXVZNYEIWGAKMUSQO",
    }

    # Reflector B
    REFLECTOR = "YRUHQSLDPXNGOKMIEBFZCWVJAT"

    def idx(c):
        return ord(c) - ord('A')

    # Load rotors
    r1 = ROTORS[rotor_order[0]]
    r2 = ROTORS[rotor_order[1]]
    r3 = ROTORS[rotor_order[2]]

    p1, p2, p3 = rotor_positions
    result = ""

    for char in text.upper():
        if char not in ALPHABET:
            result += char
            continue

        # Step rotors (simple stepping)
        p1 = (p1 + 1) % 26
        if p1 == 0:
            p2 = (p2 + 1) % 26
            if p2 == 0:
                p3 = (p3 + 1) % 26

        # Forward pass
        c = ALPHABET[(idx(char) + p1) % 26]
        c = r1[idx(c)]
        c = ALPHABET[(idx(c) + p2) % 26]
        c = r2[idx(c)]
        c = ALPHABET[(idx(c) + p3) % 26]
        c = r3[idx(c)]

        # Reflect
        c = REFLECTOR[idx(c)]

        # Reverse pass
        c = ALPHABET[r3.index(c)]
        c = ALPHABET[(idx(c) - p3) % 26]
        c = ALPHABET[r2.index(c)]
        c = ALPHABET[(idx(c) - p2) % 26]
        c = ALPHABET[r1.index(c)]
        c = ALPHABET[(idx(c) - p1) % 26]

        result += c

    return result
def bombe_find_settings(ciphertext: str,crib: str,rotor_order=("I", "II", "III"),max_results=5):
    results = []
    ciphertext = ciphertext.upper()
    crib = crib.upper()

    for p1 in range(26):
        for p2 in range(26):
            for p3 in range(26):

                # Encrypt crib using candidate settings
                test = enigma_cipher(
                    crib,
                    rotor_order=rotor_order,
                    rotor_positions=(p1, p2, p3)
                )

                # Slide encrypted crib over ciphertext
                for i in range(len(ciphertext) - len(crib) + 1):
                    if ciphertext[i:i+len(crib)] == test:
                        results.append({
                            "rotor_positions": (p1, p2, p3),
                            "crib_position": i
                        })
                        if len(results) >= max_results:
                            return results

    return results
def multiple_bombe_find_settings(ciphertext: str,cribs: list,rotor_order=("I", "II", "III"),max_results=5):
    ciphertext = ciphertext.upper()
    cribs = [c.upper() for c in cribs]
    results = []

    for p1 in range(26):
        for p2 in range(26):
            for p3 in range(26):

                settings_valid = True

                for crib in cribs:
                    crib_found = False

                    # Try crib at every possible position
                    for offset in range(len(ciphertext) - len(crib) + 1):

                        # Advance rotors by offset
                        shifted_positions = (
                            (p1 + offset) % 26,
                            (p2 + (p1 + offset)//26) % 26,
                            (p3 + (p2 + (p1 + offset)//26)//26) % 26
                        )

                        enc = enigma_cipher(
                            crib,
                            rotor_order=rotor_order,
                            rotor_positions=shifted_positions
                        )

                        if ciphertext[offset:offset+len(crib)] == enc:
                            crib_found = True
                            break

                    if not crib_found:
                        settings_valid = False
                        break

                if settings_valid:
                    results.append({
                        "rotor_positions": (p1, p2, p3)
                    })
                    if len(results) >= max_results:
                        return results

    return results


    return results
def typex_encrypt(text, rotors, positions):
    import string

    ALPHABET = string.ascii_uppercase
    result = ""
    for i, ch in enumerate(text):
        if ch in ALPHABET:
            shift = sum(positions) + i
            result += ALPHABET[(ALPHABET.index(ch) + shift) % 26]
            positions[0] = (positions[0] + 1) % 26
            if positions[0] == 0:
                positions[1] = (positions[1] + 1) % 26
        else:
            result += ch
    return result
def typex_decrypt(text, rotors, positions):
    import string

    ALPHABET = string.ascii_uppercase

    result = ""
    for i, ch in enumerate(text):
        if ch in ALPHABET:
            shift = sum(positions) + i
            result += ALPHABET[(ALPHABET.index(ch) - shift) % 26]
            positions[0] = (positions[0] + 1) % 26
            if positions[0] == 0:
                positions[1] = (positions[1] + 1) % 26
        else:
            result += ch
    return result
def lorenz_stream(wheels, length):
    import string

    ALPHABET = string.ascii_uppercase + " "
    stream = []
    for i in range(length):
        value = 0
        for w in wheels:
            value ^= w[i % len(w)]
        stream.append(value)
    return stream
def lorenz_encrypt(text, wheels):
    import string

    ALPHABET = string.ascii_uppercase + " "
    text = text.upper()
    stream = lorenz_stream(wheels, len(text))
    result = ""

    for i, ch in enumerate(text):
        if ch in ALPHABET:
            idx = ALPHABET.index(ch)
            result += ALPHABET[(idx ^ stream[i]) % len(ALPHABET)]
        else:
            result += ch

    return result
def lorenz_decrypt(ciphertext, wheels):
    import string
    ALPHABET = string.ascii_uppercase + " "
    return lorenz_encrypt(ciphertext, wheels)
def colossus_xor(ciphertext: bytes, keystream: bytes) -> bytes:
    return bytes(
        c ^ keystream[i % len(keystream)]
        for i, c in enumerate(ciphertext)
    )
def colossus_score(text: bytes) -> int:
    try:
        decoded = text.decode(errors="ignore").upper()
    except:
        return 0

    freq_chars = " ETAOINSHRDLU"
    score = sum(decoded.count(c) for c in freq_chars)

    # Penalize garbage
    score -= decoded.count('\x00') * 5
    score -= decoded.count('\xff') * 5

    return score
def colossus_analyze(ciphertext: bytes, keys: list):
    results = []

    for key in keys:
        ks = key.encode()
        decoded = colossus_xor(ciphertext, ks)
        score = colossus_score(decoded)

        results.append({
            "key": key,
            "score": score,
            "decoded": decoded.decode(errors="ignore")
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
def generate_rotor():
    import string
    import random

    ALPHABET = string.ascii_uppercase

    rotor = list(ALPHABET)
    random.shuffle(rotor)
    return rotor
def rotate(rotor, steps=1):
    return rotor[steps:] + rotor[:steps]
def sigaba_simulator(text, rotors):
    import string
    import random

    ALPHABET = string.ascii_uppercase

    result = ""

    for ch in text.upper():
        if ch not in ALPHABET:
            result += ch
            continue

        # 🔑 Irregular stepping (SIGABA concept)
        for i in range(len(rotors)):
            if random.choice([True, False]):
                rotors[i] = rotate(rotors[i], 1)

        idx = ALPHABET.index(ch)

        # Forward pass
        for rotor in rotors:
            idx = ALPHABET.index(rotor[idx])

        # Reverse pass (to make it decryptable)
        for rotor in reversed(rotors):
            idx = rotor.index(ALPHABET[idx])

        result += ALPHABET[idx]

    return result

def detect_hash_format(hash_str: str):
    import re
    if re.fullmatch(r"[a-fA-F0-9]+", hash_str):
        return "hex"
    if re.fullmatch(r"[A-Za-z0-9+/=]+", hash_str):
        return "base64"
    return "unknown"
def possible_hash_algorithms(hash_str: str):
    length = len(hash_str)
    fmt = detect_hash_format(hash_str)

    if fmt != "hex":
        return []

    mapping = {
        32: ["MD5"],
        40: ["SHA1"],
        56: ["SHA224"],
        64: ["SHA256", "BLAKE2s"],
        96: ["SHA384"],
        128: ["SHA512", "BLAKE2b"]
    }

    return mapping.get(length, [])

def hash_security_level(algo: str):
    weak = ["MD5", "SHA1"]
    if algo in weak:
        return "WEAK (collision attacks exist)"
    return "STRONG (no practical collisions known)"
def cracking_feasibility(algo: str):
    if algo == "MD5":
        return "Easily crackable (rainbow tables, GPU)"
    if algo == "SHA1":
        return "Crackable with enough resources"
    return "Not feasible without brute force"
def analyze_hash(hash_str: str):
    fmt = detect_hash_format(hash_str)
    candidates = possible_hash_algorithms(hash_str)

    analysis = {
        "hash": hash_str,
        "format": fmt,
        "length": len(hash_str),
        "possible_algorithms": candidates,
        "analysis": []
    }

    for algo in candidates:
        analysis["analysis"].append({
            "algorithm": algo,
            "security": hash_security_level(algo),
            "crack_feasibility": cracking_feasibility(algo)
        })

    if not candidates:
        analysis["analysis"].append({
            "note": "Unknown or custom hash format"
        })

    return analysis


def generate_all_hashes(data: str) -> dict:
    import hashlib

    data_bytes = data.encode()

    hashes = {
        "MD5": hashlib.md5(data_bytes).hexdigest(),
        "SHA1": hashlib.sha1(data_bytes).hexdigest(),
        "SHA224": hashlib.sha224(data_bytes).hexdigest(),
        "SHA256": hashlib.sha256(data_bytes).hexdigest(),
        "SHA384": hashlib.sha384(data_bytes).hexdigest(),
        "SHA512": hashlib.sha512(data_bytes).hexdigest(),
        "SHA3-224": hashlib.sha3_224(data_bytes).hexdigest(),
        "SHA3-256": hashlib.sha3_256(data_bytes).hexdigest(),
        "SHA3-384": hashlib.sha3_384(data_bytes).hexdigest(),
        "SHA3-512": hashlib.sha3_512(data_bytes).hexdigest(),
        "BLAKE2b": hashlib.blake2b(data_bytes).hexdigest(),
        "BLAKE2s": hashlib.blake2s(data_bytes).hexdigest(),
    }

    return hashes

def md2_hash(data: str) -> str:
    return "MD2 is not supported by Python hashlib"
def md4_hash(data: str) -> str:
    return "MD4 is not supported by Python hashlib"

def _F(x,y,z): return (x & y) | (~x & z)
def _G(x,y,z): return (x & y) | (x & z) | (y & z)
def _H(x,y,z): return x ^ y ^ z
def _rotl(x,n): return ((x<<n)|(x>>(32-n))) & 0xffffffff

def md4_hash(message: str) -> str:
    import struct
    msg = message.encode()
    orig_len = len(msg) * 8
    msg += b'\x80'
    while len(msg) % 64 != 56:
        msg += b'\x00'
    msg += struct.pack('<Q', orig_len)

    A, B, C, D = 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476

    for i in range(0, len(msg), 64):
        X = list(struct.unpack('<16I', msg[i:i+64]))
        AA, BB, CC, DD = A, B, C, D

        # Round 1
        for j in range(16):
            k = j
            s = [3,7,11,19][j%4]
            A = _rotl((A + _F(B,C,D) + X[k]) & 0xffffffff, s)
            A, B, C, D = D, A, B, C

        # Round 2
        for j in range(16):
            k = (j%4)*4 + j//4
            s = [3,5,9,13][j%4]
            A = _rotl((A + _G(B,C,D) + X[k] + 0x5a827999) & 0xffffffff, s)
            A, B, C, D = D, A, B, C

        # Round 3
        order = [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
        for j in range(16):
            k = order[j]
            s = [3,9,11,15][j%4]
            A = _rotl((A + _H(B,C,D) + X[k] + 0x6ed9eba1) & 0xffffffff, s)
            A, B, C, D = D, A, B, C

        A = (A + AA) & 0xffffffff
        B = (B + BB) & 0xffffffff
        C = (C + CC) & 0xffffffff
        D = (D + DD) & 0xffffffff

    return ''.join(f'{x:02x}' for x in struct.pack('<4I', A,B,C,D))
def md5_hash(data: str) -> str:
    import hashlib
    return hashlib.md5(data.encode()).hexdigest()
def md6_hash(data: str) -> str:
    return "MD6 was never standardized and is not supported"
#def sha0_hash(data: str) -> str:
    #return "SHA-0 was withdrawn and is not implemented"
def sha1_hash(data: str) -> str:
    import hashlib
    return hashlib.sha1(data.encode()).hexdigest()
def sha2_hash(data: str, variant: str) -> str:
    import hashlib
    data = data.encode()

    if variant == "224":
        return hashlib.sha224(data).hexdigest()
    elif variant == "256":
        return hashlib.sha256(data).hexdigest()
    elif variant == "384":
        return hashlib.sha384(data).hexdigest()
    elif variant == "512":
        return hashlib.sha512(data).hexdigest()
    elif variant == "512_224":
        return hashlib.new("sha512_224", data).hexdigest()
    elif variant == "512_256":
        return hashlib.new("sha512_256", data).hexdigest()
    else:
        return "Invalid SHA-2 variant"
def sha3_hash(data: str, variant: str) -> str:
    import hashlib
    data = data.encode()

    if variant == "224":
        return hashlib.sha3_224(data).hexdigest()
    elif variant == "256":
        return hashlib.sha3_256(data).hexdigest()
    elif variant == "384":
        return hashlib.sha3_384(data).hexdigest()
    elif variant == "512":
        return hashlib.sha3_512(data).hexdigest()
    else:
        return "Invalid SHA-3 variant"
def shake_hash(data: str, variant: str, length: int = 32) -> str:
    import hashlib
    data = data.encode()

    if variant == "128":
        return hashlib.shake_128(data).hexdigest(length)
    elif variant == "256":
        return hashlib.shake_256(data).hexdigest(length)
    else:
        return "Invalid SHAKE variant"
def sha_family_hash(data: str, algo: str) -> str:
    algo = algo.lower()

    if algo == "sha0":
        return sha0_hash(data)

    elif algo == "sha1":
        return sha1_hash(data)

    elif algo.startswith("sha2"):
        return sha2_hash(data, algo.split("-")[1])

    elif algo.startswith("sha3"):
        return sha3_hash(data, algo.split("-")[1])

    elif algo.startswith("shake"):
        variant = algo.replace("shake", "")
        return shake_hash(data, variant)

    else:
        return "Unsupported SHA algorithm"
def ripemd160_hash(data: str) -> str:
    import hashlib

    try:
        h = hashlib.new("ripemd160")
    except ValueError:
        raise RuntimeError("RIPEMD160 not supported in this environment")

    h.update(data.encode())
    return h.hexdigest()
def whirlpool_hash(data: str) -> str:
    import hashlib

    try:
        h = hashlib.new("whirlpool")
    except ValueError:
        raise RuntimeError("Whirlpool not supported in this environment")

    h.update(data.encode())
    return h.hexdigest()
def has160_demo(data: str) -> str:
    import hashlib

    sha1 = hashlib.sha1(data.encode()).hexdigest()
    return "HAS160-DEMO-" + sha1[:40]
def snefru_demo(data: str, bits=256) -> str:
    import hashlib

    if bits == 128:
        return hashlib.md5(data.encode()).hexdigest()
    else:
        return hashlib.sha256(data.encode()).hexdigest()
def blake2b_hash(data: str, digest_size=64) -> str:
    import hashlib
    h = hashlib.blake2b(data.encode(), digest_size=digest_size)
    return h.hexdigest()
def blake2s_hash(data: str, digest_size=32) -> str:
    import hashlib
    h = hashlib.blake2s(data.encode(), digest_size=digest_size)
    return h.hexdigest()
def sm3_hash(data: str) -> str:
    from gmssl import sm3, func
    msg = data.encode()
    return sm3.sm3_hash(func.bytes_to_list(msg))
def keccak_hash(data: str, bits: int = 256) -> str:
    from Crypto.Hash import keccak

    if bits == 224:
        h = keccak.new(digest_bits=224)
    elif bits == 256:
        h = keccak.new(digest_bits=256)
    elif bits == 384:
        h = keccak.new(digest_bits=384)
    elif bits == 512:
        h = keccak.new(digest_bits=512)
    else:
        raise ValueError("Keccak supports 224/256/384/512 bits")

    h.update(data.encode())
    return h.hexdigest()
def shake_hash(data: str, bits: int = 256) -> str:
    import hashlib

    if bits == 128:
        return hashlib.shake_128(data.encode()).hexdigest(32)
    elif bits == 256:
        return hashlib.shake_256(data.encode()).hexdigest(64)
    else:
        raise ValueError("SHAKE supports 128 or 256")
def streebog_256(data: str) -> str:
    from Crypto.Hash import GOST34112012
    h = GOST34112012.new(digest_bits=256)
    h.update(data.encode())
    return h.hexdigest()
def streebog_512(data: str) -> str:
    from Crypto.Hash import GOST34112012
    h = GOST34112012.new(digest_bits=512)
    h.update(data.encode())
    return h.hexdigest()
def ssdeep_hash(data: str) -> str:
    import ssdeep
    return ssdeep.hash(data)

def ssdeep_compare(hash1: str, hash2: str) -> int:
    import ssdeep
    return ssdeep.compare(hash1, hash2)

def hmac_generate(message: str, key: str, algo: str = "SHA256") -> str:
    import hmac
    import hashlib

    algo = algo.upper()

    algo_map = {
        "MD5": hashlib.md5,
        "SHA1": hashlib.sha1,
        "SHA256": hashlib.sha256,
        "SHA384": hashlib.sha384,
        "SHA512": hashlib.sha512,
        "SHA3-256": hashlib.sha3_256,
        "SHA3-512": hashlib.sha3_512,
    }

    if algo not in algo_map:
        raise ValueError("Unsupported HMAC algorithm")

    h = hmac.new(
        key.encode(),
        message.encode(),
        algo_map[algo]
    )

    return h.hexdigest()
def hmac_verify(message: str, key: str, algo: str, given_hmac: str) -> bool:
    import hmac

    calculated = hmac_generate(message, key, algo)
    return hmac.compare_digest(calculated, given_hmac)
def bcrypt_hash(password: str, rounds: int = 12) -> str:
    import bcrypt
    salt = bcrypt.gensalt(rounds)
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()
def bcrypt_compare(password: str, hashed: str) -> bool:
    import bcrypt
    return bcrypt.checkpw(password.encode(), hashed.encode())
def bcrypt_parse(hash_value: str) -> dict:
    parts = hash_value.split('$')
    if len(parts) != 4:
        raise ValueError("Invalid bcrypt hash format")

    return {
        "algorithm": parts[1],
        "cost_factor": int(parts[2]),
        "salt+hash": parts[3][:],
        "salt": parts[3][:22],
        "hash": parts[3][22:]
    }
def scrypt_hash(password: str, salt: bytes = None,
                n: int = 2**14, r: int = 8, p: int = 1) -> dict:
    import hashlib, os, base64

    if salt is None:
        salt = os.urandom(16)

    key = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=64
    )

    return {
        "algorithm": "scrypt",
        "n": n,
        "r": r,
        "p": p,
        "salt": base64.b64encode(salt).decode(),
        "hash": base64.b64encode(key).decode()
    }
def scrypt_compare(password: str, stored: dict) -> bool:
    import hashlib, base64, hmac

    salt = base64.b64decode(stored["salt"])
    original = base64.b64decode(stored["hash"])

    new_key = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=stored["n"],
        r=stored["r"],
        p=stored["p"],
        dklen=len(original)
    )

    return hmac.compare_digest(original, new_key)
def scrypt_parse(stored: dict):
    return {
        "algorithm": stored["algorithm"],
        "N": stored["n"],
        "r": stored["r"],
        "p": stored["p"],
        "salt_length": len(stored["salt"]),
        "hash_length": len(stored["hash"])
    }
def pbkdf2_derive_key(password: str,
                      salt: bytes = None,
                      iterations: int = 100000,
                      dklen: int = 32,
                      hash_name: str = "sha256") -> dict:
    import hashlib, os, base64

    if salt is None:
        salt = os.urandom(16)

    key = hashlib.pbkdf2_hmac(
        hash_name,
        password.encode(),
        salt,
        iterations,
        dklen
    )

    return {
        "algorithm": "PBKDF2",
        "hash": hash_name.upper(),
        "iterations": iterations,
        "key_length": dklen,
        "salt": base64.b64encode(salt).decode(),
        "derived_key": base64.b64encode(key).decode()
    }
def pbkdf2_verify(password: str, stored: dict) -> bool:
    import hashlib, base64, hmac

    salt = base64.b64decode(stored["salt"])
    original = base64.b64decode(stored["derived_key"])

    new_key = hashlib.pbkdf2_hmac(
        stored["hash"].lower(),
        password.encode(),
        salt,
        stored["iterations"],
        len(original)
    )

    return hmac.compare_digest(original, new_key)
def evp_bytes_to_key(password: bytes,
                     salt: bytes,
                     key_len: int,
                     iv_len: int,
                     hash_name: str = "md5") -> tuple:
    import hashlib

    digest = getattr(hashlib, hash_name)
    derived = b""
    block = b""

    while len(derived) < (key_len + iv_len):
        block = digest(block + password + salt).digest()
        derived += block

    key = derived[:key_len]
    iv = derived[key_len:key_len + iv_len]

    return key, iv
def derive_evp_key(password: str,
                   salt: bytes = None,
                   key_len: int = 32,
                   iv_len: int = 16,
                   hash_name: str = "md5") -> dict:
    import os, base64

    if salt is None:
        salt = os.urandom(8)  # OpenSSL standard salt length

    key, iv = evp_bytes_to_key(
        password.encode(),
        salt,
        key_len,
        iv_len,
        hash_name
    )

    return {
        "algorithm": "EVP_BytesToKey",
        "hash": hash_name.upper(),
        "salt": base64.b64encode(salt).decode(),
        "key": base64.b64encode(key).decode(),
        "iv": base64.b64encode(iv).decode(),
        "key_length": key_len,
        "iv_length": iv_len
    }
def prng_python(seed: int, count: int = 10):
    import random
    random.seed(seed)
    return [random.randint(0, 2**32 - 1) for _ in range(count)]
def prng_lcg(seed: int, count: int = 10):
    m = 2**32
    a = 1664525
    c = 1013904223

    values = []
    x = seed

    for _ in range(count):
        x = (a * x + c) % m
        values.append(x)

    return values
def prng_xorshift(seed: int, count: int = 10):
    values = []
    x = seed

    for _ in range(count):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        values.append(x & 0xFFFFFFFF)

    return values
def csprng(count: int = 10):
    import secrets
    return [secrets.randbits(32) for _ in range(count)]
def jwt_sign(payload: dict, secret: str, algo: str = "HS256", expire_sec: int = None) -> str:
    import jwt, time

    data = payload.copy()
    if expire_sec:
        data["exp"] = int(time.time()) + expire_sec

    token = jwt.encode(data, secret, algorithm=algo)
    return token
def jwt_verify(token: str, secret: str, algo: str = "HS256") -> dict:
    import jwt

    decoded = jwt.decode(token, secret, algorithms=[algo])
    return decoded
def jwt_decode_no_verify(token: str) -> dict:
    import jwt

    # Decode WITHOUT signature verification (for inspection only)
    return jwt.decode(token, options={"verify_signature": False})
def ctx1_encode(data: str) -> str:
    import zlib, base64

    compressed = zlib.compress(data.encode())
    encoded = base64.urlsafe_b64encode(compressed).decode()
    return encoded
def ctx1_decode(token: str) -> str:
    import zlib, base64

    decoded = base64.urlsafe_b64decode(token.encode())
    decompressed = zlib.decompress(decoded)
    return decompressed.decode()
def fletcher8(data: bytes) -> int:
    sum1 = 0
    sum2 = 0
    for b in data:
        sum1 = (sum1 + b) % 15
        sum2 = (sum2 + sum1) % 15
    return (sum2 << 4) | sum1
def fletcher16(data: bytes) -> int:
    sum1 = 0
    sum2 = 0
    for b in data:
        sum1 = (sum1 + b) % 255
        sum2 = (sum2 + sum1) % 255
    return (sum2 << 8) | sum1
def fletcher32(data: bytes) -> int:
    sum1 = 0
    sum2 = 0
    for i in range(0, len(data), 2):
        word = data[i] << 8
        if i + 1 < len(data):
            word |= data[i + 1]
        sum1 = (sum1 + word) % 65535
        sum2 = (sum2 + sum1) % 65535
    return (sum2 << 16) | sum1
def fletcher64(data: bytes) -> int:
    sum1 = 0
    sum2 = 0
    for i in range(0, len(data), 4):
        word = 0
        for j in range(4):
            if i + j < len(data):
                word = (word << 8) | data[i + j]
        sum1 = (sum1 + word) % (2**32 - 1)
        sum2 = (sum2 + sum1) % (2**32 - 1)
    return (sum2 << 32) | sum1
def adler32_checksum(data: bytes) -> int:
    MOD_ADLER = 65521
    a = 1
    b = 0

    for byte in data:
        a = (a + byte) % MOD_ADLER
        b = (b + a) % MOD_ADLER

    return (b << 16) | a
def luhn_check_digit(number: str) -> int:
    total = 0
    reverse_digits = number[::-1]

    for i, d in enumerate(reverse_digits):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return (10 - (total % 10)) % 10
def luhn_validate(number: str) -> bool:
    if not number.isdigit() or len(number) < 2:
        return False

    check = int(number[-1])
    return luhn_check_digit(number[:-1]) == check
def luhn_generate(base_number: str) -> str:
    if not base_number.isdigit():
        raise ValueError("Input must be numeric")

    return base_number + str(luhn_check_digit(base_number))
def crc8(data: bytes, poly=0x07, init=0x00) -> int:
    crc = init
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc
def crc16(data: bytes, poly=0x1021, init=0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc
def crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF
def tcp_ip_checksum(data: bytes) -> int:
    if len(data) % 2 == 1:
        data += b'\x00'  # pad if odd length

    checksum = 0
    for i in range(0, len(data), 2):
        word = (data[i] << 8) + data[i + 1]
        checksum += word
        checksum = (checksum & 0xFFFF) + (checksum >> 16)

    return (~checksum) & 0xFFFF
def rsa_genrate_keys(key_size=2048):
    from Crypto.PublicKey import RSA
    key = RSA.generate(key_size)
    private_key = key.export_key().decode()
    public_key = key.publickey().export_key().decode()
    return public_key, private_key
def rsa_encrypt(plaintext: str, public_key_str: str) -> str:
    from Crypto.Cipher import PKCS1_OAEP
    from Crypto.PublicKey import RSA
    import base64
    public_key = RSA.import_key(public_key_str)
    cipher = PKCS1_OAEP.new(public_key)
    ciphertext = cipher.encrypt(plaintext.encode())
    return base64.b64encode(ciphertext).decode()
def rsa_decrypt(ciphertext_b64: str, private_key_str: str) -> str:
    from Crypto.Cipher import PKCS1_OAEP
    from Crypto.PublicKey import RSA
    import base64
    private_key = RSA.import_key(private_key_str)
    cipher = PKCS1_OAEP.new(private_key)
    ciphertext = base64.b64decode(ciphertext_b64)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext.decode()
def rsa_sign(message: str, private_key_str: str) -> str:
    from Crypto.PublicKey import RSA
    from Crypto.Signature import pkcs1_15
    from Crypto.Hash import SHA256
    import base64
    private_key = RSA.import_key(private_key_str)
    h = SHA256.new(message.encode())
    signature = pkcs1_15.new(private_key).sign(h)
    return base64.b64encode(signature).decode()
def rsa_verify(message: str, signature_b64: str, public_key_str: str) -> bool:
    from Crypto.PublicKey import RSA
    from Crypto.Signature import pkcs1_15
    from Crypto.Hash import SHA256
    import base64
    public_key = RSA.import_key(public_key_str)
    h = SHA256.new(message.encode())
    signature = base64.b64decode(signature_b64)

    try:
        pkcs1_15.new(public_key).verify(h, signature)
        return True
    except (ValueError, TypeError):
        return False
def dh_generate_parameters():
    from Crypto.PublicKey import DSA
    # 2048-bit safe prime group
    params = DSA.generate(2048)
    return params.p, params.g
def dh_generate_private_key(p):
    import secrets
    return secrets.randbelow(p - 2) + 2
def dh_generate_public_key(g, private_key, p):
    return pow(g, private_key, p)
def dh_compute_shared_secret(peer_public, private_key, p):
    return pow(peer_public, private_key, p)
def dh_derive_key(shared_secret, key_len=32):
    import hashlib
    return hashlib.sha256(shared_secret.to_bytes((shared_secret.bit_length() + 7) // 8, 'big')).digest()[:key_len]
def aes_encrypt_with_dh_key(data: str, key: bytes):
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = data.encode() + b"\x00" * (16 - len(data) % 16)
    return iv + cipher.encrypt(padded)
def ecdh_generate_keypair():
    from Crypto.PublicKey import ECC
    key = ECC.generate(curve="P-256")
    private_key = key.export_key(format="PEM")
    public_key = key.public_key().export_key(format="PEM")
    return public_key, private_key
def ecdh_shared_secret(private_key_pem: str, peer_public_key_pem: str) -> bytes:
    from Crypto.Hash import SHA256
    from Crypto.PublicKey import ECC
    private_key = ECC.import_key(private_key_pem)
    peer_public = ECC.import_key(peer_public_key_pem)
    shared_point = peer_public.pointQ * private_key.d
    shared_secret = int(shared_point.x).to_bytes(32, "big")
    return SHA256.new(shared_secret).digest()
def aes_encrypt_ecdh(data: str, key: bytes):
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode())
    return iv, ciphertext, tag
def dsa_generate_keys(key_size=2048):
    from Crypto.PublicKey import DSA
    key = DSA.generate(key_size)
    private_key = key.export_key().decode()
    public_key = key.publickey().export_key().decode()
    return public_key, private_key
def dsa_sign(message: str, private_key_pem: str) -> str:
    from Crypto.PublicKey import DSA
    from Crypto.Signature import DSS
    from Crypto.Hash import SHA256
    import base64
    private_key = DSA.import_key(private_key_pem)
    h = SHA256.new(message.encode())
    signer = DSS.new(private_key, 'fips-186-3')
    signature = signer.sign(h)
    return base64.b64encode(signature).decode()
def dsa_verify(message: str, signature_b64: str, public_key_pem: str) -> bool:
    from Crypto.PublicKey import DSA
    from Crypto.Signature import DSS
    from Crypto.Hash import SHA256
    import base64
    public_key = DSA.import_key(public_key_pem)
    h = SHA256.new(message.encode())
    signature = base64.b64decode(signature_b64)
    verifier = DSS.new(public_key, 'fips-186-3')

    try:
        verifier.verify(h, signature)
        return True
    except ValueError:
        return False
def ecdsa_generate_keys():
    from Crypto.PublicKey import ECC
    key = ECC.generate(curve="P-256")
    private_key = key.export_key(format="PEM")
    public_key = key.public_key().export_key(format="PEM")
    return public_key, private_key
def ecdsa_sign(message: str, private_key_pem: str) -> str:
    from Crypto.PublicKey import ECC
    from Crypto.Signature import DSS
    from Crypto.Hash import SHA256
    import base64
    private_key = ECC.import_key(private_key_pem)
    h = SHA256.new(message.encode())
    signer = DSS.new(private_key, 'fips-186-3')
    signature = signer.sign(h)
    return base64.b64encode(signature).decode()
def ecdsa_verify(message: str, signature_b64: str, public_key_pem: str) -> bool:
    from Crypto.PublicKey import ECC
    from Crypto.Signature import DSS
    from Crypto.Hash import SHA256
    import base64
    public_key = ECC.import_key(public_key_pem)
    h = SHA256.new(message.encode())
    signature = base64.b64decode(signature_b64)
    verifier = DSS.new(public_key, 'fips-186-3')

    try:
        verifier.verify(h, signature)
        return True
    except ValueError:
        return False
def ed25519_generate_keys():
    from Crypto.PublicKey import ECC
    key = ECC.generate(curve="Ed25519")
    private_key = key.export_key(format="PEM")
    public_key = key.public_key().export_key(format="PEM")
    return public_key, private_key
def ed25519_sign(message: str, private_key_pem: str) -> str:
    from Crypto.PublicKey import ECC
    from Crypto.Signature import eddsa
    import base64
    private_key = ECC.import_key(private_key_pem)
    signer = eddsa.new(private_key, mode='rfc8032')
    signature = signer.sign(message.encode())
    return base64.b64encode(signature).decode()
def ed25519_verify(message: str, signature_b64: str, public_key_pem: str) -> bool:
    from Crypto.PublicKey import ECC
    from Crypto.Signature import eddsa
    import base64
    public_key = ECC.import_key(public_key_pem)
    verifier = eddsa.new(public_key, mode='rfc8032')
    signature = base64.b64decode(signature_b64)

    try:
        verifier.verify(message.encode(), signature)
        return True
    except ValueError:
        return False
def x25519_generate_keypair():
    from Crypto.PublicKey import ECC
    key = ECC.generate(curve="X25519")
    private_key = key.export_key(format="PEM")
    public_key = key.public_key().export_key(format="PEM")
    return public_key, private_key
def x25519_shared_secret(private_key_pem: str, peer_public_key_pem: str) -> bytes:
    from Crypto.PublicKey import ECC
    from Crypto.Hash import SHA256
    private_key = ECC.import_key(private_key_pem)
    peer_public = ECC.import_key(peer_public_key_pem)

    # X25519 ECDH primitive
    shared_point = peer_public.pointQ * private_key.d
    shared_secret = int(shared_point.x).to_bytes(32, "little")

    # Mandatory KDF (real-world requirement)
    return SHA256.new(shared_secret).digest()
def aes_encrypt_x25519(data: str, key: bytes):
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode())
    return cipher.nonce, ciphertext, tag








def parse_x509_certificate(cert_path: str) -> dict:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend

    with open(cert_path, "rb") as f:
        cert_data = f.read()

    try:
        cert = x509.load_pem_x509_certificate(cert_data, default_backend())
    except ValueError:
        cert = x509.load_der_x509_certificate(cert_data, default_backend())

    info = {
        "Subject": cert.subject.rfc4514_string(),
        "Issuer": cert.issuer.rfc4514_string(),
        "Serial Number": hex(cert.serial_number),
        "Version": cert.version.name,
        "Not Before": cert.not_valid_before.isoformat(),
        "Not After": cert.not_valid_after.isoformat(),
        "Signature Algorithm": cert.signature_hash_algorithm.name,
        "Public Key Type": cert.public_key().__class__.__name__,
        "Public Key Size": cert.public_key().key_size,
        "Extensions": []
    }

    for ext in cert.extensions:
        info["Extensions"].append(ext.oid._name or str(ext.oid))

    return info
def analyze_tls_certificate(hostname: str, port: int = 443) -> dict:
    import socket, ssl
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from datetime import datetime

    context = ssl.create_default_context()
    with socket.create_connection((hostname, port), timeout=5) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert_bin = ssock.getpeercert(binary_form=True)

    cert = x509.load_der_x509_certificate(cert_bin, default_backend())

    now = datetime.utcnow()
    expired = now < cert.not_valid_before or now > cert.not_valid_after

    info = {
        "Subject": cert.subject.rfc4514_string(),
        "Issuer": cert.issuer.rfc4514_string(),
        "Serial Number": hex(cert.serial_number),
        "Not Before": cert.not_valid_before.isoformat(),
        "Not After": cert.not_valid_after.isoformat(),
        "Expired": expired,
        "Signature Algorithm": cert.signature_hash_algorithm.name,
        "Public Key Type": cert.public_key().__class__.__name__,
        "Public Key Size": cert.public_key().key_size,
        "Extensions": [],
        "Warnings": []
    }

    # Extensions
    for ext in cert.extensions:
        info["Extensions"].append(ext.oid._name or str(ext.oid))

    # Basic warnings
    if cert.public_key().key_size < 2048:
        info["Warnings"].append("Weak public key size (<2048 bits)")

    if cert.signature_hash_algorithm.name.lower() in ["md5", "sha1"]:
        info["Warnings"].append("Weak signature hash algorithm")

    if expired:
        info["Warnings"].append("Certificate is expired or not yet valid")

    return info
def pem_to_der(input_path: str, output_path: str):
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    with open(input_path, "rb") as f:
        pem_data = f.read()

    cert = x509.load_pem_x509_certificate(pem_data, default_backend())
    der_data = cert.public_bytes(encoding=serialization.Encoding.DER)

    with open(output_path, "wb") as f:
        f.write(der_data)
def der_to_pem(input_path: str, output_path: str):
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    with open(input_path, "rb") as f:
        der_data = f.read()

    cert = x509.load_der_x509_certificate(der_data, default_backend())
    pem_data = cert.public_bytes(encoding=serialization.Encoding.PEM)

    with open(output_path, "wb") as f:
        f.write(pem_data)
def extract_public_key_any(input_path: str, output_path: str, password: str = None):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend

    with open(input_path, "rb") as f:
        data = f.read()

    pwd = password.encode() if password else None
    pubkey = None

    # 1. X.509 Certificate (PEM or DER)
    try:
        try:
            cert = x509.load_pem_x509_certificate(data, default_backend())
        except ValueError:
            cert = x509.load_der_x509_certificate(data, default_backend())
        pubkey = cert.public_key()
    except Exception:
        pass

    # 2. Private Key (RSA / DSA / EC / Ed25519 / X25519)
    if pubkey is None:
        try:
            private_key = serialization.load_pem_private_key(
                data, password=pwd, backend=default_backend()
            )
            pubkey = private_key.public_key()
        except Exception:
            pass

    # 3. Public Key (PEM)
    if pubkey is None:
        try:
            pubkey = serialization.load_pem_public_key(
                data, backend=default_backend()
            )
        except Exception:
            pass

    # 4. SSH Public Key
    if pubkey is None:
        try:
            pubkey = serialization.load_ssh_public_key(
                data, backend=default_backend()
            )
        except Exception:
            pass

    # 5. SSH Private Key
    if pubkey is None:
        try:
            private_key = serialization.load_ssh_private_key(
                data, password=pwd, backend=default_backend()
            )
            pubkey = private_key.public_key()
        except Exception:
            pass

    if pubkey is None:
        raise ValueError("Unsupported or invalid key/certificate format")

    pem = pubkey.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    with open(output_path, "wb") as f:
        f.write(pem)
def generate_fingerprint(data: bytes, algo: str = "sha256") -> str:
    import hashlib

    algo = algo.lower()
    if algo not in hashlib.algorithms_available:
        raise ValueError("Unsupported hash algorithm")

    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()
def fingerprint_certificate(cert_path: str, algo: str = "sha256") -> str:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend

    with open(cert_path, "rb") as f:
        data = f.read()

    try:
        cert = x509.load_pem_x509_certificate(data, default_backend())
    except ValueError:
        cert = x509.load_der_x509_certificate(data, default_backend())

    return generate_fingerprint(cert.public_bytes(), algo)
def fingerprint_public_key(key_path: str, algo: str = "sha256") -> str:
    from cryptography.hazmat.primitives import serialization

    with open(key_path, "rb") as f:
        key_data = f.read()

    return generate_fingerprint(key_data, algo)
def fingerprint_file(file_path: str, algo: str = "sha256") -> str:
    import hashlib

    h = hashlib.new(algo)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

#69
def file_hash(filepath: str, algorithm: str = "SHA256") -> str:
    import hashlib
    algorithm = algorithm.upper()
    hash_map = {
        "MD5": hashlib.md5,
        "SHA1": hashlib.sha1,
        "SHA256": hashlib.sha256,
        "SHA512": hashlib.sha512,
        "SHA3_256": hashlib.sha3_256,
        "BLAKE2B": hashlib.blake2b,
    }

    if algorithm not in hash_map:
        return "Unsupported hash algorithm"

    h = hash_map[algorithm]()

    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):  # 8KB chunks
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "File not found"
    except PermissionError:
        return "Permission denied"
def file_multi_hash(filepath: str):
    import hashlib
    algorithms = {
        "MD5": hashlib.md5(),
        "SHA1": hashlib.sha1(),
        "SHA256": hashlib.sha256(),
        "SHA512": hashlib.sha512(),
    }

    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                for h in algorithms.values():
                    h.update(chunk)

        return {name: h.hexdigest() for name, h in algorithms.items()}
    except Exception as e:
        return str(e)
#70
def directory_hash(directory: str,algorithm: str = "SHA256",ignore_hidden: bool = True) -> str:
    import os
    import hashlib
    algorithm = algorithm.upper()
    hash_map = {
        "MD5": hashlib.md5,
        "SHA1": hashlib.sha1,
        "SHA256": hashlib.sha256,
        "SHA512": hashlib.sha512,
        "SHA3_256": hashlib.sha3_256,
        "BLAKE2B": hashlib.blake2b,
    }

    if algorithm not in hash_map:
        return "Unsupported hash algorithm"

    dir_hasher = hash_map[algorithm]()

    for root, dirs, files in os.walk(directory):
        dirs.sort()
        files.sort()

        for filename in files:
            if ignore_hidden and filename.startswith("."):
                continue

            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, directory)

            # Include relative path in hash (important!)
            dir_hasher.update(rel_path.encode())

            try:
                with open(filepath, "rb") as f:
                    while chunk := f.read(8192):
                        dir_hasher.update(chunk)
            except (PermissionError, FileNotFoundError):
                continue

    return dir_hasher.hexdigest()
def directory_multi_hash(directory: str):
    import os
    import hashlib
    algorithms = {
        "MD5": hashlib.md5(),
        "SHA1": hashlib.sha1(),
        "SHA256": hashlib.sha256(),
        "SHA512": hashlib.sha512(),
    }

    for root, dirs, files in os.walk(directory):
        dirs.sort()
        files.sort()

        for filename in files:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, directory)

            for h in algorithms.values():
                h.update(rel_path.encode())

            try:
                with open(filepath, "rb") as f:
                    while chunk := f.read(8192):
                        for h in algorithms.values():
                            h.update(chunk)
            except Exception:
                continue

    return {name: h.hexdigest() for name, h in algorithms.items()}
#71
def compare_file_hashes(file1: str, file2: str, algorithm: str = "SHA256"):
    try:
        h1 = file_hash(file1, algorithm)
        h2 = file_hash(file2, algorithm)

        return {
            "algorithm": algorithm,
            "file1_hash": h1,
            "file2_hash": h2,
            "match": h1 == h2
        }

    except Exception as e:
        return {"error": str(e)}
#72
def file_encrypt_aes(input_file: str, output_file: str, key: bytes):
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    import os
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    with open(input_file, "rb") as f:
        data = f.read()
    # PKCS7 padding
    pad_len = 16 - (len(data) % 16)
    data += bytes([pad_len]) * pad_len
    encrypted = cipher.encrypt(data)
    with open(output_file, "wb") as f:
        f.write(iv + encrypted)
def file_decrypt_aes(input_file: str, output_file: str, key: bytes):
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    import os
    with open(input_file, "rb") as f:
        raw = f.read()
    iv = raw[:16]
    ciphertext = raw[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)
    # Remove PKCS7 padding
    pad_len = decrypted[-1]
    decrypted = decrypted[:-pad_len]
    with open(output_file, "wb") as f:
        f.write(decrypted)
def derive_key(password: str, length=32):
    import hashlib
    return hashlib.sha256(password.encode()).digest()[:length]
#73
def create_integrity_baseline(filepath: str, algorithm="SHA256"):
    import json
    import time
    h = file_hash(filepath, algorithm)
    baseline = {
        "file": filepath,
        "algorithm": algorithm,
        "hash": h,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(filepath + ".integrity", "w") as f:
        json.dump(baseline, f, indent=4)
    return "Integrity baseline created"
def verify_file_integrity(filepath: str):
    try:
        with open(filepath + ".integrity", "r") as f:
            baseline = json.load(f)

        current_hash = file_hash(filepath, baseline["algorithm"])

        return {
            "file": filepath,
            "algorithm": baseline["algorithm"],
            "original_hash": baseline["hash"],
            "current_hash": current_hash,
            "intact": baseline["hash"] == current_hash
        }
    except FileNotFoundError:
        return {"error": "Integrity baseline not found"}
    except Exception as e:
        return {"error": str(e)}
#74
def calculate_entropy(filepath: str) -> float:
    import math
    with open(filepath, "rb") as f:
        data = f.read()
    if not data:
        return 0.0
    freq = [0] * 256
    for byte in data:
        freq[byte] += 1
    entropy = 0.0
    data_len = len(data)
    for count in freq:
        if count == 0:
            continue
        p = count / data_len
        entropy -= p * math.log2(p)
    return round(entropy, 4)
def interpret_entropy(entropy: float) -> str:
    if entropy < 3:
        return "Very low entropy (plain text / structured)"
    elif entropy < 6:
        return "Moderate entropy (binary / executable)"
    elif entropy < 7.5:
        return "High entropy (compressed)"
    else:
        return "Very high entropy (encrypted / packed)"
#75
def load_binary_data(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
def byte_frequency_test(data: bytes):
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    return freq
def bit_balance_test(data: bytes):
    ones = 0
    zeros = 0
    for byte in data:
        bits = bin(byte)[2:].zfill(8)
        ones += bits.count("1")
        zeros += bits.count("0")
    total = ones + zeros
    ratio = ones / total if total else 0
    return {
        "ones": ones,
        "zeros": zeros,
        "ratio": round(ratio, 4)
    }
def runs_test(data: bytes):
    bits = "".join(bin(b)[2:].zfill(8) for b in data)
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            runs += 1
    return {
        "total_bits": len(bits),
        "runs": runs
    }
def chi_square_test(data: bytes):
    freq = byte_frequency_test(data)
    expected = len(data) / 256
    chi = 0.0
    for count in freq:
        chi += ((count - expected) ** 2) / expected if expected else 0
    return round(chi, 4)
def randomness_test_suite(filepath: str):
    data = load_binary_data(filepath)
    entropy = calculate_entropy(filepath)
    bit_balance = bit_balance_test(data)
    runs = runs_test(data)
    chi = chi_square_test(data)
    return {
        "entropy": entropy,
        "bit_balance": bit_balance,
        "runs": runs,
        "chi_square": chi
    }
#76






def tlsh_hash_bytes(data: bytes) -> str:
    import tlsh

    h = tlsh.hash(data)
    if not h or h == "TNULL":
        raise ValueError("Data too small or low entropy for TLSH")
    return h
def tlsh_hash_text(text: str) -> str:
    return tlsh_hash_bytes(text.encode())
def tlsh_hash_file(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return tlsh_hash_bytes(data)
def tlsh_compare(hash1: str, hash2: str) -> int:
    import tlsh

    score = tlsh.diff(hash1, hash2)
    return score
def generate_imphash(pe_path: str) -> str:
    import pefile
    import hashlib

    pe = pefile.PE(pe_path)
    imports = []

    if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        raise ValueError("No imports found in PE file")

    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        dll = entry.dll.decode(errors="ignore").lower()
        for imp in entry.imports:
            if imp.name:
                func = imp.name.decode(errors="ignore").lower()
                imports.append(f"{dll}.{func}")

    if not imports:
        raise ValueError("No valid imports for Imphash")

    imports.sort()
    joined = ",".join(imports)
    return hashlib.md5(joined.encode()).hexdigest()
def pe_hash_analyzer(pe_path: str) -> dict:
    import pefile
    import hashlib
    import os

    results = {}

    # File hashes
    with open(pe_path, "rb") as f:
        data = f.read()

    results["MD5"] = hashlib.md5(data).hexdigest()
    results["SHA1"] = hashlib.sha1(data).hexdigest()
    results["SHA256"] = hashlib.sha256(data).hexdigest()

    # PE parsing
    pe = pefile.PE(pe_path)

    # Imphash
    try:
        results["Imphash"] = pe.get_imphash()
    except Exception:
        results["Imphash"] = "N/A"

    # Sections
    sections = []
    for sec in pe.sections:
        sec_info = {
            "Name": sec.Name.decode(errors="ignore").strip("\x00"),
            "VirtualSize": sec.Misc_VirtualSize,
            "RawSize": sec.SizeOfRawData,
            "Entropy": round(sec.get_entropy(), 2)
        }
        sections.append(sec_info)

    results["Sections"] = sections

    # Suspicious indicators
    flags = []
    for sec in sections:
        if sec["Entropy"] > 7.2:
            flags.append(f"High entropy section: {sec['Name']}")

    if len(sections) < 3:
        flags.append("Very few sections (possible packing)")

    results["Warnings"] = flags

    return results
def base58check_decode(addr: str) -> bytes:
    num = 0
    for c in addr:
        if c not in BASE58_ALPHABET:
            raise ValueError("Invalid Base58 character")
        num = num * 58 + BASE58_ALPHABET.index(c)

    combined = num.to_bytes((num.bit_length() + 7) // 8, byteorder="big")

    # Restore leading zeros
    n_pad = len(addr) - len(addr.lstrip("1"))
    data = b"\x00" * n_pad + combined

    payload, checksum = data[:-4], data[-4:]

    import hashlib
    h = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]

    if h != checksum:
        raise ValueError("Invalid Base58 checksum")

    return payload
def bech32_verify(addr: str) -> bool:
    CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    addr = addr.lower()

    if not addr.startswith("bc1"):
        return False

    if not all(c in CHARSET or c.isalnum() for c in addr[3:]):
        return False

    # Bech32 checksum verification (minimal)
    return True  # format + prefix validated
def validate_bitcoin_address(addr: str) -> dict:
    addr = addr.strip()

    try:
        # Legacy & P2SH
        if addr.startswith("1") or addr.startswith("3"):
            payload = base58check_decode(addr)
            version = payload[0]
            addr_type = "P2PKH" if addr.startswith("1") else "P2SH"

            return {
                "Valid": True,
                "Type": addr_type,
                "Network": "Mainnet"
            }

        # Bech32 (SegWit)
        elif addr.lower().startswith("bc1"):
            if bech32_verify(addr):
                return {
                    "Valid": True,
                    "Type": "Bech32 (SegWit)",
                    "Network": "Mainnet"
                }

        return {"Valid": False}

    except Exception:
        return {"Valid": False}
def is_valid_eth_hex(address: str) -> bool:
    if not address.startswith("0x"):
        return False
    addr = address[2:]
    return len(addr) == 40 and all(c in "0123456789abcdefABCDEF" for c in addr)
def is_checksum_eth_address(address: str) -> bool:
    import hashlib

    addr = address.replace("0x", "")
    addr_lower = addr.lower()

    keccak = hashlib.new("sha3_256")
    keccak.update(addr_lower.encode())
    hash_hex = keccak.hexdigest()

    for i in range(40):
        char = addr[i]
        if char.isalpha():
            if (int(hash_hex[i], 16) >= 8 and char.upper() != char) or \
               (int(hash_hex[i], 16) < 8 and char.lower() != char):
                return False
    return True
def validate_ethereum_address(address: str) -> dict:
    address = address.strip()

    if not is_valid_eth_hex(address):
        return {"Valid": False}

    addr = address[2:]

    # All lower or all upper = valid (no checksum)
    if addr.islower() or addr.isupper():
        return {
            "Valid": True,
            "Checksum": False,
            "Type": "Non-checksummed"
        }

    # Mixed case → checksum required
    if is_checksum_eth_address(address):
        return {
            "Valid": True,
            "Checksum": True,
            "Type": "EIP-55 checksummed"
        }

    return {"Valid": False}
def keccak256_eth(data: bytes) -> str:
    import sha3  # from pysha3

    k = sha3.keccak_256()
    k.update(data)
    return k.hexdigest()
def merkle_hash(data: bytes, algo: str = "sha256") -> bytes:
    import hashlib
    h = hashlib.new(algo)
    h.update(data)
    return h.digest()
def build_merkle_tree(items: list, algo: str = "sha256") -> dict:
    if not items:
        raise ValueError("No items provided")

    # Convert leaves to hashes
    level = [merkle_hash(item.encode(), algo) for item in items]
    tree = [level]

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])  # duplicate last if odd

        next_level = []
        for i in range(0, len(level), 2):
            combined = level[i] + level[i + 1]
            next_level.append(merkle_hash(combined, algo))

        level = next_level
        tree.append(level)

    return {
        "Root": tree[-1][0].hex(),
        "Levels": [[h.hex() for h in lvl] for lvl in tree]
    }
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def base58_encode(data: bytes) -> str:
    num = int.from_bytes(data, "big")
    encoded = ""
    while num > 0:
        num, rem = divmod(num, 58)
        encoded = BASE58_ALPHABET[rem] + encoded

    # Preserve leading zeros
    pad = 0
    for b in data:
        if b == 0:
            pad += 1
        else:
            break

    return "1" * pad + encoded
def base58_decode(s: str) -> bytes:
    num = 0
    for c in s:
        if c not in BASE58_ALPHABET:
            raise ValueError("Invalid Base58 character")
        num = num * 58 + BASE58_ALPHABET.index(c)

    combined = num.to_bytes((num.bit_length() + 7) // 8, "big")

    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + combined
def double_sha256(data: bytes) -> bytes:
    import hashlib
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()
def wif_decode(wif: str) -> dict:
    raw = base58_decode(wif)
    payload, checksum = raw[:-4], raw[-4:]

    if double_sha256(payload)[:4] != checksum:
        raise ValueError("Invalid WIF checksum")

    version = payload[0]
    compressed = len(payload) == 34 and payload[-1] == 0x01

    key = payload[1:-1] if compressed else payload[1:]

    network = "Mainnet" if version == 0x80 else "Testnet"

    return {
        "PrivateKeyHex": key.hex(),
        "Compressed": compressed,
        "Network": network
    }
def wif_encode(private_key_hex: str, compressed=True, testnet=False) -> str:
    key = bytes.fromhex(private_key_hex)

    if len(key) != 32:
        raise ValueError("Private key must be 32 bytes")

    version = b"\xEF" if testnet else b"\x80"
    payload = version + key + (b"\x01" if compressed else b"")

    checksum = double_sha256(payload)[:4]
    return base58_encode(payload + checksum)
def steg_text_encode(cover_text: str, secret: str) -> str:
    ZWSP = "\u200b"   # 0
    ZWNJ = "\u200c"   # 1
    ZWJ  = "\u200d"   # end marker

    bits = "".join(format(ord(c), "08b") for c in secret)
    encoded = "".join(ZWNJ if b == "1" else ZWSP for b in bits)

    return cover_text + encoded + ZWJ
def steg_text_decode(stego_text: str) -> str:
    ZWSP = "\u200b"
    ZWNJ = "\u200c"
    ZWJ  = "\u200d"

    bits = ""
    for c in stego_text:
        if c == ZWSP:
            bits += "0"
        elif c == ZWNJ:
            bits += "1"
        elif c == ZWJ:
            break

    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return "".join(chr(int(b, 2)) for b in chars if len(b) == 8)
def _text_to_bits(text: str) -> str:
    bits = ''.join(format(ord(c), '08b') for c in text)
    return bits + '1111111111111110'  # end marker
def _bits_to_text(bits: str) -> str:
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)
def image_lsb_encode(input_img: str, output_img: str, secret: str):
    from PIL import Image

    img = Image.open(input_img).convert("RGB")
    pixels = img.load()

    bits = _text_to_bits(secret)
    idx = 0

    for y in range(img.height):
        for x in range(img.width):
            if idx >= len(bits):
                img.save(output_img)
                return

            r, g, b = pixels[x, y]
            r = (r & ~1) | int(bits[idx]); idx += 1
            if idx < len(bits):
                g = (g & ~1) | int(bits[idx]); idx += 1
            if idx < len(bits):
                b = (b & ~1) | int(bits[idx]); idx += 1

            pixels[x, y] = (r, g, b)

    raise ValueError("Image too small to hold secret")
def image_lsb_decode(img_path: str) -> str:
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    pixels = img.load()

    bits = ""
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = pixels[x, y]
            bits += str(r & 1)
            bits += str(g & 1)
            bits += str(b & 1)

            if bits.endswith("1111111111111110"):
                return _bits_to_text(bits[:-16])

    return _bits_to_text(bits)
def _audio_text_to_bits(text: str) -> str:
    bits = ''.join(format(ord(c), '08b') for c in text)
    return bits + '1111111111111110'  # end marker
def _audio_bits_to_text(bits: str) -> str:
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)
def audio_lsb_encode(input_wav: str, output_wav: str, secret: str):
    import wave
    import struct

    with wave.open(input_wav, 'rb') as wf:
        params = wf.getparams()
        frames = bytearray(wf.readframes(wf.getnframes()))

    bits = _audio_text_to_bits(secret)
    if len(bits) > len(frames):
        raise ValueError("Audio file too small to hold secret")

    bit_idx = 0
    for i in range(len(frames)):
        if bit_idx >= len(bits):
            break
        frames[i] = (frames[i] & 0xFE) | int(bits[bit_idx])
        bit_idx += 1

    with wave.open(output_wav, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(frames)
def audio_lsb_decode(stego_wav: str) -> str:
    import wave

    with wave.open(stego_wav, 'rb') as wf:
        frames = bytearray(wf.readframes(wf.getnframes()))

    bits = ""
    for b in frames:
        bits += str(b & 1)
        if bits.endswith("1111111111111110"):
            return _audio_bits_to_text(bits[:-16])

    return _audio_bits_to_text(bits)

def password_entropy(password: str) -> float:
    import math
    import re
    pool = 0

    if re.search(r"[a-z]", password):
        pool += 26
    if re.search(r"[A-Z]", password):
        pool += 26
    if re.search(r"[0-9]", password):
        pool += 10
    if re.search(r"[^a-zA-Z0-9]", password):
        pool += 32  # symbols approx

    if pool == 0:
        return 0.0

    return round(len(password) * math.log2(pool), 2)

def interpret_password_strength(entropy: float) -> str:
    if entropy < 28:
        return "VERY WEAK (trivial to crack)"
    elif entropy < 36:
        return "WEAK"
    elif entropy < 60:
        return "MODERATE"
    elif entropy < 80:
        return "STRONG"
    else:
        return "VERY STRONG"

def symmetric_key_strength(key_bytes: bytes, algorithm="AES"):
    bits = len(key_bytes) * 8

    if algorithm.upper() == "AES":
        if bits < 128:
            return f"WEAK ({bits}-bit key)"
        elif bits in (128, 192, 256):
            return f"SECURE ({bits}-bit AES key)"
        else:
            return f"NON-STANDARD ({bits}-bit AES key)"

    return f"{bits}-bit key"

def asymmetric_key_strength(bits: int, algorithm="RSA"):
    if algorithm.upper() == "RSA":
        if bits < 2048:
            return "WEAK (below modern security level)"
        elif bits < 3072:
            return "SECURE"
        else:
            return "VERY STRONG"

    if algorithm.upper() in ("ECC", "ECDSA", "ED25519", "X25519"):
        return "SECURE (elliptic-curve cryptography)"

    return "Unknown algorithm"

def estimate_password_entropy(password: str) -> float:
    import math
    import re
    pool = 0

    if re.search(r"[a-z]", password):
        pool += 26
    if re.search(r"[A-Z]", password):
        pool += 26
    if re.search(r"[0-9]", password):
        pool += 10
    if re.search(r"[^a-zA-Z0-9]", password):
        pool += 32  # symbols approx

    if pool == 0:
        return 0.0

    entropy = len(password) * math.log2(pool)
    return round(entropy, 2)

def password_weakness_checks(password: str):
    import re
    issues = []

    if len(password) < 8:
        issues.append("Too short (less than 8 characters)")

    if password.lower() == password:
        issues.append("No uppercase letters")

    if not re.search(r"[0-9]", password):
        issues.append("No digits")

    if not re.search(r"[^a-zA-Z0-9]", password):
        issues.append("No special characters")

    if password.isalnum():
        issues.append("Only letters and digits (predictable)")

    return issues

def password_strength_verdict(entropy: float) -> str:
    if entropy < 28:
        return "VERY WEAK"
    elif entropy < 36:
        return "WEAK"
    elif entropy < 60:
        return "MODERATE"
    elif entropy < 80:
        return "STRONG"
    else:
        return "VERY STRONG"

def password_strength_estimator(password: str):
    entropy = estimate_password_entropy(password)
    issues = password_weakness_checks(password)
    verdict = password_strength_verdict(entropy)

    return {
        "length": len(password),
        "entropy": entropy,
        "verdict": verdict,
        "issues": issues if issues else ["No obvious weaknesses detected"]
    }


def salt_random(length=16, encoding="hex"):
    import os, base64
    salt = os.urandom(length)
    return salt.hex() if encoding == "hex" else base64.b64encode(salt).decode()
def salt_fixed(value="fixed_salt"):
    return value
def salt_from_password(password: str, length=16):
    import hashlib
    return hashlib.sha256(password.encode()).digest()[:length].hex()
def salt_time_based(length=16):
    import time, hashlib
    return hashlib.sha256(str(time.time()).encode()).digest()[:length].hex()
def salt_user_specific(username: str, length=16):
    import hashlib
    return hashlib.sha256(username.encode()).digest()[:length].hex()
def salt_combined(password: str, username: str, length=16):
    import os, hashlib
    random_part = os.urandom(length)
    context = f"{password}:{username}".encode()
    return hashlib.sha256(random_part + context).digest().hex()
def salt_from_file(path: str, length=16):
    import hashlib
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).digest()[:length].hex()
def salt_session(length=16):
    import os
    return os.urandom(length).hex()

def generate_nonce(length: int = 12) -> bytes:
    import secrets
    """
    Generates a cryptographically secure nonce.
    Default: 12 bytes (recommended for AES-GCM, ChaCha20)
    """
    return secrets.token_bytes(length)

def nonce_hex(length: int = 12) -> str:
    import base64
    return generate_nonce(length).hex()

def nonce_base64(length: int = 12) -> str:
    import base64
    return base64.b64encode(generate_nonce(length)).decode()
def load_diceware_wordlist(path: str) -> dict:
    wordlist = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                key, word = parts
                wordlist[key] = word
    return wordlist
def roll_die() -> str:
    import secrets
    return str(secrets.randbelow(6) + 1)
def diceware_generate(wordlist: dict, words: int = 6, separator: str = " ") -> str:
    phrase = []

    for _ in range(words):
        roll = "".join(roll_die() for _ in range(5))
        phrase.append(wordlist[roll])

    return separator.join(phrase)


def coder_ops():
    while True:
        op = int(input("Enter which decoder or encoder you want to use (enter 999 for menu (or) 0 to exit):"))
        if op == 1:
            def caeser_ops():
                data = input("Enter text for caeser encrypt/decrypt:")
                while True:
                    try:
                        shift = int(input("Enter the shift (1-26):"))
                        if shift >= 1 and shift <= 26:
                            break
                        else:
                            print("Shift must be between 1 and 26. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter an integer between 1 and 26.")
                result_data = CAESAR_encrypt_and_decrypt(data,shift)
                print("The result is :",result_data)
                print(f"To decrypt, re-enter the encrypted text and use a shift value of {26 - shift} (i.e., the reverse shift). Note: encrypting with shift {shift} and then decrypting with shift {26 - shift} will return the original text.")
            caeser_ops()
        elif op == 2:
            def vigenere_en_de_coders_ops():
                while True:
                    op1 = int(input("Enter '1' to encode or '2' to decode with Vigenere cipher:"))
                    if op1 == 1:
                        data = input("Enter text:")
                        keyword = input("Enter the keyword for Vigenere cipher:")
                        encoded_text = vigenere_encoder(keyword, data)
                        print("The encoded text is:", encoded_text)
                        break
                    elif op1 == 2:
                        data = input("Enter text:")
                        keyword = input("Enter the keyword for Vigenere cipher:")
                        decrypted_text = vigenere_decoder(data, keyword)
                        print("The decoded text is:", decrypted_text)
                        break
                    else:
                        print("Entered wrong option!!, Try again.")
            vigenere_en_de_coders_ops()
        elif op == 3:
            def atbash_ops():
                data = input("Enter text for Atbash cipher:")
                result_data = atbash_cipher(data)
                print("The result is :",result_data)
            atbash_ops()
        elif op == 4:
            def becon_en_de_coders_ops():
                while True:
                    op1 = int(input("Enter '1' to encode or '2' to decode with Becon cipher:"))
                    if op1 == 1:
                        data = input("Enter text:")
                        encoded_text = becon_encoder(data)
                        print("The encoded text is:", encoded_text)
                        break
                    elif op1 == 2:
                        data = input("Enter text:")
                        decoded_text = becon_decoder(data)
                        print("The decoded text is:", decoded_text)
                        break
                    else:
                        print("Entered wrong option!!, Try again.")
            becon_en_de_coders_ops()
        elif op == 5:
            def bifid_en_de_coders_ops():
                while True:
                    op1 = int(input("Enter '1' to encrypt or '2' to decypt with Bifid cipher:"))
                    if op1 == 1:
                        data = input("Enter text:")
                        keyword = input("Enter the keyword for Bifid cipher:")
                        encoded_text = bifid_encrypt(data, keyword)
                        print("The encrypted text is:", encoded_text)
                        break
                    elif op1 == 2:
                        data = input("Enter text:")
                        keyword = input("Enter the keyword for Bifid cipher:")
                        decrypted_text = bifid_decrypt(data, keyword)
                        print("The decrypted text is:", decrypted_text)
                        break
                    else:
                        print("Entered wrong option!!, Try again.")
            bifid_en_de_coders_ops()
        elif op == 6:
            def affine_en_de_ops():
                while True:
                    op1 = int(input("Enter '1' to encrypt or '2' to decrypt using Affine Cipher: "))
                    text = input("Enter text: ")
                    a = int(input("Enter key 'a' (coprime with 26): "))
                    b = int(input("Enter key 'b': "))

                    try:
                        if op1 == 1:
                            print("Encrypted text:", affine_encrypt(text, a, b))
                            break
                        elif op1 == 2:
                            print("Decrypted text:", affine_decrypt(text, a, b))
                            break
                        else:
                            print("Invalid option. Try again.")
                    except ValueError as e:
                        print("Error:", e)

            affine_en_de_ops()
        elif op == 7:
            def A1Z26_ops():
                while True:
                    op1 = int(input("enter '1' to encode (or) enter '2' to decode:"))
                    if op1 == 1:
                        data = input("Enter the plain text to encode:")
                        encoded_text = A1Z26_encoder(data)
                        print("The encoded text is:",encoded_text)
                        break
                    elif op1 == 2:
                        data = input("Enter the encoded text to decode:")
                        decoded_text = A1Z26_decoder(data)
                        print("The decoded text is:",decoded_text)
                        break
                    else:
                        print("Entered wrong option!!,Try again")
            A1Z26_ops()
        elif op == 8:
            def rail_fence_ops():
                while True:
                    op1 = int(input("Enter '1' to encrypt or '2' to decrypt with Rail Fence Cipher: "))
                    text = input("Enter text: ")
                    rails = int(input("Enter number of rails: "))

                    if rails < 2:
                        print("Rails must be >= 2")
                        continue

                    if op1 == 1:
                        print("Encrypted text:", rail_fence_encrypt(text, rails))
                        break
                    elif op1 == 2:
                        print("Decrypted text:", rail_fence_decrypt(text, rails))
                        break
                    else:
                        print("Invalid option. Try again.")

            rail_fence_ops()
        elif op == 9:
            def substitution_ops():
                while True:
                    op1 = int(input("Enter '1' to encrypt or '2' to decrypt with Substitution Cipher: "))
                    text = input("Enter text: ")
                    key = input("Enter 26-letter substitution key: ")

                    try:
                        if op1 == 1:
                            print("Encrypted text:", substitution_encrypt(text, key))
                            break
                        elif op1 == 2:
                            print("Decrypted text:", substitution_decrypt(text, key))
                            break
                        else:
                            print("Invalid option. Try again.")
                    except ValueError as e:
                        print("Error:", e)

            substitution_ops()
        elif op == 10:
            def xor_ops():
                while True:
                    op1 = int(input("Enter '1' for XOR Encrypt/Decrypt or '2' for XOR Bruteforce: "))
                    if op1 == 1:
                        op2 = int(input("Enter '1' for XOR Encrypt or '2' for XOR Decrypt: "))  
                        if op2 == 1:
                            text = input("Enter text: ")
                            key = input("Enter XOR key: ")
                            print("Result:", xor_cipher(text, key))
                            break
                        elif op2 == 2:
                            text = input("Enter text: ")
                            key = input("Enter XOR key: ")
                            print("Result:", xor_decipher(text, key))
                            break
                        else:
                            print("Invalid option. Try again.")

                    elif op1 == 2:
                        results = xor_bruteforce(text)
                        print("Possible XOR decryptions:")
                        for r in results[:20]:
                            print(f"Key {r['key']:3} → {r['text']}")
                        break

                    else:
                        print("Invalid option. Try again.")

            xor_ops()
        elif op == 11:
            def rot_ops():
                while True:
                    op1 = int(input("Enter '1' for ROT13 or '2' for ROT47: "))
                    text = input("Enter text: ")

                    if op1 == 1:
                        print("Result:", CAESAR_encrypt_and_decrypt(text,13))
                        break
                    elif op1 == 2:
                        print("Result:", rot47_encoder_decoder(text))
                        break
                    else:
                        print("Invalid option. Try again.")
            rot_ops()
        elif op == 12:
            def CipherSaber2_ops():
                while True:
                    ops1=input("Enter '1' to encrypt or '2' to decrypt with CipherSaber2 cipher:")
                    if ops1 == '1':
                        data = input("Enter text:")
                        keyword = input("Enter the keyword for CipherSaber2 cipher:")
                        encrypt_text = ciphersaber2_encrypt(data, keyword)
                        print("The encoded text is:", encrypt_text)
                        break
                    elif ops1 == '2':
                        data = input("Enter text:")
                        keyword = input("Enter the keyword for CipherSaber2 cipher:")
                        decrypted_text = ciphersaber2_decrypt(data, keyword)
                        print("The decoded text is:", decrypted_text)
                        break
                    else:
                        print("Invalid option. Try again.")
            cipher_saber2_ops()
        elif op == 13:
            def RC2_ops():
                while True:
                    ops1=input("Enter '1' to encrypt or '2' to decrypt with RC2 cipher:")
                    if ops1 == '1':
                        data = input("Enter text:")
                        keyword = input("Enter the keyword for RC2 cipher:")
                        encrypted_text = rc2_encrypt(data, keyword)
                        print("The encrypted text is:", encrypted_text)
                        break
                    elif ops1 == '2':
                        data = input("Enter text:")
                        keyword = input("Enter the keyword for RC2 cipher:")
                        decrypted_text = rc2_decrypt(data, keyword)
                        print("The decrypted text is:", decrypted_text)
                        break
                    else:
                        print("Invalid option. Try again.")
            rc2_ops()
        elif op == 14:
            def RC4_ops():
                print("\nRC4 STREAM CIPHER")
                print("1. RC4")
                print("2. RC4-Drop")

                choice = input("Choose mode: ")
                text = input("Enter text: ").encode()
                key = input("Enter key: ").encode()

                if choice == "1":
                    result = rc4_crypt(text, key)

                elif choice == "2":
                    drop = int(input("Drop bytes (e.g. 768 / 1024): "))
                    result = rc4_drop_crypt(text, key, drop)

                else:
                    print("Invalid option.")
                    return

                print("Output (hex):", result.hex())
                print("Output (text):", result.decode(errors="ignore"))

            RC4_ops()

        elif op == 15:
            def AES_ops():
                while True:
                    print("\nAES MODES:")
                    print("1. ECB")
                    print("2. CBC")
                    print("3. CFB")
                    print("4. OFB")
                    print("5. CTR")
                    print("6. GCM")
                    print("7. CCM")
                    print("8. OCB")
                    print("9. XTS")
                    print("10. SIV")

                    mode_choice = input("Choose AES mode (1-10): ")

                    mode_map = {
                        "1": "ECB",
                        "2": "CBC",
                        "3": "CFB",
                        "4": "OFB",
                        "5": "CTR",
                        "6": "GCM",
                        "7": "CCM",
                        "8": "OCB",
                        "9": "XTS",
                        "10": "SIV"
                    }

                    if mode_choice not in mode_map:
                        print("Invalid AES mode. Try again.")
                        continue

                    mode = mode_map[mode_choice]

                    ops = input("Enter '1' to Encrypt or '2' to Decrypt: ")

                    if ops == '1':
                        data = input("Enter text: ")
                        password = input("Enter AES password/key: ")
                        key_size = int(input("Enter key size (128 / 192 / 256): ")) // 8

                        encrypted_text = aes_encrypt(
                            data,
                            password,
                            mode=mode,
                            key_size=key_size
                        )
                        print("Encrypted text:", encrypted_text)
                        break

                    elif ops == '2':
                        data = input("Enter encrypted text: ")
                        password = input("Enter AES password/key: ")
                        key_size = int(input("Enter key size (128 / 192 / 256): ")) // 8

                        decrypted_text = aes_decrypt(
                            data,
                            password,
                            mode=mode,
                            key_size=key_size
                        )
                        print("Decrypted text:", decrypted_text)
                        break

                    else:
                        print("Invalid option. Try again.")
            AES_ops()
        elif op == 16:
            def DES_ops():
                while True:
                    print("\nDES MODES:")
                    print("1. ECB")
                    print("2. CBC")
                    print("3. CFB")
                    print("4. OFB")
                    print("5. CTR")

                    mode_choice = input("Choose DES mode (1-5): ")

                    mode_map = {
                        "1": "ECB",
                        "2": "CBC",
                        "3": "CFB",
                        "4": "OFB",
                        "5": "CTR"
                    }

                    if mode_choice not in mode_map:
                        print("Invalid DES mode. Try again.")
                        continue

                    mode = mode_map[mode_choice]

                    ops = input("Enter '1' to Encrypt or '2' to Decrypt: ")

                    if ops == '1':
                        data = input("Enter text: ")
                        password = input("Enter DES password/key: ")

                        encrypted_text = des_encrypt(
                            data,
                            password,
                            mode=mode
                        )
                        print("Encrypted text:", encrypted_text)
                        break

                    elif ops == '2':
                        data = input("Enter encrypted text: ")
                        password = input("Enter DES password/key: ")

                        decrypted_text = des_decrypt(
                            data,
                            password,
                            mode=mode
                        )
                        print("Decrypted text:", decrypted_text)
                        break

                    else:
                        print("Invalid option. Try again.")

            DES_ops()
        elif op == 17:
            def TDES_ops():
                while True:
                    print("\nTRIPLE DES (3DES) MODES:")
                    print("1. ECB")
                    print("2. CBC")
                    print("3. CFB")
                    print("4. OFB")
                    print("5. CTR")
                    print("6. EAX")

                    mode_choice = input("Choose 3DES mode (1-6): ")

                    mode_map = {
                        "1": "ECB",
                        "2": "CBC",
                        "3": "CFB",
                        "4": "OFB",
                        "5": "CTR",
                        "6": "EAX"
                    }

                    if mode_choice not in mode_map:
                        print("Invalid 3DES mode. Try again.")
                        continue

                    mode = mode_map[mode_choice]

                    ops = input("Enter '1' to Encrypt or '2' to Decrypt: ")

                    if ops == '1':
                        data = input("Enter text: ")
                        password = input("Enter 3DES password/key: ")

                        encrypted_text = tdes_encrypt(
                            data,
                            password,
                            mode=mode
                        )
                        print("Encrypted text:", encrypted_text)
                        break

                    elif ops == '2':
                        data = input("Enter encrypted text: ")
                        password = input("Enter 3DES password/key: ")

                        decrypted_text = tdes_decrypt(
                            data,
                            password,
                            mode=mode
                        )
                        print("Decrypted text:", decrypted_text)
                        break

                    else:
                        print("Invalid option. Try again.")

            TDES_ops()
        elif op == 18:
            def Blowfish_ops():
                while True:
                    print("\nBLOWFISH MODES:")
                    print("1. ECB")
                    print("2. CBC")
                    print("3. CFB")
                    print("4. OFB")
                    print("5. CTR")

                    mode_map = {
                        "1": "ECB",
                        "2": "CBC",
                        "3": "CFB",
                        "4": "OFB",
                        "5": "CTR"
                    }

                    m = input("Choose mode (1-5): ")
                    if m not in mode_map:
                        print("Invalid mode.")
                        continue

                    mode = mode_map[m]
                    op2 = input("Enter '1' to Encrypt or '2' to Decrypt: ")

                    if op2 == '1':
                        text = input("Enter text: ")
                        key = input("Enter password: ")
                        print("Encrypted:", blowfish_encrypt(text, key, mode))
                        break

                    elif op2 == '2':
                        text = input("Enter encrypted text: ")
                        key = input("Enter password: ")
                        print("Decrypted:", blowfish_decrypt(text, key, mode))
                        break

                    else:
                        print("Invalid option.")

            Blowfish_ops()
        elif op == 19:
            def SM4_ops():
                while True:
                    print("\nSM4 MODES (gmssl):")
                    print("1. ECB")
                    print("2. CBC")

                    mode_choice = input("Choose SM4 mode (1-2): ")

                    mode_map = {
                        "1": "ECB",
                        "2": "CBC"
                    }

                    if mode_choice not in mode_map:
                        print("Invalid SM4 mode. Try again.")
                        continue

                    mode = mode_map[mode_choice]
                    ops = input("Enter '1' to Encrypt or '2' to Decrypt: ")

                    if ops == '1':
                        data = input("Enter text: ")
                        password = input("Enter SM4 password/key: ")
                        print("Encrypted:", sm4_encrypt(data, password, mode))
                        break

                    elif ops == '2':
                        data = input("Enter encrypted text: ")
                        password = input("Enter SM4 password/key: ")
                        print("Decrypted:", sm4_decrypt(data, password, mode))
                        break

                    else:
                        print("Invalid option. Try again.")

            SM4_ops()
        elif op == 20:
            def Enigma_ops():
                text = input("Enter text: ")
                r1 = input("Rotor 1 (I / II / III): ").upper()
                r2 = input("Rotor 2 (I / II / III): ").upper()
                r3 = input("Rotor 3 (I / II / III): ").upper()
                p1 = int(input("Rotor 1 position (0-25): "))
                p2 = int(input("Rotor 2 position (0-25): "))
                p3 = int(input("Rotor 3 position (0-25): "))
                result = enigma_cipher(text,rotor_order=(r1, r2, r3),rotor_positions=(p1, p2, p3))
                print("Result:", result)
            Enigma_ops()
        elif op == 21:
            def Bombe_ops():
                ciphertext = input("Enter Enigma ciphertext: ")
                crib = input("Enter crib (guessed plaintext): ")

                r1 = input("Rotor 1 (I/II/III): ").upper()
                r2 = input("Rotor 2 (I/II/III): ").upper()
                r3 = input("Rotor 3 (I/II/III): ").upper()

                results = bombe_find_settings(
                    ciphertext,
                    crib,
                    rotor_order=(r1, r2, r3)
                )

                if not results:
                    print("No settings found.")
                else:
                    for r in results:
                        print(r)

            Bombe_ops()
        elif op == 22:
            def Multiple_Bombe_ops():
                ciphertext = input("Enter Enigma ciphertext: ").replace(" ", "").upper()
                cribs_input = input("Enter cribs (comma separated): ")

                cribs = [c.strip() for c in cribs_input.split(",")]

                r1 = input("Rotor 1 (I/II/III): ").upper()
                r2 = input("Rotor 2 (I/II/III): ").upper()
                r3 = input("Rotor 3 (I/II/III): ").upper()

                results = multiple_bombe_find_settings(ciphertext=ciphertext,cribs=cribs,rotor_order=(r1, r2, r3))

                if not results:
                    print("No settings found.")
                else:
                    for r in results:
                        print("Possible settings:", r)

            Multiple_Bombe_ops()
        elif op == 23:
            def Typex_ops():
                print("\nTYPEX MACHINE")

                ops = input("Enter '1' to Encrypt or '2' to Decrypt: ")

                text = input("Enter text (A–Z only): ").upper()

                rotor_positions = []
                for i in range(5):
                    pos = int(input(f"Enter rotor {i+1} start (0–25): "))
                    rotor_positions.append(pos)

                if ops == '1':
                    output = typex_encrypt(text, rotors=5, positions=rotor_positions)
                    print("Encrypted text:", output)

                elif ops == '2':
                    output = typex_decrypt(text, rotors=5, positions=rotor_positions)
                    print("Decrypted text:", output)

                else:
                    print("Invalid option.")

            Typex_ops()
        elif op == 24:
            def Lorenz_ops():
                print("\nLORENZ SZ-40/42")

                ops = input("Enter '1' to Encrypt or '2' to Decrypt: ")
                text = input("Enter text: ").upper()

                print("Enter wheel patterns (binary, e.g. 101011)")
                wheels = []

                for i in range(5):   # simplified version
                    wheel = input(f"Wheel {i+1}: ")
                    wheels.append([int(b) for b in wheel])

                if ops == '1':
                    output = lorenz_encrypt(text, wheels)
                    print("Encrypted text:", output)

                elif ops == '2':
                    output = lorenz_decrypt(text, wheels)
                    print("Decrypted text:", output)

                else:
                    print("Invalid option.")

            Lorenz_ops()
        elif op == 25:
            def colossus_ops():
                print("\n[ COLOSSUS — STATISTICAL ANALYZER ]")
                print("⚠ Historical code-breaking simulator (NOT encryption)\n")

                ciphertext = input("Enter intercepted ciphertext: ").encode()

                print("\nEnter possible keystream guesses (comma-separated)")
                print("Example: ABC, KEY, SECRET")
                keys = input("Keys: ").split(",")

                results = colossus_analyze(ciphertext, keys)

                print("\nTop Results:")
                print("=" * 50)

                for r in results[:5]:
                    print(f"Key Guess : {r['key']}")
                    print(f"Score     : {r['score']}")
                    print("Decoded   :", r['decoded'])
                    print("-" * 50)

            colossus_ops()
        elif op == 26:
            def sigaba_cli():
                print("\n[ SIGABA SIMULATOR ]")
                print("⚠ Educational simulator — NOT real SIGABA\n")

                try:
                    rotor_count = int(input("Enter number of rotors (recommended 5): "))
                    if rotor_count < 3:
                        print("Using minimum of 3 rotors.")
                        rotor_count = 3
                except:
                    rotor_count = 5

                # Generate rotors
                rotors = [generate_rotor() for _ in range(rotor_count)]

                text = input("Enter text to encrypt/decrypt: ")

                result = sigaba_simulator(text, rotors)

                print("\nResult:")
                print(result)
            sigaba_cli()
        elif op == 27:
            import base64
            # encoder or decoder options
            def base64_en_de_coders_ops():
                while True:
                    op1 = int(input("enter '1' to encode (or) enter '2' to decode:"))
                    if op1 == 1:
                        data = input("enter text to encode to base64:")
                        encoded_text = base64_encoder(data)
                        print("The encoded text is:",encoded_text)
                        break
                    elif op1 == 2:
                        data = input("enter base64-encoded text:")
                        decoded_text = base64_decoder(data)
                        print("The decoded text is:",decoded_text)
                        break
                    else:
                        print("Entered wrong option!!,Try again")
            base64_en_de_coders_ops()
        elif op == 28:
            # encoder or decoder options
            def HEX_en_de_coders_ops():
                while True:
                    op1 = int(input("enter '1' to encode (or) enter '2' to decode:"))
                    if op1 == 1:
                        data = input("Enter the plain text to encode:")
                        encoded_text = hex_encoder(data)
                        print("The encoded text is:",encoded_text)
                        break
                    elif op1 == 2:
                        data = input("Enter the encoded text to decode:")
                        decoded_text = hex_decoder(data)
                        print("The decoded text is:",decoded_text)
                        break
                    else:
                        print("Entered wrong option!!,Try again")
            HEX_en_de_coders_ops()
        elif op == 29:
            # encoder or decoder options
            def url_en_de_coders_ops():
                while True:
                    op1 = int(input("enter '1' to encode (or) enter '2' to decode:"))
                    if op1 == 1:
                        data = input("enter the url text to encode:")
                        encoded_text = url_encoder(data)
                        print("The encoded text is:",encoded_text)
                        break
                    elif op1 == 2:
                        data = input("Enter the encoded url to decode:")
                        decoded_text = url_decoder(data)
                        print("The decoded text is:",decoded_text)
                        break
                    else:
                        print("Entered wrong option!!,Try again")
            url_en_de_coders_ops()
        elif op == 30:
            import hashlib
            def binary_en_de_coders_ops():
                while True:
                    op1 = int(input("enter '1' to encode (or) enter '2' to decode:"))
                    if op1 == 1:
                        data = input("Enter the text to encode with :")
                        binary = binary_encoder(data)
                        print("The binary encoded value is:", binary)
                        break
                    elif op1 == 2:
                        data = input("Enter the binary to decode :")
                        binary = binary_decoder(data)
                        print("The binary decoded value is:", binary)
                        break
                    else :
                        print("Entered wrong option!!,Try again")
            binary_en_de_coders_ops()
        elif op == 31:
            import hashlib
            def ascii_en_de_coders_ops():
                while True:
                    op1 = int(input("enter '1' to encode (or) enter '2' to decode:"))
                    if op1 == 1:
                        data = input("Enter the text to encode with ASCII:")
                        ascii_encoded = ascii_encoder(data)
                        print("The ASCII encoded value is:", ascii_encoded)
                        break
                    elif op1 == 2:
                        data = input("Enter the ASCII to decode :")
                        ascii_decoded = ascii_decoder(data)
                        print("The ASCII decoded value is:", ascii_decoded)
                        break
                    else :
                        print("Entered wrong option!!,Try again")
            ascii_en_de_coders_ops()
        elif op == 32:
            # encoder or decoder options
            def base32_en_de_coders_ops():
                while True:
                    op1 = int(input("enter '1' to encode (or) enter '2' to decode:"))
                    if op1 == 1:
                        data = input("enter text to encode to base32:")
                        encoded_text = base32_encoder(data)
                        print("The encoded text is:",encoded_text)
                        break
                    elif op1 == 2:
                        data = input("enter base32-encoded text:")
                        decoded_text = base32_decoder(data)
                        print("The decoded text is:",decoded_text)
                        break
                    else:
                        print("Entered wrong option!!,Try again")
            base32_en_de_coders_ops()
        elif op == 33:
            def morse_en_de_coders_ops():
                while True:
                    op1 = int(input("enter '1' to encode (or) enter '2' to decode:"))
                    if op1 == 1:
                        data = input("enter text to encode to morse code:")
                        encoded_text = morse_encoder(data)
                        print("The encoded text is:",encoded_text)
                        break
                    elif op1 == 2:
                        data = input("enter morse code to decode:")
                        decoded_text = morse_decoder(data)
                        print("The decoded text is:",decoded_text)
                        break
                    else:
                        print("Entered wrong option!!,Try again")
            morse_en_de_coders_ops()
        elif op == 34:
            def analyze_hash_ops():
                h = input("Enter hash value: ")
                result = analyze_hash(h)

                print("\n🔍 HASH ANALYSIS REPORT")
                print("----------------------")
                print("Format:", result["format"])
                print("Length:", result["length"])

                for item in result["analysis"]:
                    if "algorithm" in item:
                        print("\nAlgorithm:", item["algorithm"])
                        print("Security:", item["security"])
                        print("Crack Feasibility:", item["crack_feasibility"])
                    else:
                        print(item["note"])
            analyze_hash_ops()

        elif op == 35:
            def generate_all_hashes_cli():
                print("\n[ GENERATE ALL HASHES ]")
                data = input("Enter text to hash: ")

                results = generate_all_hashes(data)

                print("\nGenerated Hashes:")
                print("=" * 60)
                for algo, value in results.items():
                    print(f"{algo:<12}: {value}")
                print("=" * 60)

            generate_all_hashes_cli()
        elif op == 36:
            def MD_ops():
                text = input("Enter text to hash: ")
                #print("\n1. MD2")
                print("2. MD4")
                print("3. MD5")
                #print("4. MD6")

                choice = input("Choose hash: ")

                if choice == '1':
                    print(md2_hash(text))
                elif choice == '2':
                    print(md4_hash(text))
                elif choice == '3':
                    print(md5_hash(text))
                elif choice == '4':
                    print(md6_hash(text))
                else:
                    print("Invalid option")

            MD_ops()
        elif op == 37:
            def SHA_ops():
                text = input("Enter text to hash: ")

                print("""
                    1. SHA-0
                    2. SHA-1
                    3. SHA-224
                    4. SHA-256
                    5. SHA-384
                    6. SHA-512
                    7. SHA-512/224
                    8. SHA-512/256
                    9. SHA3-224
                    10. SHA3-256
                    11. SHA3-384
                    12. SHA3-512
                    13. SHAKE128
                    14. SHAKE256
                """)

                choice_map = {
                    "1": "sha0",
                    "2": "sha1",
                    "3": "sha2-224",
                    "4": "sha2-256",
                    "5": "sha2-384",
                    "6": "sha2-512",
                    "7": "sha2-512_224",
                    "8": "sha2-512_256",
                    "9": "sha3-224",
                    "10": "sha3-256",
                    "11": "sha3-384",
                    "12": "sha3-512",
                    "13": "shake128",
                    "14": "shake256",
                }

                ch = input("Choose option: ")
                algo = choice_map.get(ch)

                if algo:
                    print("Hash:", sha_family_hash(text, algo))
                else:
                    print("Invalid choice")

            SHA_ops()
        elif op == 38:
            def sponge_hash_ops():
                while True:
                    print("\nSM3 / KECCAK / SHAKE")
                    print("1. SM3")
                    print("2. Keccak")
                    print("3. SHAKE")

                    choice = input("Choose algorithm (1-3): ")

                    if choice not in ["1", "2", "3"]:
                        print("Invalid option")
                        continue

                    data = input("Enter text: ")

                    if choice == "1":
                        print("SM3:", sm3_hash(data))
                        break

                    elif choice == "2":
                        bits = int(input("Keccak bits (224/256/384/512): "))
                        print("Keccak:", keccak_hash(data, bits))
                        break

                    elif choice == "3":
                        bits = int(input("SHAKE bits (128/256): "))
                        print("SHAKE:", shake_hash(data, bits))
                        break

            sponge_hash_ops()
        elif op == 39:
            def Hash_ops():
                print("\nHASH FUNCTIONS")
                print("1. RIPEMD-160")
                print("2. HAS-160 (Demo)")
                print("3. Whirlpool")
                print("4. Snefru (Demo)")

                choice = input("Choose hash type: ")
                text = input("Enter text: ")

                if choice == "1":
                    print("RIPEMD-160:", ripemd160_hash(text))

                elif choice == "2":
                    print("HAS-160:", has160_demo(text))

                elif choice == "3":
                    print("Whirlpool:", whirlpool_hash(text))

                elif choice == "4":
                    bits = input("Enter output size (128/256): ")
                    print("Snefru:", snefru_demo(text, int(bits)))

                else:
                    print("Invalid choice.")

            Hash_ops()
        elif op == 40:
            def Blake_ops():
                print("\nBLAKE2 HASH FUNCTIONS")
                print("1. BLAKE2b")
                print("2. BLAKE2s")

                choice = input("Choose hash type: ")
                text = input("Enter text: ")

                if choice == "1":
                    size = int(input("Digest size in bytes (1–64): "))
                    print("BLAKE2b:", blake2b_hash(text, size))

                elif choice == "2":
                    size = int(input("Digest size in bytes (1–32): "))
                    print("BLAKE2s:", blake2s_hash(text, size))

                else:
                    print("Invalid choice.")

            Blake_ops()
        elif op == 41:
            def gost_streebog_ops():
                while True:
                    print("\nGOST / STREEBOG")
                    print("1. Streebog-256")
                    print("2. Streebog-512")

                    choice = input("Choose option (1-2): ")
                    if choice not in ["1", "2"]:
                        print("Invalid option")
                        continue

                    data = input("Enter text: ")

                    if choice == "1":
                        print("Streebog-256:", streebog_256(data))
                        break

                    elif choice == "2":
                        print("Streebog-512:", streebog_512(data))
                        break

            gost_streebog_ops()
        elif op == 42:
            def ssdeep_ops():
                while True:
                    print("\nSSDEEP / CTPH")
                    print("1. Generate SSDEEP hash")
                    print("2. Compare SSDEEP hashes")

                    choice = input("Choose option (1-2): ")

                    if choice not in ["1", "2"]:
                        print("Invalid option")
                        continue

                    if choice == "1":
                        data = input("Enter text or data: ")
                        print("SSDEEP Hash:")
                        print(ssdeep_hash(data))
                        break

                    elif choice == "2":
                        h1 = input("Enter first SSDEEP hash: ")
                        h2 = input("Enter second SSDEEP hash: ")

                        score = ssdeep_compare(h1, h2)
                        print(f"Similarity Score: {score}%")

                        if score >= 80:
                            print("🔴 Very High Similarity")
                        elif score >= 50:
                            print("🟠 Moderate Similarity")
                        elif score >= 20:
                            print("🟡 Low Similarity")
                        else:
                            print("🟢 No meaningful similarity")
                        break

            ssdeep_ops()
        elif op == 43:
            def hmac_ops():
                while True:
                    print("\nHMAC")
                    print("1. Generate HMAC")
                    print("2. Verify HMAC")

                    choice = input("Choose option (1-2): ")

                    if choice not in ["1", "2"]:
                        print("Invalid option")
                        continue

                    print("\nSupported Algorithms:")
                    print("MD5 | SHA1 | SHA256 | SHA384 | SHA512 | SHA3-256 | SHA3-512")

                    algo = input("Choose HMAC algorithm: ").upper()
                    message = input("Enter message: ")
                    key = input("Enter secret key: ")

                    if choice == "1":
                        mac = hmac_generate(message, key, algo)
                        print("\nGenerated HMAC:")
                        print(mac)
                        break

                    elif choice == "2":
                        given = input("Enter HMAC to verify: ")
                        if hmac_verify(message, key, algo, given):
                            print("✅ HMAC is VALID (message authentic)")
                        else:
                            print("❌ HMAC is INVALID (tampered or wrong key)")
                        break

            hmac_ops()
        elif op == 44:
            def bcrypt_ops():
                while True:
                    print("\nBCRYPT")
                    print("1. Hash password")
                    print("2. Compare password")
                    print("3. Parse bcrypt hash")

                    choice = input("Choose option (1-3): ")

                    if choice not in ["1", "2", "3"]:
                        print("Invalid option")
                        continue

                    if choice == "1":
                        pwd = input("Enter password: ")
                        rounds = int(input("Cost factor (10–14 recommended): "))
                        print("Bcrypt Hash:")
                        print(bcrypt_hash(pwd, rounds))
                        break

                    elif choice == "2":
                        pwd = input("Enter password: ")
                        hashed = input("Enter bcrypt hash: ")
                        if bcrypt_compare(pwd, hashed):
                            print("✅ Password MATCHES hash")
                        else:
                            print("❌ Password does NOT match")
                        break

                    elif choice == "3":
                        hashed = input("Enter bcrypt hash: ")
                        info = bcrypt_parse(hashed)
                        print("\nParsed bcrypt hash:")
                        for k, v in info.items():
                            print(f"{k:<15}: {v}")
                        break

            bcrypt_ops()
        elif op == 45:
            def scrypt_ops():
                while True:
                    print("\nSCRYPT")
                    print("1. Hash password")
                    print("2. Compare password")
                    print("3. Parse scrypt hash")

                    choice = input("Choose option (1-3): ")

                    if choice not in ["1", "2", "3"]:
                        print("Invalid option")
                        continue

                    if choice == "1":
                        pwd = input("Enter password: ")
                        print("Recommended values: N=16384 r=8 p=1")
                        n = int(input("Enter N (CPU/memory cost): "))
                        r = int(input("Enter r (block size): "))
                        p = int(input("Enter p (parallelization): "))

                        result = scrypt_hash(pwd, n=n, r=r, p=p)
                        print("\nGenerated scrypt hash:")
                        for k, v in result.items():
                            print(f"{k:<10}: {v}")
                        break

                    elif choice == "2":
                        pwd = input("Enter password: ")
                        print("Paste stored scrypt parameters")

                        stored = {
                            "algorithm": "scrypt",
                            "n": int(input("N: ")),
                            "r": int(input("r: ")),
                            "p": int(input("p: ")),
                            "salt": input("salt (base64): "),
                            "hash": input("hash (base64): ")
                        }

                        if scrypt_compare(pwd, stored):
                            print("✅ Password MATCHES hash")
                        else:
                            print("❌ Password does NOT match")
                        break

                    elif choice == "3":
                        stored = {
                            "algorithm": "scrypt",
                            "n": int(input("N: ")),
                            "r": int(input("r: ")),
                            "p": int(input("p: ")),
                            "salt": input("salt (base64): "),
                            "hash": input("hash (base64): ")
                        }

                        info = scrypt_parse(stored)
                        print("\nParsed scrypt hash:")
                        for k, v in info.items():
                            print(f"{k:<15}: {v}")
                        break

            scrypt_ops()
        elif op == 46:
            def pbkdf2_ops():
                while True:
                    print("\nPBKDF2 KEY DERIVATION")
                    print("1. Derive key")
                    print("2. Verify key")

                    choice = input("Choose option (1-2): ")

                    if choice not in ["1", "2"]:
                        print("Invalid option")
                        continue

                    print("\nSupported hashes:")
                    print("SHA1 | SHA256 | SHA384 | SHA512")

                    if choice == "1":
                        password = input("Enter password: ")
                        hash_name = input("Hash algorithm: ").lower()
                        iterations = int(input("Iterations (recommended ≥100000): "))
                        dklen = int(input("Key length (bytes): "))

                        result = pbkdf2_derive_key(
                            password,
                            iterations=iterations,
                            dklen=dklen,
                            hash_name=hash_name
                        )

                        print("\nDerived PBKDF2 key:")
                        for k, v in result.items():
                            print(f"{k:<15}: {v}")
                        break

                    elif choice == "2":
                        password = input("Enter password: ")
                        stored = {
                            "algorithm": "PBKDF2",
                            "hash": input("Hash: "),
                            "iterations": int(input("Iterations: ")),
                            "salt": input("Salt (base64): "),
                            "derived_key": input("Derived key (base64): ")
                        }

                        if pbkdf2_verify(password, stored):
                            print("✅ Password produces SAME derived key")
                        else:
                            print("❌ Password does NOT match derived key")
                        break

            pbkdf2_ops()
        elif op == 47:
            def evp_ops():
                while True:
                    print("\nEVP KEY DERIVATION (OpenSSL Compatible)")
                    print("1. Derive EVP key")

                    choice = input("Choose option (1): ")

                    if choice != "1":
                        print("Invalid option")
                        continue

                    password = input("Enter password: ")

                    print("\nSupported hashes:")
                    print("MD5 | SHA1 | SHA256")

                    hash_name = input("Hash algorithm: ").lower()

                    key_len = int(input("Key length (bytes, e.g. 16/24/32): "))
                    iv_len = int(input("IV length (bytes, e.g. 16): "))

                    result = derive_evp_key(
                        password,
                        key_len=key_len,
                        iv_len=iv_len,
                        hash_name=hash_name
                    )

                    print("\nDerived EVP Key:")
                    print("=" * 60)
                    for k, v in result.items():
                        print(f"{k:<15}: {v}")
                    print("=" * 60)

                    break

            evp_ops()
        elif op == 48:
            def prng_ops():
                while True:
                    print("\nPSEUDO-RANDOM NUMBER GENERATOR")
                    print("1. Python PRNG (Mersenne Twister)")
                    print("2. Linear Congruential Generator (LCG)")
                    print("3. XORShift")
                    print("4. Secure RNG (CSPRNG)")

                    choice = input("Choose PRNG (1-4): ")

                    if choice not in ["1", "2", "3", "4"]:
                        print("Invalid option")
                        continue

                    count = int(input("How many random numbers?: "))

                    if choice == "4":
                        nums = csprng(count)
                    else:
                        seed = int(input("Enter seed value: "))

                        if choice == "1":
                            nums = prng_python(seed, count)
                        elif choice == "2":
                            nums = prng_lcg(seed, count)
                        elif choice == "3":
                            nums = prng_xorshift(seed, count)

                    print("\nGenerated Numbers:")
                    print("=" * 60)
                    for i, n in enumerate(nums, 1):
                        print(f"{i:02}: {n}")
                    print("=" * 60)
                    break

            prng_ops()
        elif op == 49:
            def jwt_ops():
                while True:
                    print("\nJWT OPERATIONS")
                    print("1. Sign JWT")
                    print("2. Verify JWT")
                    print("3. Decode JWT (No Verify)")

                    choice = input("Choose option (1-3): ")

                    if choice not in ["1", "2", "3"]:
                        print("Invalid option")
                        continue

                    if choice == "1":
                        print("\nSupported Algorithms: HS256 | HS384 | HS512")
                        algo = input("Algorithm: ").upper()
                        secret = input("Secret key: ")

                        payload = {}
                        print("Enter payload key=value (empty line to finish)")
                        while True:
                            line = input("> ")
                            if not line:
                                break
                            k, v = line.split("=", 1)
                            payload[k.strip()] = v.strip()

                        exp = input("Expiration seconds (optional): ")
                        exp = int(exp) if exp else None

                        token = jwt_sign(payload, secret, algo, exp)
                        print("\nJWT Token:")
                        print(token)
                        break

                    elif choice == "2":
                        token = input("Enter JWT token: ")
                        secret = input("Secret key: ")
                        algo = input("Algorithm (HS256 default): ") or "HS256"

                        try:
                            data = jwt_verify(token, secret, algo)
                            print("\n✅ JWT is VALID")
                            print("Payload:")
                            for k, v in data.items():
                                print(f"{k}: {v}")
                        except Exception as e:
                            print("\n❌ JWT verification FAILED")
                            print("Error:", str(e))
                        break

                    elif choice == "3":
                        token = input("Enter JWT token: ")
                        data = jwt_decode_no_verify(token)

                        print("\nDecoded JWT (NO verification):")
                        for k, v in data.items():
                            print(f"{k}: {v}")
                        break

            jwt_ops()
        elif op == 50:
            def ctx1_ops():
                while True:
                    print("\nCITRIX CTX1")
                    print("1. Encode")
                    print("2. Decode")

                    choice = input("Choose option (1-2): ")

                    if choice not in ["1", "2"]:
                        print("Invalid option")
                        continue

                    if choice == "1":
                        data = input("Enter text to encode: ")
                        result = ctx1_encode(data)
                        print("\nCTX1 Encoded:")
                        print(result)
                        break

                    elif choice == "2":
                        token = input("Enter CTX1 token: ")
                        try:
                            result = ctx1_decode(token)
                            print("\nCTX1 Decoded:")
                            print(result)
                        except Exception as e:
                            print("\n❌ Decode failed")
                            print("Error:", str(e))
                        break
        
            ctx1_ops()
        elif op == 51:
            def fletcher_ops():
                print("\nFLETCHER CHECKSUM")
                print("1. Fletcher-8")
                print("2. Fletcher-16")
                print("3. Fletcher-32")
                print("4. Fletcher-64")

                choice = input("Choose option (1-4): ")
                data = input("Enter text: ").encode()

                if choice == "1":
                    print("Fletcher-8 :", hex(fletcher8(data)))
                elif choice == "2":
                    print("Fletcher-16:", hex(fletcher16(data)))
                elif choice == "3":
                    print("Fletcher-32:", hex(fletcher32(data)))
                elif choice == "4":
                    print("Fletcher-64:", hex(fletcher64(data)))
                else:
                    print("Invalid option")

            fletcher_ops()
        elif op == 52:
            def adler32_ops():
                print("\nADLER-32 CHECKSUM")

                data = input("Enter text: ").encode()
                checksum = adler32_checksum(data)

                print("Adler-32 Checksum:", hex(checksum))

            adler32_ops()
        elif op == 53:
            def luhn_ops():
                print("\nLUHN CHECKSUM")
                print("1. Validate number")
                print("2. Generate check digit")

                choice = input("Choose option (1-2): ")

                if choice == "1":
                    number = input("Enter full number: ")
                    if luhn_validate(number):
                        print("✅ Valid Luhn number")
                    else:
                        print("❌ Invalid Luhn number")

                elif choice == "2":
                    base = input("Enter number without check digit: ")
                    try:
                        full = luhn_generate(base)
                        print("Generated Luhn number:", full)
                    except Exception as e:
                        print("Error:", str(e))
                else:
                    print("Invalid option")

            luhn_ops()
        elif op == 54:
            def crc_ops():
                print("\nCRC CHECKSUM")
                print("1. CRC-8")
                print("2. CRC-16")
                print("3. CRC-32")

                choice = input("Choose option (1-3): ")
                data = input("Enter text: ").encode()

                if choice == "1":
                    print("CRC-8 :", hex(crc8(data)))
                elif choice == "2":
                    print("CRC-16:", hex(crc16(data)))
                elif choice == "3":
                    print("CRC-32:", hex(crc32(data)))
                else:
                    print("Invalid option")

            crc_ops()
        elif op == 55:
            def tcp_ip_checksum_ops():
                print("\nTCP/IP CHECKSUM")

                data = input("Enter text: ").encode()
                checksum = tcp_ip_checksum(data)

                print("TCP/IP Checksum:", hex(checksum))

            tcp_ip_checksum_ops()
        elif op == 56:
            
            def RSA_ops():
                print("1. Generate Keys")
                print("2. Encrypt")
                print("3. Decrypt")
                ch = input("Choose option: ")

                if ch == "1":
                    pub, priv = rsa_generate_keys()
                    print("Public Key:\n", pub)
                    print("Private Key:\n", priv)

                elif ch == "2":
                    text = input("Enter plaintext: ")
                    pub = input("Paste public key:\n")
                    print("Encrypted:", rsa_encrypt(text, pub))

                elif ch == "3":
                    cipher = input("Enter ciphertext (Base64): ")
                    priv = input("Paste private key:\n")
                    print("Decrypted:", rsa_decrypt(cipher, priv))

                else:
                    print("Invalid option")

            RSA_ops()
        elif op == 57:
            def RSA_sign_verify_ops():
                print("1. Sign")
                print("2. Verify")
                ch = input("Choose option: ")

                if ch == "1":
                    msg = input("Enter message: ")
                    priv = input("Paste PRIVATE key:\n")
                    print("Signature:", rsa_sign(msg, priv))

                elif ch == "2":
                    msg = input("Enter message: ")
                    sig = input("Enter signature (Base64): ")
                    pub = input("Paste PUBLIC key:\n")
                    result = rsa_verify(msg, sig, pub)
                    print("VALID SIGNATURE" if result else "INVALID SIGNATURE")

                else:
                    print("Invalid option")

            RSA_sign_verify_ops()
        elif op == 58:
            def DH_ops():
                print("Performing real Diffie–Hellman key exchange (2048-bit)")
                p, g = dh_generate_parameters()

                a = dh_generate_private_key(p)
                b = dh_generate_private_key(p)

                A = dh_generate_public_key(g, a, p)
                B = dh_generate_public_key(g, b, p)

                secret = dh_compute_shared_secret(B, a, p)
                key = dh_derive_key(secret)

                print("Shared secret established")
                print("Derived AES key:", key.hex())

            DH_ops()
        elif op == 59:
            def ECDH_ops():
                print("Elliptic Curve Diffie-Hellman (P-256)")

                alice_pub, alice_priv = ecdh_generate_keypair()
                bob_pub, bob_priv = ecdh_generate_keypair()

                key = ecdh_shared_secret(alice_priv, bob_pub)

                print("Shared secret (AES key):", key.hex())

            ECDH_ops()
        elif op == 60:
            def DSA_ops():
                print("1. Generate Keys")
                print("2. Sign")
                print("3. Verify")
                ch = input("Choose option: ")

                if ch == "1":
                    pub, priv = dsa_generate_keys()
                    print("Public Key:\n", pub)
                    print("Private Key:\n", priv)

                elif ch == "2":
                    msg = input("Enter message: ")
                    priv = input("Paste PRIVATE key:\n")
                    print("Signature:", dsa_sign(msg, priv))

                elif ch == "3":
                    msg = input("Enter message: ")
                    sig = input("Enter signature (Base64): ")
                    pub = input("Paste PUBLIC key:\n")
                    print(
                        "VALID SIGNATURE"
                        if dsa_verify(msg, sig, pub)
                        else "INVALID SIGNATURE"
                    )

                else:
                    print("Invalid option")

            DSA_ops()
        elif op == 61:
            def ECDSA_ops():
                print("1. Generate Keys")
                print("2. Sign")
                print("3. Verify")
                ch = input("Choose option: ")

                if ch == "1":
                    pub, priv = ecdsa_generate_keys()
                    print("Public Key:\n", pub)
                    print("Private Key:\n", priv)

                elif ch == "2":
                    msg = input("Enter message: ")
                    priv = input("Paste PRIVATE key:\n")
                    print("Signature:", ecdsa_sign(msg, priv))

                elif ch == "3":
                    msg = input("Enter message: ")
                    sig = input("Enter signature (Base64): ")
                    pub = input("Paste PUBLIC key:\n")
                    print(
                        "VALID SIGNATURE"
                        if ecdsa_verify(msg, sig, pub)
                        else "INVALID SIGNATURE"
                    )

                else:
                    print("Invalid option")

            ECDSA_ops()
        elif op == 62:
            def Ed25519_ops():
                print("1. Generate Keys")
                print("2. Sign")
                print("3. Verify")
                ch = input("Choose option: ")

                if ch == "1":
                    pub, priv = ed25519_generate_keys()
                    print("Public Key:\n", pub)
                    print("Private Key:\n", priv)

                elif ch == "2":
                    msg = input("Enter message: ")
                    priv = input("Paste PRIVATE key:\n")
                    print("Signature:", ed25519_sign(msg, priv))

                elif ch == "3":
                    msg = input("Enter message: ")
                    sig = input("Enter signature (Base64): ")
                    pub = input("Paste PUBLIC key:\n")
                    print(
                        "VALID SIGNATURE"
                        if ed25519_verify(msg, sig, pub)
                        else "INVALID SIGNATURE")
                else:
                    print("Invalid option")

                Ed25519_ops()
        elif op == 63:
            def X25519_ops():
                print("X25519 Key Exchange")

                alice_pub, alice_priv = x25519_generate_keypair()
                bob_pub, bob_priv = x25519_generate_keypair()

                key = x25519_shared_secret(alice_priv, bob_pub)

                print("Shared secret (AES key):", key.hex())

            X25519_ops()
        elif op == 64:
            def x509_ops():
                print("\nX.509 CERTIFICATE PARSER")

                path = input("Enter certificate file path (.pem / .der): ")

                try:
                    info = parse_x509_certificate(path)

                    print("\nCERTIFICATE DETAILS")
                    print("=" * 60)
                    for k, v in info.items():
                        if k != "Extensions":
                            print(f"{k}: {v}")

                    print("\nExtensions:")
                    for e in info["Extensions"]:
                        print(" -", e)

                    print("=" * 60)

                except Exception as e:
                    print("\n❌ Failed to parse certificate")
                    print("Error:", str(e))

            x509_ops()
        elif op == 65:
            def tls_cert_ops():
                print("\nTLS CERTIFICATE ANALYZER")

                host = input("Enter domain (example.com): ")
                port = input("Enter port (default 443): ")
                port = int(port) if port else 443

                try:
                    info = analyze_tls_certificate(host, port)

                    print("\nTLS CERTIFICATE DETAILS")
                    print("=" * 60)
                    for k, v in info.items():
                        if k not in ["Extensions", "Warnings"]:
                            print(f"{k}: {v}")

                    print("\nExtensions:")
                    for e in info["Extensions"]:
                        print(" -", e)

                    if info["Warnings"]:
                        print("\n⚠ WARNINGS:")
                        for w in info["Warnings"]:
                            print(" -", w)
                    else:
                        print("\n✔ No obvious certificate issues detected")

                    print("=" * 60)

                except Exception as e:
                    print("\n❌ Failed to analyze TLS certificate")
                    print("Error:", str(e))

            tls_cert_ops()
        elif op == 66:
            def pem_der_ops():
                print("\nPEM ↔ DER CONVERTER")
                print("1. PEM → DER")
                print("2. DER → PEM")

                choice = input("Choose option (1-2): ")

                if choice == "1":
                    inp = input("Enter PEM file path: ")
                    out = input("Enter output DER file path: ")
                    try:
                        pem_to_der(inp, out)
                        print("✔ Converted PEM → DER successfully")
                    except Exception as e:
                        print("❌ Conversion failed:", e)

                elif choice == "2":
                    inp = input("Enter DER file path: ")
                    out = input("Enter output PEM file path: ")
                    try:
                        der_to_pem(inp, out)
                        print("✔ Converted DER → PEM successfully")
                    except Exception as e:
                        print("❌ Conversion failed:", e)
                else:
                    print("Invalid option")

            pem_der_ops()
        elif op == 67:
            def public_key_extractor_ops():
                print("\nPUBLIC KEY EXTRACTOR")
                print("1. Extract from Certificate (PEM/DER)")
                print("2. Extract from Private Key (PEM)")

                choice = input("Choose option (1-2): ")

                if choice == "1":
                    inp = input("Enter certificate file path: ")
                    out = input("Enter output public key PEM path: ")
                    try:
                        extract_public_key_from_cert(inp, out)
                        print("✔ Public key extracted from certificate")
                    except Exception as e:
                        print("❌ Failed:", e)
                elif choice == "2":
                    inp = input("Enter private key PEM path: ")
                    pwd = input("Private key password (leave empty if none): ")
                    out = input("Enter output public key PEM path: ")
                    try:
                        extract_public_key_from_private_key(inp, out, pwd if pwd else None)
                        print("✔ Public key extracted from private key")
                    except Exception as e:
                        print("❌ Failed:", e)
                else:
                    print("Invalid option")
            public_key_extractor_ops()
        elif op == 68:
            def fingerprint_ops():
                print("\nFINGERPRINT GENERATOR")
                print("1. Certificate Fingerprint")
                print("2. Public Key Fingerprint")
                print("3. File Fingerprint")
                print("4. Text Fingerprint")
                algo = input("Hash algorithm (md5 / sha1 / sha256 / sha384 / sha512): ").lower()
                choice = input("Choose option (1-4): ")
                try:
                    if choice == "1":
                        path = input("Certificate file path: ")
                        print("Fingerprint:", fingerprint_certificate(path, algo))
                    elif choice == "2":
                        path = input("Public key PEM path: ")
                        print("Fingerprint:", fingerprint_public_key(path, algo))
                    elif choice == "3":
                        path = input("File path: ")
                        print("Fingerprint:", fingerprint_file(path, algo))
                    elif choice == "4":
                        text = input("Enter text: ")
                        print("Fingerprint:", generate_fingerprint(text.encode(), algo))
                    else:
                        print("Invalid option")
                except Exception as e:
                    print("❌ Error:", str(e))
            fingerprint_ops()
        elif op == 69:
            def file_hash_ops():
                path = input("Enter file path: ")
                print("""
                    1. MD5
                    2. SHA1
                    3. SHA256
                    4. SHA512
                    5. SHA3-256
                    6. BLAKE2b
                    7. ALL (Forensics)
                """)
                choice = input("Choose option: ")
                algo_map = {
                    "1": "MD5",
                    "2": "SHA1",
                    "3": "SHA256",
                    "4": "SHA512",
                    "5": "SHA3_256",
                    "6": "BLAKE2B"
                }
                if choice in algo_map:
                    print("Hash:", file_hash(path, algo_map[choice]))
                elif choice == "7":
                    hashes = file_multi_hash(path)
                    for k, v in hashes.items():
                        print(f"{k}: {v}")
                else:
                    print("Invalid option")
            file_hash_ops()
        elif op == 70:
            def directory_hash_ops():
                path = input("Enter directory path: ")
                print("""
                    1. MD5
                    2. SHA1
                    3. SHA256
                    4. SHA512
                    5. SHA3-256
                    6. BLAKE2b
                    7. ALL (Forensics)
                """)
                choice = input("Choose option: ")
                algo_map = {
                    "1": "MD5",
                    "2": "SHA1",
                    "3": "SHA256",
                    "4": "SHA512",
                    "5": "SHA3_256",
                    "6": "BLAKE2B",
                }
                if choice in algo_map:
                    print("Directory Hash:", directory_hash(path, algo_map[choice]))
                elif choice == "7":
                    hashes = directory_multi_hash(path)
                    for k, v in hashes.items():
                        print(f"{k}: {v}")
                else:
                    print("Invalid option")
            directory_hash_ops()
        elif op == 71:
            def compare_file_hash_ops():
                file1 = input("Enter first file path: ")
                file2 = input("Enter second file path: ")
                print("""
                    1. MD5
                    2. SHA1
                    3. SHA256
                    4. SHA512
                    5. SHA3-256
                    6. BLAKE2b
                """)
                choice = input("Choose option: ")
                algo_map = {
                    "1": "MD5",
                    "2": "SHA1",
                    "3": "SHA256",
                    "4": "SHA512",
                    "5": "SHA3_256",
                    "6": "BLAKE2B",
                }
                if choice in algo_map:
                    result = compare_file_hashes(file1, file2, algo_map[choice])
                    print("MATCH" if result["match"] else "NO MATCH")
                    print(result)
                else:
                    print("Invalid option")
            compare_file_hash_ops()
        elif op == 72:
            def file_encrypt_decrypt_ops():
                print("1. Encrypt File (AES)")
                print("2. Decrypt File (AES)")
                ch = input("Choose option: ")
                if ch == "1":
                    infile = input("Enter input file path: ")
                    outfile = input("Enter output encrypted file path: ")
                    password = input("Enter password: ")
                    key = derive_key(password)
                    file_encrypt_aes(infile, outfile, key)
                    print("File encrypted successfully")
                elif ch == "2":
                    infile = input("Enter encrypted file path: ")
                    outfile = input("Enter output decrypted file path: ")
                    password = input("Enter password: ")
                    key = derive_key(password)
                    file_decrypt_aes(infile, outfile, key)
                    print("File decrypted successfully")
                else:
                    print("Invalid option")
            file_encrypt_decrypt_ops()
        elif op == 73:
            def file_integrity_ops():
                print("1. Create Integrity Baseline")
                print("2. Verify File Integrity")
                ch = input("Choose option: ")
                if ch == "1":
                    path = input("Enter file path: ")
                    algo = input("Hash algorithm (default SHA256): ") or "SHA256"
                    print(create_integrity_baseline(path, algo))
                elif ch == "2":
                    path = input("Enter file path: ")
                    result = verify_file_integrity(path)
                    if "error" in result:
                        print(result["error"])
                    else:
                        print("FILE INTACT" if result["intact"] else "FILE MODIFIED")
                        print(result)
                else:
                    print("Invalid option")
            file_integrity_ops()
        elif op == 74:
            def entropy_analyzer_ops():
                path = input("Enter file path: ")
                try:
                    entropy = calculate_entropy(path)
                    print("\n🧠 WHY ENTROPY MATTERS IN FORENSICS")
                    print("---------------------------------")
                    print("Entropy Range     Interpretation")
                    print("---------------------------------")
                    print("0.0 – 3.0         Plain text / simple data")
                    print("3.0 – 6.0         Structured binary")
                    print("6.0 – 7.5         Compressed")
                    print("7.5 – 8.0         Encrypted / packed")
                    print("---------------------------------")

                    print(f"\nCalculated Entropy: {entropy}")
                    print("Analysis:", interpret_entropy(entropy))
                except FileNotFoundError:
                    print("File not found")
                except PermissionError:
                    print("Permission denied")
            entropy_analyzer_ops()

        elif op == 75:
            def randomness_test_ops():
                path = input("Enter file path: ")
                try:
                    results = randomness_test_suite(path)
                    print("Entropy:", results["entropy"])
                    print("Bit balance:", results["bit_balance"])
                    print("Runs:", results["runs"])
                    print("Chi-square:", results["chi_square"])
                except FileNotFoundError:
                    print("File not found")
                except PermissionError:
                    print("Permission denied")
            randomness_test_ops()
          
        elif op == 76:
            def key_strength_ops():
                print("1. Password / Passphrase")
                print("2. Symmetric Key")
                print("3. Asymmetric Key")
                ch = input("Choose option: ")

                if ch == "1":
                    pwd = input("Enter password: ")
                    ent = password_entropy(pwd)
                    print("Estimated entropy:", ent)
                    print("Strength:", interpret_password_strength(ent))

                elif ch == "2":
                    key = input("Enter key (hex or raw): ")
                    algo = input("Algorithm (AES): ") or "AES"

                    try:
                        key_bytes = bytes.fromhex(key)
                    except ValueError:
                        key_bytes = key.encode()

                    print("Strength:", symmetric_key_strength(key_bytes, algo))

                elif ch == "3":
                    bits = int(input("Enter key size in bits: "))
                    algo = input("Algorithm (RSA/ECC): ") or "RSA"
                    print("Strength:", asymmetric_key_strength(bits, algo))

                else:
                    print("Invalid option")

            key_strength_ops()


        elif op == 77:
            def tlsh_ops():
                print("\nTLSH FUZZY HASHING")
                print("1. Hash Text")
                print("2. Hash File")
                print("3. Compare TLSH Hashes")
                choice = input("Choose option (1-3): ")
                try:
                    if choice == "1":
                        text = input("Enter text: ")
                        h = tlsh_hash_text(text)
                        print("TLSH Hash:", h)
                    elif choice == "2":
                        path = input("Enter file path: ")
                        h = tlsh_hash_file(path)
                        print("TLSH Hash:", h)
                    elif choice == "3":
                        h1 = input("Enter TLSH hash 1: ")
                        h2 = input("Enter TLSH hash 2: ")
                        score = tlsh_compare(h1, h2)
                        print("TLSH Distance Score:", score)
                        if score == 0:
                            print("✔ Identical")
                        elif score < 50:
                            print("✔ Very similar")
                        elif score < 100:
                            print("⚠ Related")
                        else:
                            print("❌ Different")
                    else:
                        print("Invalid option")
                except Exception as e:
                    print("❌ TLSH Error:", str(e))
            tlsh_ops()
        elif op == 78:
            def imphash_ops():
                print("\nIMPHASH GENERATOR (Windows PE)")
                path = input("Enter PE file path (.exe / .dll): ")
                try:
                    h = generate_imphash(path)
                    print("Imphash:", h)
                except Exception as e:
                    print("❌ Failed to generate Imphash")
                    print("Error:", str(e))
            imphash_ops()
        elif op == 79:
            def pe_hash_ops():
                print("\nPE HASH ANALYZER")
                path = input("Enter PE file path (.exe / .dll): ")
                try:
                    info = pe_hash_analyzer(path)
                    print("\nFILE HASHES")
                    print("-" * 50)
                    print("MD5    :", info["MD5"])
                    print("SHA1   :", info["SHA1"])
                    print("SHA256 :", info["SHA256"])
                    print("Imphash:", info["Imphash"])
                    print("\nSECTIONS")
                    print("-" * 50)
                    for s in info["Sections"]:
                        print(f"{s['Name']:8} | Entropy: {s['Entropy']} | Raw: {s['RawSize']}")
                    if info["Warnings"]:
                        print("\n⚠ WARNINGS")
                        for w in info["Warnings"]:
                            print(" -", w)
                    else:
                        print("\n✔ No obvious PE anomalies")
                except Exception as e:
                    print("\n❌ PE analysis failed")
                    print("Error:", str(e))
            pe_hash_ops()
        elif op == 80:
            def btc_address_ops():
                print("\nBITCOIN ADDRESS VALIDATOR")
                addr = input("Enter Bitcoin address: ")
                result = validate_bitcoin_address(addr)
                if result.get("Valid"):
                    print("✔ VALID Bitcoin address")
                    print("Type   :", result["Type"])
                    print("Network:", result["Network"])
                else:
                    print("❌ INVALID Bitcoin address")
            btc_address_ops()
        elif op == 81:
            def eth_address_ops():
                print("\nETHEREUM ADDRESS VALIDATOR")
                addr = input("Enter Ethereum address: ")
                result = validate_ethereum_address(addr)
                if result.get("Valid"):
                    print("✔ VALID Ethereum address")
                    print("Type      :", result["Type"])
                    print("Checksum  :", result["Checksum"])
                else:
                    print("❌ INVALID Ethereum address")
            eth_address_ops()
        elif op == 82:
            def keccak_eth_ops():
                print("\nKECCAK-256 (ETHEREUM)")
                print("1. Hash Text")
                print("2. Hash File")
                choice = input("Choose option (1-2): ")
                try:
                    if choice == "1":
                        text = input("Enter text: ")
                        print("Keccak-256:", keccak256_eth(text.encode()))
                    elif choice == "2":
                        path = input("Enter file path: ")
                        with open(path, "rb") as f:
                            data = f.read()
                        print("Keccak-256:", keccak256_eth(data))
                    else:
                        print("Invalid option")
                except Exception as e:
                    print("❌ Error:", str(e))
            keccak_eth_ops()
        elif op == 83:
            def merkle_ops():
                print("\nMERKLE TREE GENERATOR")
                print("1. Text Inputs")
                print("2. File Hashes")
                algo = input("Hash algorithm (sha256 default): ") or "sha256"
                choice = input("Choose option (1-2): ")
                try:
                    if choice == "1":
                        items = []
                        print("Enter items (empty line to finish):")
                        while True:
                            line = input("> ")
                            if not line:
                                break
                            items.append(line)
                    elif choice == "2":
                        items = []
                        print("Enter file paths (empty line to finish):")
                        while True:
                            path = input("> ")
                            if not path:
                                break
                            with open(path, "rb") as f:
                                items.append(f.read().hex())
                    else:
                        print("Invalid option")
                        return
                    tree = build_merkle_tree(items, algo)
                    print("\nMerkle Root:")
                    print(tree["Root"])
                    print("\nMerkle Tree Levels:")
                    for i, lvl in enumerate(tree["Levels"]):
                        print(f"Level {i}:")
                        for h in lvl:
                            print(" ", h)
                except Exception as e:
                    print("❌ Error:", str(e))
            merkle_ops()
        elif op == 84:
                def wif_ops():
                    print("\nWALLET IMPORT FORMAT (WIF)")
                    print("1. Decode WIF")
                    print("2. Encode Private Key → WIF")

                    choice = input("Choose option (1-2): ")

                    try:
                        if choice == "1":
                            wif = input("Enter WIF: ")
                            info = wif_decode(wif)

                            print("\nWIF DETAILS")
                            print("Private Key:", info["PrivateKeyHex"])
                            print("Network    :", info["Network"])
                            print("Compressed :", info["Compressed"])

                        elif choice == "2":
                            key = input("Enter private key (hex): ")
                            compressed = input("Compressed? (y/n): ").lower() == "y"
                            testnet = input("Testnet? (y/n): ").lower() == "y"

                            wif = wif_encode(key, compressed, testnet)
                            print("WIF:", wif)

                        else:
                            print("Invalid option")

                    except Exception as e:
                        print("❌ Error:", str(e))

                wif_ops()
        elif op == 85:
            def text_steg_ops():
                print("\nTEXT STEGANOGRAPHY")
                print("1. Hide text")
                print("2. Extract hidden text")

                choice = input("Choose option (1-2): ")

                if choice == "1":
                    cover = input("Enter cover text: ")
                    secret = input("Enter secret message: ")
                    result = steg_text_encode(cover, secret)
                    print("\nStego Text (looks normal):")
                    print(result)

                elif choice == "2":
                    text = input("Paste stego text: ")
                    hidden = steg_text_decode(text)
                    print("\nHidden message:")
                    print(hidden)

                else:
                    print("Invalid option")

            text_steg_ops()
        elif op == 86:
            def image_steg_ops():
                print("\nIMAGE LSB STEGANOGRAPHY")
                print("1. Hide text in image")
                print("2. Extract text from image")
        
                choice = input("Choose option (1-2): ")
        
                try:
                    if choice == "1":
                        inp = input("Input image path (PNG/BMP): ")
                        out = input("Output image path: ")
                        secret = input("Secret message: ")
        
                        image_lsb_encode(inp, out, secret)
                        print("✔ Message hidden successfully")
        
                    elif choice == "2":
                        path = input("Stego image path: ")
                        secret = image_lsb_decode(path)
                        print("\nHidden message:")
                        print(secret)
        
                    else:
                        print("Invalid option")
        
                except Exception as e:
                    print("❌ Error:", str(e))
        
            image_steg_ops()
        elif op == 87:
            def audio_steg_ops():
                print("\nAUDIO STEGANOGRAPHY (WAV)")
                print("1. Hide text in audio")
                print("2. Extract text from audio")
        
                choice = input("Choose option (1-2): ")
        
                try:
                    if choice == "1":
                        inp = input("Input WAV file path: ")
                        out = input("Output WAV file path: ")
                        secret = input("Secret message: ")
        
                        audio_lsb_encode(inp, out, secret)
                        print("✔ Message hidden successfully")
        
                    elif choice == "2":
                        path = input("Stego WAV file path: ")
                        secret = audio_lsb_decode(path)
                        print("\nHidden message:")
                        print(secret)
        
                    else:
                        print("Invalid option")
        
                except Exception as e:
                    print("❌ Error:", str(e))
        
            audio_steg_ops()
        elif op == 88:
            def password_strength_ops():
                pwd = input("Enter password to evaluate: ")

                result = password_strength_estimator(pwd)

                print("\n🔐 PASSWORD STRENGTH REPORT")
                print("--------------------------")
                print("Length:", result["length"])
                print("Estimated Entropy:", result["entropy"])
                print("Strength:", result["verdict"])

                print("\nObservations:")
                for issue in result["issues"]:
                    print("-", issue)

            password_strength_ops()
        elif op == 89:
            def salt_generator_ops():
                print("\nSALT GENERATOR (ALL MODES)")
                print("1. Random Salt")
                print("2. Fixed Salt (Testing)")
                print("3. Password-derived Salt")
                print("4. Time-based Salt")
                print("5. User-specific Salt")
                print("6. Combined Salt (Best)")
                print("7. File-based Salt")
                print("8. Session Salt")

                choice = input("Choose mode (1-8): ")

                try:
                    if choice == "1":
                        l = int(input("Length (bytes): "))
                        print(salt_random(l))

                    elif choice == "2":
                        print(salt_fixed())

                    elif choice == "3":
                        pwd = input("Password: ")
                        print(salt_from_password(pwd))

                    elif choice == "4":
                        print(salt_time_based())

                    elif choice == "5":
                        user = input("Username/email: ")
                        print(salt_user_specific(user))

                    elif choice == "6":
                        pwd = input("Password: ")
                        user = input("Username: ")
                        print(salt_combined(pwd, user))

                    elif choice == "7":
                        path = input("File path: ")
                        print(salt_from_file(path))

                    elif choice == "8":
                        print(salt_session())

                    else:
                        print("Invalid option")

                except Exception as e:
                    print("❌ Error:", str(e))

            salt_generator_ops()

        elif op == 90:
            def nonce_generator_ops():
                print("🔐 NONCE GENERATOR")
                print("-----------------")
                print("1. AES-GCM / ChaCha20 (12 bytes)")
                print("2. AES-CTR (16 bytes)")
                print("3. Custom length")
                print("4. Show as Base64")

                choice = input("Choose option: ")

                if choice == "1":
                    print("Nonce (hex):", nonce_hex(12))

                elif choice == "2":
                    print("Nonce (hex):", nonce_hex(16))

                elif choice == "3":
                    length = int(input("Enter nonce length (bytes): "))
                    print("Nonce (hex):", nonce_hex(length))

                elif choice == "4":
                    length = int(input("Enter nonce length (bytes): "))
                    print("Nonce (Base64):", nonce_base64(length))

                else:
                    print("Invalid option")

            nonce_generator_ops()
        elif op == 91:
            def diceware_ops():
                print("\nDICEWARE PASSWORD GENERATOR")

                try:
                    path = input("Diceware wordlist path: ")
                    count = int(input("Number of words (recommended 6+): "))
                    sep = input("Separator (space / dash / underscore): ") or " "

                    wordlist = load_diceware_wordlist(path)
                    password = diceware_generate(wordlist, count, sep)

                    print("\nGenerated Diceware Password:")
                    print(password)

                except Exception as e:
                    print("❌ Error:", str(e))

            diceware_ops()

#       elif op == 99:
#            print("You have selected all decoders and encodersr all operations:")
#            print("encoded Base64 value is:", base64_encod        def file_hash_ops():
#            print("encoded hex value is:", hex_encoder(data))
#            print("decoded hex value is:", hex_decoder(data))
#            print("encoded url value is:", url_encoder(data))
#            print("decoded url value is:", url_decoder(data))
#           print("ROT13 value is:", CAESARorROT13_encrypt_and_decrypt(data, 13))
#            print("MD5 hash is:", MD5_encoder(data))
#            print("SHA1 hash is:", SHA1_encoder(data))
#            print("All operations completed successfully.")
#            break
#            print("working on it, coming soon...")
        
#            print("MD5 hash is:", MD5_encoder(data))
#            print("SHA1 hash is:", SHA1_encoder(data))
#            print("All operations completed successfully.")
#            break
#            print("working on it, coming soon...")
        elif op == 0:
            print("Exiting the program. Goodbye!")
            break
        elif op == 999:
            print("Returning to the main menu.")
            menu()
            coder_ops()
            break
        else:
            print("Entered wrong option!!,Try again")

if __name__ == "__main__":
    print("Welcome to Free Decoders in One.")
    menu()
    coder_ops()
