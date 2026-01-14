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
    print("41. GOST / Streebog")
    print("42. SSDEEP / CTPH / Compare SSDEEP or CTPH hashes")

    # Passwords and HMACs
    print("\n[PASSWORDS & MACS]")
    print("43. HMAC")
    print("44. Bcrypt (Hash/Compare/Parse)")
    print("45. Scrypt")

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
def sha0_hash(data: str) -> str:
    return "SHA-0 was withdrawn and is not implemented"
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
#        elif op == 34:
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

  #      elif op == 49:
  #      elif op == 50:
  #      elif op == 51:
  #      elif op == 52:
  #      elif op == 53:
  #      elif op == 54:
  #      elif op == 55:
  #      elif op == 56:
  #      elif op == 57:
  #      elif op == 58:
  #      elif op == 59:
  #      elif op == 60:
  #      elif op == 61:
  #      elif op == 62:
  #      elif op == 63:
  #      elif op == 64:
  #      elif op == 65:
  #      elif op == 66:
  #      elif op == 67:
  #      elif op == 68:
  #      elif op == 69:
  #      elif op == 70:
  #      elif op == 71:
#       elif op == 99:
#            print("You have selected all decoders and encoders at once.")
#            data = input("Enter the text for all operations:")
#            print("encoded Base64 value is:", base64_encoder(data))
#            print("decoded Base64 value is:", base64_decoder(data))
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

print("Welcome to Free Decoders in One.")
menu()
coder_ops()