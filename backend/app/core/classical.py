def generate_polybius_square(key=""):
    ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # J removed
    key = key.upper().replace("J", "I")
    seen = set()
    square = []
    for c in key + ALPHABET:
        if c.isalpha() and c not in seen:
            seen.add(c)
            square.append(c)
    char_to_pos = {}
    pos_to_char = {}
    idx = 0
    for r in range(1, 6):
        for c in range(1, 6):
            char_to_pos[square[idx]] = (r, c)
            pos_to_char[(r, c)] = square[idx]
            idx += 1
    return char_to_pos, pos_to_char

def mod_inverse(a, m=26):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None

def caeser(data: str,shift: int)->str:
    result=''
    for char in data:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base + shift) % 26 + base)
        else:
            result += char
    return result
#done routes
def vigenere_encrypter(keyword, data):
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
#done routes
def vigenere_decrypter(data, keyword):
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
#done routes
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
#done routes
def becon_encoder(data):
    BACON_TABLE = {
        'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA',
        'F': 'AABAB', 'G': 'AABBA', 'H': 'AABBB', 'I': 'ABAAA', 'J': 'ABAAB',
        'K': 'ABABA', 'L': 'ABABB', 'M': 'ABBAA', 'N': 'ABBAB', 'O': 'ABBBA',
        'P': 'ABBBB', 'Q': 'BAAAA', 'R': 'BAAAB', 'S': 'BAABA', 'T': 'BAABB',
        'U': 'BABAA', 'V': 'BABAB', 'W': 'BABBA', 'X': 'BABBB', 'Y': 'BBAAA',
        'Z': 'BBAAB'
    }
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
    REVERSE_BACON_TABLE = {v: k for k, v in BACON_TABLE.items()}
    data = data.replace(" ", "").upper()
    result = []
    for i in range(0, len(data), 5):
        chunk = data[i:i+5]
        if len(chunk) == 5 and chunk in REVERSE_BACON_TABLE:
            result.append(REVERSE_BACON_TABLE[chunk])
        else:
            result.append('?')  
    return "".join(result)

def bifid_encrypt(plaintext: str, key="") -> str:

    char_to_pos, pos_to_char = generate_polybius_square(key)

    plaintext = plaintext.upper().replace("J", "I")
    plaintext = "".join(c for c in plaintext if c.isalpha())

    rows = []
    cols = []

    for ch in plaintext:
        r, c = char_to_pos[ch]
        rows.append(r)
        cols.append(c)
    merged = rows + cols

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

    for ch in ciphertext:
        r, c = char_to_pos[ch]
        coords.append(r)
        coords.append(c)

    half = len(coords) // 2
    rows = coords[:half]
    cols = coords[half:]

    plaintext = []
    for r, c in zip(rows, cols):
        plaintext.append(pos_to_char[(r, c)])

    return "".join(plaintext)

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
            result.append('/')  
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
    pattern = [[None] * len(ciphertext) for _ in range(rails)]
    rail = 0
    direction = 1
    for i in range(len(ciphertext)):
        pattern[rail][i] = '*'
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction *= -1
    index = 0
    for r in range(rails):
        for c in range(len(ciphertext)):
            if pattern[r][c] == '*' and index < len(ciphertext):
                pattern[r][c] = ciphertext[index]
                index += 1
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