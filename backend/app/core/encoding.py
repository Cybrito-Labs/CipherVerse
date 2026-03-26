import base64
import urllib.parse

def base64_encoder(data: str) -> str:
    return base64.b64encode(data.encode()).decode()

def base64_decoder(data: str) -> str:
    return base64.b64decode(data.encode()).decode()

def hex_encoder(data: str) -> str:
    return data.encode().hex()

def hex_decoder(data: str) -> str:
    return bytes.fromhex(data).decode()

def url_encoder(data: str) -> str:
    return urllib.parse.quote(data)

def url_decoder(data: str) -> str:
    return urllib.parse.unquote(data)

def binary_encoder(data: str) -> str:
    return ' '.join(format(ord(c), '08b') for c in data)

def binary_decoder(data: str) -> str:
    data = data.replace(' ', '')
    return ''.join(chr(int(data[i:i+8], 2)) for i in range(0, len(data), 8))

def ascii_encoder(data: str) -> str:
    return ','.join(str(ord(c)) for c in data)

def ascii_decoder(data: str) -> str:
    return ''.join(chr(int(c)) for c in data.split(','))

def base32_encoder(data: str) -> str:
    return base64.b32encode(data.encode()).decode()

def base32_decoder(data: str) -> str:
    return base64.b32decode(data.encode()).decode()

def morse_encoder(data: str) -> str:
    MORSE_CODE_DICT = { 'A':'.-', 'B':'-...', 'C':'-.-.', 'D':'-..', 'E':'.', 'F':'..-.', 'G':'--.', 'H':'....', 'I':'..', 'J':'.---', 'K':'-.-', 'L':'.-..', 'M':'--', 'N':'-.', 'O':'---', 'P':'.--.', 'Q':'--.-', 'R':'.-.', 'S':'...', 'T':'-', 'U':'..-', 'V':'...-', 'W':'.--', 'X':'-..-', 'Y':'-.--', 'Z':'--..', '1':'.----', '2':'..---', '3':'...--', '4':'....-', '5':'.....', '6':'-....', '7':'--...', '8':'---..', '9':'----.', '0':'-----', ', ':'--..--', '.':'.-.-.-', '?':'..--..', '/':'-..-.', '-':'-....-', '(':'-.--.', ')':'-.--.-', ' ': ' '}
    return ' '.join(MORSE_CODE_DICT.get(c.upper(), '') for c in data)

def morse_decoder(data: str) -> str:
    MORSE_CODE_DICT = { 'A':'.-', 'B':'-...', 'C':'-.-.', 'D':'-..', 'E':'.', 'F':'..-.', 'G':'--.', 'H':'....', 'I':'..', 'J':'.---', 'K':'-.-', 'L':'.-..', 'M':'--', 'N':'-.', 'O':'---', 'P':'.--.', 'Q':'--.-', 'R':'.-.', 'S':'...', 'T':'-', 'U':'..-', 'V':'...-', 'W':'.--', 'X':'-..-', 'Y':'-.--', 'Z':'--..', '1':'.----', '2':'..---', '3':'...--', '4':'....-', '5':'.....', '6':'-....', '7':'--...', '8':'---..', '9':'----.', '0':'-----', ', ':'--..--', '.':'.-.-.-', '?':'..--..', '/':'-..-.', '-':'-....-', '(':'-.--.', ')':'-.--.-', ' ': ' '}
    REVERSE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}
    return ''.join(REVERSE_DICT.get(c, '') for c in data.split(' '))

def rot47_encoder_decoder(data: str) -> str:
    result = []
    for i in range(len(data)):
        j = ord(data[i])
        if 33 <= j <= 126:
            result.append(chr(33 + ((j + 14) % 94)))
        else:
            result.append(data[i])
    return ''.join(result)
