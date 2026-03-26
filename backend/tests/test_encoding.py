from app.core import encoding

def test_base64_encode_decode():
    data = "hello world"
    encoded = encoding.base64_encoder(data)
    assert encoding.base64_decoder(encoded) == data

def test_hex_encode_decode():
    data = "hello world"
    encoded = encoding.hex_encoder(data)
    assert encoding.hex_decoder(encoded) == data

def test_morse_encode_decode():
    data = "HELLO"
    encoded = encoding.morse_encoder(data)
    # H = .... E = . L = .-.. L = .-.. O = ---
    assert encoded == ".... . .-.. .-.. ---"
    assert encoding.morse_decoder(encoded) == "HELLO"

def test_rot47():
    data = "Hello World123"
    # rot47(Hello World123) = 26==@ r@C=5`ab
    # Let's just check cycle
    encoded = encoding.rot47_encoder_decoder(data)
    assert encoding.rot47_encoder_decoder(encoded) == data

