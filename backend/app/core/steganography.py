from PIL import Image
import wave
import struct

def steg_text_encode(cover_text: str, secret: str) -> str:
    ZWSP = "\u200b"   # 0
    ZWNJ = "\u200c"   # 1
    ZWJ  = "\u200d"   # end marker
    bits = "".join(format(ord(c), "08b") for c in secret)
    encoded = "".join(ZWNJ if b == "1" else ZWSP for b in bits)
    return cover_text + encoded + ZWJ

def steg_text_decode(stego_text: str) -> str:
    ZWSP = "\u200b"; ZWNJ = "\u200c"; ZWJ  = "\u200d"
    bits = ""
    for c in stego_text:
        if c == ZWSP: bits += "0"
        elif c == ZWNJ: bits += "1"
        elif c == ZWJ: break
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return "".join(chr(int(b, 2)) for b in chars if len(b) == 8)

def image_lsb_encode(input_img: str, output_img: str, secret: str):
    img = Image.open(input_img).convert("RGB")
    pixels = img.load()
    bits = ''.join(format(ord(c), '08b') for c in secret) + '1111111111111110'
    idx = 0
    for y in range(img.height):
        for x in range(img.width):
            if idx >= len(bits): 
                img.save(output_img)
                return
            r, g, b = pixels[x, y]
            r = (r & ~1) | int(bits[idx]); idx += 1
            if idx < len(bits): g = (g & ~1) | int(bits[idx]); idx += 1
            if idx < len(bits): b = (b & ~1) | int(bits[idx]); idx += 1
            pixels[x, y] = (r, g, b)
    img.save(output_img)

def image_lsb_decode(img_path: str) -> str:
    img = Image.open(img_path).convert("RGB")
    pixels = img.load()
    bits = ""
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = pixels[x, y]
            bits += str(r & 1) + str(g & 1) + str(b & 1)
            if bits.endswith("1111111111111110"):
                clean_bits = bits[:-16]
                chars = [clean_bits[i:i+8] for i in range(0, len(clean_bits), 8)]
                return "".join(chr(int(b, 2)) for b in chars if len(b) == 8)
    return ""

def audio_lsb_encode(input_wav: str, output_wav: str, secret: str):
    with wave.open(input_wav, 'rb') as wf:
        params = wf.getparams()
        frames = bytearray(wf.readframes(wf.getnframes()))
    bits = ''.join(format(ord(c), '08b') for c in secret) + '1111111111111110'
    if len(bits) > len(frames): raise ValueError("Audio too small")
    for i in range(len(bits)):
        frames[i] = (frames[i] & 0xFE) | int(bits[i])
    with wave.open(output_wav, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(frames)

def audio_lsb_decode(stego_wav: str) -> str:
    with wave.open(stego_wav, 'rb') as wf:
        frames = bytearray(wf.readframes(wf.getnframes()))
    bits = "".join(str(b & 1) for b in frames)
    if "1111111111111110" in bits:
        clean_bits = bits[:bits.find("1111111111111110")]
        chars = [clean_bits[i:i+8] for i in range(0, len(clean_bits), 8)]
        return "".join(chr(int(b, 2)) for b in chars if len(b) == 8)
    return ""

