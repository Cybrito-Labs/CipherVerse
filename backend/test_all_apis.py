import requests
import json

BASE = "http://127.0.0.1:8000/symmetric"
results = []

def test(name, enc_url, enc_body, dec_url=None, dec_body_fn=None):
    try:
        r = requests.post(BASE + enc_url, json=enc_body)
        if r.status_code != 200:
            results.append(f"FAIL {name} encrypt: {r.status_code} - {r.text[:100]}")
            return
        enc_result = r.json()["result"]
        
        if dec_url and dec_body_fn:
            dec_body = dec_body_fn(enc_result)
            r2 = requests.post(BASE + dec_url, json=dec_body)
            if r2.status_code != 200:
                results.append(f"FAIL {name} decrypt: {r2.status_code} - {r2.text[:100]}")
                return
            dec_result = r2.json()["result"]
            if "Hello" in dec_result:
                results.append(f"PASS {name} (encrypt + decrypt roundtrip)")
            else:
                results.append(f"WARN {name}: decrypted to '{dec_result[:50]}'")
        else:
            results.append(f"PASS {name} encrypt only")
    except Exception as e:
        results.append(f"ERROR {name}: {e}")


# ==================== XOR ====================
test("XOR", "/xor/encrypt",
     {"text": "Hello World", "key": "secret"},
     "/xor/decrypt",
     lambda r: {"hex_data": r, "key": "secret"})

# XOR Bruteforce (special case)
try:
    r = requests.post(BASE + "/xor/bruteforce", json={"hex_data": "3b000f1e"})
    if r.status_code == 200:
        results.append(f"PASS XOR bruteforce ({len(r.json()['results'])} results)")
    else:
        results.append(f"FAIL XOR bruteforce: {r.status_code}")
except Exception as e:
    results.append(f"ERROR XOR bruteforce: {e}")

# ==================== RC2 ====================
test("RC2", "/rc2/encrypt",
     {"text": "Hello World", "password": "mysecretpass"},
     "/rc2/decrypt",
     lambda r: {"ciphertext": r, "password": "mysecretpass"})

# ==================== RC4 ====================
test("RC4", "/rc4/encrypt",
     {"text": "Hello World", "key": "secret"},
     "/rc4/decrypt",
     lambda r: {"hex_data": r, "key": "secret"})

# ==================== RC4 Drop ====================
test("RC4-Drop", "/rc4/drop/encrypt",
     {"text": "Hello World", "key": "secret", "drop_n": 768},
     "/rc4/drop/decrypt",
     lambda r: {"text": r, "key": "secret", "drop_n": 768})

# ==================== CipherSaber2 ====================
test("CipherSaber2", "/ciphersaber2/encrypt",
     {"text": "Hello World", "password": "mysecretpass"},
     "/ciphersaber2/decrypt",
     lambda r: {"ciphertext": r, "password": "mysecretpass"})

# ==================== AES (multiple modes) ====================
for mode in ["CBC", "ECB", "CFB", "OFB", "CTR", "GCM"]:
    test(f"AES-{mode}", "/aes/encrypt",
         {"text": "Hello World", "password": "mysecretpass", "mode": mode, "key_size": 32},
         "/aes/decrypt",
         lambda r, m=mode: {"ciphertext": r, "password": "mysecretpass", "mode": m, "key_size": 32})

# ==================== DES ====================
test("DES-CBC", "/des/encrypt",
     {"text": "Hello World", "password": "mysecretpass", "mode": "CBC"},
     "/des/decrypt",
     lambda r: {"ciphertext": r, "password": "mysecretpass", "mode": "CBC"})

# ==================== Triple DES ====================
test("3DES-CBC", "/3des/encrypt",
     {"text": "Hello World", "password": "mysecretpass", "mode": "CBC"},
     "/3des/decrypt",
     lambda r: {"ciphertext": r, "password": "mysecretpass", "mode": "CBC"})

# ==================== Blowfish ====================
test("Blowfish-CBC", "/blowfish/encrypt",
     {"text": "Hello World", "password": "mysecretpass", "mode": "CBC"},
     "/blowfish/decrypt",
     lambda r: {"ciphertext": r, "password": "mysecretpass", "mode": "CBC"})

# ==================== SM4 ====================
test("SM4-ECB", "/sm4/encrypt",
     {"text": "Hello World", "password": "mysecretpass", "mode": "ECB"},
     "/sm4/decrypt",
     lambda r: {"ciphertext": r, "password": "mysecretpass", "mode": "ECB"})

# ==================== Print Results ====================
print("\n" + "=" * 60)
print("CipherVerse API Test Results")
print("=" * 60)
for r in results:
    icon = "[OK]" if r.startswith("PASS") else ("[XX]" if r.startswith("FAIL") else ("[??]" if r.startswith("WARN") else "[!!]"))
    print(f"  {icon} {r}")

passed = len([r for r in results if r.startswith("PASS")])
failed = len([r for r in results if r.startswith("FAIL")])
warned = len([r for r in results if r.startswith("WARN")])
errored = len([r for r in results if r.startswith("ERROR")])
print(f"\n  Total: {passed} passed, {failed} failed, {warned} warnings, {errored} errors out of {len(results)} tests")
print("=" * 60)
