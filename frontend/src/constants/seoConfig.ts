export interface PageSEO {
  title: string;
  description: string;
  keywords: string[];
  category?: string;
  ogImage?: string;
  faqs?: { question: string; answer: string }[];
}

export const SITE_NAME = 'CipherVerse';
export const SITE_URL = 'https://cipherverse.vercel.app';
export const DEFAULT_OG_IMAGE = 'https://cipherverse.vercel.app/og-image.png';
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
  'online cipher solver'
];

export const seoConfigMap: Record<string, PageSEO> = {
  '/': {
    title: 'CipherVerse — Next-Gen Professional Cybersecurity & Cryptography Platform',
    description: 'Explore 40+ interactive online cryptography, malware analysis, file forensics, steganography, and historical cipher tools in one high-performance platform.',
    keywords: ['cybersecurity platform', 'online cryptography', 'cipher tools', 'encryption tools', 'malware analysis online', 'steganography online'],
    category: 'Overview',
  },

  // Classical Ciphers
  '/classical': {
    title: 'Classical Ciphers Hub — Substitution & Transposition Solvers | CipherVerse',
    description: 'Interactive online suite for classical ciphers including Caesar, Vigenère, Atbash, Bacon, Bifid, Affine, A1Z26, Rail Fence, and Substitution ciphers.',
    keywords: ['classical ciphers', 'historical ciphers', 'substitution cipher solver', 'transposition ciphers', 'cryptanalysis tools'],
    category: 'Classical Ciphers',
  },
  '/classical/caesar': {
    title: 'Caesar Cipher Encoder & Decoder | Online Shift Cipher Tool',
    description: 'Free online Caesar Cipher encoder, decoder, and brute-force solver. Shift text by any key instantly with detailed letter frequency analysis.',
    keywords: ['caesar cipher', 'shift cipher', 'rot13 solver', 'caesar cipher decoder', 'caesar cipher encoder', 'caesar brute force'],
    category: 'Classical Ciphers',
    faqs: [
      {
        question: 'What is the Caesar Cipher?',
        answer: 'The Caesar Cipher is a classic substitution cipher where each letter in the plaintext is shifted by a fixed number of positions down the alphabet.'
      },
      {
        question: 'How do I decrypt a Caesar Cipher without a key?',
        answer: 'Use the brute-force mode in CipherVerse to inspect all 25 possible shifts and identify legible plain text automatically.'
      }
    ]
  },
  '/classical/vigenere': {
    title: 'Vigenère Cipher Encoder, Decoder & Key Solver | CipherVerse',
    description: 'Encrypt and decrypt messages using the polyalphabetic Vigenère Cipher. Includes automatic keyword analysis and tabular visualization.',
    keywords: ['vigenere cipher', 'polyalphabetic cipher', 'vigenere decoder', 'vigenere key solver', 'vigenere square'],
    category: 'Classical Ciphers',
  },
  '/classical/atbash': {
    title: 'Atbash Cipher Tool — Reverse Alphabet Encryption | CipherVerse',
    description: 'Fast online Atbash Cipher tool to substitute alphabet letters in reverse order (A to Z, B to Y).',
    keywords: ['atbash cipher', 'atbash decoder', 'atbash encoder', 'reverse alphabet cipher', 'hebrew cipher'],
    category: 'Classical Ciphers',
  },
  '/classical/bacon': {
    title: 'Baconian Cipher Encoder & Steganographic Decoder | CipherVerse',
    description: 'Encode and decode messages using Francis Bacon\'s binary steganographic cipher system (5-letter A/B patterns).',
    keywords: ['bacon cipher', 'baconian cipher', 'binary steganography', 'bacon decoder', 'francis bacon cipher'],
    category: 'Classical Ciphers',
  },
  '/classical/bifid': {
    title: 'Bifid Cipher Tool — Polybius Square Fractionation | CipherVerse',
    description: 'Encrypt and decrypt using the Bifid Cipher, combining Polybius square substitution with vertical transposition fractionation.',
    keywords: ['bifid cipher', 'fractionation cipher', 'polybius square', 'bifid decoder', 'bifid solver'],
    category: 'Classical Ciphers',
  },
  '/classical/affine': {
    title: 'Affine Cipher Solver — Mathematical Substitution Tool | CipherVerse',
    description: 'Online Affine Cipher tool to perform mathematical substitution encryption using linear modular arithmetic E(x) = (ax + b) mod 26.',
    keywords: ['affine cipher', 'modular arithmetic cipher', 'affine decoder', 'coprime key cipher'],
    category: 'Classical Ciphers',
  },
  '/classical/a1z26': {
    title: 'A1Z26 Cipher & Number Substitution Tool | CipherVerse',
    description: 'Convert text to numbers and numbers back to text instantly with the A1Z26 letter-number substitution converter.',
    keywords: ['a1z26 cipher', 'letter to number cipher', 'a1z26 decoder', 'number substitution'],
    category: 'Classical Ciphers',
  },
  '/classical/rail-fence': {
    title: 'Rail Fence Cipher Encoder & Decoder — Zigzag Transposition | CipherVerse',
    description: 'Online Rail Fence Cipher calculator. Encrypt and decrypt text using multi-rail zigzag transposition paths.',
    keywords: ['rail fence cipher', 'zigzag cipher', 'transposition cipher', 'rail fence decoder'],
    category: 'Classical Ciphers',
  },
  '/classical/substitution': {
    title: 'Monoalphabetic Substitution Cipher Solver | CipherVerse',
    description: 'Custom alphabet substitution cipher solver with letter frequency analysis and interactive key mapping.',
    keywords: ['substitution cipher', 'monoalphabetic cipher', 'cipher key mapping', 'letter frequency analysis'],
    category: 'Classical Ciphers',
  },

  // Encoding & Decoding
  '/encoding': {
    title: 'Online Encoding & Decoding Tools Hub | Base64, Hex, URL, Binary',
    description: 'Comprehensive suite of developer encoding and decoding utilities including Base64, Hexadecimal, URL, Binary, and Morse code.',
    keywords: ['encoding tools', 'base64 encoder', 'hex decoder', 'url encode decode', 'binary converter'],
    category: 'Encoding & Decoding',
  },
  '/encoding/base64': {
    title: 'Base64 Encoder & Decoder Online | CipherVerse',
    description: 'Instant online Base64 text and binary data encoder/decoder with UTF-8 and URL-safe support.',
    keywords: ['base64 encode', 'base64 decode', 'base64 converter', 'url safe base64'],
    category: 'Encoding & Decoding',
  },
  '/encoding/hex': {
    title: 'Hexadecimal (Hex) Encoder & Decoder | CipherVerse',
    description: 'Convert plain text to Hex bytes and decode Hex strings to readable text instantly with formatting options.',
    keywords: ['hex encoder', 'hex decoder', 'hex to string', 'string to hex', 'hexadecimal converter'],
    category: 'Encoding & Decoding',
  },
  '/encoding/url': {
    title: 'URL Percent Encoder & Decoder | CipherVerse',
    description: 'Encode special characters into percent-encoded URI strings and decode encoded URLs securely.',
    keywords: ['url encode', 'url decode', 'percent encoding', 'uri component encoder'],
    category: 'Encoding & Decoding',
  },
  '/encoding/binary': {
    title: 'Text to Binary & Binary to Text Converter | CipherVerse',
    description: 'Convert ASCII and UTF-8 text into 8-bit binary numbers (0s and 1s) and decode binary byte arrays.',
    keywords: ['text to binary', 'binary decoder', 'binary to text', '8 bit binary converter'],
    category: 'Encoding & Decoding',
  },
  '/encoding/morse': {
    title: 'Morse Code Translator — Audio & Visual Signals | CipherVerse',
    description: 'Translate text into International Morse Code dots and dashes, with audio playback and visual light simulation.',
    keywords: ['morse code translator', 'morse code decoder', 'text to morse', 'morse code audio'],
    category: 'Encoding & Decoding',
  },

  // Symmetric Cryptography
  '/symmetric': {
    title: 'Symmetric Cryptography Tools — AES, DES, 3DES, Blowfish, RC4 | CipherVerse',
    description: 'Interactive modern symmetric block and stream cipher toolset featuring AES-GCM/CBC, Triple DES, Blowfish, RC4, SM4, and XOR bruteforce.',
    keywords: ['symmetric encryption', 'aes online', 'des cipher', 'blowfish encryption', 'stream ciphers', 'block ciphers'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/aes': {
    title: 'AES Encryption & Decryption Online (AES-128, 192, 256) | CipherVerse',
    description: 'Secure online Advanced Encryption Standard (AES) calculator supporting CBC, GCM, CTR modes with key and IV generators.',
    keywords: ['aes encryption online', 'aes-256 calculator', 'aes cbc mode', 'aes gcm online', 'aes decrypt'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/des': {
    title: 'DES (Data Encryption Standard) Online Tool | CipherVerse',
    description: 'Demonstration and educational DES encryption/decryption tool with mode configuration and subkey visualization.',
    keywords: ['des cipher', 'data encryption standard', 'des online', 'des decrypt'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/3des': {
    title: 'Triple DES (3DES / TDEA) Encryption Tool | CipherVerse',
    description: 'Online Triple DES calculator supporting 2-key and 3-key EDE modes for legacy cipher inspection.',
    keywords: ['triple des', '3des encryption', 'tdea cipher', '3des online'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/rc2': {
    title: 'RC2 Block Cipher Tool | CipherVerse',
    description: 'Encrypt and decrypt using Ron Rivest\'s RC2 variable key-size block cipher.',
    keywords: ['rc2 cipher', 'rc2 encryption', 'ron rivest cipher', 'rc2 online'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/rc4': {
    title: 'RC4 Stream Cipher Generator & Decrypter | CipherVerse',
    description: 'Online RC4 stream cipher state generator and PRGA output inspector for cryptographic research.',
    keywords: ['rc4 cipher', 'rc4 online', 'stream cipher rc4', 'rc4 keystream'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/rc4-drop': {
    title: 'RC4-Drop Cipher Utility | CipherVerse',
    description: 'Enhanced RC4-Drop implementation discarding initial N keystream bytes to eliminate initial state bias.',
    keywords: ['rc4 drop', 'rc4 drop initial bytes', 'strengthened rc4'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/blowfish': {
    title: 'Blowfish Encryption & Decryption Tool | CipherVerse',
    description: 'Bruce Schneier\'s Blowfish symmetric block cipher tool with key sizes up to 448 bits.',
    keywords: ['blowfish cipher', 'blowfish encryption online', 'bruce schneier cipher'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/sm4': {
    title: 'SM4 National Standard Block Cipher Tool | CipherVerse',
    description: 'Chinese National Standard SM4 128-bit block cipher online encoder and decrypter.',
    keywords: ['sm4 cipher', 'sm4 encryption', 'chinese national standard cipher', 'sm4 block cipher'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/ciphersaber2': {
    title: 'CipherSaber-2 Key Stream Encryption Tool | CipherVerse',
    description: 'CipherSaber-2 cryptographic protocol tool utilizing RC4 with initial state mixing rounds.',
    keywords: ['ciphersaber', 'ciphersaber-2', 'rc4 IV mixing'],
    category: 'Symmetric Crypto',
  },
  '/symmetric/xor': {
    title: 'XOR Cipher & Multi-Byte Key Bruteforce Tool | CipherVerse',
    description: 'Single-byte and multi-byte XOR cipher encryption, decryption, and frequency-based key cracking tool.',
    keywords: ['xor cipher', 'xor solver', 'xor bruteforce', 'xor key breaker'],
    category: 'Symmetric Crypto',
  },

  // Asymmetric Cryptography
  '/asymmetric': {
    title: 'Asymmetric Cryptography — Public-Key RSA & DSA Suite | CipherVerse',
    description: 'Generate RSA and DSA keypairs, perform public key encryption, digital signature creation, and signature verification.',
    keywords: ['asymmetric cryptography', 'public key encryption', 'rsa key generator', 'dsa signature', 'digital signatures'],
    category: 'Asymmetric Crypto',
  },
  '/asymmetric/rsa': {
    title: 'RSA Key Generator, Encryption & Digital Signature Tool | CipherVerse',
    description: 'Online RSA tool to generate PEM keypairs (1024, 2048, 4096 bit), encrypt messages, decrypt ciphertexts, sign, and verify signatures.',
    keywords: ['rsa online tool', 'rsa keypair generator', 'rsa encrypt decrypt', 'rsa digital signature'],
    category: 'Asymmetric Crypto',
  },
  '/asymmetric/dsa': {
    title: 'DSA Key Generator & Digital Signature Algorithm | CipherVerse',
    description: 'Digital Signature Algorithm (DSA) parameter generator, signature generation, and verification toolkit.',
    keywords: ['dsa signature', 'digital signature algorithm', 'dsa key generation', 'dsa verify'],
    category: 'Asymmetric Crypto',
  },

  // Hashing & KDFs
  '/hashing': {
    title: 'Online Hash Generator & Key Derivation Functions (KDFs) | CipherVerse',
    description: 'Calculate cryptographic hashes (SHA-256, SHA-512, MD5, SHA-3), HMAC signatures, and run PBKDF2, Scrypt, bcrypt KDFs.',
    keywords: ['hash generator online', 'sha256 calculator', 'hmac generator', 'pbkdf2 calculator', 'scrypt tool'],
    category: 'Hashing & KDFs',
  },

  // Certificates & TLS
  '/certificates': {
    title: 'X.509 Certificate Inspector & TLS Handshake Analyzer | CipherVerse',
    description: 'Parse X.509 PEM/DER certificates, inspect SANs and issuer chains, test TLS connections, and compute SSL fingerprints.',
    keywords: ['x509 parser', 'ssl certificate inspector', 'tls handshake analyzer', 'certificate fingerprint'],
    category: 'Certificates & TLS',
  },
  '/certificates/x509': {
    title: 'X.509 SSL/TLS Certificate Parser Online | CipherVerse',
    description: 'Parse X.509 certificates to extract subject, issuer, serial number, validity dates, subjectAltNames, and public keys.',
    keywords: ['x509 certificate parser', 'ssl cert viewer', 'pem parser online', 'san inspector'],
    category: 'Certificates & TLS',
  },
  '/certificates/tls': {
    title: 'TLS Protocol & Cipher Suite Inspection Tool | CipherVerse',
    description: 'Analyze SSL/TLS handshake sequences, supported cipher suites, and protocol security parameters.',
    keywords: ['tls inspector', 'ssl handshake tool', 'cipher suite checker'],
    category: 'Certificates & TLS',
  },
  '/certificates/fingerprint': {
    title: 'SSL Certificate Fingerprint Calculator (SHA-1, SHA-256) | CipherVerse',
    description: 'Calculate SHA-1, SHA-256, and MD5 fingerprints for SSL/TLS certificates instantly.',
    keywords: ['certificate fingerprint', 'sha256 cert fingerprint', 'ssl thumbprint calculator'],
    category: 'Certificates & TLS',
  },

  // Blockchain Tools
  '/blockchain': {
    title: 'Blockchain Validation Suite — Bitcoin, Ethereum & Merkle Trees | CipherVerse',
    description: 'Validate Bitcoin and Ethereum public addresses, construct Merkle tree proofs, and encode WIF private keys.',
    keywords: ['blockchain tools', 'bitcoin address validator', 'ethereum address checker', 'merkle tree generator', 'wif encoder'],
    category: 'Blockchain',
  },
  '/blockchain/bitcoin': {
    title: 'Bitcoin Address Validator & Script Inspector | CipherVerse',
    description: 'Validate Bitcoin legacy (P2PKH), SegWit (P2SH), and Bech32/Taproot addresses with checksum verification.',
    keywords: ['bitcoin address validator', 'bech32 validator', 'segwit address checker', 'btc checksum'],
    category: 'Blockchain',
  },
  '/blockchain/ethereum': {
    title: 'Ethereum (ETH) Address Validator & EIP-55 Checksum Tool | CipherVerse',
    description: 'Verify Ethereum 0x hex addresses and test EIP-55 mixed-case checksum validity online.',
    keywords: ['ethereum address validator', 'eip55 checker', 'eth 0x address checksum'],
    category: 'Blockchain',
  },
  '/blockchain/merkle': {
    title: 'Merkle Tree Construction & Visual Proof Generator | CipherVerse',
    description: 'Build interactive Merkle Trees from transaction hashes, calculate root hash, and verify audit path inclusion proofs.',
    keywords: ['merkle tree generator', 'merkle root calculator', 'merkle proof verifier', 'blockchain merkle tree'],
    category: 'Blockchain',
  },
  '/blockchain/wif': {
    title: 'Wallet Import Format (WIF) Private Key Encoder | CipherVerse',
    description: 'Encode and decode Bitcoin raw ECDSA private keys into compressed and uncompressed WIF format.',
    keywords: ['wif encoder', 'wallet import format', 'bitcoin private key wif', 'wif decoder'],
    category: 'Blockchain',
  },

  // Steganography
  '/steganography': {
    title: 'Online Steganography Tools — Text, Image & Audio Hiding | CipherVerse',
    description: 'Hide hidden payload messages inside digital images (LSB), audio files, and zero-width unicode text steganography.',
    keywords: ['steganography online', 'image steganography', 'hide text in image', 'audio steganography', 'zero width steganography'],
    category: 'Steganography',
  },
  '/steganography/text': {
    title: 'Zero-Width Text Steganography Encoder & Decoder | CipherVerse',
    description: 'Hide invisible secret messages within plain text using zero-width non-joiner unicode characters.',
    keywords: ['text steganography', 'zero width space hide text', 'invisible text encoder', 'stego text'],
    category: 'Steganography',
  },
  '/steganography/image': {
    title: 'Image Steganography Tool — LSB Secret Text Hiding | CipherVerse',
    description: 'Embed and extract hidden encrypted messages inside PNG/JPEG image pixels using Least Significant Bit (LSB) encoding.',
    keywords: ['image steganography online', 'lsb steganography', 'hide secret text in photo', 'steganography decoder online'],
    category: 'Steganography',
  },
  '/steganography/audio': {
    title: 'Audio Steganography Tool — LSB Waveform Hiding | CipherVerse',
    description: 'Embed secret text messages inside WAV audio sample data with playback preservation.',
    keywords: ['audio steganography', 'wav steganography', 'hide message in sound file'],
    category: 'Steganography',
  },

  // Malware Analysis
  '/malware-analysis': {
    title: 'Malware Analysis Toolkit — Hashes, TLSH & PE Header Analysis | CipherVerse',
    description: 'Online malware triage platform featuring multi-hash lookup, Trend Micro Locality Sensitive Hashing (TLSH), and PE binary header parsing.',
    keywords: ['malware analysis online', 'tlsh similarity', 'pe header parser', 'file hash lookup', 'malware triage'],
    category: 'Malware Analysis',
  },
  '/malware-analysis/hash': {
    title: 'Malware File Hash Lookup & Analysis | CipherVerse',
    description: 'Extract and query MD5, SHA-1, SHA-256 malware signatures for threat intelligence triage.',
    keywords: ['malware hash lookup', 'sha256 threat intelligence', 'virustotal hash lookup'],
    category: 'Malware Analysis',
  },
  '/malware-analysis/tlsh': {
    title: 'TLSH (Trend Micro Locality Sensitive Hash) Comparator | CipherVerse',
    description: 'Calculate TLSH fuzzy hashes and compute distance scores to detect malware family variants and binary similarity.',
    keywords: ['tlsh comparison', 'locality sensitive hashing', 'fuzzy hashing malware', 'tlsh distance score'],
    category: 'Malware Analysis',
  },
  '/malware-analysis/pe': {
    title: 'PE (Portable Executable) Header Parser & Section Inspector | CipherVerse',
    description: 'Parse Windows PE/EXE headers, section tables (.text, .data, .rsrc), import/export tables, and compile timestamps.',
    keywords: ['pe header parser online', 'exe section analyzer', 'windows pe inspection', 'import table parser'],
    category: 'Malware Analysis',
  },

  // File Forensics
  '/file-forensics': {
    title: 'File Forensics Suite — Entropy, Multi-Hash & Randomness Testing | CipherVerse',
    description: 'Perform digital forensic file analysis: calculate Shannon entropy curves, compute simultaneous file hashes, and evaluate PRNG randomness.',
    keywords: ['file forensics online', 'shannon entropy calculator', 'file randomness test', 'digital forensics tools'],
    category: 'File Forensics',
  },
  '/file-forensics/hash': {
    title: 'Single File Hash Generator | CipherVerse',
    description: 'Calculate cryptographic digests for local files directly in browser without uploading to server.',
    keywords: ['file hash generator', 'sha256 file hash', 'md5 file check'],
    category: 'File Forensics',
  },
  '/file-forensics/multi-hash': {
    title: 'Multi-Algorithm Concurrent File Hasher | CipherVerse',
    description: 'Calculate MD5, SHA-1, SHA-256, SHA-512, and RIPEMD-160 file hashes in a single pass.',
    keywords: ['multi file hash', 'simultaneous hashing', 'file integrity check'],
    category: 'File Forensics',
  },
  '/file-forensics/entropy': {
    title: 'Shannon File Entropy Analyzer & Visualization | CipherVerse',
    description: 'Analyze binary file entropy to detect compressed files, packed executables, or encrypted payloads.',
    keywords: ['file entropy calculator', 'shannon entropy', 'detect encrypted file', 'packed exe detection'],
    category: 'File Forensics',
  },
  '/file-forensics/randomness': {
    title: 'Cryptographic Randomness Test Suite (NIST / Chi-Square) | CipherVerse',
    description: 'Evaluate random number generator output with Chi-square, byte distribution, and serial correlation tests.',
    keywords: ['randomness test online', 'chi square test', 'prng quality test', 'entropy evaluation'],
    category: 'File Forensics',
  },

  // Utilities
  '/utilities': {
    title: 'Cybersecurity Utilities — Password Entropy, JWT Signer, Salt Generator | CipherVerse',
    description: 'Developer security utilities including password strength analyzer, JWT signature generator, cryptographically secure salt generator, and checksum tools.',
    keywords: ['security utilities', 'password strength tester', 'jwt sign tool', 'crypto salt generator'],
    category: 'Utilities',
  },
  '/utilities/password': {
    title: 'Password Strength & Entropy Calculator | CipherVerse',
    description: 'Evaluate password entropy (bits), estimate brute-force cracking time, and detect common dictionary vulnerability patterns.',
    keywords: ['password strength analyzer', 'password entropy calculator', 'time to crack password'],
    category: 'Utilities',
  },
  '/utilities/jwt': {
    title: 'JWT (JSON Web Token) HMAC Signer & Verifier | CipherVerse',
    description: 'Sign and verify JWT tokens online using HS256/HS384/HS512 secret keys with custom header and payload JSON.',
    keywords: ['jwt signer online', 'jwt hmac sign', 'jwt secret tester', 'jwt generator'],
    category: 'Utilities',
  },
  '/utilities/salt': {
    title: 'Cryptographic Salt & Token Generator | CipherVerse',
    description: 'Generate high-entropy cryptographically secure random salts in Hex, Base64, and raw byte formats.',
    keywords: ['crypto salt generator', 'random salt generator', 'secure random byte token'],
    category: 'Utilities',
  },
  '/utilities/fletcher16': {
    title: 'Fletcher-16 Checksum Calculator | CipherVerse',
    description: 'Calculate Fletcher-16 error-detecting checksums for data integrity verification.',
    keywords: ['fletcher 16 checksum', 'fletcher checksum calculator', 'data integrity checksum'],
    category: 'Utilities',
  },

  // Historical Machines
  '/historical': {
    title: 'Historical Cipher Machines — Enigma, Bombe & Typex Simulators | CipherVerse',
    description: 'Experience WWII cryptographic history with accurate software simulators of the German Enigma machine, Turing Bombe, and British Typex cipher machine.',
    keywords: ['historical cipher machines', 'enigma machine simulator', 'turing bombe simulator', 'typex cipher machine'],
    category: 'Historical Machines',
  },
  '/historical/enigma': {
    title: 'Enigma Machine Simulator (I, M3, M4) Online | CipherVerse',
    description: 'Authentic 3-rotor and 4-rotor Enigma machine simulator. Configure rotors, ring settings, reflector, and plugboard (Steckerbrett).',
    keywords: ['enigma machine simulator', 'online enigma machine', 'ww2 enigma cipher', 'steckerbrett simulator'],
    category: 'Historical Machines',
  },
  '/historical/bombe': {
    title: 'Alan Turing Bombe Crib Analysis Simulator | CipherVerse',
    description: 'Interactive simulation of Alan Turing\'s Bombe electromechanical machine used at Bletchley Park to break Enigma rotor keys.',
    keywords: ['turing bombe simulator', 'bletchley park bombe', 'enigma crib solver'],
    category: 'Historical Machines',
  },
  '/historical/typex': {
    title: 'British Typex Cipher Machine Simulator | CipherVerse',
    description: 'Simulate the British WWII Typex rotor cipher machine with multi-rotor transposition logic.',
    keywords: ['typex cipher machine', 'british enigma', 'typex simulator'],
    category: 'Historical Machines',
  },

  // Miscellaneous Pages
  '/api-explorer': {
    title: 'API Explorer — Cryptographic REST Endpoints | CipherVerse',
    description: 'Test and integrate CipherVerse REST API endpoints for automated cipher calculation and threat intelligence.',
    keywords: ['cryptography api', 'cipher rest api', 'api explorer', 'security api endpoints'],
    category: 'Developer',
  },
  '/settings': {
    title: 'Platform Settings & Preferences | CipherVerse',
    description: 'Customize CipherVerse theme, default key options, and local workspace preferences.',
    keywords: ['cipherverse settings', 'preferences'],
    category: 'System',
  },
  '/404': {
    title: '404 - Page Not Found | CipherVerse',
    description: 'The requested cipher or security tool page could not be found on CipherVerse.',
    keywords: ['404', 'page not found'],
    category: 'System',
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
    title: 'CipherVerse — Professional Cybersecurity & Cryptography Platform',
    description: 'Interactive online suite for ciphers, hashing, malware analysis, file forensics, steganography, and historical cryptographic machines.',
    keywords: DEFAULT_KEYWORDS,
    category: 'Cybersecurity',
  };
}
