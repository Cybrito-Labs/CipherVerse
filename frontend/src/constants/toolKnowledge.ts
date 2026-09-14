export interface HowToStep {
  step: number;
  title: string;
  description: string;
}

export interface FAQItem {
  question: string;
  answer: string;
}

export interface ToolKnowledge {
  howTo: HowToStep[];
  faqs: FAQItem[];
  guideTitle?: string;
  guideSubtitle?: string;
}

export const TOOL_KNOWLEDGE_MAP: Record<string, ToolKnowledge> = {
  // ==========================================
  // Category Hub Pages
  // ==========================================
  '/classical': {
    guideTitle: 'Guide to Classical Ciphers & Cryptanalysis',
    guideSubtitle: 'Understanding substitution, transposition, and historic pencil-and-paper ciphers',
    howTo: [
      {
        step: 1,
        title: 'Identify the Cipher Type',
        description: 'Determine whether your secret text uses substitution (letters replaced by other letters/numbers) or transposition (letter order scrambled).',
      },
      {
        step: 2,
        title: 'Choose the Specialized Solver',
        description: 'Select Caesar/ROT13 for single-shift text, Vigenère for keyword polyalphabetic text, Atbash for inverted alphabets, or Rail Fence for zigzag patterns.',
      },
      {
        step: 3,
        title: 'Run Frequency Analysis or Brute-Force',
        description: 'Inspect letter distribution histograms comparing against standard English frequencies (E, T, A, O, I, N) to confirm decryption accuracy.',
      },
    ],
    faqs: [
      {
        question: 'What is the fundamental difference between substitution and transposition ciphers?',
        answer: 'Substitution ciphers replace plaintext letters with different characters, numbers, or symbols while preserving their positions. Transposition ciphers keep the original letters intact but rearrange their physical order.',
      },
      {
        question: 'Which classical cipher was historically called "le chiffre indéchiffrable" (the indecipherable cipher)?',
        answer: 'The Vigenère cipher held this reputation for nearly three centuries until Friedrich Kasiski published a mathematical method in 1863 to determine keyword length and break the cipher.',
      },
      {
        question: 'Are classical ciphers safe for modern data transmission?',
        answer: 'No. Modern computers can brute-force all classical key spaces in less than a second. Classical ciphers are studied today for education, cryptographic history, and CTF competitions.',
      },
    ],
  },

  '/encoding': {
    guideTitle: 'Developer Guide to Encodings & Representations',
    guideSubtitle: 'Translating between binary bytes, hexadecimal, Base64, and telecommunication signals',
    howTo: [
      {
        step: 1,
        title: 'Identify Required Data Format',
        description: 'Determine whether you need Base64 for web transfers, Hex for byte-level memory dumps, URL percent-encoding for HTTP parameters, or Binary for bitwise analysis.',
      },
      {
        step: 2,
        title: 'Select Conversion Tool & Parameters',
        description: 'Choose delimiters, character encodings (UTF-8, ASCII), or URL-safe character sets.',
      },
      {
        step: 3,
        title: 'Encode or Decode with One Click',
        description: 'Instantly view representations, inspect byte lengths, and copy formatted strings.',
      },
    ],
    faqs: [
      {
        question: 'What is the difference between encoding and encryption?',
        answer: 'Encoding transforms data into a standard format so different computer systems can process or transmit it; it requires no secret key and provides zero confidentiality. Encryption transforms data to keep it confidential and requires a secret key to reverse.',
      },
      {
        question: 'Why does Base64 increase data size by ~33%?',
        answer: 'Base64 represents 3 binary bytes (24 bits) using 4 ASCII characters (each carrying 6 bits of data). This 4/3 ratio creates an approximate 33% overhead.',
      },
    ],
  },

  '/symmetric': {
    guideTitle: 'Modern Symmetric Cryptography Guide',
    guideSubtitle: 'Secret-key block and stream ciphers protecting digital data at rest and in transit',
    howTo: [
      {
        step: 1,
        title: 'Select Cipher Algorithm',
        description: 'Choose AES for modern production security, Blowfish for variable key sizes, Triple DES for legacy compliance, or RC4 for stream cipher research.',
      },
      {
        step: 2,
        title: 'Configure Mode & Secret Key',
        description: 'Select operational modes like GCM (authenticated encryption) or CBC, and set keys ranging from 128 to 256 bits.',
      },
      {
        step: 3,
        title: 'Generate IV & Authenticated Output',
        description: 'Provide an Initialization Vector (IV/Nonce) to ensure identical plaintexts encrypt to unique ciphertexts, and verify authentication tags.',
      },
    ],
    faqs: [
      {
        question: 'What is the difference between block ciphers and stream ciphers?',
        answer: 'Block ciphers (like AES, DES, Blowfish) process data in fixed-size chunks (e.g. 128 bits) using modes of operation. Stream ciphers (like RC4, ChaCha20) generate a continuous pseudorandom keystream XORed bit-by-bit with the data.',
      },
      {
        question: 'Why is AES-GCM preferred over AES-CBC in modern applications?',
        answer: 'AES-GCM provides Authenticated Encryption with Associated Data (AEAD), ensuring both confidentiality and message integrity in a single pass while preventing padding-oracle and bit-flipping attacks.',
      },
    ],
  },

  '/asymmetric': {
    guideTitle: 'Public-Key Asymmetric Cryptography Suite',
    guideSubtitle: 'Mathematical keypairs for secure key exchange, encryption, and digital signatures',
    howTo: [
      {
        step: 1,
        title: 'Generate or Import Keypair',
        description: 'Create a mathematically linked public and private keypair in PEM or DER format.',
      },
      {
        step: 2,
        title: 'Distribute Public Key / Protect Private Key',
        description: 'Share the public key openly with communicators while keeping the private key strictly confidential.',
      },
      {
        step: 3,
        title: 'Execute Asymmetric Operations',
        description: 'Encrypt data using the recipient\'s public key or sign messages with your private key for non-repudiation.',
      },
    ],
    faqs: [
      {
        question: 'How do public and private keys interact mathematically?',
        answer: 'Data encrypted with a public key can only be decrypted by its corresponding private key. Conversely, a digital signature created with a private key can be verified by anyone holding the matching public key.',
      },
      {
        question: 'Why is asymmetric encryption rarely used to encrypt large files directly?',
        answer: 'Asymmetric operations are computationally expensive (thousands of times slower than symmetric ciphers). In practice, hybrid encryption is used: an asymmetric cipher encrypts a fast symmetric session key (like AES-256), which encrypts the file.',
      },
    ],
  },

  '/certificates': {
    guideTitle: 'X.509 & TLS Certificate Inspector Guide',
    guideSubtitle: 'Public Key Infrastructure (PKI), handshake negotiation, and SSL fingerprints',
    howTo: [
      {
        step: 1,
        title: 'Paste Certificate or Query Hostname',
        description: 'Provide an X.509 PEM certificate string or query an active TLS domain name.',
      },
      {
        step: 2,
        title: 'Inspect ASN.1 Certificate Hierarchy',
        description: 'Review Subject, Issuer, Validity NotBefore/NotAfter dates, Serial Number, and Subject Alternative Names (SANs).',
      },
      {
        step: 3,
        title: 'Verify Thumbprints & Supported Ciphers',
        description: 'Calculate SHA-256/SHA-1 fingerprints and verify handshake security parameters.',
      },
    ],
    faqs: [
      {
        question: 'What is a Certificate Authority (CA) chain of trust?',
        answer: 'A chain of trust links an end-entity SSL certificate to intermediate CAs, which ultimately lead to a trusted Root CA pre-installed in your operating system or browser trust store.',
      },
      {
        question: 'What happens if a certificate\'s Subject Alternative Name (SAN) is missing?',
        answer: 'Modern web browsers strictly require valid SAN entries matching the accessed hostname. Without SANs, browsers reject the connection as invalid, even if the Common Name (CN) matches.',
      },
    ],
  },

  '/blockchain': {
    guideTitle: 'Blockchain Validation & Cryptography Suite',
    guideSubtitle: 'Cryptographic address verification, Merkle audit proofs, and wallet key encoding',
    howTo: [
      {
        step: 1,
        title: 'Select Blockchain Network or Tool',
        description: 'Choose between Bitcoin address validation, Ethereum checksum verification, Merkle proof generation, or WIF private key encoding.',
      },
      {
        step: 2,
        title: 'Input Cryptographic Payload',
        description: 'Enter addresses, raw private keys, or transaction hash lists.',
      },
      {
        step: 3,
        title: 'Verify Mathematical Checksums & Proofs',
        description: 'Confirm Base58Check, Bech32 BCH error correction, or EIP-55 mixed-case validity.',
      },
    ],
    faqs: [
      {
        question: 'Why do cryptocurrencies use checksummed addresses?',
        answer: 'Checksums mathematically detect typos, miscopied characters, or transcription errors before a transaction is broadcast to the blockchain network, preventing irreversible loss of funds.',
      },
      {
        question: 'What is a Merkle tree root hash?',
        answer: 'The Merkle root is a single 32-byte cryptographic digest at the top of a binary hash tree that immutably summarizes every transaction included within a block.',
      },
    ],
  },

  '/steganography': {
    guideTitle: 'Digital Steganography Suite Guide',
    guideSubtitle: 'Concealing confidential payloads inside image pixels, audio waveforms, and unicode text',
    howTo: [
      {
        step: 1,
        title: 'Choose Carrier Medium',
        description: 'Select Image Steganography for lossless PNG/BMP files, Audio Steganography for WAV files, or Text Steganography for zero-width unicode characters.',
      },
      {
        step: 2,
        title: 'Enter Secret Message & Passphrase',
        description: 'Type the confidential payload and set an optional AES passphrase for layered security.',
      },
      {
        step: 3,
        title: 'Encode or Extract Carrier',
        description: 'Download the modified carrier file or extract hidden messages from an existing steganographic file.',
      },
    ],
    faqs: [
      {
        question: 'How does steganography differ from cryptography?',
        answer: 'Cryptography scrambles the contents of a message so an eavesdropper cannot read it, but the existence of the secret message is obvious. Steganography conceals the very fact that a secret communication is taking place.',
      },
      {
        question: 'What is steganalysis?',
        answer: 'Steganalysis is the forensic science of detecting whether a digital file contains hidden information by analyzing pixel noise, color histograms, or high-frequency audio components.',
      },
    ],
  },

  '/malware-analysis': {
    guideTitle: 'Static Malware Analysis & Triage Suite',
    guideSubtitle: 'Safe, client-side reverse engineering of Windows PE headers, hashes, and fuzzy similarities',
    howTo: [
      {
        step: 1,
        title: 'Select Binary File or Signature',
        description: 'Upload a Windows PE executable (.exe, .dll) or enter hash signatures for static triage.',
      },
      {
        step: 2,
        title: 'Parse Headers & Dependencies',
        description: 'Examine DOS/NT headers, section characteristics, compiler stamps, and imported Win32 API calls.',
      },
      {
        step: 3,
        title: 'Evaluate Indicators of Compromise (IoCs)',
        description: 'Check section entropy for packers/crypters and calculate TLSH fuzzy similarity distance against known threat samples.',
      },
    ],
    faqs: [
      {
        question: 'Is it safe to analyze malware files in CipherVerse?',
        answer: 'Yes. CipherVerse performs pure static analysis—parsing binary byte offsets and header structures without executing any instructions, ensuring your host operating system is never compromised.',
      },
      {
        question: 'What is a packed executable in malware analysis?',
        answer: 'A packed executable has its original code compressed or encrypted by a packer (like UPX or custom crypters). It unpacks itself into memory at runtime to evade static antivirus signature detection.',
      },
    ],
  },

  '/file-forensics': {
    guideTitle: 'Digital File Forensics & Integrity Suite',
    guideSubtitle: 'Cryptographic hashing, Shannon entropy distributions, and randomness audits',
    howTo: [
      {
        step: 1,
        title: 'Provide Evidence File',
        description: 'Drag and drop any file into the forensic workspace for in-memory analysis.',
      },
      {
        step: 2,
        title: 'Select Forensic Test Suite',
        description: 'Choose multi-algorithm concurrent hashing, Shannon entropy calculation, or NIST Chi-Square statistical randomness tests.',
      },
      {
        step: 3,
        title: 'Generate Chain-of-Custody Manifest',
        description: 'Export verifiable cryptographic checksums (SHA-256, SHA-512, MD5) and entropy score graphs.',
      },
    ],
    faqs: [
      {
        question: 'Why are cryptographic hashes vital in legal digital forensics?',
        answer: 'Hashes serve as digital fingerprints. By calculating and logging a file\'s SHA-256 hash at the time of seizure, investigators can prove in court that evidence was not modified or tampered with during analysis.',
      },
      {
        question: 'How does Shannon entropy identify hidden encryption?',
        answer: 'Normal files (executables, plaintext, images) have characteristic entropy values (between 3.0 and 6.5). Truly encrypted or compressed payloads produce near-maximum entropy (~7.9 to 8.0) because every byte value appears with equal probability.',
      },
    ],
  },

  '/utilities': {
    guideTitle: 'Cybersecurity Utilities Suite',
    guideSubtitle: 'Password strength estimation, JWT tokens, CSPRNG salts, and checksum calculators',
    howTo: [
      {
        step: 1,
        title: 'Choose Utility Tool',
        description: 'Select Password Strength to test credential entropy, JWT Signer to inspect web tokens, Salt Generator for CSPRNG bytes, or Fletcher-16 for data integrity.',
      },
      {
        step: 2,
        title: 'Configure Parameters & Inputs',
        description: 'Set token payloads, password phrases, byte lengths, or checksum inputs.',
      },
      {
        step: 3,
        title: 'Export Verified Output',
        description: 'Copy cryptographically secure tokens, verified claims, and entropy metrics.',
      },
    ],
    faqs: [
      {
        question: 'What makes a password resistant to modern GPU cracking attacks?',
        answer: 'Length and entropy. While complex 8-character passwords can be cracked in hours with dedicated GPU clusters running hashcat, multi-word passphrases with >70 bits of entropy take centuries to crack.',
      },
    ],
  },

  '/historical': {
    guideTitle: 'Historical Cryptographic Machines Suite',
    guideSubtitle: 'Simulating the mechanical and electrical cipher machines that shaped 20th-century warfare',
    howTo: [
      {
        step: 1,
        title: 'Select Historical Machine',
        description: 'Choose between the German Wehrmacht Enigma machine, Alan Turing\'s Bletchley Park Bombe crib solver, or the British 5-rotor Typex machine.',
      },
      {
        step: 2,
        title: 'Configure Mechanical Components',
        description: 'Set rotor types (I-V), initial ground settings, ringstellung ring offsets, and wire the plugboard (Steckerbrett).',
      },
      {
        step: 3,
        title: 'Simulate Keystrokes & Decoding',
        description: 'Type wartime intercepts to witness mechanical rotor stepping and observe electrical lampboard permutations.',
      },
    ],
    faqs: [
      {
        question: 'How did the Enigma machine work electrically?',
        answer: 'Pressing a key sent an electrical current through a plugboard, through three or four rotating wired rotors, into a fixed reflector, and back through the rotors via a different path to illuminate an output lamp.',
      },
      {
        question: 'What was the Turing Bombe?',
        answer: 'The Bombe was an electromechanical machine designed by Alan Turing and Gordon Welchman at Bletchley Park to rapidly deduce Enigma rotor setups by testing logical deductions based on suspected plaintext cribs.',
      },
    ],
  },

  // ==========================================
  // Individual Tool Pages
  // ==========================================

  // Classical Ciphers
  '/classical/caesar': {
    howTo: [
      {
        step: 1,
        title: 'Input Plaintext or Ciphertext',
        description: 'Enter the text you want to encrypt or decrypt into the main text area.',
      },
      {
        step: 2,
        title: 'Set Shift Key (0–25) or Enable Brute Force',
        description: 'Choose a numeric shift offset or toggle brute-force mode to test all 25 possible rotation keys simultaneously.',
      },
      {
        step: 3,
        title: 'Inspect Output & Frequency Distribution',
        description: 'View the shifted output instantly and review the letter frequency analysis comparing against standard English frequencies.',
      },
    ],
    faqs: [
      {
        question: 'What is the Caesar Cipher?',
        answer: 'The Caesar Cipher is a classic monoalphabetic substitution cipher where each letter in the plaintext is shifted by a fixed number of positions down the alphabet.',
      },
      {
        question: 'What is ROT13 and how is it related to Caesar Cipher?',
        answer: 'ROT13 is a specific Caesar cipher with a shift key of 13. Because the English alphabet has 26 letters, shifting twice by 13 results in a complete 26-position rotation, making encryption and decryption completely identical.',
      },
      {
        question: 'How do I decrypt a Caesar Cipher without knowing the key?',
        answer: 'Use the brute-force mode in CipherVerse to inspect all 25 possible shifts simultaneously and identify legible plain text automatically via English letter frequency analysis.',
      },
    ],
  },

  '/classical/vigenere': {
    howTo: [
      {
        step: 1,
        title: 'Enter Plaintext and Keyword',
        description: 'Type your secret message and provide an alphabetic keyword (e.g., "SECRET" or "CIPHER").',
      },
      {
        step: 2,
        title: 'Polyalphabetic Shift Matching',
        description: 'The keyword repeats cyclically across your message length to assign a distinct Caesar shift to each individual letter.',
      },
      {
        step: 3,
        title: 'Inspect Vigenère Square Result',
        description: 'Review the encrypted or decrypted text and observe the Tabula Recta row-column intersections.',
      },
    ],
    faqs: [
      {
        question: 'How does the Vigenère Cipher differ from a Caesar Cipher?',
        answer: 'While Caesar shifts every character by the same constant key, Vigenère uses a keyword to vary the shift per character, making simple frequency analysis much harder.',
      },
      {
        question: 'How is the Vigenère Cipher broken?',
        answer: 'Historically, Vigenère was broken using Kasiski examination and the Friedman test (index of coincidence) to determine key length, followed by frequency analysis on each Caesar sub-cipher.',
      },
    ],
  },

  '/classical/atbash': {
    howTo: [
      {
        step: 1,
        title: 'Input Message',
        description: 'Type or paste the message you want to encrypt or decrypt.',
      },
      {
        step: 2,
        title: 'Symmetric Alphabet Inversion',
        description: 'The tool maps the alphabet in reverse order: A becomes Z, B becomes Y, C becomes X, and vice-versa.',
      },
      {
        step: 3,
        title: 'Copy or Re-encode',
        description: 'Copy your result. Running Atbash a second time automatically restores the original text because Atbash is an involution.',
      },
    ],
    faqs: [
      {
        question: 'What is the origin of the Atbash Cipher?',
        answer: 'The Atbash cipher is a classical substitution cipher originally used for the Hebrew alphabet. It maps the first letter (Aleph) to the last (Tav), and the second (Bet) to the second-to-last (Shin).',
      },
      {
        question: 'Is a secret key required for Atbash?',
        answer: 'No secret key is required. The transformation is completely symmetric and deterministic based on the fixed reverse alphabet mapping.',
      },
    ],
  },

  '/classical/bacon': {
    howTo: [
      {
        step: 1,
        title: 'Enter Secret Text',
        description: 'Input the text to encode or the 5-character A/B steganographic sequence to decode.',
      },
      {
        step: 2,
        title: 'Select Baconian Alphabet Variant',
        description: 'Choose between the traditional 24-letter alphabet (I=J and U=V) or modern 26-letter standard.',
      },
      {
        step: 3,
        title: 'Generate Binary Steganographic Output',
        description: 'Inspect the resulting 5-character groups of A/B or binary digits ready for concealment in cover texts.',
      },
    ],
    faqs: [
      {
        question: 'What is Francis Bacon\'s Cipher?',
        answer: 'Invented by Francis Bacon in 1605, it is a binary steganographic system where each letter of the alphabet is represented by a 5-character group of two symbols (A and B).',
      },
      {
        question: 'How was Bacon\'s cipher used for steganography?',
        answer: 'In printed books, Bacon encoded messages by using two slightly different typographic fonts (typeface A and typeface B), making the secret message invisible to casual readers.',
      },
    ],
  },

  '/classical/bifid': {
    howTo: [
      {
        step: 1,
        title: 'Enter Message and Polybius Keyword',
        description: 'Type your message and enter a keyword to construct the 5x5 Polybius square matrix (I/J combined).',
      },
      {
        step: 2,
        title: 'Vertical Coordinate Fractionation',
        description: 'Each letter is mapped to (row, column) coordinates, written vertically underneath the plaintext.',
      },
      {
        step: 3,
        title: 'Read Horizontally & Substitute',
        description: 'The coordinate numbers are read horizontally in periodic groups and converted back into ciphertext letters.',
      },
    ],
    faqs: [
      {
        question: 'What makes the Bifid cipher unique in classical cryptography?',
        answer: 'Invented by Félix Delastelle around 1901, Bifid was the first classical cipher to achieve both substitution and transposition simultaneously through a process called fractionation.',
      },
      {
        question: 'Why does Bifid use a 5x5 Polybius square?',
        answer: 'A 5x5 grid contains 25 cells, exactly enough to hold the 26 letters of the English alphabet when I and J share a single cell.',
      },
    ],
  },

  '/classical/affine': {
    howTo: [
      {
        step: 1,
        title: 'Input Plaintext Message',
        description: 'Enter the text you want to encrypt using modular arithmetic.',
      },
      {
        step: 2,
        title: 'Set Multiplicative Key (a) and Shift Key (b)',
        description: 'Select a key "a" that is coprime to 26 (e.g. 1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25) and any shift "b" (0–25).',
      },
      {
        step: 3,
        title: 'Compute E(x) = (ax + b) mod 26',
        description: 'View the mathematically transformed ciphertext or compute modular inverses for decryption.',
      },
    ],
    faqs: [
      {
        question: 'Why must key "a" be coprime to 26 in the Affine cipher?',
        answer: 'If key "a" shares a common divisor with 26 (e.g. 2 or 13), distinct letters map to identical ciphertexts, making unique decryption mathematically impossible.',
      },
      {
        question: 'How is an Affine cipher decrypted?',
        answer: 'Decryption uses the modular multiplicative inverse: D(y) = a⁻¹ · (y - b) mod 26, where a · a⁻¹ ≡ 1 (mod 26).',
      },
    ],
  },

  '/classical/a1z26': {
    howTo: [
      {
        step: 1,
        title: 'Select Conversion Direction',
        description: 'Choose Text-to-Numbers (encryption) or Numbers-to-Text (decryption).',
      },
      {
        step: 2,
        title: 'Choose Number Separator',
        description: 'Select whether numbers are separated by hyphens (e.g. 3-9-16-8-5-18), spaces, or commas.',
      },
      {
        step: 3,
        title: 'Inspect Letter-Number Values',
        description: 'View the instant conversion where A=1, B=2, C=3, up to Z=26.',
      },
    ],
    faqs: [
      {
        question: 'What is the A1Z26 cipher?',
        answer: 'A1Z26 is a direct number substitution cipher where each alphabet letter is replaced by its 1-indexed position in the alphabet (A=1, Z=26).',
      },
      {
        question: 'Is A1Z26 considered secure?',
        answer: 'No. A1Z26 provides no cryptographic security and is purely a numerical encoding scheme used in puzzles, geocaching, and ARG games.',
      },
    ],
  },

  '/classical/rail-fence': {
    howTo: [
      {
        step: 1,
        title: 'Enter Text to Transpose',
        description: 'Provide the plaintext message to encode or zigzag ciphertext to reconstruct.',
      },
      {
        step: 2,
        title: 'Choose Rail Count',
        description: 'Select the number of rails (levels) for the zigzag path (typically 2 to 6 rails).',
      },
      {
        step: 3,
        title: 'Trace Zigzag Paths',
        description: 'The characters travel diagonally across the rails and are read row-by-row to form the transposed output.',
      },
    ],
    faqs: [
      {
        question: 'What type of cipher is the Rail Fence Cipher?',
        answer: 'It is a transposition cipher that alters the spatial positions of letters without changing their alphabetical identity, unlike substitution ciphers.',
      },
      {
        question: 'How secure is the Rail Fence Cipher?',
        answer: 'It provides virtually no modern security. An attacker can test small rail numbers in seconds or use anagramming to reconstruct the message.',
      },
    ],
  },

  '/classical/substitution': {
    howTo: [
      {
        step: 1,
        title: 'Input Text & Target Alphabet',
        description: 'Enter your message and specify a 26-character custom cipher alphabet (or generate a random permutation).',
      },
      {
        step: 2,
        title: 'Review Character Mapping Matrix',
        description: 'Check that every plaintext letter (A–Z) is uniquely mapped to exactly one cipher character without duplicates.',
      },
      {
        step: 3,
        title: 'Analyze Letter Frequencies',
        description: 'Compare ciphertext character counts against English frequency benchmarks to solve unknown keys.',
      },
    ],
    faqs: [
      {
        question: 'How many keys exist for a monoalphabetic substitution cipher?',
        answer: 'There are 26! (26 factorial) possible keys, which is approximately 4.03 × 10²⁶ keys—far too large for manual brute-force.',
      },
      {
        question: 'If the key space is so large, why is monoalphabetic substitution easy to crack?',
        answer: 'Because letter frequencies are preserved. In English, \'E\' occurs ~12.7% of the time, followed by \'T\', \'A\', and \'O\'. Frequency analysis breaks simple substitution rapidly.',
      },
    ],
  },

  // Encoding & Decoding
  '/encoding/base64': {
    howTo: [
      {
        step: 1,
        title: 'Input Text or Binary String',
        description: 'Enter plain text, ASCII characters, or hexadecimal bytes to convert.',
      },
      {
        step: 2,
        title: 'Select Encoding Variant',
        description: 'Choose Standard Base64 (with + and /) or URL-Safe Base64 (with - and _).',
      },
      {
        step: 3,
        title: 'Instant Conversion & Copy',
        description: 'View the padded radix-64 representation and copy the result with one click.',
      },
    ],
    faqs: [
      {
        question: 'What is Base64 encoding?',
        answer: 'Base64 is a binary-to-text encoding scheme that represents binary data in an ASCII string format by translating it into a radix-64 representation.',
      },
      {
        question: 'Is Base64 an encryption algorithm?',
        answer: 'No. Base64 is purely an encoding format to transfer binary data over text-only protocols. It provides zero confidentiality and can be decoded by anyone instantly.',
      },
      {
        question: 'What is URL-safe Base64?',
        answer: 'Standard Base64 uses + and / characters which have reserved meanings in URLs. URL-safe Base64 substitutes - and _ in their place to prevent routing conflicts.',
      },
    ],
  },

  '/encoding/hex': {
    howTo: [
      {
        step: 1,
        title: 'Input Plaintext or Hex Bytes',
        description: 'Enter readable text to encode, or space-separated / continuous hex characters (e.g. "48 65 6c 6c 6f") to decode.',
      },
      {
        step: 2,
        title: 'Select Delimiter & Format',
        description: 'Choose formatting options including spaces, colons, 0x prefixes, or uppercase hex characters.',
      },
      {
        step: 3,
        title: 'Convert & Inspect Byte Array',
        description: 'Inspect hexadecimal byte values and corresponding ASCII/UTF-8 character representations.',
      },
    ],
    faqs: [
      {
        question: 'What is Hexadecimal encoding?',
        answer: 'Hexadecimal represents binary byte values in base-16 using digits 0-9 and letters A-F, with each byte represented by two hex characters.',
      },
      {
        question: 'Why do developers use Hex encoding?',
        answer: 'Hexadecimal provides a human-readable representation of raw computer memory, network packets, cryptographic digests, and compiled binary bytes.',
      },
    ],
  },

  '/encoding/url': {
    howTo: [
      {
        step: 1,
        title: 'Paste URL or Query Parameter',
        description: 'Enter a full website URL, query parameter, or text containing special characters.',
      },
      {
        step: 2,
        title: 'Select Component vs Full URI Mode',
        description: 'Choose encodeURIComponent (escapes symbols like ? and &) or encodeURI (preserves URL protocol and slash delimiters).',
      },
      {
        step: 3,
        title: 'Copy Percent-Encoded Output',
        description: 'View the safely escaped URI string where spaces become %20 and special characters are safely escaped.',
      },
    ],
    faqs: [
      {
        question: 'What is URL percent-encoding?',
        answer: 'Defined in RFC 3986, percent-encoding replaces unsafe ASCII characters and non-ASCII bytes with a \'%\' followed by their two-digit hexadecimal value.',
      },
      {
        question: 'When should I use encodeURIComponent vs encodeURI?',
        answer: 'Use encodeURI when encoding a complete URL (preserving http:// and slashes). Use encodeURIComponent when encoding individual parameter values so query delimiters like \'=\' and \'&\' do not corrupt query parsing.',
      },
    ],
  },

  '/encoding/binary': {
    howTo: [
      {
        step: 1,
        title: 'Select Conversion Direction',
        description: 'Choose Text-to-Binary to convert ASCII/UTF-8 into bits, or Binary-to-Text to decode 0s and 1s.',
      },
      {
        step: 2,
        title: 'Set Byte Delimiters',
        description: 'Choose space-separated 8-bit byte groups (e.g. 01001000 01101001) or continuous raw bitstreams.',
      },
      {
        step: 3,
        title: 'Inspect Bit Pattern Output',
        description: 'View the exact machine-level binary byte representations.',
      },
    ],
    faqs: [
      {
        question: 'How does ASCII translate into binary bytes?',
        answer: 'Each ASCII character corresponds to a numerical value between 0 and 127, which is represented as an 8-bit byte (e.g., \'A\' = 65 = 01000001).',
      },
    ],
  },

  '/encoding/morse': {
    howTo: [
      {
        step: 1,
        title: 'Input Text or Morse Dots/Dashes',
        description: 'Enter plain English text or input Morse code using dots (.), dashes (-), and slashes (/) for word boundaries.',
      },
      {
        step: 2,
        title: 'Configure Playback Speed (WPM)',
        description: 'Adjust audio frequency (Hz) and Words Per Minute (WPM) playback speed.',
      },
      {
        step: 3,
        title: 'Play Audio Tone or Optical Light',
        description: 'Listen to the synthesized audio beeps or watch the synchronized optical light simulation.',
      },
    ],
    faqs: [
      {
        question: 'What are the timing standards in International Morse Code?',
        answer: 'A dash is three times the duration of a dot. The space between parts of the same letter is 1 dot duration, between letters is 3 dot durations, and between words is 7 dot durations.',
      },
      {
        question: 'Who invented Morse code?',
        answer: 'Samuel Morse and Alfred Vail developed the electric telegraph and Morse code in the 1830s and 1840s to transmit text across long distances via electrical pulses.',
      },
    ],
  },

  // Symmetric Cryptography
  '/symmetric/aes': {
    howTo: [
      {
        step: 1,
        title: 'Input Text or Hex Payload',
        description: 'Paste your plaintext message or hexadecimal ciphertext into the input panel.',
      },
      {
        step: 2,
        title: 'Select Key Size (128, 192, 256) & Mode',
        description: 'Choose AES-128, AES-192, or AES-256 and select an operational mode like GCM (authenticated) or CBC.',
      },
      {
        step: 3,
        title: 'Generate or Input Key & IV',
        description: 'Provide your secret key and Initialization Vector (IV/Nonce), or use the built-in cryptographically secure generator.',
      },
    ],
    faqs: [
      {
        question: 'What is AES (Advanced Encryption Standard)?',
        answer: 'AES is a symmetric block cipher established by the U.S. NIST in 2001. It processes 128-bit blocks of data using cryptographic keys of 128, 192, or 256 bits and is the global standard for secure encryption.',
      },
      {
        question: 'What is the difference between AES-GCM and AES-CBC?',
        answer: 'AES-CBC provides confidentiality but requires padding and a separate HMAC for integrity. AES-GCM provides Authenticated Encryption with Associated Data (AEAD), combining high-performance encryption and message authentication in one pass.',
      },
      {
        question: 'Why is an Initialization Vector (IV) required?',
        answer: 'An IV ensures that identical plaintexts encrypted with the same key produce completely unique ciphertexts, preventing pattern leakage and replay attacks.',
      },
    ],
  },

  '/symmetric/des': {
    howTo: [
      {
        step: 1,
        title: 'Enter 64-bit Block or Text',
        description: 'Input plaintext or ciphertext to process with the Data Encryption Standard.',
      },
      {
        step: 2,
        title: 'Provide 56-bit Key',
        description: 'Input an 8-byte key (56 effective key bits plus 8 parity bits).',
      },
      {
        step: 3,
        title: 'Execute 16 Feistel Rounds',
        description: 'View the 16 rounds of S-box substitution, subkey generation, and permutation results.',
      },
    ],
    faqs: [
      {
        question: 'Why is standard DES obsolete?',
        answer: 'DES uses a 56-bit key size, which can be brute-forced in less than a day using modern hardware. It was superseded by Triple DES and ultimately AES.',
      },
      {
        question: 'What is the Feistel structure in DES?',
        answer: 'The Feistel structure splits the 64-bit block into left and right 32-bit halves, iteratively applying a round function so that encryption and decryption logic are nearly identical.',
      },
    ],
  },

  '/symmetric/3des': {
    howTo: [
      {
        step: 1,
        title: 'Input Plaintext or Ciphertext Blocks',
        description: 'Enter data to process through the Triple Data Encryption Standard.',
      },
      {
        step: 2,
        title: 'Select 2-Key or 3-Key EDE Mode',
        description: 'Choose 2-Key 3DES (112-bit effective key) or 3-Key 3DES (168-bit effective key) using Encrypt-Decrypt-Encrypt sequence.',
      },
      {
        step: 3,
        title: 'Compute Triple Feistel Transformation',
        description: 'View the processed 64-bit blocks with legacy backward compatibility.',
      },
    ],
    faqs: [
      {
        question: 'How does Triple DES (3DES) work?',
        answer: 'Triple DES applies DES three times: Encrypt with Key 1, Decrypt with Key 2, and Encrypt with Key 3. If Key 1 equals Key 2, it falls back to standard single DES.',
      },
      {
        question: 'Why was Triple DES phased out despite its 112/168 bit key?',
        answer: 'Because of its small 64-bit block size. Under high-throughput TLS traffic, the birthday bound can be reached after only 32 GB of data (Sweet32 attack), allowing collision recovery.',
      },
    ],
  },

  '/symmetric/blowfish': {
    howTo: [
      {
        step: 1,
        title: 'Input Data & Key (Up to 448 Bits)',
        description: 'Provide message text and any secret key from 32 to 448 bits in length.',
      },
      {
        step: 2,
        title: 'Select Mode of Operation',
        description: 'Choose ECB or CBC mode with optional initialization vector.',
      },
      {
        step: 3,
        title: 'Execute 16-Round Feistel Network',
        description: 'Inspect the transformed 64-bit blocks through key-dependent S-boxes.',
      },
    ],
    faqs: [
      {
        question: 'Who designed the Blowfish cipher?',
        answer: 'Bruce Schneier designed Blowfish in 1993 as a fast, free, unpatented alternative to DES. It is notable for its large key-dependent S-boxes.',
      },
    ],
  },

  '/symmetric/rc2': {
    howTo: [
      {
        step: 1,
        title: 'Input Message Payload',
        description: 'Enter data to encrypt or decrypt with Ron Rivest\'s 64-bit block cipher.',
      },
      {
        step: 2,
        title: 'Configure Effective Key Bits',
        description: 'Set your secret key and configure effective key bits (from 8 to 1024 bits).',
      },
      {
        step: 3,
        title: 'Execute Mixing & Mashing Rounds',
        description: 'View output after 16 mixing and mashing rounds.',
      },
    ],
    faqs: [
      {
        question: 'What is RC2?',
        answer: 'RC2 is a 64-bit block cipher designed in 1987 by Ron Rivest for RSA Security, featuring variable key sizes for export compliance.',
      },
    ],
  },

  '/symmetric/rc4': {
    howTo: [
      {
        step: 1,
        title: 'Enter Stream Plaintext or Hex',
        description: 'Input the data to encrypt or decrypt with the RC4 stream cipher.',
      },
      {
        step: 2,
        title: 'Set Secret Key (1–256 Bytes)',
        description: 'Provide an encryption passphrase of any arbitrary length.',
      },
      {
        step: 3,
        title: 'Generate Keystream & Bitwise XOR',
        description: 'Observe the Key Scheduling Algorithm (KSA) permutation and PRGA keystream output.',
      },
    ],
    faqs: [
      {
        question: 'How does RC4 generate its pseudorandom keystream?',
        answer: 'RC4 initializes a 256-byte state array with the key (KSA), then continuously swaps array elements using two index pointers (PRGA) to output one byte per cycle.',
      },
    ],
  },

  '/symmetric/rc4-drop': {
    howTo: [
      {
        step: 1,
        title: 'Input Message Data & Key',
        description: 'Provide the data payload and secret passphrase.',
      },
      {
        step: 2,
        title: 'Set Initial Drop Count (N)',
        description: 'Specify how many initial keystream bytes to discard (commonly 768 or 3072 bytes).',
      },
      {
        step: 3,
        title: 'Stream Encryption Post-Drop',
        description: 'Encrypt or decrypt using the stabilized PRGA state with initial bias eliminated.',
      },
    ],
    faqs: [
      {
        question: 'Why discard initial keystream bytes in RC4-drop?',
        answer: 'Standard RC4 has statistical biases in its first few hundred bytes that correlate with the key. Discarding the first N bytes removes this vulnerability.',
      },
    ],
  },

  '/symmetric/sm4': {
    howTo: [
      {
        step: 1,
        title: 'Input 128-bit Data Block',
        description: 'Enter plaintext or ciphertext blocks.',
      },
      {
        step: 2,
        title: 'Provide 128-bit Key & Mode',
        description: 'Input a 16-byte key and choose ECB or CBC mode.',
      },
      {
        step: 3,
        title: 'Execute 32 Unbalanced Feistel Rounds',
        description: 'View block encryption conforming to Chinese National Standard GB/T 32907-2016.',
      },
    ],
    faqs: [
      {
        question: 'What is the SM4 block cipher?',
        answer: 'SM4 is the Chinese National Standard 128-bit block cipher used for wireless LAN (WAPI) and authorized commercial cryptographic applications in China.',
      },
    ],
  },

  '/symmetric/ciphersaber2': {
    howTo: [
      {
        step: 1,
        title: 'Enter Secret Message & Password',
        description: 'Type your message and provide a shared secret passphrase.',
      },
      {
        step: 2,
        title: 'Set Mixing Rounds (Default 20)',
        description: 'Configure the number of KSA key schedule mixing iterations.',
      },
      {
        step: 3,
        title: 'Prepend 10-Byte Random IV',
        description: 'Generates ciphertext with a randomly generated 10-byte IV prepended to the output.',
      },
    ],
    faqs: [
      {
        question: 'What is CipherSaber-2?',
        answer: 'CipherSaber-2 is an enhanced version of Arnold Reinhold\'s CipherSaber protocol that runs 20 rounds of key scheduling to eliminate RC4 key setup weaknesses.',
      },
    ],
  },

  '/symmetric/xor': {
    howTo: [
      {
        step: 1,
        title: 'Input Raw Text or Hex Bytes',
        description: 'Enter your string or hex values to process.',
      },
      {
        step: 2,
        title: 'Choose Key or Brute Force Mode',
        description: 'Enter a single-byte (0–255) or multi-byte key, or run XOR Brute Force to test all 256 single-byte keys automatically.',
      },
      {
        step: 3,
        title: 'Inspect Frequency-Scored Results',
        description: 'Review the bitwise XOR output and frequency-scored candidate plaintexts ranked by English readability.',
      },
    ],
    faqs: [
      {
        question: 'What is the fundamental property of bitwise XOR in cryptography?',
        answer: 'XOR is self-inverting: A ⊕ B ⊕ B = A. Applying XOR with the same key twice restores the original value, making encryption and decryption identical.',
      },
      {
        question: 'Is XOR encryption secure?',
        answer: 'A One-Time Pad using a truly random, non-repeating key equal in length to the message is mathematically unbreakable. However, repeating short keys is vulnerable to frequency analysis.',
      },
    ],
  },

  // Asymmetric Cryptography
  '/asymmetric/rsa': {
    howTo: [
      {
        step: 1,
        title: 'Generate or Paste Keypair',
        description: 'Generate a new 1024, 2048, or 4096-bit RSA PEM keypair or paste an existing private/public key.',
      },
      {
        step: 2,
        title: 'Choose Operation Mode',
        description: 'Select Public Key Encryption, Private Key Decryption, Digital Signature Creation, or Signature Verification.',
      },
      {
        step: 3,
        title: 'Inspect Output or Signature Status',
        description: 'View the Base64-encoded ciphertext or verify RSA-PSS / PKCS#1 v1.5 signature validity.',
      },
    ],
    faqs: [
      {
        question: 'What is RSA asymmetric cryptography?',
        answer: 'RSA is a public-key cryptosystem that relies on the computational difficulty of factoring large composite integers. It enables secure communication without sharing secret keys in advance.',
      },
      {
        question: 'What RSA key size is secure for production use?',
        answer: 'NIST recommends a minimum key size of 2048 bits for current applications, with 4096 bits recommended for long-term security archives.',
      },
      {
        question: 'Can RSA be used to encrypt large files directly?',
        answer: 'No. RSA can only encrypt plaintexts smaller than its key modulus. In practice, RSA is used in hybrid encryption: encrypting a fast symmetric key (like AES-256) which encrypts the file.',
      },
    ],
  },

  '/asymmetric/dsa': {
    howTo: [
      {
        step: 1,
        title: 'Generate or Load DSA Parameters',
        description: 'Configure prime modulus p, subprime q, and generator g, and generate a DSA keypair.',
      },
      {
        step: 2,
        title: 'Hash & Sign Message',
        description: 'Compute SHA-256 digest and sign with the private key to produce signature pair (r, s).',
      },
      {
        step: 3,
        title: 'Verify Signature Proof',
        description: 'Confirm mathematical signature validity against the public key.',
      },
    ],
    faqs: [
      {
        question: 'Can DSA be used for data encryption?',
        answer: 'No. DSA (Digital Signature Algorithm, NIST FIPS 186) is strictly a digital signature mechanism. It does not support encryption or key exchange.',
      },
    ],
  },

  // Hashing
  '/hashing': {
    howTo: [
      {
        step: 1,
        title: 'Input Message or File Payload',
        description: 'Type text, paste hexadecimal bytes, or enter secret keys for message authentication.',
      },
      {
        step: 2,
        title: 'Select Hash or KDF Function',
        description: 'Choose between SHA-256, SHA-512, SHA-3, MD5, HMAC-SHA256, or PBKDF2/Scrypt/Bcrypt key derivation.',
      },
      {
        step: 3,
        title: 'Inspect Cryptographic Digest',
        description: 'View the fixed-length hexadecimal digest and copy checksums for verification.',
      },
    ],
    faqs: [
      {
        question: 'What is a cryptographic hash function?',
        answer: 'A cryptographic hash function takes arbitrary data as input and produces a fixed-size deterministic string (digest). It is computationally infeasible to reverse or find two distinct inputs with the same hash.',
      },
      {
        question: 'Can a SHA-256 hash be decrypted or reversed?',
        answer: 'No. Hashes are mathematical one-way functions, not ciphers. The only way to find the original input is brute force or dictionary lookup of precomputed hashes (rainbow tables).',
      },
      {
        question: 'Why should MD5 and SHA-1 no longer be used for security?',
        answer: 'Both MD5 and SHA-1 have proven collision vulnerabilities, meaning attackers can generate two different files with the exact same hash. Use SHA-256, SHA-512, or SHA-3 instead.',
      },
    ],
  },

  // Certificates & TLS
  '/certificates/x509': {
    howTo: [
      {
        step: 1,
        title: 'Paste PEM Certificate Block',
        description: 'Paste your -----BEGIN CERTIFICATE----- block or upload a .crt/.cer file.',
      },
      {
        step: 2,
        title: 'Parse ASN.1 Structure',
        description: 'Extracts Subject Common Name, Organization, Issuer CA, Serial Number, and Validity Period.',
      },
      {
        step: 3,
        title: 'Inspect Subject Alternative Names (SANs)',
        description: 'Verify domain names, public key algorithm (RSA/ECC), and signature algorithm.',
      },
    ],
    faqs: [
      {
        question: 'What is an X.509 certificate?',
        answer: 'An X.509 certificate is a digital document that binds a public key to an entity\'s identity, verified by a trusted Certificate Authority (CA) for HTTPS and TLS.',
      },
    ],
  },

  '/certificates/tls': {
    howTo: [
      {
        step: 1,
        title: 'Input Hostname & Port',
        description: 'Enter a domain (e.g. google.com) and TLS port (default 443).',
      },
      {
        step: 2,
        title: 'Inspect Handshake Sequence',
        description: 'Observe ClientHello, ServerHello, TLS protocol negotiation (TLS 1.2 vs 1.3), and ALPN.',
      },
      {
        step: 3,
        title: 'Review Cipher Suite Security',
        description: 'Confirm perfect forward secrecy (ECDHE) and AEAD cipher support.',
      },
    ],
    faqs: [
      {
        question: 'What is Perfect Forward Secrecy (PFS)?',
        answer: 'PFS ensures that session keys are negotiated ephemerally (via ECDHE). Even if a server\'s long-term private key is compromised in the future, past recorded sessions cannot be decrypted.',
      },
    ],
  },

  '/certificates/fingerprint': {
    howTo: [
      {
        step: 1,
        title: 'Paste Certificate Text',
        description: 'Provide your X.509 PEM certificate string.',
      },
      {
        step: 2,
        title: 'Calculate SHA-256 & SHA-1 Hashes',
        description: 'Computes cryptographic digests of the DER-encoded binary certificate payload.',
      },
      {
        step: 3,
        title: 'Copy Fingerprint Thumbprints',
        description: 'Use formatted fingerprint strings for certificate pinning in mobile apps and APIs.',
      },
    ],
    faqs: [
      {
        question: 'What is certificate pinning?',
        answer: 'Certificate pinning is a security mechanism where client applications hardcode the expected certificate fingerprint, rejecting rogue certificates issued by compromised CAs.',
      },
    ],
  },

  // Blockchain
  '/blockchain/bitcoin': {
    howTo: [
      {
        step: 1,
        title: 'Paste Bitcoin Address',
        description: 'Enter a Legacy (1...), SegWit (3...), or Bech32/Taproot (bc1...) address.',
      },
      {
        step: 2,
        title: 'Automatic Script & Checksum Detection',
        description: 'The tool decodes Base58Check or BIP173 BCH checksums to verify mathematical validity.',
      },
      {
        step: 3,
        title: 'Inspect Address Metadata',
        description: 'View the network type (Mainnet/Testnet), witness version, and public key hash (Hash160).',
      },
    ],
    faqs: [
      {
        question: 'What does a Bitcoin address checksum protect against?',
        answer: 'The 4-byte Base58Check or Bech32 checksum ensures that any typo, omitted character, or transcription error is caught immediately before sending funds.',
      },
    ],
  },

  '/blockchain/ethereum': {
    howTo: [
      {
        step: 1,
        title: 'Input 0x Ethereum Address',
        description: 'Paste a 42-character hexadecimal Ethereum public address.',
      },
      {
        step: 2,
        title: 'Verify Length & Characters',
        description: 'The validator verifies 20-byte length, hexadecimal characters, and prefix format.',
      },
      {
        step: 3,
        title: 'Validate EIP-55 Mixed-Case Checksum',
        description: 'Computes Keccak-256 hash to confirm valid capitalization and prevent accidental typographical errors.',
      },
    ],
    faqs: [
      {
        question: 'What is an EIP-55 checksummed Ethereum address?',
        answer: 'EIP-55 uses mixed-case letters based on the Keccak-256 hash of the lowercase address. If a single character is miskeyed, the capitalization check fails.',
      },
    ],
  },

  '/blockchain/merkle': {
    howTo: [
      {
        step: 1,
        title: 'Enter Transaction Hashes',
        description: 'Provide a list of leaf data blocks or transaction IDs.',
      },
      {
        step: 2,
        title: 'Generate Binary Tree',
        description: 'The engine recursively pairs hashes and computes SHA-256 parent digests up to the Merkle Root.',
      },
      {
        step: 3,
        title: 'Verify Inclusion Proofs',
        description: 'Audit individual transaction paths to prove membership without downloading the entire dataset.',
      },
    ],
    faqs: [
      {
        question: 'Why are Merkle trees crucial for blockchain scalability?',
        answer: 'They allow light clients to verify that a transaction is included in a block using only O(log N) hashes instead of downloading the entire block history.',
      },
    ],
  },

  '/blockchain/wif': {
    howTo: [
      {
        step: 1,
        title: 'Input 256-bit Private Key Hex',
        description: 'Paste a 64-character hexadecimal raw ECDSA private key.',
      },
      {
        step: 2,
        title: 'Choose Network & Compression',
        description: 'Select Mainnet (0x80) or Testnet (0xEF), and specify compressed or uncompressed public key.',
      },
      {
        step: 3,
        title: 'Generate Base58 WIF Key',
        description: 'Computes network prefix, optional 0x01 compression byte, and double-SHA256 checksum to export WIF.',
      },
    ],
    faqs: [
      {
        question: 'What is Bitcoin WIF (Wallet Import Format)?',
        answer: 'WIF is a Base58Check-encoded private key format that makes private keys easier to copy and import into wallets with built-in checksum error prevention.',
      },
    ],
  },

  // Steganography
  '/steganography/image': {
    howTo: [
      {
        step: 1,
        title: 'Upload Carrier Image (PNG/BMP)',
        description: 'Select a lossless cover image to act as the carrier for your secret payload.',
      },
      {
        step: 2,
        title: 'Enter Secret Message & Passphrase',
        description: 'Type the confidential text to hide and optionally set an encryption password to secure the payload.',
      },
      {
        step: 3,
        title: 'Download Stego Image or Decode',
        description: 'The tool modifies the Least Significant Bits (LSB) of pixel channels without visible alteration, exporting a pristine stego image.',
      },
    ],
    faqs: [
      {
        question: 'Why are PNG images preferred over JPEG for steganography?',
        answer: 'PNG uses lossless compression, preserving exact pixel color values. JPEG uses lossy compression that alters low-order pixel bits, corrupting LSB steganographic payloads.',
      },
      {
        question: 'How much data can be hidden in an image?',
        answer: 'With 1-bit LSB encoding across 3 color channels (RGB), an image can store approximately 3 bits per pixel (e.g. a 1920x1080 image can hide up to ~777 KB of data).',
      },
    ],
  },

  '/steganography/text': {
    howTo: [
      {
        step: 1,
        title: 'Enter Innocent Cover Text',
        description: 'Provide an ordinary paragraph of public text to carry the hidden information.',
      },
      {
        step: 2,
        title: 'Enter Secret Message',
        description: 'Type the secret string you want to conceal invisibly.',
      },
      {
        step: 3,
        title: 'Copy Invisible Steganographic Text',
        description: 'The tool encodes your secret into binary and injects zero-width non-joiner unicode characters into the cover text.',
      },
    ],
    faqs: [
      {
        question: 'What are zero-width characters in text steganography?',
        answer: 'Zero-width characters (like U+200B and U+200C) are valid Unicode characters that occupy zero visual space, allowing secret binary data to hide inside ordinary visible text.',
      },
    ],
  },

  '/steganography/audio': {
    howTo: [
      {
        step: 1,
        title: 'Upload Uncompressed WAV Audio',
        description: 'Select an uncompressed 16-bit PCM WAV audio file to act as the acoustic carrier.',
      },
      {
        step: 2,
        title: 'Enter Payload & Secret Passphrase',
        description: 'Type your message and set an optional encryption key.',
      },
      {
        step: 3,
        title: 'Encode or Decode Waveform Samples',
        description: 'Modifies the least significant bits of audio PCM samples without causing audible acoustic distortion.',
      },
    ],
    faqs: [
      {
        question: 'Why does audio steganography require WAV instead of MP3?',
        answer: 'MP3 compression discards inaudible sound data and recalculates waveform coefficients, destroying embedded LSB bits. Lossless WAV audio preserves raw sample values.',
      },
    ],
  },

  // Malware & Forensics
  '/malware-analysis/hash': {
    howTo: [
      {
        step: 1,
        title: 'Input Known File Hash',
        description: 'Paste an MD5, SHA-1, or SHA-256 hash signature from a security alert or log.',
      },
      {
        step: 2,
        title: 'Detect Hash Format & Family',
        description: 'Identifies hash bit-length and algorithm standard.',
      },
      {
        step: 3,
        title: 'Cross-Reference Threat Intelligence',
        description: 'Evaluate against known ransomware strains, malware families, or benign software registries.',
      },
    ],
    faqs: [
      {
        question: 'What is a file hash in threat intelligence?',
        answer: 'A cryptographic hash acts as an immutable digital signature for a file, allowing security analysts to detect malware strains across network telemetry without transferring the full binary.',
      },
    ],
  },

  '/malware-analysis/tlsh': {
    howTo: [
      {
        step: 1,
        title: 'Input Two TLSH Fuzzy Hashes',
        description: 'Provide Trend Micro Locality Sensitive Hash strings for comparison.',
      },
      {
        step: 2,
        title: 'Compute Statistical Distance Score',
        description: 'Calculates the distance difference between byte distributions and quartiles.',
      },
      {
        step: 3,
        title: 'Interpret Similarity Confidence',
        description: 'Scores < 30 indicate high confidence of variant similarity; scores > 100 indicate distinct software.',
      },
    ],
    faqs: [
      {
        question: 'How does TLSH differ from traditional cryptographic hashes?',
        answer: 'In SHA-256, changing a single bit changes the entire hash randomly (avalanche effect). In TLSH, minor code changes produce very similar hashes, making it ideal for tracking mutated malware variants.',
      },
    ],
  },

  '/malware-analysis/pe': {
    howTo: [
      {
        step: 1,
        title: 'Upload Windows Executable',
        description: 'Select a PE binary (.exe, .dll, .sys) for static forensic inspection.',
      },
      {
        step: 2,
        title: 'Parse Headers & Data Directories',
        description: 'Inspect the DOS header (MZ magic), COFF file header, optional header, and section headers.',
      },
      {
        step: 3,
        title: 'Evaluate Section Entropy & Imports',
        description: 'Analyze suspicious imported API calls (e.g. VirtualAlloc, WriteProcessMemory) and section entropy indicating packers.',
      },
    ],
    faqs: [
      {
        question: 'What is a PE (Portable Executable) file?',
        answer: 'PE is the standard binary format used by Windows for executables, DLLs, and kernel drivers, defining headers, sections, and dynamic import tables.',
      },
      {
        question: 'How does high section entropy indicate packed malware?',
        answer: 'Standard compiled code has an entropy of ~6.0. An entropy score above 7.0 strongly indicates encrypted, compressed, or packed malware designed to evade antivirus signatures.',
      },
    ],
  },

  '/file-forensics/hash': {
    howTo: [
      {
        step: 1,
        title: 'Select File for Hashing',
        description: 'Drag and drop any file directly into the client-side forensic hasher.',
      },
      {
        step: 2,
        title: 'Generate Multiple Cryptographic Checksums',
        description: 'Streaming workers calculate SHA-256, SHA-512, MD5, and SHA-1 simultaneously without uploading files to any external server.',
      },
      {
        step: 3,
        title: 'Verify File Integrity & Signatures',
        description: 'Compare calculated digests against vendor software releases, download checksums, or threat intelligence feeds.',
      },
    ],
    faqs: [
      {
        question: 'Does CipherVerse upload my file to a server to calculate hashes?',
        answer: 'No. All file hashing is performed locally inside your browser memory using Web Cryptography API streaming workers for maximum privacy and security.',
      },
    ],
  },

  '/file-forensics/multi-hash': {
    howTo: [
      {
        step: 1,
        title: 'Drag & Drop File',
        description: 'Provide any local file to compute multi-algorithm cryptographic verification.',
      },
      {
        step: 2,
        title: 'Compute 6+ Hashes in Streaming Pass',
        description: 'Calculates MD5, SHA-1, SHA-256, SHA-384, SHA-512, and CRC32 concurrently.',
      },
      {
        step: 3,
        title: 'Export Forensic Checksum Manifest',
        description: 'Copy or export verified hash tables for audit logs and legal custody reports.',
      },
    ],
    faqs: [
      {
        question: 'Why generate multiple hash algorithms simultaneously?',
        answer: 'Different security systems and forensic databases use different hash standards. Calculating multiple digests in one streaming pass avoids reading the file multiple times.',
      },
    ],
  },

  '/file-forensics/entropy': {
    howTo: [
      {
        step: 1,
        title: 'Upload File or Paste Raw Bytes',
        description: 'Select any binary file or hex string for forensic randomness measurement.',
      },
      {
        step: 2,
        title: 'Compute Shannon Entropy Metric',
        description: 'Calculates the degree of randomness on a scale from 0.0 (completely uniform) to 8.0 (pure cryptographic randomness).',
      },
      {
        step: 3,
        title: 'Review Byte Density Distribution',
        description: 'Inspect the histogram of byte frequencies to distinguish between plaintext, compressed data, and encrypted ciphers.',
      },
    ],
    faqs: [
      {
        question: 'What is Shannon Entropy in digital forensics?',
        answer: 'Shannon entropy measures the average information density per byte. It is defined as H(X) = -Σ P(x) log₂(P(x)) and ranges from 0 to 8 bits per byte.',
      },
    ],
  },

  '/file-forensics/randomness': {
    howTo: [
      {
        step: 1,
        title: 'Input Binary Data or PRNG Output',
        description: 'Upload a binary sample or supply byte streams to evaluate.',
      },
      {
        step: 2,
        title: 'Run NIST Statistical Tests',
        description: 'Executes Chi-Square distribution test, Monte Carlo Pi calculation, and serial correlation.',
      },
      {
        step: 3,
        title: 'Inspect P-Values & Bias Diagnostics',
        description: 'Verify whether the byte distribution exhibits statistical randomness suitable for cryptographic key generation.',
      },
    ],
    faqs: [
      {
        question: 'What does the Chi-Square test measure in randomness?',
        answer: 'The Chi-Square test compares observed byte frequencies against the expected uniform distribution. A p-value between 0.01 and 0.99 indicates acceptable cryptographic randomness.',
      },
    ],
  },

  // Utilities
  '/utilities/password': {
    howTo: [
      {
        step: 1,
        title: 'Enter Password to Evaluate',
        description: 'Type a candidate password or passphrase into the secure evaluation input.',
      },
      {
        step: 2,
        title: 'Calculate Information Entropy',
        description: 'The engine evaluates character pool size, length, dictionary patterns, and zxcvbn strength metrics.',
      },
      {
        step: 3,
        title: 'View Estimated Crack Time',
        description: 'Review estimated cracking times against supercomputers, offline GPU clusters, and online rate-limited attackers.',
      },
    ],
    faqs: [
      {
        question: 'How many bits of password entropy are considered secure?',
        answer: 'Passphrases with at least 60–80 bits of entropy are considered resistant to offline GPU dictionary attacks, while 128 bits matches symmetric encryption strength.',
      },
    ],
  },

  '/utilities/jwt': {
    howTo: [
      {
        step: 1,
        title: 'Paste JWT or Create Claims',
        description: 'Enter an existing three-part JWT or define JSON payload claims (iss, sub, exp).',
      },
      {
        step: 2,
        title: 'Configure Signing Secret & Algorithm',
        description: 'Select HMAC-SHA256 (HS256) or RSA (RS256) and provide your secret key or private PEM.',
      },
      {
        step: 3,
        title: 'Inspect Decoded Token & Signature Validity',
        description: 'View the color-coded Header, Payload, and verified cryptographic signature status.',
      },
    ],
    faqs: [
      {
        question: 'What are the three parts of a JSON Web Token (JWT)?',
        answer: 'A JWT consists of Header (algorithm & token type), Payload (claims and data), and Signature (cryptographic verification), separated by dots.',
      },
    ],
  },

  '/utilities/salt': {
    howTo: [
      {
        step: 1,
        title: 'Specify Desired Byte Length',
        description: 'Choose 16, 32, or 64 bytes depending on cryptographic algorithm requirements.',
      },
      {
        step: 2,
        title: 'Select Output Encoding Format',
        description: 'Choose Hexadecimal, Base64, or URL-safe Base64 representation.',
      },
      {
        step: 3,
        title: 'Generate Cryptographically Secure Salt',
        description: 'Generates nonces using OS kernel entropy via Web Cryptography CSPRNG.',
      },
    ],
    faqs: [
      {
        question: 'What is a cryptographic salt and why is it essential?',
        answer: 'A salt is random data appended to a password before hashing. It ensures that identical passwords have completely different hashes, neutralizing precomputed rainbow tables.',
      },
    ],
  },

  '/utilities/fletcher16': {
    howTo: [
      {
        step: 1,
        title: 'Input Data String or Bytes',
        description: 'Enter the message string or hexadecimal bytes to checksum.',
      },
      {
        step: 2,
        title: 'Compute Running Modulo 255 Sums',
        description: 'Accumulates two running 8-bit sums over the data stream.',
      },
      {
        step: 3,
        title: 'Inspect 16-Bit Checksum Result',
        description: 'View the combined 16-bit integer and hexadecimal verification checksum.',
      },
    ],
    faqs: [
      {
        question: 'How is Fletcher-16 superior to a simple additive checksum?',
        answer: 'Fletcher-16 is position-dependent. Swapping two adjacent bytes changes the second sum, allowing Fletcher-16 to detect transposition errors that standard byte addition misses.',
      },
    ],
  },

  // Historical Machines
  '/historical/enigma': {
    howTo: [
      {
        step: 1,
        title: 'Configure Rotors and Ring Settings',
        description: 'Select three rotor models (I to V), initial rotor ground settings, and Ringstellung ring offsets.',
      },
      {
        step: 2,
        title: 'Wire the Steckerbrett (Plugboard)',
        description: 'Connect letter pairs on the plugboard to swap characters prior to rotor entry.',
      },
      {
        step: 3,
        title: 'Simulate Keystroke Encryption',
        description: 'Type letters to witness mechanical rotor stepping and observe corresponding output lamps illuminating.',
      },
    ],
    faqs: [
      {
        question: 'Why could an Enigma machine never encrypt a letter as itself?',
        answer: 'The Enigma used a reflector (Umkehrwalze) that sent the electrical signal back through the rotors via a different path, making it impossible for an output letter to equal the input letter—a vital flaw codebreakers exploited.',
      },
    ],
  },

  '/historical/bombe': {
    howTo: [
      {
        step: 1,
        title: 'Enter Ciphertext & Plaintext Crib',
        description: 'Input intercepted German ciphertext and a suspected German plaintext phrase (the crib).',
      },
      {
        step: 2,
        title: 'Configure Menu Graph Connections',
        description: 'The machine maps closed loops of deductions linking ciphertext letters to crib letters.',
      },
      {
        step: 3,
        title: 'Run Turing-Welchman Contradiction Engine',
        description: 'Electromechanical drums test rotor positions simultaneously, halting on candidate daily keys with zero contradictions.',
      },
    ],
    faqs: [
      {
        question: 'How did the Turing Bombe break Enigma without testing all keys?',
        answer: 'Instead of searching for correct keys, the Bombe searched for logical contradictions. If a hypothesized letter deduction led to a contradiction, that entire branch of rotor positions was rejected simultaneously.',
      },
    ],
  },

  '/historical/typex': {
    howTo: [
      {
        step: 1,
        title: 'Select 5-Rotor Configuration',
        description: 'Configure the British Typex rotor assembly using two stationary stators and three revolving rotors.',
      },
      {
        step: 2,
        title: 'Set Multiple Turnover Notches',
        description: 'Configure multiple notches per rotor to produce irregular, non-odometer mechanical stepping.',
      },
      {
        step: 3,
        title: 'Simulate Printed Teleprinter Output',
        description: 'Type messages to generate printed tape output with significantly greater cryptographic complexity than Wehrmacht Enigma.',
      },
    ],
    faqs: [
      {
        question: 'Why was the British Typex machine superior to Enigma?',
        answer: 'Typex used 5 rotors instead of 3, had multiple turnover notches on each rotor for irregular stepping, and printed directly onto paper tape, eliminating human transcription errors.',
      },
    ],
  },
};

export function getToolKnowledge(path: string, category?: string): ToolKnowledge {
  const clean = path.length > 1 && path.endsWith('/') ? path.slice(0, -1) : path;

  // 1. Exact match for specific tool or hub page
  if (TOOL_KNOWLEDGE_MAP[clean]) {
    return TOOL_KNOWLEDGE_MAP[clean];
  }

  // 2. Exact match on category hub
  if (category && TOOL_KNOWLEDGE_MAP[category]) {
    return TOOL_KNOWLEDGE_MAP[category];
  }

  // 3. Fallback
  return {
    howTo: [
      {
        step: 1,
        title: 'Provide Input Data',
        description: 'Enter your message, binary string, or parameters into the workspace panel.',
      },
      {
        step: 2,
        title: 'Configure Algorithm Parameters',
        description: 'Set required keys, operational modes, or formatting options.',
      },
      {
        step: 3,
        title: 'Execute & Verify Output',
        description: 'Inspect the transformed cryptographic result and copy verified data.',
      },
    ],
    faqs: [
      {
        question: 'How does this tool process sensitive data?',
        answer: 'CipherVerse operates directly inside your local browser session or via isolated backend routines. Your sensitive keys and plaintexts are never logged, tracked, or persisted.',
      },
    ],
  };
}
