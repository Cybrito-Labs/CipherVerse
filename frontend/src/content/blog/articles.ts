export interface TableOfContentsItem {
  id: string;
  title: string;
  level: number;
}

export interface RelatedTool {
  name: string;
  path: string;
  description: string;
  category: string;
}

export interface BlogArticle {
  slug: string;
  title: string;
  description: string;
  category: 'Classical Cryptography' | 'Modern Cryptography' | 'Steganography' | 'Public-Key & Protocols' | 'Forensics';
  publishedAt: string;
  readTime: string;
  author: {
    name: string;
    role: string;
  };
  tags: string[];
  coverGradient: string;
  featured?: boolean;
  seriesBadge?: string;
  relatedTools: RelatedTool[];
  tableOfContents: TableOfContentsItem[];
  sections: Array<{
    id: string;
    heading: string;
    level?: number;
    paragraphs?: string[];
    postCodeParagraphs?: string[];
    codeBlock?: {
      language: string;
      code: string;
      caption?: string;
    };
    callout?: {
      type: 'info' | 'warning' | 'tip';
      title: string;
      text: string;
    };
    list?: {
      ordered?: boolean;
      items: string[];
    };
    toolCta?: RelatedTool;
  }>;
}

export const blogArticles: BlogArticle[] = [
  {
    slug: 'caesar-cipher',
    title: 'Caesar Cipher & ROT13: The Complete Mathematical & Cryptanalysis Guide',
    description: 'An exhaustive educational breakdown of the Caesar cipher: historical Roman origins, modular arithmetic formulas in Z26, ROT13 involution, automated Chi-Square frequency cryptanalysis, and runnable Python cracking scripts.',
    category: 'Classical Cryptography',
    publishedAt: '2026-09-14',
    readTime: '8 min read',
    author: {
      name: 'CipherVerse Cryptography Academy',
      role: 'Classical Cryptography & Cryptanalysis',
    },
    tags: [
      'Caesar Cipher',
      'ROT13',
      'Classical Cryptography',
      'Modular Arithmetic',
      'Frequency Analysis',
      'Chi-Square Test',
      'Cryptanalysis',
    ],
    coverGradient: 'from-amber-500/20 via-yellow-500/10 to-red-500/20',
    featured: true,
    seriesBadge: 'Start Here • Lesson 1: Foundational Shift Ciphers',
    relatedTools: [
      {
        name: 'Caesar Cipher Solver & ROT13',
        path: '/classical/caesar',
        description: 'Encode, decode, and brute-force all 25 Caesar shifts in real time with interactive controls.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Monoalphabetic Substitution Cipher',
        path: '/classical/substitution',
        description: 'Generalized substitution cipher with custom alphabet mappings and frequency tables.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Atbash Cipher Tool',
        path: '/classical/atbash',
        description: 'Biblical reciprocal substitution cipher reversing the alphabet (A <-> Z, B <-> Y).',
        category: 'Classical Ciphers',
      },
    ],
    tableOfContents: [
      { id: 'historical-origins', title: '1. Historical Roman Origins & Julius Caesar', level: 2 },
      { id: 'mathematical-formulation', title: '2. Mathematical Formulation: Congruence in Z26', level: 2 },
      { id: 'the-rot13-involution', title: '3. The ROT13 Special Case & Mathematical Involution', level: 2 },
      { id: 'step-by-step-example', title: '4. Step-by-Step Encryption & Decryption Example', level: 2 },
      { id: 'cryptanalysis-and-breaking', title: '5. Cryptanalysis & Automated Chi-Square Cracking', level: 2 },
      { id: 'code-implementation', title: '6. Complete Python Implementation & Auto-Solver', level: 2 },
      { id: 'practice-challenge', title: '7. Practice Challenge: The Roman Legion Dispatch', level: 2 },
      { id: 'interactive-workbench', title: '8. Interactive Caesar Cipher Workbench', level: 2 },
    ],
    sections: [
      {
        id: 'historical-origins',
        heading: '1. Historical Roman Origins & Julius Caesar',
        paragraphs: [
          'The Caesar cipher is one of the earliest documented encryption algorithms in human history. As recorded by the Roman biographer Suetonius in "De Vita Caesarum" (The Lives of the Caesars, 121 CE), Gaius Julius Caesar utilized a secret substitution scheme to protect sensitive military dispatches during the Gallic Wars (58–50 BCE).',
          'Suetonius notes: "If he had anything confidential to say, he wrote it in cipher, that is, by so changing the order of the letters of the alphabet, that not a word could be made out. If anyone wishes to decipher these, and get at their meaning, he must substitute the fourth letter of the alphabet, namely D, for A, and so with the others."',
          'Interestingly, Caesar\'s nephew and successor, Augustus Caesar, also used a substitution cipher, but with a fixed shift of only 1 position (A became B, B became C). When Augustus reached the final letter of the Latin alphabet (X at the time), he wrote "AA" rather than wrapping around to A.',
          'In antiquity, the Caesar cipher was remarkably effective not because of its mathematical complexity, but because the vast majority of Rome\'s barbarian adversaries (Gauls, Germanic tribes, and Britons) were illiterate. Even educated adversaries who intercepted a dispatch assumed the garbled text was written in an obscure, unfamiliar foreign dialect.',
        ],
        callout: {
          type: 'info',
          title: 'Historical Milestone',
          text: 'The Caesar cipher represents the transition from physical steganography (such as shaving a slave\'s head, tattooing a message, and waiting for hair to regrow) to mathematical cryptography—protecting the meaning of a message rather than merely hiding its physical presence.',
        },
      },
      {
        id: 'mathematical-formulation',
        heading: '2. Mathematical Formulation: Congruence in Z26',
        paragraphs: [
          'From a modern algebraic perspective, the Caesar cipher is a monoalphabetic shift cipher operating over the finite ring of integers modulo 26, denoted as Z26 = {0, 1, 2, ..., 25}.',
          'First, we define a bijective mapping between the 26 letters of the standard Latin alphabet and their zero-indexed integers: A ↦ 0, B ↦ 1, C ↦ 2, ..., Z ↦ 25.',
          'Let x in Z26 denote the numerical value of a plaintext letter, and let k in {0, 1, ..., 25} denote the secret numerical shift key. The encryption function E_k(x) and decryption function D_k(y) are formally defined as:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Formal mathematical definition of Caesar Cipher encryption and decryption',
          code: `Encryption:
  E_k(x) = (x + k) mod 26

Decryption:
  D_k(y) = (y - k) mod 26
         = (y + 26 - k) mod 26`,
        },
        postCodeParagraphs: [
          'The modular addition guarantees that shifts past the end of the alphabet cleanly "wrap around" back to the beginning. For example, if x = 24 ("Y") and the key k = 3, we calculate: (24 + 3) mod 26 = 27 mod 26 = 1 ("B").',
          'Similarly, during decryption, adding 26 prior to calculating modulo 26 prevents negative numbers in programming languages that implement truncated integer division rather than true Euclidean modulo.',
        ],
      },
      {
        id: 'the-rot13-involution',
        heading: '3. The ROT13 Special Case & Mathematical Involution',
        paragraphs: [
          'A famous variant of the Caesar cipher is ROT13 ("Rotate by 13 places"), where the shift key is fixed at k = 13.',
          'Because the English alphabet contains exactly 26 letters, shifting by 13 positions divides the alphabet into two symmetric halves: A ↔ N, B ↔ O, C ↔ P, ..., M ↔ Z.',
          'Mathematically, ROT13 is an involution—a function that is its own inverse. Applying the encryption function twice returns the original value:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Mathematical proof of ROT13 involution',
          code: `E_13(E_13(x)) = ((x + 13) mod 26 + 13) mod 26
              = (x + 26) mod 26
              = x mod 26
              = x`,
        },
        callout: {
          type: 'tip',
          title: 'Modern Use of ROT13',
          text: 'ROT13 provides zero cryptographic confidentiality today, but it is widely used in online forums, Reddit, Usenet, and Geocaching to obscure spoilers, puzzle solutions, movie endings, and offensive punchlines from accidental viewing.',
        },
      },
      {
        id: 'step-by-step-example',
        heading: '4. Step-by-Step Encryption & Decryption Example',
        paragraphs: [
          'To see the algorithm in action, let us encrypt the message "CIPHER" using a Caesar shift key of k = 7.',
          'Step 1: Convert each letter of the plaintext into its numerical integer index (0-25).',
          'Step 2: Add the shift key k = 7 to each index.',
          'Step 3: Apply modulo 26 to compute the ciphertext index.',
          'Step 4: Convert each resulting index back into its corresponding character.',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Worked trace table for encrypting "CIPHER" with k = 7',
          code: `Letter | Plaintext Index (x) | x + k (x + 7) | (x + 7) mod 26 | Ciphertext Letter
-------+---------------------+---------------+----------------+------------------
  C    |          2          |       9       |       9        |        J
  I    |          8          |      15       |      15        |        P
  P    |         15          |      22       |      22        |        W
  H    |          7          |      14       |      14        |        O
  E    |          4          |      11       |      11        |        L
  R    |         17          |      24       |      24        |        Y

Plaintext:  CIPHER
Ciphertext: JPWOLY`,
        },
      },
      {
        id: 'cryptanalysis-and-breaking',
        heading: '5. Cryptanalysis & Automated Chi-Square Cracking',
        paragraphs: [
          'How secure is the Caesar cipher? By modern security benchmarks, it provides virtually zero security due to three critical vulnerabilities:',
          '1. Tiny Key Space (Brute Force): Because there are only 26 possible letters in the alphabet, a shift of 0 does nothing, leaving exactly 25 potential keys. A human can test all 25 shifts by hand on paper in under 3 minutes; a computer can test all 25 shifts in less than 10 microseconds.',
          '2. Preserved Letter Frequencies: Monoalphabetic substitution ciphers do not alter the statistical distribution of letters. In English text, the letter "E" is by far the most frequent (12.7%), followed by "T" (9.1%), "A" (8.2%), and "O" (7.5%). In a Caesar ciphertext, the most common letter will almost always correspond to the shifted equivalent of "E".',
          '3. Automated Chi-Square Goodness-of-Fit Test: How does an automated computer program know which of the 25 shifts is readable English without human intervention? We use the Chi-Square statistic:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Chi-Square statistic formula for automated cryptanalysis',
          code: `         26  (Observed_i - Expected_i)^2
  chi^2 = Σ   ---------------------------
         i=1          Expected_i

Where:
  Observed_i = Actual count of letter i in the candidate plaintext
  Expected_i = (Length of text) * (Typical English frequency of letter i)`,
        },
        callout: {
          type: 'warning',
          title: 'The Lowest Chi-Square Wins',
          text: 'When computing Chi-Square across all 25 shifts, the candidate text that yields the absolute lowest chi-squared score is statistically guaranteed to be the genuine English plaintext!',
        },
      },
      {
        id: 'code-implementation',
        heading: '6. Complete Python Implementation & Auto-Solver',
        paragraphs: [
          'Here is a production-grade, standalone Python script demonstrating both Caesar encryption/decryption and an automated Chi-Square frequency cracking engine that cracks unknown Caesar ciphertexts instantly with zero human guessing:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'caesar_cipher_complete.py: Modular cipher and Chi-Square frequency cracker',
          code: `import string

# Standard English letter frequencies (A-Z) in percentages
ENGLISH_FREQS = {
    'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
    'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
    'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
    'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
    'U': 0.02758, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974,
    'Z': 0.00074
}

def caesar(text: str, shift: int, decrypt: bool = False) -> str:
    """Encrypts or decrypts text with a Caesar shift."""
    if decrypt:
        shift = -shift
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + shift) % 26
            result.append(chr(base + shifted))
        else:
            result.append(char)
    return "".join(result)

def calculate_chi_square(text: str) -> float:
    """Calculates Chi-Square statistic comparing text to English distribution."""
    letters = [c.upper() for c in text if c.isalpha()]
    total = len(letters)
    if total == 0:
        return float('inf')

    counts = {c: letters.count(c) for c in string.ascii_uppercase}
    chi_square = 0.0
    for char, expected_prob in ENGLISH_FREQS.items():
        expected_count = total * expected_prob
        observed_count = counts[char]
        chi_square += ((observed_count - expected_count) ** 2) / expected_count
    return chi_square

def auto_crack_caesar(ciphertext: str):
    """Automatically cracks a Caesar cipher using Chi-Square minimization."""
    best_shift = 0
    lowest_chi = float('inf')
    best_plaintext = ""

    for shift in range(26):
        candidate = caesar(ciphertext, shift, decrypt=True)
        chi = calculate_chi_square(candidate)
        if chi < lowest_chi:
            lowest_chi = chi
            best_shift = shift
            best_plaintext = candidate

    return best_shift, best_plaintext, lowest_chi

# Demonstration
secret = "KHOOR ZRUOG WKLV LV DQ DXWRPDWHG FUDSWRJUDSKLF VROSHU"
shift, plaintext, chi = auto_crack_caesar(secret)
print(f"Detected Shift: {shift}")
print(f"Decrypted Text: {plaintext}")
print(f"Chi-Square Score: {chi:.2f}")`,
        },
      },
      {
        id: 'practice-challenge',
        heading: '7. Practice Challenge: The Roman Legion Dispatch',
        paragraphs: [
          'Ready to test your cryptanalysis skills? Here is an authentic historical puzzle:',
          'An intercepted messenger from Julius Caesar\'s Tenth Legion (Legio X Equestris) carried the following scrambled dispatch:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Challenge Ciphertext',
          code: `YHQL YLGL YLFL! WKH JDOOLF WULEHV KDYH VXUUHQGHUHG DW DOHVLD.
DOO KDLW FDHVDU!`,
        },
        postCodeParagraphs: [
          'Can you determine the secret shift key and uncover the Latin victory declaration and historical military report?',
          'Hint: Test it directly using the CipherVerse Caesar Cipher Brute-Force feature!',
        ],
        toolCta: {
          name: 'Solve with CipherVerse Caesar Tool',
          path: '/classical/caesar',
          description: 'Paste the puzzle ciphertext into CipherVerse to test all shifts in real time.',
          category: 'Classical Ciphers',
        },
      },
      {
        id: 'interactive-workbench',
        heading: '8. Interactive Caesar Cipher Workbench',
        paragraphs: [
          'You can explore real-time rotation, automated ROT13 toggle, brute-force tables, and live letter-frequency visualizers directly in CipherVerse.',
          'Every computation executes entirely client-side in your browser with zero latency and zero data transmitted to any external server.',
        ],
        toolCta: {
          name: 'Launch Caesar Cipher & ROT13 Solver',
          path: '/classical/caesar',
          description: 'Instant encoding, decoding, and automated brute-force cracking.',
          category: 'Classical Ciphers',
        },
      },
    ],
  },
  {
    slug: 'evolution-of-cryptography',
    title: "The Cryptographer's Journey: From Ancient Caesar Ciphers to Modern AES-256",
    description: "Explore the 2,000-year history of cryptographic evolution: how simple monoalphabetic substitution ciphers collapsed under frequency analysis, paving the way for polyalphabetic machines and modern Rijndael block ciphers.",
    category: 'Classical Cryptography',
    publishedAt: '2026-09-14',
    readTime: '7 min read',
    author: {
      name: 'CipherVerse Research Lab',
      role: 'Cryptographic Security & Algorithms',
    },
    tags: ['Classical Ciphers', 'Caesar Cipher', 'Vigenere', 'AES-256', 'Cryptanalysis', 'History'],
    coverGradient: 'from-amber-500/20 via-orange-500/10 to-purple-500/20',
    featured: false,
    relatedTools: [
      {
        name: 'Caesar Cipher Solver & ROT13',
        path: '/classical/caesar',
        description: 'Encode, decode, and brute-force all 25 Caesar shifts in real time.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Vigenere Cipher Solver',
        path: '/classical/vigenere',
        description: 'Polyalphabetic substitution cipher with custom key and autokey support.',
        category: 'Classical Ciphers',
      },
      {
        name: 'AES-GCM / CBC Encryption Suite',
        path: '/symmetric/aes',
        description: 'Military-grade 128/192/256-bit symmetric block cipher encryptor.',
        category: 'Symmetric Crypto',
      },
    ],
    tableOfContents: [
      { id: 'ancient-beginnings', title: '1. Ancient Beginnings: The Caesar Shift', level: 2 },
      { id: 'the-fall-of-substitution', title: '2. The Fall of Substitution: Frequency Analysis', level: 2 },
      { id: 'polyalphabetic-leap', title: '3. The Polyalphabetic Leap: The Vigenère Cipher', level: 2 },
      { id: 'breaking-polyalphabetic', title: '4. The Kasiski Examination & Friedman Test', level: 2 },
      { id: 'modern-block-ciphers', title: '5. The Modern Era: Claude Shannon & AES-256', level: 2 },
      { id: 'hands-on-practice', title: '6. Hands-on Practice with CipherVerse Tools', level: 2 },
    ],
    sections: [
      {
        id: 'ancient-beginnings',
        heading: '1. Ancient Beginnings: The Caesar Shift',
        paragraphs: [
          'More than two thousand years ago, Julius Caesar communicated military instructions to his generals using a simple substitution cipher. By shifting every letter of the Latin alphabet by three positions, plaintext messages like "ATTACK AT DAWN" were transformed into "DWWDFN DW GDZQ".',
          'Mathematically, if each letter A through Z is assigned a numerical index from 0 to 25, the Caesar cipher encryption and decryption functions can be expressed as modular arithmetic operations:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'Caesar Cipher modular arithmetic implementation in Python',
          code: `# Caesar cipher shift with modulus 26
def caesar_cipher(text: str, shift: int = 3, decrypt: bool = False) -> str:
    if decrypt:
        shift = -shift
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + shift) % 26
            result.append(chr(base + shifted))
        else:
            result.append(char)
    return "".join(result)

# Example:
cipher = caesar_cipher("DEFEND THE GATES", shift=3)
print("Ciphertext:", cipher) # GHIHQG WKH JDWHV`,
        },
        callout: {
          type: 'info',
          title: 'Did you know?',
          text: 'ROT13 is a special case of the Caesar cipher with a fixed shift of 13. Because 13 is exactly half the 26-letter English alphabet, ROT13 is symmetric: applying the exact same algorithm twice restores the original plaintext!',
        },
      },
      {
        id: 'the-fall-of-substitution',
        heading: '2. The Fall of Substitution: Frequency Analysis',
        paragraphs: [
          'For centuries, substitution ciphers were believed to be impenetrable. However, around 850 CE, Arab polymath Al-Kindi published "A Manuscript on Deciphering Cryptographic Messages", inventing the foundational science of cryptanalysis through frequency analysis.',
          'Al-Kindi realized that in any natural language, certain characters appear with predictable frequencies. In the English language, the letter "E" accounts for roughly 12.7% of all text, followed by "T" (9.1%) and "A" (8.2%). Letters like "Q", "X", and "Z" occur less than 0.2% of the time.',
          'Because monoalphabetic substitution does not alter the underlying statistical distribution, any Caesar or monoalphabetic ciphertext longer than 50 characters can be broken in seconds simply by correlating peak letter frequencies.',
        ],
        toolCta: {
          name: 'Caesar Cipher Solver',
          path: '/classical/caesar',
          description: 'Try brute-forcing and frequency-shifting Caesar cipher texts in CipherVerse.',
          category: 'Classical Ciphers',
        },
      },
      {
        id: 'polyalphabetic-leap',
        heading: '3. The Polyalphabetic Leap: The Vigenère Cipher',
        paragraphs: [
          'To defeat Al-Kindi’s frequency analysis, 16th-century cryptographers led by Giovan Battista Bellaso and Blaise de Vigenère introduced polyalphabetic substitution. Instead of using a single alphabet shift throughout the message, the Vigenère cipher uses a repeating secret keyword.',
          'Each character of the keyword determines a distinct Caesar shift for the corresponding character in the plaintext. For example, if the keyword is "LEMON", the first letter shifts by 11 (L), the second by 4 (E), the third by 12 (M), and so on. The letter "E" in the plaintext might encrypt as "P" in one word, and as "R" in another, flattening the single-letter frequency distribution curve.',
          'For nearly three centuries, European diplomats referred to the Vigenère cipher as "le chiffre indéchiffrable" (the indecipherable cipher).',
        ],
      },
      {
        id: 'breaking-polyalphabetic',
        heading: '4. The Kasiski Examination & Friedman Test',
        paragraphs: [
          'In 1863, Prussian infantry officer Friedrich Kasiski shattered the illusion of Vigenère’s invulnerability. Kasiski noticed that when recurring phrases in the plaintext coincide with the cycle length of the keyword, identical ciphertext fragments are generated.',
          'By finding the greatest common divisor (GCD) of the distances between repeated ciphertext strings, an analyst can accurately deduce the exact length of the keyword (L). Once the keyword length is known, the ciphertext is partitioned into L separate monoalphabetic substitution ciphers, and each column is solved using standard frequency analysis.',
        ],
        callout: {
          type: 'warning',
          title: 'The Perfect Cipher: One-Time Pad',
          text: 'If a polyalphabetic key is truly random, never reused, kept completely secret, and as long as the message itself, it becomes a One-Time Pad (OTP). Claude Shannon mathematically proved in 1949 that the One-Time Pad provides information-theoretic security that can never be broken, even with infinite computing power!',
        },
      },
      {
        id: 'modern-block-ciphers',
        heading: '5. The Modern Era: Claude Shannon & AES-256',
        paragraphs: [
          'In 1949, Claude Shannon formulated the twin pillars of modern cryptographic design: Confusion and Diffusion. Confusion obscures the relationship between the key and the ciphertext (typically via substitution S-Boxes), while Diffusion spreads the influence of each plaintext bit across the entire ciphertext (via permutations and linear mixing).',
          'In 2001, the National Institute of Standards and Technology (NIST) adopted the Rijndael algorithm as the Advanced Encryption Standard (AES). Unlike classical character-level ciphers, AES operates on 128-bit blocks of raw binary data organized into a 4x4 matrix of bytes.',
          'AES-256 executes 14 iterative rounds comprising SubBytes, ShiftRows, MixColumns, and AddRoundKey. With 2^256 possible keys (approximately 1.15 x 10^77), cracking an AES-256 key by brute force would require more energy than all stars in the observable universe produce over billions of years.',
        ],
        codeBlock: {
          language: 'typescript',
          caption: 'Encrypting payload using modern Web Crypto API (AES-256-GCM)',
          code: `// Modern AES-256-GCM Encryption in TypeScript / Web Crypto
async function encryptWithAES256(plaintext: string, secretKey: CryptoKey) {
  const iv = crypto.getRandomValues(new Uint8Array(12)); // 96-bit nonce
  const encoded = new TextEncoder().encode(plaintext);

  const ciphertext = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    secretKey,
    encoded
  );

  return {
    iv: Array.from(iv),
    ciphertext: Array.from(new Uint8Array(ciphertext))
  };
}`,
        },
      },
      {
        id: 'hands-on-practice',
        heading: '6. Hands-on Practice with CipherVerse Tools',
        paragraphs: [
          'Understanding theoretical cryptography is most effective when paired with hands-on experimentation. In CipherVerse, you can directly compare classical and modern encryption algorithms in real time.',
          'Start with the Caesar and Vigenère solvers to observe character rotation and frequency curves, then explore the AES-256 and Triple-DES tools to see how block ciphers transform raw binary payloads with initialization vectors and Galois counter modes.',
        ],
        toolCta: {
          name: 'Explore AES-256 Encryption Suite',
          path: '/symmetric/aes',
          description: 'Encrypt and decrypt messages using AES with GCM, CBC, and ECB modes.',
          category: 'Symmetric Crypto',
        },
      },
    ],
  },
  {
    slug: 'steganography-guide',
    title: 'Steganography Masterclass: How to Conceal Secrets in Digital Images and Audio',
    description: 'A practical, mathematical guide to modern steganography: LSB spatial embedding, discrete cosine transforms (DCT), high-frequency audio phase encoding, and forensic steganalysis detection.',
    category: 'Steganography',
    publishedAt: '2026-09-14',
    readTime: '6 min read',
    author: {
      name: 'CipherVerse Security Team',
      role: 'Forensics & Information Hiding',
    },
    tags: ['Steganography', 'LSB Embedding', 'Image Forensics', 'WAV Audio', 'Information Security', 'CTF'],
    coverGradient: 'from-emerald-500/20 via-teal-500/10 to-cyan-500/20',
    relatedTools: [
      {
        name: 'Image Steganography Suite',
        path: '/steganography/image',
        description: 'Conceal text or binary files into PNG and BMP images using LSB bit manipulation.',
        category: 'Steganography',
      },
      {
        name: 'Audio Steganography Suite',
        path: '/steganography/audio',
        description: 'Encode secret payloads inside uncompressed WAV audio PCM waveforms.',
        category: 'Steganography',
      },
      {
        name: 'Zero-Width Text Steganography',
        path: '/steganography/text',
        description: 'Hide invisible unicode data inside innocent-looking text documents.',
        category: 'Steganography',
      },
    ],
    tableOfContents: [
      { id: 'stego-vs-crypto', title: '1. Steganography vs. Cryptography', level: 2 },
      { id: 'lsb-image-encoding', title: '2. Least Significant Bit (LSB) Image Encoding', level: 2 },
      { id: 'audio-steganography', title: '3. Audio Steganography: Waveform Manipulation', level: 2 },
      { id: 'steganalysis-detection', title: '4. Steganalysis: How Investigators Detect Secrets', level: 2 },
      { id: 'practical-tools', title: '5. Testing Steganography on CipherVerse', level: 2 },
    ],
    sections: [
      {
        id: 'stego-vs-crypto',
        heading: '1. Steganography vs. Cryptography',
        paragraphs: [
          'While cryptography aims to make messages unintelligible to eavesdroppers, steganography aims to conceal the very existence of the communication. The word derives from the ancient Greek words "steganos" (covered) and "graphein" (writing).',
          'In many adversarial scenarios, sending an encrypted message is itself an alert to intelligence agencies or censors that sensitive information is being transmitted. Steganography embeds the payload into everyday carrier media—such as family vacation photos, background audio tracks, or plain text articles—so that observers never suspect a secret is present.',
        ],
        callout: {
          type: 'tip',
          title: 'The Golden Rule of Covert Channels',
          text: 'Security through obscurity is never enough. The gold standard for modern covert communication is combining both disciplines: first encrypt the payload using AES-256, and then embed the resulting ciphertext into a carrier medium with steganography.',
        },
      },
      {
        id: 'lsb-image-encoding',
        heading: '2. Least Significant Bit (LSB) Image Encoding',
        paragraphs: [
          'In a standard 24-bit RGB bitmap image, each pixel consists of three color channels: Red, Green, and Blue. Each channel is represented by an 8-bit byte with values ranging from 0 to 255.',
          'The most significant bit (MSB) carries 50% of the pixel color value (128), while the least significant bit (LSB) represents a value of just 1. Changing the LSB from a 0 to a 1 changes the luminance of that color channel by approximately 0.39%—a difference completely imperceptible to the human eye.',
        ],
        codeBlock: {
          language: 'python',
          caption: 'Concept of LSB bit replacement in Python',
          code: `# Replacing the LSB of a byte with a payload bit
def embed_bit(pixel_byte: int, secret_bit: int) -> int:
    # Clear the last bit using bitwise AND with 0xFE (11111110), then OR with secret bit
    return (pixel_byte & ~1) | secret_bit

# Example: Original Red channel is 200 (11001000b)
original = 200
secret_bit = 1
modified = embed_bit(original, secret_bit)
print(f"Original: {original} (bin: {bin(original)})")
print(f"Modified: {modified} (bin: {bin(modified)})") # 201 (11001001b)`,
        },
      },
      {
        id: 'audio-steganography',
        heading: '3. Audio Steganography: Waveform Manipulation',
        paragraphs: [
          'Digital audio consists of thousands of discrete sound pressure samples recorded per second (typically 44,100 Hz in CD-quality audio). In 16-bit Pulse Code Modulation (PCM) WAV files, each sample is a 16-bit signed integer ranging from -32,768 to +32,767.',
          'By manipulating the lowest 1 or 2 bits of each PCM audio sample, secret files can be hidden within songs or voice recordings. Because human auditory perception is less sensitive to minuscule amplitudes in high-frequency noise bands, LSB-encoded WAV files sound identical to the original uncompressed master.',
        ],
        toolCta: {
          name: 'Audio Steganography Suite',
          path: '/steganography/audio',
          description: 'Try embedding and extracting secret audio messages in CipherVerse.',
          category: 'Steganography',
        },
      },
      {
        id: 'steganalysis-detection',
        heading: '4. Steganalysis: How Investigators Detect Secrets',
        paragraphs: [
          'Steganalysis is the counter-science of detecting hidden communications. Modern forensic investigators use several mathematical tools to expose LSB embedding:',
          '1. Chi-Square Analysis: Natural images exhibit uneven distributions of even and odd pixel values. Naive LSB embedding tends to equalize adjacent color pairs (PoVs: Pairs of Values), leaving an identifiable statistical signature.',
          '2. Sample Pair Analysis (SPA): Analyzes the transition probabilities between neighboring pixels to estimate the exact percentage of modified carrier bytes.',
          '3. Visual Steganography Attacks: Isolating and viewing only the 0th bit plane will render pure noise in a natural image, but displays visible text, barcodes, or patterns in naive implementations.',
        ],
      },
      {
        id: 'practical-tools',
        heading: '5. Testing Steganography on CipherVerse',
        paragraphs: [
          'You can experiment with both LSB visual encoding and waveform audio steganography directly in your browser using CipherVerse’s Steganography Suite. The tools run client-side using HTML5 Canvas and the Web Audio API, guaranteeing your carrier media and secret messages are processed with 100% privacy.',
        ],
        toolCta: {
          name: 'Launch Image Steganography Suite',
          path: '/steganography/image',
          description: 'Embed and extract text or files with zero external server uploads.',
          category: 'Steganography',
        },
      },
    ],
  },
  {
    slug: 'understanding-rsa-cryptography',
    title: 'How Public-Key Cryptography Works: The Mathematics of RSA & Prime Factorization',
    description: 'An intuitive breakdown of asymmetric cryptography: trapdoor one-way functions, Euler’s Totient Theorem, modular exponentiation, and how 2048-bit RSA secures HTTPS across the world.',
    category: 'Public-Key & Protocols',
    publishedAt: '2026-09-14',
    readTime: '8 min read',
    author: {
      name: 'CipherVerse Research Lab',
      role: 'Asymmetric Cryptography & Protocols',
    },
    tags: ['RSA', 'Asymmetric Cryptography', 'Public Key', 'Euler Totient', 'Number Theory', 'Cybersecurity'],
    coverGradient: 'from-blue-500/20 via-indigo-500/10 to-violet-500/20',
    relatedTools: [
      {
        name: 'RSA Key Generator & Cipher Suite',
        path: '/asymmetric/rsa',
        description: 'Generate public/private keypairs, encrypt data, and sign messages using RSA.',
        category: 'Asymmetric Crypto',
      },
      {
        name: 'DSA Signature Verification',
        path: '/asymmetric/dsa',
        description: 'Create and verify digital signatures with the Digital Signature Algorithm.',
        category: 'Asymmetric Crypto',
      },
      {
        name: 'X.509 Certificate Inspector',
        path: '/certificates/x509',
        description: 'Inspect ASN.1 certificate parameters, public keys, and validity chains.',
        category: 'Certificates',
      },
    ],
    tableOfContents: [
      { id: 'the-key-exchange-problem', title: '1. The Key Exchange Dilemma', level: 2 },
      { id: 'the-trapdoor-function', title: '2. The Trapdoor One-Way Function', level: 2 },
      { id: 'rsa-mathematical-walkthrough', title: '3. Step-by-Step RSA Key Generation', level: 2 },
      { id: 'encryption-and-decryption', title: '4. Encryption and Decryption Mechanics', level: 2 },
      { id: 'quantum-threat', title: '5. Shor’s Algorithm & The Quantum Horizon', level: 2 },
      { id: 'try-rsa-online', title: '6. Generate RSA Keys in CipherVerse', level: 2 },
    ],
    sections: [
      {
        id: 'the-key-exchange-problem',
        heading: '1. The Key Exchange Dilemma',
        paragraphs: [
          'Before the mid-1970s, all encryption was symmetric: Alice and Bob had to agree on the exact same secret key prior to transmitting confidential data. But how could they securely agree on a key across an insecure channel like the public internet without an eavesdropper intercepting it?',
          'In 1976, Whitfield Diffie and Martin Hellman published "New Directions in Cryptography", proposing asymmetric cryptography: every participant holds two mathematically linked keys—a Public Key that anyone can see, and a Private Key kept strictly secret.',
          'In 1977, Ron Rivest, Adi Shamir, and Leonard Adleman at MIT turned this theoretical concept into a practical algorithm: RSA.',
        ],
      },
      {
        id: 'the-trapdoor-function',
        heading: '2. The Trapdoor One-Way Function',
        paragraphs: [
          'Asymmetric encryption relies on a mathematical concept called a "Trapdoor One-Way Function". It is extraordinarily easy to compute in one direction, but computationally infeasible to invert unless you possess a secret piece of auxiliary information (the "trapdoor").',
          'For RSA, this trapdoor is Integer Factorization: multiplying two large prime numbers together is nearly instantaneous, but finding the original prime factors of a composite 2048-bit integer would take supercomputers thousands of years.',
        ],
        callout: {
          type: 'info',
          title: 'Euler’s Totient Theorem',
          text: 'Euler’s Totient Function, denoted φ(n), counts the positive integers up to n that are coprime to n. If n is the product of two distinct primes p and q, then φ(n) = (p - 1) * (q - 1). This value is the mathematical key that unlocks RSA decryption.',
        },
      },
      {
        id: 'rsa-mathematical-walkthrough',
        heading: '3. Step-by-Step RSA Key Generation',
        paragraphs: [
          'To generate an RSA keypair:',
          'Step 1: Choose two distinct, large prime numbers, p and q (in practice, each is 1024 bits long).',
          'Step 2: Compute their product, n = p * q. This number n is called the modulus and is made public.',
          'Step 3: Compute Euler’s totient: φ(n) = (p - 1) * (q - 1).',
          'Step 4: Choose an integer e such that 1 < e < φ(n) and gcd(e, φ(n)) = 1. (65537 is standard due to efficient binary weight). This e is the public exponent.',
          'Step 5: Compute the modular multiplicative inverse d of e modulo φ(n), satisfying: d * e ≡ 1 (mod φ(n)). This d is the private exponent.',
        ],
        codeBlock: {
          language: 'python',
          caption: 'Toy RSA implementation demonstrating the underlying number theory',
          code: `# Toy RSA demonstration with small primes (do not use small keys in production!)
p, q = 61, 53
n = p * q                # 3233
phi_n = (p - 1) * (q - 1) # 3120
e = 17                   # Coprime to 3120

# Extended Euclidean Algorithm to find modular inverse:
def egcd(a, b):
    if a == 0: return (b, 0, 1)
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    return x % m

d = modinv(e, phi_n)     # 2753

print(f"Public Key: (e={e}, n={n})")
print(f"Private Key: (d={d}, n={n})")

# Encrypting letter 'A' (ASCII 65):
message = 65
ciphertext = pow(message, e, n)
decrypted = pow(ciphertext, d, n)
print(f"Ciphertext: {ciphertext}, Decrypted: {decrypted} ('{chr(decrypted)}')")`,
        },
      },
      {
        id: 'encryption-and-decryption',
        heading: '4. Encryption and Decryption Mechanics',
        paragraphs: [
          'Once the keys are established:',
          'Encryption: Anyone can take a plaintext message m (represented as an integer < n) and calculate ciphertext c = m^e mod n using the recipient’s public key.',
          'Decryption: Only the recipient who holds private exponent d can compute m = c^d mod n. By Euler’s Totient Theorem, (m^e)^d ≡ m^(e*d) ≡ m (mod n).',
          'Digital Signatures: RSA can also be reversed to prove authenticity. The sender encrypts a hash of the document with their private key (signature = hash^d mod n). Anyone with the sender’s public key can verify the signature by computing signature^e mod n and checking if it matches the document hash.',
        ],
        toolCta: {
          name: 'RSA Key Generator & Cipher Tool',
          path: '/asymmetric/rsa',
          description: 'Generate real RSA keypairs, sign payloads, and decrypt messages in CipherVerse.',
          category: 'Asymmetric Crypto',
        },
      },
      {
        id: 'quantum-threat',
        heading: '5. Shor’s Algorithm & The Quantum Horizon',
        paragraphs: [
          'While classical supercomputers cannot factor 2048-bit RSA numbers, a sufficiently large fault-tolerant quantum computer running Peter Shor’s algorithm (1994) could factor large integers in polynomial time O((log n)^3).',
          'This has sparked global migration toward Post-Quantum Cryptography (PQC), such as lattice-based cryptography (ML-KEM / Crystals-Kyber) and stateless hash-based signatures (ML-DSA / Crystals-Dilithium), which are resistant to quantum attack.',
        ],
      },
      {
        id: 'try-rsa-online',
        heading: '6. Generate RSA Keys in CipherVerse',
        paragraphs: [
          'Want to explore key generation, public key formats (PEM, PKCS#8), and asymmetric encryption in your browser? CipherVerse provides an interactive RSA workbench equipped with real Web Crypto API key generation.',
        ],
        toolCta: {
          name: 'Open CipherVerse RSA Tool',
          path: '/asymmetric/rsa',
          description: 'Generate 1024/2048/4096-bit RSA keys with zero server retention.',
          category: 'Asymmetric Crypto',
        },
      },
    ],
  },
];
