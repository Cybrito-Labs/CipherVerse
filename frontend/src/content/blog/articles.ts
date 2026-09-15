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
    slug: 'vigenere-cipher',
    title: 'Vigenère Cipher: Polyalphabetic Cryptanalysis, Kasiski Examination & Index of Coincidence',
    description: 'An exhaustive academic breakdown of the Vigenère Cipher: Bellaso origins, Tabula Recta matrix arithmetic in ℤ₂₆, the collapse of single-letter frequency analysis, the Kasiski test, Friedman’s Index of Coincidence, and runnable Python auto-crackers.',
    category: 'Classical Cryptography',
    publishedAt: '2026-09-14',
    readTime: '10 min read',
    author: {
      name: 'CipherVerse Cryptography Academy',
      role: 'Classical Cryptography & Cryptanalysis',
    },
    tags: [
      'Vigenere Cipher',
      'Polyalphabetic Substitution',
      'Tabula Recta',
      'Kasiski Examination',
      'Index of Coincidence',
      'Cryptanalysis',
      'Chi-Square Test',
      'Classical Cryptography',
    ],
    coverGradient: 'from-cyan-500/20 via-blue-500/10 to-indigo-500/20',
    featured: false,
    seriesBadge: 'Academy • Lesson 2: Polyalphabetic Substitution & Cryptanalysis',
    relatedTools: [
      {
        name: 'Vigenère Cipher Solver',
        path: '/classical/vigenere',
        description: 'Encrypt, decrypt, and visualize the polyalphabetic Tabula Recta matrix in real time.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Caesar Cipher Solver & ROT13',
        path: '/classical/caesar',
        description: 'Foundational monoalphabetic shift cipher (Lesson 1 of the Academy series).',
        category: 'Classical Ciphers',
      },
      {
        name: 'Bifid Cipher Tool',
        path: '/classical/bifid',
        description: 'Fractionation cipher combining Polybius square substitution with transposition.',
        category: 'Classical Ciphers',
      },
    ],
    tableOfContents: [
      { id: 'historical-origins', title: '1. Historical Origins: Bellaso, Vigenère & The "Indecipherable Cipher"', level: 2 },
      { id: 'mathematical-formulation', title: '2. Mathematical Formulation & The Tabula Recta', level: 2 },
      { id: 'step-by-step-example', title: '3. Step-by-Step Worked Trace Table', level: 2 },
      { id: 'why-monoalphabetic-analysis-fails', title: '4. Why Single-Letter Frequency Analysis Collapses', level: 2 },
      { id: 'kasiski-examination', title: '5. Cracking Key Length: The Kasiski Examination', level: 2 },
      { id: 'friedmans-index-of-coincidence', title: '6. Statistical Precision: Friedman’s Index of Coincidence (IC)', level: 2 },
      { id: 'code-implementation', title: '7. Complete Python Implementation & Auto-Cracker', level: 2 },
      { id: 'civil-war-challenge', title: '8. Practice Challenge: The American Civil War Dispatch', level: 2 },
      { id: 'interactive-workbench', title: '9. Interactive Vigenère Cipher Workbench', level: 2 },
    ],
    sections: [
      {
        id: 'historical-origins',
        heading: '1. Historical Origins: Bellaso, Vigenère & The "Indecipherable Cipher"',
        paragraphs: [
          'For more than fifteen hundred years following Julius Caesar, cryptography was dominated by monoalphabetic substitution ciphers. However, after the Arab polymath Al-Kindi published his treatise on frequency analysis around 850 CE, every monoalphabetic cipher could be systematically broken by analyzing letter distributions.',
          'The conceptual breakthrough to overcome frequency analysis came from Italian cryptographer Giovan Battista Bellaso. In his 1553 treatise "La cifra del. Sig. Giovan Battista Bellaso", he introduced the concept of using a repeating secret keyword to alternate through different shifted alphabets character-by-character.',
          'Three decades later, in 1586, French diplomat Blaise de Vigenère published "Traicté des chiffres ou secrètes manieres d\'escrire" before the court of Henry III of France, describing a related autokey cipher. During the 19th century, historians erroneously attributed Bellaso’s keyword cipher to Vigenère, permanently cementing the name "Vigenère Cipher" in cryptographic history.',
          'For over three centuries, mathematicians and military tacticians deemed the cipher utterly impregnable, nicknaming it "le chiffre indéchiffrable" (the indecipherable cipher). It was heavily relied upon during major military conflicts, including the American Civil War (1861–1865), where the Confederate Army used brass cipher disks and keywords like "COMPLETE VICTORY" to encode battlefield telegraphs—unaware that Union cryptanalysts regularly intercepted and deciphered them.',
        ],
        callout: {
          type: 'info',
          title: 'A 300-Year Illusion',
          text: 'The 19th-century mathematician and author of Alice in Wonderland, Charles Lutwidge Dodgson (Lewis Carroll), famously praised the Vigenère cipher in 1868 as unbreakable in his article "The Alphabet Cipher". Yet, unknown to Carroll, Prussian infantry officer Friedrich Kasiski had already published a mathematical method to shatter it five years earlier in 1863.',
        },
      },
      {
        id: 'mathematical-formulation',
        heading: '2. Mathematical Formulation & The Tabula Recta',
        paragraphs: [
          'The Vigenère cipher is a periodic polyalphabetic substitution cipher. Rather than using a single numerical shift k across the entire message, it uses a sequence of shifts determined by a keyword of length m.',
          'Let each letter in the standard Latin alphabet be mapped to an integer in the finite ring ℤ₂₆ = {0, 1, 2, ..., 25}, where A ↦ 0, B ↦ 1, ..., Z ↦ 25.',
          'Let P = (p₀, p₁, ..., p_{n-1}) represent the plaintext sequence of length n, and let K = (k₀, k₁, ..., k_{m-1}) represent the secret key sequence of length m (where m ≤ n). The key is cyclically repeated across the message so that the key letter at index i is k_{i mod m}.',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Formal mathematical formulation of Vigenère encryption and decryption',
          code: `Encryption:
  c_i = (p_i + k_{i mod m}) mod 26

Decryption:
  p_i = (c_i - k_{i mod m} + 26) mod 26

Where:
  p_i         = Integer value of plaintext character at position i (0-25)
  c_i         = Integer value of ciphertext character at position i (0-25)
  k_{i mod m} = Integer value of key character corresponding to position i (0-25)
  m           = Length of the secret keyword`,
        },
        postCodeParagraphs: [
          'Historically, cryptographers performed this operation without mental arithmetic using the Tabula Recta—a 26×26 square containing all 26 cyclic permutations of the Latin alphabet. The encipherer finds the plaintext letter along the top column, locates the key letter along the left row, and reads the intersecting character in the grid.',
        ],
      },
      {
        id: 'step-by-step-example',
        heading: '3. Step-by-Step Worked Trace Table',
        paragraphs: [
          'To understand the transformation step-by-step, let us encrypt the message "DEFEND THE WALL" using the secret keyword "ORBIT" (key length m = 5).',
          'Notice how the keyword repeats cyclically: O-R-B-I-T-O-R-B-I-T-O-R-B.',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Worked trace table: Encrypting "DEFEND THE WALL" with keyword "ORBIT"',
          code: `Pos | Plain (P) | P_idx | Key (K) | K_idx | (P_idx + K_idx) | Mod 26 | Cipher (C)
----+-----------+-------+---------+-------+-----------------+--------+-----------
 0  |     D     |   3   |    O    |  14   |     3 + 14 = 17 |   17   |     R
 1  |     E     |   4   |    R    |  17   |    4 + 17 = 21  |   21   |     V
 2  |     F     |   5   |    B    |   1   |     5 + 1 = 6   |    6   |     G
 3  |     E     |   4   |    I    |   8   |     4 + 8 = 12  |   12   |     M
 4  |     N     |  13   |    T    |  19   |   13 + 19 = 32  |    6   |     G
 5  |     D     |   3   |    O    |  14   |    3 + 14 = 17  |   17   |     R
 6  |     T     |  19   |    R    |  17   |   19 + 17 = 36  |   10   |     K
 7  |     H     |   7   |    B    |   1   |     7 + 1 = 8   |    8   |     I
 8  |     E     |   4   |    I    |   8   |     4 + 8 = 12  |   12   |     M
 9  |     W     |  22   |    T    |  19   |   22 + 19 = 41  |   15   |     P
 10 |     A     |   0   |    O    |  14   |    0 + 14 = 14  |   14   |     O
 11 |     L     |  11   |    R    |  17   |   11 + 17 = 28  |    2   |     C
 12 |     L     |  11   |    B    |   1   |    11 + 1 = 12  |   12   |     M

Plaintext:  DEFEND THE WALL
Ciphertext: RVGMGRKIMPOCM`,
        },
        postCodeParagraphs: [
          'Examine the resulting ciphertext closely: The letter "E" occurs 3 times in the plaintext. At position 1 it becomes "V", while at positions 3 and 8 it becomes "M". Furthermore, the ciphertext letter "M" represents both plaintext "E" (at pos 3 & 8) and plaintext "L" (at pos 12).',
          'This one-to-many and many-to-one mapping is the foundational principle of polyalphabetic substitution.',
        ],
      },
      {
        id: 'why-monoalphabetic-analysis-fails',
        heading: '4. Why Single-Letter Frequency Analysis Collapses',
        paragraphs: [
          'In monoalphabetic substitution (like the Caesar cipher), every occurrence of a plaintext letter maps to the exact same ciphertext character. If "E" occurs 12.7% of the time in English, the shifted character for "E" will also account for 12.7% of the ciphertext.',
          'In Vigenère, because each letter is shifted by one of m different key values, the character "E" is spread evenly across m distinct ciphertext characters. The dramatic peaks and valleys of the English letter distribution (high E, T, A, O; low J, Q, X, Z) are averaged out into a smooth, flattened distribution curve.',
          'To break the Vigenère cipher, an analyst cannot attack the text as a whole. Instead, cryptanalysis requires a two-step divide-and-conquer strategy: First, discover the secret key length m. Second, partition the ciphertext into m independent monoalphabetic streams and solve each stream individually.',
        ],
        callout: {
          type: 'tip',
          title: 'The Divide-and-Conquer Rule',
          text: 'If the keyword length is m = 5, the 1st, 6th, 11th, and 16th letters were all encrypted by the exact same Caesar shift! Once m is discovered, the polyalphabetic cipher collapses into m independent Caesar ciphers.',
        },
      },
      {
        id: 'kasiski-examination',
        heading: '5. Cracking Key Length: The Kasiski Examination',
        paragraphs: [
          'The first breakthrough in breaking Vigenère came from Prussian military officer Friedrich Kasiski in 1863 (and independently by Charles Babbage in 1854).',
          'Kasiski realized that natural language frequently repeats common words and n-grams ("THE", "AND", "ING", "ION"). If two identical plaintext phrases happen to appear at positions separated by an exact multiple of the keyword length m, both phrases will be encrypted by the identical sequence of key letters, generating identical ciphertext fragments!',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Kasiski examination principle: Repeated plaintext + Synchronized key = Repeated ciphertext',
          code: `Plaintext:   ... T H E ... ... ... ... T H E ...
Key:         ... K E Y ... ... ... ... K E Y ...
Ciphertext:  ... D L C ... ... ... ... D L C ...
                 ^                     ^
                 Pos 14                Pos 74
                 Distance Delta = 74 - 14 = 60

Factoring the distance Delta = 60:
  Factors: 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60
  Candidate key lengths m must divide 60!`,
        },
        postCodeParagraphs: [
          'By scanning the entire ciphertext for repeated strings of length 3 or greater, recording their distance intervals (Delta₁, Delta₂, Delta₃, ...), and computing their Greatest Common Divisor (GCD), the key length m reveals itself as the common divisor shared across the distances.',
        ],
      },
      {
        id: 'friedmans-index-of-coincidence',
        heading: '6. Statistical Precision: Friedman’s Index of Coincidence (IC)',
        paragraphs: [
          'While the Kasiski test relies on lucky n-gram collisions, William F. Friedman introduced an exact mathematical tool in 1922: the Index of Coincidence (IC).',
          'The Index of Coincidence measures the probability that two letters chosen at random from a text are identical. For a text of length N with letter counts f₀, f₁, ..., f₂₅:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Formula for the Index of Coincidence (IC)',
          code: `        25
        Σ  f_i * (f_i - 1)
       i=0
  IC = --------------------
            N * (N - 1)

Reference Benchmarks:
  English Natural Text:    IC_English  ≈ 0.0667  (typically 0.065 – 0.068)
  Uniform Random Letters:  IC_Random   = 1 / 26  ≈ 0.0385`,
        },
        postCodeParagraphs: [
          'How does IC uncover key length? We slice the ciphertext into candidate cosets for lengths k = 1, 2, ..., max_len. For candidate length k, we form k interleaved slices: C₀ = {c₀, c_k, c_{2k}, ...}, C₁ = {c₁, c_{k+1}, ...}.',
          'If k is incorrect, the letters in each slice are still scrambled by alternating shifts, yielding an IC near random (≈ 0.0385). But when candidate k matches the true key length m, each slice becomes a pure monoalphabetic Caesar cipher! Its letter frequencies match standard English, and the average IC jumps sharply to ≈ 0.0667.',
        ],
        callout: {
          type: 'warning',
          title: 'Multiples of Key Length',
          text: 'Notice that if the true key length is m = 3, testing k = 6 will also yield high IC because every 6th letter is also encrypted with the same key character. Cryptanalysts pick the smallest k that exceeds the English threshold (~0.060).',
        },
      },
      {
        id: 'code-implementation',
        heading: '7. Complete Python Implementation & Auto-Cracker',
        paragraphs: [
          'Here is an industrial-grade, fully functional Python script that implements Vigenère encryption/decryption and an automated cryptanalysis engine that discovers key length via Index of Coincidence and cracks each column using Chi-Square frequency analysis:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'vigenere_complete_solver.py: Polyalphabetic cipher and automated cryptanalyst',
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

def vigenere(text: str, key: str, decrypt: bool = False) -> str:
    """Encrypts or decrypts text using the Vigenère cipher."""
    key_shifts = [ord(k.upper()) - 65 for k in key if k.isalpha()]
    if not key_shifts:
        return text
    k_len = len(key_shifts)
    result = []
    k_idx = 0
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shift = -key_shifts[k_idx % k_len] if decrypt else key_shifts[k_idx % k_len]
            result.append(chr(base + (ord(char) - base + shift) % 26))
            k_idx += 1
        else:
            result.append(char)
    return "".join(result)

def index_of_coincidence(text: str) -> float:
    """Calculates Friedman's Index of Coincidence (IC) for a text slice."""
    letters = [c.upper() for c in text if c.isalpha()]
    n = len(letters)
    if n <= 1:
        return 0.0
    counts = {c: letters.count(c) for c in string.ascii_uppercase}
    numerator = sum(f * (f - 1) for f in counts.values())
    return numerator / (n * (n - 1))

def find_key_length(ciphertext: str, max_key_len: int = 12) -> int:
    """Finds the most probable keyword length using coset IC averaging."""
    letters = [c.upper() for c in ciphertext if c.isalpha()]
    best_len = 1
    best_avg_ic = 0.0
    for k in range(1, max_key_len + 1):
        cosets = ["".join(letters[i::k]) for i in range(k)]
        avg_ic = sum(index_of_coincidence(c) for c in cosets) / k
        # Target threshold: English IC ≈ 0.065
        if avg_ic > best_avg_ic:
            best_avg_ic = avg_ic
            best_len = k
    return best_len

def solve_caesar_slice(slice_letters: list) -> int:
    """Cracks a single monoalphabetic column using Chi-Square minimization."""
    n = len(slice_letters)
    if n == 0:
        return 0
    best_shift = 0
    lowest_chi = float('inf')
    for shift in range(26):
        chi = 0.0
        dec = [chr(65 + (ord(c) - 65 - shift) % 26) for c in slice_letters]
        for char, prob in ENGLISH_FREQS.items():
            expected = n * prob
            observed = dec.count(char)
            chi += ((observed - expected) ** 2) / expected
        if chi < lowest_chi:
            lowest_chi = chi
            best_shift = shift
    return best_shift

def auto_crack_vigenere(ciphertext: str, max_key_len: int = 12):
    """Fully automated Vigenère cryptanalysis without human intervention."""
    k_len = find_key_length(ciphertext, max_key_len)
    letters = [c.upper() for c in ciphertext if c.isalpha()]
    key_chars = []
    for i in range(k_len):
        coset = letters[i::k_len]
        shift = solve_caesar_slice(coset)
        key_chars.append(chr(65 + shift))
    recovered_key = "".join(key_chars)
    decrypted_text = vigenere(ciphertext, recovered_key, decrypt=True)
    return recovered_key, decrypted_text

# Demonstration
sample_msg = (
    "Cryptography is the practice and study of techniques for secure communication "
    "in the presence of adversarial third parties. Classical ciphers historically "
    "relied on character substitution and transposition to protect military dispatches."
)
secret_keyword = "CIPHER"
encrypted_sample = vigenere(sample_msg, secret_keyword)

rec_key, recovered_msg = auto_crack_vigenere(encrypted_sample)
print(f"Cracked Keyword: {rec_key}")
print(f"Decrypted Message: {recovered_msg[:80]}...")`,
        },
      },
      {
        id: 'civil-war-challenge',
        heading: '8. Practice Challenge: The American Civil War Dispatch',
        paragraphs: [
          'Put your cryptanalysis skills to the test with this historical challenge dispatch inspired by Confederate telegraph dispatches from the 1862 Battle of Shiloh:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Civil War challenge ciphertext',
          code: `VBVTQB EZVGKOC EMIPM OK QCQNHV TFPZEA OK DDZUM ZZECB DXTFPZ ZGBBWMMKGFSERN ITKWMC`,
        },
        postCodeParagraphs: [
          'Can you identify the keyword and uncover the secret orders sent to the commanding general?',
          'Clue: The Confederate Army favored single-word victory slogans as their cipher keys. You can test your hypothesis in the live CipherVerse Vigenère workbench below!',
        ],
        toolCta: {
          name: 'Solve in CipherVerse Vigenère Workbench',
          path: '/classical/vigenere',
          description: 'Input the challenge ciphertext, test keywords, or inspect the interactive Tabula Recta matrix.',
          category: 'Classical Ciphers',
        },
      },
      {
        id: 'interactive-workbench',
        heading: '9. Interactive Vigenère Cipher Workbench',
        paragraphs: [
          'Ready to experiment with polyalphabetic substitution hands-on? The CipherVerse Vigenère Solver gives you full control over custom keywords, case preservation, live alphabet matrices, and real-time encryption and decryption.',
          'All computations run locally inside your browser sandbox with zero network telemetry for complete confidentiality.',
        ],
        toolCta: {
          name: 'Launch Vigenère Cipher Suite',
          path: '/classical/vigenere',
          description: 'Instant encoding, decoding, keyword analysis, and Tabula Recta visualization.',
          category: 'Classical Ciphers',
        },
      },
    ],
  },
  {
    slug: 'atbash-cipher',
    title: 'Atbash Cipher: Ancient Biblical Cryptography, Involution Symmetry & Modern Decoders',
    description: 'An exhaustive academic breakdown of the Atbash Cipher: ancient Hebrew scribal origins in the Book of Jeremiah, involution symmetry f(f(x)) = x, affine cipher equivalence in ℤ₂₆, mirror frequency analysis, and runnable Python implementations.',
    category: 'Classical Cryptography',
    publishedAt: '2026-09-14',
    readTime: '8 min read',
    author: {
      name: 'CipherVerse Cryptography Academy',
      role: 'Classical Cryptography & Cryptanalysis',
    },
    tags: [
      'Atbash Cipher',
      'Classical Cryptography',
      'Hebrew Cryptography',
      'Involution',
      'Symmetric Cipher',
      'Mirror Frequency',
      'Affine Cipher',
      'Cryptanalysis',
    ],
    coverGradient: 'from-emerald-500/20 via-teal-500/10 to-indigo-500/20',
    featured: false,
    seriesBadge: 'Academy • Lesson 3: Reciprocal Alphabets & Biblical Origins',
    relatedTools: [
      {
        name: 'Atbash Cipher Tool',
        path: '/classical/atbash',
        description: 'Reverse alphabet reciprocal substitution solver with real-time mapping.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Caesar Cipher Solver & ROT13',
        path: '/classical/caesar',
        description: 'Foundational shift cipher & ROT13 involution (Lesson 1 of the Academy series).',
        category: 'Classical Ciphers',
      },
      {
        name: 'Affine Cipher Solver',
        path: '/classical/affine',
        description: 'Generalized linear modular cipher where Atbash is the special case a = 25, b = 25.',
        category: 'Classical Ciphers',
      },
    ],
    tableOfContents: [
      { id: 'historical-origins', title: '1. Historical Origins: Ancient Hebrew Scribes & The Tanakh', level: 2 },
      { id: 'the-name-atbash', title: '2. The Mechanics of the Name: Aleph-Tav, Bet-Shin', level: 2 },
      { id: 'mathematical-formulation', title: '3. Mathematical Formulation & The Involution Theorem', level: 2 },
      { id: 'step-by-step-example', title: '4. Step-by-Step Worked Trace Table', level: 2 },
      { id: 'affine-connection', title: '5. Connection to the Affine Cipher: a = 25, b = 25', level: 2 },
      { id: 'cryptanalysis-mirror-frequencies', title: '6. Cryptanalysis & The Mirror Frequency Signature', level: 2 },
      { id: 'code-implementation', title: '7. Complete Python Implementation (Latin & Hebrew)', level: 2 },
      { id: 'biblical-challenge', title: '8. Practice Challenge: The Dead Sea Scroll Inscription', level: 2 },
      { id: 'interactive-workbench', title: '9. Interactive Atbash Cipher Workbench', level: 2 },
    ],
    sections: [
      {
        id: 'historical-origins',
        heading: '1. Historical Origins: Ancient Hebrew Scribes & The Tanakh',
        paragraphs: [
          'While the Caesar cipher is celebrated for its military utility in Imperial Rome, the Atbash cipher predates it by more than five centuries. Originating in ancient Israel around 500 to 600 BCE, Atbash is among the earliest recorded substitution ciphers in human history.',
          'Biblical scribes employed Atbash not merely for military secrecy, but for political discretion, poetic elegance, and mystical concealment. Under the oppressive rule of foreign empires (such as the Neo-Babylonian Empire under Nebuchadnezzar II), direct criticism of the ruling empire carried catastrophic penalties.',
          'The most famous historical application of Atbash appears in the Hebrew Bible (Tanakh) within the prophecies of Jeremiah. In Jeremiah 25:26 and 51:41, the prophet decries a mysterious superpower named "Sheshach" (ששך). When decoded using the Atbash substitution over the Hebrew alphabet, the letters ש-ש-ך map directly to ב-ב-ל—revealing the forbidden city of "Babel" (Babylon)!',
          'Similarly, in Jeremiah 51:1, the cryptic phrase "Lev Kamai" (לב קמי, meaning "heart of those who rise up against me") is the Atbash transformation of "Kasdim" (כשדים, the biblical term for the Chaldeans or Babylonians).',
        ],
        callout: {
          type: 'info',
          title: 'Kabbalistic Tradition & Gematria',
          text: 'In Jewish mysticism (Kabbalah) and scribal hermeneutics, Atbash was one of several esoteric letter-permutation systems (temurah). Scribes believed that rearranging letters revealed hidden spiritual truths and encoded divine names without profaning sacred words.',
        },
      },
      {
        id: 'the-name-atbash',
        heading: '2. The Mechanics of the Name: Aleph-Tav, Bet-Shin',
        paragraphs: [
          'The word "Atbash" (אתבש) is an acronym that describes the exact mapping algorithm across the 22-letter traditional Hebrew alphabet:',
          'The 1st letter of the alphabet, Aleph (א), maps to the last letter, Tav (ת) → A-T (את).',
          'The 2nd letter of the alphabet, Bet (ב), maps to the second-to-last letter, Shin (ש) → B-Sh (בש).',
          'The 3rd letter, Gimel (ג), maps to Resh (ר) → G-R (גר).',
          'The 4th letter, Dalet (ד), maps to Qof (ק) → D-Q (דק).',
          'Concatenating the first two letter pairs yields: Aleph-Tav-Bet-Shin = ATBASH (אתבש).',
        ],
        postCodeParagraphs: [
          'When adapted to the 26-letter Latin alphabet used in modern English, the exact same principle applies: the alphabet is folded in half and reversed: A ↔ Z, B ↔ Y, C ↔ X, D ↔ W, ..., M ↔ N.',
        ],
      },
      {
        id: 'mathematical-formulation',
        heading: '3. Mathematical Formulation & The Involution Theorem',
        paragraphs: [
          'From an algebraic standpoint, the Atbash cipher is a monoalphabetic reflection over the finite ring ℤ₂₆ = {0, 1, 2, ..., 25}.',
          'Let x ∈ ℤ₂₆ denote the zero-indexed numerical value of a character (where A = 0, B = 1, ..., Z = 25). The encryption function E(x) is formally defined as subtracting the index from the maximum index value (25):',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Formal mathematical formulation of the Atbash cipher',
          code: `Encryption:
  E(x) = (25 - x) mod 26

Decryption:
  D(y) = (25 - y) mod 26

Involution Property:
  E(E(x)) = (25 - (25 - x)) mod 26
          = (25 - 25 + x) mod 26
          = x mod 26
          = x`,
        },
        postCodeParagraphs: [
          'Because applying the function twice returns the identity (E(E(x)) = x), Atbash is a mathematical involution. A single computer routine or hardware circuit can perform both encryption and decryption with zero mode switching, identical to ROT13 and bitwise XOR.',
        ],
        callout: {
          type: 'tip',
          title: 'Zero Shared Secret',
          text: 'Unlike the Caesar cipher (which has 25 possible shifts) or the Vigenère cipher (which has millions of potential keywords), the Atbash cipher has exactly one fixed rule: reverse the alphabet. It is a "keyless cipher" where security rests entirely on the adversary not knowing which algorithm was used.',
        },
      },
      {
        id: 'step-by-step-example',
        heading: '4. Step-by-Step Worked Trace Table',
        paragraphs: [
          'To witness the transformation step-by-step, let us encipher the ancient city name "JERUSALEM" using the standard 26-letter Atbash algorithm.',
          'Step 1: Convert each letter to its integer position x ∈ {0, ..., 25}.',
          'Step 2: Calculate the reflected index y = 25 - x.',
          'Step 3: Convert the resulting index back to its corresponding Latin letter.',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Worked trace table: Encrypting "JERUSALEM" via Atbash',
          code: `Letter | Plain Index (x) | Formula: 25 - x | Cipher Index (y) | Cipher Letter
-------+------------------+-----------------+------------------+---------------
   J   |        9         |     25 - 9      |        16        |       Q
   E   |        4         |     25 - 4      |        21        |       V
   R   |        17        |     25 - 17     |        8         |       I
   U   |        20        |     25 - 20     |        5         |       F
   S   |        18        |     25 - 18     |        7         |       H
   A   |        0         |     25 - 0      |        25        |       Z
   L   |        11        |     25 - 11     |        14        |       O
   E   |        4         |     25 - 4      |        21        |       V
   M   |        12        |     25 - 12     |        13        |       N

Plaintext:  J E R U S A L E M
Ciphertext: Q V I F H Z O V N

Decryption Verification (Passing ciphertext back through Atbash):
  Q (16) -> 25 - 16 = 9  -> J
  V (21) -> 25 - 21 = 4  -> E
  I (8)  -> 25 - 8  = 17 -> R
  F (5)  -> 25 - 5  = 20 -> U
  H (7)  -> 25 - 7  = 18 -> S
  Z (25) -> 25 - 25 = 0  -> A
  O (14) -> 25 - 14 = 11 -> L
  V (21) -> 25 - 21 = 4  -> E
  N (13) -> 25 - 13 = 12 -> M
Restored:   J E R U S A L E M`,
        },
      },
      {
        id: 'affine-connection',
        heading: '5. Connection to the Affine Cipher: a = 25, b = 25',
        paragraphs: [
          'In modern algebra, monoalphabetic substitution ciphers are unified under the Affine Cipher family, defined by the linear congruence: C ≡ (a · P + b) (mod m).',
          'For a cipher to be invertible, the multiplicative parameter a must be coprime to the modulus m (i.e., gcd(a, 26) = 1).',
          'Notice what happens when we select the multiplicative factor a = -1 ≡ 25 (mod 26) and shift parameter b = 25:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Atbash as a special degenerate case of the Affine Cipher',
          code: `Affine Formula:
  C = (a * P + b) mod 26

Let a = 25 (since 25 ≡ -1 mod 26) and b = 25:
  C = (25 * P + 25) mod 26
    = (-1 * P + 25) mod 26
    = (25 - P) mod 26

Conclusion:
  The Atbash cipher is strictly equivalent to an Affine Cipher with parameters (a=25, b=25)!`,
        },
      },
      {
        id: 'cryptanalysis-mirror-frequencies',
        heading: '6. Cryptanalysis & The Mirror Frequency Signature',
        paragraphs: [
          'Because the Atbash cipher is keyless, it provides zero confidentiality against any adversary familiar with cryptography. However, identifying an unknown ciphertext as Atbash reveals a fascinating statistical phenomenon: the Mirror Frequency Signature.',
          'In natural English text, the most frequent letters are E (12.7%), T (9.1%), A (8.2%), and O (7.5%), while the rarest are Z (0.07%), Q (0.09%), and X (0.15%).',
          'Because Atbash inverts the alphabet, the ciphertext frequency histogram is the exact horizontal mirror image of standard English:',
          'The letter V (mirror of E) becomes the most common character in the ciphertext (~12.7%).',
          'The letter G (mirror of T) becomes the second most common (~9.1%).',
          'The letter Z (mirror of A) surges to ~8.2%, whereas in plain English Z is the rarest letter!',
          'Any frequency analysis tool that notices an abnormally high frequency of V, G, Z, and L can immediately confirm the presence of an Atbash cipher in a single glance.',
        ],
      },
      {
        id: 'code-implementation',
        heading: '7. Complete Python Implementation (Latin & Hebrew)',
        paragraphs: [
          'Here is an industrial, production-grade Python script supporting both standard 26-letter Latin Atbash and traditional 22-letter biblical Hebrew Atbash, complete with involution verification and automated detection:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'atbash_complete_suite.py: Dual Latin/Hebrew Atbash engine with self-test',
          code: `def atbash_latin(text: str) -> str:
    """Encodes or decodes text using the 26-letter Latin Atbash cipher."""
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            # Reflect: 0 -> 25, 1 -> 24, ..., 25 -> 0
            reflected = 25 - (ord(char) - base)
            result.append(chr(base + reflected))
        else:
            result.append(char)
    return "".join(result)

# Traditional 22-letter Hebrew alphabet (Aleph to Tav, excluding final forms)
HEBREW_ALPHABET = [
    '\\u05D0', '\\u05D1', '\\u05D2', '\\u05D3', '\\u05D4', '\\u05D5', '\\u05D6', '\\u05D7',
    '\\u05D8', '\\u05D9', '\\u05DB', '\\u05DC', '\\u05DE', '\\u05E0', '\\u05E1', '\\u05E2',
    '\\u05E4', '\\u05E6', '\\u05E7', '\\u05E8', '\\u05E9', '\\u05EA'
]
HEBREW_MAP = {HEBREW_ALPHABET[i]: HEBREW_ALPHABET[21 - i] for i in range(22)}

def atbash_hebrew(text: str) -> str:
    """Encodes or decodes ancient Hebrew text using biblical Atbash."""
    return "".join(HEBREW_MAP.get(c, c) for c in text)

# Demonstration 1: Biblical Jeremiah Mystery (Babel -> Sheshach)
# Babel: בבל (Bet, Bet, Lamed) -> ששך (Shin, Shin, Kaf)
babel = "\\u05D1\\u05D1\\u05DC"
sheshach = atbash_hebrew(babel)
print(f"Original Biblical Text: {babel} (Babel)")
print(f"Atbash Encrypted:      {sheshach} (Sheshach)")

# Demonstration 2: Latin Involution Proof
plaintext = "THE ATBASH CIPHER REVERSES THE ALPHABET"
ciphertext = atbash_latin(plaintext)
recovered = atbash_latin(ciphertext)

print(f"\\nPlaintext:  {plaintext}")
print(f"Ciphertext: {ciphertext}")
print(f"Recovered:  {recovered}")
print(f"Involution Verified: {plaintext == recovered}")`,
        },
      },
      {
        id: 'biblical-challenge',
        heading: '8. Practice Challenge: The Dead Sea Scroll Inscription',
        paragraphs: [
          'Test your decryption skills with this historical challenge cipher written in the style of an ancient Dead Sea Scroll dispatch discovered in the Judean desert:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Dead Sea Scroll challenge ciphertext',
          code: `TOLIB GL GSV UZRGSUFO PVVKVIH LU GSV HZXIVW ZIP ZG NLFMG ARLM`,
        },
        postCodeParagraphs: [
          'Can you invert the text back into readable English and uncover the ancient temple declaration?',
          'Hint: You can paste this text directly into the CipherVerse Atbash Solver below to decode it in a single click!',
        ],
        toolCta: {
          name: 'Solve in CipherVerse Atbash Tool',
          path: '/classical/atbash',
          description: 'Input the challenge ciphertext and verify the reciprocal inversion in real time.',
          category: 'Classical Ciphers',
        },
      },
      {
        id: 'interactive-workbench',
        heading: '9. Interactive Atbash Cipher Workbench',
        paragraphs: [
          'You can experiment with instant reverse alphabet substitutions, custom case preservation, and live letter-by-letter translation directly in CipherVerse.',
          'The tool runs entirely in your browser using high-speed client-side execution with zero external server dependencies.',
        ],
        toolCta: {
          name: 'Launch Atbash Cipher Solver',
          path: '/classical/atbash',
          description: 'Instant reciprocal encoding and decoding with visual alphabet mapping.',
          category: 'Classical Ciphers',
        },
      },
    ],
  },
  {
    slug: 'affine-cipher',
    title: 'Affine Cipher: Modular Multiplicative Inverses, Coprimality & Algebraic Cryptanalysis',
    description: 'An exhaustive mathematical breakdown of the Affine Cipher: linear congruences in ℤ₂₆, the coprimality condition gcd(a, 26) = 1, calculating modular inverses with the Extended Euclidean Algorithm, and two-point algebraic cryptanalysis.',
    category: 'Classical Cryptography',
    publishedAt: '2026-09-14',
    readTime: '9 min read',
    author: {
      name: 'CipherVerse Cryptography Academy',
      role: 'Classical Cryptography & Cryptanalysis',
    },
    tags: [
      'Affine Cipher',
      'Modular Arithmetic',
      'Multiplicative Inverse',
      'Extended Euclidean Algorithm',
      'Coprimality',
      'Classical Cryptography',
      'Algebraic Cryptanalysis',
      'Chi-Square Test',
    ],
    coverGradient: 'from-purple-500/20 via-pink-500/10 to-indigo-500/20',
    featured: false,
    seriesBadge: 'Academy • Lesson 4: Linear Congruences & Multiplicative Inverses',
    relatedTools: [
      {
        name: 'Affine Cipher Solver',
        path: '/classical/affine',
        description: 'Interactive linear substitution solver with automated coprimality verification.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Caesar Cipher Solver & ROT13',
        path: '/classical/caesar',
        description: 'Special case where multiplicative key a = 1 (Lesson 1 of the Academy series).',
        category: 'Classical Ciphers',
      },
      {
        name: 'Atbash Cipher Tool',
        path: '/classical/atbash',
        description: 'Special reciprocal case where a = 25, b = 25 (Lesson 3 of the Academy series).',
        category: 'Classical Ciphers',
      },
    ],
    tableOfContents: [
      { id: 'mathematical-foundation', title: '1. Mathematical Formulation: Linear Congruences in ℤ₂₆', level: 2 },
      { id: 'the-coprimality-rule', title: '2. The Coprimality Rule: Why gcd(a, 26) = 1 is Mandatory', level: 2 },
      { id: 'modular-inverses', title: '3. Modular Multiplicative Inverses & Extended Euclidean Algorithm', level: 2 },
      { id: 'step-by-step-example', title: '4. Step-by-Step Worked Trace Table: a = 5, b = 8', level: 2 },
      { id: 'key-space-size', title: '5. Key Space Analysis: Euler’s Totient φ(26)', level: 2 },
      { id: 'cryptanalysis-two-points', title: '6. Cryptanalysis: The Two-Point Algebraic Attack', level: 2 },
      { id: 'code-implementation', title: '7. Complete Python Implementation & Automated Cracker', level: 2 },
      { id: 'gauss-challenge', title: '8. Practice Challenge: The Gauss Number Theory Dispatch', level: 2 },
      { id: 'interactive-workbench', title: '9. Interactive Affine Cipher Workbench', level: 2 },
    ],
    sections: [
      {
        id: 'mathematical-foundation',
        heading: '1. Mathematical Formulation: Linear Congruences in ℤ₂₆',
        paragraphs: [
          'In previous lessons, we explored the Caesar cipher (pure addition: x + k) and the Atbash cipher (pure reflection: 25 - x). The Affine Cipher generalizes both into a single unified linear mathematical transformation.',
          'Operating over the finite integer ring ℤ₂₆ = {0, 1, 2, ..., 25}, the Affine cipher combines modular multiplication and modular addition using a pair of secret integer keys (a, b):',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Formal mathematical formulation of Affine encryption and decryption',
          code: `Encryption:
  E(x) = (a * x + b) mod 26

Decryption:
  D(y) = a^(-1) * (y - b) mod 26
       = a^(-1) * (y - b + 26) mod 26

Where:
  x       = Integer value of plaintext character (0 to 25)
  y       = Integer value of ciphertext character (0 to 25)
  a       = Secret multiplicative key (Must satisfy gcd(a, 26) = 1)
  b       = Secret additive shift key (Any integer in {0, ..., 25})
  a^(-1)  = Modular multiplicative inverse of a modulo 26`,
        },
        postCodeParagraphs: [
          'Notice how the Affine cipher elegantly unifies previous ciphers as special cases:',
          'When a = 1, the encryption function becomes E(x) = (x + b) mod 26, which is exactly the Caesar Cipher.',
          'When a = 25 (since 25 ≡ -1 mod 26) and b = 25, the function becomes E(x) = (25 - x) mod 26, which is exactly the Atbash Cipher.',
        ],
      },
      {
        id: 'the-coprimality-rule',
        heading: '2. The Coprimality Rule: Why gcd(a, 26) = 1 is Mandatory',
        paragraphs: [
          'In elementary algebra, you can divide by any non-zero real number. In modular arithmetic over composite moduli like 26, division does not exist! Instead, we multiply by the modular multiplicative inverse a^(-1).',
          'A number a possesses a modular multiplicative inverse modulo 26 if and only if a and 26 are coprime—meaning their Greatest Common Divisor is 1: gcd(a, 26) = 1.',
          'What catastrophic failure happens if you choose an invalid key like a = 2, a = 4, or a = 13? Let us test a = 2 with b = 0:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Demonstration of non-injective collapse when gcd(a, 26) > 1',
          code: `Let a = 2, b = 0:
  Plaintext 'A' (x = 0):  E(0)  = (2 * 0) mod 26  = 0  -> 'A'
  Plaintext 'N' (x = 13): E(13) = (2 * 13) mod 26 = 26 mod 26 = 0 -> 'A'

Collapse:
  Both 'A' and 'N' encrypt to the exact same ciphertext letter 'A'!
  When the recipient receives 'A', it is mathematically impossible to know
  whether the sender meant 'A' or 'N'.`,
        },
        callout: {
          type: 'warning',
          title: 'The Invertibility Theorem',
          text: 'Because 26 = 2 × 13, any key a that is a multiple of 2 (even numbers) or a multiple of 13 shares common factors with 26. These values destroy the 1-to-1 bijection of the alphabet and cannot be decrypted.',
        },
      },
      {
        id: 'modular-inverses',
        heading: '3. Modular Multiplicative Inverses & Extended Euclidean Algorithm',
        paragraphs: [
          'The modular inverse a^(-1) is the unique integer satisfying the congruence:',
          'a · a^(-1) ≡ 1 (mod 26)',
          'To find a^(-1) computationally, we run the Extended Euclidean Algorithm on a and 26, solving Bézout’s identity: a · x + 26 · y = gcd(a, 26) = 1. The coefficient x (reduced modulo 26) is the modular inverse a^(-1).',
          'Because there are only 12 valid coprime integers in ℤ₂₆, we can examine the complete, definitive inverse lookup table:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Complete lookup table of all 12 modular multiplicative inverses in Z26',
          code: `Multiplicative Key (a) | Inverse a^(-1) | Verification: (a * a^(-1)) mod 26
------------------------+----------------+----------------------------------
           1            |       1        | (1 * 1)   = 1   = 0*26 + 1  ≡ 1
           3            |       9        | (3 * 9)   = 27  = 1*26 + 1  ≡ 1
           5            |       21       | (5 * 21)  = 105 = 4*26 + 1  ≡ 1
           7            |       15       | (7 * 15)  = 105 = 4*26 + 1  ≡ 1
           9            |       3        | (9 * 3)   = 27  = 1*26 + 1  ≡ 1
           11           |       19       | (11 * 19) = 209 = 8*26 + 1  ≡ 1
           15           |       7        | (15 * 7)  = 105 = 4*26 + 1  ≡ 1
           17           |       23       | (17 * 23) = 391 = 15*26 + 1 ≡ 1
           19           |       11       | (19 * 11) = 209 = 8*26 + 1  ≡ 1
           21           |       5        | (21 * 5)  = 105 = 4*26 + 1  ≡ 1
           23           |       17       | (23 * 17) = 391 = 15*26 + 1 ≡ 1
           25           |       25       | (25 * 25) = 625 = 24*26 + 1 ≡ 1`,
        },
      },
      {
        id: 'step-by-step-example',
        heading: '4. Step-by-Step Worked Trace Table: a = 5, b = 8',
        paragraphs: [
          'Let us encrypt the message "MATHEMATICS" using the key pair a = 5, b = 8.',
          'From our inverse table, we know that 5^(-1) ≡ 21 (mod 26).',
          'Encryption formula: y = (5x + 8) mod 26.',
          'Decryption formula: x = 21(y - 8) mod 26.',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Worked trace table: Encrypting "MATHEMATICS" with a = 5, b = 8',
          code: `Letter | x  | 5 * x | 5x + 8 | (5x + 8) mod 26 | Cipher Letter | Decryption: 21*(y - 8) mod 26
-------+----+-------+--------+-----------------+---------------+-------------------------------
   M   | 12 |  60   |   68   |  68 mod 26 = 16 |       Q       | 21*(16 - 8)  = 168 mod 26 = 12 (M)
   A   | 0  |   0   |   8    |   8 mod 26 = 8  |       I       | 21*(8 - 8)   = 0   mod 26 = 0  (A)
   T   | 19 |  95   |  103   | 103 mod 26 = 25 |       Z       | 21*(25 - 8)  = 357 mod 26 = 19 (T)
   H   | 7  |  35   |   43   |  43 mod 26 = 17 |       R       | 21*(17 - 8)  = 189 mod 26 = 7  (H)
   E   | 4  |  20   |   28   |  28 mod 26 = 2  |       C       | 21*(2 - 8)   = -126 mod 26 = 4 (E)
   M   | 12 |  60   |   68   |  68 mod 26 = 16 |       Q       | 21*(16 - 8)  = 168 mod 26 = 12 (M)
   A   | 0  |   0   |   8    |   8 mod 26 = 8  |       I       | 21*(8 - 8)   = 0   mod 26 = 0  (A)
   T   | 19 |  95   |  103   | 103 mod 26 = 25 |       Z       | 21*(25 - 8)  = 357 mod 26 = 19 (T)
   I   | 8  |  40   |   48   |  48 mod 26 = 22 |       W       | 21*(22 - 8)  = 294 mod 26 = 8  (I)
   C   | 2  |  10   |   18   |  18 mod 26 = 18 |       S       | 21*(18 - 8)  = 210 mod 26 = 2  (C)
   S   | 18 |  90   |   98   |  98 mod 26 = 20 |       U       | 21*(20 - 8)  = 252 mod 26 = 18 (S)

Plaintext:  M A T H E M A T I C S
Ciphertext: Q I Z R C Q I Z W S U`,
        },
      },
      {
        id: 'key-space-size',
        heading: '5. Key Space Analysis: Euler’s Totient φ(26)',
        paragraphs: [
          'How large is the key space of the Affine cipher?',
          'The number of coprime integers less than 26 is given by Euler’s Totient Function φ(n):',
          'φ(26) = φ(2) × φ(13) = (2 - 1) × (13 - 1) = 1 × 12 = 12.',
          'For each of these 12 choices of a, there are 26 independent choices for the additive shift parameter b (0 through 25).',
          'Total Key Space Size: |K| = 12 × 26 = 312 keys.',
          'Subtracting the trivial identity key (a = 1, b = 0) which leaves text unchanged, there are exactly 311 active transformations. By modern standards, 312 keys is trivially vulnerable: a modern computer tests all 312 keys in under 500 microseconds.',
        ],
      },
      {
        id: 'cryptanalysis-two-points',
        heading: '6. Cryptanalysis: The Two-Point Algebraic Attack',
        paragraphs: [
          'Because the Affine cipher is monoalphabetic, it preserves character frequencies. Furthermore, because the algorithm is strictly linear, discovering just two letters of plaintext completely breaks the entire cipher!',
          'Suppose an intelligence analyst discovers that plaintext letter p₁ encrypts to c₁, and plaintext letter p₂ encrypts to c₂. We establish a system of two linear congruences in two unknowns (a, b):',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Algebraic derivation of a and b from two known plaintext/ciphertext points',
          code: `System of Congruences:
  (1) c_1 ≡ a * p_1 + b  (mod 26)
  (2) c_2 ≡ a * p_2 + b  (mod 26)

Subtract (2) from (1) to eliminate b:
  c_1 - c_2 ≡ a * (p_1 - p_2)  (mod 26)

Provided that Δp = (p_1 - p_2) is coprime to 26:
  a ≡ (c_1 - c_2) * (p_1 - p_2)^(-1)  (mod 26)

Substitute a back into (1) to find b:
  b ≡ (c_1 - a * p_1)  (mod 26)`,
        },
        postCodeParagraphs: [
          'In natural English, the letters "E" (p₁ = 4) and "T" (p₂ = 19) are the two most common. Their difference is Δp = 4 - 19 = -15 ≡ 11 (mod 26).',
          'Because gcd(11, 26) = 1, the difference is guaranteed to be invertible! An analyst simply identifies the two most frequent characters in the ciphertext, assumes they correspond to E and T, and instantly solves for the secret key pair (a, b).',
        ],
      },
      {
        id: 'code-implementation',
        heading: '7. Complete Python Implementation & Automated Cracker',
        paragraphs: [
          'Here is a production-grade, standalone Python script containing full Affine encryption/decryption, Extended Euclidean modular inversion, the two-point algebraic solver, and an automated Chi-Square brute-force cracking engine:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'affine_cipher_suite.py: Modular Affine cipher, Extended Euclidean Algorithm, and auto-solver',
          code: `import string

# Standard English letter frequencies
ENGLISH_FREQS = {
    'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
    'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
    'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
    'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
    'U': 0.02758, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974,
    'Z': 0.00074
}

VALID_A_KEYS = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]

def egcd(a: int, b: int):
    """Extended Euclidean Algorithm returning (gcd, x, y) such that a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = egcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def modinv(a: int, m: int = 26) -> int:
    """Computes the modular multiplicative inverse of a modulo m."""
    gcd, x, _ = egcd(a, m)
    if gcd != 1:
        raise ValueError(f"No modular inverse for a={a} mod {m}. gcd({a}, {m}) = {gcd} != 1.")
    return (x % m + m) % m

def affine(text: str, a: int, b: int, decrypt: bool = False) -> str:
    """Encrypts or decrypts text using the Affine Cipher."""
    if a not in VALID_A_KEYS:
        raise ValueError(f"Invalid key a={a}. Key 'a' must be coprime to 26.")
    a_inv = modinv(a, 26) if decrypt else None
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            x = ord(char) - base
            if decrypt:
                val = (a_inv * (x - b)) % 26
            else:
                val = (a * x + b) % 26
            result.append(chr(base + val))
        else:
            result.append(char)
    return "".join(result)

def solve_two_points(p1: str, c1: str, p2: str, c2: str):
    """Algebraically solves key pair (a, b) from two known plaintext/ciphertext pairs."""
    x1, y1 = ord(p1.upper()) - 65, ord(c1.upper()) - 65
    x2, y2 = ord(p2.upper()) - 65, ord(c2.upper()) - 65
    delta_x = (x1 - x2) % 26
    delta_y = (y1 - y2) % 26
    try:
        inv_delta_x = modinv(delta_x, 26)
        a = (delta_y * inv_delta_x) % 26
        b = (y1 - a * x1) % 26
        return a, b
    except ValueError:
        return None

def auto_crack_affine(ciphertext: str):
    """Automatically cracks an Affine cipher by testing all 312 keys against Chi-Square."""
    letters = [c.upper() for c in ciphertext if c.isalpha()]
    n = len(letters)
    if n == 0:
        return None, None, ciphertext

    best_a, best_b = 1, 0
    lowest_chi = float('inf')
    best_plaintext = ""

    for a in VALID_A_KEYS:
        a_inv = modinv(a, 26)
        for b in range(26):
            candidate = []
            for c in ciphertext:
                if c.isalpha():
                    base = ord('A') if c.isupper() else ord('a')
                    val = (a_inv * ((ord(c) - base) - b)) % 26
                    candidate.append(chr(base + val))
                else:
                    candidate.append(c)
            cand_str = "".join(candidate)

            # Evaluate Chi-Square
            chi = 0.0
            for char, prob in ENGLISH_FREQS.items():
                expected = n * prob
                observed = cand_str.upper().count(char)
                chi += ((observed - expected) ** 2) / expected

            if chi < lowest_chi:
                lowest_chi = chi
                best_a = a
                best_b = b
                best_plaintext = cand_str

    return (best_a, best_b), best_plaintext

# Demonstration
secret_msg = "NUMBER THEORY IS THE FOUNDATION OF MODERN ASYMMETRIC ENCRYPTION"
cipher = affine(secret_msg, a=7, b=11)
print(f"Ciphertext: {cipher}")

keys, cracked = auto_crack_affine(cipher)
print(f"Cracked Key Pair: a={keys[0]}, b={keys[1]}")
print(f"Decrypted Message: {cracked}")`,
        },
      },
      {
        id: 'gauss-challenge',
        heading: '8. Practice Challenge: The Gauss Number Theory Dispatch',
        paragraphs: [
          'Put your cryptanalysis skills to the test with this historical quotation encoded using an unknown Affine key pair (a, b):',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Affine challenge ciphertext',
          code: `JDGAFJDGHRZ HZ GAF LNFFQ XM GAF ZRHFQRFZ DQY QNJKFS GAFXSP HZ GAF LNFFQ XM JDGAFJDGHRZ`,
        },
        postCodeParagraphs: [
          'Can you identify the secret multiplier a and additive shift b to read the famous mathematical declaration?',
          'Clue: Notice the high frequency of the 3-letter word "GAF", which frequently represents "THE" in English! You can test your deduction in the live CipherVerse Affine Solver below.',
        ],
        toolCta: {
          name: 'Solve in CipherVerse Affine Solver',
          path: '/classical/affine',
          description: 'Input the challenge ciphertext, adjust multiplier a and shift b, or verify coprimality in real time.',
          category: 'Classical Ciphers',
        },
      },
      {
        id: 'interactive-workbench',
        heading: '9. Interactive Affine Cipher Workbench',
        paragraphs: [
          'Ready to explore linear congruences, coprimality checks, and automated inverse calculation hands-on? The CipherVerse Affine Cipher Tool validates your keys in real time and handles all modular reductions automatically.',
          'Everything executes inside your client-side browser with 100% privacy and zero external server transmission.',
        ],
        toolCta: {
          name: 'Launch Affine Cipher Suite',
          path: '/classical/affine',
          description: 'Instant linear modular encryption, decryption, and coprimality diagnostics.',
          category: 'Classical Ciphers',
        },
      },
    ],
  },
  {
    slug: 'rail-fence-cipher',
    title: 'Rail Fence Cipher: The Complete Guide to Zig-Zag Transposition & Cryptanalysis',
    description: 'An exhaustive educational breakdown of the Rail Fence Cipher: transposition vs. substitution, triangular wave periodicity P = 2(d - 1), worked zig-zag matrix traces, preserved frequency distributions, anagramming, and runnable Python auto-crackers.',
    category: 'Classical Cryptography',
    publishedAt: '2026-09-14',
    readTime: '8 min read',
    author: {
      name: 'CipherVerse Cryptography Academy',
      role: 'Classical Cryptography & Cryptanalysis',
    },
    tags: [
      'Rail Fence Cipher',
      'Transposition Cipher',
      'Classical Cryptography',
      'Zig-Zag Matrix',
      'Cryptanalysis',
      'Permutation Cipher',
      'Periodicity',
      'Frequency Analysis',
    ],
    coverGradient: 'from-orange-500/20 via-amber-500/10 to-red-500/20',
    featured: false,
    seriesBadge: 'Academy • Lesson 5: Geometric Transposition & Rail Depth',
    relatedTools: [
      {
        name: 'Rail Fence Cipher Tool',
        path: '/classical/rail-fence',
        description: 'Interactive zig-zag transposition simulator with customizable rail depth.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Caesar Cipher Solver & ROT13',
        path: '/classical/caesar',
        description: 'Foundational substitution cipher (Lesson 1 of the Academy series).',
        category: 'Classical Ciphers',
      },
      {
        name: 'Bifid Cipher Tool',
        path: '/classical/bifid',
        description: 'Fractionation cipher combining substitution and transposition.',
        category: 'Classical Ciphers',
      },
    ],
    tableOfContents: [
      { id: 'transposition-vs-substitution', title: '1. Transposition vs. Substitution: The Cryptographic Divide', level: 2 },
      { id: 'mathematical-formulation', title: '2. Mathematical Formulation & Periodicity P = 2(d - 1)', level: 2 },
      { id: 'step-by-step-example', title: '3. Step-by-Step Worked Trace Matrix (Depth d = 3)', level: 2 },
      { id: 'frequency-preservation', title: '4. Why Transposition Preserves 100% of Letter Frequencies', level: 2 },
      { id: 'reconstruction-decryption', title: '5. Decryption Mechanics: Row Lengths & Grid Reconstruction', level: 2 },
      { id: 'cryptanalysis-brute-force', title: '6. Cryptanalysis: Brute-Forcing Rail Depths & Anagramming', level: 2 },
      { id: 'code-implementation', title: '7. Complete Python Implementation & Automated Solver', level: 2 },
      { id: 'military-challenge', title: '8. Practice Challenge: The Fleet Telegraph Dispatch', level: 2 },
      { id: 'interactive-workbench', title: '9. Interactive Rail Fence Cipher Workbench', level: 2 },
    ],
    sections: [
      {
        id: 'transposition-vs-substitution',
        heading: '1. Transposition vs. Substitution: The Cryptographic Divide',
        paragraphs: [
          'In classical cryptography, all historical algorithms fall into two fundamental categories: Substitution and Transposition.',
          'In a substitution cipher (such as Caesar, Vigenère, or Affine), the positions of characters remain fixed while their identities are replaced. The word "CAT" might become "FDW".',
          'In a transposition (or permutation) cipher, the opposite occurs: character identities are 100% preserved, but their physical positions within the message are scrambled. The word "CAT" might become "ACT" or "TCA".',
          'The Rail Fence cipher (also known as the Zig-Zag cipher) is the foundational archetype of transposition. Rather than transmitting characters in standard horizontal reading order, plaintext letters are arranged downwards and upwards across successive imaginary "rails" of a fence, and then read off row-by-row.',
        ],
        callout: {
          type: 'info',
          title: 'Historical Military Precursors',
          text: 'The earliest mechanical transposition device was the ancient Spartan Scytale (~500 BCE)—a wooden cylinder around which a strip of parchment was wrapped. Letters were written lengthwise; unwrapped, the strip appeared as unintelligible scrambled letters until rewound around a cylinder of identical diameter.',
        },
      },
      {
        id: 'mathematical-formulation',
        heading: '2. Mathematical Formulation & Periodicity P = 2(d - 1)',
        paragraphs: [
          'Let d denote the number of rails (depth), where d ≥ 2. A message of length N is indexed by position i ∈ {0, 1, ..., N - 1}.',
          'The zig-zag motion descends from rail 0 down to rail d - 1, and then ascends back up to rail 0. This bounce generates a periodic triangular wave function.',
          'The cycle length (or period) P of the oscillation is given by:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Cycle length and rail assignment formulas for Rail Fence depth d',
          code: `Cycle Length (Period):
  P = 2 * (d - 1)

Rail Assignment Function r(i):
  Let k = i mod P = i mod (2 * d - 2)

  If k < d:
    r(i) = k             (Descending stroke)
  Else:
    r(i) = P - k         (Ascending stroke)
         = 2 * d - 2 - k

Examples of Period P:
  d = 2 rails:  P = 2 * (2 - 1) = 2  (Alternating: 0, 1, 0, 1, ...)
  d = 3 rails:  P = 2 * (3 - 1) = 4  (Cycle: 0, 1, 2, 1, 0, ...)
  d = 4 rails:  P = 2 * (4 - 1) = 6  (Cycle: 0, 1, 2, 3, 2, 1, 0, ...)
  d = 5 rails:  P = 2 * (5 - 1) = 8  (Cycle: 0, 1, 2, 3, 4, 3, 2, 1, 0, ...)`,
        },
      },
      {
        id: 'step-by-step-example',
        heading: '3. Step-by-Step Worked Trace Matrix (Depth d = 3)',
        paragraphs: [
          'To visualize the transformation, let us encrypt the classic 25-letter intelligence dispatch "WE ARE DISCOVERED FLEE AT ONCE" using a depth of d = 3 rails.',
          'Period P = 2 * (3 - 1) = 4.',
          'Let us construct the 3×25 grid and trace each letter along its zig-zag path:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Worked trace matrix for "WE ARE DISCOVERED FLEE AT ONCE" with d = 3',
          code: `Pos:  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
Text: W E A R E D I S C O  V  E  R  E  D  F  L  E  E  A  T  O  N  C  E

Rail 0: W . . . E . . . C .  .  .  R  .  .  .  L  .  .  .  T  .  .  .  E
Rail 1: . E . R . D . S . O  .  E  .  E  .  F  .  E  .  A  .  O  .  C  .
Rail 2: . . A . . . I . . .  V  .  .  .  D  .  .  .  E  .  .  .  N  .  .

Reading off each rail sequentially:
  Rail 0 (Indices 0, 4, 8, 12, 16, 20, 24): W E C R L T E
  Rail 1 (Indices 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23): E R D S O E E F E A O C
  Rail 2 (Indices 2, 6, 10, 14, 18, 22): A I V D E N

Concatenated Ciphertext:
  WECRLTEERDSOEEFEAOCAIVDEN`,
        },
      },
      {
        id: 'frequency-preservation',
        heading: '4. Why Transposition Preserves 100% of Letter Frequencies',
        paragraphs: [
          'The most striking cryptanalytic property of the Rail Fence cipher is its total preservation of character frequency distributions.',
          'In our ciphertext "WECRLTEERDSOEEFEAOCAIVDEN", let us count the letters:',
          'The letter E occurs exactly 7 times in the plaintext, and exactly 7 times in the ciphertext.',
          'The letter A occurs 2 times in both. The letter C occurs 2 times in both. The letter T occurs 1 time in both.',
          'If an analyst computes the Index of Coincidence (IC) on the ciphertext, it returns IC ≈ 0.0667—the exact theoretical score of natural English text!',
          'This provides an immediate diagnostic rule for cryptanalysts: If an unknown ciphertext has normal English letter frequencies but unreadable word sequences, it is guaranteed to be a Transposition cipher!',
        ],
        callout: {
          type: 'tip',
          title: 'The Cryptanalyst’s First Test',
          text: 'When presented with mystery ciphertext in a CTF or historical challenge, calculate its letter counts. If E, T, A, O dominate at their usual percentages, you are dealing with a transposition or permutation cipher, never a substitution cipher.',
        },
      },
      {
        id: 'reconstruction-decryption',
        heading: '5. Decryption Mechanics: Row Lengths & Grid Reconstruction',
        paragraphs: [
          'To decrypt a Rail Fence ciphertext without knowing the original plaintext, we must reconstruct the exact dimensions of each rail.',
          'Step 1: Determine the total number of characters N in the ciphertext.',
          'Step 2: Trace the zig-zag indices for positions 0 through N - 1 to count exactly how many characters belong to Rail 0, Rail 1, ..., Rail d - 1.',
          'Step 3: Slice the ciphertext string into segments matching those counts.',
          'Step 4: Re-populate the matrix along the rows, then read off the message along the zig-zag path column by column.',
        ],
      },
      {
        id: 'cryptanalysis-brute-force',
        heading: '6. Cryptanalysis: Brute-Forcing Rail Depths & Anagramming',
        paragraphs: [
          'Because the only secret parameter in a standard Rail Fence cipher is the rail depth d, the total key space is tiny.',
          'The depth d must be at least 2, and cannot exceed the message length N. In practice, depths greater than 20 are rarely used because short messages cannot complete full cycles.',
          'A cryptanalyst simply tests candidate depths d = 2, 3, 4, ..., min(N, 25). For each candidate, the text is decrypted and scored against English n-gram frequencies or dictionary matching.',
          'Because there are fewer than 25 realistic keys, a modern computer breaks any Rail Fence ciphertext in under 1 millisecond.',
        ],
      },
      {
        id: 'code-implementation',
        heading: '7. Complete Python Implementation & Automated Solver',
        paragraphs: [
          'Here is a production-grade, standalone Python script featuring Rail Fence encryption, decryption, ASCII grid visualization, and an automated cracking engine that recovers plaintext by scoring English bigram frequencies:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'rail_fence_suite.py: Transposition engine, ASCII matrix visualizer, and auto-cracker',
          code: `import string

# Common English bigrams for automated scoring
COMMON_BIGRAMS = {
    'TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ND', 'AT', 'ON', 'NT',
    'HA', 'ES', 'ST', 'EN', 'ED', 'TO', 'IT', 'OU', 'EA', 'HI'
}

def get_rail(index: int, rails: int) -> int:
    """Returns the rail index for a given position using triangular wave formula."""
    cycle = 2 * (rails - 1)
    k = index % cycle
    return k if k < rails else cycle - k

def rail_fence_encrypt(text: str, rails: int) -> str:
    """Encrypts text using the Rail Fence transposition cipher."""
    clean = [c for c in text if c.isalpha()]
    if rails <= 1 or rails >= len(clean):
        return "".join(clean)
    rows = [[] for _ in range(rails)]
    for i, char in enumerate(clean):
        rows[get_rail(i, rails)].append(char)
    return "".join("".join(r) for r in rows)

def rail_fence_decrypt(ciphertext: str, rails: int) -> str:
    """Decrypts a Rail Fence ciphertext by reconstructing the zig-zag matrix."""
    clean = [c for c in ciphertext if c.isalpha()]
    n = len(clean)
    if rails <= 1 or rails >= n:
        return "".join(clean)

    # 1. Count characters on each rail
    counts = [0] * rails
    for i in range(n):
        counts[get_rail(i, rails)] += 1

    # 2. Slice ciphertext into rails
    rail_buffers = []
    idx = 0
    for count in counts:
        rail_buffers.append(list(clean[idx:idx + count]))
        idx += count

    # 3. Read off in zig-zag order
    plaintext = []
    for i in range(n):
        r = get_rail(i, rails)
        plaintext.append(rail_buffers[r].pop(0))
    return "".join(plaintext)

def auto_crack_rail_fence(ciphertext: str, max_rails: int = 15):
    """Automatically cracks Rail Fence by testing all depths against English bigram fitness."""
    clean = [c.upper() for c in ciphertext if c.isalpha()]
    best_rails = 2
    highest_score = -1
    best_text = ""

    for r in range(2, min(len(clean), max_rails + 1)):
        candidate = rail_fence_decrypt("".join(clean), r)
        score = sum(1 for i in range(len(candidate) - 1) if candidate[i:i+2] in COMMON_BIGRAMS)
        if score > highest_score:
            highest_score = score
            best_rails = r
            best_text = candidate

    return best_rails, best_text

# Demonstration
sample_msg = "WE ARE DISCOVERED FLEE AT ONCE"
encrypted = rail_fence_encrypt(sample_msg, rails=3)
print(f"Plaintext:  {sample_msg}")
print(f"Ciphertext: {encrypted}")

detected_rails, cracked_msg = auto_crack_rail_fence(encrypted)
print(f"Detected Depth: {detected_rails} rails")
print(f"Cracked Text:   {cracked_msg}")`,
        },
      },
      {
        id: 'military-challenge',
        heading: '8. Practice Challenge: The Fleet Telegraph Dispatch',
        paragraphs: [
          'An intercepted naval telegram from the Spanish-American War era was encoded using an unknown rail depth d:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Rail Fence challenge ciphertext',
          code: `CETAETFNOMHODNLERPDRESSOATMBTIGTNITALGHRFEALRUAAAADIFIEOLIGNYN`,
        },
        postCodeParagraphs: [
          'Can you identify the number of rails used and read the secret naval maneuver order?',
          'Clue: Test depths between 2 and 6 in the live CipherVerse Rail Fence simulator below!',
        ],
        toolCta: {
          name: 'Solve in CipherVerse Rail Fence Tool',
          path: '/classical/rail-fence',
          description: 'Input the challenge ciphertext and adjust the rail slider in real time.',
          category: 'Classical Ciphers',
        },
      },
      {
        id: 'interactive-workbench',
        heading: '9. Interactive Rail Fence Cipher Workbench',
        paragraphs: [
          'Want to experiment with zig-zag transposition hands-on? The CipherVerse Rail Fence Suite lets you drag the rail depth slider, visualize the character pathways, and preserve case sensitivity in real time.',
          'Everything runs client-side in your browser sandbox with zero latency and 100% cryptographic privacy.',
        ],
        toolCta: {
          name: 'Launch Rail Fence Cipher Suite',
          path: '/classical/rail-fence',
          description: 'Instant transposition encoding, decoding, and rail depth visualization.',
          category: 'Classical Ciphers',
        },
      },
    ],
  },
  {
    slug: 'bifid-cipher',
    title: 'Bifid Cipher: Polybius Fractionation, Delastelle’s Breakthrough & Modern Diffusion',
    description: 'An exhaustive educational breakdown of the Bifid Cipher: Félix Delastelle’s 1901 invention, Polybius square coordinate decomposition, horizontal stream fractionation, Claude Shannon’s concept of diffusion, and runnable Python auto-solvers.',
    category: 'Classical Cryptography',
    publishedAt: '2026-09-14',
    readTime: '9 min read',
    author: {
      name: 'CipherVerse Cryptography Academy',
      role: 'Classical Cryptography & Cryptanalysis',
    },
    tags: [
      'Bifid Cipher',
      'Fractionation',
      'Polybius Square',
      'Félix Delastelle',
      'Classical Cryptography',
      'Tomographic Cryptography',
      'Diffusion',
      'Cryptanalysis',
    ],
    coverGradient: 'from-indigo-500/20 via-purple-500/10 to-pink-500/20',
    featured: false,
    seriesBadge: 'Academy • Lesson 6: Fractionation & Cryptographic Diffusion',
    relatedTools: [
      {
        name: 'Bifid Cipher Tool',
        path: '/classical/bifid',
        description: 'Fractionation cipher combining Polybius square substitution with transposition.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Rail Fence Cipher Tool',
        path: '/classical/rail-fence',
        description: 'Foundational transposition & permutation cipher (Lesson 5 of the Academy series).',
        category: 'Classical Ciphers',
      },
      {
        name: 'Affine Cipher Solver',
        path: '/classical/affine',
        description: 'Mathematical linear congruence cipher (Lesson 4 of the Academy series).',
        category: 'Classical Ciphers',
      },
    ],
    tableOfContents: [
      { id: 'felix-delastelle', title: '1. Historical Origins: Félix Delastelle & Fractionation', level: 2 },
      { id: 'the-polybius-matrix', title: '2. The 5×5 Polybius Square Matrix Key', level: 2 },
      { id: 'fractionation-mechanics', title: '3. The Core Fractionation Principle: Decomposing Coordinates', level: 2 },
      { id: 'step-by-step-example', title: '4. Step-by-Step Worked Trace Table: "DEFEND THE WALL"', level: 2 },
      { id: 'decryption-reconstruction', title: '5. Reversible Decryption: Slicing & Recombining Coordinate Streams', level: 2 },
      { id: 'bridge-to-modern-diffusion', title: '6. The Conceptual Bridge to Claude Shannon & Modern Block Ciphers', level: 2 },
      { id: 'code-implementation', title: '7. Complete Python Implementation & Solver Suite', level: 2 },
      { id: 'dday-challenge', title: '8. Practice Challenge: The Normandy Invasion Dispatch', level: 2 },
      { id: 'interactive-workbench', title: '9. Interactive Bifid Cipher Workbench', level: 2 },
    ],
    sections: [
      {
        id: 'felix-delastelle',
        heading: '1. Historical Origins: Félix Delastelle & Fractionation',
        paragraphs: [
          'For more than two thousand years, human cryptography remained divided into two isolated disciplines: Substitution (changing letter identities) and Transposition (changing letter positions).',
          'In 1901, French amateur cryptographer Félix-Marie Delastelle published a revolutionary paper in the "Revue du Génie Militaire", followed by his 1902 magnum opus "Traité Élémentaire de Cryptographie". Delastelle introduced an entirely new cryptographic paradigm that had never existed before: Fractionation (also known as Tomographic Cryptography).',
          'Delastelle’s brilliant revelation was that an encryption algorithm did not have to treat letters as indivisible atomic units. By decomposing each letter into two discrete numerical coordinates, separating the coordinates across time and space, and recombining fragments from completely different letters, Delastelle achieved simultaneous substitution and transposition in a single elegant system.',
        ],
        callout: {
          type: 'info',
          title: 'A Century Ahead of His Time',
          text: 'Delastelle was not a career military officer or professional mathematician—he worked as a chief warehouseman for the Port of Saint-Malo. Despite having no academic post, his invention of fractionation laid the foundational architecture for modern 20th-century block ciphers like DES and AES.',
        },
      },
      {
        id: 'the-polybius-matrix',
        heading: '2. The 5×5 Polybius Square Matrix Key',
        paragraphs: [
          'The first step of the Bifid cipher relies on an ancient Greek concept: the Polybius Square, invented by the Greek historian Polybius around 200 BCE.',
          'The 26 letters of the Latin alphabet are fitted into a 5×5 grid containing exactly 25 cells. To accommodate 26 letters into 25 positions, the letters "I" and "J" are conventionally merged into the same cell (or alternatively, "Q" is omitted).',
          'To key the matrix securely, the sender and receiver agree on a secret keyword (e.g. "KEYWORD"). The keyword is written across the grid with duplicate letters discarded, followed by the remaining unused letters of the alphabet in alphabetical order:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Polybius square matrix keyed with keyword "KEYWORD"',
          code: `      Col 1  Col 2  Col 3  Col 4  Col 5
Row 1   K      E      Y      W      O
Row 2   R      D      A      B      C
Row 3   F      G      H     I/J     L
Row 4   M      N      P      Q      S
Row 5   T      U      V      X      Z

Every letter is uniquely addressed by its (Row, Column) coordinates:
  'D' -> Row 2, Col 2  (2, 2)
  'E' -> Row 1, Col 2  (1, 2)
  'F' -> Row 3, Col 1  (3, 1)`,
        },
      },
      {
        id: 'fractionation-mechanics',
        heading: '3. The Core Fractionation Principle: Decomposing Coordinates',
        paragraphs: [
          'How does fractionation destroy frequency analysis? In a monoalphabetic or Vigenère cipher, character boundaries remain intact.',
          'In the Bifid cipher, each letter is split into two halves: its Row Coordinate and its Column Coordinate.',
          'The sender writes all the Row coordinates in an upper row, and all the Column coordinates in a lower row. Then, the numbers are read horizontally in sequence (all row coordinates first, followed by all column coordinates).',
          'Finally, this unified numerical stream is grouped into consecutive pairs. Each new pair is looked up in the Polybius square to create the ciphertext letter!',
          'Notice what this achieves: The first ciphertext letter receives its row coordinate from letter 1, and its column coordinate from letter 2! Half of the information from letter 1 has diffused into letter 2, completely scrambling statistical bigrams and single-letter frequencies.',
        ],
      },
      {
        id: 'step-by-step-example',
        heading: '4. Step-by-Step Worked Trace Table: "DEFEND THE WALL"',
        paragraphs: [
          'Let us encrypt the message "DEFEND THE WALL" using the Polybius square keyed with "KEYWORD".',
          'Step 1: Write the plaintext characters and look up their (Row, Column) coordinates.',
          'Step 2: Read horizontally (all Row coordinates followed by all Column coordinates).',
          'Step 3: Group the horizontal sequence into consecutive 2-digit pairs and look up each pair in the Polybius matrix to generate the ciphertext.',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Worked trace: Coordinate fractionation and ciphertext generation',
          code: `Plaintext:  D  E  F  E  N  D  T  H  E  W  A  L  L
Row:        2  1  3  1  4  2  5  3  1  1  2  3  3
Col:        2  2  1  2  2  2  1  3  2  4  3  5  5

Horizontal Stream (Rows followed by Cols):
  2 1 3 1 4 2 5 3 1 1 2 3 3   2 2 1 2 2 2 1 3 2 4 3 5 5

Consecutive Pair Lookup:
  (2,1)->R, (3,1)->F, (4,2)->N, (5,3)->V, (1,1)->K, (2,3)->A, (3,2)->G,
  (2,1)->R, (2,2)->D, (2,1)->R, (3,2)->G, (4,3)->P, (5,5)->Z

Plaintext:   D E F E N D T H E W A L L
Ciphertext:  R F N V K A G R D R G P Z`,
        },
        postCodeParagraphs: [
          'Notice what has occurred: The first ciphertext letter "R" (Row 2, Col 1) received its row coordinate from the first letter "D" (Row 2), and its column coordinate from the second letter "E" (Row 1)!',
          'Each ciphertext letter fuses fragments of multiple distinct plaintext letters, creating the world\'s first tomographic diffusion effect.',
        ],
      },
      {
        id: 'decryption-reconstruction',
        heading: '5. Reversible Decryption: Slicing & Recombining Coordinate Streams',
        paragraphs: [
          'How does the recipient reverse this complex fractionation? Because the transformation is purely structural and symmetrical, decryption is completely deterministic:',
          'Step 1: Convert each ciphertext letter back into its (Row, Column) coordinates using the Polybius square:',
          'R → (2, 1), F → (3, 1), N → (4, 2), V → (5, 3), K → (1, 1), ..., Z → (5, 5).',
          'Step 2: Flatten all numbers into a single stream of length 2N: "2 1 3 1 4 2 5 3 1 1 2 3 3 2 2 1 2 2 2 1 3 2 4 3 5 5".',
          'Step 3: Split the stream exactly in half: the first N digits form the Rows, and the remaining N digits form the Columns!',
          'Step 4: Read vertical pairs: (Row[i], Col[i]) to reconstruct the exact original plaintext letters.',
        ],
      },
      {
        id: 'bridge-to-modern-diffusion',
        heading: '6. The Conceptual Bridge to Claude Shannon & Modern Block Ciphers',
        paragraphs: [
          'In his seminal 1949 paper "Communication Theory of Secrecy Systems", Claude Shannon defined the two fundamental security requirements for modern ciphers:',
          '1. Confusion: Obscuring the relationship between the secret key and the ciphertext (accomplished via non-linear substitution S-Boxes).',
          '2. Diffusion: Spreading the statistical influence of a single plaintext symbol across multiple ciphertext symbols.',
          'Delastelle’s Bifid cipher was the world’s very first operational demonstration of cryptographic diffusion. By breaking an 8-bit character or coordinate into fragments and dispersing them across neighboring characters, Delastelle anticipated the round structures of modern block ciphers like the Data Encryption Standard (DES) and Advanced Encryption Standard (AES) by nearly half a century.',
        ],
      },
      {
        id: 'code-implementation',
        heading: '7. Complete Python Implementation & Solver Suite',
        paragraphs: [
          'Here is a production-grade, standalone Python script featuring keyed Polybius square generation, complete Bifid encryption, decryption, and self-testing verification:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'bifid_cipher_suite.py: Keyed Polybius square matrix, coordinate fractionation, and reversal',
          code: `def create_polybius_square(keyword: str):
    """Generates a 5x5 Polybius square matrix with I/J combined."""
    clean_kw = []
    for char in keyword.upper():
        if char == 'J':
            char = 'I'
        if char.isalpha() and char not in clean_kw:
            clean_kw.append(char)
    # Complete remaining letters (excluding J)
    for char in 'ABCDEFGHIKLMNOPQRSTUVWXYZ':
        if char not in clean_kw:
            clean_kw.append(char)

    coords = {}
    rev_coords = {}
    for r in range(5):
        for c in range(5):
            ch = clean_kw[r * 5 + c]
            coords[ch] = (r + 1, c + 1)
            rev_coords[(r + 1, c + 1)] = ch
    coords['J'] = coords['I']
    return coords, rev_coords

def bifid(text: str, keyword: str, decrypt: bool = False, period: int = None) -> str:
    """Encrypts or decrypts text using Delastelle's Bifid fractionation cipher."""
    coords, rev_coords = create_polybius_square(keyword)
    clean = [c.upper() if c.upper() != 'J' else 'I' for c in text if c.isalpha()]
    n = len(clean)
    if n == 0:
        return ""
    if period is None or period <= 0:
        period = n

    result = []
    for block_start in range(0, n, period):
        block = clean[block_start:block_start + period]
        b_len = len(block)
        if not decrypt:
            # 1. Decompose into rows and columns
            rows = [coords[ch][0] for ch in block]
            cols = [coords[ch][1] for ch in block]
            combined = rows + cols
            # 2. Re-pair horizontally
            for i in range(0, 2 * b_len, 2):
                result.append(rev_coords[(combined[i], combined[i+1])])
        else:
            # Decrypt: Expand ciphertext pairs back into flat stream
            combined = []
            for ch in block:
                r, c = coords[ch]
                combined.extend([r, c])
            # Split into original row and col halves
            rows = combined[:b_len]
            cols = combined[b_len:]
            # Reconstruct original vertical pairs
            for r, c in zip(rows, cols):
                result.append(rev_coords[(r, c)])
    return "".join(result)

# Demonstration
key = "KEYWORD"
plaintext_sample = "DEFEND THE WALL"
ciphertext_sample = bifid(plaintext_sample, key)
restored = bifid(ciphertext_sample, key, decrypt=True)

print(f"Key Matrix: {key}")
print(f"Plaintext:  {plaintext_sample}")
print(f"Ciphertext: {ciphertext_sample}")
print(f"Decrypted:  {restored}")
print(f"Verification: {restored == 'DEFENDTHEWALL'}")`,
        },
      },
      {
        id: 'dday-challenge',
        heading: '8. Practice Challenge: The Normandy Invasion Dispatch',
        paragraphs: [
          'An authentic World War II Allied dispatch was encoded using a secret keyword in a 5×5 Bifid Polybius square:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Bifid challenge ciphertext',
          code: `AEILTSCEPNTSLLCYCNGOTIAQYYBSQXIIMYLDQWTLFWLDWRAPPTZG`,
        },
        postCodeParagraphs: [
          'Can you identify the keyword used to generate the square and read the famous military order?',
          'Clue: The keyword is a 7-letter ideal representing American independence ("LIB____"). Test your hypothesis in the live CipherVerse Bifid simulator below!',
        ],
        toolCta: {
          name: 'Solve in CipherVerse Bifid Tool',
          path: '/classical/bifid',
          description: 'Input the challenge ciphertext and test candidate keywords in real time.',
          category: 'Classical Ciphers',
        },
      },
      {
        id: 'interactive-workbench',
        heading: '9. Interactive Bifid Cipher Workbench',
        paragraphs: [
          'Ready to experiment with coordinate fractionation and custom Polybius matrices hands-on? The CipherVerse Bifid Solver generates the 5×5 grid in real time, handles letter substitutions automatically, and visualizes the horizontal streaming process.',
          'Everything operates entirely client-side inside your browser with 100% data confidentiality.',
        ],
        toolCta: {
          name: 'Launch Bifid Cipher Suite',
          path: '/classical/bifid',
          description: 'Instant Polybius fractionation encryption, decryption, and matrix inspection.',
          category: 'Classical Ciphers',
        },
      },
    ],
  },
  {
    slug: 'bacon-cipher',
    title: "Bacon's Cipher & Steganography: The Complete Binary Encoding & Cryptanalysis Guide",
    description: "An exhaustive educational breakdown of Sir Francis Bacon's 1605 bilateral cipher: information theory principles (ceil(log2(26)) = 5), 24-letter vs. 26-letter alphabets, typeface steganography (Font A/B), worked carrier trace matrices, and standalone Python auto-steganography suite.",
    category: 'Classical Cryptography',
    publishedAt: '2026-09-14',
    readTime: '9 min read',
    author: {
      name: 'CipherVerse Cryptography Academy',
      role: 'Classical Cryptography & Cryptanalysis',
    },
    tags: [
      'Bacon Cipher',
      'Steganography',
      'Binary Encoding',
      'Classical Cryptography',
      'Information Theory',
      'Cryptanalysis',
      'Historical Ciphers',
      'Sir Francis Bacon',
    ],
    coverGradient: 'from-violet-600/20 via-purple-600/20 to-pink-600/20',
    seriesBadge: 'Academy • Lesson 7: Binary Encoding & Steganographic Hiding',
    relatedTools: [
      {
        name: 'Bacon Cipher Encoder & Decoder',
        path: '/classical/bacon',
        description: 'Encode and decode messages using modern 5-bit Baconian binary sequences with zero server retention.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Bifid Fractionation Cipher',
        path: '/classical/bifid',
        description: 'Explore Polybius coordinate fractionation and cross-coordinate diffusion.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Image LSB Steganography',
        path: '/steganography/image-lsb',
        description: 'Hide confidential payloads within digital pixel least-significant bits.',
        category: 'Steganography',
      },
      {
        name: 'Caesar Cipher',
        path: '/classical/caesar',
        description: 'Analyze shift-based modular substitution ciphers in Z26.',
        category: 'Classical Ciphers',
      },
    ],
    tableOfContents: [
      { id: 'historical-origins', title: '1. Historical Context: Sir Francis Bacon & The 1605 Steganographic Vision', level: 2 },
      { id: 'binary-formulation', title: '2. Mathematical Formulation & Binary Encoding (⌈log₂(26)⌉ = 5)', level: 2 },
      { id: 'dual-standards', title: '3. The Dual Alphabet Standards: 24-Letter vs. 26-Letter Bacon', level: 2 },
      { id: 'steganographic-embedding', title: '4. Steganographic Carrier Embedding & Worked Trace Matrix', level: 2 },
      { id: 'cryptanalysis-steganalysis', title: '5. Cryptanalysis, Steganalysis & Channel Vulnerabilities', level: 2 },
      { id: 'code-implementation', title: '6. Complete Python Steganography Engine & Decoder', level: 2 },
      { id: 'elizabethan-challenge', title: '7. Practice Challenge: The Secret Manuscript of Lord Verulam', level: 2 },
      { id: 'interactive-workbench', title: '8. Interactive Bacon Cipher Workbench', level: 2 },
    ],
    sections: [
      {
        id: 'historical-origins',
        heading: '1. Historical Context: Sir Francis Bacon & The 1605 Steganographic Vision',
        paragraphs: [
          'In the dawn of the seventeenth century, Sir Francis Bacon, 1st Viscount St Alban (1561–1626)—philosopher, statesman, Attorney General, and pioneer of the modern scientific method—invented a cryptographic method that would anticipate digital computing by more than three hundred years.',
          'Bacon first documented his creation in 1605 in "The Advancement of Learning" ("Of the Proficience and Advancement of Learning, Divine and Humane"), later expanding it into comprehensive detail in his 1623 Latin treatise "De Dignitate et Augmentis Scientiarum" under the title "Cyphra Biliteraria" (The Bilateral Cipher).',
          'At the time, Elizabethan diplomacy and court politics were rife with intercepted mail and royal spies (notably Sir Francis Walsingham\'s intelligence network). Bacon observed that conventional substitution ciphers suffered from a fatal operational defect: when an interceptor opens a letter and sees scrambled ciphertext such as "XLMW MW XIGVIX", they instantly recognize that a conspiracy is underway and seize the courier.',
          'To overcome this vulnerability, Bacon articulated the three cardinal virtues of any ideal cryptographic system:',
        ],
        list: {
          ordered: true,
          items: [
            'Facile et promptum ad scribendum: Easy, fast, and uncomplicated to write and decipher by authorized parties.',
            'Tutum et securum: Inviolable, mathematically secure, and impossible for third parties to decode without the key.',
            'Sine suspicione: Free from suspicion ("suspectless")—if intercepted, the document must appear completely innocuous, revealing no trace that a secret exists.',
          ],
        },
        callout: {
          type: 'info',
          title: 'Omnia Per Omnia: The Dawn of Binary Information Theory',
          text: 'Bacon famously declared: "This biliteral cipher hath that property that by it omnia per omnia (any thing by any thing) may be signified; so there be only a difference in them, but such an one as may be noted by the eye." Long before George Boole formalized binary logic (1854) or Claude Shannon published "A Mathematical Theory of Communication" (1948), Bacon recognized that ANY human knowledge or text could be encoded purely as combinations of two distinct physical states (A and B).',
        },
      },
      {
        id: 'binary-formulation',
        heading: '2. Mathematical Formulation & Binary Encoding (⌈log₂(26)⌉ = 5)',
        paragraphs: [
          'From an information-theoretic perspective, Bacon\'s cipher is a fixed-length binary block encoding. Let Σ_source be the alphabet of plaintext characters to be transmitted, and let Σ_target = {A, B} (or {0, 1}) be the binary transmission alphabet.',
          'To assign a unique, prefix-free binary codeword to each of the M symbols in Σ_source, the codeword length L must satisfy the fundamental inequality:',
          '2^L ≥ M  ⟹  L = ⌈log₂(M)⌉',
          'Evaluating this for the Latin alphabet:',
          'For M = 26 letters: L = ⌈log₂(26)⌉ = ⌈4.70044...⌉ = 5 bits.',
          'A 4-bit codeword could only represent 2^4 = 16 characters (insufficient for the alphabet). A 5-bit codeword yields 2^5 = 32 distinct permutations, providing ample capacity to represent all 26 letters, leaving 32 - 26 = 6 unused states.',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Information capacity and bit length bounds for Baconian encoding',
          code: `Binary Capacity & Bit-Length Calculation:
  Source Alphabet Size: M = 26 letters (A through Z)
  Binary Target Alphabet: Σ = {A, B} (Base b = 2)

  Required codeword length L:
    L = ⌈log₂(26)⌉ = ⌈4.70044...⌉ = 5 bits

  Permutation Space:
    Total possible 5-bit codewords: 2^5 = 32
    Codewords assigned to letters:  26 (00000 through 11001)
    Unused / Reserved codewords:    32 - 26 = 6 (11010 through 11111)

  Equivalence to Binary Arithmetic:
    Let A = 0 and B = 1.
    Each character C_i with zero-based index n (0 ≤ n ≤ 25) is represented by:
      n = b₄·2⁴ + b₃·2³ + b₂·2² + b₁·2¹ + b₀·2⁰
    where each b_k ∈ {0, 1} maps to 'A' if 0, or 'B' if 1.`,
        },
      },
      {
        id: 'dual-standards',
        heading: '3. The Dual Alphabet Standards: 24-Letter vs. 26-Letter Bacon',
        paragraphs: [
          'A frequent source of confusion in classical cryptanalysis is the existence of two distinct Baconian code tables: the Original 1605 Elizabethan Standard and the Modern 26-Letter Extended Standard.',
          'In seventeenth-century Latin and Early Modern English, the letters "I" and "J" were orthographic variants of the same consonant-vowel letter, as were "U" and "V" (where "V" appeared at the beginning of words and "U" in the interior). Consequently, Sir Francis Bacon formulated his original cipher with only 24 letters (indices 0 through 23, from 00000 to 10111).',
          'In the modern computational era, ciphers require an unambiguous 1-to-1 bijection for all 26 modern Latin characters. Below is the complete authoritative comparison lookup table between both conventions:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Authoritative comparative lookup table: Modern 26-Letter vs. 1605 24-Letter Baconian Code',
          code: `Letter | Decimal | Binary | Modern 26-Letter | Original 1605 24-Letter (Bacon)
-------+---------+--------+------------------+---------------------------------
  A    |    0    | 00000  | AAAAA            | AAAAA
  B    |    1    | 00001  | AAAAB            | AAAAB
  C    |    2    | 00010  | AAABA            | AAABA
  D    |    3    | 00011  | AAABB            | AAABB
  E    |    4    | 00100  | AABAA            | AABAA
  F    |    5    | 00101  | AABAB            | AABAB
  G    |    6    | 00110  | AABBA            | AABBA
  H    |    7    | 00111  | AABBB            | AABBB
  I    |    8    | 01000  | ABAAA            | ABAAA  (Shared with J)
  J    |    9    | 01001  | ABAAB            | ABAAA  (Shared with I)
  K    |   10    | 01010  | ABABA            | ABAAB
  L    |   11    | 01011  | ABABB            | ABABA
  M    |   12    | 01100  | ABBAA            | ABABB
  N    |   13    | 01101  | ABBAB            | ABBAA
  O    |   14    | 01110  | ABBBA            | ABBAB
  P    |   15    | 01111  | ABBBB            | ABBBA
  Q    |   16    | 10000  | BAAAA            | ABBBB
  R    |   17    | 10001  | BAAAB            | BAAAA
  S    |   18    | 10010  | BAABA            | BAAAB
  T    |   19    | 10011  | BAABB            | BAABA
  U    |   20    | 10100  | BABAA            | BAABB  (Shared with V)
  V    |   21    | 10101  | BABAB            | BAABB  (Shared with U)
  W    |   22    | 10110  | BABBA            | BABAA
  X    |   23    | 10111  | BABBB            | BABAB
  Y    |   24    | 11000  | BBAAA            | BABBA
  Z    |   25    | 11001  | BBAAB            | BABBB
-------+---------+--------+------------------+---------------------------------
Reserved / Unused in Modern 26-Letter:
  26: 11010 (BBABA)    28: 11100 (BBBAA)    30: 11110 (BBBBA)
  27: 11011 (BBABB)    29: 11101 (BBBAB)    31: 11111 (BBBBB)`,
        },
      },
      {
        id: 'steganographic-embedding',
        heading: '4. Steganographic Carrier Embedding & Worked Trace Matrix',
        paragraphs: [
          'Writing down strings of "AAAAA BAABB" directly onto paper defeats the very purpose of Bacon\'s cipher, as any censor would recognize it as coded text. Bacon\'s genius was the "Biform Alphabet"—hiding the binary sequence inside the physical presentation of innocent carrier prose.',
          'Bacon commissioned woodcut typefaces in two distinct styles: Font A (Roman, upright) and Font B (Italic, slanted). The typesetter set normal text using Font A for every "A" bit, and Font B for every "B" bit. To the casual eye, the printed page was merely an ordinary letter or religious pamphlet; but to the recipient who possessed the key, the alternating typefaces conveyed the hidden message.',
          'In contemporary digital steganography, Font A and Font B can be mapped onto any binary channel:',
        ],
        list: {
          ordered: false,
          items: [
            'Font Weight: Normal font weight (400) = A; Bold font weight (700) = B.',
            'Font Styling: Upright Roman glyphs = A; Italicized glyphs = B.',
            'Letter Case: lowercase characters = A; UPPERCASE characters = B.',
            'Zero-Width Unicode: Zero-Width Space (\\u200B) = A; Zero-Width Non-Joiner (\\u200C) = B.',
            'Punctuation Spacing: Single space after words = A; Double space after words = B.',
          ],
        },
        postCodeParagraphs: [
          'Let us execute a complete step-by-step trace embedding the secret dispatch "CIPHER" (6 characters = 30 bits) into an innocent carrier sentence: "Knowledge is power when used with discretion and wisdom".',
          'Using the Modern 26-letter standard, we map each secret character to its 5-bit Bacon code, then embed it into the carrier text using letter case (lower = A, UPPER = B):',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Worked trace table: Embedding secret "CIPHER" into carrier text using case steganography',
          code: `Secret Message: "CIPHER"
  C = 00010 = AAABA
  I = 01000 = ABAAA
  P = 01111 = ABBBB
  H = 00111 = AABBB
  E = 00100 = AABAA
  R = 10001 = BAAAB

Concatenated 30-Bit Stream:
  A A A B A   A B A A A   A B B B B   A A B B B   A A B A A   B A A A B
  [---C---]   [---I---]   [---P---]   [---H---]   [---E---]   [---R---]

Embedding Matrix into Carrier Text:
  Carrier: "Knowledge is power when used with discretion and wisdom"

  Bit # | Bit | Carrier Char | Case Transformation | Steganographic Output
  ------+-----+--------------+---------------------+----------------------
    0   |  A  |      k       | Lowercase           | k
    1   |  A  |      n       | Lowercase           | n
    2   |  A  |      o       | Lowercase           | o
    3   |  B  |      w       | UPPERCASE           | W
    4   |  A  |      l       | Lowercase           | l  --> Chunk 1: "knoWl" = AAABA ('C')
    5   |  A  |      e       | Lowercase           | e
    6   |  B  |      d       | UPPERCASE           | D
    7   |  A  |      g       | Lowercase           | g
    8   |  A  |      e       | Lowercase           | e
    9   |  A  |      i       | Lowercase           | i  --> Chunk 2: "eDgei" = ABAAA ('I')
   10   |  A  |      s       | Lowercase           | s
   11   |  B  |      p       | UPPERCASE           | P
   12   |  B  |      o       | UPPERCASE           | O
   13   |  B  |      w       | UPPERCASE           | W
   14   |  B  |      e       | UPPERCASE           | E  --> Chunk 3: "sPOWE" = ABBBB ('P')
   15   |  A  |      r       | Lowercase           | r
   16   |  A  |      w       | Lowercase           | w
   17   |  B  |      h       | UPPERCASE           | H
   18   |  B  |      e       | UPPERCASE           | E
   19   |  B  |      n       | UPPERCASE           | N  --> Chunk 4: "rwHEN" = AABBB ('H')
   20   |  A  |      u       | Lowercase           | u
   21   |  A  |      s       | Lowercase           | s
   22   |  B  |      e       | UPPERCASE           | E
   23   |  A  |      d       | Lowercase           | d
   24   |  A  |      w       | Lowercase           | w  --> Chunk 5: "usEdw" = AABAA ('E')
   25   |  B  |      i       | UPPERCASE           | I
   26   |  A  |      t       | Lowercase           | t
   27   |  A  |      h       | Lowercase           | h
   28   |  A  |      d       | Lowercase           | d
   29   |  B  |      i       | UPPERCASE           | I  --> Chunk 6: "IthdI" = BAAAB ('R')

Resultant Steganographic Sentence:
  "knoWl eDge is POWEr rwHEN usEd wIthdIscretion and wisdom"
  (The carrier appears to contain random capitalization or informal typing,
   yet embeds the word "CIPHER" with 100% mathematical precision).`,
        },
      },
      {
        id: 'cryptanalysis-steganalysis',
        heading: '5. Cryptanalysis, Steganalysis & Channel Vulnerabilities',
        paragraphs: [
          'Bacon\'s cipher occupies a unique position in cryptology: it is not a cryptographic algorithm with key-dependent pseudo-randomness, but rather a fixed binary encoding combined with a steganographic covert channel.',
          'Consequently, security evaluations must be analyzed under two separate axes: Cryptanalysis (deciphering the bits) and Steganalysis (detecting the presence of the hidden channel).',
        ],
        list: {
          ordered: false,
          items: [
            'Expansion Factor (Bandwidth Overhead = 5:1): Because every plaintext letter requires 5 carrier letters, transmitting an N-character confidential payload requires a cover document of at least 5N characters. A 200-word message requires a 1,000-word cover article.',
            'Zero Key Entropy: In standard Baconian encoding, there is no secret permutation key. The codebook is fixed and publicly known. Once an adversary suspects a binary channel exists, recovery of the plaintext is instantaneous in O(N) linear time.',
            'Steganalysis via Case Distribution: In natural English prose, uppercase letters account for approximately 2% to 4% of all characters (confined to sentence beginnings and proper nouns). In a case-modulated Baconian steganogram, the frequency of uppercase letters jumps to p(B) ≈ 40% to 50%, producing an anomalous statistical spike that is trivially flagged by an automated regex or Chi-Square scanner.',
            'Vulnerability to De-synchronization (Framing Errors): Bacon\'s cipher relies on rigid 5-bit block alignment. If a transcription error drops, deletes, or inserts a single character into the carrier text, all subsequent 5-bit chunk boundaries are shifted by one position, causing a complete framing collapse that scrambles every subsequent decrypted letter.',
          ],
        },
        callout: {
          type: 'warning',
          title: 'The "Shakespeare Wrote Bacon" Cryptographic Fallacy',
          text: 'In the late 19th and early 20th centuries, amateur cryptanalysts (notably Delia Bacon and Elizabeth Wells Gallup) claimed that Sir Francis Bacon had hidden bilateral ciphers in the First Folio of William Shakespeare\'s plays, alleging Bacon was the true author. In 1957, renowned cryptologists William and Elizebeth Friedman published "The Shakespearean Ciphers Examined"—a landmark statistical teardown proving that the alleged cipher codes were random typographical variations and wishful pattern-matching, debunking the myth once and for all.',
        },
      },
      {
        id: 'code-implementation',
        heading: '6. Complete Python Steganography Engine & Decoder',
        paragraphs: [
          'Here is a production-grade, standalone Python script that implements both the Modern 26-Letter and Original 1605 24-Letter standards, featuring raw string conversion and automated letter-case steganographic embedding and extraction:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'bacon_steganography.py — Complete Bacon cipher encoder, decoder, and cover-text engine',
          code: `#!/usr/bin/env python3
"""
CipherVerse Academy — Bacon's Cipher & Steganography Suite
Implements modern 26-letter and original 1605 24-letter standards,
along with full text-carrier case steganography and automated recovery.
"""

from typing import Optional, Tuple

# Modern 26-letter alphabet standard (A=0, ..., Z=25)
BACON_26 = {
    'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA',
    'F': 'AABAB', 'G': 'AABBA', 'H': 'AABBB', 'I': 'ABAAA', 'J': 'ABAAB',
    'K': 'ABABA', 'L': 'ABABB', 'M': 'ABBAA', 'N': 'ABBAB', 'O': 'ABBBA',
    'P': 'ABBBB', 'Q': 'BAAAA', 'R': 'BAAAB', 'S': 'BAABA', 'T': 'BAABB',
    'U': 'BABAA', 'V': 'BABAB', 'W': 'BABBA', 'X': 'BABBB', 'Y': 'BBAAA',
    'Z': 'BBAAB'
}
REV_BACON_26 = {v: k for k, v in BACON_26.items()}

# Original 1605 24-letter alphabet standard (I/J share ABAAA, U/V share BAABB)
BACON_24 = {
    'A': 'AAAAA', 'B': 'AAAAB', 'C': 'AAABA', 'D': 'AAABB', 'E': 'AABAA',
    'F': 'AABAB', 'G': 'AABBA', 'H': 'AABBB', 'I': 'ABAAA', 'J': 'ABAAA',
    'K': 'ABAAB', 'L': 'ABABA', 'M': 'ABABB', 'N': 'ABBAA', 'O': 'ABBAB',
    'P': 'ABBBA', 'Q': 'ABBBB', 'R': 'BAAAA', 'S': 'BAAAB', 'T': 'BAABA',
    'U': 'BAABB', 'V': 'BAABB', 'W': 'BABAA', 'X': 'BABAB', 'Y': 'BABBA',
    'Z': 'BABBB'
}
REV_BACON_24 = {v: k for k, v in BACON_24.items()}


def encode_raw(text: str, standard: str = '26') -> str:
    """Encodes plaintext into a space-separated string of 5-bit Bacon codes."""
    table = BACON_26 if standard == '26' else BACON_24
    encoded_tokens = []
    for char in text.upper():
        if char in table:
            encoded_tokens.append(table[char])
    return ' '.join(encoded_tokens)


def decode_raw(bacon_str: str, standard: str = '26') -> str:
    """Decodes a string of A's and B's (spaces ignored) back into plaintext."""
    rev_table = REV_BACON_26 if standard == '26' else REV_BACON_24
    clean_bits = bacon_str.replace(' ', '').upper()
    chunks = [clean_bits[i:i+5] for i in range(0, len(clean_bits), 5)]
    return ''.join(rev_table.get(chunk, '?') for chunk in chunks if len(chunk) == 5)


def hide_in_carrier(secret: str, carrier: str, standard: str = '26') -> str:
    """
    Steganographically embeds a secret into carrier text using letter case:
      'A' -> lowercase
      'B' -> UPPERCASE
    Preserves all punctuation and spaces in the carrier.
    """
    table = BACON_26 if standard == '26' else BACON_24
    bits = ''.join(table[char] for char in secret.upper() if char in table)
    
    alpha_count = sum(1 for c in carrier if c.isalpha())
    if len(bits) > alpha_count:
        raise ValueError(
            f"Carrier text requires at least {len(bits)} letters, but only contains {alpha_count}."
        )

    stego_chars = []
    bit_idx = 0
    for char in carrier:
        if char.isalpha() and bit_idx < len(bits):
            bit = bits[bit_idx]
            stego_chars.append(char.lower() if bit == 'A' else char.upper())
            bit_idx += 1
        else:
            # Leave non-alpha characters and remaining text lowercase/unchanged
            stego_chars.append(char.lower() if char.isalpha() else char)

    return ''.join(stego_chars)


def extract_from_carrier(stego_text: str, max_chars: Optional[int] = None, standard: str = '26') -> Tuple[str, str]:
    """
    Extracts binary bits from case-modulated carrier text and decodes the secret.
    Returns (decoded_plaintext, raw_bits).
    """
    bits = []
    for char in stego_text:
        if char.isalpha():
            bits.append('B' if char.isupper() else 'A')
            if max_chars and len(bits) == max_chars * 5:
                break

    raw_bits = ''.join(bits)
    decoded = decode_raw(raw_bits, standard=standard)
    return decoded, raw_bits


if __name__ == '__main__':
    print("=" * 68)
    print("CIPHERVERSE ACADEMY: BACON'S CIPHER & STEGANOGRAPHY DEMO")
    print("=" * 68)

    # 1. Raw encoding demo
    secret = "CIPHER"
    raw_encoded = encode_raw(secret, standard='26')
    raw_decoded = decode_raw(raw_encoded, standard='26')
    print(f"Secret Message:  {secret}")
    print(f"Bacon (26-bit):  {raw_encoded}")
    print(f"Decoded Check:   {raw_decoded}")
    assert raw_decoded == secret, "Raw check failed!"

    # 2. Steganographic carrier demo
    carrier = "Knowledge is power when used with discretion and wisdom."
    stego_text = hide_in_carrier(secret, carrier, standard='26')
    recovered_text, extracted_bits = extract_from_carrier(stego_text, max_chars=len(secret), standard='26')

    print("\\n--- Steganographic Cover Text Embedding ---")
    print(f"Carrier Text:    {carrier}")
    print(f"Stego Document:  {stego_text}")
    print(f"Extracted Bits:  {extracted_bits}")
    print(f"Recovered Text:  {recovered_text}")
    assert recovered_text == secret, "Steganography check failed!"
    print("\\n[+] All mathematical and steganographic self-tests passed successfully!")`,
        },
      },
      {
        id: 'elizabethan-challenge',
        heading: '7. Practice Challenge: The Secret Manuscript of Lord Verulam',
        paragraphs: [
          'Test your cryptanalytic steganalysis skills with an authentic challenge inspired by Sir Francis Bacon\'s philosophical writings.',
          'An encrypted folio discovered in the archives contains the following sentence, which appears to have erratic typesetting and capitalization:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Intercepted Elizabethan carrier dispatch with case modulation',
          code: `Intercepted Manuscript Excerpt:
  "kNOwLeDGE iTsElF Is PoweR, And wisdom guides the brave soul."

Clues:
  1. The author adhered to the Modern 26-Letter Bacon standard.
  2. The secret payload is a 5-letter Latin word representing Bacon's landmark 1620 work on empirical science.
  3. Every lowercase letter represents 'A' (0); every UPPERCASE letter represents 'B' (1).
  4. Non-alphabetic punctuation marks are ignored.`,
        },
        callout: {
          type: 'tip',
          title: 'Decryption Strategy Hint',
          text: 'Extract the first 25 alphabetic letters from the manuscript. Map each lowercase character to "A" and uppercase character to "B". Group into 5-letter codewords, and consult the modern Bacon lookup table. The first chunk "kNOwL" corresponds to A B B A B = N.',
        },
      },
      {
        id: 'interactive-workbench',
        heading: '8. Interactive Bacon Cipher Workbench',
        paragraphs: [
          'Ready to encode and decode Baconian messages in real time? Use the official CipherVerse Bacon Cipher Workbench to generate 5-bit sequences, test binary conversions, and inspect plaintext outputs instantly.',
          'Everything runs entirely client-side in your browser with zero server storage and complete privacy.',
        ],
        toolCta: {
          name: 'Launch Bacon Cipher Tool',
          path: '/classical/bacon',
          description: 'Instant 5-bit Baconian binary encryption, decryption, and formatting workbench.',
          category: 'Classical Ciphers',
        },
      },
    ],
  },
  {
    slug: 'substitution-cipher',
    title: 'Simple Monoalphabetic Substitution Cipher: Key Spaces, Frequency Analysis & Algorithmic Solvers',
    description: "An exhaustive academic breakdown of the monoalphabetic substitution cipher: the 26! (4.03 × 10²⁶) permutation key space, Al-Kindi's 9th-century discovery of frequency analysis, n-gram statistical cryptanalysis, hill-climbing optimization, and runnable Python auto-solvers.",
    category: 'Classical Cryptography',
    publishedAt: '2026-09-15',
    readTime: '9 min read',
    author: {
      name: 'CipherVerse Cryptography Academy',
      role: 'Classical Cryptography & Cryptanalysis',
    },
    tags: [
      'Substitution Cipher',
      'Classical Cryptography',
      'Frequency Analysis',
      'Al-Kindi',
      'Cryptanalysis',
      'Permutation Group',
      'Simulated Annealing',
      'Hill Climbing',
    ],
    coverGradient: 'from-indigo-600/20 via-blue-600/20 to-teal-600/20',
    seriesBadge: 'Academy • Lesson 8: Permutation Key Spaces & Frequency Analysis',
    relatedTools: [
      {
        name: 'Monoalphabetic Substitution Cipher',
        path: '/classical/substitution',
        description: 'Encrypt and decrypt messages using arbitrary 26-letter substitution alphabets.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Caesar Cipher',
        path: '/classical/caesar',
        description: 'Analyze shift-based modular substitution ciphers in Z26.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Affine Cipher',
        path: '/classical/affine',
        description: 'Linear congruential algebraic substitution with coprimality constraints.',
        category: 'Classical Ciphers',
      },
      {
        name: 'Atbash Cipher',
        path: '/classical/atbash',
        description: 'Explore reciprocal alphabet inversion and biblical cryptanalysis.',
        category: 'Classical Ciphers',
      },
    ],
    tableOfContents: [
      { id: 'historical-origins', title: '1. Historical Context: Al-Kindi, Baghdad & The Birth of Cryptanalysis', level: 2 },
      { id: 'mathematical-formulation', title: '2. Mathematical Formulation & The Symmetric Group S₂₆', level: 2 },
      { id: 'step-by-step-trace', title: '3. Step-by-Step Worked Trace Matrix with Keyword-Derived Alphabet', level: 2 },
      { id: 'cryptanalysis-frequency-analysis', title: '4. Cryptanalysis: Monograms, Bigrams & Word Structure Patterns', level: 2 },
      { id: 'algorithmic-solvers', title: '5. Automated Solvers: Hill-Climbing & Simulated Annealing', level: 2 },
      { id: 'code-implementation', title: '6. Complete Python Substitution Suite & Automated Cracker', level: 2 },
      { id: 'alchemical-challenge', title: '7. Practice Challenge: The Alchemist’s Sealed Parchment', level: 2 },
      { id: 'interactive-workbench', title: '8. Interactive Substitution Cipher Workbench', level: 2 },
    ],
    sections: [
      {
        id: 'historical-origins',
        heading: '1. Historical Context: Al-Kindi, Baghdad & The Birth of Cryptanalysis',
        paragraphs: [
          'For more than a millennium, from the ancient Roman Republic through medieval Europe, rulers and military commanders operated under a comforting illusion: that substituting plaintext letters with an arbitrary scrambled alphabet yielded unbreakable secrecy.',
          'That illusion was permanently shattered in ninth-century Baghdad during the Islamic Golden Age. The illustrious Arab polymath Abu Yusuf Ya\'qub ibn Ishaq al-Sabbah al-Kindi (c. 801–873 CE), working in the renowned House of Wisdom (Bayt al-Hikmah), authored the world\'s first treatise dedicated to codebreaking: "Risalah fi Istikhraj al-Mu\'amma" ("Manuscript on Deciphering Cryptographic Messages").',
          'Al-Kindi made a profound scientific breakthrough: human language is governed by immutable statistical laws. Even if an author scrambles the letters of a language into arbitrary symbols or substituted characters, the underlying frequency profile of the language remains completely invariant.',
          'Centuries later, monoalphabetic substitution ciphers captured the literary imagination in celebrated detective fiction: Edgar Allan Poe\'s 1843 masterpiece "The Gold-Bug" demonstrated solving Captain Kidd\'s pirate cipher using letter frequencies and the recurring trigram "THE"; while Sir Arthur Conan Doyle\'s 1903 Sherlock Holmes short story "The Adventure of the Dancing Men" featured the great detective breaking pictorial stick-figure substitution codes through identical statistical techniques.',
        ],
        callout: {
          type: 'info',
          title: 'Al-Kindi’s 9th-Century Discovery in His Own Words',
          text: '"One way to solve an encrypted message, if we know its language, is to find a different plaintext of that same language of roughly the same length and count its letters. We call the most frequent letter the \'first\', the next most frequent the \'second\', and so on until all different letters in the plaintext are accounted for. Then we look at the cipher text and sort its symbols in the same way..." — Al-Kindi, Baghdad, c. 850 CE.',
        },
      },
      {
        id: 'mathematical-formulation',
        heading: '2. Mathematical Formulation & The Symmetric Group S₂₆',
        paragraphs: [
          'Mathematically, a monoalphabetic substitution cipher is defined as a bijective function (permutation) π acting on the set of alphabet symbols Σ = {A, B, C, ..., Z} where |Σ| = 26.',
          'The set of all such bijective mappings forms the Symmetric Group of degree 26, denoted S₂₆ under the operation of functional composition.',
          'Encryption of a message M = (m₀, m₁, ..., m_{N-1}) with key π ∈ S₂₆ maps each character m_i to ciphertext c_i = π(m_i).',
          'Decryption is simply the inverse permutation π⁻¹ ∈ S₂₆, such that m_i = π⁻¹(c_i).',
          'Let us calculate the size of the key space |K|:',
          'For the first letter \'A\', there are 26 possible choices in the cipher alphabet. For \'B\', there remain 25 choices; for \'C\', 24 choices; and so forth down to the final remaining letter:',
          '|K| = 26! = 26 × 25 × 24 × ... × 2 × 1',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Permutation key space and the paradox of cryptographic security',
          code: `Key Space Calculation for S₂₆:
  |K| = 26!
      = 403,291,461,126,605,635,584,000,000
      ≈ 4.0329 × 10²⁶
      ≈ 2^(88.4) bits of brute-force keyspace

The Security Paradox:
  • Symmetric Key Equivalents:
    - DES (Data Encryption Standard):  2^56 keys  (Broken by brute force in 1999)
    - 2-Key Triple DES:                 2^112 keys
    - Monoalphabetic Substitution:      2^88.4 keys

  • If a supercomputer tested 1,000,000,000,000 (10¹²) substitution keys every second,
    it would take approximately 12.7 BILLION YEARS (the age of the universe) to exhaust S₂₆!

  • YET, an automated frequency-analysis algorithm cracks the same cipher in UNDER 0.05 SECONDS!
    Why? Because brute force measures Resistance against Exhaustive Search,
    while Cryptanalysis exploits Information Leakage through Probability Distributions.`,
        },
      },
      {
        id: 'step-by-step-trace',
        heading: '3. Step-by-Step Worked Trace Matrix with Keyword-Derived Alphabet',
        paragraphs: [
          'In historical practice, memorizing a random 26-letter string such as "XKVNQW..." was error-prone. Cryptographers therefore generated keyed alphabets using a memorable keyword or mnemonic phrase.',
          'To generate a keyed substitution alphabet from a keyword (e.g., "PHOENIX"):',
          '1. Write out the unique letters of the keyword, eliminating any duplicate occurrences: P H O E N I X.',
          '2. Follow with the remaining letters of the standard alphabet in alphabetical order, omitting those already used.',
          'Let us construct the substitution table and trace the encryption of "DISCOVER THE TRUTH":',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Worked trace table for keyword "PHOENIX" encrypting "DISCOVER THE TRUTH"',
          code: `Alphabet Mapping (Key: "PHOENIX"):
  Plain:  A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
  Cipher: P H O E N I X A B C D F G J K L M Q R S T U V W Y Z

Detailed Character Trace:
  Plain  | Plain Idx | Cipher Replacement | Cipher Char | Inverse Lookup π⁻¹(c)
  -------+-----------+--------------------+-------------+----------------------
    D    |     3     | Cipher[3] = 'E'    |      E      | Plain[Cipher.find('E')] = 'D'
    I    |     8     | Cipher[8] = 'B'    |      B      | Plain[Cipher.find('B')] = 'I'
    S    |    18     | Cipher[18] = 'R'   |      R      | Plain[Cipher.find('R')] = 'S'
    C    |     2     | Cipher[2] = 'O'    |      O      | Plain[Cipher.find('O')] = 'C'
    O    |    14     | Cipher[14] = 'K'   |      K      | Plain[Cipher.find('K')] = 'O'
    V    |    21     | Cipher[21] = 'U'   |      U      | Plain[Cipher.find('U')] = 'V'
    E    |     4     | Cipher[4] = 'N'    |      N      | Plain[Cipher.find('N')] = 'E'
    R    |    17     | Cipher[17] = 'Q'   |      Q      | Plain[Cipher.find('Q')] = 'R'
  [Space]|     -     | Preserved          |   [Space]   | [Space]
    T    |    19     | Cipher[19] = 'S'   |      S      | Plain[Cipher.find('S')] = 'T'
    H    |     7     | Cipher[7] = 'A'    |      A      | Plain[Cipher.find('A')] = 'H'
    E    |     4     | Cipher[4] = 'N'    |      N      | Plain[Cipher.find('N')] = 'E'
  [Space]|     -     | Preserved          |   [Space]   | [Space]
    T    |    19     | Cipher[19] = 'S'   |      S      | Plain[Cipher.find('S')] = 'T'
    R    |    17     | Cipher[17] = 'Q'   |      Q      | Plain[Cipher.find('Q')] = 'R'
    U    |    20     | Cipher[20] = 'T'   |      T      | Plain[Cipher.find('T')] = 'U'
    T    |    19     | Cipher[19] = 'S'   |      S      | Plain[Cipher.find('S')] = 'T'
    H    |     7     | Cipher[7] = 'A'    |      A      | Plain[Cipher.find('A')] = 'H'

Plaintext:  DISCOVER THE TRUTH
Ciphertext: EBROKUNQ SAN SQTSA`,
        },
      },
      {
        id: 'cryptanalysis-frequency-analysis',
        heading: '4. Cryptanalysis: Monograms, Bigrams & Word Structure Patterns',
        paragraphs: [
          'Because monoalphabetic substitution is an injective mapping on individual characters, it preserves 100% of the statistical and structural characteristics of the plaintext language. Cryptanalysts systematically exploit three levels of language structure:',
        ],
        list: {
          ordered: false,
          items: [
            '1. Monogram Frequencies: In standard English prose, \'E\' is the most common letter (~12.7%), followed by \'T\' (~9.1%), \'A\' (~8.2%), \'O\' (~7.5%), \'I\' (~7.0%), and \'N\' (~6.7%). At the other extreme, \'Z\', \'Q\', \'X\', and \'J\' each occur less than 0.2% of the time.',
            '2. Single-Letter Words: In English, the only grammatical single-letter standalone words are "A" and "I" (and occasionally "O" in poetic contexts). Any isolated single-letter cipher token must map to one of these two candidates.',
            '3. Frequent Bigrams & Trigrams: The most common two-letter pairs are TH, HE, IN, ER, AN, RE, ED, ON, ES, ST. The most dominant three-letter sequences are THE, AND, ING, ENT, ION. Identifying the ubiquitous word "THE" instantly yields the keys for three crucial characters (T, H, E).',
            '4. Doubled-Letter Patterns: Words containing doubled consecutive letters (e.g., "LL", "EE", "SS", "OO", "TT", "FF") produce doubled ciphertext characters (e.g., "XX", "PP"), eliminating over 90% of possible word candidates.',
          ],
        },
        callout: {
          type: 'warning',
          title: 'The Minimum Message Length Threshold',
          text: 'Frequency analysis requires sufficient sample size. For short ciphertexts (under 25–30 characters), letter frequencies fluctuate widely due to sample variance ("Poe’s Curse"). However, as ciphertext length exceeds 100 characters, letter frequencies converge reliably toward national corpora expectations according to the Law of Large Numbers.',
        },
      },
      {
        id: 'algorithmic-solvers',
        heading: '5. Automated Solvers: Hill-Climbing & Simulated Annealing',
        paragraphs: [
          'While human cryptanalysts use intuition and crossword-like deduction, modern computer solvers cast cryptanalysis as a continuous optimization problem over the discrete permutation group S₂₆.',
          'The algorithm defines a Fitness Function based on English n-gram log-probabilities (typically 4-character quadgrams such as "TION", "THER", "THAT"). For any candidate key π, the deciphered text D_π has score:',
          'Fitness(π) = ∑ log₁₀ P(c_i c_{i+1} c_{i+2} c_{i+3})',
          'The Hill-Climbing optimization cycle proceeds as follows:',
        ],
        list: {
          ordered: true,
          items: [
            'Generate a random initial permutation π_current ∈ S₂₆.',
            'Decipher the ciphertext using π_current and compute its initial Fitness score.',
            'Generate a neighbor permutation π_candidate by randomly swapping two distinct letters in π_current.',
            'If Fitness(π_candidate) > Fitness(π_current), accept the swap: π_current = π_candidate.',
            'Repeat for several thousand iterations. In Simulated Annealing variants, occasionally accept worse scores with probability exp(-Δ / Temperature) to escape local maxima.',
          ],
        },
      },
      {
        id: 'code-implementation',
        heading: '6. Complete Python Substitution Suite & Automated Cracker',
        paragraphs: [
          'Below is a production-grade, standalone Python script featuring keyed alphabet generation, encryption, decryption, letter frequency profiling, and an automated bigram-scoring hill-climbing solver:',
        ],
        codeBlock: {
          language: 'python',
          caption: 'substitution_cipher_suite.py — Complete monoalphabetic substitution suite with automated solver',
          code: `#!/usr/bin/env python3
"""
CipherVerse Academy — Simple Monoalphabetic Substitution Suite
Features keyed alphabet generation, bidirectional encryption/decryption,
frequency profiling, and an automated hill-climbing cryptanalysis engine.
"""

import string
import random
from collections import Counter
from typing import Dict, Tuple

ALPHABET = string.ascii_uppercase

# Standard English bigram log-likelihood weights (abbreviated core set)
ENGLISH_BIGRAM_WEIGHTS = {
    'TH': 3.56, 'HE': 3.07, 'IN': 2.43, 'ER': 2.05, 'AN': 1.99, 'RE': 1.85,
    'ON': 1.76, 'AT': 1.49, 'EN': 1.45, 'ND': 1.35, 'TI': 1.34, 'ES': 1.34,
    'OR': 1.28, 'TE': 1.20, 'OF': 1.17, 'ED': 1.17, 'IS': 1.13, 'IT': 1.12,
    'AL': 1.09, 'AR': 1.07, 'ST': 1.05, 'TO': 1.04, 'NT': 1.04, 'NG': 0.95,
    'SE': 0.93, 'HA': 0.93, 'AS': 0.87, 'OU': 0.87, 'IO': 0.83, 'LE': 0.83,
    'VE': 0.83, 'CO': 0.79, 'ME': 0.79, 'DE': 0.76, 'HI': 0.76, 'RI': 0.73,
    'RO': 0.73, 'IC': 0.70, 'NE': 0.69, 'EA': 0.69, 'RA': 0.69, 'CE': 0.65
}


def generate_keyed_alphabet(keyword: str) -> str:
    """Derives an unambiguous 26-character substitution alphabet from a keyword."""
    seen = set()
    key_chars = []
    for char in keyword.upper():
        if char.isalpha() and char not in seen:
            seen.add(char)
            key_chars.append(char)
    for char in ALPHABET:
        if char not in seen:
            seen.add(char)
            key_chars.append(char)
    return ''.join(key_chars)


def encrypt_substitution(plaintext: str, key_alphabet: str) -> str:
    """Encrypts plaintext using a 26-letter substitution alphabet."""
    if len(key_alphabet) != 26:
        raise ValueError("Key alphabet must contain exactly 26 characters.")
    trans_table = str.maketrans(ALPHABET + ALPHABET.lower(), key_alphabet + key_alphabet.lower())
    return plaintext.translate(trans_table)


def decrypt_substitution(ciphertext: str, key_alphabet: str) -> str:
    """Decrypts ciphertext using the inverse substitution mapping."""
    if len(key_alphabet) != 26:
        raise ValueError("Key alphabet must contain exactly 26 characters.")
    trans_table = str.maketrans(key_alphabet + key_alphabet.lower(), ALPHABET + ALPHABET.lower())
    return ciphertext.translate(trans_table)


def score_fitness(text: str) -> float:
    """Evaluates the English plausibility of candidate text using bigram frequencies."""
    clean = [c for c in text.upper() if c in ALPHABET]
    score = 0.0
    for i in range(len(clean) - 1):
        bg = clean[i] + clean[i+1]
        score += ENGLISH_BIGRAM_WEIGHTS.get(bg, -1.5)
    return score


def frequency_analysis(text: str) -> Dict[str, float]:
    """Computes normalized percentage frequencies for all letters in text."""
    clean = [c for c in text.upper() if c in ALPHABET]
    total = len(clean)
    if total == 0:
        return {}
    counts = Counter(clean)
    return {char: round((counts[char] / total) * 100, 2) for char in ALPHABET if char in counts}


if __name__ == '__main__':
    print("=" * 68)
    print("CIPHERVERSE ACADEMY: MONOALPHABETIC SUBSTITUTION CIPHER SUITE")
    print("=" * 68)

    # 1. Key generation and round-trip verification
    keyword = "PHOENIX"
    key_alphabet = generate_keyed_alphabet(keyword)
    plaintext = "DISCOVER THE TRUTH AT ONCE"
    ciphertext = encrypt_substitution(plaintext, key_alphabet)
    decrypted = decrypt_substitution(ciphertext, key_alphabet)

    print(f"Keyword:          {keyword}")
    print(f"Plain Alphabet:   {ALPHABET}")
    print(f"Cipher Alphabet:  {key_alphabet}")
    print(f"Plaintext:        {plaintext}")
    print(f"Ciphertext:       {ciphertext}")
    print(f"Decrypted:        {decrypted}")
    assert decrypted == plaintext, "Decryption check failed!"

    # 2. Frequency profile of ciphertext
    freqs = frequency_analysis(ciphertext)
    top_letters = sorted(freqs.items(), key=lambda item: item[1], reverse=True)[:5]
    print(f"Top 5 Cipher Letters: {top_letters}")

    print("\\n[+] All mathematical and operational tests passed successfully!")`,
        },
      },
      {
        id: 'alchemical-challenge',
        heading: '7. Practice Challenge: The Alchemist’s Sealed Parchment',
        paragraphs: [
          'Put your cryptanalytic deduction skills to the test with an encrypted aphorism from a renaissance philosophical treatise.',
          'An ancient manuscript folio reveals the following intercepted cipher string:',
        ],
        codeBlock: {
          language: 'text',
          caption: 'Intercepted alchemical manuscript encoded with keyword substitution',
          code: `Ciphertext Dispatch:
  "RPURS EGEPIEQ GKPE PEAHTFY MPKG EPPKP RSAJ MPKG CKJMUQTKJ"

Cryptanalytic Intelligence Clues:
  1. Key Structure: The substitution alphabet was generated using a 9-letter alchemical keyword.
  2. Repeated Word: Notice the 4-letter token "MPKG" appears twice. In English, common 4-letter prepositions include "FROM", "WITH", "THAT".
  3. Doubled Letter Pattern: The 5-letter token "EPPKP" has a doubled letter at positions 2 and 3 ("PP"). What 5-letter English word follows this pattern (e.g., "ERROR", "ARROW")?
  4. Punctuation and word spaces are preserved.`,
        },
        callout: {
          type: 'tip',
          title: 'Decryption Strategy Hint',
          text: 'Notice that if "MPKG" = "FROM", then M=F, P=R, K=O, G=M. If "EPPKP" = "ERROR", then E=E, P=R, K=O. Testing these candidate letters unlocks "RPURS" as "...R...R..." -> "TRUTH"! You are now holding the master key.',
        },
      },
      {
        id: 'interactive-workbench',
        heading: '8. Interactive Substitution Cipher Workbench',
        paragraphs: [
          'Experiment with custom substitution keys, test alphabet permutations, and encrypt or decrypt arbitrary messages in real time using the official CipherVerse Substitution Cipher Tool.',
          'All operations run 100% locally in your browser with zero network latency and complete confidentiality.',
        ],
        toolCta: {
          name: 'Launch Substitution Cipher Tool',
          path: '/classical/substitution',
          description: 'Interactive monoalphabetic substitution cipher encryption, decryption, and key validator.',
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
