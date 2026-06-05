export interface ToolDefinition {
  name: string;
  description: string;
  path: string;
  category: string;
  endpoint: string;
}

export const toolDefinitions: ToolDefinition[] = [
  // Classical Ciphers
  { name: 'Caesar Cipher', description: 'Shift-based substitution cipher', path: '/classical/caesar', category: 'Classical Ciphers', endpoint: '/classical/caesar' },
  { name: 'Vigenère Cipher', description: 'Polyalphabetic substitution cipher', path: '/classical/vigenere', category: 'Classical Ciphers', endpoint: '/classical/vigenere/encrypt' },
  { name: 'Atbash Cipher', description: 'Reverse alphabet substitution', path: '/classical/atbash', category: 'Classical Ciphers', endpoint: '/classical/atbash' },
  { name: 'Bacon Cipher', description: 'Binary steganographic cipher', path: '/classical/bacon', category: 'Classical Ciphers', endpoint: '/classical/bacon/encrypt' },
  { name: 'Bifid Cipher', description: 'Polybius square fractionation cipher', path: '/classical/bifid', category: 'Classical Ciphers', endpoint: '/classical/bifid/encrypt' },
  { name: 'Affine Cipher', description: 'Mathematical substitution cipher', path: '/classical/affine', category: 'Classical Ciphers', endpoint: '/classical/affine/encrypt' },
  { name: 'A1Z26 Cipher', description: 'Letter-to-number encoding', path: '/classical/a1z26', category: 'Classical Ciphers', endpoint: '/classical/a1z26/encrypt' },
  { name: 'Rail Fence Cipher', description: 'Zigzag transposition cipher', path: '/classical/rail-fence', category: 'Classical Ciphers', endpoint: '/classical/rail_fence/encrypt' },
  { name: 'Substitution Cipher', description: 'Custom alphabet substitution', path: '/classical/substitution', category: 'Classical Ciphers', endpoint: '/classical/substitute/encrypt' },

  // Encoding
  { name: 'Base64', description: 'Base64 encoding and decoding', path: '/encoding/base64', category: 'Encoding & Decoding', endpoint: '/encoding/base64/encode' },
  { name: 'Hex', description: 'Hexadecimal encoding and decoding', path: '/encoding/hex', category: 'Encoding & Decoding', endpoint: '/encoding/hex/encode' },
  { name: 'URL', description: 'URL encoding and decoding', path: '/encoding/url', category: 'Encoding & Decoding', endpoint: '/encoding/url/encode' },
  { name: 'Binary', description: 'Binary encoding and decoding', path: '/encoding/binary', category: 'Encoding & Decoding', endpoint: '/encoding/binary/encode' },
  { name: 'Morse Code', description: 'Morse code encoding and decoding', path: '/encoding/morse', category: 'Encoding & Decoding', endpoint: '/encoding/morse/encode' },

  // Symmetric (placeholders for search)
  { name: 'AES', description: 'Advanced Encryption Standard', path: '/symmetric', category: 'Symmetric Crypto', endpoint: '/symmetric/aes/encrypt' },
  { name: 'DES', description: 'Data Encryption Standard', path: '/symmetric', category: 'Symmetric Crypto', endpoint: '/symmetric/des/encrypt' },
  { name: 'Blowfish', description: 'Blowfish block cipher', path: '/symmetric', category: 'Symmetric Crypto', endpoint: '/symmetric/blowfish/encrypt' },
  { name: 'XOR', description: 'XOR cipher with bruteforce', path: '/symmetric', category: 'Symmetric Crypto', endpoint: '/symmetric/xor/encrypt' },

  // Asymmetric
  { name: 'RSA', description: 'RSA key generation, encrypt, decrypt, sign, verify', path: '/asymmetric', category: 'Asymmetric Crypto', endpoint: '/asymmetric/rsa/generate-keys' },
  { name: 'DSA', description: 'DSA key generation', path: '/asymmetric', category: 'Asymmetric Crypto', endpoint: '/asymmetric/dsa/generate-keys' },

  // Hashing
  { name: 'Hash Generator', description: 'Generate hashes with various algorithms', path: '/hashing', category: 'Hashing & KDFs', endpoint: '/hashing/hash' },
  { name: 'HMAC', description: 'Hash-based Message Authentication Code', path: '/hashing', category: 'Hashing & KDFs', endpoint: '/hashing/hmac' },
  { name: 'PBKDF2', description: 'Password-Based Key Derivation Function 2', path: '/hashing', category: 'Hashing & KDFs', endpoint: '/hashing/pbkdf2' },
  { name: 'Scrypt', description: 'Memory-hard key derivation function', path: '/hashing', category: 'Hashing & KDFs', endpoint: '/hashing/scrypt' },

  // Blockchain
  { name: 'Bitcoin Validator', description: 'Validate Bitcoin addresses', path: '/blockchain', category: 'Blockchain', endpoint: '/blockchain/bitcoin/validate' },
  { name: 'Ethereum Validator', description: 'Validate Ethereum addresses', path: '/blockchain', category: 'Blockchain', endpoint: '/blockchain/ethereum/validate' },
  { name: 'Merkle Tree', description: 'Build and visualize Merkle trees', path: '/blockchain', category: 'Blockchain', endpoint: '/blockchain/merkle' },

  // Utilities
  { name: 'Password Strength', description: 'Analyze password entropy and strength', path: '/utilities', category: 'Utilities', endpoint: '/utilities/password/strength' },
  { name: 'JWT Sign', description: 'Sign JSON Web Tokens', path: '/utilities', category: 'Utilities', endpoint: '/utilities/jwt/sign' },
  { name: 'Salt Generator', description: 'Generate cryptographic salts', path: '/utilities', category: 'Utilities', endpoint: '/utilities/salt/generate' },

  // Historical
  { name: 'Enigma Machine', description: 'WWII Enigma cipher simulator', path: '/historical', category: 'Historical Machines', endpoint: '/historic/enigma' },
  { name: 'Bombe', description: 'Turing Bombe crib analysis', path: '/historical', category: 'Historical Machines', endpoint: '/historic/bombe' },
  { name: 'Typex', description: 'British Typex cipher machine', path: '/historical', category: 'Historical Machines', endpoint: '/historic/typex' },
];
