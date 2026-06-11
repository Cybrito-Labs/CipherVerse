import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, X, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';

const TOOLS_DATA: Record<string, string[]> = {
  "Caesar Cipher": [
    "One of the oldest substitution ciphers.",
    "Each letter is shifted by a fixed number of positions in the alphabet.",
    "Encryption and decryption use the same shift value.",
    "Commonly used to teach cryptographic fundamentals.",
    "Easily broken through brute force because only 25 possible keys exist."
  ],
  "Vigenère Cipher": [
    "A polyalphabetic substitution cipher.",
    "Uses a keyword to generate multiple Caesar shifts.",
    "More secure than a simple Caesar Cipher.",
    "Encrypts identical letters differently depending on position.",
    "Historically considered difficult to crack before frequency-analysis improvements."
  ],
  "Atbash Cipher": [
    "A simple monoalphabetic substitution cipher.",
    "Alphabet is reversed (A→Z, B→Y, etc.).",
    "Uses no key.",
    "Encryption and decryption are identical operations.",
    "Mainly educational and historical."
  ],
  "Bacon Cipher": [
    "Encodes letters using patterns of A and B characters.",
    "Can hide messages inside text formatting.",
    "Considered an early steganographic technique.",
    "Converts each letter into a 5-bit representation.",
    "Useful for demonstrating hidden communication methods."
  ],
  "Bifid Cipher": [
    "Combines substitution and transposition techniques.",
    "Uses a Polybius square for letter coordinates.",
    "Developed by Félix Delastelle.",
    "Spreads plaintext statistics across ciphertext.",
    "More resistant than simple substitution ciphers."
  ],
  "Affine Cipher": [
    "Mathematical substitution cipher.",
    "Uses formula: E(x)=(ax+b) mod 26.",
    "Requires two key values.",
    "Demonstrates modular arithmetic in cryptography.",
    "Vulnerable to frequency analysis."
  ],
  "A1Z26 Cipher": [
    "Converts letters into numbers.",
    "A=1, B=2, ..., Z=26.",
    "Not encryption in a cryptographic sense.",
    "Commonly used in puzzles and CTF challenges.",
    "Useful for introducing character encoding concepts."
  ],
  "Rail Fence Cipher": [
    "Transposition cipher.",
    "Writes text diagonally across multiple rails.",
    "Ciphertext is formed by reading row-wise.",
    "Does not alter characters themselves.",
    "Popular educational example of transposition encryption."
  ],
  "Substitution Cipher": [
    "Replaces each letter with another alphabet character.",
    "Uses a custom alphabet mapping.",
    "Key space is significantly larger than Caesar Cipher.",
    "Vulnerable to frequency analysis.",
    "Foundation for understanding modern cryptanalysis."
  ],
  "AES": [
    "Advanced Encryption Standard.",
    "Current industry standard symmetric encryption algorithm.",
    "Supports AES-128, AES-192, and AES-256.",
    "Used in VPNs, HTTPS, disk encryption, and cloud security.",
    "Known for strong security and high performance."
  ],
  "DES": [
    "Data Encryption Standard.",
    "Uses a 56-bit key.",
    "Once a government encryption standard.",
    "Now considered insecure due to brute-force attacks.",
    "Important historically in cryptography education."
  ],
  "3DES": [
    "Triple DES performs DES three times.",
    "Created to extend DES lifespan.",
    "More secure than DES but slower.",
    "Being phased out in modern systems.",
    "Useful for legacy compatibility."
  ],
  "RC2": [
    "Symmetric block cipher by Ron Rivest.",
    "Variable key length support.",
    "Designed as a DES alternative.",
    "Rarely used in modern systems.",
    "Mostly encountered in legacy applications."
  ],
  "RC4": [
    "Stream cipher developed by Ron Rivest.",
    "Extremely fast and simple.",
    "Once widely used in SSL/TLS and Wi-Fi.",
    "Multiple vulnerabilities discovered.",
    "Deprecated in secure applications."
  ],
  "RC4 Drop": [
    "Modified RC4 implementation.",
    "Discards initial keystream bytes.",
    "Reduces early RC4 biases.",
    "Improves security over raw RC4.",
    "Still considered outdated today."
  ],
  "Blowfish": [
    "Symmetric block cipher designed by Bruce Schneier.",
    "Fast and free to use.",
    "Supports variable key lengths.",
    "Predecessor to Twofish.",
    "Still found in some legacy applications."
  ],
  "SM4": [
    "Chinese national encryption standard.",
    "Uses a 128-bit key.",
    "Similar role to AES in China.",
    "Common in Chinese government and enterprise systems.",
    "Provides strong modern encryption."
  ],
  "CipherSaber2": [
    "RC4-based encryption scheme.",
    "Adds key stretching and iteration.",
    "Improves security compared to raw RC4.",
    "Popular among hobbyist cryptography users.",
    "Educational rather than enterprise-focused."
  ],
  "XOR Cipher": [
    "Uses XOR operation between data and key.",
    "Extremely simple encryption method.",
    "Foundation of many stream ciphers.",
    "Weak if key is reused.",
    "Frequently used in malware obfuscation."
  ],
  "RSA": [
    "Public-key cryptography algorithm.",
    "Uses separate public and private keys.",
    "Supports encryption and digital signatures.",
    "Security relies on integer factorization difficulty.",
    "One of the most widely used asymmetric systems."
  ],
  "DSA": [
    "Digital Signature Algorithm.",
    "Designed specifically for signatures.",
    "Does not perform general encryption.",
    "Provides authentication and integrity.",
    "Commonly used in government and enterprise systems."
  ],
  "Hash Generator": [
    "Produces cryptographic hashes from input data.",
    "Supports algorithms like SHA-256.",
    "One-way operation.",
    "Used for integrity verification.",
    "Fundamental security utility."
  ],
  "HMAC": [
    "Hash-based Message Authentication Code.",
    "Combines secret key and hash function.",
    "Verifies integrity and authenticity.",
    "Widely used in APIs and authentication systems.",
    "Resistant to tampering."
  ],
  "PBKDF2": [
    "Password-Based Key Derivation Function 2.",
    "Strengthens passwords using repeated hashing.",
    "Protects against brute-force attacks.",
    "Industry standard password hashing method.",
    "Used in many authentication systems."
  ],
  "Scrypt": [
    "Memory-hard key derivation function.",
    "Designed to resist GPU attacks.",
    "Requires significant memory resources.",
    "Popular in password storage.",
    "Stronger than PBKDF2 against specialized hardware."
  ],
  "Bcrypt": [
    "Password hashing algorithm.",
    "Automatically includes salting.",
    "Adjustable work factor.",
    "Widely used in web applications.",
    "Trusted for secure password storage."
  ],
  "Base64": [
    "Converts binary data into ASCII text.",
    "Common in email and APIs.",
    "Not encryption.",
    "Easily reversible.",
    "Used for data transport and storage."
  ],
  "Hex": [
    "Represent binary data as hexadecimal characters.",
    "Human-readable format.",
    "Frequently used in cryptography.",
    "Useful for hashes and binary inspection.",
    "Not a security mechanism."
  ],
  "URL Encoding": [
    "Encodes special URL characters.",
    "Uses percent notation.",
    "Ensures safe data transmission.",
    "Required for web requests.",
    "Prevents URL parsing issues."
  ],
  "Binary": [
    "Converts data to base-2 representation.",
    "Shows underlying computer storage format.",
    "Useful for learning digital systems.",
    "Common in educational exercises.",
    "Not encryption."
  ],
  "Morse Code": [
    "Encodes text into dots and dashes.",
    "Historical communication method.",
    "Easily reversible.",
    "Used in radio communication.",
    "Educational and recreational."
  ],
  "X.509 Parser": [
    "Parses X.509 digital certificates used in PKI environments.",
    "Extracts issuer, subject, validity period, serial number, and extensions.",
    "Helps analyze certificate trust chains.",
    "Useful for debugging SSL/TLS deployment issues.",
    "Commonly used by security analysts and system administrators."
  ],
  "TLS Analyzer": [
    "Connects to remote servers and inspects TLS configurations.",
    "Retrieves certificate chains and protocol information.",
    "Detects weak cipher suites and outdated protocols.",
    "Helps identify SSL/TLS security misconfigurations.",
    "Valuable for website and infrastructure security audits."
  ],
  "Fingerprint Generator": [
    "Generates cryptographic fingerprints from certificates or raw data.",
    "Supports common hashing algorithms.",
    "Used for certificate verification and comparison.",
    "Helps identify unique digital assets.",
    "Frequently used in incident response and forensics."
  ],
  "Bitcoin Validation": [
    "Validates Bitcoin wallet addresses.",
    "Supports formats such as P2PKH, P2SH, and Bech32.",
    "Performs checksum verification.",
    "Detects malformed or invalid addresses.",
    "Useful for blockchain developers and investigators."
  ],
  "Ethereum Validation": [
    "Validates Ethereum addresses.",
    "Supports EIP-55 checksum verification.",
    "Detects formatting errors.",
    "Helps reduce transaction mistakes.",
    "Useful for smart contract and wallet development."
  ],
  "Merkle Tree": [
    "Builds cryptographic Merkle Trees from input data.",
    "Demonstrates blockchain data integrity concepts.",
    "Allows verification of individual records.",
    "Widely used in Bitcoin and other blockchains.",
    "Helps visualize hierarchical hash structures."
  ],
  "WIF Encoder": [
    "Converts private keys into Wallet Import Format.",
    "Makes Bitcoin private keys easier to store and transfer.",
    "Applies version bytes and checksums.",
    "Used when importing wallets into Bitcoin software.",
    "Demonstrates cryptocurrency key management concepts."
  ],
  "Text Steganography": [
    "Hides secret messages inside normal-looking text.",
    "Often uses invisible or zero-width characters.",
    "Enables covert communication.",
    "Difficult to detect visually.",
    "Useful for educational steganography demonstrations."
  ],
  "Image Steganography": [
    "Embeds hidden data inside images.",
    "Commonly uses Least Significant Bit (LSB) techniques.",
    "Allows secret communication through pictures.",
    "Demonstrates digital steganography principles.",
    "Useful for cybersecurity and forensic training."
  ],
  "Audio Steganography": [
    "Conceals information inside audio files.",
    "Modifies low-impact portions of sound samples.",
    "Maintains perceived audio quality.",
    "Demonstrates hidden communication channels.",
    "Useful for learning multimedia steganography."
  ],
  "File Hash": [
    "Calculates a cryptographic hash for a file.",
    "Generates fingerprints using algorithms such as SHA-256.",
    "Verifies file integrity.",
    "Detects modifications and tampering.",
    "Widely used in digital forensics investigations."
  ],
  "Multi Hash": [
    "Computes multiple hashes simultaneously.",
    "Supports MD5, SHA1, SHA256, and others.",
    "Speeds up forensic workflows.",
    "Helps compare files across different systems.",
    "Useful for malware and evidence analysis."
  ],
  "Shannon Entropy": [
    "Measures randomness within a file.",
    "Helps identify encrypted or compressed content.",
    "Detects packed malware samples.",
    "Produces numerical entropy values.",
    "Important in malware triage and forensic investigations."
  ],
  "Randomness Test": [
    "Performs statistical analysis on file contents.",
    "Evaluates randomness quality.",
    "Helps identify cryptographic material.",
    "Useful for security research.",
    "Supports forensic examination of suspicious files."
  ],
  "Hash Analysis": [
    "Analyzes malware hashes against threat intelligence sources.",
    "Assists in identifying known malicious files.",
    "Provides quick threat categorization.",
    "Useful during incident response.",
    "Helps determine whether a sample has been previously observed."
  ],
  "TLSH Compare": [
    "Uses Trend Micro Locality Sensitive Hashing.",
    "Compares files based on similarity rather than exact matches.",
    "Detects related malware variants.",
    "Useful for malware family classification.",
    "Helps identify modified or obfuscated samples."
  ],
  "PE Analysis": [
    "Analyzes Windows Portable Executable (PE) files.",
    "Extracts headers, sections, imports, and metadata.",
    "Calculates indicators such as Import Hashes (Imphash).",
    "Detects packing and suspicious characteristics.",
    "Essential for Windows malware investigations."
  ],
  "Password Strength": [
    "Evaluates password complexity and entropy.",
    "Identifies weak password patterns.",
    "Provides security recommendations.",
    "Helps users create stronger credentials.",
    "Useful for awareness and training purposes."
  ],
  "JWT Sign": [
    "Generates JSON Web Tokens.",
    "Supports custom payloads and signing algorithms.",
    "Demonates token-based authentication.",
    "Useful for API development and testing.",
    "Helps developers understand JWT structures."
  ],
  "Salt Generator": [
    "Produces cryptographically secure random salts.",
    "Used with password hashing algorithms.",
    "Prevents rainbow table attacks.",
    "Improves password storage security.",
    "Essential in authentication systems."
  ],
  "Fletcher-16 Checksum": [
    "Computes Fletcher-16 integrity checksums.",
    "Detects accidental data corruption.",
    "Faster than many cryptographic hashes.",
    "Common in networking and embedded systems.",
    "Intended for error detection rather than security."
  ],
  "Enigma Machine": [
    "Simulates the famous German WWII encryption machine.",
    "Uses rotating rotors and plugboard substitutions.",
    "Produces complex polyalphabetic encryption.",
    "Important in cryptography history.",
    "Demonstrates mechanical encryption principles."
  ],
  "Bombe Simulator": [
    "Simulates Alan Turing's Bombe machine.",
    "Used to attack Enigma-encrypted communications.",
    "Demonstrates historical cryptanalysis techniques.",
    "Shows how known plaintext attacks work.",
    "Valuable educational cybersecurity tool."
  ],
  "Typex Machine": [
    "Simulates the British Typex cipher machine.",
    "Based on Enigma but significantly enhanced.",
    "Used by Allied forces during WWII.",
    "Provides stronger operational security.",
    "Demonstrates evolution of mechanical cryptography."
  ],
  "Classical Ciphers": [
    "A collection of historical encryption techniques developed before modern computers.",
    "Includes substitution, transposition, and polyalphabetic ciphers.",
    "Primarily used for education, research, and cryptanalysis training.",
    "Helps users understand the foundations of modern cryptography.",
    "Demonstrates how encryption evolved over time."
  ],
  "Symmetric Cryptography": [
    "Contains algorithms that use a single shared key for encryption and decryption.",
    "Includes both modern and legacy block and stream ciphers.",
    "Focuses on confidentiality and secure data protection.",
    "Commonly used for file encryption, network security, and secure communications.",
    "Demonstrates the principles behind modern encryption systems."
  ],
  "Asymmetric Cryptography": [
    "Provides tools based on public-key cryptography.",
    "Uses separate public and private keys for secure operations.",
    "Supports encryption, decryption, digital signatures, and authentication.",
    "Forms the foundation of PKI, SSL/TLS, and secure online communication.",
    "Essential for modern cybersecurity infrastructures."
  ],
  "Hashing & KDFs": [
    "Contains cryptographic hash functions and key derivation mechanisms.",
    "Supports integrity verification and password protection.",
    "Includes tools for generating digests, HMACs, and password hashes.",
    "Demonstrates one-way cryptographic transformations.",
    "Critical for authentication and secure credential storage."
  ],
  "Encoding & Decoding": [
    "Provides data representation and transformation utilities.",
    "Converts information between formats such as Base64, Hex, and Binary.",
    "Helps users understand how data is stored and transmitted.",
    "Not intended for encryption or security protection.",
    "Useful for development, analysis, and debugging tasks."
  ],
  "Certificates & TLS": [
    "Focuses on Public Key Infrastructure (PKI) and secure communication protocols.",
    "Provides tools for analyzing digital certificates and TLS configurations.",
    "Helps verify trust relationships between systems.",
    "Useful for SSL/TLS troubleshooting and security audits.",
    "Essential for understanding secure internet communications."
  ],
  "Blockchain": [
    "Contains tools related to cryptocurrency and blockchain technologies.",
    "Supports address validation, key encoding, and Merkle Tree generation.",
    "Demonstrates cryptographic principles used in decentralized systems.",
    "Useful for blockchain developers and researchers.",
    "Helps users understand cryptocurrency security mechanisms."
  ],
  "Steganography": [
    "Provides methods for concealing information inside other media.",
    "Supports hiding data within text, images, and audio files.",
    "Focuses on covert communication rather than encryption.",
    "Demonstrates techniques used in information hiding.",
    "Useful for cybersecurity education and digital forensics."
  ],
  "File Forensics": [
    "Provides forensic analysis tools for digital files.",
    "Calculates hashes, entropy values, and statistical properties.",
    "Helps identify tampering, encryption, compression, or malware.",
    "Supports digital investigations and incident response activities.",
    "Essential for forensic examiners and security analysts."
  ],
  "Malware Analysis": [
    "Focuses on static analysis of suspicious files and binaries.",
    "Helps identify malware characteristics and relationships.",
    "Includes hash intelligence, similarity analysis, and executable inspection.",
    "Supports threat hunting and malware research workflows.",
    "Useful for cybersecurity professionals and researchers."
  ],
  "Utilities": [
    "A collection of general-purpose security and cryptography utilities.",
    "Includes password analysis, token generation, checksum calculation, and random data generation.",
    "Provides supporting functions used across multiple security domains.",
    "Designed for developers, analysts, and students.",
    "Enhances productivity during security testing and development."
  ],
  "Historical Machines": [
    "Simulates famous cryptographic machines from history.",
    "Includes wartime encryption devices and codebreaking systems.",
    "Demonstrates how mechanical cryptography operated before digital computers.",
    "Provides interactive learning experiences for cryptography enthusiasts.",
    "Helps users understand the historical development of secure communication."
  ]
};

const ALL_TOOLS = Object.keys(TOOLS_DATA);

const INITIAL_QUICK_QUESTIONS = [
  "What is Asymmetric Cryptography?",
  "What are Hashing & KDFs?",
  "What is Encoding & Decoding?",
  "What are Certificates & TLS?",
  "What are Blockchain Tools?",
];

interface Message {
  role: 'bot' | 'user';
  content: string | string[];
}

export function ChatbotWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'bot',
      content: "Hello! I can explain any tool on this website. What tool would you like to know about?"
    }
  ]);
  const [input, setInput] = useState('');
  const [quickQuestions, setQuickQuestions] = useState(INITIAL_QUICK_QUESTIONS);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      setQuickQuestions(INITIAL_QUICK_QUESTIONS);
    }
  }, [isOpen]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isOpen]);

  const findToolDefinition = (query: string) => {
    const lowerQuery = query.toLowerCase().replace(/what is|what are|explain|tell me about/g, '').trim();
    
    // Exact match first
    const exactMatch = ALL_TOOLS.find(t => t.toLowerCase() === lowerQuery);
    if (exactMatch) return { name: exactMatch, definition: TOOLS_DATA[exactMatch] };
    
    // Partial match
    const partialMatch = ALL_TOOLS.find(t => t.toLowerCase().includes(lowerQuery) || lowerQuery.includes(t.toLowerCase()));
    if (partialMatch) return { name: partialMatch, definition: TOOLS_DATA[partialMatch] };
    
    return null;
  };

  const sendMessage = (text: string) => {
    if (!text.trim()) return;
    
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput('');

    setTimeout(() => {
      const match = findToolDefinition(text);
      if (match) {
        setMessages(prev => [
          ...prev, 
          { 
            role: 'bot', 
            content: match.definition 
          }
        ]);
      } else {
        const fallbackTools = ["Caesar Cipher", "AES", "RSA", "Hash Generator", "File Hash"];
        setMessages(prev => [
          ...prev, 
          { 
            role: 'bot', 
            content: `I can only answer questions about the tools on this website. Try asking about ${fallbackTools.join(', ')}, etc.`
          }
        ]);
      }
    }, 400);
  };

  const handleSend = () => {
    sendMessage(input);
  };

  const handleQuickQuestion = (question: string) => {
    setQuickQuestions(prev => prev.filter(q => q !== question));
    sendMessage(question);
  };

  return (
    <>
      {/* Floating Button */}
      <motion.button
        className="fixed bottom-6 right-6 w-14 h-14 bg-primary text-primary-foreground rounded-full flex items-center justify-center shadow-lg hover:shadow-xl hover:scale-105 transition-all z-50"
        onClick={() => setIsOpen(true)}
        initial={{ scale: 0 }}
        animate={{ scale: isOpen ? 0 : 1 }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <MessageCircle size={24} />
      </motion.button>

      {/* Chat Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed bottom-6 right-6 w-[350px] max-w-[calc(100vw-3rem)] h-[500px] max-h-[calc(100vh-3rem)] glass rounded-2xl flex flex-col overflow-hidden z-50 card-shadow border border-border"
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
          >
            {/* Header */}
            <div className="bg-primary text-primary-foreground px-4 py-3 flex items-center justify-between border-b border-primary-foreground/10">
              <div className="flex items-center gap-2">
                <MessageCircle size={20} />
                <h3 className="font-semibold text-sm">CipherVerse Guide</h3>
              </div>
              <button 
                onClick={() => setIsOpen(false)}
                className="p-1 hover:bg-primary-foreground/20 rounded-md transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            {/* Messages Area */}
            <div 
              ref={scrollRef}
              className="flex-1 p-4 overflow-y-auto flex flex-col gap-4 bg-background/50"
            >
              {messages.map((msg, idx) => (
                <motion.div 
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div 
                    className={`max-w-[85%] rounded-2xl px-4 py-2 text-sm ${
                      msg.role === 'user' 
                        ? 'bg-primary text-primary-foreground rounded-tr-sm' 
                        : 'bg-card border border-border text-foreground rounded-tl-sm shadow-sm'
                    }`}
                  >
                    {Array.isArray(msg.content) ? (
                      <ul className="list-disc list-outside ml-4 space-y-1">
                        {msg.content.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Quick Questions Area */}
            {quickQuestions.length > 0 && (
              <div className="px-4 py-3 border-t border-border bg-card">
                <p className="text-xs text-muted-foreground mb-2 font-medium">Quick Questions</p>
                <div className="flex flex-wrap gap-2">
                  {quickQuestions.map((q) => (
                    <button
                      key={q}
                      onClick={() => handleQuickQuestion(q)}
                      className="text-xs px-3 py-1.5 rounded-full bg-secondary text-secondary-foreground border border-border hover:bg-primary hover:text-primary-foreground transition-colors text-left"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Input Area */}
            <div className="p-3 border-t border-border bg-card">
              <div className="relative">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Type a tool name or ask a question..."
                  className="pr-10 bg-background"
                />
                <Button 
                  size="icon" 
                  variant="ghost" 
                  className="absolute right-1 top-1 h-7 w-7 text-primary hover:text-primary hover:bg-primary/10"
                  onClick={handleSend}
                >
                  <Send size={16} />
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
