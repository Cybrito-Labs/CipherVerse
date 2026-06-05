import { motion } from 'framer-motion';
import { Shield, Zap, Lock, Hash, FileCode, Award, Blocks, Image, FileSearch, Bug, Wrench, Clock, KeyRound, ShieldCheck } from 'lucide-react';
import { CategoryCard } from '@/components/shared/CategoryCard';

const categories = [
  { title: 'Classical Ciphers', description: 'Caesar, Vigenère, Atbash, Affine, Bifid, and more classical encryption algorithms', icon: KeyRound, path: '/classical', toolCount: 9 },
  { title: 'Symmetric Cryptography', description: 'AES, DES, 3DES, Blowfish, SM4, RC4, XOR encryption and decryption', icon: ShieldCheck, path: '/symmetric', toolCount: 10 },
  { title: 'Asymmetric Cryptography', description: 'RSA and DSA key generation, encryption, signing, and verification', icon: Lock, path: '/asymmetric', toolCount: 6 },
  { title: 'Hashing & KDFs', description: 'SHA, MD5, HMAC, PBKDF2, Scrypt, bcrypt key derivation functions', icon: Hash, path: '/hashing', toolCount: 5 },
  { title: 'Encoding & Decoding', description: 'Base64, Hex, URL, Binary, and Morse code encoding tools', icon: FileCode, path: '/encoding', toolCount: 5 },
  { title: 'Certificates & TLS', description: 'X.509 certificate parsing, TLS analysis, and fingerprint generation', icon: Award, path: '/certificates', toolCount: 3 },
  { title: 'Blockchain Tools', description: 'Bitcoin and Ethereum validation, Merkle trees, WIF encoding', icon: Blocks, path: '/blockchain', toolCount: 4 },
  { title: 'Steganography', description: 'Hide and extract data in text, images, and audio files', icon: Image, path: '/steganography', toolCount: 4 },
  { title: 'File Forensics', description: 'File hashing, entropy analysis, and randomness testing', icon: FileSearch, path: '/file-forensics', toolCount: 4 },
  { title: 'Malware Analysis', description: 'Hash analysis, TLSH comparison, and PE file analysis', icon: Bug, path: '/malware-analysis', toolCount: 3 },
  { title: 'Utilities', description: 'Password strength, salt generation, JWT signing, checksums', icon: Wrench, path: '/utilities', toolCount: 4 },
  { title: 'Historical Machines', description: 'Enigma, Bombe, and Typex cipher machine simulators', icon: Clock, path: '/historical', toolCount: 3 },
];

const stats = [
  { label: 'Tools Available', value: '60+', icon: Zap },
  { label: 'Categories', value: '12', icon: Shield },
  { label: 'API Endpoints', value: '50+', icon: Lock },
  { label: 'Algorithms', value: '40+', icon: Hash },
];

export default function Dashboard() {
  return (
    <div className="max-w-7xl mx-auto space-y-10">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative overflow-hidden rounded-2xl glass p-8 md:p-12"
      >
        <div className="absolute inset-0 bg-grid-pattern opacity-30" />
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center animate-pulse-glow">
              <Shield className="w-7 h-7 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl md:text-4xl font-bold text-foreground">
                Cipher<span className="text-primary">Verse</span>
              </h1>
              <p className="text-muted-foreground text-sm">
                Professional Cybersecurity & Cryptography Platform
              </p>
            </div>
          </div>
          <p className="text-muted-foreground max-w-2xl text-sm md:text-base leading-relaxed mt-4">
            A comprehensive suite of cryptographic tools, forensics analyzers, and security utilities.
            Built for security researchers, ethical hackers, digital forensics analysts, and cryptography enthusiasts.
          </p>
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: idx * 0.08 }}
              className="glass rounded-xl p-5 text-center"
            >
              <div className="w-10 h-10 mx-auto mb-3 rounded-lg bg-primary/10 flex items-center justify-center">
                <Icon className="w-5 h-5 text-primary" />
              </div>
              <p className="text-2xl font-bold text-foreground">{stat.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
            </motion.div>
          );
        })}
      </div>

      {/* Categories Grid */}
      <div>
        <motion.h2
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-xl font-semibold text-foreground mb-5"
        >
          Tool Categories
        </motion.h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {categories.map((cat, idx) => (
            <CategoryCard
              key={cat.path}
              title={cat.title}
              description={cat.description}
              icon={cat.icon}
              path={cat.path}
              toolCount={cat.toolCount}
              delay={0.05 * idx}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
