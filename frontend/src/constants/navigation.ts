import {
  LayoutDashboard,
  KeyRound,
  ShieldCheck,
  Lock,
  Hash,
  FileCode,
  Award,
  Blocks,
  Image,
  FileSearch,
  Bug,
  Wrench,
  Clock,
  Globe,
  Settings,
  type LucideIcon,
} from 'lucide-react';

export interface NavItem {
  label: string;
  path: string;
  icon: LucideIcon;
  description: string;
  toolCount?: number;
  badge?: string;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

export const navigationGroups: NavGroup[] = [
  {
    label: 'Overview',
    items: [
      {
        label: 'Dashboard',
        path: '/',
        icon: LayoutDashboard,
        description: 'Platform overview and quick launch',
      },
    ],
  },
  {
    label: 'Cryptography',
    items: [
      {
        label: 'Classical Ciphers',
        path: '/classical',
        icon: KeyRound,
        description: 'Caesar, Vigenere, Atbash, and more',
        toolCount: 9,
      },
      {
        label: 'Symmetric Crypto',
        path: '/symmetric',
        icon: ShieldCheck,
        description: 'AES, DES, Blowfish, RC4, XOR',
        toolCount: 10,
      },
      {
        label: 'Asymmetric Crypto',
        path: '/asymmetric',
        icon: Lock,
        description: 'RSA, DSA key generation and operations',
        toolCount: 6,
      },
      {
        label: 'Hashing & KDFs',
        path: '/hashing',
        icon: Hash,
        description: 'SHA, MD5, HMAC, PBKDF2, Scrypt, bcrypt',
        toolCount: 5,
      },
      {
        label: 'Encoding & Decoding',
        path: '/encoding',
        icon: FileCode,
        description: 'Base64, Hex, URL, Binary, Morse',
        toolCount: 5,
      },
    ],
  },
  {
    label: 'Security',
    items: [
      {
        label: 'Certificates & TLS',
        path: '/certificates',
        icon: Award,
        description: 'X.509 parsing, TLS analysis',
        toolCount: 3,
      },
      {
        label: 'Blockchain',
        path: '/blockchain',
        icon: Blocks,
        description: 'Bitcoin, Ethereum, Merkle trees',
        toolCount: 4,
      },
      {
        label: 'Steganography',
        path: '/steganography',
        icon: Image,
        description: 'Text, image, and audio steganography',
        toolCount: 4,
      },
    ],
  },
  {
    label: 'Analysis',
    items: [
      {
        label: 'File Forensics',
        path: '/file-forensics',
        icon: FileSearch,
        description: 'File hashing, entropy, randomness tests',
        toolCount: 4,
      },
      {
        label: 'Malware Analysis',
        path: '/malware-analysis',
        icon: Bug,
        description: 'Hash analysis, TLSH, PE analysis',
        toolCount: 3,
      },
    ],
  },
  {
    label: 'Tools',
    items: [
      {
        label: 'Utilities',
        path: '/utilities',
        icon: Wrench,
        description: 'Password strength, salt, JWT, checksums',
        toolCount: 4,
      },
      {
        label: 'Historical Machines',
        path: '/historical',
        icon: Clock,
        description: 'Enigma, Bombe, Typex simulators',
        toolCount: 3,
      },
      {
        label: 'API Explorer',
        path: '/api-explorer',
        icon: Globe,
        description: 'Browse and test all API endpoints',
      },
      {
        label: 'Settings',
        path: '/settings',
        icon: Settings,
        description: 'Configuration and preferences',
      },
    ],
  },
];

export const allNavItems: NavItem[] = navigationGroups.flatMap((g) => g.items);
