import { ShieldCheck, ShieldAlert, KeyRound, Shuffle, Lock, SearchCode } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const ciphers = [
  { title: 'AES', description: 'Advanced Encryption Standard (AES-128, AES-192, AES-256).', icon: ShieldCheck, path: '/symmetric/aes' },
  { title: 'DES', description: 'Data Encryption Standard (Legacy).', icon: ShieldAlert, path: '/symmetric/des' },
  { title: '3DES', description: 'Triple Data Encryption Standard.', icon: ShieldAlert, path: '/symmetric/3des' },
  { title: 'RC2', description: 'Rivest Cipher 2 block cipher.', icon: KeyRound, path: '/symmetric/rc2' },
  { title: 'RC4', description: 'Rivest Cipher 4 stream cipher.', icon: KeyRound, path: '/symmetric/rc4' },
  { title: 'RC4 Drop', description: 'RC4 stream cipher with initial keystream bytes dropped.', icon: KeyRound, path: '/symmetric/rc4-drop' },
  { title: 'Blowfish', description: 'Keyed, symmetric cryptographic block cipher.', icon: Lock, path: '/symmetric/blowfish' },
  { title: 'SM4', description: 'Chinese standard block cipher.', icon: Lock, path: '/symmetric/sm4' },
  { title: 'CipherSaber2', description: 'RC4-based encryption with key stretching.', icon: SearchCode, path: '/symmetric/ciphersaber2' },
  { title: 'XOR', description: 'Simple exclusive OR cipher and bruteforcing.', icon: Shuffle, path: '/symmetric/xor' },
];

export default function SymmetricPage() {
  return (
    <ToolPageLayout
      title="Symmetric Cryptography"
      description="Modern and legacy symmetric encryption algorithms using the same cryptographic keys for both encryption of plaintext and decryption of ciphertext."
      icon={ShieldCheck}
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {ciphers.map((cipher) => (
          <ToolCard
            key={cipher.path}
            title={cipher.title}
            description={cipher.description}
            icon={cipher.icon}
            path={cipher.path}
          />
        ))}
      </div>
    </ToolPageLayout>
  );
}
