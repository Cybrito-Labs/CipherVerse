import { Lock, FileSignature } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const ciphers = [
  { title: 'RSA', description: 'Rivest-Shamir-Adleman algorithm. Generate key pairs, encrypt, decrypt, sign, and verify messages.', icon: Lock, path: '/asymmetric/rsa' },
  { title: 'DSA', description: 'Digital Signature Algorithm. Generate keys, create digital signatures, and verify authenticity.', icon: FileSignature, path: '/asymmetric/dsa' },
];

export default function AsymmetricPage() {
  return (
    <ToolPageLayout
      title="Asymmetric Cryptography"
      description="Public-key cryptography uses pairs of keys: public keys which may be disseminated widely, and private keys which are known only to the owner."
      icon={Lock}
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
