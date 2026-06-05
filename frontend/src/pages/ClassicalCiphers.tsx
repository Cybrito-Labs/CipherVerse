import { KeyRound, Sigma, RotateCcw, BookOpen, Grid3X3, Calculator, Hash, Train, Shuffle } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const ciphers = [
  { title: 'Caesar Cipher', description: 'Shift-based substitution cipher using a fixed shift value', icon: RotateCcw, path: '/classical/caesar' },
  { title: 'Vigenère Cipher', description: 'Polyalphabetic substitution using a keyword', icon: KeyRound, path: '/classical/vigenere' },
  { title: 'Atbash Cipher', description: 'Reverse alphabet substitution (A↔Z, B↔Y, ...)', icon: Shuffle, path: '/classical/atbash' },
  { title: 'Bacon Cipher', description: 'Binary steganographic cipher (A/B encoding)', icon: BookOpen, path: '/classical/bacon' },
  { title: 'Bifid Cipher', description: 'Polybius square fractionation cipher', icon: Grid3X3, path: '/classical/bifid' },
  { title: 'Affine Cipher', description: 'Mathematical substitution with ax + b mod 26', icon: Calculator, path: '/classical/affine' },
  { title: 'A1Z26 Cipher', description: 'Letter-to-number encoding (A=1, B=2, ... Z=26)', icon: Hash, path: '/classical/a1z26' },
  { title: 'Rail Fence Cipher', description: 'Zigzag transposition cipher across rails', icon: Train, path: '/classical/rail-fence' },
  { title: 'Substitution Cipher', description: 'Custom alphabet substitution with keyword', icon: Sigma, path: '/classical/substitution' },
];

export default function ClassicalCiphers() {
  return (
    <ToolPageLayout
      title="Classical Ciphers"
      description="Historical and educational encryption algorithms including substitution, transposition, and polyalphabetic ciphers."
      icon={KeyRound}
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
